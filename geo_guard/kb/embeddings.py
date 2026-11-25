# -*- coding: utf-8 -*-
"""嵌入后端（design.md D3）：确定性哈希嵌入为默认，API 嵌入为增强。

- HashEmbedding：字符 bigram/trigram + ASCII 词 → FNV-1a 哈希 → 384 维有符号
  计数 → 子线性 tf → L2 归一化。同一文本永远同一向量（可测试、可复现、零依赖），
  离线演示与 CI 的默认后端。语义泛化弱于真实模型——已知局限，由 BM25 字面通道
  兜基本盘（头脑风暴对抗评审 #2），且阈值随后端自动偏移（config.strong_threshold）。
- ApiEmbedding：OpenAI 兼容 /embeddings 端点（DashScope text-embedding-v3 等），
  sha256 键控磁盘缓存，任何失败自动回退 HashEmbedding。
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..config import Config
from .bm25 import tokenize

# FNV-1a 32bit 常数
_FNV_OFFSET = 0x811C9DC5
_FNV_PRIME = 0x01000193
_MASK32 = 0xFFFFFFFF


def _fnv1a(s: str) -> int:
    h = _FNV_OFFSET
    for byte in s.encode("utf-8"):
        h = ((h ^ byte) * _FNV_PRIME) & _MASK32
    return h


def _char_ngrams(text: str, n_min: int = 2, n_max: int = 3) -> List[str]:
    """CJK 字符 n-gram（bigram+trigram），与 tokenize 的 ASCII 词合并作为嵌入 token。"""
    grams: List[str] = []
    cjk_run: List[str] = []
    for ch in text:
        if "一" <= ch <= "鿿":
            cjk_run.append(ch)
        else:
            if cjk_run:
                grams.extend(_grams_of_run("".join(cjk_run), n_min, n_max))
                cjk_run = []
    if cjk_run:
        grams.extend(_grams_of_run("".join(cjk_run), n_min, n_max))
    return grams


def _grams_of_run(run: str, n_min: int, n_max: int) -> List[str]:
    out = []
    for n in range(n_min, n_max + 1):
        if len(run) >= n:
            out.extend(run[i:i + n] for i in range(len(run) - n + 1))
        elif n == n_min:
            out.append(run)
    return out


class HashEmbedding:
    """本地确定性哈希嵌入（零外部依赖）。"""

    name = "hash"

    def __init__(self, dim: int = 384):
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: Sequence[str]) -> np.ndarray:
        mat = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            toks = tokenize(text) + _char_ngrams(text)
            counts: Dict[str, int] = {}
            for t in toks:
                counts[t] = counts.get(t, 0) + 1
            vec = mat[row]
            for t, c in counts.items():
                h = _fnv1a(t)
                idx = h % self.dim
                sign = 1.0 if (h >> 16) & 1 else -1.0
                vec[idx] += sign * (1.0 + np.log(c))  # 子线性 tf
            norm = float(np.linalg.norm(vec))
            if norm > 0:
                vec /= norm
        return mat


class ApiEmbedding:
    """OpenAI 兼容 /embeddings 后端，带磁盘缓存与 Hash 回退。"""

    name = "api"

    def __init__(self, base_url: str, api_key: str, model: str,
                 cache_dir: Path, dim: int = 384, timeout: float = 20.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self._fallback = HashEmbedding(dim)
        self.dim = dim
        self._cache_path = cache_dir / "embed_cache.json"
        self._cache: Dict[str, List[float]] = {}
        self._cache_dirty = False
        self._load_cache()
        self.available = True  # 首次失败后置 False，避免反复超时拖慢管线

    def _load_cache(self) -> None:
        if self._cache_path.exists():
            try:
                self._cache = json.loads(self._cache_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                self._cache = {}

    def _save_cache(self) -> None:
        if self._cache_dirty:
            try:
                self._cache_path.parent.mkdir(parents=True, exist_ok=True)
                self._cache_path.write_text(
                    json.dumps(self._cache, ensure_ascii=False), encoding="utf-8")
                self._cache_dirty = False
            except OSError:
                pass

    @staticmethod
    def _key(text: str, model: str) -> str:
        return hashlib.sha256(f"{model}\x00{text}".encode("utf-8")).hexdigest()[:32]

    def embed(self, text: str) -> np.ndarray:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: Sequence[str]) -> np.ndarray:
        texts = list(texts)
        vectors: List[Optional[np.ndarray]] = [None] * len(texts)
        misses: List[int] = []
        for i, t in enumerate(texts):
            cached = self._cache.get(self._key(t, self.model))
            if cached is not None:
                vectors[i] = np.asarray(cached, dtype=np.float32)
            else:
                misses.append(i)
        if misses and self.available:
            fetched = self._call_api([texts[i] for i in misses])
            if fetched is not None and len(fetched) == len(misses):
                for i, vec in zip(misses, fetched):
                    vectors[i] = vec
                    self._cache[self._key(texts[i], self.model)] = vec.tolist()
                    self._cache_dirty = True
                self._save_cache()
            else:
                self.available = False  # 本进程内不再尝试，回退 Hash
        for i in misses:
            if vectors[i] is None:
                vectors[i] = self._fallback.embed(texts[i])
        return np.vstack([v for v in vectors if v is not None])

    def _call_api(self, texts: List[str]) -> Optional[List[np.ndarray]]:
        body = json.dumps({"model": self.model, "input": texts}).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/embeddings",
            data=body,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return [np.asarray(item["embedding"], dtype=np.float32)
                    for item in sorted(data["data"], key=lambda d: d["index"])]
        except Exception:
            return None


def get_embedder(config: Config):
    """嵌入后端选择：显式配置 → 环境变量探测 → HashEmbedding 兜底。"""
    setting = (config.embedding or "auto").lower()
    if setting == "hash":
        return HashEmbedding(config.embed_dim)
    if setting in ("api", "auto"):
        # DashScope 优先（与母团队技术栈一致），其次 OpenAI
        ds_key = os.environ.get("DASHSCOPE_API_KEY", "")
        oa_key = os.environ.get("OPENAI_API_KEY", "")
        if setting == "api" or ds_key or oa_key:
            if ds_key:
                base = os.environ.get("DASHSCOPE_BASE_URL",
                                      "https://dashscope.aliyuncs.com/compatible-mode/v1")
                model = os.environ.get("GEO_GUARD_EMBED_MODEL", "text-embedding-v3")
                return ApiEmbedding(base, ds_key, model, config.cache_path, dim=1024)
            if oa_key:
                base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
                model = os.environ.get("GEO_GUARD_EMBED_MODEL", "text-embedding-3-small")
                return ApiEmbedding(base, oa_key, model, config.cache_path, dim=1536)
    return HashEmbedding(config.embed_dim)
