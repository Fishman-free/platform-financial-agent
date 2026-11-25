# -*- coding: utf-8 -*-
"""中文特化的 BM25 稀疏检索（design.md D4）。

token 策略：
- CJK 连续串 → 字符 bigram（单字串保留单字）；
- ASCII 字母数字串（含 % . · - + 等内嵌符号）→ 整体小写 token，
  保证 "3·15"、"70%"、"GB2760"、"12315" 类法规/标准/百分比 token 不被拆碎；
- 其余符号丢弃。

不引入 jieba/rank_bm25：短文本 bigram 已足够，且零词典依赖（design.md D4）。
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Callable, Dict, Iterable, List, Optional, Tuple

# ASCII 串：字母数字开头，允许内嵌 . % · - + _ 等（法规号/百分比/日期形态）
_RE_ASCII = re.compile(r"[A-Za-z0-9][A-Za-z0-9.%·+\-_]*")
# CJK 基本区 + 扩展A（覆盖常用中文）
_RE_CJK = re.compile(r"[㐀-䶿一-鿿]+")


def tokenize(text: str) -> List[str]:
    """把文本切分为检索 token（中文 bigram + ASCII 整串）。"""
    tokens: List[str] = []
    for m in _RE_ASCII.finditer(text):
        tokens.append(m.group().lower())
    for m in _RE_CJK.finditer(text):
        run = m.group()
        if len(run) == 1:
            tokens.append(run)
        else:
            tokens.extend(run[i:i + 2] for i in range(len(run) - 1))
    return tokens


class BM25Index:
    """Okapi BM25 倒排索引（增量 add_doc，全量 search）。"""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._doc_ids: List[str] = []
        self._doc_len: Dict[str, int] = {}
        self._tf: Dict[str, Counter] = {}          # doc_id -> token 计数
        self._inverted: Dict[str, Dict[str, int]] = {}  # token -> {doc_id: tf}
        self._total_len = 0

    @property
    def doc_count(self) -> int:
        return len(self._doc_ids)

    @property
    def avg_len(self) -> float:
        return self._total_len / self.doc_count if self.doc_count else 0.0

    def add_doc(self, doc_id: str, text: str) -> None:
        if doc_id in self._tf:
            return  # 幂等：同 id 不重复入索引
        toks = tokenize(text)
        tf = Counter(toks)
        self._doc_ids.append(doc_id)
        self._tf[doc_id] = tf
        self._doc_len[doc_id] = len(toks)
        self._total_len += len(toks)
        for tok, cnt in tf.items():
            self._inverted.setdefault(tok, {})[doc_id] = cnt

    def remove_doc(self, doc_id: str) -> None:
        if doc_id not in self._tf:
            return
        for tok in self._tf[doc_id]:
            postings = self._inverted.get(tok)
            if postings:
                postings.pop(doc_id, None)
                if not postings:
                    del self._inverted[tok]
        self._total_len -= self._doc_len.pop(doc_id)
        del self._tf[doc_id]
        self._doc_ids.remove(doc_id)

    def search(self, query: str, top_k: int = 10,
               allowed: Optional[Iterable[str]] = None,
               ) -> List[Tuple[str, float]]:
        """返回 [(doc_id, score)] 降序。allowed 为候选 doc_id 集合（元数据过滤前置）。"""
        q_toks = tokenize(query)
        if not q_toks or not self.doc_count:
            return []
        allowed_set = set(allowed) if allowed is not None else None
        n = self.doc_count
        avgdl = self.avg_len or 1.0
        scores: Dict[str, float] = {}
        for tok in set(q_toks):
            postings = self._inverted.get(tok)
            if not postings:
                continue
            df = len(postings)
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            for doc_id, tf in postings.items():
                if allowed_set is not None and doc_id not in allowed_set:
                    continue
                dl = self._doc_len[doc_id]
                denom = tf + self.k1 * (1 - self.b + self.b * dl / avgdl)
                scores[doc_id] = scores.get(doc_id, 0.0) + idf * (tf * (self.k1 + 1)) / denom
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return ranked[:top_k]

    def search_multi(self, queries: List[str], top_k: int = 10,
                     allowed: Optional[Iterable[str]] = None) -> Dict[str, List[Tuple[str, float]]]:
        """多查询批量检索（查询改写场景），返回 {query: ranked}。"""
        return {q: self.search(q, top_k, allowed) for q in queries}
