# -*- coding: utf-8 -*-
"""全局配置：阈值、权重、路径的唯一来源（design.md D1）。

所有默认值可被环境变量覆写（前缀 GEO_GUARD_），保证部署环境无需改代码。
判定相关阈值（t_high/t_low、融合权重）集中在此，供 fusion 与校准器读写，
避免魔法数字散落各模块。
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Dict, List

# 仓库根目录（geo_guard/ 的上一级）
REPO_ROOT = Path(__file__).resolve().parent.parent

# 权威白名单锚点域名（良性对照集，I-22）。
# 白名单 != 免检：命中白名单仍跑完整管线，话术强命中时降档并提示人工复核。
WHITELIST_DOMAINS: List[str] = [
    "gov.cn",
    "samr.gov.cn",
    "std.samr.gov.cn",
    "12315.cn",
    "xinhuanet.com",
    "news.cn",
    "people.com.cn",
    "cctv.com",
    "chinanews.com.cn",
    "cac.gov.cn",
    "miit.gov.cn",
    "nmpa.gov.cn",
]


def _env(name: str, default):
    """按类型读取 GEO_GUARD_ 前缀环境变量。"""
    raw = os.environ.get(f"GEO_GUARD_{name}")
    if raw is None or raw == "":
        return default
    if isinstance(default, bool):
        return raw.lower() in ("1", "true", "yes", "on")
    if isinstance(default, int) and not isinstance(default, bool):
        return int(raw)
    if isinstance(default, float):
        return float(raw)
    return raw


@dataclass
class Config:
    # ---- 路径 ----
    repo_root: Path = REPO_ROOT
    db_path: str = ""            # 空 = repo_root/knowledge/geoguard.db
    seed_dir: str = ""           # 空 = repo_root/knowledge/seed
    cache_dir: str = ""          # 空 = repo_root/knowledge/cache
    # 合成对抗集隔离目录：仅评测用，任何加载路径 MUST NOT 读取（data-ingest 规格）
    adversarial_dir: str = ""    # 空 = repo_root/knowledge/adversarial

    # ---- 后端 ----
    # off|auto|dashscope|openai —— auto: 探测环境变量密钥，均无则 NullBackend
    llm: str = "auto"
    llm_model: str = ""          # 空 = 按后端取默认模型
    # hash|api —— api 失败自动回退 hash
    embedding: str = "auto"
    embed_dim: int = 384

    # ---- 检索（knowledge-retrieval 规格）----
    rrf_k: int = 60              # RRF 融合常数
    recall_top: int = 50         # 粗召回规模
    rerank_top: int = 5          # 重排后保留
    pattern_strong: float = 0.70 # 高危话术阈值（继承母项目"70% 告警"设定）
    pattern_weak: float = 0.50   # 弱证据下限
    # 本地哈希嵌入分布更尖，阈值整体下调（对抗评审 #2 决议；按后端名索引）
    hash_threshold_offset: float = -0.05

    # ---- 判定融合（geo-screening 规格，docs/02 §5.2）----
    fusion_weights: Dict[str, float] = field(default_factory=lambda: {
        "rhetoric": 0.30,        # s1 话术信号
        "structural": 0.15,      # s2 结构信号
        "factual": 0.25,         # s3 事实信号
        "source": 0.15,          # s4 来源信号
        "case": 0.05,            # s5 判例信号
        "network": 0.10,         # s6 协同信号（端侧缺失时重归一化）
    })
    t_high: float = 0.72
    t_low: float = 0.40
    thin_evidence_count: int = 3   # 证据 < 3 条触发"证据单薄"降档
    multi_pattern_boost: float = 1.2  # top3 多模式命中加成

    # ---- 记忆与画像（memory-profile 规格，docs/05）----
    work_memory_k: int = 20        # 工作记忆环形缓冲轮数
    lambda_screening: float = math.log(2) / 90.0   # 甄别结论半衰期 90 天
    lambda_preference: float = math.log(2) / 30.0  # 一般偏好半衰期 30 天
    importance_protect: float = 0.8  # >= 该值不参与衰减清除
    ema_alpha: float = 0.3           # 画像隐式更新步长
    recall_top_k: int = 5

    # ---- 数据管线（data-ingest 规格，docs/04）----
    simhash_distance: int = 3        # 海明距离 <= 3 判近重复
    feedback_dispute_threshold: int = 3  # 同条目异议累计触发复审

    # ---- 交互（dialogue-guidance 规格，docs/06）----
    max_evidence_cards: int = 3      # 单轮证据卡片上限（分级披露）

    def __post_init__(self) -> None:
        # 环境变量覆写（仅标量字段；dict 字段用 GEO_GUARD_T_HIGH 等单项覆写）
        for f in fields(self):
            if f.name in ("fusion_weights", "repo_root"):
                continue
            setattr(self, f.name, _env(f.name.upper(), getattr(self, f.name)))
        # 融合权重的单项覆写：GEO_GUARD_W_RHETORIC 等
        for key in list(self.fusion_weights):
            env_val = os.environ.get(f"GEO_GUARD_W_{key.upper()}")
            if env_val:
                self.fusion_weights[key] = float(env_val)
        self.repo_root = Path(self.repo_root)

    # ---- 派生路径（保证 dataclass 可哈希字段不变，运行时解析）----
    @property
    def db_file(self) -> Path:
        p = Path(self.db_path) if self.db_path else self.repo_root / "knowledge" / "geoguard.db"
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def seed_path(self) -> Path:
        return Path(self.seed_dir) if self.seed_dir else self.repo_root / "knowledge" / "seed"

    @property
    def cache_path(self) -> Path:
        p = Path(self.cache_dir) if self.cache_dir else self.repo_root / "knowledge" / "cache"
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def adversarial_path(self) -> Path:
        return Path(self.adversarial_dir) if self.adversarial_dir else self.repo_root / "knowledge" / "adversarial"

    def strong_threshold(self, backend_name: str) -> float:
        """按嵌入后端返回高危话术阈值（哈希嵌入自动下调偏移量）。"""
        if backend_name == "hash":
            return max(0.0, self.pattern_strong + self.hash_threshold_offset)
        return self.pattern_strong

    def weak_threshold(self, backend_name: str) -> float:
        if backend_name == "hash":
            return max(0.0, self.pattern_weak + self.hash_threshold_offset)
        return self.pattern_weak
