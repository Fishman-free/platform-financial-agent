# -*- coding: utf-8 -*-
"""geo_guard —— GEO 逆向甄别内核。

面向用户的平台治理支持智能体系统 · 反 GEO 投毒模块的代码落地。
分层依赖（自上而下，禁止反向）：
    agent/plugin → dialogue/explain → screening/detection → retrieval → kb → schema/config

所有 LLM 触点（查询改写、claim 抽取、重排、意图路由、生成润色）均有规则兜底，
无 API 密钥时全链路可离线运行（NullBackend）。
"""

from .config import Config
from .schema import (
    Claim,
    Evidence,
    KBEntry,
    RiskLevel,
    RiskReport,
    SourceMeta,
    TraceReport,
)

__version__ = "0.1.0"

__all__ = [
    "Config",
    "Claim",
    "Evidence",
    "KBEntry",
    "RiskLevel",
    "RiskReport",
    "SourceMeta",
    "TraceReport",
    "__version__",
]
