# -*- coding: utf-8 -*-
"""F1 话术模式匹配（docs/02 §4，docs/03 路径二）。

内容切分为证据单元（句/段），逐单元与 KB1 话术模式库做混合检索；
相似度分级：>=strong 为高危话术信号（强证据），weak~strong 为弱证据。
阈值随嵌入后端偏移（config.strong_threshold）。
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from ..config import Config
from ..retrieval import HybridRetriever, RetrievalHit
from ..schema import Evidence

# 证据单元切分：中文句末标点 + 换行；过短片段并入前句
_SENT_SPLIT = re.compile(r"(?<=[。！？；!?;\n])")
_MIN_UNIT = 8
_MAX_UNIT = 300


def split_units(text: str) -> List[Tuple[int, int, str]]:
    """把文本切成 (start, end, unit) 证据单元列表。"""
    units: List[Tuple[int, int, str]] = []
    pos = 0
    for part in _SENT_SPLIT.split(text):
        if not part:
            continue
        start, end = pos, pos + len(part)
        pos = end
        stripped = part.strip()
        if not stripped:
            continue
        if units and len(stripped) < _MIN_UNIT:
            ps, pe, pu = units[-1]
            units[-1] = (ps, end, pu + part)
        else:
            units.append((start, end, part))
    # 超长单元按窗口再切（保证检索粒度）
    out: List[Tuple[int, int, str]] = []
    for s, e, u in units:
        if len(u) <= _MAX_UNIT:
            out.append((s, e, u))
        else:
            step = _MAX_UNIT
            for i in range(0, len(u), step):
                out.append((s + i, min(s + i + step, e), u[i:i + step]))
    return out


class PatternMatcher:
    def __init__(self, retriever: HybridRetriever, config: Optional[Config] = None):
        self.retriever = retriever
        self.config = config or retriever.config

    def match(self, text: str) -> List[Evidence]:
        """返回话术证据列表（强/弱证据都返回，强度记在 score，融合层分级使用）。"""
        backend = getattr(self.retriever.store.embedder, "name", "hash")
        strong_t = self.config.strong_threshold(backend)
        weak_t = self.config.weak_threshold(backend)
        evidences: List[Evidence] = []
        seen_categories: dict = {}
        for start, end, unit in split_units(text):
            if len(unit.strip()) < _MIN_UNIT:
                continue
            hits: List[RetrievalHit] = self.retriever.pattern_similarity(unit)
            for h in hits:
                sim = h.similarity
                if sim < weak_t:
                    continue
                cat = h.entry.category
                # 同一模式类别保留最强一条证据，防止同模式重复计分
                prev = seen_categories.get(cat)
                ev = Evidence(
                    channel="rhetoric",
                    label=h.entry.title,
                    score=min(sim, 1.0),
                    severity=h.entry.severity,
                    span=(start, end),
                    excerpt=unit.strip()[:120],
                    kb_refs=[h.entry.entry_id],
                    detail={
                        "category": cat,
                        "strength": "strong" if sim >= strong_t else "weak",
                        "similarity": round(sim, 4),
                        "bm25_norm": round(h.bm25_norm, 4),
                        "dense": round(h.dense, 4),
                    },
                )
                if prev is None or ev.score > prev.score:
                    seen_categories[cat] = ev
        evidences = list(seen_categories.values())
        evidences.sort(key=lambda e: e.score * (0.5 + e.severity), reverse=True)
        return evidences
