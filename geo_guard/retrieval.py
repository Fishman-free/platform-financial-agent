# -*- coding: utf-8 -*-
"""混合检索管线（knowledge-retrieval 规格，docs/02 §3）。

双通道召回（BM25 + 稠密向量）→ RRF 融合 → 元数据过滤 → 两段式重排
（LLM 打分 / 规则特征兜底）。话术相似度采用 docs/02 §3.3 语义：
sim = 0.6×dense + 0.4×bm25_norm，阈值随嵌入后端自动偏移。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .config import Config
from .kb.bm25 import tokenize
from .kb.store import KnowledgeStore
from .llm import LLMBackend, NullBackend, touch_query_rewrite, touch_rerank
from .schema import KBEntry, KBName


@dataclass
class RetrievalHit:
    """单条检索命中（含通道分解，供证据链溯源）。"""

    entry: KBEntry
    score: float                      # 融合后 0~1
    bm25_norm: float = 0.0
    dense: float = 0.0
    rrf: float = 0.0
    rerank: Optional[float] = None

    @property
    def similarity(self) -> float:
        """话术相似度语义（docs/02 §3.3）：0.6×dense + 0.4×bm25_norm。"""
        return 0.6 * max(self.dense, 0.0) + 0.4 * self.bm25_norm


class HybridRetriever:
    def __init__(self, store: KnowledgeStore, config: Optional[Config] = None,
                 llm: Optional[LLMBackend] = None):
        self.store = store
        self.config = config or Config()
        self.llm = llm or NullBackend()

    # ---------------- 主检索入口 ----------------

    def retrieve(self, query: str,
                 kbs: Optional[Sequence[str]] = None,
                 categories: Optional[Sequence[str]] = None,
                 top_k: Optional[int] = None,
                 entities: Optional[Dict[str, str]] = None,
                 rewrite: bool = True) -> List[RetrievalHit]:
        """混合检索：多查询并集召回 + RRF + 重排。"""
        top_k = top_k or self.config.rerank_top
        allowed = self.store.active_ids(kbs, categories)
        if not allowed or not query.strip():
            return []
        queries = [query]
        if rewrite:
            queries = touch_query_rewrite(self.llm, query, entities)

        # 通道 1：BM25（多查询取各 doc 最优排名）
        bm25_rank: Dict[str, int] = {}
        bm25_raw: Dict[str, float] = {}
        for q in queries:
            for rank, (doc_id, sc) in enumerate(self.store.bm25_search(
                    q, self.config.recall_top, allowed)):
                if doc_id not in bm25_rank or rank < bm25_rank[doc_id]:
                    bm25_rank[doc_id] = rank
                bm25_raw[doc_id] = max(bm25_raw.get(doc_id, 0.0), sc)

        # 通道 2：稠密向量（多查询取最大余弦）
        dense: Dict[str, float] = {}
        for q in queries:
            qv = self.store.embed(q)
            for doc_id, s in self.store.dense_scores(qv, allowed).items():
                dense[doc_id] = max(dense.get(doc_id, -1.0), s)

        # RRF 融合（两通道各自排名）
        cand = set(bm25_rank) | {d for d, s in dense.items() if s > 0}
        dense_rank = {d: i for i, (d, _) in enumerate(
            sorted(dense.items(), key=lambda kv: kv[1], reverse=True))}
        rrf: Dict[str, float] = {}
        k = self.config.rrf_k
        for doc_id in cand:
            s = 0.0
            if doc_id in bm25_rank:
                s += 1.0 / (k + bm25_rank[doc_id] + 1)
            if doc_id in dense_rank:
                s += 1.0 / (k + dense_rank[doc_id] + 1)
            rrf[doc_id] = s

        # BM25 归一化（对候选集内最大值归一）
        max_bm25 = max(bm25_raw.values()) if bm25_raw else 0.0
        hits: List[RetrievalHit] = []
        for doc_id, r in rrf.items():
            entry = self.store.get(doc_id)
            if entry is None:
                continue
            hits.append(RetrievalHit(
                entry=entry,
                score=r,
                bm25_norm=(bm25_raw.get(doc_id, 0.0) / max_bm25) if max_bm25 > 0 else 0.0,
                dense=max(dense.get(doc_id, 0.0), 0.0),
                rrf=r,
            ))
        hits.sort(key=lambda h: h.score, reverse=True)
        shortlist = hits[:self.config.recall_top]

        # 两段式重排
        reranked = self._rerank(query, shortlist)
        return reranked[:top_k]

    # ---------------- 重排（触点 3：LLM / 规则双路径） ----------------

    def _rerank(self, query: str, hits: List[RetrievalHit]) -> List[RetrievalHit]:
        if not hits:
            return hits
        llm_scores = touch_rerank(self.llm, query,
                                  [{"title": h.entry.title, "content": h.entry.content}
                                   for h in hits[:20]])
        q_toks = set(tokenize(query))
        max_rrf = max(h.rrf for h in hits) or 1.0
        for i, h in enumerate(hits):
            e_toks = set(tokenize(h.entry.title + h.entry.content))
            overlap = len(q_toks & e_toks) / len(q_toks) if q_toks else 0.0
            if llm_scores is not None and i < len(llm_scores):
                h.rerank = llm_scores[i] / 10.0
            else:
                # 规则重排：0.5×RRF归一 + 0.3×实体重合 + 0.2×severity 先验
                h.rerank = (0.5 * (h.rrf / max_rrf) + 0.3 * overlap
                            + 0.2 * h.entry.severity)
        hits.sort(key=lambda h: (h.rerank or 0.0, h.rrf), reverse=True)
        return hits

    # ---------------- 便捷封装 ----------------

    def match_patterns(self, segment: str, top_k: int = 5) -> List[RetrievalHit]:
        """KB1 话术模式匹配专用入口。"""
        return self.retrieve(segment, kbs=[KBName.PATTERNS.value], top_k=top_k,
                             rewrite=False)

    def match_facts(self, claim_text: str, top_k: int = 3) -> List[RetrievalHit]:
        """KB3 权威事实核验专用入口。"""
        return self.retrieve(claim_text, kbs=[KBName.FACTS.value], top_k=top_k,
                             rewrite=False)

    def match_cases(self, text: str, top_k: int = 3) -> List[RetrievalHit]:
        """KB4 判例检索专用入口。"""
        return self.retrieve(text, kbs=[KBName.CASES.value], top_k=top_k)

    def match_rules(self, query: str, top_k: int = 3) -> List[RetrievalHit]:
        """KB5 平台规则（维权路径）检索专用入口。"""
        return self.retrieve(query, kbs=[KBName.RULES.value], top_k=top_k)

    def pattern_similarity(self, segment: str) -> List[RetrievalHit]:
        """返回带 similarity 语义的 KB1 命中（阈值判断在 detection 层做）。"""
        return self.match_patterns(segment, top_k=self.config.rerank_top)
