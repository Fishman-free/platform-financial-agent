# -*- coding: utf-8 -*-
"""混合检索管线（knowledge-retrieval 规格 + 验收基线 #4）。"""

from geo_guard.schema import KBName


def test_trigger_example_recall_at_5(retriever, store):
    """验收基线 #4：KB1 触发例句 Recall@5 >= 0.8（Hash 兜底模式）。"""
    pattern_entries = [e for e in store.iter_entries(kb=KBName.PATTERNS.value)
                       if e.metadata.get("trigger_examples")]
    assert len(pattern_entries) >= 10
    hit = 0
    for entry in pattern_entries:
        for ex in entry.metadata["trigger_examples"]:
            hits = retriever.match_patterns(ex, top_k=5)
            if any(h.entry.entry_id == entry.entry_id for h in hits):
                hit += 1
                break  # 每条目任一例句命中即算召回
    recall = hit / len(pattern_entries)
    assert recall >= 0.8, f"触发例句 Recall@5={recall:.2f} < 0.8"


def test_kb_filter_isolation(retriever):
    """库别过滤生效（规格场景）：仅查 KB5 不返回其他库条目。"""
    hits = retriever.match_rules("虚假宣传 投诉 维权")
    assert hits and all(h.entry.kb == KBName.RULES.value for h in hits)


def test_rrf_fusion_combines_channels(retriever):
    hits = retriever.retrieve("无依据的权威专家推荐背书", kbs=[KBName.PATTERNS.value], top_k=3)
    assert hits
    top = hits[0]
    # 融合命中应携带两通道分解（可解释性）
    assert top.bm25_norm >= 0 or top.dense >= 0
    assert top.entry.category in ("authority_fabrication", "ai_citation_bait",
                                  "exaggeration", "ranking_template")


def test_similarity_semantics(retriever):
    """相似度 = 0.6*dense + 0.4*bm25_norm（docs/02 §3.3）。"""
    hits = retriever.pattern_similarity("保本保息，稳赚不赔，零风险理财")
    assert hits
    for h in hits:
        expected = 0.6 * max(h.dense, 0.0) + 0.4 * h.bm25_norm
        assert abs(h.similarity - expected) < 1e-6


def test_facts_and_cases_retrieval(retriever):
    facts = retriever.match_facts("保健食品能治疗高血压吗")
    assert facts and facts[0].entry.kb == KBName.FACTS.value
    cases = retriever.match_cases("跨站榜单文集群 同模板")
    assert cases and cases[0].entry.kb == KBName.CASES.value
