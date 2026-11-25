# -*- coding: utf-8 -*-
"""知识库存储治理与嵌入后端（knowledge-retrieval 规格）。"""

import numpy as np
import pytest

from geo_guard.kb.bm25 import BM25Index, tokenize
from geo_guard.kb.embeddings import HashEmbedding
from geo_guard.kb.store import KnowledgeStore
from geo_guard.schema import KBEntry, KBName


def _entry(entry_id="kb1-test-001", **kw):
    base = dict(entry_id=entry_id, kb=KBName.PATTERNS.value, title="测试模式",
                content="识别要点：测试内容", source="测试来源", severity=0.7,
                category="exaggeration", metadata={}, checked_at="2026-11-25")
    base.update(kw)
    return KBEntry(**base)


def test_tokenize_cjk_bigram_and_numeric():
    toks = tokenize("2026年十大空气净化器排行榜，去除率达99.97%，符合GB2760标准")
    assert "2026" in toks            # 数字整体保留
    assert "99.97%" in toks          # 百分比不拆碎
    assert "gb2760" in toks          # 标准号小写整体
    assert "空气" in toks and "净化" in toks  # CJK bigram


def test_bm25_ranks_relevant_first():
    idx = BM25Index()
    idx.add_doc("a", "空气净化器排行榜 十大排名")
    idx.add_doc("b", "今天天气不错适合散步")
    ranked = idx.search("空气净化器排行榜")
    assert ranked and ranked[0][0] == "a"


def test_seed_build_populates_five_kbs(store):
    for kb in KBName:
        assert store.count(kb=kb.value) >= 3, f"{kb.value} 条目不足"


def test_seed_build_idempotent(config, store):
    """重复构建幂等（规格场景）。"""
    from geo_guard.ingest import IngestPipeline
    before = store.count(status=None)
    IngestPipeline(store, config).seed_from_dir()
    assert store.count(status=None) == before


def test_empty_source_rejected(store):
    with pytest.raises(ValueError, match="source"):
        store.upsert(_entry("kb1-bad-001", source="  "))


def test_pending_review_invisible(store, retriever):
    store.upsert(_entry("kb1-pend-001", title="独特测试词zzzqqq",
                        content="独特测试词zzzqqq 的识别要点", status="pending_review"))
    hits = retriever.match_patterns("独特测试词zzzqqq")
    assert all(h.entry.entry_id != "kb1-pend-001" for h in hits)
    # 转 active 后可见
    store.set_status("kb1-pend-001", "active")
    hits = retriever.match_patterns("独特测试词zzzqqq")
    assert any(h.entry.entry_id == "kb1-pend-001" for h in hits)


def test_versioned_update_keeps_history(store):
    store.upsert(_entry("kb1-ver-001", content="旧版本内容"))
    store.upsert(_entry("kb1-ver-001", content="新版本内容", version="1.1.0"))
    row = store.conn.execute(
        "SELECT version FROM kb_entry_versions WHERE entry_id='kb1-ver-001'").fetchone()
    assert row is not None and row["version"] == "1.0.0"  # 旧版本可查
    assert store.get("kb1-ver-001").content == "新版本内容"


def test_hash_embedding_deterministic_and_normalized():
    emb = HashEmbedding(384)
    v1 = emb.embed("同一段文本")
    v2 = emb.embed("同一段文本")
    assert np.allclose(v1, v2)
    assert abs(float(np.linalg.norm(v1)) - 1.0) < 1e-5
    assert v1.shape == (384,)
    # 不同文本向量不同
    assert not np.allclose(v1, emb.embed("完全不同的另一段文本内容"))


def test_store_reopen_recovers_index(config, store):
    """崩溃恢复：重启后从 SQLite 重建内存索引。"""
    n_before = store.count()
    db_path = store.db_path
    store.close()
    s2 = KnowledgeStore(config, db_path=db_path)
    assert s2.count() == n_before
    assert s2.bm25.doc_count > 0 and s2._matrix.shape[0] > 0
    s2.close()
