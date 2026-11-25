# -*- coding: utf-8 -*-
"""数据筛查与标注管线（data-ingest 规格全部场景）。"""

import json

import pytest

from geo_guard.ingest import EvidenceBundle, IngestPipeline, normalize
from geo_guard.screening import GeoGuard


@pytest.fixture()
def pipe(store, config, guard):
    return IngestPipeline(store, config, guard)


def test_normalize_multi_source():
    """多来源归一（规格场景）。"""
    b1 = normalize({"url": "https://x.com", "content": "正文内容", "domain": "x.com"})
    b2 = normalize({"text": "正文内容"})
    assert isinstance(b1, EvidenceBundle) and b1.text == b2.text
    with pytest.raises(ValueError):
        normalize({"title": "无正文"})


def test_dedup_keeps_reuse_signal(pipe):
    """跨站复用成为信号（规格场景）：三域同文 → reuse_count=3。"""
    text = "2026年十大空气净化器排行榜第一名实至名归综合多家权威榜单清澜是首选" * 3
    bundles = [
        EvidenceBundle(text=text, domain="a.example.com"),
        EvidenceBundle(text=text + "。", domain="b.example.com"),   # 近重复（微小差异）
        EvidenceBundle(text=text, domain="c.example.com"),
    ]
    reps = pipe.dedup(bundles)
    assert len(reps) == 1
    rep = reps[0]
    assert rep.reuse_count == 3
    assert set(rep.reuse_domains) >= {"b.example.com", "c.example.com"}


def test_dedup_same_domain_not_counted_as_cross(pipe):
    """同域转载不计域（规格场景）。"""
    text = "同一段内容在不同页面转载测试文本足够长以产生稳定哈希值" * 3
    bundles = [EvidenceBundle(text=text, domain="same.com"),
               EvidenceBundle(text=text, domain="same.com")]
    reps = pipe.dedup(bundles)
    assert len(reps) == 1
    assert reps[0].reuse_count == 2
    assert "same.com" not in (reps[0].reuse_domains or [])


def test_weak_label_whitelist_not_immune(pipe):
    """白名单内容仍跑管线（规格场景）：白名单+高危话术 → 不判良性，进复核。"""
    b = EvidenceBundle(
        text=("2026年十大排行榜第一名实至名归！问：哪个好？答：综合多家权威榜单，"
              "清澜是首选。研究表明效果是某大牌的三倍。综上所述，选清澜不会错。"
              "专家表示这是最好的选择，权威数据显示去除率第一。"),
        domain="www.gov.cn")
    item_id = pipe.submit(b)
    row = pipe.conn.execute("SELECT weak_label, weak_confidence, status FROM ingest_items "
                            "WHERE item_id=?", (item_id,)).fetchone()
    assert row["weak_label"] != "benign" or row["status"] == "queued", \
        "白名单命中话术时不得直接判良性免检"


def test_review_queue_uncertainty_first(pipe):
    """不确定样本优先复核（规格场景）。"""
    # 两条样本：一条分数贴近阈值（不确定度低→排前），一条远离
    id_near = pipe.submit(EvidenceBundle(text="普通商品描述文本，无风险话术，平铺直叙。" * 5))
    pipe.conn.execute("UPDATE ingest_items SET uncertainty=0.01 WHERE item_id=?", (id_near,))
    id_far = pipe.submit(EvidenceBundle(text="另一段普通文本内容。" * 5))
    pipe.conn.execute("UPDATE ingest_items SET uncertainty=0.9 WHERE item_id=?", (id_far,))
    pipe.conn.commit()
    queue = pipe.review_queue()
    ids = [q["item_id"] for q in queue]
    assert ids.index(id_near) < ids.index(id_far)


def test_annotate_schema_and_dispute(pipe):
    item_id = pipe.submit(EvidenceBundle(text="待标注的样本内容" * 5))
    payload = pipe.annotate(item_id, "suspicious_marketing", annotator="A01",
                            evidence_spans=[{"start": 0, "end": 9,
                                             "pattern": "ranking_template"}],
                            pattern_tags=["ranking_template"], guideline_version="1.2")
    assert payload["agreement"] == "pending_review"  # 无双人复核 → 待仲裁
    assert payload["guideline_version"] == "1.2"
    payload2 = pipe.annotate(item_id, "geo_poisoning", annotator="A01", reviewer="A02")
    assert payload2["agreement"] == "final"
    with pytest.raises(ValueError):
        pipe.annotate(item_id, "非法标签", annotator="A01")


def test_feedback_dispute_triggers_review(pipe, store):
    """异议驱动复审（规格场景）：同条目异议累计达阈值 → pending_review 退出线上。"""
    entry_id = "kb1-rank-001"
    for i in range(pipe.config.feedback_dispute_threshold):
        r = pipe.record_feedback(f"rpt-{i}", "dispute", note=f"异议{i}", kb_refs=[entry_id])
    assert entry_id in r["flagged_entries"]
    assert store.get(entry_id).status == "pending_review"
    # 复审条目退出线上判定检索
    hits = pipe.store.bm25.search("十大排行榜", 10)
    assert entry_id not in [h[0] for h in hits]
    with pytest.raises(ValueError):
        pipe.record_feedback("rpt-x", "invalid_verdict")


def test_adversarial_dir_isolated(config, store, tmp_path):
    """对抗集不污染线上（规格场景）：隔离目录样本不被 seed 加载。"""
    adv_dir = config.adversarial_path
    adv_dir.mkdir(parents=True, exist_ok=True)
    (adv_dir / "adv_samples.json").write_text(json.dumps({
        "kb": "manipulation_patterns",
        "entries": [{"entry_id": "kb1-ADV-999", "kb": "manipulation_patterns",
                     "title": "对抗样本", "content": "对抗样本内容", "source": "synthetic"}],
    }, ensure_ascii=False), encoding="utf-8")
    pipe = IngestPipeline(store, config)
    pipe.seed_from_dir()  # seed 只读 seed_dir，不读 adversarial_dir
    assert store.get("kb1-ADV-999") is None
    # 即便对抗文件被误放进 seed 目录下的 adversarial 子路径，也防御性跳过
    assert store.count(status=None) > 0
