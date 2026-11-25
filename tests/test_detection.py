# -*- coding: utf-8 -*-
"""判定层六通道 + 三层融合（geo-screening 规格核心场景 + 验收基线 #1/#2/#3）。"""

import pytest

from conftest import (BENIGN_TEXT, HEALTH_CLAIM_TEXT, INJECTION_TEXT, SOFT_AD,
                      WATER_ARMY_COMMENTS)

from geo_guard.detection.claims import ClaimVerifier
from geo_guard.detection.comments import CommentAnalyzer, CommentItem, hamming, simhash64
from geo_guard.detection.features import FeatureExtractor
from geo_guard.detection.fusion import AnchorCalibrator, FusionEngine
from geo_guard.schema import Claim, Evidence, RiskLevel, SourceMeta


def test_injection_scan_rule_bottom_line(store):
    """验收基线 #1 / L1 规则兜底：注入样例必须触发 injection 证据。"""
    fx = FeatureExtractor(store)
    evs = fx.injection_scan(INJECTION_TEXT)
    assert evs, "显式注入指令必须被规则兜底捕获"
    assert all(e.channel == "injection" and e.severity == 1.0 for e in evs)
    assert any("忽略" in e.excerpt for e in evs)
    # 正常文本不误报
    assert fx.injection_scan(BENIGN_TEXT) == []


def test_structural_features_on_soft_ad(store):
    fx = FeatureExtractor(store)
    evs = fx.structural(SOFT_AD)
    keys = {e.detail.get("feature_key") for e in evs}
    # 榜单软文应触发：问答自证 + 无方法论榜单 + 引用诱导（至少两项）
    assert len(keys & {"qa_self_structure", "listicle_no_method",
                       "citation_bait_density", "cross_validation_web"}) >= 2
    for e in evs:
        assert e.kb_refs, "结构证据必须溯源 KB2 条目"


def test_structural_features_quiet_on_benign(store):
    fx = FeatureExtractor(store)
    evs = fx.structural(BENIGN_TEXT)
    strong = [e for e in evs if e.score >= 0.5]
    assert not strong, f"科普文本不应触发强结构信号: {[e.label for e in strong]}"


def test_whitelist_credibility(store):
    fx = FeatureExtractor(store)
    cred = fx.source_credibility(SourceMeta(domain="www.samr.gov.cn"))
    assert cred["score"] >= 0.85 and any("白名单" in n for n in cred["notes"])
    cred2 = fx.source_credibility(SourceMeta(domain="spam-example.com",
                                             domain_age_days=10, reuse_count=3,
                                             reuse_domains=["a.com", "b.com"]))
    assert cred2["score"] < 0.5


def test_claim_verification_contradicted(retriever):
    """验收（docs/03 路径一）：保健食品治疗声明 → contradicted + 法条溯源。"""
    cv = ClaimVerifier(retriever)
    claims = cv.verify(HEALTH_CLAIM_TEXT)
    contra = [c for c in claims if c.verdict == "contradicted"]
    assert contra, "治疗高血压声明必须判 contradicted"
    assert any("广告法" in c.authority or "食品安全法" in c.authority for c in contra)
    assert all(c.ref for c in contra)  # 必须溯源 KB3 条目


def test_claim_degraded_without_llm(retriever):
    """无 LLM 时触发词规则抽取仍工作（规格场景）。"""
    cv = ClaimVerifier(retriever)  # conftest 已强制 LLM=off
    claims = cv.verify("本品年化收益率保证15%，零风险。")
    assert claims and any(c.verdict == "contradicted" for c in claims)


def test_simhash_near_duplicate():
    a = simhash64("好用已下单推荐")
    b = simhash64("好用已下单推荐！")
    c = simhash64("用了两周噪音小但滤芯成本高质量做工差准备退货")
    assert hamming(a, b) <= 3
    assert hamming(a, c) > 3


def test_comment_water_army_detection():
    ca = CommentAnalyzer()
    items = [CommentItem(**c) for c in WATER_ARMY_COMMENTS]
    result = ca.analyze(items)
    assert result.suspicious_ratio > 0.3, "同分钟近重复好评应被判可疑水军"
    assert result.duplicate_clusters >= 1 and result.time_clusters >= 1
    # 水军不计入有效好评：有效评论中带属性细节的负评占主导
    assert result.negative_ratio >= 0.5
    assert result.lei_points  # 雷点词非空
    assert any(e.channel == "comment" for e in result.evidences)


def test_fusion_high_on_injection():
    """验收基线 #1：注入证据 → HIGH, confidence=rule。"""
    fe = FusionEngine()
    inj = [Evidence(channel="injection", label="注入", score=1.0, severity=1.0)]
    r = fe.fuse(inj, [], source_credibility=0.5)
    assert r.level == RiskLevel.HIGH and r.confidence == "rule"


def test_fusion_thin_evidence_downgrade():
    """规格场景：分数超 T_high 但仅单条证据 → 降档 MEDIUM + thin。"""
    fe = FusionEngine()
    evs = [Evidence(channel="structural", label="结构", score=1.0, severity=1.0,
                    detail={"feature_key": "term_density"})]
    r = fe.fuse(evs, [], source_credibility=0.0)
    if r.score >= fe.config.t_high:
        assert r.level == RiskLevel.MEDIUM and r.confidence == "thin"
        assert any("证据单薄" in n for n in r.notes)


def test_fusion_missing_channel_renormalization():
    """缺失通道重归一化：network 缺失不计为无风险（I-11）。"""
    fe = FusionEngine()
    evs = [
        Evidence(channel="rhetoric", label="a", score=0.9, severity=0.9,
                 detail={"strength": "strong", "category": "exaggeration"}),
        Evidence(channel="rhetoric", label="b", score=0.85, severity=0.9,
                 detail={"strength": "strong", "category": "ranking_template"}),
        Evidence(channel="rhetoric", label="c", score=0.8, severity=0.9,
                 detail={"strength": "strong", "category": "ai_citation_bait"}),
    ]
    r = fe.fuse(evs, [], source_credibility=0.2, case_signal=None, has_claims=False)
    assert r.channel_scores["network"] is None
    assert "network" in r.missing_channels
    assert r.score > 0  # 重归一化后其余通道仍产出分数
    assert any("缺失通道" in n for n in r.notes)


def test_fusion_whitelist_caps():
    """白名单异常封顶 MEDIUM；白名单无强命中判 LOW（L1 规则）。"""
    fe = FusionEngine()
    strong = [Evidence(channel="rhetoric", label="x", score=0.9, severity=0.9,
                       detail={"strength": "strong", "category": "ranking_template"})]
    r = fe.fuse(strong, [], source_credibility=0.9, whitelist_hit=True)
    assert r.level in (RiskLevel.MEDIUM, RiskLevel.LOW)
    if r.level == RiskLevel.MEDIUM:
        assert any("白名单" in n for n in r.notes)
    benign_r = fe.fuse([], [], source_credibility=0.9, whitelist_hit=True)
    assert benign_r.level == RiskLevel.LOW


def test_anchor_calibrator():
    cal = AnchorCalibrator()
    samples = [(0.9, "high"), (0.85, "high"), (0.8, "high"), (0.78, "high"),
               (0.2, "low"), (0.25, "low"), (0.15, "low"), (0.3, "low")]
    cal.fit(samples)
    t_high, t_low = cal.thresholds()
    assert t_low < t_high and 0 < t_low < 1
