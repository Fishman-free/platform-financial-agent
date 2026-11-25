# -*- coding: utf-8 -*-
"""schema 往返序列化与 RiskLevel 半档语义。"""

from geo_guard.schema import (Claim, DAGNode, DialogueState, Evidence, Phase,
                              RiskLevel, RiskReport, TraceReport)


def test_risk_level_downgrade_half():
    assert RiskLevel.HIGH.downgrade_half() == RiskLevel.MEDIUM
    assert RiskLevel.MEDIUM.downgrade_half() == RiskLevel.LOW
    assert RiskLevel.LOW.downgrade_half() == RiskLevel.LOW  # 永不低于 LOW


def test_evidence_roundtrip():
    e = Evidence(channel="rhetoric", label="测试", score=0.8, span=(3, 10),
                 excerpt="abc", kb_refs=["kb1-x"], detail={"strength": "strong"})
    e2 = Evidence.from_dict(e.to_dict())
    assert e2.span == (3, 10) and e2.kb_refs == ["kb1-x"] and e2.detail["strength"] == "strong"


def test_report_roundtrip():
    r = RiskReport(report_id="rpt-1", level=RiskLevel.HIGH, score=0.9,
                   evidence=[Evidence(channel="injection", label="inj", score=1.0)],
                   claims=[Claim(text="治疗高血压", verdict="contradicted", ref="kb3-fact-0001")],
                   channel_scores={"rhetoric": 0.8, "network": None},
                   missing_channels=["network"], notes=["n"], degraded=["d"])
    r2 = RiskReport.from_dict(r.to_dict())
    assert r2.level == RiskLevel.HIGH
    assert r2.evidence[0].channel == "injection"
    assert r2.claims[0].verdict == "contradicted"
    assert r2.channel_scores["network"] is None


def test_dialogue_state_snapshot_roundtrip():
    s = DialogueState(session_id="s1", phase=Phase.EXECUTING,
                      dag=[DAGNode(node_id="t1", action="call:screen", kind="recommend")],
                      history=[{"role": "user", "content": "hi"}], turn=3)
    s2 = DialogueState.from_json(s.to_json())
    assert s2.phase == Phase.EXECUTING and s2.turn == 3
    assert s2.dag[0].kind == "recommend" and s2.history[0]["content"] == "hi"


def test_trace_report_serializable():
    t = TraceReport(report_id="trace-1", suspicion_level=RiskLevel.MEDIUM,
                    subject="清澜", concentration=0.5,
                    resonance=[{"category": "ai_citation_bait", "label": "x",
                                "source_indices": [0, 1]}])
    d = t.to_dict()
    assert d["suspicion_level"] == "medium" and d["resonance"][0]["source_indices"] == [0, 1]
