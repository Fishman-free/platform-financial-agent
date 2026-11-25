# -*- coding: utf-8 -*-
"""可解释输出（分级披露 / 措辞纪律 / 降级标注）。"""

from conftest import SOFT_AD

from geo_guard.explain import render_actions, render_report, render_trace
from geo_guard.schema import Claim, Evidence, RiskLevel, RiskReport, SourceRisk, TraceReport


def _fake_report(n_evidence=7, level=RiskLevel.MEDIUM):
    evs = [Evidence(channel="rhetoric", label=f"模式{i}", score=0.9 - i * 0.05,
                    severity=0.8, span=(0, 10), excerpt=f"片段{i}",
                    kb_refs=[f"kb1-x-{i}"], detail={"category": "exaggeration"})
           for i in range(n_evidence)]
    return RiskReport(report_id="rpt-test", level=level, score=0.66,
                      evidence=evs, claims=[Claim(text="治疗高血压", verdict="contradicted",
                                                  ref="kb3-fact-0001", authority="《广告法》第十八条")],
                      channel_scores={"rhetoric": 0.8, "network": None},
                      missing_channels=["network"], notes=["测试说明"],
                      degraded=["LLM 降级"], proactive_prompts=["要查历史差评吗？"],
                      subject="测试主体")


def test_summary_folds_beyond_top3():
    """折叠超额证据（规格场景）：7 条证据默认只呈现 top-3。"""
    out = render_report(_fake_report(7), detail="summary")
    shown = out.count("[证据")
    assert shown == 3, f"summary 档应只展示 3 张证据卡片，实际 {shown}"
    assert "还有 4 条证据" in out


def test_standard_shows_all_and_breakdown():
    out = render_report(_fake_report(7), detail="standard")
    assert out.count("[证据") == 7
    assert "通道分解" in out and "network=缺失" in out


def test_full_shows_claims_and_missing():
    out = render_report(_fake_report(3), detail="full")
    assert "事实核验明细" in out and "矛盾" in out
    assert "缺失通道" in out and "network" in out


def test_degraded_banner_and_wording():
    out = render_report(_fake_report(1, RiskLevel.HIGH))
    assert "降级" in out  # 降级模式头部标注
    assert "风险提示" in out and "非法律结论" in out
    # 措辞纪律：法律结论式断言被替换
    from geo_guard.explain import _wording_guard
    assert "构成违法" not in _wording_guard("该内容构成违法")
    assert "监管部门认定" in _wording_guard("该内容构成违法")


def test_render_trace_sections():
    t = TraceReport(report_id="tr-1", suspicion_level=RiskLevel.HIGH, subject="清澜",
                    per_source=[SourceRisk(index=0, title="源A", level=RiskLevel.HIGH,
                                           score=0.9, patterns=["ai_citation_bait"])],
                    concentration=0.8,
                    resonance=[{"category": "ai_citation_bait", "label": "AI 引用诱导",
                                "source_indices": [0, 1]}],
                    missing_negatives=[Claim(text="治疗声明矛盾", verdict="contradicted")],
                    notes=["共振说明"], degraded=["LLM 降级"])
    out = render_trace(t)
    assert "AI 回答回溯" in out and "话术共振" in out
    assert "被过滤的反面证据" in out and "源1" in out


def test_render_actions_staleness():
    out = render_actions([{"title": "电商投诉", "channel": "订单页", "materials": ["快照"],
                           "deadline_hint": "尽早", "checked_at": "2020-01-01",
                           "kb_ref": "kb5-rule-0001", "stale": True}])
    assert "超过核查周期" in out
    assert render_actions([])  # 空列表有兜底文案
