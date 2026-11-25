# -*- coding: utf-8 -*-
"""可解释输出（docs/02 §5.4，docs/06 §4）：证据链卡片、分级披露、
降级模式标注、"风险提示"措辞纪律（禁止法律结论表述）。

三档披露：summary（结论+top3）/ standard（+全部证据+通道分解）/
full（+claims 明细+缺失通道+复算贡献）。默认档由画像 evidence_appetite 决定。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .config import Config
from .schema import RiskReport, RiskLevel, TraceReport

_LEVEL_BADGE = {
    RiskLevel.HIGH: "🔴 高风险",
    RiskLevel.MEDIUM: "🟡 中风险",
    RiskLevel.LOW: "🟢 低风险",
}

# 措辞纪律：输出必须使用"风险提示"框架，禁止法律结论式断言
_FORBIDDEN_PHRASES = ("构成违法", "肯定合法", "绝对安全", "构成犯罪", "必定是骗局")


def render_report(report: RiskReport, detail: str = "summary",
                  config: Optional[Config] = None,
                  style_hint: str = "") -> str:
    """渲染风险报告。detail ∈ summary|standard|full。"""
    cfg = config or Config()
    lines: List[str] = []
    if report.degraded:
        lines.append("⚠️ " + "；".join(report.degraded) + "（功能完整，语言与语义泛化降级）")
        lines.append("")
    badge = _LEVEL_BADGE[report.level]
    lines.append(f"{badge}（风险分 {report.score:.2f} · 置信类型 {report.confidence}）")
    lines.append(f"对象：{report.subject or '（未识别主体）'}")
    lines.append("")
    lines.append("【风险提示】本结果为多信号融合的风险提示，非法律结论。")
    lines.append("")

    # 证据卡片：默认 top3，超额折叠（dialogue-guidance 规格）
    evidences = _sorted_evidence(report)
    shown = evidences[:cfg.max_evidence_cards] if detail == "summary" else evidences
    lines.append(f"证据链（{len(evidences)} 条）：")
    for i, e in enumerate(shown, 1):
        lines.append(f"  [证据{i}] {_evidence_line(e)}")
    if len(evidences) > len(shown):
        lines.append(f"  ……还有 {len(evidences) - len(shown)} 条证据，回复\"展开\"查看")
    lines.append("")

    if detail in ("standard", "full"):
        lines.append("通道分解：" + _channel_breakdown(report))
        lines.append("")
    if detail == "full":
        if report.claims:
            lines.append("事实核验明细：")
            for c in report.claims:
                mark = {"supported": "✓一致", "contradicted": "✗矛盾",
                        "unverified": "?无法确证"}[c.verdict]
                lines.append(f"  - [{mark}] {c.text[:60]}"
                             + (f"（依据：{c.authority}）" if c.authority else ""))
            lines.append("")
        if report.missing_channels:
            lines.append("缺失通道（按未知处理，不计为无风险）："
                         + ", ".join(report.missing_channels))
            lines.append("")
    if report.notes:
        lines.append("说明：")
        for n in report.notes:
            lines.append(f"  · {n}")
        lines.append("")
    if report.proactive_prompts:
        lines.append("可以继续问我：")
        for p in report.proactive_prompts:
            lines.append(f"  → {p}")
    text = "\n".join(lines)
    return _wording_guard(text)


def render_trace(report: TraceReport, detail: str = "summary") -> str:
    lines: List[str] = []
    if report.degraded:
        lines.append("⚠️ " + "；".join(report.degraded))
        lines.append("")
    badge = _LEVEL_BADGE[report.suspicion_level]
    lines.append(f"【AI 回答回溯】{badge}（疑似被 GEO 牵引 · 集中度 {report.concentration:.2f}）")
    if report.subject:
        lines.append(f"回答推荐主体：{report.subject}")
    lines.append("")
    lines.append("本结果为风险提示，非法律结论。")
    lines.append("")
    if report.per_source:
        lines.append("逐源风险：")
        for s in report.per_source:
            lines.append(f"  [源{s.index + 1}] {_LEVEL_BADGE[s.level]} "
                         f"{s.title or s.url or '(无标题)'}"
                         + (f" 命中：{', '.join(s.patterns)}" if s.patterns else ""))
        lines.append("")
    if report.resonance:
        lines.append("话术共振（多源命中同一模式——投毒集群的端侧可见投影）：")
        for r in report.resonance:
            idx = ", ".join(f"源{i + 1}" for i in r["source_indices"])
            lines.append(f"  · {r['label']}（{r['category']}）：{idx}")
        lines.append("")
    if report.missing_negatives:
        lines.append("被过滤的反面证据（回答未呈现）：")
        for c in report.missing_negatives[:5]:
            lines.append(f"  · {c.text[:80]}" + (f"（依据：{c.authority}）" if c.authority else ""))
        if detail == "summary" and len(report.missing_negatives) > 5:
            lines.append(f"  ……还有 {len(report.missing_negatives) - 5} 条，回复\"展开\"查看")
        lines.append("")
    if report.notes:
        lines.append("说明：")
        for n in report.notes:
            lines.append(f"  · {n}")
    return _wording_guard("\n".join(lines))


def render_actions(actions: List[Dict[str, Any]]) -> str:
    if not actions:
        return "（平台规则库暂无匹配条目；建议直接通过 12315 平台反映）"
    lines = ["【行动引导】以下为维权路径参考（以平台/监管部门最新规则为准）："]
    for a in actions:
        stale = "（⚠️ 条目超过核查周期，请自行核实最新流程）" if a.get("stale") else ""
        lines.append(f"\n◆ {a['title']}{stale}")
        if a.get("channel"):
            lines.append(f"  渠道：{a['channel']}")
        if a.get("materials"):
            lines.append(f"  材料：{'、'.join(a['materials'])}")
        if a.get("deadline_hint"):
            lines.append(f"  时限：{a['deadline_hint']}")
        lines.append(f"  来源：{a['kb_ref']}（核查于 {a['checked_at']}）")
    return "\n".join(lines)


def _sorted_evidence(report: RiskReport) -> List:
    """证据排序：通道严重度 × 分数，注入类置顶。"""
    def key(e):
        base = e.score * (0.5 + e.severity)
        if e.channel == "injection":
            base += 10
        return base
    return sorted(report.evidence, key=key, reverse=True)


def _evidence_line(e) -> str:
    parts = [f"{e.label}（强度 {e.score:.2f}）"]
    expl = e.detail.get("explanation") if e.detail else None
    if expl:
        parts.append(expl)
    if e.excerpt:
        excerpt = e.excerpt.replace("\n", " ")[:60]
        parts.append(f"原文：\"{excerpt}\"")
    if e.kb_refs:
        parts.append(f"溯源：{', '.join(e.kb_refs)}")
    return " | ".join(parts)


def _channel_breakdown(report: RiskReport) -> str:
    items = []
    for ch, sc in report.channel_scores.items():
        items.append(f"{ch}={'缺失' if sc is None else f'{sc:.2f}'}")
    return " ".join(items)


def _wording_guard(text: str) -> str:
    """措辞校验：法律结论式断言替换为风险提示框架（agent-integration 规格）。"""
    replacements = {
        "构成违法": "涉嫌违反相关规定（以监管部门认定为准）",
        "肯定合法": "未发现明显风险信号",
        "绝对安全": "风险较低",
        "构成犯罪": "涉嫌违法（以司法机关认定为准）",
        "必定是骗局": "存在多项骗术特征信号",
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    return text
