# -*- coding: utf-8 -*-
"""三层判定融合（docs/02 §5，geo-screening 规格，design.md D5）。

L1 规则兜底：高确定性模式（注入/黑名单/白名单异常）直接定档或封顶；
L2 加权融合：六通道线性和，缺失通道权重重归一化（缺失 != 无风险）；
L3 分档映射：T_high/T_low 切档 + 证据单薄降半档（规则定档不降）。

显式线性模型：每条 Evidence 携带 channel/score/weight，报告分数可逐项
分解复算——可解释性由数据结构保证，无需后置归因。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence, Set, Tuple

from ..config import Config
from ..schema import Claim, Evidence, RiskLevel

# 通道名 → 融合信号键（与 config.fusion_weights 对应）
CHANNEL_SIGNAL = {
    "rhetoric": "rhetoric",
    "structural": "structural",
    "factual": "factual",
    "source": "source",
    "case": "case",
    "comment": "source",      # 评论信号并入来源可信面
    "network": "network",
}


class Calibrator(Protocol):
    """分数→阈值校准器接口（I-13）。本期实现 AnchorCalibrator，
    团队标注数据充足后可替换 Platt/Isotonic 实现。"""

    def fit(self, samples: Sequence[Tuple[float, str]]) -> "Calibrator": ...
    def thresholds(self) -> Tuple[float, float]: ...


class AnchorCalibrator:
    """锚点法：用少量已标注样本观察分数分布，取分位点定阈（对抗评审 #4 决议）。"""

    def __init__(self, t_high: float = 0.72, t_low: float = 0.40):
        self.t_high = t_high
        self.t_low = t_low

    def fit(self, samples: Sequence[Tuple[float, str]]) -> "AnchorCalibrator":
        pos = sorted(s for s, lab in samples if lab in ("geo_poisoning", "high"))
        neg = sorted(s for s, lab in samples if lab in ("benign", "low"))
        if len(pos) >= 3 and len(neg) >= 3:
            # t_low 取良性上四分位与投毒下四分位的中点；t_high 取投毒中位数
            import statistics
            neg_q3 = neg[int(0.75 * (len(neg) - 1))]
            pos_q1 = pos[int(0.25 * (len(pos) - 1))]
            self.t_low = round((neg_q3 + pos_q1) / 2, 3)
            self.t_high = round(statistics.median(pos), 3)
            if self.t_high <= self.t_low:
                self.t_high = min(0.95, self.t_low + 0.15)
        return self

    def thresholds(self) -> Tuple[float, float]:
        return self.t_high, self.t_low


@dataclass
class FusionResult:
    level: RiskLevel
    score: float
    confidence: str                     # rule | model | thin
    channel_scores: Dict[str, Optional[float]]
    channel_contrib: Dict[str, float]   # 各通道加权贡献（复算用）
    missing_channels: List[str]
    notes: List[str] = field(default_factory=list)


class FusionEngine:
    def __init__(self, config: Optional[Config] = None,
                 calibrator: Optional[Calibrator] = None,
                 blacklist_domains: Optional[Set[str]] = None):
        self.config = config or Config()
        self.calibrator = calibrator or AnchorCalibrator(self.config.t_high, self.config.t_low)
        self.blacklist = blacklist_domains or set()

    # ---------------- L2：通道分计算 ----------------

    def channel_scores(self, evidences: List[Evidence], claims: List[Claim],
                       source_credibility: float, case_signal: Optional[float],
                       has_claims: bool,
                       network_signal: Optional[float] = None) -> Dict[str, Optional[float]]:
        w = self.config.fusion_weights
        scores: Dict[str, Optional[float]] = {k: None for k in w}

        # s1 话术：最强命中 × severity；>=3 个不同模式时 top3 均值加成
        rhe = [e for e in evidences if e.channel == "rhetoric"]
        if rhe:
            rhe_sorted = sorted(rhe, key=lambda e: e.score * (0.5 + e.severity), reverse=True)
            base = rhe_sorted[0].score * (0.4 + 0.6 * rhe_sorted[0].severity)
            if len(rhe_sorted) >= 3:
                top3 = sum(e.score * (0.4 + 0.6 * e.severity) for e in rhe_sorted[:3]) / 3
                base = min(1.0, top3 * self.config.multi_pattern_boost)
            scores["rhetoric"] = round(min(base, 1.0), 4)

        # s2 结构：全部结构特征均值（含 cross_validation_web 端侧投影——它是
        # 结构可见的局部协同信号；完整图谱证据走 s6 云侧槽位）
        struct = [e for e in evidences if e.channel == "structural"]
        if struct:
            scores["structural"] = round(
                sum(e.score * e.severity for e in struct) / len(struct), 4)

        # s3 事实：(contradicted + 0.5*unverified) / claims 总数
        if has_claims and claims:
            contra = sum(1 for c in claims if c.verdict == "contradicted")
            unver = sum(1 for c in claims if c.verdict == "unverified" and c.ref)
            scores["factual"] = round((contra + 0.5 * unver) / len(claims), 4)

        # s4 来源：1 - 可信度；评论水军/负评证据上调
        s4 = 1.0 - source_credibility
        comm = [e for e in evidences if e.channel == "comment"]
        if comm:
            s4 = min(1.0, s4 + max(e.score * 0.5 for e in comm))
        scores["source"] = round(s4, 4)

        # s5 判例
        if case_signal is not None:
            scores["case"] = round(case_signal, 4)

        # s6 协同（云侧 F6 槽位）：端侧无数据 → None → 权重重归一化（I-11）
        if network_signal is not None:
            scores["network"] = round(max(0.0, min(1.0, network_signal)), 4)
        return scores

    # ---------------- 主入口 ----------------

    def fuse(self, evidences: List[Evidence], claims: List[Claim],
             source_credibility: float, domain: str = "",
             case_signal: Optional[float] = None,
             whitelist_hit: bool = False,
             has_claims: bool = True,
             network_signal: Optional[float] = None) -> FusionResult:
        notes: List[str] = []
        scores = self.channel_scores(evidences, claims, source_credibility,
                                     case_signal, has_claims, network_signal)

        # ---- L1 规则兜底 ----
        inj = [e for e in evidences if e.channel == "injection"]
        if inj:
            return FusionResult(RiskLevel.HIGH, 1.0, "rule", scores,
                                self._contrib(scores), self._missing(scores),
                                ["命中显式指令注入模式（L1 规则兜底，不依赖统计信号）"])
        if domain and domain in self.blacklist:
            return FusionResult(RiskLevel.HIGH, 1.0, "rule", scores,
                                self._contrib(scores), self._missing(scores),
                                [f"域名 {domain} 在确认黑名单（云侧下发）"])

        # ---- L2 加权融合（缺失通道重归一化）----
        w = self.config.fusion_weights
        avail = {k: v for k, v in scores.items() if v is not None}
        missing = [k for k, v in scores.items() if v is None]
        if missing:
            notes.append("缺失通道按未知处理（不计为无风险）：" + ", ".join(missing))
        total_w = sum(w[k] for k in avail) or 1.0
        score = sum(w[k] * v for k, v in avail.items()) / total_w
        score = round(min(max(score, 0.0), 1.0), 4)
        contrib = {k: round(w[k] * v / total_w, 4) for k, v in avail.items()}

        # ---- L3 分档 ----
        t_high, t_low = self.calibrator.thresholds()
        if score >= t_high:
            level = RiskLevel.HIGH
        elif score >= t_low:
            level = RiskLevel.MEDIUM
        else:
            level = RiskLevel.LOW

        # 证据单薄降档：证据 < thin_count 或全部来自单一通道（规则定档不降）
        real_evs = [e for e in evidences if e.channel != "injection"]
        channels_used: Set[str] = {CHANNEL_SIGNAL.get(e.channel, e.channel) for e in real_evs
                                   if e.score > 0.05}
        confidence = "model"
        if len(real_evs) < self.config.thin_evidence_count or len(channels_used) <= 1:
            if level != RiskLevel.LOW:
                level = level.downgrade_half()
                confidence = "thin"
                notes.append("证据单薄（数量不足或单一通道），已降半档；建议多源交叉验证")
            if len(real_evs) <= 1 and real_evs:
                notes.append("单一信号，建议交叉验证")

        # ---- L1 白名单封顶（在统计分档之后生效）----
        if whitelist_hit:
            strong = any(e.channel == "rhetoric"
                         and e.detail.get("strength") == "strong" for e in evidences)
            contradicted = any(c.verdict == "contradicted" for c in claims)
            if strong or contradicted:
                if level == RiskLevel.HIGH:
                    level = RiskLevel.MEDIUM
                notes.append("权威白名单站点内容异常（话术强命中或事实矛盾），封顶 MEDIUM，建议人工复核")
            else:
                level = RiskLevel.LOW
                notes.append("来源命中权威白名单且无话术强命中，判 LOW")
        return FusionResult(level, score, confidence, scores, contrib, missing, notes)

    @staticmethod
    def _contrib(scores: Dict[str, Optional[float]]) -> Dict[str, float]:
        return {k: round(v or 0.0, 4) for k, v in scores.items()}

    @staticmethod
    def _missing(scores: Dict[str, Optional[float]]) -> List[str]:
        return [k for k, v in scores.items() if v is None]
