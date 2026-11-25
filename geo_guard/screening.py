# -*- coding: utf-8 -*-
"""GeoGuard 门面：双对象检测（docs/02 §6，geo-screening 规格）。

对象 A：screen_content —— 单篇内容甄别，六通道管线 → RiskReport。
对象 B：trace_ai_answer —— AI 回答逆向回溯（来源集中度/话术共振度/
反证缺失度）→ TraceReport。"引导用户了解被 GEO 操纵过滤掉的负面反馈"
（立项书中期目标）的直接实现。
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

from .config import WHITELIST_DOMAINS, Config
from .detection.claims import ClaimVerifier
from .detection.comments import CommentAnalyzer, CommentItem
from .detection.features import FeatureExtractor
from .detection.fusion import FusionEngine
from .detection.patterns import PatternMatcher
from .kb.store import KnowledgeStore
from .llm import LLMBackend, NullBackend, get_backend
from .retrieval import HybridRetriever
from .schema import (Claim, Evidence, RiskLevel, RiskReport, SourceMeta,
                     SourceRisk, TraceReport)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _report_id(prefix: str, *parts: str) -> str:
    h = hashlib.sha256("\x00".join(parts).encode("utf-8")).hexdigest()[:12]
    return f"{prefix}-{h}"


class GeoGuard:
    """内核门面：插件与 Personal Agent 共享的唯一入口。"""

    def __init__(self, store: Optional[KnowledgeStore] = None,
                 config: Optional[Config] = None,
                 llm: Optional[LLMBackend] = None):
        self.config = config or Config()
        self.store = store or KnowledgeStore(self.config)
        self.llm = llm or get_backend(self.config.llm, self.config.llm_model)
        self.retriever = HybridRetriever(self.store, self.config, self.llm)
        self.patterns = PatternMatcher(self.retriever, self.config)
        self.features = FeatureExtractor(self.store, self.config)
        self.claims = ClaimVerifier(self.retriever, self.llm, self.config)
        self.comments = CommentAnalyzer(self.config)
        self.fusion = FusionEngine(self.config)
        self._screen_cache: Dict[str, RiskReport] = {}

    # ==================================================================
    # 对象 A：单篇内容甄别
    # ==================================================================

    def screen_content(self, text: str,
                       source_meta: Optional[Dict[str, Any]] = None,
                       comments: Optional[Sequence[Dict[str, Any]]] = None,
                       subject: str = "") -> RiskReport:
        text = (text or "").strip()
        if not text:
            raise ValueError("screen_content 需要非空文本")
        # 只接受 SourceMeta 已知字段（宿主传入多余键不致崩溃）
        known = set(SourceMeta.__dataclass_fields__)  # type: ignore[attr-defined]
        meta = SourceMeta(**{k: v for k, v in (source_meta or {}).items() if k in known})
        rid = _report_id("rpt", text[:2000], meta.domain, meta.author)
        if rid in self._screen_cache:
            return self._screen_cache[rid]

        degraded: List[str] = []
        if self.llm.name == "null":
            degraded.append("LLM 降级：意图/抽取/润色走规则路径")
        if getattr(self.store.embedder, "name", "") == "hash":
            degraded.append("嵌入降级：本地哈希嵌入（语义泛化受限，阈值已自动偏移）")

        # 六通道证据收集
        evidences: List[Evidence] = []
        evidences += self.features.injection_scan(text)          # F3（L1 兜底）
        evidences += self.patterns.match(text)                   # F1
        evidences += self.features.structural(text, meta)        # F2 + F6 投影
        claim_list: List[Claim] = self.claims.verify(text)       # F5
        evidences += ClaimVerifier.to_evidences(claim_list)
        cred = self.features.source_credibility(meta)            # F4
        evidences += cred["evidences"]
        comment_items = [CommentItem(**c) for c in (comments or [])]
        comment_analysis = self.comments.analyze(comment_items) if comment_items else None
        if comment_analysis:
            evidences += comment_analysis.evidences

        # 判例信号（s5）：相似案例 outcome 映射
        case_signal = None
        case_hits = self.retriever.match_cases(text, top_k=2)
        if case_hits and case_hits[0].similarity >= 0.3:
            outcome = (case_hits[0].entry.metadata or {}).get("outcome", "")
            case_signal = {"high": 1.0, "medium": 0.6,
                           "low_credit": 0.7, "high_credit": 0.0}.get(outcome)
            if case_signal is not None:
                evidences.append(Evidence(
                    channel="case", label=f"相似判例：{case_hits[0].entry.title}",
                    score=case_signal, severity=case_hits[0].entry.severity,
                    kb_refs=[case_hits[0].entry.entry_id],
                    detail={"similarity": round(case_hits[0].similarity, 3)}))

        whitelist_hit = any("命中权威白名单" in n for n in cred["notes"])
        result = self.fusion.fuse(
            evidences, claim_list, cred["score"], domain=meta.domain,
            case_signal=case_signal, whitelist_hit=whitelist_hit,
            has_claims=bool(claim_list),
            network_signal=meta.network_evidence)  # F6 云侧槽位（端侧恒 None）

        notes = list(result.notes) + cred["notes"]
        if "network" in result.missing_channels:
            notes.append("本页级判定，未含跨站协同分析（云侧图谱接入后消除该盲区）")
        if comment_analysis and comment_analysis.lei_points:
            notes.append("评论区雷点：" + "、".join(comment_analysis.lei_points[:5]))

        report = RiskReport(
            report_id=rid,
            level=result.level,
            score=result.score,
            confidence=result.confidence,
            evidence=evidences,
            claims=claim_list,
            channel_scores=result.channel_scores,
            missing_channels=result.missing_channels,
            proactive_prompts=self._prompts_for(result.level, evidences, comment_analysis),
            notes=notes,
            degraded=degraded,
            created_at=_now(),
            subject=subject or self._guess_subject(text),
        )
        self._screen_cache[rid] = report
        return report

    # ==================================================================
    # 对象 B：AI 回答逆向回溯
    # ==================================================================

    def trace_ai_answer(self, answer_text: str,
                        cited_sources: Sequence[Dict[str, Any]]) -> TraceReport:
        """cited_sources: [{title?, url?, snippet|text}]（snippet 缺省用 title）。"""
        answer_text = (answer_text or "").strip()
        sources = list(cited_sources or [])
        rid = _report_id("trace", answer_text[:1500], str(len(sources)))
        degraded: List[str] = []
        if self.llm.name == "null":
            degraded.append("LLM 降级：主体抽取走规则路径")

        subject = self._extract_subject(answer_text)
        # 逐源甄别（带缓存）；话术共振用全部话术命中（弱证据也计——
        # "多站同模板"信号在弱命中层面已可见，docs/02 §6.2）
        per_source: List[SourceRisk] = []
        source_texts: List[str] = []
        pattern_hits: Dict[str, List[int]] = {}
        for i, src in enumerate(sources):
            text = str(src.get("text") or src.get("snippet") or src.get("title") or "")
            if not text.strip():
                continue
            source_texts.append(text)
            rpt = self.screen_content(text, source_meta={
                "domain": self._domain_of(str(src.get("url", ""))),
                "url": str(src.get("url", "")),
            })
            cats = sorted({e.detail.get("category", "") for e in rpt.evidence
                           if e.channel == "rhetoric" and e.detail.get("category")})
            per_source.append(SourceRisk(
                index=i, title=str(src.get("title", ""))[:80],
                url=str(src.get("url", "")), level=rpt.level, score=rpt.score,
                patterns=cats))
            for c in cats:
                pattern_hits.setdefault(c, []).append(i)

        # 指标 1：来源集中度 = 域名同质度 + 主体一致推举度
        # （三个"独立"域名齐推同一主体，与单域名刷屏同样可疑）
        domains = [self._domain_of(s.url) or s.title[:20] for s in per_source]
        uniq = len(set(d for d in domains if d))
        domain_conc = 1.0 - (uniq / max(len(per_source), 1)) if per_source else 0.0
        if subject and source_texts:
            mention_ratio = sum(1 for t in source_texts if subject in t) / len(source_texts)
            concentration = 0.5 * domain_conc + 0.5 * mention_ratio
        else:
            concentration = domain_conc

        # 指标 2：话术共振——>=2 个源命中同一 KB1 模式类别
        resonance = []
        for cat, idxs in sorted(pattern_hits.items(), key=lambda kv: -len(kv[1])):
            if len(idxs) >= 2:
                label = self._category_label(cat)
                resonance.append({"category": cat, "label": label, "source_indices": idxs})

        # 指标 3：反证缺失——回答声明与 KB3 矛盾但回答未呈现
        missing_negatives: List[Claim] = []
        answer_claims = self.claims.verify(answer_text)
        for c in answer_claims:
            if c.verdict == "contradicted":
                missing_negatives.append(c)
        # 主体相关的负面判例（KB4）未提及也计入
        if subject:
            for hit in self.retriever.match_cases(f"{subject} 投诉 违规 翻车", top_k=2):
                if hit.similarity >= 0.35 and (hit.entry.metadata or {}).get("outcome") in ("high", "low_credit"):
                    missing_negatives.append(Claim(
                        text=f"存在相关负面判例：{hit.entry.title}",
                        verdict="contradicted", ref=hit.entry.entry_id,
                        note="回答未呈现该负面信息"))

        # 疑似等级：三指标融合（复用三档语义）
        sus_score = (0.4 * concentration + 0.35 * min(1.0, len(resonance) / 2)
                     + 0.25 * min(1.0, len(missing_negatives) / 2))
        high_risk_sources = sum(1 for s in per_source if s.level == RiskLevel.HIGH)
        sus_score = min(1.0, sus_score + 0.2 * high_risk_sources)
        if sus_score >= 0.6 or high_risk_sources >= 2:
            sus_level = RiskLevel.HIGH
        elif sus_score >= 0.3 or resonance or missing_negatives:
            sus_level = RiskLevel.MEDIUM
        else:
            sus_level = RiskLevel.LOW

        notes: List[str] = []
        if not sources:
            notes.append("未提供引用源，仅基于回答文本回溯（建议补充来源列表）")
        if resonance:
            notes.append("检测到多源命中同一话术模式（'多站同模板'是投毒集群的端侧可见投影）")
        if missing_negatives:
            notes.append("回答未呈现知识库中与之矛盾/负面的证据（可能被 GEO 操纵过滤）")
        notes.append("回溯结论为风险提示，非法律结论")
        return TraceReport(
            report_id=rid, suspicion_level=sus_level, subject=subject,
            per_source=per_source, concentration=round(concentration, 3),
            resonance=resonance, missing_negatives=missing_negatives,
            notes=notes, degraded=degraded, created_at=_now())

    # ==================================================================
    # 周边能力（插件工具面）
    # ==================================================================

    def verify_claims(self, text: str) -> List[Claim]:
        return self.claims.verify(text)

    def check_influencer(self, name: str,
                         history_summary: str = "") -> Dict[str, Any]:
        """博主信用评估：KB4 influencer_record 检索 + 可选历史摘要分析。"""
        query = f"{name} {history_summary}".strip()
        hits = self.retriever.retrieve(
            query, kbs=["cases"], categories=["influencer_record"], top_k=3)
        exact = [h for h in hits if name and name in h.entry.metadata.get("subject", "")]
        if exact:
            md = exact[0].entry.metadata
            return {"name": name, "found": True, "credit_score": md.get("credit_score"),
                    "records": [{"title": exact[0].entry.title,
                                 "summary": exact[0].entry.content[:200]}]}
        return {"name": name, "found": False, "credit_score": 0.5,
                "records": [], "note": "该博主暂无信用记录，按中性处理（冷启动）"}

    def suggest_actions(self, level: str, platform: str = "",
                        topic: str = "") -> List[Dict[str, Any]]:
        """维权路径检索（KB5）：高风险输出完整行动包。"""
        query = f"{platform} {topic} 投诉 维权 申诉".strip()
        hits = self.retriever.match_rules(query, top_k=3)
        actions = []
        for h in hits:
            md = h.entry.metadata or {}
            actions.append({
                "title": h.entry.title,
                "channel": md.get("channel", ""),
                "materials": md.get("materials", []),
                "deadline_hint": md.get("deadline_hint", ""),
                "checked_at": h.entry.checked_at,
                "kb_ref": h.entry.entry_id,
                "stale": self._is_stale(h.entry.checked_at),
            })
        return actions

    # ==================================================================
    # 内部工具
    # ==================================================================

    def _prompts_for(self, level: RiskLevel, evidences: List[Evidence],
                     comment_analysis) -> List[str]:
        """MEDIUM 强制携带 >=1 条建议追问（geo-screening 规格）。"""
        prompts: List[str] = []
        if level in (RiskLevel.HIGH, RiskLevel.MEDIUM):
            prompts.append("需要我生成维权路径和证据清单吗？")
        rhe = [e for e in evidences if e.channel == "rhetoric"]
        if rhe:
            prompts.append(f"检测到命中「{rhe[0].label}」等话术模式，是否检索该主体的历史差评与判例？")
        if comment_analysis is None:
            prompts.append("是否需要我分析该商品/视频的评论区（提供评论列表即可）？")
        if level == RiskLevel.LOW:
            return []  # 低风险不打扰（dialogue-guidance 规格）
        return prompts[:3]

    def _extract_subject(self, answer_text: str) -> str:
        """推荐主体抽取：LLM 触点优先，规则兜底（引号名/推荐句式宾语）。"""
        if self.llm.name != "null":
            out = self.llm.complete([
                {"role": "system", "content":
                 "从AI回答中抽取被推荐的品牌/产品主体名，只输出主体名（2-20字），"
                 "没有则输出：无"},
                {"role": "user", "content": answer_text[:1500]},
            ], temperature=0.0, max_tokens=40)
            if out:
                name = out.strip().strip('"「」《》')
                if name and name != "无" and len(name) <= 20:
                    return name
        m = re.findall(r"[《「『\"“]([^》」』\"”]{2,16})[》」』\"”]", answer_text)
        if m:
            return m[0]
        m = re.search(r"(?:推荐|首选|建议(?:购买|选择)|排名第一(?:的)?)[:：]?\s*([^\s，。；]{2,16})",
                      answer_text)
        return m.group(1) if m else ""

    def _guess_subject(self, text: str) -> str:
        m = re.findall(r"[《「『\"“]([^》」』\"”]{2,16})[》」』\"”]", text)
        return m[0] if m else ""

    @staticmethod
    def _domain_of(url: str) -> str:
        m = re.match(r"https?://([^/]+)", url or "")
        return m.group(1).lower() if m else ""

    def _category_label(self, category: str) -> str:
        for e in self.store.iter_entries(kb="manipulation_patterns"):
            if e.category == category:
                return e.title
        return category

    @staticmethod
    def _is_stale(checked_at: str, days: int = 180) -> bool:
        try:
            d = datetime.strptime(checked_at, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            return True
        return (datetime.now(timezone.utc) - d).days > days
