# -*- coding: utf-8 -*-
"""F5 事实核验（docs/03 路径一，D-06 降级策略）。

claim 抽取：LLM 触点 2 优先，无 LLM 时按触发词规则筛句。
三态核验：与 KB3 权威事实库比对——
- 规则路径：claim 命中 KB3 条目的 violation_patterns → contradicted；
  命中 support_patterns → supported；检索到相关条目但无模式命中 → unverified；
- LLM 路径：给出条目上下文让 LLM 判三态（输出仍受规则路径复核兜底）。
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from ..config import Config
from ..llm import LLMBackend, NullBackend, touch_extract_claims
from ..retrieval import HybridRetriever
from ..schema import Claim, Evidence, KBName


class ClaimVerifier:
    def __init__(self, retriever: HybridRetriever, llm: Optional[LLMBackend] = None,
                 config: Optional[Config] = None):
        self.retriever = retriever
        self.config = config or retriever.config
        self.llm = llm or NullBackend()

    # ---------------- 抽取 ----------------

    def extract(self, text: str) -> List[Dict[str, Any]]:
        return touch_extract_claims(self.llm, text)

    # ---------------- 核验 ----------------

    def verify(self, text: str,
               claims: Optional[List[Dict[str, Any]]] = None) -> List[Claim]:
        """返回三态核验后的 Claim 列表（span 保留原文定位）。"""
        raw = claims if claims is not None else self.extract(text)
        out: List[Claim] = []
        for item in raw:
            claim_text = str(item.get("text", ""))[:200]
            span = tuple(item.get("span", (0, 0)))
            if not claim_text.strip():
                continue
            verdict, ref, authority, note = self._verify_one(claim_text)
            out.append(Claim(text=claim_text, verdict=verdict, ref=ref,
                             authority=authority, note=note, span=span))
        return out

    def _verify_one(self, claim_text: str):
        hits = self.retriever.match_facts(claim_text, top_k=3)
        if not hits:
            return "unverified", "", "", "权威事实库无相关条目（种子库覆盖有限）"
        # 规则路径优先：violation/support patterns 是确定性判据
        for h in hits:
            md = h.entry.metadata or {}
            for pat in md.get("violation_patterns", []) or []:
                if pat and pat in claim_text:
                    return ("contradicted", h.entry.entry_id,
                            md.get("authority", ""),
                            f"声明命中违规表述模式「{pat}」：{h.entry.title}")
            for pat in md.get("support_patterns", []) or []:
                if pat and pat in claim_text:
                    return ("supported", h.entry.entry_id,
                            md.get("authority", ""),
                            f"声明与权威事实一致：{h.entry.title}")
        # LLM 路径：规则未决时让 LLM 基于条目上下文判三态
        if self.llm.name != "null":
            verdict = self._llm_judge(claim_text, hits)
            if verdict in ("contradicted", "supported"):
                best = hits[0]
                return (verdict, best.entry.entry_id,
                        (best.entry.metadata or {}).get("authority", ""),
                        "LLM 基于权威事实条目判定")
        # 检索到相关条目但无法判定 → unverified（计入事实信号半分）
        best = hits[0]
        return ("unverified", best.entry.entry_id,
                (best.entry.metadata or {}).get("authority", ""),
                f"相关条目「{best.entry.title}」无法确证该声明")

    def _llm_judge(self, claim_text: str, hits) -> str:
        ctx = "\n".join(f"- {h.entry.title}: {h.entry.content[:200]}" for h in hits)
        out = self.llm.complete([
            {"role": "system", "content":
                "你是事实核验器。基于给定的权威事实条目判断声明的状态，只输出一个词："
                "supported（与事实一致）/ contradicted（与事实矛盾或触犯禁止性规定）/ "
                "unverified（无法判断）。"},
            {"role": "user", "content": f"声明：{claim_text}\n权威事实条目：\n{ctx}"},
        ], temperature=0.0, max_tokens=16)
        if not out:
            return "unverified"
        out = out.strip().lower()
        for v in ("contradicted", "supported", "unverified"):
            if v in out:
                return v
        return "unverified"

    # ---------------- 证据化 ----------------

    @staticmethod
    def to_evidences(claims: List[Claim]) -> List[Evidence]:
        """把 contradicted/unverified 声明转为 factual 通道证据。"""
        evs: List[Evidence] = []
        for c in claims:
            if c.verdict == "contradicted":
                score = 1.0
            elif c.verdict == "unverified" and c.ref:
                score = 0.5
            else:
                continue
            evs.append(Evidence(
                channel="factual",
                label=f"事实核验：{c.verdict}",
                score=score,
                severity=0.9 if c.verdict == "contradicted" else 0.5,
                span=c.span,
                excerpt=c.text[:120],
                kb_refs=[c.ref] if c.ref else [],
                detail={"authority": c.authority, "note": c.note},
            ))
        return evs
