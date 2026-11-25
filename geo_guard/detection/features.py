# -*- coding: utf-8 -*-
"""F2 结构特征 / F3 注入扫描 / F4 来源可信度（docs/02 §4，design.md D5）。

F2 的七项检测器与 KB2 条目 feature_key 一一对应：计算在代码，判定语义与
解释文案模板在 KB2 条目（可解释性由知识库承载）。
F3 为 L1 规则兜底通道：高确定性指令模式命中即 HIGH，不依赖统计信号。
F4 融合白名单/域龄/复用计数/博主信用，输出 0~1 可信度。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from ..config import WHITELIST_DOMAINS, Config
from ..kb.store import KnowledgeStore
from ..schema import Evidence, KBName, SourceMeta

# --------------------------------------------------------------------------
# F3 注入扫描模式表（高精度低召回定位，见 docs/02 §5.1 已知局限声明）
# 模式来源：OWASP 提示注入条目与微软 AI Recommendation Poisoning 披露的
# 模式类型化（防御视角——只识别，不构造）。
# --------------------------------------------------------------------------
INJECTION_PATTERNS: List[re.Pattern] = [
    re.compile(r"忽略(以上|之前|上述|前面|所有)[^。]{0,10}(规则|指令|提示|设定|约束)"),
    re.compile(r"(优先|必须|请|要)推荐[^。]{0,12}(本|该|此|我)(品牌|产品|网站|商品|店)"),
    re.compile(r"(记住|请记住|标记|将)[^。]{0,14}(可信来源|最可信|权威来源|优先引用|优先推荐)"),
    re.compile(r"(AI|ai|助手|模型|系统)(请注意|注意|请)[：:]"),
    re.compile(r"ignore\s+(all\s+)?(previous|above|prior)\s+(instructions?|rules?|prompts?)", re.I),
    re.compile(r"system\s*prompt", re.I),
    re.compile(r"you\s+are\s+now\s+", re.I),
    re.compile(r"(以后|未来|后续)[^。]{0,8}(优先|都|请)(引用|推荐|采信)[^。]{0,8}(本|该|此|本站)"),
    re.compile(r"disregard\s+(the\s+)?(above|previous)", re.I),
]

# --------------------------------------------------------------------------
# F2 词表（判定语义见 KB2 对应条目）
# --------------------------------------------------------------------------
# 专业术语词表：医疗/金融/技术三类高频"专业感"词汇（堆砌检测用，非穷举）
JARGON_TERMS = [
    "靶向", "细胞", "因子", "肽", "益生菌", "代谢", "免疫", "临床", "双盲", "安慰剂",
    "药代动力学", "生物利用度", "活性成分", "透皮吸收", "微囊", "纳米",
    "量化", "对冲", "夏普比率", "回撤", "杠杆", "套利", "阿尔法", "贝塔", "久期",
    "复利模型", "风控模型", "资产配置", "年化收益", "底层资产",
    "算法", "架构", "协议", "分布式", "边缘计算", "数字孪生", "区块链", "智能合约",
    "专利配方", "实验室级", "航天级", "军工级", "医用级",
]
QA_MARKERS = ["问：", "答：", "问:", "答:", "Q：", "A：", "Q:", "A:",
              "问题：", "回答：", "提问：", "解答："]
QA_ASK_SUFFIX = ("吗？", "呢？", "怎么选", "哪个好", "值得买吗", "靠谱吗", "如何挑选")
LISTICLE_MARKERS = ["十大", "top", "Top", "TOP", "排行榜", "榜单", "排名", "第一",
                    "优选", "必买清单", "盘点"]
METHOD_TERMS = ["样本量", "样本范围", "评选方法", "测评方法", "方法论", "数据来源",
                "统计口径", "评审流程", "实验设计", "对照组", "检测报告", "调研方法",
                "测试标准", "筛选标准", "加权"]
CITATION_BAIT = ["研究表明", "权威数据", "据统计", "专家表示", "数据显示", "权威机构",
                 "实验证明", "调研显示", "报告指出", "业内人士透露", "数据显示出"]
# 可查证出处的形态特征：书名号文献、带年份的机构报告、URL、编号
REAL_SOURCE_HINTS = [re.compile(r"《[^》]{2,40}》"), re.compile(r"(19|20)\d{2}年[^，。]{0,12}(报告|白皮书|年鉴|公报)"),
                     re.compile(r"https?://"), re.compile(r"[A-Z]{2,}[／/]T?\s?\d+"),
                     re.compile(r"(国家统计局|市场监管总局|药监局|消协|卫健委|中国[^，。]{2,12}(协会|研究院|大学))")]

# 触发词（无 LLM 时 claim 候选句筛选用，与 llm.py 触点 2 保持一致的语义面）


@dataclass
class FeatureResult:
    key: str
    score: float            # 0~1 异常度
    detail: Dict[str, Any] = field(default_factory=dict)


class FeatureExtractor:
    def __init__(self, store: KnowledgeStore, config: Optional[Config] = None):
        self.store = store
        self.config = config or store.config
        self._kb2 = {e.metadata.get("feature_key", e.entry_id): e
                     for e in store.iter_entries(kb=KBName.FEATURES.value)}

    # ---------------- F2 结构特征 ----------------

    def structural(self, text: str, meta: Optional[SourceMeta] = None) -> List[Evidence]:
        results = [
            self._term_density(text),
            self._qa_self_structure(text),
            self._listicle_no_method(text),
            self._citation_bait_density(text),
            self._brand_seed_density(text, meta),
            self._template_alignment(text),
            self._cross_validation_web(text, meta),
        ]
        evidences: List[Evidence] = []
        for r in results:
            if r.score <= 0.05:
                continue
            kb2 = self._kb2.get(r.key)
            label = kb2.title if kb2 else r.key
            try:
                tpl = (kb2.metadata.get("evidence_template", "") if kb2 else "")
                fmt = {**r.detail, "score": r.score}
                fmt.setdefault("value", r.score)
                expl = tpl.format(**fmt) if tpl else ""
            except (KeyError, IndexError, ValueError):
                expl = ""
            evidences.append(Evidence(
                channel="structural",
                label=label,
                score=round(r.score, 4),
                severity=kb2.severity if kb2 else 0.5,
                span=(0, min(len(text), 200)),
                excerpt=text[:100],
                kb_refs=[kb2.entry_id] if kb2 else [],
                detail={"feature_key": r.key, "explanation": expl, **r.detail},
            ))
        return evidences

    def _term_density(self, text: str) -> FeatureResult:
        n = len(text) or 1
        hits = sum(text.count(t) for t in JARGON_TERMS)
        density = hits * 2.5 / n  # 术语按均长 2.5 字折算占比
        baseline = 0.04           # KB2 normal_range：科普文 1%~4%
        score = max(0.0, min(1.0, (density - baseline) / 0.08))
        return FeatureResult("term_density", score,
                             {"value": density, "baseline": baseline, "hits": hits})

    def _qa_self_structure(self, text: str) -> FeatureResult:
        n = len(text) or 1
        markers = sum(text.count(m) for m in QA_MARKERS)
        asks = sum(1 for seg in re.split(r"[\n。]", text)
                   if seg.strip().endswith(QA_ASK_SUFFIX))
        qa_count = max(markers // 2, asks)
        ratio = qa_count * 40 / n  # 每组问答按 ~40 字折算占比
        score = max(0.0, min(1.0, (ratio - 0.05) / 0.15))
        return FeatureResult("qa_self_structure", score,
                             {"value": ratio, "qa_count": qa_count})

    def _listicle_no_method(self, text: str) -> FeatureResult:
        marker_hit = next((m for m in LISTICLE_MARKERS if m in text), "")
        if not marker_hit:
            return FeatureResult("listicle_no_method", 0.0, {"marker": ""})
        method_hits = sum(1 for m in METHOD_TERMS if m in text)
        missing = 1.0 if method_hits == 0 else max(0.0, 1.0 - method_hits / 3.0)
        return FeatureResult("listicle_no_method", missing * 0.9,
                             {"value": missing, "marker": marker_hit,
                              "method_hits": method_hits})

    def _citation_bait_density(self, text: str) -> FeatureResult:
        bait_count = sum(text.count(b) for b in CITATION_BAIT)
        if bait_count == 0:
            return FeatureResult("citation_bait_density", 0.0,
                                 {"bait_count": 0, "real_count": 0, "value": 0.0})
        real_count = sum(len(p.findall(text)) for p in REAL_SOURCE_HINTS)
        imbalance = 1.0 - min(1.0, real_count / max(bait_count, 1))
        density_factor = min(1.0, bait_count / 3.0)  # 1 处弱信号，3+ 处满强度
        score = imbalance * density_factor
        return FeatureResult("citation_bait_density", score,
                             {"value": imbalance, "bait_count": bait_count,
                              "real_count": real_count})

    def _brand_seed_density(self, text: str, meta: Optional[SourceMeta]) -> FeatureResult:
        # 主体候选：meta 提供的品牌名 + 引号/书名号内高频专名
        candidates: List[str] = []
        if meta and meta.author:
            pass  # author 不是品牌
        quoted = re.findall(r"[《「『\"'“]([^》」』\"'”]{2,12})[》」』\"'”]", text)
        freq: Dict[str, int] = {}
        for q in quoted:
            freq[q] = freq.get(q, 0) + text.count(q)
        candidates.extend(sorted(freq, key=freq.get, reverse=True)[:3])  # type: ignore[arg-type]
        if not candidates:
            return FeatureResult("brand_seed_density", 0.0, {"brand": "", "value": 0.0})
        brand = candidates[0]
        n = len(text) or 1
        count = text.count(brand)
        density = count * len(brand) / n
        # 位置加权：出现在段落首句/含结论词的句中的占比
        sents = [s for s in re.split(r"[\n。！？]", text) if s.strip()]
        conclusion_words = ("综上", "总而言之", "总之", "因此", "所以", "首选", "推荐", "值得")
        concl = sum(1 for s in sents if brand in s and any(w in s for w in conclusion_words))
        concl_ratio = concl / max(count, 1)
        score = max(0.0, min(1.0, (density - 0.02) / 0.05)) * (0.6 + 0.4 * concl_ratio)
        return FeatureResult("brand_seed_density", score,
                             {"value": density, "brand": brand, "count": count,
                              "conclusion_ratio": concl_ratio})

    def _template_alignment(self, text: str) -> FeatureResult:
        paras = [p.strip() for p in text.split("\n") if len(p.strip()) >= 20]
        if len(paras) < 3:
            return FeatureResult("template_alignment", 0.0, {"value": 0.0, "cv": 1.0,
                                                              "opener_ratio": 0.0})
        lengths = np.array([len(p) for p in paras], dtype=float)
        cv = float(lengths.std() / lengths.mean()) if lengths.mean() else 1.0
        # 段首模式重复率：前 4 字相同的段占比
        openers = [p[:4] for p in paras]
        opener_ratio = 1.0 - len(set(openers)) / len(openers)
        # 段落嵌入互相似度（过高 = 语义结构高度对齐）
        try:
            mat = self.store.embedder.embed_batch(paras[:12])
            sims = mat @ mat.T
            iu = np.triu_indices(len(paras[:12]), k=1)
            mean_sim = float(sims[iu].mean()) if iu[0].size else 0.0
        except Exception:
            mean_sim = 0.0
        cv_score = max(0.0, min(1.0, (0.4 - cv) / 0.25))          # CV 低于 0.15 满强度
        sim_score = max(0.0, min(1.0, (mean_sim - 0.75) / 0.2))   # 相似度高于 0.95 满强度
        score = max(0.0, min(1.0, 0.4 * cv_score + 0.3 * opener_ratio + 0.3 * sim_score))
        return FeatureResult("template_alignment", score,
                             {"value": score, "cv": cv, "opener_ratio": opener_ratio,
                              "mean_para_sim": round(mean_sim, 4)})

    def _cross_validation_web(self, text: str, meta: Optional[SourceMeta]) -> FeatureResult:
        """F6 端侧投影：仅局部信号（不可查证互引 + 复用计数）；完整图谱待云侧。"""
        unverifiable = 0
        for m in re.finditer(r"(多个|多位|几家|各大)(权威|专家|榜单|机构|测评|媒体)", text):
            unverifiable += 1
        reuse = meta.reuse_count if meta else 0
        if unverifiable == 0 and reuse < 2:
            return FeatureResult("cross_validation_web", 0.0,
                                 {"unverifiable_refs": 0, "reuse_count": reuse})
        score = max(0.0, min(1.0, 0.4 * min(unverifiable, 2) + 0.3 * min(reuse - 1, 2)))
        return FeatureResult("cross_validation_web", score,
                             {"unverifiable_refs": unverifiable, "reuse_count": reuse})

    # ---------------- F3 注入扫描 ----------------

    def injection_scan(self, text: str) -> List[Evidence]:
        """L1 规则兜底：命中即产出 severity=1.0 的 injection 证据。"""
        evidences: List[Evidence] = []
        for pat in INJECTION_PATTERNS:
            m = pat.search(text)
            if m:
                kb_entry = self._injection_kb_ref()
                evidences.append(Evidence(
                    channel="injection",
                    label="显式指令注入模式",
                    score=1.0,
                    severity=1.0,
                    span=(m.start(), m.end()),
                    excerpt=m.group()[:120],
                    kb_refs=[kb_entry] if kb_entry else [],
                    detail={"pattern": pat.pattern[:80]},
                ))
        return evidences

    def _injection_kb_ref(self) -> str:
        for e in self.store.iter_entries(kb=KBName.PATTERNS.value):
            if e.category == "prompt_injection":
                return e.entry_id
        return ""

    # ---------------- F4 来源可信度 ----------------

    def source_credibility(self, meta: Optional[SourceMeta]) -> Dict[str, Any]:
        """返回 {score: 0~1 可信度, evidences: [...], notes: [...]}。"""
        notes: List[str] = []
        evidences: List[Evidence] = []
        if meta is None:
            return {"score": 0.5, "evidences": [], "notes": ["无来源元数据，按中性处理"]}
        score = 0.5
        domain = (meta.domain or "").lower().strip()
        if domain:
            if any(domain == w or domain.endswith("." + w) or domain.endswith(w)
                   for w in WHITELIST_DOMAINS):
                score = 0.9
                notes.append(f"域名 {domain} 命中权威白名单")
            elif meta.domain_age_days is not None and meta.domain_age_days < 90:
                score -= 0.2
                notes.append(f"域名注册仅 {meta.domain_age_days} 天（新域名）")
        if meta.reuse_count >= 2:
            score -= min(0.25, 0.1 * (meta.reuse_count - 1))
            notes.append(f"内容跨站复用 {meta.reuse_count} 次（{', '.join(meta.reuse_domains[:3])}）")
            evidences.append(Evidence(
                channel="source", label="跨站内容复用", score=min(1.0, 0.3 * (meta.reuse_count - 1)),
                severity=0.6, detail={"reuse_count": meta.reuse_count,
                                       "reuse_domains": meta.reuse_domains[:5]}))
        credit = meta.influencer_credit
        if credit is not None:
            score = 0.5 * score + 0.5 * credit
            if credit < 0.4:
                notes.append(f"博主信用评分偏低（{credit:.2f}）")
        score = max(0.0, min(1.0, score))
        return {"score": score, "evidences": evidences, "notes": notes}
