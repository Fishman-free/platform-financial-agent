# -*- coding: utf-8 -*-
"""纯数据模型：跨层接口契约（design.md D1）。

这些 dataclass 即模块间与形态层（插件/Personal Agent）的接口契约：
云侧图谱、浏览器插件未来按同一结构接入。全部支持 dict 往返序列化，
保证可入 SQLite JSON 列、可被插件 dispatch 直接返回给宿主 Agent。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class RiskLevel(str, Enum):
    """三档风险等级（继承母项目三档处置策略）。"""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

    @property
    def rank(self) -> int:
        return {"low": 0, "medium": 1, "high": 2}[self.value]

    def downgrade_half(self) -> "RiskLevel":
        """降半档（HIGH→MEDIUM→LOW），永不低于 LOW。"""
        return {"high": RiskLevel.MEDIUM,
                "medium": RiskLevel.LOW,
                "low": RiskLevel.LOW}[self.value]


class KBName(str, Enum):
    """五库分治（docs/02 §2，D-01）。"""

    PATTERNS = "manipulation_patterns"        # KB1 话术模式库
    FEATURES = "geo_structural_features"      # KB2 GEO 结构特征库
    FACTS = "authoritative_facts"             # KB3 权威事实库
    CASES = "cases"                           # KB4 历史案例库
    RULES = "platform_rules"                  # KB5 平台规则库


@dataclass
class KBEntry:
    """知识库统一信封（knowledge-retrieval 规格）。"""

    entry_id: str
    kb: str
    title: str
    content: str
    source: str                      # 强制非空——入库治理（I-12 反哺防投毒）
    version: str = "1.0.0"
    checked_at: str = ""             # 时效核查日期 YYYY-MM-DD
    status: str = "active"           # active | deprecated | pending_review
    severity: float = 0.5            # 0~1，KB1/KB2 使用
    category: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "KBEntry":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class SourceMeta:
    """来源元数据（Evidence Bundle 的元数据面）。"""

    domain: str = ""
    author: str = ""
    published_at: str = ""
    url: str = ""
    domain_age_days: Optional[int] = None   # 云侧/WHOIS 提供，端侧可空
    reuse_count: int = 0                    # 跨站复用计数（ingest 去重产出）
    reuse_domains: List[str] = field(default_factory=list)
    influencer_credit: Optional[float] = None  # 博主信用 0~1，缺省中性 0.5
    network_evidence: Optional[float] = None   # F6 协同证据槽位（云侧图谱接入前恒为 None）


@dataclass
class Evidence:
    """单条证据：可解释性的最小单元（P3 原则）。

    每条证据都能定位原文片段（span）并溯源知识条目（kb_refs），
    报告分数可由证据逐项分解复算。
    """

    channel: str            # rhetoric|structural|injection|source|factual|case|network|comment
    label: str              # 人读标签（如"无依据权威背书"）
    score: float            # 0~1 通道内强度
    severity: float = 0.5   # 知识条目严重度
    span: Tuple[int, int] = (0, 0)     # 原文片段定位 [start, end)
    excerpt: str = ""                  # 原文摘录（截断至 120 字）
    kb_refs: List[str] = field(default_factory=list)
    detail: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["span"] = list(self.span)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Evidence":
        d = dict(d)
        if "span" in d:
            d["span"] = tuple(d["span"])
        return cls(**d)


@dataclass
class Claim:
    """原子事实声明与三态核验结果（docs/03 路径一）。"""

    text: str
    verdict: str = "unverified"   # supported | contradicted | unverified
    ref: str = ""                 # 命中的 KB3 条目 id
    authority: str = ""           # 权威出处（法规/标准名）
    note: str = ""
    span: Tuple[int, int] = (0, 0)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["span"] = list(self.span)
        return d


@dataclass
class RiskReport:
    """对象 A（单篇内容甄别）输出契约。"""

    report_id: str
    level: RiskLevel
    score: float                       # 融合后 0~1 风险分
    confidence: str = "model"          # rule | model | thin（证据单薄）
    evidence: List[Evidence] = field(default_factory=list)
    claims: List[Claim] = field(default_factory=list)
    channel_scores: Dict[str, Optional[float]] = field(default_factory=dict)
    missing_channels: List[str] = field(default_factory=list)  # 缺失声明（I-11）
    proactive_prompts: List[str] = field(default_factory=list)  # 建议追问（dialogue 触发器输入）
    notes: List[str] = field(default_factory=list)              # 降级/单薄/白名单异常等声明
    degraded: List[str] = field(default_factory=list)           # 降级模式标注（embedding/llm/claim 抽取）
    created_at: str = ""
    subject: str = ""                  # 甄别对象主体（商品/品牌/博主名，可空）

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["level"] = self.level.value
        d["evidence"] = [e.to_dict() for e in self.evidence]
        d["claims"] = [c.to_dict() for c in self.claims]
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RiskReport":
        d = dict(d)
        d["level"] = RiskLevel(d["level"])
        d["evidence"] = [Evidence.from_dict(e) for e in d.get("evidence", [])]
        d["claims"] = [Claim(**{k: v for k, v in c.items() if k != "span"} | {"span": tuple(c.get("span", (0, 0)))})
                       for c in d.get("claims", [])]
        return cls(**d)


@dataclass
class SourceRisk:
    """回溯模式下单个引用源的风险摘要。"""

    index: int
    title: str = ""
    url: str = ""
    level: RiskLevel = RiskLevel.LOW
    score: float = 0.0
    patterns: List[str] = field(default_factory=list)   # 命中的 KB1 category

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["level"] = self.level.value
        return d


@dataclass
class TraceReport:
    """对象 B（AI 回答逆向回溯）输出契约（docs/02 §6.2）。"""

    report_id: str
    suspicion_level: RiskLevel                 # 回答被 GEO 牵引的疑似等级
    subject: str = ""                          # 回答中的推荐主体
    per_source: List[SourceRisk] = field(default_factory=list)
    concentration: float = 0.0                 # 来源集中度 0~1（越高越可疑）
    resonance: List[Dict[str, Any]] = field(default_factory=list)
    # 话术共振：[{category, source_indices[], label}]
    missing_negatives: List[Claim] = field(default_factory=list)  # 反证缺失清单
    notes: List[str] = field(default_factory=list)
    degraded: List[str] = field(default_factory=list)
    created_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["suspicion_level"] = self.suspicion_level.value
        d["per_source"] = [s.to_dict() for s in self.per_source]
        d["missing_negatives"] = [c.to_dict() for c in self.missing_negatives]
        return d


@dataclass
class MemoryRecord:
    """情景记忆条目（memory-profile 规格）。"""

    memory_id: str
    kind: str            # screening | feedback | preference | event
    subject: str
    content: str
    risk_level: str = ""
    importance: float = 0.5
    created_at: float = 0.0        # unix 秒
    last_accessed_at: float = 0.0
    access_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DAGNode:
    """行动链 DAG 节点（dialogue-guidance 规格，动作集对齐 ask/show/call/confirm）。"""

    node_id: str
    action: str                        # ask | show | confirm | call:<name>
    params: Dict[str, Any] = field(default_factory=dict)
    depends_on: List[str] = field(default_factory=list)
    status: str = "pending"            # pending | running | done | skipped | cancelled
    result: Optional[Dict[str, Any]] = None
    kind: str = "task"                 # task | recommend（HIGH 重规划时 recommend 类被中断）

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DAGNode":
        return cls(**d)


class Phase(str, Enum):
    """对话状态机阶段（docs/06 §1）。"""

    IDLE = "idle"
    INTENT_PARSE = "intent_parse"
    CLARIFYING = "clarifying"
    TASK_PLANNING = "task_planning"
    EXECUTING = "executing"
    RISK_PRESENTING = "risk_presenting"
    ACTION_GUIDING = "action_guiding"
    RESPONDING = "responding"


@dataclass
class DialogueState:
    """对话状态（纯数据，可 JSON 快照——中断恢复的基础，design.md D7）。"""

    session_id: str
    phase: Phase = Phase.IDLE
    pending_question: str = ""
    pending_intent: str = ""
    dag: List[DAGNode] = field(default_factory=list)
    history: List[Dict[str, str]] = field(default_factory=list)  # 工作记忆环形缓冲
    turn: int = 0
    clarify_attempts: int = 0

    def to_json(self) -> str:
        d = asdict(self)
        d["phase"] = self.phase.value
        return json.dumps(d, ensure_ascii=False)

    @classmethod
    def from_json(cls, raw: str) -> "DialogueState":
        d = json.loads(raw)
        d["phase"] = Phase(d["phase"])
        d["dag"] = [DAGNode.from_dict(n) for n in d.get("dag", [])]
        return cls(**d)


@dataclass
class FeedbackRecord:
    """用户反馈（data-ingest 反馈闭环）。"""

    feedback_id: str
    report_id: str
    verdict: str          # agree | dispute
    note: str = ""
    kb_refs: List[str] = field(default_factory=list)
    created_at: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
