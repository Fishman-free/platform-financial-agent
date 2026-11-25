# -*- coding: utf-8 -*-
"""多轮交互与行动引导（dialogue-guidance 规格，docs/06，design.md D7）。

- 对话状态机：8 阶段，纯数据状态 + JSON 快照持久化 → 中断恢复；
- 主动质询触发器：规则 + LLM 双轨，保守触发（宁少勿滥）；
- 行动链 DAG：ask/show/call/confirm 原子动作，依赖调度，节点失败 skipped
  不中断，HIGH 输出触发动态重规划（中断推荐类节点，插入警告与维权引导）。
"""

from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .config import Config
from .llm import LLMBackend, NullBackend
from .schema import DAGNode, DialogueState, Phase, RiskLevel

_DDL = """
CREATE TABLE IF NOT EXISTS session_snapshots (
    session_id TEXT,
    seq        INTEGER,
    payload    TEXT,
    saved_at   TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (session_id, seq)
);
"""

# 意图不明时的补全质询（保守触发：仅在无法定位甄别对象时）
_CLARIFY_QUESTION = "您想甄别的是哪类内容？可以把链接、文案或 AI 回答直接发给我。"


class ProactiveTrigger:
    """主动质询触发器：规则触发 ∪ LLM 决策（docs/06 §2）。

    规则通道永远兜底；LLM 通道只做"从候选问题中选最优"，不发明新问题，
    保证无 API 时行为可预期。
    """

    def __init__(self, llm: Optional[LLMBackend] = None):
        self.llm = llm or NullBackend()

    def candidates(self, report: Optional[Any] = None,
                   conflicts: Optional[List[Dict[str, Any]]] = None,
                   ambiguous_intent: bool = False,
                   fork_choices: Optional[List[str]] = None) -> List[str]:
        cands: List[str] = []
        if ambiguous_intent:
            cands.append(_CLARIFY_QUESTION)
        if report is not None:
            cands.extend(getattr(report, "proactive_prompts", []) or [])
        for c in conflicts or []:
            cands.append(c.get("question", ""))
        if fork_choices and len(fork_choices) >= 2:
            cands.append(fork_choices[0])
        return [c for c in cands if c]

    def pick(self, context: str, candidates: List[str]) -> List[str]:
        if not candidates:
            return []
        if self.llm.name == "null":
            return candidates[:1]  # 规则通道：保守只发第一条
        listing = "\n".join(f"[{i}] {c}" for i, c in enumerate(candidates))
        out = self.llm.complete([
            {"role": "system", "content":
             "从候选追问中选出此刻最能推进任务的 1 条（宁少勿滥，避免打扰）。"
             "只输出序号数字，若都不合适输出 -1。"},
            {"role": "user", "content": f"上下文：{context[:800]}\n候选：\n{listing}"},
        ], temperature=0.0, max_tokens=8)
        try:
            idx = int((out or "-1").strip()[:3])
        except (TypeError, ValueError):
            idx = -1
        if 0 <= idx < len(candidates):
            return [candidates[idx]]
        return candidates[:1] if candidates and not out else []


class ActionDAG:
    """行动链 DAG 执行引擎（docs/06 §3）。handlers: {action_name: callable(params)->result}。"""

    def __init__(self, handlers: Optional[Dict[str, Callable[[Dict[str, Any]], Any]]] = None):
        self.handlers = handlers or {}

    @staticmethod
    def plan_screening(text: str, with_comments: bool = False,
                       with_credit: bool = False) -> List[DAGNode]:
        """典型甄别链（docs/06 §3.2）。"""
        nodes = [DAGNode(node_id="t1", action="call:screen",
                         params={"text": text}, kind="task")]
        dep = ["t1"]
        if with_comments:
            nodes.append(DAGNode(node_id="t2", action="call:comments",
                                 params={}, depends_on=["t1"], kind="task"))
            dep.append("t2")
        if with_credit:
            nodes.append(DAGNode(node_id="t3", action="call:credit",
                                 params={}, depends_on=["t1"], kind="task"))
            dep.append("t3")
        nodes.append(DAGNode(node_id="t_show", action="show",
                             params={}, depends_on=dep, kind="task"))
        nodes.append(DAGNode(node_id="t_action", action="call:actions",
                             params={}, depends_on=["t_show"], kind="task"))
        return nodes

    @staticmethod
    def plan_recommendation(query: str) -> List[DAGNode]:
        """推荐链：先甄别再推荐（HIGH 时推荐节点被中断——规格场景）。"""
        return [
            DAGNode(node_id="s1", action="call:screen", params={"text": query}, kind="task"),
            DAGNode(node_id="s2", action="call:recommend", params={"query": query},
                    depends_on=["s1"], kind="recommend"),
            DAGNode(node_id="s3", action="show", params={}, depends_on=["s2"], kind="task"),
        ]

    def execute(self, state: DialogueState,
                handlers: Optional[Dict[str, Callable[[Dict[str, Any]], Any]]] = None,
                ) -> DialogueState:
        """依赖驱动调度：入度 0 先执行；HIGH 动态重规划；异常 skipped 降级。"""
        handlers = {**self.handlers, **(handlers or {})}
        nodes = {n.node_id: n for n in state.dag}

        def deps_done(n: DAGNode) -> bool:
            return all(nodes.get(d) is not None
                       and nodes[d].status in ("done", "skipped", "cancelled")
                       for d in n.depends_on)

        progress = True
        while progress:
            progress = False
            for n in state.dag:
                if n.status != "pending" or not deps_done(n):
                    continue
                # 前驱被取消 → 级联取消（HIGH 重规划的传播）
                if any(nodes.get(d) and nodes[d].status == "cancelled" for d in n.depends_on):
                    n.status = "cancelled"
                    progress = True
                    continue
                n.status = "running"
                handler = handlers.get(n.action.split(":", 1)[-1]) \
                    if n.action.startswith("call:") else handlers.get(n.action)
                try:
                    if handler is None:
                        n.status = "skipped"
                        n.result = {"skipped": f"无执行器: {n.action}"}
                    else:
                        n.result = handler(n.params)
                        n.status = "done"
                        self._replan_on_high(state, n)
                except Exception as exc:  # 节点失败降级：skipped，DAG 继续
                    n.status = "skipped"
                    n.result = {"error": str(exc)[:200]}
                progress = True
        state.phase = Phase.RESPONDING
        return state

    @staticmethod
    def _replan_on_high(state: DialogueState, node: DAGNode) -> None:
        """甄别节点输出 HIGH → 取消后续推荐类节点，插入风险呈现与维权引导。"""
        result = node.result or {}
        level = result.get("level") or getattr(result, "value", None)
        is_high = (level == RiskLevel.HIGH.value) or (level == RiskLevel.HIGH) \
            or (isinstance(level, str) and level.lower() == "high")
        if not is_high:
            return
        for n in state.dag:
            if n.status == "pending" and n.kind == "recommend":
                n.status = "cancelled"
        existing = {n.node_id for n in state.dag}
        if "warn_inserted" not in existing:
            state.dag.append(DAGNode(node_id="warn_inserted", action="show",
                                     params={"mode": "risk_warning"},
                                     depends_on=[node.node_id], kind="task"))
            state.dag.append(DAGNode(node_id="action_inserted", action="call:actions",
                                     params={}, depends_on=["warn_inserted"], kind="task"))
        state.phase = Phase.RISK_PRESENTING


class DialogueManager:
    """状态机 + 快照持久化（中断恢复）。"""

    def __init__(self, config: Optional[Config] = None,
                 db_path: Optional[Path] = None,
                 llm: Optional[LLMBackend] = None):
        self.config = config or Config()
        db_file = Path(db_path) if db_path else self.config.db_file
        db_file.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_file))
        self.conn.executescript(_DDL)
        self.conn.commit()
        self.trigger = ProactiveTrigger(llm)
        self.dag_engine = ActionDAG()
        self._seq: Dict[str, int] = {}

    def new_state(self, session_id: Optional[str] = None) -> DialogueState:
        return DialogueState(session_id=session_id or uuid.uuid4().hex[:10])

    def save_snapshot(self, state: DialogueState) -> None:
        seq = self._seq.get(state.session_id, 0) + 1
        self._seq[state.session_id] = seq
        self.conn.execute(
            "INSERT OR REPLACE INTO session_snapshots(session_id, seq, payload) VALUES(?,?,?)",
            (state.session_id, seq, state.to_json()))
        self.conn.commit()

    def load_latest(self, session_id: str) -> Optional[DialogueState]:
        row = self.conn.execute(
            "SELECT payload FROM session_snapshots WHERE session_id=? "
            "ORDER BY seq DESC LIMIT 1", (session_id,)).fetchone()
        return DialogueState.from_json(row[0]) if row else None

    def transition(self, state: DialogueState, event: str,
                   payload: Optional[Dict[str, Any]] = None) -> DialogueState:
        """状态迁移（纯函数式：读 state → 改 state → 快照）。"""
        payload = payload or {}
        if event == "user_input":
            state.turn += 1
            state.history.append({"role": "user", "content": str(payload.get("text", ""))[:500]})
            if len(state.history) > self.config.work_memory_k * 2:
                state.history = state.history[-self.config.work_memory_k * 2:]
            if state.phase == Phase.CLARIFYING:
                state.clarify_attempts += 1
                state.phase = Phase.INTENT_PARSE
            else:
                state.phase = Phase.INTENT_PARSE
        elif event == "intent_resolved":
            state.pending_intent = str(payload.get("intent", ""))
            if state.pending_intent in ("screen", "trace", "recommend", "action", "memory"):
                state.phase = Phase.TASK_PLANNING
            else:
                state.phase = Phase.RESPONDING  # 越界/闲聊 → 边界声明
        elif event == "intent_ambiguous":
            if state.clarify_attempts >= 3:
                state.phase = Phase.RESPONDING  # 三次未明 → 降级为一般建议
                state.pending_question = ""
            else:
                state.phase = Phase.CLARIFYING
                state.pending_question = payload.get("question", _CLARIFY_QUESTION)
        elif event == "plan_ready":
            state.dag = payload.get("dag", [])
            state.phase = Phase.EXECUTING
        elif event == "executed":
            high = any((n.result or {}).get("level") == RiskLevel.HIGH.value
                       for n in state.dag if n.action == "call:screen")
            state.phase = Phase.RISK_PRESENTING if high else Phase.RESPONDING
        elif event == "risk_acknowledged":
            state.phase = Phase.ACTION_GUIDING
        elif event == "responded":
            state.phase = Phase.IDLE
            state.pending_question = ""
        self.save_snapshot(state)
        return state

    def close(self) -> None:
        self.conn.close()
