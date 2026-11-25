# -*- coding: utf-8 -*-
"""对话状态机与行动链 DAG（dialogue-guidance 规格全部场景）。"""

import pytest

from geo_guard.dialogue import ActionDAG, DialogueManager, ProactiveTrigger
from geo_guard.llm import NullBackend
from geo_guard.schema import DAGNode, DialogueState, Phase, RiskLevel


@pytest.fixture()
def dm(config):
    d = DialogueManager(config)
    yield d
    d.close()


def test_state_machine_clarify_flow(dm):
    """意图不明进入澄清（规格场景）。"""
    s = dm.new_state("sess-1")
    dm.transition(s, "user_input", {"text": "在吗"})
    assert s.phase == Phase.INTENT_PARSE
    dm.transition(s, "intent_ambiguous", {})
    assert s.phase == Phase.CLARIFYING and s.pending_question
    # 三次未明 → 降级为一般建议
    for _ in range(3):
        dm.transition(s, "user_input", {"text": "嗯"})
        dm.transition(s, "intent_ambiguous", {})
    assert s.phase == Phase.RESPONDING and s.pending_question == ""


def test_snapshot_resume(dm):
    """中断恢复（规格场景）：进程重启后从快照续行。"""
    s = dm.new_state("sess-resume")
    dm.transition(s, "user_input", {"text": "帮我看看这段文案"})
    dm.transition(s, "intent_resolved", {"intent": "screen"})
    s.dag = ActionDAG.plan_screening("某文案")
    s.dag[0].status = "done"
    s.dag[0].result = {"level": "medium"}
    dm.save_snapshot(s)
    # 模拟重启：新 DialogueManager 实例从同一 DB 恢复
    dm2 = DialogueManager(dm.config)
    restored = dm2.load_latest("sess-resume")
    assert restored is not None
    assert restored.phase == Phase.TASK_PLANNING
    assert restored.dag[0].status == "done"
    assert restored.turn == 1
    dm2.close()


def test_dag_dependency_order():
    """依赖顺序执行（规格场景）。"""
    order = []
    handlers = {
        "screen": lambda p: (order.append("screen"), {"level": "low"})[1],
        "show": lambda p: (order.append("show"), {"shown": True})[1],
        "actions": lambda p: (order.append("actions"), {"actions": []})[1],
    }
    s = DialogueState(session_id="d1", dag=ActionDAG.plan_screening("文本"))
    ActionDAG(handlers).execute(s)
    assert order[0] == "screen" and order[-1] == "actions"
    assert all(n.status == "done" for n in s.dag)


def test_dag_node_failure_skipped():
    """节点失败降级（规格场景）：异常 → skipped，DAG 继续，不静默吞。"""
    def boom(p):
        raise RuntimeError("模拟失败")
    ran = []
    handlers = {"screen": boom,
                "show": lambda p: (ran.append("show"), {"ok": 1})[1],
                "actions": lambda p: (ran.append("actions"), {"ok": 1})[1]}
    s = DialogueState(session_id="d2", dag=ActionDAG.plan_screening("文本"))
    ActionDAG(handlers).execute(s)
    screen_node = next(n for n in s.dag if n.action == "call:screen")
    assert screen_node.status == "skipped" and "模拟失败" in screen_node.result["error"]
    assert "show" in ran  # 下游仍执行（声明缺失证据面而非中断）


def test_high_risk_replanning_cancels_recommend():
    """高风险动态重规划（规格场景）：推荐节点被取消，插入警告与维权。"""
    handlers = {
        "screen": lambda p: {"level": RiskLevel.HIGH.value},
        "recommend": lambda p: {"should": "not run"},
        "show": lambda p: {"shown": True},
        "actions": lambda p: {"actions": []},
    }
    s = DialogueState(session_id="d3", dag=ActionDAG.plan_recommendation("某推荐查询"))
    ActionDAG(handlers).execute(s)
    rec = next(n for n in s.dag if n.action == "call:recommend")
    assert rec.status == "cancelled", "HIGH 时推荐节点必须被中断"
    ids = {n.node_id for n in s.dag}
    assert "warn_inserted" in ids and "action_inserted" in ids
    assert s.phase in (Phase.RISK_PRESENTING, Phase.RESPONDING)


def test_proactive_trigger_conservative():
    """质询触发器保守：LOW 无候选；MEDIUM 必有候选；规则通道只发一条。"""
    trig = ProactiveTrigger(NullBackend())

    class FakeReport:
        def __init__(self, prompts):
            self.proactive_prompts = prompts

    assert trig.candidates(report=FakeReport([])) == []
    cands = trig.candidates(report=FakeReport(["要不要查历史差评？"]))
    assert len(cands) == 1
    picked = trig.pick("上下文", ["问题A", "问题B", "问题C"])
    assert len(picked) == 1  # 宁少勿滥
    conflicts = [{"question": "您之前更关注性价比，这次想看高端款吗？"}]
    assert trig.candidates(conflicts=conflicts) == [conflicts[0]["question"]]


def test_dag_serializable_in_state():
    s = DialogueState(session_id="d4", dag=ActionDAG.plan_screening("x"))
    s2 = DialogueState.from_json(s.to_json())
    assert len(s2.dag) == len(s.dag)
    assert all(isinstance(n, DAGNode) for n in s2.dag)
