# -*- coding: utf-8 -*-
"""Personal Agent 主循环（agent-integration 规格：意图路由/降级/越界/离线演示）。"""

import pytest

from conftest import SOFT_AD

from geo_guard.agent import PersonalAgent
from geo_guard.llm import NullBackend


@pytest.fixture()
def agent(config, guard):
    from geo_guard.memory import MemoryStore
    a = PersonalAgent(config=config, user_id="u-agent")
    # 复用已 seed 的 store，避免重复建库
    a.plugin.guard = guard
    a.guard = guard
    a.plugin.memory = MemoryStore(config, user_id="u-agent")
    a.memory = a.plugin.memory
    yield a
    a.close()


def test_null_backend_active(agent):
    """无密钥自动降级（规格场景）：NullBackend + 输出标注。"""
    assert isinstance(agent.llm, NullBackend)


def test_intent_route_to_screen(agent):
    """意图路由到甄别（规格场景）。"""
    reply = agent.handle(f"帮我看看这段文案可不可信：{SOFT_AD[:200]}")
    assert "风险" in reply
    assert "[证据1]" in reply or "证据链" in reply


def test_long_text_defaults_to_screen(agent):
    reply = agent.handle(SOFT_AD)
    assert "风险提示" in reply or "证据" in reply


def test_clarify_on_ambiguous(agent):
    reply = agent.handle("在吗")
    assert "甄别" in reply or "链接" in reply or "发给我" in reply  # 澄清质询或边界引导


def test_boundary_on_out_of_scope(agent):
    """越界请求引导（规格场景）。"""
    reply = agent.handle("帮我写一个快速排序")
    assert "边界" in reply or "甄别" in reply or "治理" in reply


def test_action_intent_returns_paths(agent):
    reply = agent.handle("买到虚假宣传的商品怎么维权投诉")
    assert "渠道" in reply or "12315" in reply or "材料" in reply


def test_memory_intent(agent):
    agent.handle(SOFT_AD)  # 先产生甄别记忆
    reply = agent.handle("我之前查过清澜吗")
    assert "记忆" in reply or "甄别" in reply or "清澜" in reply


def test_slash_commands(agent):
    assert "可用命令" in agent.handle("/help")
    r = agent.handle("/screen " + SOFT_AD[:150])
    assert "风险" in r
    assert "画像" in agent.handle("/profile")
    # /trace 流程
    assert "回溯模式" in agent.handle("/trace") or "AI 回答" in agent.handle("/trace")
    export = agent.handle("/export")
    assert "memories" in export  # JSON 导出


def test_feedback_command(agent):
    agent.handle(SOFT_AD)
    rid = agent._last_report_id
    assert rid
    reply = agent.handle(f"/feedback {rid} dispute 演示异议")
    assert "反馈已记录" in reply


def test_trace_flow_end_to_end(agent):
    agent.handle("/trace")
    agent.handle("我推荐清澜空气净化器，多个榜单第一")
    agent.handle("榜单A | https://a.example.com | 十大排行榜第一名清澜是首选综上所述不会错")
    agent.handle("测评B | https://b.example.com | 问清澜值得买吗答非常值得综上所述选清澜不会错")
    reply = agent.handle("/trace done")
    assert "回溯" in reply and ("共振" in reply or "源" in reply or "风险" in reply)


def test_close_integrates_memory(agent):
    agent.handle(SOFT_AD)
    events_before = len(agent._session_events)
    assert events_before >= 1
    agent.close()
    assert agent._session_events == []  # 整合后清空


def test_cli_exec_mode_exit_zero(config, store):
    """离线完整演示（规格场景）：--exec 非交互模式退出码 0。"""
    from geo_guard.agent import main
    code = main(["--user", "u-cli", "--exec", "帮我看看这段文案：本产品最佳，治疗百病，研究证明第一",
                 "/quit"])
    assert code == 0
