# -*- coding: utf-8 -*-
"""Agent 插件形态（agent-integration 规格全部场景）。"""

import pytest

from conftest import INJECTION_TEXT, SOFT_AD

from geo_guard.plugin import HOST_SYSTEM_PROMPT, GeoGuardPlugin

EXPECTED_TOOLS = {"screen_content", "trace_ai_answer", "verify_claims", "check_influencer",
                  "recall_memory", "remember", "suggest_actions", "give_feedback"}


@pytest.fixture()
def plugin(guard, config):
    from geo_guard.memory import MemoryStore
    mem = MemoryStore(config, user_id="u-plugin")
    return GeoGuardPlugin(user_id="u-plugin", guard=guard, memory=mem, config=config)


def test_schema_completeness(plugin):
    """schema 合法性（规格场景）：8 工具全在列，各有 name/description/parameters。"""
    schemas = plugin.tool_schemas()
    names = {s["function"]["name"] for s in schemas}
    assert names == EXPECTED_TOOLS
    for s in schemas:
        assert s["type"] == "function"
        fn = s["function"]
        assert fn["description"] and fn["parameters"]["type"] == "object"
        assert isinstance(fn["parameters"]["properties"], dict)


def test_dispatch_all_tools_callable(plugin):
    """全工具可 dispatch（无 LLM 环境）。"""
    r = plugin.dispatch("screen_content", {"text": SOFT_AD})
    assert "error" not in r and r["report"]["level"] in ("high", "medium", "low")
    rid = r["report"]["report_id"]
    assert r["rendered"]  # 渲染文本非空

    r = plugin.dispatch("trace_ai_answer", {
        "answer_text": "推荐清澜", "cited_sources": [{"title": "s", "text": SOFT_AD[:200]}]})
    assert "error" not in r and "suspicion_level" in r["report"]

    r = plugin.dispatch("verify_claims", {"text": SOFT_AD})
    assert "error" not in r and isinstance(r["claims"], list)

    r = plugin.dispatch("check_influencer", {"name": "合成博主B"})
    assert "error" not in r and r["found"] and r["credit_score"] >= 0.7

    r = plugin.dispatch("remember", {"content": "偏好测试", "kind": "preference"})
    assert "error" not in r and r["memory_id"]

    r = plugin.dispatch("recall_memory", {"query": "偏好"})
    assert "error" not in r and len(r["memories"]) >= 1

    r = plugin.dispatch("suggest_actions", {"risk_level": "high"})
    assert "error" not in r and r["actions"] and r["rendered"]

    r = plugin.dispatch("give_feedback", {"report_id": rid, "verdict": "agree"})
    assert "error" not in r and r["feedback_id"]


def test_structured_error_missing_field(plugin):
    """非法参数自纠错（规格场景）：缺参返回结构化错误，不抛异常。"""
    r = plugin.dispatch("screen_content", {})
    assert r["error"]["type"] == "invalid_params"
    assert r["error"]["missing_field"] == "text"
    # 宿主补参重试成功
    r2 = plugin.dispatch("screen_content", {"text": SOFT_AD})
    assert "error" not in r2


def test_structured_error_unknown_tool(plugin):
    r = plugin.dispatch("launch_missiles", {})
    assert r["error"]["type"] == "unknown_tool"
    assert "screen_content" in r["error"]["message"]


def test_screen_content_writes_memory(plugin):
    """甄别结论自动入情景记忆（跨会话召回基础）。"""
    plugin.dispatch("screen_content", {"text": INJECTION_TEXT, "subject": "注入测试"})
    r = plugin.dispatch("recall_memory", {"query": "甄别", "kinds": ["screening"]})
    assert r["memories"], "甄别后应写入情景记忆"


def test_host_prompt_distributed(plugin):
    """宿主使用契约分发（规格场景）。"""
    assert "screen_content" in HOST_SYSTEM_PROMPT
    assert "风险提示" in HOST_SYSTEM_PROMPT
    assert "give_feedback" in HOST_SYSTEM_PROMPT
