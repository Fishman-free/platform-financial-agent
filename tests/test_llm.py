# -*- coding: utf-8 -*-
"""LLM 后端选择纪律与触点降级（agent-integration 规格）。"""

import geo_guard.llm as llm
from geo_guard.llm import (NullBackend, get_backend, touch_extract_claims,
                           touch_query_rewrite, touch_route_intent)


def test_no_key_returns_null_backend(monkeypatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert isinstance(get_backend("auto"), NullBackend)
    # 显式指定但无密钥 → 降级而非报错（规格）
    assert isinstance(get_backend("dashscope"), NullBackend)
    assert isinstance(get_backend("off"), NullBackend)


def test_null_backend_complete_returns_none():
    assert NullBackend().complete([{"role": "user", "content": "hi"}]) is None


def test_query_rewrite_fallback_includes_original():
    qs = touch_query_rewrite(NullBackend(), "清澜空气净化器怎么样",
                             entities={"brand": "清澜", "category": "空气净化器"})
    assert qs[0] == "清澜空气净化器怎么样"
    assert any("空气净化器" in q and ("虚假宣传" in q or "投诉" in q) for q in qs[1:])


def test_extract_claims_rule_path_uses_trigger_words():
    text = "本品效果最好，能治疗高血压。今天天气不错，适合散步。保证收益年化15%。"
    claims = touch_extract_claims(NullBackend(), text)
    joined = " ".join(c["text"] for c in claims)
    assert "治疗高血压" in joined and "保证收益" in joined
    assert "散步" not in joined  # 无触发词的句子不抽取


def test_route_intent_rule_path():
    r = touch_route_intent(NullBackend(), "这个博主说的话靠谱吗")
    assert r["intent"] == "screen"
    r = touch_route_intent(NullBackend(), "买到假货怎么投诉")
    assert r["intent"] == "action"
    r = touch_route_intent(NullBackend(), "帮我写一段代码")
    assert r["intent"] == "other"
