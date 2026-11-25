# -*- coding: utf-8 -*-
"""三层记忆与画像（memory-profile 规格全部场景）。"""

import time

import pytest

from geo_guard.config import Config
from geo_guard.memory import MemoryStore


@pytest.fixture()
def mem(config):
    m = MemoryStore(config, user_id="u-test")
    yield m
    m.close()


def test_cross_session_recall(mem, config):
    """跨会话召回甄别历史（规格场景）：重建实例后记忆仍在。"""
    mem.remember_screening("清澜净化器", "high", 0.88,
                           "甄别「清澜净化器」：high，分数 0.88")
    mem.close()
    m2 = MemoryStore(config, user_id="u-test")
    hits = m2.recall("清澜 净化器 甄别过吗")
    assert hits and "清澜" in hits[0][0].content
    m2.close()


def test_time_decay_ordering(mem):
    """旧偏好让位新偏好（规格场景）。"""
    old_id = mem.add("preference", "用户偏好人文景点旅游", subject="旅游", importance=0.3)
    # 手动把旧记忆时间拨回 60 天前
    mem.conn.execute("UPDATE memories SET created_at=? WHERE memory_id=?",
                     (time.time() - 60 * 86400, old_id))
    mem.conn.commit()
    mem.add("preference", "用户最近偏好自然景点徒步旅游", subject="旅游", importance=0.3)
    hits = mem.recall("旅游 景点 偏好")
    assert len(hits) >= 2
    assert "自然景点" in hits[0][0].content, "近期记忆应排在前面"


def test_importance_protected_from_decay(mem):
    """高风险结论不被遗忘（规格场景）：超半衰期仍可召回。"""
    mid = mem.remember_screening("某品牌", "high", 0.9, "甄别「某品牌」：high，投毒集群特征")
    mem.conn.execute("UPDATE memories SET created_at=?, last_accessed_at=0, access_count=0 "
                     "WHERE memory_id=?", (time.time() - 200 * 86400, mid))
    mem.conn.commit()
    hits = mem.recall("某品牌 甄别")
    ids = [rec.memory_id for rec, _ in hits]
    assert mid in ids, "importance>=0.8 的记忆衰减后仍必须可召回"


def test_explicit_statement_full_confidence(mem):
    mem.set_profile("price_sensitivity", 0.9, confidence=1.0)
    snap = mem.get_profile()
    assert snap.state["price_sensitivity"] == 0.9 and snap.confidence["price_sensitivity"] == 1.0


def test_implicit_ema_update(mem):
    mem.set_profile("risk_aversion", 0.5, confidence=0.6)
    mem.update_implicit("risk_aversion", 1.0)
    v = mem.get_profile().state["risk_aversion"]
    assert 0.5 < v < 1.0  # EMA α=0.3 → 0.65


def test_conflict_detection_triggers_question(mem):
    """冲突触发质询（规格场景）：不静默改画像。"""
    mem.set_profile("price_sensitivity", 0.9, confidence=1.0)
    before = mem.get_profile().state["price_sensitivity"]
    conflicts = mem.detect_conflicts("这次不差钱，直接上旗舰款")
    assert conflicts and "高端" in conflicts[0]["question"]
    # 检测本身不改画像
    assert mem.get_profile().state["price_sensitivity"] == before
    # 用户确认后修正 + 冲突事件入情景记忆
    mem.confirm_conflict("price_sensitivity", 0.2, note="用户确认看高端款")
    assert mem.get_profile().state["price_sensitivity"] == 0.2
    hits = mem.recall("画像修正", kinds=["event"])
    assert hits and "price_sensitivity" in hits[0][0].content


def test_entity_trust(mem):
    t = mem.set_entity_trust("某博主", "influencer", delta=-0.5)
    assert t == -0.5
    t2 = mem.set_entity_trust("某博主", "influencer", delta=0.2)
    assert abs(t2 - (-0.3)) < 1e-9
    assert mem.get_profile().entity_trust["某博主"] == pytest.approx(-0.3)


def test_session_integration(mem):
    n = mem.integrate_session([
        {"kind": "screening", "content": "甄别A：high", "subject": "A", "risk_level": "high"},
        {"kind": "preference", "content": "预算300以内"},
        {"kind": "event", "content": "不入库的类型事件"},  # kind=event 不在整合白名单
    ])
    assert n == 2


def test_export_delete_clear(mem):
    """用户数据主权（规格场景）。"""
    mid = mem.add("preference", "喜欢人文景点")
    exported = mem.export_json()
    assert "喜欢人文景点" in exported
    assert mem.delete(mid)
    assert not mem.delete("m-nonexistent")
    mem.add("preference", "临时记忆")
    with pytest.raises(PermissionError):
        mem.clear_all(confirm=False)
    n = mem.clear_all(confirm=True)
    assert n >= 1
    assert mem.recall("人文景点") == []
    assert mem.get_profile().entity_trust == {}


def test_working_memory_ring_buffer(mem):
    for i in range(mem.config.work_memory_k + 10):
        mem.push_turn("user", f"消息{i}")
    assert len(mem.working) == mem.config.work_memory_k
    assert mem.recent_turns(2)[-1]["text"] == f"消息{mem.config.work_memory_k + 9}"
