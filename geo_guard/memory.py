# -*- coding: utf-8 -*-
"""三层记忆与动态用户画像（memory-profile 规格，docs/05）。

- 工作记忆：会话内环形缓冲（deque, K=config.work_memory_k）；
- 情景记忆：SQLite + 向量召回，时间衰减 w(t)=e^(-λt) 分型半衰期
  （甄别结论 90 天 / 一般偏好 30 天），importance>=0.8 受保护不清除；
- 语义记忆：画像状态向量（风险规避/价格敏感/品牌导向/证据披露偏好）
  + 实体信任表，EMA 隐式更新，显式声明 confidence=1.0。

红线（D-08）：画像只调制呈现与引导，永不参与风险分计算——本模块不向
detection/fusion 提供任何输入，依赖方向由包结构强制。
隐私：全本地 SQLite；支持查看/删除/清空/导出（用户主权）。
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
import time
import uuid
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

from .config import Config
from .kb.embeddings import HashEmbedding
from .schema import MemoryRecord

_DDL = """
CREATE TABLE IF NOT EXISTS memories (
    memory_id TEXT PRIMARY KEY,
    user_id   TEXT NOT NULL,
    kind      TEXT NOT NULL,
    subject   TEXT DEFAULT '',
    content   TEXT NOT NULL,
    risk_level TEXT DEFAULT '',
    importance REAL DEFAULT 0.5,
    vector    BLOB,
    created_at REAL,
    last_accessed_at REAL DEFAULT 0,
    access_count INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS profile_state (
    user_id TEXT, key TEXT, value REAL, confidence REAL, updated_at REAL,
    PRIMARY KEY (user_id, key)
);
CREATE TABLE IF NOT EXISTS entity_trust (
    user_id TEXT, entity TEXT, entity_type TEXT, trust REAL,
    evidence_count INTEGER DEFAULT 1, updated_at REAL,
    PRIMARY KEY (user_id, entity, entity_type)
);
CREATE INDEX IF NOT EXISTS idx_mem_user ON memories(user_id);
"""

PROFILE_KEYS = ("risk_aversion", "price_sensitivity", "brand_orientation", "evidence_appetite")

# 画像冲突检测词表（规则路径；LLM 路径在 dialogue 层可选增强）
_PRICE_HIGH = ("不差钱", "旗舰", "顶配", "最贵的", "高端", "随便买", "价格不是问题")
_PRICE_LOW = ("便宜", "性价比", "预算", "平替", "省钱", "实惠", "低价")
_RISK_CARELESS = ("放心买", "肯定没事", "不用查了", "直接下单", "闭眼入")


@dataclass
class ProfileSnapshot:
    """画像只读快照（供呈现层调制；判定层禁止消费——D-08）。"""
    state: Dict[str, float]
    confidence: Dict[str, float]
    entity_trust: Dict[str, float]

    @property
    def detail_level(self) -> str:
        """证据披露偏好 → explain 的默认披露档。"""
        v = self.state.get("evidence_appetite", 0.5)
        return "full" if v >= 0.7 else ("standard" if v >= 0.4 else "summary")

    @property
    def warning_style(self) -> str:
        v = self.state.get("risk_aversion", 0.5)
        return "强调风险细节，建议保守决策" if v >= 0.7 else (
            "简要提示风险即可" if v <= 0.3 else "平衡呈现风险与收益")


class MemoryStore:
    def __init__(self, config: Optional[Config] = None,
                 db_path: Optional[Path] = None, user_id: str = "default"):
        self.config = config or Config()
        self.user_id = user_id
        db_file = Path(db_path) if db_path else self.config.db_file
        db_file.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_file))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_DDL)
        self.conn.commit()
        self.embedder = HashEmbedding(self.config.embed_dim)  # 记忆召回用本地嵌入即可
        self.working: Deque[Dict[str, str]] = deque(maxlen=self.config.work_memory_k)
        self.session_id = uuid.uuid4().hex[:8]

    # ---------------- 工作记忆 ----------------

    def push_turn(self, role: str, text: str) -> None:
        self.working.append({"role": role, "text": text})

    def recent_turns(self, n: Optional[int] = None) -> List[Dict[str, str]]:
        items = list(self.working)
        return items[-n:] if n else items

    # ---------------- 情景记忆 ----------------

    def add(self, kind: str, content: str, subject: str = "",
            risk_level: str = "", importance: Optional[float] = None) -> str:
        mid = f"m-{uuid.uuid4().hex[:10]}"
        if importance is None:
            importance = {"screening": 0.7, "feedback": 0.5,
                          "preference": 0.6, "event": 0.4}.get(kind, 0.5)
        vec = self.embedder.embed(content)
        self.conn.execute(
            "INSERT INTO memories(memory_id, user_id, kind, subject, content, risk_level,"
            " importance, vector, created_at) VALUES(?,?,?,?,?,?,?,?,?)",
            (mid, self.user_id, kind, subject, content, risk_level,
             importance, vec.astype(np.float32).tobytes(), time.time()))
        self.conn.commit()
        return mid

    def recall(self, query: str, top_k: Optional[int] = None,
               kinds: Optional[List[str]] = None) -> List[Tuple[MemoryRecord, float]]:
        """语义召回 × 时间衰减 × 访问加成（docs/05 §3）。"""
        top_k = top_k or self.config.recall_top_k
        rows = self.conn.execute(
            "SELECT * FROM memories WHERE user_id=?" +
            (f" AND kind IN ({','.join('?' * len(kinds))})" if kinds else ""),
            (self.user_id, *(kinds or []))).fetchall()
        if not rows:
            return []
        qv = self.embedder.embed(query)
        now = time.time()
        scored: List[Tuple[MemoryRecord, float]] = []
        for r in rows:
            vec = np.frombuffer(r["vector"], dtype=np.float32)
            cos = float(np.dot(qv, vec) / (np.linalg.norm(qv) * np.linalg.norm(vec) + 1e-9))
            lam = (self.config.lambda_screening if r["kind"] == "screening"
                   else self.config.lambda_preference)
            age_days = max(0.0, (now - r["created_at"]) / 86400.0)
            decay = math.exp(-lam * age_days)
            rec = MemoryRecord(
                memory_id=r["memory_id"], kind=r["kind"], subject=r["subject"],
                content=r["content"], risk_level=r["risk_level"],
                importance=r["importance"], created_at=r["created_at"],
                last_accessed_at=r["last_accessed_at"], access_count=r["access_count"])
            score = max(cos, 0.0) * decay * (1 + 0.1 * math.log(1 + rec.access_count))
            # 受保护记忆（高风险结论/显式声明）设置召回分数下限，衰减不清除
            if rec.importance >= self.config.importance_protect:
                score = max(score, 0.05 * max(cos, 0.0))
            scored.append((rec, score))
        scored.sort(key=lambda t: t[1], reverse=True)
        top = [(rec, s) for rec, s in scored[:top_k] if s > 0.01]
        if top:
            ids = [rec.memory_id for rec, _ in top]
            self.conn.executemany(
                "UPDATE memories SET last_accessed_at=?, access_count=access_count+1 "
                "WHERE memory_id=?", [(now, i) for i in ids])
            self.conn.commit()
        return top

    def remember_screening(self, subject: str, level: str, score: float,
                           summary: str) -> str:
        """甄别结论入情景记忆（高风险=受保护记忆）。"""
        importance = 0.9 if level == "high" else (0.7 if level == "medium" else 0.5)
        return self.add("screening", summary, subject=subject,
                        risk_level=level, importance=importance)

    # ---------------- 语义记忆：画像 ----------------

    def get_profile(self) -> ProfileSnapshot:
        state: Dict[str, float] = {k: 0.5 for k in PROFILE_KEYS}
        conf: Dict[str, float] = {k: 0.0 for k in PROFILE_KEYS}
        for r in self.conn.execute(
                "SELECT key, value, confidence FROM profile_state WHERE user_id=?",
                (self.user_id,)):
            if r["key"] in state:
                state[r["key"]] = r["value"]
                conf[r["key"]] = r["confidence"]
        trust = {r["entity"]: r["trust"] for r in self.conn.execute(
            "SELECT entity, trust FROM entity_trust WHERE user_id=?", (self.user_id,))}
        return ProfileSnapshot(state=state, confidence=conf, entity_trust=trust)

    def set_profile(self, key: str, value: float, confidence: float = 1.0) -> None:
        if key not in PROFILE_KEYS:
            raise ValueError(f"未知画像键: {key}（合法：{PROFILE_KEYS}）")
        value = max(0.0, min(1.0, float(value)))
        self.conn.execute(
            "INSERT INTO profile_state(user_id, key, value, confidence, updated_at) "
            "VALUES(?,?,?,?,?) ON CONFLICT(user_id, key) DO UPDATE SET "
            "value=excluded.value, confidence=excluded.confidence, updated_at=excluded.updated_at",
            (self.user_id, key, value, confidence, time.time()))
        self.conn.commit()

    def update_implicit(self, key: str, signal: float) -> None:
        """隐式行为 EMA 增量更新（docs/05 §4.1）。signal ∈ [0,1]。"""
        snap = self.get_profile()
        cur = snap.state.get(key, 0.5)
        a = self.config.ema_alpha
        new = (1 - a) * cur + a * max(0.0, min(1.0, signal))
        conf = min(0.9, snap.confidence.get(key, 0.0) + 0.1)  # 隐式证据置信缓升，上限0.9
        self.set_profile(key, new, confidence=conf)

    def set_entity_trust(self, entity: str, entity_type: str, delta: float,
                         absolute: Optional[float] = None) -> float:
        row = self.conn.execute(
            "SELECT trust, evidence_count FROM entity_trust "
            "WHERE user_id=? AND entity=? AND entity_type=?",
            (self.user_id, entity, entity_type)).fetchone()
        if absolute is not None:
            trust, cnt = max(-1.0, min(1.0, absolute)), 1
        elif row:
            cnt = row["evidence_count"] + 1
            trust = max(-1.0, min(1.0, row["trust"] + delta))
        else:
            trust, cnt = max(-1.0, min(1.0, delta)), 1
        self.conn.execute(
            "INSERT INTO entity_trust(user_id, entity, entity_type, trust, evidence_count,"
            " updated_at) VALUES(?,?,?,?,?,?) "
            "ON CONFLICT(user_id, entity, entity_type) DO UPDATE SET trust=excluded.trust,"
            " evidence_count=excluded.evidence_count, updated_at=excluded.updated_at",
            (self.user_id, entity, entity_type, trust, cnt, time.time()))
        self.conn.commit()
        return trust

    # ---------------- 偏好冲突检测（docs/05 §4.2） ----------------

    def detect_conflicts(self, request_text: str) -> List[Dict[str, Any]]:
        """返回冲突列表 [{key, profile_value, signal, question}]。
        不静默改画像——冲突必须经主动质询确认（memory-profile 规格）。"""
        snap = self.get_profile()
        conflicts: List[Dict[str, Any]] = []
        ps = snap.state.get("price_sensitivity", 0.5)
        pc = snap.confidence.get("price_sensitivity", 0.0)
        if pc >= 0.5:
            if ps >= 0.7 and any(w in request_text for w in _PRICE_HIGH):
                conflicts.append({
                    "key": "price_sensitivity", "profile_value": ps, "signal": "high_budget",
                    "question": "您之前更关注性价比，这次想看看高端款吗？"})
            elif ps <= 0.3 and any(w in request_text for w in _PRICE_LOW):
                conflicts.append({
                    "key": "price_sensitivity", "profile_value": ps, "signal": "low_budget",
                    "question": "您之前不太在意价格，这次想优先找高性价比的吗？"})
        ra = snap.state.get("risk_aversion", 0.5)
        if snap.confidence.get("risk_aversion", 0.0) >= 0.5 and ra >= 0.7 \
                and any(w in request_text for w in _RISK_CARELESS):
            conflicts.append({
                "key": "risk_aversion", "profile_value": ra, "signal": "risk_seeking",
                "question": "您通常比较谨慎，这次确定跳过风险核查直接下单吗？"})
        return conflicts

    def confirm_conflict(self, key: str, new_value: float, note: str = "") -> None:
        """用户确认后修正画像，并把冲突事件写入情景记忆（可审计）。"""
        self.set_profile(key, new_value, confidence=1.0)
        self.add("event", f"画像修正：{key} → {new_value:.2f}（用户确认）。{note}",
                 subject=key, importance=0.6)

    # ---------------- 会话整合与用户主权 ----------------

    def integrate_session(self, events: List[Dict[str, Any]]) -> int:
        """会话结束时把短期缓冲中有长期价值的事件提炼入情景记忆（docs/05 §3）。
        events: [{kind, content, subject?, risk_level?, importance?}]"""
        n = 0
        for ev in events:
            if ev.get("kind") in ("screening", "preference", "feedback") and ev.get("content"):
                self.add(ev["kind"], ev["content"], subject=ev.get("subject", ""),
                         risk_level=ev.get("risk_level", ""),
                         importance=ev.get("importance"))
                n += 1
        return n

    def export_json(self) -> str:
        data = {
            "user_id": self.user_id,
            "exported_at": time.time(),
            "profile": self.get_profile().state,
            "entity_trust": self.get_profile().entity_trust,
            "memories": [dict(r) for r in self.conn.execute(
                "SELECT memory_id, kind, subject, content, risk_level, importance,"
                " created_at FROM memories WHERE user_id=?", (self.user_id,))],
        }
        return json.dumps(data, ensure_ascii=False, indent=2)

    def delete(self, memory_id: str) -> bool:
        cur = self.conn.execute(
            "DELETE FROM memories WHERE memory_id=? AND user_id=?",
            (memory_id, self.user_id))
        self.conn.commit()
        return cur.rowcount > 0

    def clear_all(self, confirm: bool = False) -> int:
        """清空全部数据（需显式确认；不可恢复——用户主权条款）。"""
        if not confirm:
            raise PermissionError("清空全部数据需要 confirm=True 二次确认")
        n = self.conn.execute(
            "SELECT COUNT(*) c FROM memories WHERE user_id=?", (self.user_id,)).fetchone()["c"]
        self.conn.execute("DELETE FROM memories WHERE user_id=?", (self.user_id,))
        self.conn.execute("DELETE FROM profile_state WHERE user_id=?", (self.user_id,))
        self.conn.execute("DELETE FROM entity_trust WHERE user_id=?", (self.user_id,))
        self.conn.commit()
        self.working.clear()
        return n

    def close(self) -> None:
        self.conn.close()
