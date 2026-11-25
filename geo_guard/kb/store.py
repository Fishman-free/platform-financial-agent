# -*- coding: utf-8 -*-
"""知识库存储（design.md D2）：SQLite 持久化真相源 + 进程内 numpy 向量索引。

治理约束（knowledge-retrieval 规格）：
- source 为空拒绝入库（I-12 反哺防投毒）；
- status=pending_review / deprecated 的条目不参与判定检索；
- 更新产生新版本行（append-only 历史表），旧版本可查，不物理删除；
- 同 entry_id + 同 version 的重复 upsert 幂等（种子构建可重跑）。

崩溃恢复：启动时从 SQLite 全量重建内存索引（向量按内容 sha256 缓存于
embed_cache 表，避免重复调用 API 嵌入）。
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from ..config import Config
from ..schema import KBEntry
from .bm25 import BM25Index
from .embeddings import get_embedder

_DDL = """
CREATE TABLE IF NOT EXISTS kb_entries (
    entry_id   TEXT PRIMARY KEY,
    kb         TEXT NOT NULL,
    title      TEXT NOT NULL,
    content    TEXT NOT NULL,
    severity   REAL DEFAULT 0.5,
    category   TEXT DEFAULT '',
    metadata   TEXT DEFAULT '{}',
    source     TEXT NOT NULL,
    version    TEXT DEFAULT '1.0.0',
    checked_at TEXT DEFAULT '',
    status     TEXT DEFAULT 'active',
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS kb_entry_versions (
    entry_id   TEXT,
    version    TEXT,
    payload    TEXT,
    archived_at TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (entry_id, version)
);
CREATE TABLE IF NOT EXISTS embed_cache (
    content_hash TEXT PRIMARY KEY,
    vector       BLOB
);
CREATE INDEX IF NOT EXISTS idx_kb_name ON kb_entries(kb);
CREATE INDEX IF NOT EXISTS idx_kb_status ON kb_entries(status);
"""


class KnowledgeStore:
    """五库统一存储与检索门面。"""

    def __init__(self, config: Optional[Config] = None, db_path: Optional[Path] = None):
        self.config = config or Config()
        self.db_path = Path(db_path) if db_path else self.config.db_file
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_DDL)
        self.conn.commit()
        self.embedder = get_embedder(self.config)
        self.bm25 = BM25Index()
        self._entries: Dict[str, KBEntry] = {}
        self._ids: List[str] = []          # 与 _matrix 行对齐
        self._matrix: Optional[np.ndarray] = None
        self._load()

    # ---------------- 治理与写入 ----------------

    def upsert(self, entry: KBEntry, bump_history: bool = True) -> bool:
        """写入/更新条目。返回 True 表示发生实际写入，False 表示幂等跳过。

        治理：source 空 → ValueError；重复(entry_id, version, content) → 跳过；
        内容变更 → 旧版本归档进 kb_entry_versions 后替换。
        """
        if not (entry.source or "").strip():
            raise ValueError(f"条目 {entry.entry_id} 缺少 source，拒绝入库（入库治理约束）")
        row = self.conn.execute(
            "SELECT content, version FROM kb_entries WHERE entry_id=?",
            (entry.entry_id,)).fetchone()
        if row is not None:
            if row["content"] == entry.content and row["version"] == entry.version:
                return False  # 幂等：完全相同，跳过
            if bump_history:
                self.conn.execute(
                    "INSERT OR REPLACE INTO kb_entry_versions(entry_id, version, payload) "
                    "VALUES(?,?,?)",
                    (entry.entry_id, row["version"],
                     json.dumps(self._row_to_entry_dict(entry.entry_id, row), ensure_ascii=False)))
        self.conn.execute(
            """INSERT INTO kb_entries(entry_id, kb, title, content, severity, category,
                                      metadata, source, version, checked_at, status)
               VALUES(?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(entry_id) DO UPDATE SET
                 kb=excluded.kb, title=excluded.title, content=excluded.content,
                 severity=excluded.severity, category=excluded.category,
                 metadata=excluded.metadata, source=excluded.source,
                 version=excluded.version, checked_at=excluded.checked_at,
                 status=excluded.status, updated_at=datetime('now')""",
            (entry.entry_id, entry.kb, entry.title, entry.content, entry.severity,
             entry.category, json.dumps(entry.metadata, ensure_ascii=False),
             entry.source, entry.version, entry.checked_at, entry.status))
        self.conn.commit()
        self._index_entry(entry)
        return True

    def _row_to_entry_dict(self, entry_id: str, row: sqlite3.Row) -> dict:
        return {"entry_id": entry_id, "content": row["content"], "version": row["version"]}

    def set_status(self, entry_id: str, status: str) -> None:
        """状态迁移（active/deprecated/pending_review）。非 active 条目退出检索索引。"""
        if status not in ("active", "deprecated", "pending_review"):
            raise ValueError(f"非法状态: {status}")
        self.conn.execute("UPDATE kb_entries SET status=?, updated_at=datetime('now') "
                          "WHERE entry_id=?", (status, entry_id))
        self.conn.commit()
        # 同步内存索引：active → 重新加载；非 active → 移出
        row = self.conn.execute(
            "SELECT * FROM kb_entries WHERE entry_id=?", (entry_id,)).fetchone()
        if row is None:
            return
        if status == "active":
            self._entries[entry_id] = self._row_to_entry(row)
        else:
            self._entries.pop(entry_id, None)
        self._rebuild_index()

    def get(self, entry_id: str) -> Optional[KBEntry]:
        """按 id 取条目（含非 active 状态——回退查 DB；内存索引只存 active）。"""
        entry = self._entries.get(entry_id)
        if entry is not None:
            return entry
        row = self.conn.execute(
            "SELECT * FROM kb_entries WHERE entry_id=?", (entry_id,)).fetchone()
        return self._row_to_entry(row) if row is not None else None

    def count(self, kb: Optional[str] = None, status: Optional[str] = "active") -> int:
        """status=None 表示不过滤状态（统计全部版本状态的条目）。"""
        conds, args = [], []
        if kb:
            conds.append("kb=?"); args.append(kb)
        if status:
            conds.append("status=?"); args.append(status)
        sql = "SELECT COUNT(*) c FROM kb_entries"
        if conds:
            sql += " WHERE " + " AND ".join(conds)
        return self.conn.execute(sql, args).fetchone()["c"]

    def iter_entries(self, kb: Optional[str] = None,
                     status: Optional[str] = "active") -> Iterable[KBEntry]:
        sql = "SELECT * FROM kb_entries"
        conds, args = [], []
        if kb:
            conds.append("kb=?"); args.append(kb)
        if status:
            conds.append("status=?"); args.append(status)
        if conds:
            sql += " WHERE " + " AND ".join(conds)
        for row in self.conn.execute(sql, args):
            yield self._row_to_entry(row)

    # ---------------- 内存索引 ----------------

    def _content_hash(self, text: str) -> str:
        return hashlib.sha256(
            f"{self.embedder.name}:{self.embedder.dim}:{text}".encode("utf-8")).hexdigest()

    def _embed_cached(self, texts: Sequence[str]) -> np.ndarray:
        """带 SQLite 缓存的批量嵌入（避免重复 API 调用；Hash 后端缓存亦无害）。"""
        vectors: List[Optional[np.ndarray]] = [None] * len(texts)
        misses: List[int] = []
        for i, t in enumerate(texts):
            h = self._content_hash(t)
            row = self.conn.execute(
                "SELECT vector FROM embed_cache WHERE content_hash=?", (h,)).fetchone()
            if row is not None:
                vectors[i] = np.frombuffer(row["vector"], dtype=np.float32).copy()
            else:
                misses.append(i)
        if misses:
            mat = self.embedder.embed_batch([texts[i] for i in misses])
            for j, i in enumerate(misses):
                vectors[i] = mat[j]
                self.conn.execute(
                    "INSERT OR REPLACE INTO embed_cache(content_hash, vector) VALUES(?,?)",
                    (self._content_hash(texts[i]), mat[j].astype(np.float32).tobytes()))
            self.conn.commit()
        return np.vstack([v for v in vectors if v is not None])

    def _index_text(self, entry: KBEntry) -> str:
        """被索引的文本 = 标题 + 正文 + 触发例句（提高话术例句的字面召回）。"""
        parts = [entry.title, entry.content]
        for ex in entry.metadata.get("trigger_examples", []) or []:
            parts.append(str(ex))
        return "\n".join(parts)

    def _index_entry(self, entry: KBEntry) -> None:
        if entry.status != "active":
            self._entries.pop(entry.entry_id, None)
            self._rebuild_index()
            return
        self._entries[entry.entry_id] = entry
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        self.bm25 = BM25Index()
        self._ids = []
        texts: List[str] = []
        for eid, entry in self._entries.items():
            self.bm25.add_doc(eid, self._index_text(entry))
            self._ids.append(eid)
            texts.append(self._index_text(entry))
        if texts:
            self._matrix = self._embed_cached(texts)
        else:
            self._matrix = np.zeros((0, self.embedder.dim), dtype=np.float32)

    def _load(self) -> None:
        self._entries = {}
        for row in self.conn.execute("SELECT * FROM kb_entries WHERE status='active'"):
            entry = self._row_to_entry(row)
            self._entries[entry.entry_id] = entry
        self._rebuild_index()

    @staticmethod
    def _row_to_entry(row: sqlite3.Row) -> KBEntry:
        return KBEntry(
            entry_id=row["entry_id"], kb=row["kb"], title=row["title"],
            content=row["content"], source=row["source"], version=row["version"],
            checked_at=row["checked_at"], status=row["status"],
            severity=row["severity"], category=row["category"],
            metadata=json.loads(row["metadata"] or "{}"),
        )

    # ---------------- 检索原语 ----------------

    def active_ids(self, kbs: Optional[Sequence[str]] = None,
                   categories: Optional[Sequence[str]] = None) -> List[str]:
        ids = [eid for eid, e in self._entries.items()
               if (not kbs or e.kb in kbs) and (not categories or e.category in categories)]
        return ids

    def bm25_search(self, query: str, top_k: int,
                    allowed: Optional[Iterable[str]] = None) -> List[tuple]:
        return self.bm25.search(query, top_k, allowed)

    def dense_scores(self, query_vec: np.ndarray,
                     allowed_ids: Optional[Sequence[str]] = None) -> Dict[str, float]:
        """候选集余弦相似度（向量已 L2 归一化，点积即余弦）。"""
        if self._matrix is None or not self._ids:
            return {}
        if allowed_ids is None:
            allowed_ids = self._ids
        idx = {eid: i for i, eid in enumerate(self._ids)}
        rows = [idx[e] for e in allowed_ids if e in idx]
        if not rows:
            return {}
        sims = self._matrix[rows] @ query_vec
        return {eid: float(s) for eid, s in zip([allowed_ids[i] for i in range(len(rows))
                                                 if allowed_ids[i] in idx], sims)}

    def embed(self, text: str) -> np.ndarray:
        return self.embedder.embed(text)

    def close(self) -> None:
        self.conn.close()
