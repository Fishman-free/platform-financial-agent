# -*- coding: utf-8 -*-
"""数据筛查与标注管线（data-ingest 规格，docs/04）。

管线：标准化 Evidence Bundle → simhash 去重（保留复用计数）→ 弱标签 →
复核队列（不确定度排序）→ 入库治理 → 反馈闭环（异议累计触发复审）。

CLI：`python -m geo_guard.ingest --seed` 从 knowledge/seed/*.json 幂等构建
种子知识库。合成对抗集目录（knowledge/adversarial）永不加载（红线 I-29）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .config import Config
from .detection.comments import hamming, simhash64
from .kb.store import KnowledgeStore
from .schema import KBEntry

_DDL = """
CREATE TABLE IF NOT EXISTS ingest_items (
    item_id     TEXT PRIMARY KEY,
    text        TEXT NOT NULL,
    title       TEXT DEFAULT '',
    author      TEXT DEFAULT '',
    domain      TEXT DEFAULT '',
    published_at TEXT DEFAULT '',
    url         TEXT DEFAULT '',
    simhash     INTEGER,
    reuse_count INTEGER DEFAULT 1,
    reuse_domains TEXT DEFAULT '[]',
    weak_label  TEXT DEFAULT '',
    weak_confidence TEXT DEFAULT '',
    label       TEXT DEFAULT '',
    status      TEXT DEFAULT 'queued',
    uncertainty REAL DEFAULT 0.5,
    created_at  REAL
);
CREATE TABLE IF NOT EXISTS feedback (
    feedback_id TEXT PRIMARY KEY,
    report_id   TEXT,
    verdict     TEXT,
    note        TEXT DEFAULT '',
    kb_refs     TEXT DEFAULT '[]',
    created_at  REAL
);
CREATE INDEX IF NOT EXISTS idx_ingest_status ON ingest_items(status);
"""

WEAK_LABELS = ("geo_poisoning", "suspicious_marketing", "benign", "undetermined")


@dataclass
class EvidenceBundle:
    """标准化证据包（docs/04 §2①）：管线不感知来源形态。"""
    text: str
    title: str = ""
    author: str = ""
    domain: str = ""
    published_at: str = ""
    url: str = ""
    comments: List[Dict[str, Any]] = field(default_factory=list)
    # 去重阶段填充的跨站复用信号（docs/04 §2②：复用计数是特征而非丢弃理由）
    reuse_count: int = 1
    reuse_domains: List[str] = field(default_factory=list)


def normalize(raw: Dict[str, Any]) -> EvidenceBundle:
    """任意来源 → Evidence Bundle。"""
    text = str(raw.get("text") or raw.get("content") or raw.get("body") or "").strip()
    if not text:
        raise ValueError("证据包需要非空 text/content 字段")
    return EvidenceBundle(
        text=text,
        title=str(raw.get("title", "")),
        author=str(raw.get("author", "")),
        domain=str(raw.get("domain", "")),
        published_at=str(raw.get("published_at", "")),
        url=str(raw.get("url", "")),
        comments=list(raw.get("comments", []) or []),
    )


class IngestPipeline:
    def __init__(self, store: KnowledgeStore, config: Optional[Config] = None,
                 guard: Optional[Any] = None):
        self.store = store
        self.config = config or store.config
        self.conn = store.conn
        self.conn.executescript(_DDL)
        self.conn.commit()
        self._guard = guard  # GeoGuard 实例（弱标签需要跑甄别管线；可空 → 仅规则弱标签）

    # ---------------- ② 去重（保留复用信号） ----------------

    def dedup(self, bundles: Sequence[EvidenceBundle]) -> List[EvidenceBundle]:
        """simhash 近重复聚类：代表条目累加 reuse_count / reuse_domains。

        跨站复用本身是投毒信号（I-23 修订）：不丢弃近重复，而是把复用计数
        与复用域列表累加到代表条目上，作为 F4/F6 的输入特征。
        同域重复只计数不记域。注意：会原地更新代表 bundle 的复用字段。
        """
        reps: List[List[Any]] = []  # [[simhash, bundle], ...]
        for b in bundles:
            h = simhash64(b.text)
            merged = False
            for item in reps:
                if hamming(h, item[0]) <= self.config.simhash_distance:
                    rep = item[1]
                    rep.reuse_count += 1
                    if (b.domain and b.domain != rep.domain
                            and b.domain not in rep.reuse_domains):
                        rep.reuse_domains.append(b.domain)
                    merged = True
                    break
            if not merged:
                b.reuse_count = 1
                b.reuse_domains = []
                reps.append([h, b])
        return [item[1] for item in reps]

    # ---------------- ③ 弱标签 ----------------

    def weak_label(self, bundle: EvidenceBundle) -> Tuple[str, str, float]:
        """返回 (label, confidence, risk_score)。白名单不免检（规格）。"""
        from .config import WHITELIST_DOMAINS
        whitelist = any(bundle.domain == w or bundle.domain.endswith("." + w)
                        for w in WHITELIST_DOMAINS) if bundle.domain else False
        if self._guard is not None:
            report = self._guard.screen_content(
                bundle.text,
                source_meta={"domain": bundle.domain, "author": bundle.author,
                             "published_at": bundle.published_at, "url": bundle.url,
                             "reuse_count": getattr(bundle, "reuse_count", 1),
                             "reuse_domains": getattr(bundle, "reuse_domains", [])},
                comments=bundle.comments)
            score = report.score
            injection = any(e.channel == "injection" for e in report.evidence)
            contradicted = any(c.verdict == "contradicted" for c in report.claims)
            strong_rhet = any(e.channel == "rhetoric" and e.detail.get("strength") == "strong"
                              for e in report.evidence)
            if injection:
                return "geo_poisoning", "high", score
            if strong_rhet and contradicted:
                return "geo_poisoning", "medium", score
            if whitelist:
                # 白名单不免检：命中话术或事实矛盾 → 不判良性，进复核
                # （"权威站点内容异常，建议人工复核"——可能被入侵或评论区注入）
                if strong_rhet or contradicted:
                    return "suspicious_marketing", "low", score
                return "benign", "medium", score
            if strong_rhet:
                return "suspicious_marketing", "low", score
            return "undetermined", "", score
        # 无 guard 时的纯规则弱标签
        return ("benign", "low", 0.0) if whitelist else ("undetermined", "", 0.5)

    # ---------------- ④ 入队与复核 ----------------

    def submit(self, bundle: EvidenceBundle) -> str:
        item_id = f"it-{uuid.uuid4().hex[:10]}"
        label, conf, score = self.weak_label(bundle)
        t_high, t_low = self.config.t_high, self.config.t_low
        uncertainty = min(abs(score - t_high), abs(score - t_low))
        status = "queued" if label in ("undetermined", "suspicious_marketing") or conf == "low" \
            else "auto_labeled"
        self.conn.execute(
            "INSERT INTO ingest_items(item_id, text, title, author, domain, published_at,"
            " url, simhash, reuse_count, reuse_domains, weak_label, weak_confidence,"
            " status, uncertainty, created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (item_id, bundle.text, bundle.title, bundle.author, bundle.domain,
             bundle.published_at, bundle.url, simhash64(bundle.text),
             getattr(bundle, "reuse_count", 1),
             json.dumps(getattr(bundle, "reuse_domains", []), ensure_ascii=False),
             label, conf, status, uncertainty, time.time()))
        self.conn.commit()
        return item_id

    def review_queue(self, limit: int = 50) -> List[Dict[str, Any]]:
        """复核队列：不确定度升序（最贴近阈值的优先），其次弱标签严重度。"""
        rows = self.conn.execute(
            "SELECT * FROM ingest_items WHERE status='queued' "
            "ORDER BY uncertainty ASC, CASE weak_label WHEN 'geo_poisoning' THEN 0 "
            "WHEN 'suspicious_marketing' THEN 1 ELSE 2 END ASC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    def annotate(self, item_id: str, label: str, annotator: str,
                 reviewer: str = "", evidence_spans: Optional[List] = None,
                 pattern_tags: Optional[List[str]] = None,
                 guideline_version: str = "1.0") -> Dict[str, Any]:
        """人工标注（docs/04 §3 schema）。双人分歧 → disputed 进仲裁，不自动合并。"""
        if label not in WEAK_LABELS:
            raise ValueError(f"非法标签: {label}（合法：{WEAK_LABELS}）")
        row = self.conn.execute(
            "SELECT status FROM ingest_items WHERE item_id=?", (item_id,)).fetchone()
        if row is None:
            raise KeyError(item_id)
        payload = {
            "sample_id": item_id, "label": label,
            "evidence_spans": evidence_spans or [], "pattern_tags": pattern_tags or [],
            "claims": [], "annotator": annotator, "reviewer": reviewer,
            "agreement": "final" if reviewer else "pending_review",
            "guideline_version": guideline_version,
        }
        status = "annotated" if reviewer else "annotated_pending_review"
        self.conn.execute(
            "UPDATE ingest_items SET label=?, status=? WHERE item_id=?",
            (label, status, item_id))
        self.conn.commit()
        return payload

    # ---------------- ⑥ 反馈闭环 ----------------

    def record_feedback(self, report_id: str, verdict: str, note: str = "",
                        kb_refs: Optional[List[str]] = None) -> Dict[str, Any]:
        """认同/异议入库；异议累计达阈值 → 关联条目 pending_review（退出线上判定）。"""
        if verdict not in ("agree", "dispute"):
            raise ValueError("verdict 仅支持 agree/dispute")
        fid = f"fb-{uuid.uuid4().hex[:10]}"
        refs = kb_refs or []
        self.conn.execute(
            "INSERT INTO feedback(feedback_id, report_id, verdict, note, kb_refs,"
            " created_at) VALUES(?,?,?,?,?,?)",
            (fid, report_id, verdict, note, json.dumps(refs), time.time()))
        self.conn.commit()
        flagged: List[str] = []
        if verdict == "dispute":
            for ref in refs:
                n = self.conn.execute(
                    "SELECT COUNT(*) c FROM feedback WHERE verdict='dispute' "
                    "AND kb_refs LIKE ?", (f'%"{ref}"%',)).fetchone()["c"]
                if n >= self.config.feedback_dispute_threshold:
                    self.store.set_status(ref, "pending_review")
                    flagged.append(ref)
        return {"feedback_id": fid, "queued_for_review": verdict == "dispute",
                "flagged_entries": flagged}

    # ---------------- 种子构建 ----------------

    def seed_from_dir(self, seed_dir: Optional[Path] = None) -> Dict[str, int]:
        """从 knowledge/seed/*.json 幂等构建五库。对抗集目录永不读取（红线）。"""
        seed_dir = Path(seed_dir) if seed_dir else self.config.seed_path
        adv = self.config.adversarial_path.resolve()
        counts: Dict[str, int] = {}
        for fp in sorted(seed_dir.glob("*.json")):
            if adv in fp.resolve().parents or fp.resolve().parent == adv:
                continue  # 隔离目录防御性跳过（即便被误放进 seed 下）
            data = json.loads(fp.read_text(encoding="utf-8"))
            kb = data.get("kb", "")
            written = 0
            for e in data.get("entries", []):
                entry = KBEntry(
                    entry_id=e["entry_id"], kb=e.get("kb", kb), title=e["title"],
                    content=e["content"], source=e["source"],
                    version=e.get("version", "1.0.0"),
                    checked_at=e.get("checked_at", ""),
                    status=e.get("status", "active"),
                    severity=float(e.get("severity", 0.5)),
                    category=e.get("category", ""),
                    metadata=e.get("metadata", {}))
                if self.store.upsert(entry):
                    written += 1
            counts[fp.name] = written
        return counts


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="geo_guard.ingest",
                                     description="GEO 甄别知识库构建与数据管线")
    parser.add_argument("--seed", action="store_true", help="从 knowledge/seed 构建种子知识库")
    parser.add_argument("--db", default="", help="SQLite 路径（默认 knowledge/geoguard.db）")
    args = parser.parse_args(argv)
    config = Config()
    if args.db:
        config.db_path = args.db
    store = KnowledgeStore(config)
    pipe = IngestPipeline(store, config)
    if args.seed:
        counts = pipe.seed_from_dir()
        total = store.count(status=None)  # type: ignore[arg-type]
        active = store.count()
        print("种子知识库构建完成（幂等：重复运行不新增）：")
        for name, n in counts.items():
            print(f"  {name}: 新写入 {n} 条")
        print(f"库内条目：active={active}，总计={total}")
        print(f"嵌入后端：{getattr(store.embedder, 'name', '?')}"
              f"（dim={getattr(store.embedder, 'dim', '?')}）")
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
