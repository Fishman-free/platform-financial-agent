# -*- coding: utf-8 -*-
"""评论有效性筛选（docs/03 路径三，docs/04 §2② simhash 复用）。

水军信号：文本近重复（simhash 海明距离）、发布时间聚簇、好评命中话术模式。
有效负评：负面情感词 + 具体性（含商品属性词）。
输出：负面占比、可疑水军比例、雷点词列表 → F4/融合层与证据链。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..config import Config
from ..kb.bm25 import tokenize
from ..schema import Evidence

# 负面情感词（通用 + 消费场景）
NEGATIVE_WORDS = [
    "假货", "劣质", "坑", "骗", "翻车", "失望", "退货", "投诉", "维权", "差评",
    "难用", "不好用", "过敏", "拉肚子", "头晕", "无效", "没用", "智商税",
    "虚假", "夸大", "货不对板", "色差", "破损", "漏发", "客服差", "不退款",
    "被扣费", "自动续费", "套路", "割韭菜",
]
# 具体性属性词：负评含商品属性 → 更可信（细节难以批量编造）
ATTRIBUTE_WORDS = [
    "成分", "材质", "尺寸", "重量", "电池", "续航", "屏幕", "口感", "味道",
    "气味", "包装", "物流", "客服", "价格", "用量", "周期", "做工", "缝隙",
    "发热", "噪音", "掉色", "起球", "异味", "副作用",
]
POSITIVE_WORDS = ["好用", "推荐", "已下单", "回购", "正品", "满意", "好评", "不错", "值得"]


def simhash64(text: str) -> int:
    """64bit simhash（token 加权符号投影）。"""
    toks = tokenize(text)
    if not toks:
        return 0
    vec = [0] * 64
    for tok in toks:
        h = 0
        for b in tok.encode("utf-8"):
            h = (h * 131 + b) & 0xFFFFFFFFFFFFFFFF
        for i in range(64):
            vec[i] += 1 if (h >> i) & 1 else -1
    out = 0
    for i in range(64):
        if vec[i] > 0:
            out |= 1 << i
    return out


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


@dataclass
class CommentItem:
    """输入契约：离线评论列表（爬虫接入前的统一形态）。"""
    text: str
    author: str = ""
    created_at: str = ""   # ISO 或 'YYYY-MM-DD HH:MM' 等可解析前缀
    likes: int = 0


@dataclass
class CommentAnalysis:
    total: int = 0
    valid_total: int = 0
    negative_ratio: float = 0.0
    suspicious_ratio: float = 0.0
    lei_points: List[str] = field(default_factory=list)   # 雷点词（高频负面+属性组合）
    duplicate_clusters: int = 0
    time_clusters: int = 0
    evidences: List[Evidence] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if k != "evidences"}
        d["evidences"] = [e.to_dict() for e in self.evidences]
        return d


class CommentAnalyzer:
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()

    def analyze(self, comments: List[CommentItem]) -> CommentAnalysis:
        result = CommentAnalysis(total=len(comments))
        if not comments:
            return result

        # ① 近重复聚簇（simhash 海明距离 <= 阈值）
        hashes = [(c, simhash64(c.text)) for c in comments]
        dup_flag = [False] * len(comments)
        clusters = 0
        for i in range(len(hashes)):
            if dup_flag[i]:
                continue
            cluster_size = 1
            for j in range(i + 1, len(hashes)):
                if not dup_flag[j] and hamming(hashes[i][1], hashes[j][1]) <= self.config.simhash_distance:
                    dup_flag[j] = True
                    cluster_size += 1
            if cluster_size > 1:
                dup_flag[i] = True
                clusters += 1
        result.duplicate_clusters = clusters

        # ② 时间聚簇（同分钟前缀 >= 3 条）
        minute_count: Dict[str, int] = {}
        for c in comments:
            key = (c.created_at or "")[:16]
            if key:
                minute_count[key] = minute_count.get(key, 0) + 1
        burst_minutes = {k for k, v in minute_count.items() if v >= 3}
        time_flag = [bool((c.created_at or "")[:16] in burst_minutes) for c in comments]
        result.time_clusters = len(burst_minutes)

        # ③ 逐条分类：可疑水军 / 有效负评 / 有效好评 / 中性
        suspicious = 0
        neg_valid = 0
        pos_valid = 0
        lei_counter: Dict[str, int] = {}
        for idx, c in enumerate(comments):
            is_suspicious = dup_flag[idx] or time_flag[idx] or self._positive_bait(c.text)
            if is_suspicious:
                suspicious += 1
                continue
            neg_hits = [w for w in NEGATIVE_WORDS if w in c.text]
            attr_hits = [w for w in ATTRIBUTE_WORDS if w in c.text]
            if neg_hits:
                neg_valid += 1
                # 雷点 = 负面词 + 属性词组合（具体性加权）
                for w in neg_hits:
                    lei_counter[w] = lei_counter.get(w, 0) + (2 if attr_hits else 1)
            elif any(w in c.text for w in POSITIVE_WORDS):
                pos_valid += 1
        result.suspicious_ratio = suspicious / len(comments)
        result.valid_total = neg_valid + pos_valid
        result.negative_ratio = (neg_valid / result.valid_total) if result.valid_total else 0.0
        result.lei_points = [w for w, _ in sorted(lei_counter.items(),
                                                  key=lambda kv: kv[1], reverse=True)[:8]]

        # ④ 证据化
        if result.suspicious_ratio >= 0.3 and len(comments) >= 5:
            result.evidences.append(Evidence(
                channel="comment", label="评论区水军信号",
                score=min(1.0, result.suspicious_ratio),
                severity=0.65,
                detail={"suspicious_ratio": round(result.suspicious_ratio, 3),
                        "duplicate_clusters": clusters, "time_clusters": result.time_clusters},
                kb_refs=["kb3-fact-0010"],  # 虚构交易与编造评价的法规锚点
            ))
        if result.negative_ratio >= 0.4 and result.valid_total >= 5:
            result.evidences.append(Evidence(
                channel="comment", label="有效负评占比偏高",
                score=min(1.0, result.negative_ratio),
                severity=0.6,
                detail={"negative_ratio": round(result.negative_ratio, 3),
                        "lei_points": result.lei_points},
            ))
        return result

    @staticmethod
    def _positive_bait(text: str) -> bool:
        """好评命中话术模式（模板化好评反向计疑，docs/03 路径三）。"""
        if not any(w in text for w in POSITIVE_WORDS):
            return False
        # 模板化特征：短评 + 纯正面 + 无属性细节
        has_attr = any(w in text for w in ATTRIBUTE_WORDS)
        return (not has_attr) and len(text) <= 20 and re.search(r"(已下单|冲|回购|好评)\W*$", text) is not None
