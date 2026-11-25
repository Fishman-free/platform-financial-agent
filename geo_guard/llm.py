# -*- coding: utf-8 -*-
"""LLM 后端抽象与触点白名单（design.md D6）。

全部 LLM 触点仅 5 处：查询改写 / claim 抽取 / 重排 / 意图路由 / 生成润色。
每个触点都是「LLM 优先、规则兜底、同一输出 schema」的结构：
NullBackend 下走兜底路径，行为可预期（agent-integration 规格）。

密钥纪律：只从环境变量读取（DASHSCOPE_API_KEY / OPENAI_API_KEY /
OPENAI_BASE_URL），代码与仓库零硬编码；HTTP 用标准库 urllib，不引入依赖。
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol

_DEFAULT_MODELS = {
    "dashscope": "qwen-plus",
    "openai": "gpt-4o-mini",
}
_DEFAULT_BASE_URLS = {
    "dashscope": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "openai": "https://api.openai.com/v1",
}


class LLMBackend(Protocol):
    name: str

    def complete(self, messages: List[Dict[str, str]], temperature: float = 0.2,
                 max_tokens: int = 1024) -> Optional[str]:
        """返回补全文本；不可用时返回 None（调用方走兜底）。"""
        ...


class NullBackend:
    """规则降级后端：complete 永远返回 None，触点函数自动走兜底路径。"""

    name = "null"

    def complete(self, messages, temperature=0.2, max_tokens=1024) -> Optional[str]:
        return None


@dataclass
class OpenAICompatBackend:
    """OpenAI /v1/chat/completions 兼容后端（DashScope 兼容模式同理）。"""

    name: str
    api_key: str
    base_url: str
    model: str
    timeout: float = 20.0

    def complete(self, messages, temperature=0.2, max_tokens=1024) -> Optional[str]:
        body = json.dumps({
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url.rstrip('/')}/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"]
        except Exception:
            # 网络/限流/超时一律静默降级到规则路径；不向调用方抛异常，
            # 也不记录 prompt 内容（脱敏纪律）。
            return None


def get_backend(llm_setting: str = "auto", model: str = "") -> LLMBackend:
    """后端选择顺序：显式配置 → 环境变量探测 → NullBackend（规格约定）。"""
    setting = (llm_setting or "auto").lower()
    if setting in ("off", "null", "none"):
        return NullBackend()
    if setting in ("dashscope", "openai"):
        key = os.environ.get("DASHSCOPE_API_KEY" if setting == "dashscope" else "OPENAI_API_KEY", "")
        if not key:
            return NullBackend()  # 显式指定但无密钥 → 降级而非报错
        base = os.environ.get(
            "OPENAI_BASE_URL" if setting == "openai" else "DASHSCOPE_BASE_URL",
            _DEFAULT_BASE_URLS[setting],
        )
        return OpenAICompatBackend(name=setting, api_key=key, base_url=base,
                                   model=model or _DEFAULT_MODELS[setting])
    # auto：探测环境变量
    if os.environ.get("DASHSCOPE_API_KEY"):
        return get_backend("dashscope", model)
    if os.environ.get("OPENAI_API_KEY"):
        return get_backend("openai", model)
    return NullBackend()


# --------------------------------------------------------------------------
# 触点 1：查询改写（I-07）—— 兜底：原始查询 + 规则模板扩展
# --------------------------------------------------------------------------

def touch_query_rewrite(backend: LLMBackend, query: str,
                        entities: Optional[Dict[str, str]] = None) -> List[str]:
    queries = [query]
    ents = entities or {}
    # 规则模板兜底：实体拼接扩展（品类+风险词 / 品牌+品类）
    if ents.get("category"):
        queries.append(f"{ents['category']} 虚假宣传 风险 话术")
    if ents.get("brand"):
        queries.append(f"{ents['brand']} {ents.get('category', '')} 投诉 违规".strip())
    out = backend.complete([
        {"role": "system", "content": "你是检索查询改写器。把用户查询扩写为2-3个互补的检索查询，"
                                      "输出 JSON 数组（字符串列表），不要其他内容。"},
        {"role": "user", "content": query},
    ], max_tokens=256)
    if out:
        try:
            m = re.search(r"\[.*\]", out, re.S)
            if m:
                extra = [q.strip() for q in json.loads(m.group()) if isinstance(q, str) and q.strip()]
                for q in extra:
                    if q not in queries:
                        queries.append(q)
        except (json.JSONDecodeError, ValueError):
            pass
    return queries[:4]


# --------------------------------------------------------------------------
# 触点 2：claim 抽取（D-06）—— 兜底：触发词句子筛选
# --------------------------------------------------------------------------

# 事实性声明触发词：数字/功效/资质/对比四类（docs/02 §4）
CLAIM_TRIGGER_WORDS = [
    "最", "第一", "唯一", "顶级", "国家级", "特效", "治疗", "治愈", "根治",
    "有效率达", "成功率", "收益率", "年化", "保证", "承诺", "无效退款",
    "天见效", "快速见效", "抗癌", "降血压", "降血糖", "修复受损", "药监局批准",
    "荣获", "指定产品", "独家", "百分百", "100%", "零风险", "稳赚",
]


def touch_extract_claims(backend: LLMBackend, text: str) -> List[Dict[str, Any]]:
    """返回 [{text, span}]；span 为字符偏移 [start, end)。"""
    out = backend.complete([
        {"role": "system", "content": "从给定文本中抽取事实性声明（仅限功效/资质/数据/对比四类），"
                                      '输出 JSON 数组：[{"text": "声明原文"}]。没有则输出 []。'},
        {"role": "user", "content": text[:4000]},
    ], max_tokens=800)
    if out:
        try:
            m = re.search(r"\[.*\]", out, re.S)
            if m:
                items = json.loads(m.group())
                claims = []
                for it in items:
                    if isinstance(it, dict) and it.get("text"):
                        t = str(it["text"])[:200]
                        pos = text.find(t[:40])
                        claims.append({"text": t, "span": (max(pos, 0), max(pos, 0) + len(t))})
                if claims:
                    return claims
        except (json.JSONDecodeError, ValueError):
            pass
    # 规则兜底：含触发词的句子
    claims = []
    for sent_match in re.finditer(r"[^。！？；\n]*[。！？；]?", text):
        sent = sent_match.group().strip()
        if len(sent) >= 6 and any(w in sent for w in CLAIM_TRIGGER_WORDS):
            claims.append({"text": sent[:200], "span": (sent_match.start(), sent_match.end())})
    return claims[:20]


# --------------------------------------------------------------------------
# 触点 3：重排（I-03）—— 兜底：特征重排（在 retrieval.py 中实现）
# --------------------------------------------------------------------------

def touch_rerank(backend: LLMBackend, query: str,
                 candidates: List[Dict[str, Any]]) -> Optional[List[float]]:
    """LLM 对 (查询, 条目) 批量打 0-10 相关性分；不可用返回 None 走规则重排。"""
    if backend.name == "null" or not candidates:
        return None
    listing = "\n".join(
        f"[{i}] {c.get('title', '')}: {str(c.get('content', ''))[:150]}"
        for i, c in enumerate(candidates)
    )
    out = backend.complete([
        {"role": "system", "content": "你是检索重排器。对每个候选条目与查询的相关性打 0-10 分，"
                                      "输出 JSON 数组（与候选顺序等长的数字列表），不要其他内容。"},
        {"role": "user", "content": f"查询：{query}\n候选：\n{listing}"},
    ], max_tokens=400)
    if not out:
        return None
    try:
        m = re.search(r"\[.*\]", out, re.S)
        if not m:
            return None
        scores = [float(x) for x in json.loads(m.group())]
        if len(scores) == len(candidates):
            return scores
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    return None


# --------------------------------------------------------------------------
# 触点 4：意图路由（agent.py 使用）—— 兜底：关键词规则
# --------------------------------------------------------------------------

# 规则路由（按优先级排序的问句式正则——避免"退款/推荐"等词在长文中的子串误触发）
INTENT_RULES = [
    # 维权引导：疑问词 + 维权动作词的组合问句
    ("action", re.compile(
        r"(怎么|如何|哪里|去哪|流程|渠道)[^。？]{0,12}(维权|投诉|申诉|举报|退款|赔偿)"
        r"|(维权|投诉|申诉|举报)[^。？]{0,8}(怎么|如何|流程|渠道)")),
    # AI 回答回溯
    ("trace", re.compile(
        r"(为什么(会|要)?推荐|[Aa][Ii]\s*推荐|这个回答|引用来源|被操纵|被投毒|投毒了吗|回溯)")),
    # 记忆查询：过去时态 + 疑问
    ("memory", re.compile(
        r"(之前|上次|上回|记得|查过|问过|历史)[^。]{0,12}(吗|？|\?)|我(之前|上次|查过)")),
    # 推荐请求：明确的请求句式（而非文中偶然出现"推荐"二字）
    ("recommend", re.compile(
        r"(求推荐|有没有[^。]{0,12}推荐|推荐(一个|一款|几个|些)|买(什么|哪个|哪款)|怎么选|平替|同款)")),
    # 内容甄别
    ("screen", re.compile(
        r"(可信吗|可不可信|靠谱|真的假的|真伪|甄别|鉴别|是不是骗|虚假|看看|能买吗|能不能买|怎么样)")),
]


def touch_route_intent(backend: LLMBackend, user_text: str) -> Dict[str, Any]:
    """返回 {intent, entities:{brand?, category?}, content?}。"""
    out = backend.complete([
        {"role": "system", "content":
            "你是平台治理助手的意图路由器。把用户输入分类到以下意图之一："
            "screen(内容甄别)/trace(AI回答回溯)/recommend(推荐)/memory(记忆查询)/"
            "action(维权引导)/other(越界或闲聊)。"
            '输出 JSON：{"intent": "...", "entities": {"brand": "", "category": ""}, '
            '"content": "需要甄别的原文（如有）"}'},
        {"role": "user", "content": user_text[:2000]},
    ], max_tokens=400)
    if out:
        try:
            m = re.search(r"\{.*\}", out, re.S)
            if m:
                parsed = json.loads(m.group())
                if parsed.get("intent") in ("screen", "trace", "recommend", "memory", "action", "other"):
                    return parsed
        except (json.JSONDecodeError, ValueError):
            pass
    # 规则兜底：按优先级匹配问句式正则
    for intent, pat in INTENT_RULES:
        if pat.search(user_text):
            return {"intent": intent, "entities": {}, "content": user_text}
    return {"intent": "other", "entities": {}, "content": ""}


# --------------------------------------------------------------------------
# 触点 5：生成润色 —— 兜底：模板输出（explain.py 的渲染本身即兜底形态）
# --------------------------------------------------------------------------

def touch_polish(backend: LLMBackend, draft: str, style_hint: str = "") -> str:
    """润色必须保留证据编号与引用（docs/06 §4：证据编号不可删改）。"""
    out = backend.complete([
        {"role": "system", "content": "润色以下风险提示文案，保持全部证据编号（如[证据1]）与数字不变，"
                                      "语气客观中立，禁止使用'肯定合法/绝对安全/构成违法'等法律结论措辞。"
                                      + (f"风格提示：{style_hint}" if style_hint else "")},
        {"role": "user", "content": draft[:3000]},
    ], max_tokens=1200)
    if not out:
        return draft
    # 输出校验：证据编号丢失则回退草稿
    refs_in = set(re.findall(r"\[证据\d+\]", draft))
    refs_out = set(re.findall(r"\[证据\d+\]", out))
    if refs_in and not refs_in.issubset(refs_out):
        return draft
    return out
