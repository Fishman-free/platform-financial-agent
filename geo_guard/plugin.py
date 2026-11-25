# -*- coding: utf-8 -*-
"""形态一：Agent 插件（agent-integration 规格，docs/07 §2）。

GeoGuardPlugin 以 OpenAI function-calling 兼容的工具集形式挂载到任意宿主
Agent（Claude/Qwen/DeepSeek 等）。dispatch 为统一入口：参数校验失败返回
结构化错误（宿主可自纠错），永不抛未捕获异常。

MCP 演进：dispatch 语义与 MCP tool call 一一对应，包一层 stdio server 即成
MCP Server（本期不实现，接口不阻碍——docs/07 §5）。
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from .config import Config
from .explain import render_actions, render_report, render_trace
from .memory import MemoryStore
from .screening import GeoGuard

# 宿主系统提示词建议（随插件分发，docs/07 §2.3）
HOST_SYSTEM_PROMPT = """你挂载了 GEO 逆向甄别工具集（平台治理智能体·反 GEO 投毒模块）。规则：
1. 用户询问内容可信度 / AI 推荐合理性时，先调用 screen_content 或 trace_ai_answer，不要凭参数化知识直接下结论；
2. 判定为 high 时，必须完整转述证据链，并调用 suggest_actions 给出维权路径；
3. 返回中的 proactive_prompts 是建议的追问，结合对话语境酌情向用户发起（宁少勿滥）；
4. 禁止把 risk score 数值表述为法律结论；转述时使用"风险提示"措辞；
5. 用户表达认同/异议时调用 give_feedback（异议会驱动知识库复审，是系统进化的输入）；
6. 涉及用户历史与偏好时先 recall_memory；获得新的用户偏好事实时 remember。"""


def _tool(name: str, description: str, properties: Dict[str, Any],
          required: List[str]) -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties,
                           "required": required},
        },
    }


class GeoGuardPlugin:
    """8 工具插件面。所有工具返回 JSON 可序列化 dict。"""

    def __init__(self, user_id: str = "default", guard: Optional[GeoGuard] = None,
                 memory: Optional[MemoryStore] = None,
                 config: Optional[Config] = None):
        self.config = config or Config()
        self.guard = guard or GeoGuard(config=self.config)
        self.memory = memory or MemoryStore(self.config, user_id=user_id)
        self._tools: Dict[str, Callable[..., Dict[str, Any]]] = {
            "screen_content": self._screen_content,
            "trace_ai_answer": self._trace_ai_answer,
            "verify_claims": self._verify_claims,
            "check_influencer": self._check_influencer,
            "recall_memory": self._recall_memory,
            "remember": self._remember,
            "suggest_actions": self._suggest_actions,
            "give_feedback": self._give_feedback,
        }

    # ---------------- schema ----------------

    def tool_schemas(self) -> List[Dict[str, Any]]:
        return [
            _tool("screen_content",
                  "甄别一段内容（网页正文/带货话术/测评文案/评论列表）是否为 GEO 投毒或虚假宣传，"
                  "输出三档风险、证据链与建议追问。",
                  {"text": {"type": "string", "description": "待甄别文本"},
                   "source_meta": {"type": "object",
                                   "description": "可选来源元数据：domain/author/published_at/url/reuse_count"},
                   "comments": {"type": "array", "items": {"type": "object"},
                                "description": "可选评论列表：[{text, author?, created_at?}]"},
                   "subject": {"type": "string", "description": "可选：甄别对象主体名"}},
                  ["text"]),
            _tool("trace_ai_answer",
                  "逆向回溯一条 AI 生成的回答：判断其推荐是否被 GEO 内容网络牵引，"
                  "输出逐源风险、话术共振度与被过滤的反面证据。",
                  {"answer_text": {"type": "string", "description": "AI 回答原文"},
                   "cited_sources": {"type": "array", "items": {"type": "object"},
                                     "description": "引用源列表：[{title?, url?, snippet|text}]"}},
                  ["answer_text", "cited_sources"]),
            _tool("verify_claims",
                  "对文本做事实核验：抽取事实性声明（功效/资质/数据/对比），"
                  "与权威事实库比对，输出 supported/contradicted/unverified 三态。",
                  {"text": {"type": "string"}}, ["text"]),
            _tool("check_influencer",
                  "查询博主信用档案：历史推广频率、违规/翻车记录与信用评分。",
                  {"name": {"type": "string", "description": "博主名"},
                   "history_summary": {"type": "string", "description": "可选：历史内容摘要"}},
                  ["name"]),
            _tool("recall_memory",
                  "语义召回用户的长期记忆（历史甄别结论、偏好事件），带时间衰减排序。",
                  {"query": {"type": "string"},
                   "kinds": {"type": "array", "items": {"type": "string"},
                             "description": "可选过滤：screening/feedback/preference/event"}},
                  ["query"]),
            _tool("remember",
                  "把一条有长期价值的信息写入用户情景记忆（偏好声明、甄别结论、关键事件）。",
                  {"content": {"type": "string"},
                   "kind": {"type": "string", "enum": ["screening", "feedback", "preference", "event"]},
                   "subject": {"type": "string"},
                   "importance": {"type": "number"}},
                  ["content", "kind"]),
            _tool("suggest_actions",
                  "按风险等级与平台检索维权行动路径：申诉入口、所需材料、时限提示（来自平台规则库）。",
                  {"risk_level": {"type": "string", "enum": ["high", "medium", "low"]},
                   "platform": {"type": "string", "description": "如 generic_ecommerce/short_video/travel"},
                   "topic": {"type": "string"}},
                  ["risk_level"]),
            _tool("give_feedback",
                  "提交用户对判定结果的认同/异议。异议进入复核队列并驱动知识库复审。",
                  {"report_id": {"type": "string"},
                   "verdict": {"type": "string", "enum": ["agree", "dispute"]},
                   "note": {"type": "string"}},
                  ["report_id", "verdict"]),
        ]

    # ---------------- dispatch ----------------

    def dispatch(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """统一调用入口。结构化错误：{error: {type, message, missing_field?}}。"""
        fn = self._tools.get(tool_name)
        if fn is None:
            return {"error": {"type": "unknown_tool",
                              "message": f"未知工具 '{tool_name}'，可用：{sorted(self._tools)}"}}
        arguments = arguments if isinstance(arguments, dict) else {}
        # 必填参数校验（从 schema 读，保证单一真相源）
        schema = next((t for t in self.tool_schemas()
                       if t["function"]["name"] == tool_name), None)
        if schema:
            for req in schema["function"]["parameters"]["required"]:
                if req not in arguments or arguments[req] in (None, "", []):
                    return {"error": {"type": "invalid_params",
                                      "message": f"缺少必填参数 '{req}'",
                                      "missing_field": req}}
        try:
            return fn(**arguments)
        except (ValueError, KeyError, PermissionError) as exc:
            return {"error": {"type": "invalid_params", "message": str(exc)[:300]}}
        except Exception as exc:  # 兜底：永不让宿主因插件异常中断
            return {"error": {"type": "internal_error", "message": str(exc)[:300]}}

    # ---------------- 工具实现 ----------------

    def _screen_content(self, text: str, source_meta: Optional[Dict] = None,
                        comments: Optional[List[Dict]] = None,
                        subject: str = "", detail: str = "summary") -> Dict[str, Any]:
        report = self.guard.screen_content(text, source_meta, comments, subject)
        profile = self.memory.get_profile()
        rendered = render_report(report, detail=detail or profile.detail_level,
                                 config=self.config)
        # 甄别结论写入情景记忆（跨会话召回的基础）
        self.memory.remember_screening(
            report.subject, report.level.value, report.score,
            f"甄别「{report.subject or text[:30]}」：{report.level.value}，分数 {report.score:.2f}")
        return {"report": report.to_dict(), "rendered": rendered}

    def _trace_ai_answer(self, answer_text: str,
                         cited_sources: List[Dict], detail: str = "summary") -> Dict[str, Any]:
        report = self.guard.trace_ai_answer(answer_text, cited_sources)
        return {"report": report.to_dict(),
                "rendered": render_trace(report, detail=detail)}

    def _verify_claims(self, text: str) -> Dict[str, Any]:
        claims = self.guard.verify_claims(text)
        return {"claims": [c.to_dict() for c in claims],
                "degraded": "规则触发词抽取" if self.guard.llm.name == "null" else ""}

    def _check_influencer(self, name: str, history_summary: str = "") -> Dict[str, Any]:
        return self.guard.check_influencer(name, history_summary)

    def _recall_memory(self, query: str, kinds: Optional[List[str]] = None) -> Dict[str, Any]:
        hits = self.memory.recall(query, kinds=kinds)
        return {"memories": [{"memory": rec.to_dict(), "score": round(s, 4)}
                             for rec, s in hits]}

    def _remember(self, content: str, kind: str, subject: str = "",
                  importance: Optional[float] = None) -> Dict[str, Any]:
        mid = self.memory.add(kind, content, subject=subject, importance=importance)
        return {"memory_id": mid}

    def _suggest_actions(self, risk_level: str, platform: str = "",
                         topic: str = "") -> Dict[str, Any]:
        actions = self.guard.suggest_actions(risk_level, platform, topic)
        return {"actions": actions, "rendered": render_actions(actions)}

    def _give_feedback(self, report_id: str, verdict: str, note: str = "") -> Dict[str, Any]:
        report = self.guard._screen_cache.get(report_id)
        refs: List[str] = []
        if report:
            refs = [r for e in report.evidence for r in e.kb_refs]
        # 反馈写入 ingest 管线的 feedback 表（与复核队列同一闭环）
        from .ingest import IngestPipeline
        pipe = IngestPipeline(self.guard.store, self.config, self.guard)
        return pipe.record_feedback(report_id, verdict, note, refs)
