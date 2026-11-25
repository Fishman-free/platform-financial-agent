# -*- coding: utf-8 -*-
"""形态二：Personal Agent（agent-integration 规格，docs/07 §3）。

CLI 多轮智能体：意图路由（LLM/关键词双轨）→ 行动链 DAG → 分级披露渲染 →
主动质询 → 会话结束记忆整合。NullBackend 下全链路规则驱动（行为可预期），
输出头部标注降级模式——诚实披露，不假装满血。
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import Config
from .dialogue import ActionDAG, DialogueManager
from .explain import render_actions, render_report, render_trace
from .llm import LLMBackend, NullBackend, get_backend, touch_polish, touch_route_intent
from .memory import MemoryStore
from .plugin import GeoGuardPlugin
from .schema import DialogueState, Phase, RiskLevel

_BANNER = """╔══════════════════════════════════════════════════════════╗
  GEO 逆向甄别 · Personal Agent（平台治理智能体 · 用户侧）
  能力：内容甄别 / AI回答回溯 / 事实核验 / 博主信用 / 维权引导
  命令：/screen /trace /profile /forget /export /feedback /help /quit
╚══════════════════════════════════════════════════════════╝"""

_HELP = """可用命令：
  /screen <文本|@文件路径>   甄别一段内容（虚假宣传 / GEO 投毒）
  /trace                    粘贴 AI 回答与引用源，做逆向回溯
  /profile                  查看我的画像与记忆概览
  /forget <memory_id>       删除一条记忆（/forget all 清空全部，需确认）
  /export                   导出全部记忆为 JSON（数据主权）
  /feedback <report_id> <agree|dispute> [备注]   对判定反馈
  /help                     本帮助
  /quit                     退出（自动做记忆整合）
也可以直接用自然语言提问，例如"帮我看看这段文案可信吗：……"。"""

_BOUNDARY = ("我是平台治理支持智能体，专注于内容甄别、AI 回答回溯、事实核验与维权引导。"
             "这个请求超出了我的能力边界——试试把需要鉴别的文案、链接或 AI 回答发给我？")


class PersonalAgent:
    def __init__(self, config: Optional[Config] = None, user_id: str = "default",
                 llm: Optional[LLMBackend] = None):
        self.config = config or Config()
        self.llm = llm or get_backend(self.config.llm, self.config.llm_model)
        self.plugin = GeoGuardPlugin(user_id=user_id, config=self.config)
        self.guard = self.plugin.guard
        self.memory = self.plugin.memory
        self.dialogue = DialogueManager(self.config, llm=self.llm)
        self.state: DialogueState = self.dialogue.new_state()
        self._session_events: List[Dict[str, Any]] = []
        self._last_report_id: str = ""
        self._trace_buffer: Dict[str, Any] = {}

    # ---------------- 主入口 ----------------

    def handle(self, user_text: str) -> str:
        """处理一轮用户输入，返回回复文本（状态机全程驱动）。"""
        user_text = (user_text or "").strip()
        if not user_text:
            return ""
        self.memory.push_turn("user", user_text)
        self.dialogue.transition(self.state, "user_input", {"text": user_text})

        # 斜杠命令优先；回溯模式中的输入直接进回溯流程（不走意图路由）
        if user_text.startswith("/"):
            reply = self._command(user_text)
        elif self._trace_buffer:
            reply = self._act_trace(user_text)
        else:
            reply = self._route_and_act(user_text)

        self.memory.push_turn("assistant", reply)
        self.dialogue.transition(self.state, "responded")
        return reply

    # ---------------- 意图路由与执行 ----------------

    def _route_and_act(self, text: str) -> str:
        routed = touch_route_intent(self.llm, text)
        intent = routed.get("intent", "other")
        content = routed.get("content") or text
        entities = routed.get("entities") or {}

        # 画像冲突检测：先质询确认，不静默改画像（memory-profile 规格）
        conflicts = self.memory.detect_conflicts(text)
        if conflicts and self.state.phase != Phase.CLARIFYING:
            picked = self.dialogue.trigger.pick(text, [c["question"] for c in conflicts])
            if picked:
                self.dialogue.transition(self.state, "intent_ambiguous",
                                         {"question": picked[0]})
                self._pending_conflict = conflicts[0]
                return picked[0]
        if getattr(self, "_pending_conflict", None) and self.state.clarify_attempts:
            c = self._pending_conflict
            self._pending_conflict = None
            if re.search(r"(是|对|想|要|好)", text[:6]):
                new_val = 0.2 if c["signal"] == "high_budget" else 0.8
                self.memory.confirm_conflict(c["key"], new_val, note=text[:50])

        if intent == "screen":
            self.dialogue.transition(self.state, "intent_resolved", {"intent": "screen"})
            return self._act_screen(content, entities)
        if intent == "trace":
            self.dialogue.transition(self.state, "intent_resolved", {"intent": "trace"})
            return self._act_trace(content)
        if intent == "recommend":
            self.dialogue.transition(self.state, "intent_resolved", {"intent": "recommend"})
            return self._act_recommend(content)
        if intent == "memory":
            self.dialogue.transition(self.state, "intent_resolved", {"intent": "memory"})
            return self._act_recall(text)
        if intent == "action":
            self.dialogue.transition(self.state, "intent_resolved", {"intent": "action"})
            actions = self.guard.suggest_actions("medium", topic=text)
            return self._polish(render_actions(actions))
        # 长文本即使意图不明也先当甄别对象处理（用户粘贴文案的常见形态）
        if len(text) >= 60:
            self.dialogue.transition(self.state, "intent_resolved", {"intent": "screen"})
            return self._act_screen(text, entities)
        self.dialogue.transition(self.state, "intent_ambiguous", {})
        return self.state.pending_question or _BOUNDARY

    def _run_screen(self, text: str, subject: str = ""):
        """共享甄别执行：产出报告 + 写情景记忆 + 记会话事件（两条工作流复用）。"""
        report = self.guard.screen_content(text, subject=subject)
        self._last_report_id = report.report_id
        self.memory.remember_screening(
            report.subject, report.level.value, report.score,
            f"甄别「{report.subject or text[:30]}」：{report.level.value}，"
            f"分数 {report.score:.2f}")
        self._session_events.append({
            "kind": "screening", "subject": report.subject,
            "risk_level": report.level.value,
            "content": f"甄别 {report.subject or '内容'}：{report.level.value}"})
        return report

    def _act_screen(self, text: str, entities: Dict[str, Any]) -> str:
        # 先召回历史甄别记忆（跨会话连续性）
        history_note = ""
        subject_guess = entities.get("brand") or self.guard._guess_subject(text)
        if subject_guess:
            hits = self.memory.recall(f"{subject_guess} 甄别", top_k=1, kinds=["screening"])
            if hits:
                rec, score = hits[0]
                history_note = (f"📎 记忆召回：{rec.content}\n"
                                f"（如无新证据，本次结论与历史一致；有新证据则以下方为准）\n\n")
        # 行动链 DAG：甄别 → 展示 → 维权引导（HIGH 动态重规划在引擎内）
        dag = ActionDAG.plan_screening(text)
        self.dialogue.transition(self.state, "plan_ready", {"dag": dag})
        results: Dict[str, Any] = {}

        def h_screen(params):
            report = self._run_screen(params["text"], subject_guess)
            results["report"] = report
            return {"level": report.level.value, "report_id": report.report_id}

        def h_show(params):
            report = results.get("report")
            if report is None:
                return {"shown": False}
            profile = self.memory.get_profile()
            results["rendered"] = render_report(report, detail=profile.detail_level,
                                                config=self.config)
            return {"shown": True}

        def h_actions(params):
            report = results.get("report")
            if report is None or report.level == RiskLevel.LOW:
                return {"actions": []}
            actions = self.guard.suggest_actions(report.level.value, topic=report.subject)
            results["actions"] = actions
            return {"actions": len(actions)}

        self.dialogue.dag_engine.execute(
            self.state, {"screen": h_screen, "show": h_show, "actions": h_actions})
        self.dialogue.transition(self.state, "executed")

        rendered = results.get("rendered") or "（甄别节点未产出结果）"
        if results.get("actions"):
            rendered += "\n\n" + render_actions(results["actions"])
        rendered = history_note + self._polish(rendered)
        # 中高风险：更新风险规避画像的隐式信号 + 提示反馈入口
        report = results.get("report")
        if report and report.level in (RiskLevel.HIGH, RiskLevel.MEDIUM):
            rendered += f"\n\n（反馈：/feedback {report.report_id} agree|dispute —— 异议会驱动知识库复审）"
        return rendered

    def _act_trace(self, text: str) -> str:
        """回溯模式：先收 AI 回答，再逐行收引用源，done 生成回溯报告。"""
        if self._trace_buffer.get("await_answer") or "answer" not in self._trace_buffer:
            self._trace_buffer.pop("await_answer", None)
            self._trace_buffer["answer"] = text
            return ("已收到 AI 回答。请逐行粘贴它引用的来源"
                    "（每行：标题 | URL | 关键句），全部粘贴后发送 done 生成回溯报告。")
        if text.strip().lower() in ("done", "/trace done"):
            answer = self._trace_buffer.pop("answer", "")
            sources = self._trace_buffer.pop("sources", [])
            self._trace_buffer = {}
            report = self.guard.trace_ai_answer(answer, sources)
            self._last_report_id = report.report_id
            return self._polish(render_trace(report, detail="standard"))
        self._trace_buffer.setdefault("sources", []).append(self._parse_source_line(text))
        return (f"已记录来源 {len(self._trace_buffer['sources'])} 条。"
                f"继续粘贴，或发送 done 生成回溯报告。")

    def _act_recommend(self, query: str) -> str:
        """推荐 = 先甄别再推荐（利好用户筛选，docs/03 §4 防误伤）。"""
        dag = ActionDAG.plan_recommendation(query)
        self.dialogue.transition(self.state, "plan_ready", {"dag": dag})
        out: Dict[str, Any] = {}

        def h_screen(params):
            report = self._run_screen(params["text"])
            out["report"] = report
            return {"level": report.level.value}

        def h_recommend(params):
            report = out.get("report")
            if report and report.level == RiskLevel.HIGH:
                return {"skipped": "高风险，推荐已中断"}
            # 演示级推荐：从案例库取高信用对照 + 画像调制话术
            profile = self.memory.get_profile()
            out["recommendation"] = (
                "基于'推荐前先风险过滤'原则：该主体未通过甄别前不作推荐。"
                f"（画像提示：{profile.warning_style}）"
                "建议先用 /screen 甄别候选内容，或让我检索同类平替的信用记录。")
            return {"done": True}

        def h_show(params):
            report = out.get("report")
            parts = []
            if report:
                parts.append(render_report(report, detail="summary", config=self.config))
            if out.get("recommendation"):
                parts.append("\n【推荐环节】" + out["recommendation"])
            out["rendered"] = "\n".join(parts)
            return {"shown": True}

        self.dialogue.dag_engine.execute(
            self.state, {"screen": h_screen, "recommend": h_recommend, "show": h_show})
        self.dialogue.transition(self.state, "executed")
        return out.get("rendered") or _BOUNDARY

    def _act_recall(self, query: str) -> str:
        hits = self.memory.recall(query, top_k=5)
        if not hits:
            return "没有找到相关记忆。多跟我聊几次甄别与偏好，我会记住它们（全部本地存储）。"
        lines = ["相关记忆（时间衰减加权排序）："]
        for rec, score in hits:
            lines.append(f"  · [{rec.kind}] {rec.content[:80]}（相关度 {score:.2f}，id={rec.memory_id}）")
        return "\n".join(lines)

    # ---------------- 命令 ----------------

    def _command(self, text: str) -> str:
        parts = text.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""
        if cmd == "/help":
            return _HELP
        if cmd == "/screen":
            if arg.startswith("@"):
                p = Path(arg[1:])
                if not p.exists():
                    return f"文件不存在：{p}"
                arg = p.read_text(encoding="utf-8", errors="replace")
            if not arg:
                return "用法：/screen <文本|@文件路径>"
            return self._act_screen(arg, {})
        if cmd == "/trace":
            if not arg:
                self._trace_buffer = {"await_answer": True}
                return "进入回溯模式：请粘贴 AI 回答原文（下一条消息）。"
            if arg.strip().lower() == "done":
                return self._act_trace("done")
            return self._act_trace(arg)
        if cmd == "/profile":
            snap = self.memory.get_profile()
            n_mem = self.memory.conn.execute(
                "SELECT COUNT(*) c FROM memories WHERE user_id=?",
                (self.memory.user_id,)).fetchone()["c"]
            lines = ["【我的画像】（只影响呈现与引导，不影响判定分数——审计公正性）"]
            name_map = {"risk_aversion": "风险规避", "price_sensitivity": "价格敏感",
                        "brand_orientation": "品牌导向", "evidence_appetite": "证据披露偏好"}
            for k, v in snap.state.items():
                lines.append(f"  {name_map.get(k, k)}: {v:.2f}（置信 {snap.confidence.get(k, 0):.2f}）")
            if snap.entity_trust:
                lines.append("实体信任：" + ", ".join(f"{e}={t:.2f}"
                                                      for e, t in list(snap.entity_trust.items())[:8]))
            lines.append(f"情景记忆条数：{n_mem}（本地存储，可 /forget 删除、/export 导出）")
            return "\n".join(lines)
        if cmd == "/forget":
            if arg.lower() == "all":
                n = self.memory.clear_all(confirm=True)
                return f"已清空全部记忆与画像（{n} 条）。此操作不可恢复。"
            ok = self.memory.delete(arg)
            return f"已删除记忆 {arg}。" if ok else f"未找到记忆 {arg}。"
        if cmd == "/export":
            return self.memory.export_json()
        if cmd == "/feedback":
            m = re.match(r"(\S+)\s+(agree|dispute)\s*(.*)", arg)
            if not m:
                return "用法：/feedback <report_id> agree|dispute [备注]"
            rid, verdict, note = m.group(1), m.group(2), m.group(3)
            result = self.plugin.dispatch("give_feedback",
                                          {"report_id": rid, "verdict": verdict, "note": note})
            if result.get("error"):
                return f"反馈失败：{result['error']['message']}"
            flagged = result.get("flagged_entries") or []
            msg = "反馈已记录，感谢！异议样本已进入复核队列。"
            if flagged:
                msg += f"\n异议累计达标，以下知识条目已转人工复审（退出线上判定）：{flagged}"
            return msg
        return f"未知命令 {cmd}。输入 /help 查看用法。"

    # ---------------- 工具 ----------------

    def _parse_source_line(self, line: str) -> Dict[str, Any]:
        parts = [p.strip() for p in line.split("|")]
        src: Dict[str, Any] = {}
        if len(parts) >= 1:
            src["title"] = parts[0]
        if len(parts) >= 2:
            src["url"] = parts[1]
        if len(parts) >= 3:
            src["snippet"] = parts[2]
        else:
            src["text"] = parts[0]
        return src

    def _polish(self, draft: str) -> str:
        """触点 5：生成润色（证据编号不可删改——llm.py 内校验回退）。"""
        profile = self.memory.get_profile()
        return touch_polish(self.llm, draft, style_hint=profile.warning_style)

    def close(self) -> None:
        """会话结束：记忆整合（docs/05 §3）。"""
        if self._session_events:
            self.memory.integrate_session(self._session_events)
            self._session_events = []
        self.dialogue.close()
        self.memory.close()
        self.guard.store.close()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="geo-guard",
                                     description="GEO 逆向甄别 Personal Agent")
    parser.add_argument("--user", default="default", help="用户命名空间（本地数据隔离）")
    parser.add_argument("--exec", dest="exec_inputs", nargs="*", default=None,
                        help="非交互模式：依次执行输入后退出（演示/测试用）")
    parser.add_argument("--stdin", action="store_true",
                        help="非交互模式：从 stdin 逐行读取输入")
    args = parser.parse_args(argv)

    agent = PersonalAgent(user_id=args.user)
    degraded = "（LLM 降级：规则驱动模式）" if isinstance(agent.llm, NullBackend) else ""
    print(_BANNER)
    if degraded:
        print(f"⚠️ 当前为规则驱动模式{degraded}——功能完整，语言生成走模板路径。")

    def run_line(line: str) -> bool:
        if not line.strip():
            return True
        if line.strip() in ("/quit", "/exit"):
            return False
        reply = agent.handle(line)
        if reply:
            print(f"\n🤖 {reply}\n")
        return True

    try:
        if args.exec_inputs is not None:
            for line in args.exec_inputs:
                if not run_line(line):
                    break
        elif args.stdin:
            for line in sys.stdin:
                if not run_line(line):
                    break
        else:
            print("\n输入 /help 查看命令，/quit 退出。\n")
            while True:
                try:
                    line = input("🧑 > ")
                except EOFError:
                    break
                if not run_line(line):
                    break
    finally:
        agent.close()
        print("记忆已整合，再见。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
