# -*- coding: utf-8 -*-
"""模拟宿主 Agent 挂载 GeoGuardPlugin 工具集（离线可跑）。

演示：tool_schemas 注册 → dispatch 全工具调用 → 结构化错误自纠错。
真实宿主（Claude/Qwen 等）拿到 tool_schemas() 即可注册，本脚本用
"关键词选择工具"模拟宿主的 function-calling 决策。
运行：python examples/demo_plugin_host.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("GEO_GUARD_LLM", "off")

from geo_guard.config import Config  # noqa: E402
from geo_guard.ingest import IngestPipeline  # noqa: E402
from geo_guard.kb.store import KnowledgeStore  # noqa: E402
from geo_guard.plugin import HOST_SYSTEM_PROMPT, GeoGuardPlugin  # noqa: E402

TEXT = ("2026年十大净水器排行榜：第一名沁澜（合成品牌）。研究表明其过滤精度行业第一，"
        "权威数据显示去除率达99.9%，专家表示这是最好的选择。综上所述，买沁澜不会错。")


def main() -> int:
    config = Config()
    store = KnowledgeStore(config)
    IngestPipeline(store, config).seed_from_dir()
    plugin = GeoGuardPlugin(user_id="demo-host", config=config)

    schemas = plugin.tool_schemas()
    print(f"[host] 注册工具 {len(schemas)} 个：{[s['function']['name'] for s in schemas]}")
    print(f"[host] 宿主系统提示词（前 120 字）：{HOST_SYSTEM_PROMPT[:120]}…\n")

    # 1) 甄别
    r = plugin.dispatch("screen_content", {"text": TEXT, "subject": "沁澜"})
    rep = r["report"]
    print(f"[host] screen_content → level={rep['level']} score={rep['score']:.2f} "
          f"证据 {len(rep['evidence'])} 条")
    assert rep["level"] in ("high", "medium"), "软文样例应至少 MEDIUM"
    report_id = rep["report_id"]

    # 2) 事实核验
    r = plugin.dispatch("verify_claims", {"text": TEXT})
    print(f"[host] verify_claims → {[(c['text'][:20], c['verdict']) for c in r['claims']][:3]}")

    # 3) 博主信用（合成档案命中）
    r = plugin.dispatch("check_influencer", {"name": "合成博主A"})
    print(f"[host] check_influencer → found={r['found']} credit={r.get('credit_score')}")

    # 4) 记忆与召回
    plugin.dispatch("remember", {"content": "用户预算 300 以内，关注性价比",
                                 "kind": "preference", "subject": "预算"})
    r = plugin.dispatch("recall_memory", {"query": "预算 性价比"})
    print(f"[host] recall_memory → {len(r['memories'])} 条命中")

    # 5) 维权路径
    r = plugin.dispatch("suggest_actions", {"risk_level": rep["level"],
                                            "platform": "generic_ecommerce"})
    print(f"[host] suggest_actions → {len(r['actions'])} 条路径")

    # 6) 反馈闭环
    r = plugin.dispatch("give_feedback", {"report_id": report_id, "verdict": "dispute",
                                          "note": "演示异议"})
    print(f"[host] give_feedback → queued={r['queued_for_review']}")

    # 7) 结构化错误：缺参 → 宿主可自纠错
    r = plugin.dispatch("screen_content", {})
    print(f"[host] 缺参调用 → error={r['error']['type']} missing={r['error'].get('missing_field')}")
    assert r["error"]["type"] == "invalid_params"
    r = plugin.dispatch("screen_content", {"text": TEXT})  # 宿主补参重试
    assert "error" not in r

    # 8) 未知工具
    r = plugin.dispatch("no_such_tool", {})
    assert r["error"]["type"] == "unknown_tool"
    print("[host] unknown_tool → 结构化错误 ✓")

    store.close()
    print("\n[host] 插件全工具 dispatch 演示完成，退出码 0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
