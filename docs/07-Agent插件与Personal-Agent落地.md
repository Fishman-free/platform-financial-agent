# 07 · Agent 插件与 Personal Agent 落地方案

> 决策 D-04：双形态共享同一内核（`geo_guard` 包），插件验证"可挂载性"，
> Personal Agent 验证"完整体验"。本文档定义两种形态的接口契约。

## 1. 分层结构

```text
┌─────────────────────────────────────────────────────┐
│  形态层                                               │
│  plugin.py: GeoGuardPlugin     agent.py: PersonalAgent│
│  (function-calling 工具集)      (CLI 多轮循环+状态机)     │
├─────────────────────────────────────────────────────┤
│  编排层  dialogue.py(状态机/DAG)  memory.py(记忆/画像)   │
├─────────────────────────────────────────────────────┤
│  判定层  detection/{features,patterns,claims,fusion}   │
├─────────────────────────────────────────────────────┤
│  检索层  retrieval.py(混合检索/重排)                     │
├─────────────────────────────────────────────────────┤
│  知识层  kb/{store,bm25,embeddings}  knowledge/seed/*   │
└─────────────────────────────────────────────────────┘
```

依赖方向严格向下；形态层之间互不依赖。内核零 LLM 强依赖：所有 LLM 调用点
（查询改写、claim 抽取、重排、生成润色、意图理解）都有规则驱动的降级路径，
保证 `pip install numpy` 后离线可演示。

## 2. 形态一：Agent 插件（GeoGuardPlugin）

面向任意支持 function calling 的宿主 Agent（Claude/Qwen/DeepSeek 等）。

### 2.1 工具清单（tools schema）

| 工具 | 参数 | 返回 | 对应用户场景 |
|---|---|---|---|
| `screen_content` | text, source_meta{domain,author,published_at}?, comments[]? | RiskReport(JSON): level, score, confidence, evidence_chain[], proactive_prompts[] | "这段文案/这个博主可信吗" |
| `trace_ai_answer` | answer_text, cited_sources[{title,url?,snippet}] | TraceReport: suspicion_level, per_source_risk[], resonance(话术共振), missing_negatives[] | "AI 为什么推荐它？被操纵了吗" |
| `verify_claims` | text | claims[{text, verdict, ref}] | "成分/功效宣传是真的吗" |
| `check_influencer` | name, history_summary? | credit_score, records[] | "这个博主的历史口碑" |
| `recall_memory` | query, user_id? | memories[]（衰减加权） | "我之前查过这个吗" |
| `remember` | content, kind, importance? | memory_id | 宿主 Agent 主动存记忆 |
| `suggest_actions` | risk_level, platform? | actions[]（申诉路径/材料/时限, 来自 KB5） | "我该怎么维权" |
| `give_feedback` | report_id, verdict(agree/dispute), note? | queued: true | 反馈闭环入口 |

### 2.2 挂载协议

```python
from geo_guard.plugin import GeoGuardPlugin

plugin = GeoGuardPlugin(user_id="demo")          # 本地 KB, 无 LLM 也能跑
tools = plugin.tool_schemas()                     # → OpenAI 风格 tools JSON
result = plugin.dispatch("screen_content", {"text": "……"})
```

- `tool_schemas()` 输出 OpenAI function-calling 兼容格式（Claude/Gemini 可机械转换）；
- `dispatch(name, args)` 统一入口，参数校验失败返回结构化错误（宿主可自纠错）；
- **MCP 演进路径**：dispatch 语义与 MCP tool call 一一对应，后续包一层
  stdio server 即成为 MCP Server（本期不实现，接口不阻碍）。

### 2.3 宿主系统提示词建议（随插件分发）

```text
你挂载了 GEO 逆向甄别工具集。规则：
1. 用户询问内容可信度/AI推荐合理性时，先调用 screen_content / trace_ai_answer，
   不要凭参数化知识直接下结论；
2. 判定为 HIGH 时，必须完整转述证据链并调用 suggest_actions；
3. 返回中的 proactive_prompts 是建议的追问，酌情向用户发起；
4. 禁止把 risk_score 数值当作法律结论表述；转述时使用"风险提示"措辞；
5. give_feedback 在用户表达认同/异议时调用。
```

## 3. 形态二：Personal Agent（CLI）

### 3.1 主循环

```text
启动 → 加载 KB/记忆/画像 → REPL:
  用户输入
   → 意图路由(LLM 或规则):
       甄别类 → 状态机 INTENT_PARSE → DAG(screen→…) → 报告
       回溯类 → trace_ai_answer 流程
       推荐类 → 画像召回 + 候选检索 + 风险过滤(docs/03 §4)
       记忆类 → recall/remember
       维权类 → suggest_actions
       闲聊/越界 → 边界声明 + 引导回治理能力
   → 分级披露渲染(结论+top3证据, 可展开)
   → 主动质询(如有触发)
  会话结束 → 记忆整合 + 状态快照
```

### 3.2 LLM 后端抽象

```python
class LLMBackend(Protocol):
    def complete(self, messages, **kw) -> str: ...

# 实现: DashScopeBackend(OpenAI兼容协议) / OpenAIBackend / NullBackend(规则降级)
```

- 配置顺序：环境变量 `GEO_GUARD_LLM=off|dashscope|openai` → `.env`（不入库）→ 自动探测
  `DASHSCOPE_API_KEY`/`OPENAI_API_KEY` → 均无则 NullBackend；
- **NullBackend 下的体验保底**：意图路由用关键词规则、质询用模板、报告用结构化渲染——
  功能全在，语言生硬度上升；输出头部标注 `[规则驱动模式]`，诚实披露降级状态；
- 密钥纪律：任何密钥只从环境读取，代码与仓库零硬编码（.gitignore 已覆盖 .env）。

### 3.3 CLI 命令

```text
python -m geo_guard.agent                # 进入对话
  /screen <文本或文件路径>                 # 显式甄别
  /trace                                  # 粘贴 AI 回答+来源做回溯
  /profile                                # 查看我的画像
  /forget <id> | /forget-all              # 记忆删除(二次确认)
  /export                                 # 导出记忆 JSON
  /feedback <report_id> agree|dispute     # 反馈
  /quit                                   # 退出(触发记忆整合)
```

## 4. 验收对照（tests 覆盖）

| 验收项 | 测试 |
|---|---|
| 插件工具 schema 合法、dispatch 全工具可用 | test_plugin.py |
| 无 LLM 全链路可跑（NullBackend） | 全部测试默认无 API 环境运行 |
| 高风险注入样例判 HIGH 且证据链非空 | test_detection.py |
| 回溯模式检出"三源同模板"共振 | test_detection.py |
| 记忆衰减与画像冲突质询 | test_memory.py / test_dialogue.py |
| 状态机中断恢复 | test_dialogue.py |

## 5. 演进路线

1. **MCP Server 化**：dispatch → stdio MCP，接入 Claude Code 等宿主（接口已对齐）；
2. **浏览器插件端**：screen_content 轻通道（L1+F1/F3）移植 content-script，
   云侧 F6 协同信号接入（母项目端云两级架构的本仓库落位）；
3. **团队数据接入**：5000+ URL 语料走 ingest 管线，阈值切换校准器（docs/04 §5）。
