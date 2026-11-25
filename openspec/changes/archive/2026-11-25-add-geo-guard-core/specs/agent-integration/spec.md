# Spec Delta

## Purpose

定义 GEO 甄别内核的两种落地形态契约：以工具集形式挂载到任意
function-calling 宿主 Agent 的插件形态，与自带多轮循环的 Personal Agent
形态；保证两者共享同一内核、无 LLM 时全链路可降级运行。

## ADDED Requirements

### Requirement: 插件工具集 schema
插件形态 SHALL 输出 OpenAI function-calling 兼容的工具 schema 清单，
至少包含：screen_content、trace_ai_answer、verify_claims、check_influencer、
recall_memory、remember、suggest_actions、give_feedback；每个工具 MUST
声明参数类型与返回结构，schema 可被宿主 Agent 直接注册。

#### Scenario: schema 合法性
- **WHEN** 宿主 Agent 请求工具清单
- **THEN** 返回的每个工具均含 name、description、parameters(JSON Schema)，
  且 8 个工具全部在列

### Requirement: 统一 dispatch 与结构化错误
插件 SHALL 提供统一调用入口 dispatch(tool_name, arguments)；参数校验失败
或工具不存在时 MUST 返回结构化错误（含错误类型与修复提示），MUST NOT
抛出未捕获异常导致宿主中断。

#### Scenario: 非法参数自纠错
- **WHEN** 宿主以缺少必填参数的方式调用 screen_content
- **THEN** 返回含 missing_field 与提示信息的结构化错误，宿主可据此重试

### Requirement: Personal Agent 主循环
Personal Agent SHALL 提供命令行交互循环：意图路由（甄别/回溯/推荐/记忆/
维权/越界六类）、调用对应内核能力、分级披露渲染、主动质询、会话结束
触发记忆整合；越界请求 MUST 得到边界声明并被引导回治理能力。

#### Scenario: 意图路由到甄别
- **WHEN** 用户输入"帮我看看这段文案可不可信"并附文本
- **THEN** 路由至内容甄别流程并输出带证据链的报告

#### Scenario: 越界请求引导
- **WHEN** 用户请求与平台治理无关的任务（如写代码）
- **THEN** Agent 声明能力边界并引导回甄别/推荐/维权等支持场景

### Requirement: LLM 后端抽象与配置纪律
系统 SHALL 抽象统一的 LLM 后端接口，支持 DashScope/OpenAI 兼容后端与
空后端（规则降级）；后端选择顺序 MUST 为：显式配置 → 环境变量探测 →
空后端；API 密钥 MUST 仅从环境读取，代码与仓库 MUST NOT 含任何硬编码密钥。

#### Scenario: 无密钥自动降级
- **WHEN** 环境无任何 API 密钥且未显式配置后端
- **THEN** 系统使用空后端启动，意图路由/质询/报告改由规则与模板完成，
  输出头部标注降级模式

#### Scenario: 密钥不入库
- **WHEN** 审查仓库全部文件与提交历史
- **THEN** 不存在任何明文 API 密钥；.env 类文件被版本控制忽略

### Requirement: 无 LLM 全链路可用
在空后端下，系统 MUST 保持全部核心能力可用（甄别、回溯、核验、记忆、
画像、质询、行动引导），仅语言生成质量降级；降级路径的行为 MUST 可预期
（规则驱动、模板化输出）。

#### Scenario: 离线完整演示
- **WHEN** 在断网且无密钥环境运行演示脚本
- **THEN** 演示完成"甄别→追问→报告→维权引导→记忆召回"全流程且退出码为 0

### Requirement: 宿主使用契约分发
插件 SHALL 随附宿主系统提示词建议（何时调用工具、HIGH 必须转述证据链、
禁止把风险分表述为法律结论、反馈时机），供接入方直接采用或改写。

#### Scenario: 提示词随插件可得
- **WHEN** 接入方读取插件文档或模块常量
- **THEN** 可获得完整的宿主系统提示词建议文本
