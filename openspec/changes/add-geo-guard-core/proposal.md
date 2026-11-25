# Proposal

## Why

央视 3·15 曝光的 GEO"投毒"灰黑产通过批量铺设软文/榜单/测评内容网络，污染联网 AI 的
证据供给，操控用户决策入口；母项目"平台治理智能体"需要一个可运行、可解释、可在用户侧
低成本部署的逆向甄别内核来支撑反 GEO 投毒模块与虚假宣传鉴别工作流。目前设计已完成
（docs/01~07 头脑风暴与方案文档），需要将其落地为代码骨架与种子知识库，交付
"Agent 插件 + Personal Agent"双形态原型用于演示、灰度与后续云侧图谱接入。

## What Changes

- 新建 `geo_guard` Python 包：五库知识库（SQLite+numpy 持久化）、混合检索管线
  （BM25 字符 bigram + 可插拔嵌入 + RRF 融合 + 两段式重排）；
- 新建六通道特征提取（话术匹配/结构特征/注入扫描/来源可信度/事实核验/协同证据槽位）
  与三层判定融合（规则兜底→加权融合→可校准分档），输出三档风险与证据链；
- 新建双对象检测：单篇内容甄别 `screen_content` 与 AI 回答逆向回溯 `trace_ai_answer`
  （来源集中度/话术共振度/反证缺失度）；
- 新建三层记忆（工作/情景/语义）与动态用户画像：时间衰减 w(t)=e^(-λt)、偏好冲突检测、
  画像只调制呈现不改判定分数；
- 新建对话状态机与行动链 DAG 执行引擎：主动质询触发器（规则+LLM 双轨）、中断恢复、
  高风险动态重规划、分级披露；
- 新建数据筛查与标注管线 `ingest.py`：simhash 去重（保留复用计数）、弱标签、
  复核队列、反馈闭环；
- 新建双形态落地：`plugin.py`（OpenAI function-calling 兼容工具集，8 个工具）与
  `agent.py`（CLI Personal Agent，LLM 后端可插拔，NullBackend 规则降级保证离线演示）；
- 新建种子知识库（knowledge/seed/）：9 类话术模式、7 项结构特征、权威事实摘编、
  合成演示案例、平台规则模板——全部防御视角抽象，source 强制溯源；
- 新建测试套件与演示脚本，落实 docs/02 §8.3 的五项验收基线。

## Capabilities

### New Capabilities

- `knowledge-retrieval`: 五库知识库存储、条目治理（source/version/status 强制）、
  混合检索（BM25+稠密+RRF）、重排与嵌入后端可插拔、"70% 阈值"的相似度语义。
- `geo-screening`: GEO 逆向甄别判定能力——六通道特征提取、三层判定融合、三档处置、
  双对象检测（内容甄别 + AI 回答回溯）、虚假宣传识别三路径（权威核验/话术相似度/
  评论筛选）、可解释证据链输出。
- `memory-profile`: 三层记忆结构、时间衰减召回、记忆整合、动态画像状态向量、
  偏好冲突检测与确认式修正、隐私主权（本地存储/可删除/可导出）。
- `dialogue-guidance`: 对话状态机、主动质询触发器、行动链 DAG 编排与动态重规划、
  中断恢复、信息密度控制与分级披露、行动引导（维权路径检索）。
- `data-ingest`: 数据筛查与标注机制——标准化、去重与复用计数、弱标签、复核队列、
  标注 schema、反馈闭环、合成对抗集隔离。
- `agent-integration`: 双形态集成契约——插件工具集 schema 与 dispatch 协议、
  Personal Agent 主循环与 LLM 后端抽象、无 LLM 全链路降级、宿主系统提示词分发。

### Modified Capabilities

（无——本变更全部为新增能力，`openspec/specs/` 当前为空。）

## Impact

- 新增代码：`geo_guard/`（约 15 个模块）、`knowledge/seed/`（5 个 JSON 种子库）、
  `tests/`、`examples/`；
- 依赖：Python ≥ 3.10，硬依赖仅 `numpy`；可选 LLM API（DashScope/OpenAI 兼容，
  仅从环境变量读取密钥，零硬编码）；
- 数据：本地 SQLite（knowledge/*.db 与 runtime/ 已入 .gitignore，不提交运行产物）；
- 对接：为云侧图谱团队预留 F6 协同证据槽位接口；为浏览器插件形态预留
  screen_content 轻通道移植路径；团队 5000+ URL 语料经 ingest 管线接入后
  阈值从锚点法切换校准器。
