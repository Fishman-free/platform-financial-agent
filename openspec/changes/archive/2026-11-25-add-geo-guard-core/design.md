# Design

## Context

动机见 proposal.md - Why；产品级完整设计见 `docs/02-GEO逆向甄别RAG设计.md`
（本文档只沉淀**代码架构层**的决策，不复述产品方案）。约束现状：

- 全新仓库，无既有代码；团队语料（5000+ URL、400 组标注）暂不在本仓库内；
- 演示环境可能无 GPU、无外网、无任何 API 密钥（答辩/课堂场景）；
- 开发者环境为 Windows + Python 3.14，需跨平台可跑（ teammates 用 macOS/Linux）；
- 母项目云侧图谱/GNN 由其他成员后续开发，本期内 F6 协同通道只有接口没有数据。

## Goals / Non-Goals

**Goals:**
- 一个 `pip install numpy` 后即可离线全链路运行的内核（KB→检索→判定→对话→双形态）；
- 所有 LLM 触点收敛到单一 `LLMBackend` 协议，降级路径全覆盖；
- 数据结构（Evidence Bundle / KB Entry / RiskReport / TraceReport）即接口契约，
  云侧与插件端未来按同一契约接入；
- 测试即验收：docs/02 §8.3 五条验收基线全部固化为 pytest 断言。

**Non-Goals:**
- 不实现云侧 GNN/图谱（只留 F6 槽位）；不做浏览器插件前端；
- 不接真实平台爬虫（评论/博主数据以离线列表输入，接口预留）；
- 不做模型训练（Transformer 分类器/校准器拟合属于团队数据接入后的阶段）；
- 不做多用户服务端（user_id 只是本地数据命名空间）。

## Decisions

### D1. 包结构与依赖方向

```text
geo_guard/
  config.py        # 全部阈值/权重/路径的单一配置源（dataclass + 环境变量覆写）
  schema.py        # 纯数据模型（dataclass）：KBEntry/Evidence/RiskReport/TraceReport/...
  kb/              # store.py(SQLite+索引) bm25.py embeddings.py seed.py(加载knowledge/seed)
  retrieval.py     # HybridRetriever：双通道召回+RRF+重排（LLM/规则双路径）
  detection/       # features.py patterns.py claims.py comments.py fusion.py
  ingest.py        # 标准化/simhash去重/弱标签/复核队列/CLI(--seed)
  memory.py        # MemoryStore + UserProfile（SQLite，同库不同表）
  dialogue.py      # DialogueStateMachine + ActionDAG + 质询触发器
  explain.py       # 证据链卡片渲染（summary/standard/full 三档）
  llm.py           # LLMBackend 协议 + DashScope/OpenAI/Null 实现 + 意图路由
  plugin.py        # GeoGuardPlugin（8 工具 schema + dispatch）
  agent.py         # PersonalAgent CLI REPL（__main__ 入口）
```

依赖只允许自上而下：`agent/plugin → dialogue/explain → detection → retrieval → kb → schema/config`。
`memory` 被 dialogue 与 plugin 使用，但 memory 不反向依赖它们（画像不改判定的
结构性保证——D-08 由依赖方向强制，而非仅靠约定）。

**为什么不单文件/不更细分**：15 个模块是"一人可维护、职责单一"的平衡点；
再细分（如每通道一个文件）会产生大量 <100 行的碎片模块。

### D2. 存储：单 SQLite 多表 + 进程内 numpy 索引

- `knowledge/geoguard.db`：kb_entries、memories、profile_state、entity_trust、
  feedback、review_queue、session_snapshots 全部表；
- 向量不入 SQLite BLOB 逐条读，而是启动时全量加载为 numpy 矩阵常驻内存
  （千级种子条目 <10MB；5 万条目 ×384 维 ≈77MB，可接受），SQLite 只做持久化真相源；
- 写路径：先写 SQLite 再增量更新内存索引，崩溃后从 SQLite 重建。

**备选与否决**：JSON 文件存储（并发写易损坏，查询弱）；外部向量库 Qdrant/Milvus
（部署重，违背离线演示目标，留作 5 万+ 规模的迁移路径）。

### D3. 嵌入：确定性哈希嵌入为默认，API 嵌入为增强

`HashEmbedding`：字符 bigram+trigram → murmur 风格哈希（纯 Python 实现，
不用 hashlib 以控制速度）→ 384 维有符号计数 → 子线性 tf → L2 归一化。
同一文本永远同一向量（可测试、可复现、零依赖）。
`ApiEmbedding`：OpenAI 兼容 `/embeddings` 端点（DashScope text-embedding-v3 等），
带 30 天磁盘缓存（向量按内容 sha256 键控），失败自动回退 Hash。
阈值偏移（-0.05）在 `config.py` 按后端名配置——两后端的分数分布不同，
共用阈值会产生系统性偏差（头脑风暴对抗评审 #2 的决议）。

### D4. BM25 自实现，token 策略中文特化

`kb/bm25.py` 约 120 行纯 Python：倒排索引 + Okapi BM25。token =
中文字符 bigram（单字仅当 bigram 无命中时回退）∪ 英文小写词 ∪ 数字串整体
（"3·15"、"70%"、"GB2760"不拆碎——法规/标准号检索是 KB3 的真实查询形态）。
不用 rank_bm25/jieba：前者接口不稳定，后者引入词典依赖且对短文本 bigram 无优势。

### D5. 判定融合：显式线性加权 + 规则前置，拒绝黑盒

`fusion.py` 实现三层：L1 规则表（注入模式/黑名单/白名单异常）→ L2 六通道
加权和（权重/阈值全在 config，缺失通道重归一化）→ L3 分档 + 证据单薄降档。
每条 Evidence 携带 `channel, score, weight, span, kb_refs`，报告分数可
逐项分解复算——可解释性由数据结构保证（P3 原则），SHAP 之类的后置解释
在显式线性模型上是多余的。`Calibrator` 留 Protocol 接口，本期实现
`AnchorCalibrator`（读锚点样本 JSON 定阈）供团队 400 组标注接入。

### D6. LLM 触点白名单制

全部 LLM 调用点（仅 5 处）：查询改写、claim 抽取、重排、意图路由、生成润色。
每处 MUST 有规则兜底实现且兜底输出与 LLM 输出同 schema；`NullBackend` 下
系统在 import 时即可确定走兜底路径（无运行时试探性调用）。密钥只经
`os.environ` 读取；`llm.py` 对请求日志脱敏（不落盘 prompt 全文）。

### D7. 状态机与 DAG：数据驱动 + 快照表

对话状态与 DAG 都是纯数据（dataclass + JSON 序列化入 session_snapshots 表），
执行引擎是无状态函数 `step(state, event) -> state'`——中断恢复=读快照反序列化，
测试=构造 state 断言迁移。不引入 langgraph 等框架：本状态的迁移图 <20 条边，
框架的学习/依赖成本高于手写；且母项目 PPT 的"快照保存·断点续行"语义需要
对存储的完全控制。

### D8. 种子知识库构造原则

`knowledge/seed/*.json` 五个文件对应五库；话术模式条目全部来自公开渠道
（广告法禁止性表述、央视/第一财经报道中已披露的话术**类型**描述），
条目内容写"识别要点与判定语义"，不写"如何构造"（红线 I-29）；
案例库条目 `synthetic: true`；每条 source/checked_at 非空（入库治理自证）。
规模：KB1≈30 条、KB2=7 条、KB3≈15 条、KB4≈6 条、KB5≈8 条——足以支撑
验收基线与演示，不追求覆盖度（覆盖度是团队数据管线的任务）。

### D9. 测试策略：验收基线即回归测试

`tests/` 五个文件与六能力对应；所有测试默认 `GEO_GUARD_LLM=off` 环境运行
（NullBackend 全覆盖是 CI 可复现的前提）；docs/02 §8.3 五条验收基线 +
规格中每个 Scenario 至少一条断言；合成样例（软文/注入文/白名单文/三源共振）
固化为 `tests/fixtures/`，与对抗集隔离原则一致——fixtures 不进 KB。

## Risks / Trade-offs

- [HashEmbedding 语义弱，同义改写漏检] → BM25 字面通道兜基本盘；验收基线 #4
  量化召回下限；文档明示降级状态；API 嵌入即插即换（D3）。
- [线性融合对通道间交互建模不足（如"话术高×来源白名单"应特殊处理）] →
  L1 规则层承接已知交互模式（白名单异常降档）；未知交互靠灰度反馈发现，
  Calibrator 接口留了升级位（D5）。
- [单 SQLite 并发写锁] → 本期单用户单进程，WAL 模式足够；多用户服务端在
  Non-Goals，真发生时迁移 PostgreSQL（表结构不变）。
- [Python 3.14 太新，依赖兼容风险] → 硬依赖仅 numpy（3.14 已有 wheel）；
  测试矩阵在 3.10+ 声明，CI 可后补。
- [规则兜底的注入模式可被同义变形绕过（已知局限）] → 定位为高精度低召回层，
  召回由统计通道补；对抗评测集（data-ingest 隔离目录）持续度量衰减。
- [种子 KB 覆盖度低导致演示外场景判定质量差] → 输出永远附"证据单薄/建议
  交叉验证"置信声明（规格已定），诚实降级优于假装全能。

## Migration Plan

全新仓库无迁移。部署=clone + pip install numpy + `python -m geo_guard.ingest --seed`。
回滚=删除 knowledge/*.db 重建（种子构建幂等，规格已定）。
团队语料接入路径：ingest 管线（docs/04 §5）→ AnchorCalibrator 换阈值 →
（可选）API 嵌入重建索引（entry_id 不变，向量列重算）。

## Open Questions

- 重排的 LLM 打分批大小与缓存 TTL 的具体值（性能调优项，不影响接口与规格）；
- `reuse_count` 进入 F4 还是 F6 的权重归属（云侧接入时与图谱团队对齐，
  本期默认计入 F4 来源信号）；
- 演示脚本的叙事剧本（答辩前与团队定，examples/ 先给功能完整版）。
