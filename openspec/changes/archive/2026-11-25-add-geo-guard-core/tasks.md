# Tasks

## 1. 基础设施与数据模型

- [x] 1.1 创建 `geo_guard/config.py`：全部阈值/权重/路径的 dataclass 配置源，支持环境变量覆写（`GEO_GUARD_LLM`、`GEO_GUARD_DB` 等）；验证：`python -c "from geo_guard.config import Config; print(Config().t_high)"` 输出 0.72
- [x] 1.2 创建 `geo_guard/schema.py`：KBEntry/Evidence/RiskReport/TraceReport/Claim/MemoryRecord/DialogueState 等 dataclass 与 JSON 序列化；验证：schema 往返序列化测试通过（tests/test_schema.py）
- [x] 1.3 创建 `geo_guard/llm.py`：LLMBackend 协议 + NullBackend + OpenAI 兼容后端（DashScope/OpenAI，密钥仅环境变量），触点白名单注释；验证：无密钥环境下 `get_backend()` 返回 NullBackend（tests/test_llm.py）

## 2. 知识层（knowledge-retrieval）

- [x] 2.1 编写 `knowledge/seed/*.json` 五个种子库（KB1≈30 条九类话术、KB2=7 特征、KB3≈15 事实、KB4≈6 合成案例、KB5≈8 规则模板），每条 source/checked_at 非空、防御抽象红线自查；验证：`python -m geo_guard.ingest --seed` 构建成功且五库条目数达标、无 source 缺失
- [x] 2.2 实现 `geo_guard/kb/bm25.py`：中文 bigram+英文词+数字串 token 化的 Okapi BM25 倒排索引；验证：触发例句查询命中对应条目 top-5（tests/test_retrieval.py）
- [x] 2.3 实现 `geo_guard/kb/embeddings.py`：HashEmbedding（确定性 384 维）+ ApiEmbedding（OpenAI 兼容 + sha256 磁盘缓存 + 失败回退 Hash）；验证：同文本两次嵌入向量完全一致、维度正确（tests/test_kb.py）
- [x] 2.4 实现 `geo_guard/kb/store.py`：SQLite 持久化 + 启动加载 numpy 内存索引 + 幂等 upsert + pending_review 隔离 + 版本化更新；验证：重复 seed 构建条目数不变；pending_review 条目检索不可见；source 为空拒绝入库（tests/test_kb.py）
- [x] 2.5 实现 `geo_guard/retrieval.py`：双通道召回 + RRF(k=60) + 元数据过滤 + 两段式重排（LLM/规则双路径）+ 查询改写（LLM/模板兜底）+ 按后端阈值偏移；验证：验收基线 #4——KB1 触发例句 Recall@5 ≥ 0.8（Hash 模式）（tests/test_retrieval.py）

## 3. 判定层（geo-screening）

- [x] 3.1 实现 `geo_guard/detection/patterns.py`：分段话术匹配（F1），相似度分级语义（≥0.70 强证据/0.50~0.70 弱/<0.50 无），片段定位与 KB 引用；验证：合成软文样例命中 ranking_template 与 authority_fabrication（tests/test_detection.py）
- [x] 3.2 实现 `geo_guard/detection/features.py`：F2 七项结构特征检测器（term_density/qa_self_structure/listicle_no_method/citation_bait_density/brand_seed_density/template_alignment/cross_validation_web）+ F3 注入扫描规则表 + F4 来源可信度（白名单/域龄/复用计数）；验证：注入样例触发 F3、白名单文本 F4 高分（tests/test_detection.py）
- [x] 3.3 实现 `geo_guard/detection/claims.py`：claim 抽取（LLM/触发词规则降级）+ KB3 三态核验（supported/contradicted/unverified）+ 法条映射输出；验证："保健食品治疗高血压"判 contradicted；无 LLM 时降级抽取仍工作（tests/test_detection.py）
- [x] 3.4 实现 `geo_guard/detection/comments.py`：评论有效性筛选——simhash 近重复、时间聚簇、好评命中话术反向计疑、有效负评占比与雷点词；验证：水军样例（同分钟近重复好评）被标记且不计入有效好评（tests/test_detection.py）
- [x] 3.5 实现 `geo_guard/detection/fusion.py`：L1 规则兜底 → L2 六通道加权（缺失重归一化）→ L3 分档 + 证据单薄降档 + Calibrator 接口（AnchorCalibrator）；验证：验收基线 #1/#2/#3——注入样例 HIGH(rule)、软文 ≥MEDIUM、白名单 LOW（tests/test_detection.py）
- [x] 3.6 实现 `geo_guard/screening.py` 门面：`screen_content`（对象 A）与 `trace_ai_answer`（对象 B：来源集中度/话术共振度/反证缺失度）组装 RiskReport/TraceReport + proactive_prompts；验证：验收基线 #5——三源同模板输入报告话术共振（tests/test_screening.py）
- [x] 3.7 实现 `geo_guard/explain.py`：证据链卡片渲染三档披露（summary/standard/full）、超额折叠、降级模式标注、"风险提示"措辞校验；验证：7 证据报告默认只呈现 top-3 + 折叠提示（tests/test_explain.py）

## 4. 记忆与对话层（memory-profile / dialogue-guidance）

- [x] 4.1 实现 `geo_guard/memory.py`：三层记忆（环形工作记忆/情景向量召回/画像表）+ 时间衰减 e^(-λt) 分型半衰期 + importance 保护 + EMA 画像更新 + 实体信任表 + 导出/删除/清空；验证：跨"会话"（重建实例）召回甄别历史；高 importance 记忆衰减后仍可召回（tests/test_memory.py）
- [x] 4.2 实现偏好冲突检测：请求语义 vs 高置信画像矛盾 → 生成确认质询而非静默修改；验证：价格敏感画像 + "不差钱"请求触发确认质询（tests/test_memory.py）
- [x] 4.3 实现 `geo_guard/dialogue.py`：状态机（8 阶段 + 快照持久化 + 中断恢复）、主动质询触发器（规则 + LLM 双轨、保守触发）、行动链 DAG（ask/show/call/confirm、依赖调度、失败 skipped、HIGH 动态重规划）；验证：执行中杀进程重启后从快照续行；甄别 HIGH 时推荐节点不执行（tests/test_dialogue.py）
- [x] 4.4 实现 `geo_guard/ingest.py` 管线：标准化 Evidence Bundle → simhash 去重（reuse_count/reuse_domains 保留）→ 弱标签（白名单不免检）→ 复核队列（不确定度排序）→ 反馈闭环（异议累计触发 pending_review）→ 合成对抗集隔离加载检查；验证：三域同文 reuse_count=3 且域列表正确；对抗集目录不被 seed 加载（tests/test_ingest.py）

## 5. 形态层（agent-integration）

- [x] 5.1 实现 `geo_guard/plugin.py`：8 工具 OpenAI function-calling schema + dispatch 统一入口 + 结构化错误（missing_field 等）+ 宿主系统提示词常量；验证：schema 含 8 工具且各有 parameters；缺参调用返回结构化错误不抛异常（tests/test_plugin.py）
- [x] 5.2 实现 `geo_guard/agent.py` + `__main__.py`：CLI REPL、六类意图路由（LLM/关键词规则降级）、`/screen /trace /profile /forget /export /feedback /quit` 命令、会话结束记忆整合、越界请求边界声明、降级模式头部标注；验证：NullBackend 下 `python -m geo_guard.agent` 非交互注入输入可完成甄别并退出码 0（tests/test_agent.py）
- [x] 5.3 编写 `examples/demo_personal_agent.py`（甄别→追问→报告→维权引导→记忆召回全流程）与 `examples/demo_plugin_host.py`（模拟宿主 Agent 挂载工具集）；验证：断网无密钥环境下两个 demo 均退出码 0

## 6. 验收与文档收口

- [x] 6.1 全量测试通过：`python -m pytest tests/ -q` 零失败；验收基线五条（docs/02 §8.3）全部有对应绿色断言
- [x] 6.2 `openspec validate add-geo-guard-core` 通过；规格每个 Scenario 在测试中有映射（在 tasks 完成后的验证清单中逐条核对）
- [x] 6.3 README 快速开始命令逐条实测可复现（ingest --seed / demo / pytest）

## Workflow follow-up

- 全部任务完成后执行 `/openspec-archive-change`：归档本变更至 `openspec/changes/archive/` 并把六个能力 delta 同步进 `openspec/specs/`（归档后 specs 成为真相源）
- 团队评审会（遗留风险 R2）：答辩前安排真实用户/法务/云侧图谱三视角人工评审，评审结论作为新 change 输入
