# platform-financial-agent

> 面向用户的平台治理支持智能体系统 —— **产品与技术方案**（马珠淇负责部分）
>
> 中国人民大学 2026 年大学生创业训练计划《面向用户的平台治理支持智能体系统设计与实践》

## 这个仓库是什么

母项目"平台治理智能体"针对央视 3·15 曝光的 **GEO（生成式引擎优化）"投毒"** 问题：
灰黑产批量生产软文/榜单/测评，污染 RAG 与联网搜索的证据供给，操控 AI 回答误导用户决策。

本仓库承载其中 **产品与技术方案** 职责域，并将核心能力落地为可运行代码：

| 职责 | 方案文档 | 代码落地 |
|---|---|---|
| GEO 逆向甄别（RAG） | [docs/02](docs/02-GEO逆向甄别RAG设计.md) | `geo_guard/kb` `geo_guard/retrieval.py` `geo_guard/detection` |
| 虚假宣传识别 | [docs/03](docs/03-虚假宣传识别方案.md) | `geo_guard/detection/claims.py` `geo_guard/detection/patterns.py` |
| 数据筛查与标注机制 | [docs/04](docs/04-数据筛查与标注机制.md) | `geo_guard/ingest.py` |
| 长期记忆 + 用户画像 | [docs/05](docs/05-长期记忆与用户画像.md) | `geo_guard/memory.py` |
| 多轮交互 + 行动引导 | [docs/06](docs/06-多轮交互与行动引导.md) | `geo_guard/dialogue.py` |
| Agent 落地形态 | [docs/07](docs/07-Agent插件与Personal-Agent落地.md) | `geo_guard/plugin.py` `geo_guard/agent.py` |

## 两种落地形态

1. **Agent 插件**（`geo_guard.plugin.GeoGuardPlugin`）：以 function-calling 工具集形式，
   挂载到任意支持工具调用的 LLM Agent（Claude / Qwen / DeepSeek 等），提供
   `screen_content`、`verify_claims`、`recall_memory`、`suggest_actions` 等工具。
2. **Personal Agent**（`geo_guard.agent.PersonalAgent`）：自带多轮对话循环的 CLI 个人智能体，
   无 LLM API 时降级为规则驱动，保证离线可演示；有 API 时由 LLM 负责意图理解与生成。

## 快速开始

```bash
# 依赖极少：Python >= 3.10，仅 numpy 必装
pip install numpy

# 构建种子知识库（本地哈希嵌入 + BM25，无需联网）
python -m geo_guard.ingest --seed

# 运行完整演示：甄别一篇"榜单软文" + 多轮交互 + 行动引导
python examples/demo_personal_agent.py

# 运行测试
python -m pytest tests/ -q
```

可选：配置 `DASHSCOPE_API_KEY`（或 `OPENAI_API_KEY` + `OPENAI_BASE_URL`）后，
Personal Agent 将启用 LLM 意图理解与自然语言生成，嵌入可切换为云端向量模型。

## 开发流程（OpenSpec 规范驱动）

本仓库使用 [OpenSpec](https://github.com/Fission-AI/OpenSpec) 管理需求与变更：

```text
openspec/
├── config.yaml          # schema: spec-driven，产物语言 zh-CN
├── specs/               # 当前已生效的能力规格（真相源）
└── changes/             # 进行中的变更提案
    └── <change-name>/
        ├── proposal.md  # 为什么做、改什么
        ├── design.md    # 技术方案
        ├── tasks.md     # 实施清单
        └── specs/       # 规格增量（ADDED/MODIFIED Requirements）
```

流程：`/openspec-propose` 提案 → 评审 → `/openspec-apply-change` 实施 → `/openspec-archive-change` 归档并同步 specs。
设计前期的结构化头脑风暴记录见 [docs/01](docs/01-头脑风暴记录-GEO逆向甄别RAG.md)。

## 边界声明

- 本仓库全部内容仅用于**防御性甄别与平台治理研究**：知识库只描述"如何识别"操纵特征，
  不提供任何"如何实施 GEO 投毒"的可操作指引（与母项目技术报告的防御视角抽象原则一致）。
- 种子知识库中的法规/案例条目为**教学演示用途的摘编**，实际执法与维权请以权威原文为准。
- 判定输出为风险提示而非法律结论；高风险判定偏向精度（宁可少报、报出即可信）。

## 团队

- 纪力宁（负责人）· 马珠淇（产品与技术方案，本仓库）· 杨启航 · 许诗鹏

## 引用

设计与头脑风暴过程使用了 Scientific Agent Skills 的 scientific-brainstorming 技能：

> Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). Scientific Agent Skills:
> A Library of Procedural Knowledge for Research Agents. arXiv:2609.00065.
> https://doi.org/10.48550/arXiv.2609.00065
