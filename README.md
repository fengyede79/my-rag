# RAG System

面向中文问答场景的端到端检索增强生成系统。当前项目以食谱问答为落地场景，完整覆盖数据清洗、索引构建、混合检索、多轮状态管理、结构化生成、流式服务和 Live E2E 大模型实测闭环。

这个仓库不是一个只调用 LLM API 的问答 Demo，而是一次完整的 RAG 工程实践：系统把一次用户请求拆解为安全检查、轮次理解、指代解析、执行计划、检索执行、证据质量判断、上下文打包、答案生成和状态提交，并用真实服务与真实模型请求进行端到端验证。

## 项目亮点

- **完整 RAG Runtime**：从用户输入到状态写回形成稳定主链路，不把检索、生成和会话状态混在单个函数里。
- **混合检索与证据质量控制**：结合向量检索、BM25、RRF 融合、元数据过滤、fallback 标记和低证据保护。
- **上下文优先的多轮理解**：支持推荐列表后的序号引用、当前菜品继承、替换/约束追问、闲聊与越界问题隔离。
- **模块化执行边界**：`RetrievalExecutor` 统一管理检索策略，`ContextPacker` 控制上下文输入，`StateUpdatePolicy` 统一状态写入。
- **结构化生成优先**：在食材、步骤、技巧等明确内容类型下优先使用可控模板，减少不必要的模型自由发挥。
- **真实 Live E2E 测评**：启动真实 Flask 服务，调用真实 HTTP/SSE 接口，使用真实大模型完成 85 轮端到端验证。

## 系统架构

```mermaid
flowchart TD
    U["User Query"] --> SG["Basic Safety Gate"]
    SG --> SS["Session Snapshot + state_version"]
    SS --> TU["Turn Understanding"]
    TU --> RR["Reference Resolution"]
    RR --> EP["Execution Plan"]
    EP --> QP["Query Plan"]
    QP --> RE["Retrieval Executor"]
    RE --> HY["Vector + BM25 + Metadata Weighting"]
    HY --> EQ["Evidence Quality Check"]
    EQ -->|enough| CP["Context Packer"]
    EQ -->|insufficient| FB["Controlled Fallback / Low Evidence"]
    FB --> CP
    CP --> AG["Answer Generation / SSE Streaming"]
    AG --> SUP["StateUpdatePolicy"]
    SUP --> CM["Versioned State Commit"]
    CM --> OUT["Response"]
    OUT --> E2E["Live E2E Report"]
```

主链路遵循 `code/C8/docs/architecture/main-runtime-architecture-spec.md` 中的冻结架构基线。Stage 06 验收报告已确认：聊天检索统一经过 `RetrievalExecutor`，生成函数不直接扩展父文档、不直接写会话状态，写回由 `StateUpdatePolicy` 和版本化提交控制。

## 核心模块

| 模块 | 作用 |
| --- | --- |
| 数据与索引层 | 解析食谱数据，构建结构化文档块和 FAISS 向量索引。 |
| 轮次理解层 | 判断用户意图、答案模式、是否依赖上下文，以及是否需要检索。 |
| 指代解析层 | 处理“第一个”“这个”“刚才那个”等多轮引用，并给出置信度和证据来源。 |
| 检索执行层 | 统一执行主检索、fallback、证据质量判断、重排和父文档扩展。 |
| 上下文打包层 | 根据答案模式选择相关段落，限制上下文长度，避免生成层直接接触原始检索噪音。 |
| 生成与流式层 | 支持结构化答案、LLM 生成、无结果回答和 SSE 流式输出。 |
| 状态提交层 | 通过状态 diff 写回会话，避免低证据或中断流错误污染业务状态。 |
| 测评层 | 提供单元测试、架构验收测试和 Live E2E 真实服务测试报告。 |

## 测评结果

最终展示口径来自 2026-07-08 的 Live E2E 结果，模型为 `qwen-plus-2025-07-28`。测试通过真实 Flask 服务和真实 HTTP/SSE 请求执行，不使用 Flask `test_client` 或 mock 检索/生成。

| 测试集 | 轮次 | 通过 | 失败 | 通过率 |
| --- | ---: | ---: | ---: | ---: |
| Core 50 | 50 | 48 | 2 | 96.0% |
| Extended 35 | 35 | 29 | 6 | 82.9% |
| Total 85 | 85 | 77 | 8 | 90.6% |

补充指标：

- `INFRA_ERROR = 0`
- `RATE_LIMITED = 0`
- Core 集覆盖单轮菜谱详情、推荐列表、多轮引用、替换约束、低证据、越界拒答、SSE 流式和快速追问冲突。
- Extended 集用于暴露更复杂的多轮指代、低证据 fallback 和约束追问边界。

剩余失败主要集中在两类场景：复杂多轮指代在较长上下文中的稳定性，以及低证据情况下的 fallback 误召回控制。这些问题被保留为后续优化方向，不影响当前项目作为一个完整 RAG 工程闭环的展示价值。

## 工程演进闭环

项目经历了从基础 RAG 到冻结 runtime 架构的演进：

1. 数据准备与索引构建：完成食谱知识库解析、分块、元数据提取和向量索引。
2. 检索增强：加入 BM25、RRF、元数据过滤、别名与安全子串匹配。
3. 多轮对话：引入会话快照、指代解析、推荐列表引用和当前实体继承。
4. Runtime 架构化：拆分 Turn Understanding、Reference Resolution、Execution Plan、Retrieval Executor、Context Packer 和 StateUpdatePolicy。
5. 端到端验收：用 Stage 06 deterministic acceptance 验证冻结主链路。
6. Live E2E 实测：用真实服务和真实模型跑 Core + Extended 场景集，形成可审计报告。

## 快速开始

### 安装依赖

```powershell
cd code/C8
pip install -r requirements.txt
```

### 配置模型密钥

在 `code/C8/.env` 中配置：

```env
DASHSCOPE_API_KEY=your_api_key_here
```

可选配置：

```env
RAG_LLM_MODEL=qwen-plus-2025-07-28
```

### 启动命令行问答

```powershell
cd code/C8
python main.py
```

### 启动 Web 服务

```powershell
cd code/C8
python web_app.py
```

默认访问：

```text
http://127.0.0.1:5000
```

## 测试与评测

### 单元测试

```powershell
cd code/C8
pytest tests -q
```

### 检索执行器测试

```powershell
cd code/C8
pytest tests/test_retrieval_executor.py -q
```

### Live E2E 运行器

```powershell
cd code/C8
python e2e/live_e2e_runner.py `
  --models qwen-plus-2025-07-28 `
  --limit-turns 50 `
  --delay-seconds 5 `
  --max-retries 3 `
  --request-timeout-seconds 120 `
  --stream-timeout-seconds 180
```

生成结果位于：

```text
code/C8/e2e/results/
```

## 项目结构

```text
code/C8/
  main.py                         RAG 主系统入口
  web_app.py                      Flask Web / SSE 服务
  config.py                       配置管理
  rag_modules/
    turn_understanding.py         轮次理解
    reference_resolution.py       指代解析
    execution_planner.py          执行计划
    retrieval_executor.py         检索执行与证据质量
    retrieval_optimization.py     向量/BM25/RRF/元数据检索
    context_packer.py             上下文选择与裁剪
    structured_generation.py      结构化答案生成
    state_update_policy.py        状态 diff 写回策略
    turn_runtime.py               运行时生命周期
  e2e/
    live_e2e_runner.py            Live E2E CLI
    scenarios/                    Core / Extended 场景集
    reporting.py                  JSONL / Markdown 报告生成
  tests/                          单元测试与验收测试
  docs/
    architecture/                 冻结 runtime 架构文档
    architecture/evolution/       分阶段架构演进与验收报告
```

## 后续优化方向

项目当前已经完成展示闭环，后续若继续推进，优先考虑：

- 提升复杂多轮引用在长上下文中的稳定性。
- 加强低证据场景的 fallback 身份校验，进一步降低误召回。
- 扩充菜谱数据规模和元数据质量。
- 引入更细粒度的召回率、答案忠实度和状态一致性指标。

## 简历表述建议

可以这样概括本项目：

> 设计并实现了一个面向中文问答场景的端到端 RAG 系统，以食谱问答为落地场景，覆盖数据清洗、索引构建、混合检索、多轮状态管理、结构化生成、SSE 流式服务和 Live E2E 大模型实测。系统将主链路拆分为 Turn Understanding、Reference Resolution、Execution Plan、Retrieval Executor、Context Packer 和 StateUpdatePolicy 等模块，并通过 85 轮真实服务端到端测试验证整体闭环，最终达到 90.6% 总体通过率、Core 集 96.0% 通过率，且无基础设施错误和限流错误。
