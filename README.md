# 面向垂直知识库的 RAG 问答系统

一个面向食谱知识库场景的 RAG 问答项目。系统围绕“检索、路由、上下文组织、流式返回、多轮会话”构建完整应用闭环，而不是只演示一次模型调用。

## 项目价值

这个仓库的重点不是“做一个能答题的 Demo”，而是把垂直知识库问答系统里真正重要的几件事串起来：

- 用户问题如何先经过检索与路由，而不是直接丢给模型
- 多轮对话中如何处理指代消解、实体继承与话题切换
- 检索结果如何经过重排与过滤，减少上下文污染
- 流式接口与非流式路径如何共享一致的会话状态
- 项目效果如何通过评测集和诊断报告进行量化

因此，这个仓库更适合作为“RAG 应用工程实践”来看，而不仅仅是一个菜谱问答样例。

## 核心能力

### 1. 混合检索链路

系统组合了：

- 向量检索
- BM25 关键词检索
- RRF 重排
- Metadata Filter

用于支撑不同问法、不同内容类型与不同菜品范围下的召回。相比单一路径检索，这条链路更适合真实问答场景中的模糊表达与混合条件，也更容易观察每种召回手段对结果纯度的影响。

### 2. 三层路由

系统不是把所有问题都交给一次 LLM 路由判断，而是使用：

- 规则路由
- 语义路由
- LLM 路由

形成由快到慢的分层决策路径，兼顾响应速度和复杂问题覆盖，减少所有问题都直接进入 LLM 判断带来的额外开销。

### 3. 多轮会话

系统支持：

- 指代消解
- 实体继承
- 意图切换检测
- 历史压缩
- 会话状态管理

这使它能处理连续追问，而不是每次都当成一次无状态请求。

### 4. 流式交互

项目通过 Flask + SSE 提供 `/api/chat/stream`，让问答过程更接近真实产品交互形态，同时也暴露出会话一致性、状态共享与异常兜底这些工程问题。

## 当前效果

项目已经具备一套可落地的评测与诊断结果。当前数据来自 `104` 个问题、`128` 轮多轮对话评测，以及检索诊断报告：

| 指标 | 结果 |
|---|---:|
| 评测 case 数 | `104` |
| 对话轮次 | `128` |
| 整体通过率 | `92.97%` |
| 平均规则得分 | `0.9661` |
| 最终目标纯度 | `92.75%` |
| 平均纯度提升 | `2.02%` |

这些结果来自：

- `code/C8/evaluation/latest_report.json`

这些结果说明：

- 系统不只是“能跑”
- 检索质量、多轮会话和结果生成已经有一套最小可用的量化依据

## 技术栈

| 模块 | 技术 |
|---|---|
| 应用框架 | `Flask` |
| 检索框架 | `LangChain` |
| 向量索引 | `FAISS` |
| 关键词检索 | `BM25` |
| 重排 | `RRF` |
| Embedding | `BAAI/bge-small-zh-v1.5` |
| 大模型 | `qwen-turbo` |
| 流式返回 | `SSE` |
| 评测 | 规则评分 + LLM Judge |

## 项目结构

```text
my-rag/
├── code/C8/
│   ├── main.py
│   ├── web_app.py
│   ├── config.py
│   ├── rag_modules/
│   │   ├── retrieval_optimization.py   # 混合检索与重排
│   │   ├── generation_integration.py   # 生成、路由、会话集成
│   │   ├── hybrid_router.py            # 三层路由
│   │   ├── conversation_manager.py     # 多轮会话管理
│   │   └── stream_handler.py           # 流式异常兜底
│   ├── evaluation/
│   │   ├── dataset_builder.py
│   │   ├── run_evaluation.py
│   │   ├── process_diagnostics.py
│   │   └── latest_report.json
│   └── tests/
├── data/C8/
└── README.md
```

## 快速开始

### 1. 安装依赖

```bash
cd code/C8
pip install -r requirements.txt
```

### 2. 配置环境变量

在 `code/C8/` 目录下创建 `.env` 文件：

```env
DASHSCOPE_API_KEY=your_api_key_here
```

### 3. 启动命令行问答

```bash
python main.py
```

### 4. 启动 Web 服务

```bash
python web_app.py
```

访问：

```text
http://127.0.0.1:5000
```

## 你应该重点看什么

如果你是面试官、协作者或者正在学习这个项目，我建议优先看这几个文件：

- [code/C8/README.md](code/C8/README.md)
  作用：更详细的模块说明和开发入口
- [code/C8/rag_modules/retrieval_optimization.py](code/C8/rag_modules/retrieval_optimization.py)
  作用：混合检索、过滤与 RRF 重排主逻辑
- [code/C8/rag_modules/generation_integration.py](code/C8/rag_modules/generation_integration.py)
  作用：生成、路由、会话和流式能力的主集成点
- [code/C8/rag_modules/conversation_manager.py](code/C8/rag_modules/conversation_manager.py)
  作用：多轮对话状态管理
- [code/C8/evaluation/latest_report.json](code/C8/evaluation/latest_report.json)
  作用：项目效果基线与诊断数据

## 详细文档入口

根 README 负责说明“这个项目为什么值得看”。
更细的模块说明、评测命令和功能解释，请继续看：

- [code/C8/README.md](code/C8/README.md)
