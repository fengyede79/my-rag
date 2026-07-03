# 尝尝咸淡 RAG 系统

一个基于食谱知识库的检索增强生成（RAG）系统，支持多轮对话、意图路由、混合检索和流式输出。

## 项目概述

本项目实现了一个面向食谱领域的 RAG 系统，具备以下核心能力：

- **混合检索**：向量检索 + BM25 + RRF 重排，支持精确/模糊元数据过滤
- **三层路由**：规则路由 → 语义路由 → LLM 路由，兼顾速度与准确性
- **多轮对话**：会话状态管理、指代消解、实体继承、意图切换检测
- **结构化生成**：支持按内容类型（食材/步骤/技巧/介绍）定向生成
- **边界护栏**：Query Guardrails 拦截越界问题
- **评测体系**：规则评分 + LLM 裁判评分

## 技术栈

| 组件 | 技术选型 |
|------|----------|
| 向量数据库 | FAISS |
| Embedding | BAAI/bge-small-zh-v1.5 |
| LLM | 通义千问（qwen-turbo） |
| 框架 | LangChain + Flask |
| 评测 | 规则引擎 + LLM Judge |

## 项目结构

```text
code/C8/
├── main.py                      # RAG 系统主程序
├── config.py                    # 配置管理
├── web_app.py                   # Web 服务（流式 SSE）
├── requirements.txt             # 依赖清单
├── rag_modules/                  # 核心 RAG 模块
│   ├── __init__.py
│   ├── data_preparation.py      # 数据加载与分块
│   ├── index_construction.py    # 向量索引构建
│   ├── retrieval_optimization.py # 混合检索与重排
│   ├── generation_integration.py # LLM 生成与路由
│   ├── hybrid_router.py         # 三层混合路由
│   └── conversation_manager.py  # 多轮会话管理
├── evaluation/                   # 评测模块
│   ├── dataset_builder.py       # 测试集构建
│   ├── run_evaluation.py        # 评测执行器
│   ├── scoring.py               # 评分逻辑
│   └── testset.json             # 测试集
└── tests/                       # 单元测试
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

### 3. 启动交互式问答

```bash
python main.py
```

### 4. 启动 Web 服务

```bash
python web_app.py
# 访问 http://127.0.0.1:5000
```

## 核心模块说明

### 数据准备（DataPreparationModule）

负责从 Markdown 文件加载菜谱文档，按结构分块（食材/步骤/技巧等），并提取元数据（菜品名、分类、难度）。

### 索引构建（IndexConstructionModule）

基于 `sentence-transformers` 生成向量，使用 FAISS 构建向量索引，支持持久化存储。

### 检索优化（RetrievalOptimizationModule）

- **混合检索**：向量检索 + BM25，RRF 融合
- **元数据过滤**：支持按分类、难度、内容类型、食材过滤
- **降级策略**：精确匹配 → 模糊匹配 → 无过滤
- **同菜品去重**：同一菜品最多保留 3 个子块

### 三层路由（HybridRouter）

```text
用户查询
    │
    ▼
规则路由（微秒级）── 命中 → 返回 content_type
    │
    ▼ 未命中
语义路由（毫秒级）── 命中 → 返回 content_type + 置信度
    │
    ▼ 未命中
LLM 路由（秒级）───── 兜底返回 content_type
```

### 生成集成（GenerationIntegrationModule）

- **结构化答案优先**：文档结构明确时直接组装答案，减少 LLM 调用
- **张冠李戴校验**：生成前校验菜品一致性
- **Query Guardrails**：识别越界问题并给出保守回答
- **流式输出**：支持 SSE 流式响应

### 会话管理（ConversationManager）

- **指代消解**：识别"它"、"这个"等代词并还原为菜品名
- **实体继承**：多轮追问时自动继承当前菜品
- **意图切换检测**：识别"换个话题"等切换信号
- **历史压缩**：超过阈值时压缩早期对话

## 评测

```bash
# 构建测试集
python evaluation/run_evaluation.py --build-dataset --limit 12

# 运行评测
python evaluation/run_evaluation.py --limit 12 --use-llm-judge
```

评测维度包括：
- 检索质量（召回率、相关性）
- 生成质量（答案正确性、引用准确性）
- 路由准确率
- 护栏有效率

## 交互示例

```text
您的问题: 推荐几个素菜
查询类型: list (置信度: 0.95)
检索相关文档...
找到 3 个相关文档块: 凉拌黄瓜[技巧], 蒜蓉西兰花[综合], 蚝油生菜[综合]

为您推荐以下菜品：
1. 凉拌黄瓜
2. 蒜蓉西兰花
3. 蚝油生菜

您的问题: 1需要什么食材
序号引用解析: '1需要什么食材' -> '凉拌黄瓜需要什么食材'
查询类型: detail (置信度: 0.95)
提取过滤条件: {'content_type': 'ingredients'}
检索相关文档...
找到 3 个相关文档块: 凉拌黄瓜[食材]

## 所需食材
根据【食谱：凉拌黄瓜】整理：
- 黄瓜：2根
- 蒜末：适量
- 生抽：1勺
- 醋：1勺
- 香油：少许
- 辣椒油：适量
```

## 配置说明

主要配置项（`config.py`）：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `embedding_model` | `BAAI/bge-small-zh-v1.5` | 向量模型 |
| `llm_model` | `qwen-turbo` | LLM 模型 |
| `top_k` | `3` | 检索返回条数 |
| `temperature` | `0.1` | 生成温度 |
| `max_tokens` | `2048` | 最大 token 数 |
| `rebuild_index_if_missing` | `True` | 索引缺失时自动重建 |

可通过环境变量覆盖：`RAG_EMBEDDING_MODEL`、`RAG_LLM_MODEL`、`RAG_TOP_K` 等。
