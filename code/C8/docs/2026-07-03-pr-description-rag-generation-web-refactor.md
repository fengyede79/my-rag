# PR 说明：RAG 生成链路与 Web 结构重构

日期：`2026-07-03`

建议 PR 标题：`[codex] 重构 RAG 生成链路与 Web 应用结构`

## 摘要

这次改动主要围绕食谱 RAG 应用的模块边界和运行时行为做了一轮整理与收敛。

- 将原本过大的生成集成逻辑拆分为几个职责更清晰的辅助模块：
  - `guardrail.py`
  - `prompts.py`
  - `stream_handler.py`
  - `structured_generation.py`
- 将 Flask 聊天页面从 Python 内嵌 HTML 中拆出，迁移到 `templates/index.html`
- 将查询过滤条件提取逻辑统一收口到检索层，不再保留在 `main.py`
- 顺带修正了一些会话、检索、缓存与评测相关的小问题
- 同步更新了测试、依赖声明和仓库说明文档

## 背景与目的

在这次调整之前，`generation_integration.py` 和 Flask 入口承担了过多职责：

- Prompt 构建、护栏判断、结构化回答、流式兜底都混在同一个模块里
- Web 页面模板直接写在 Python 字符串中，维护成本较高
- 一些检索和元数据处理逻辑存在重复分散
- 缓存读取和父文档查找路径还有进一步增强稳定性的空间

这次重构并不是为了增加一批新功能，而是为了在保持原有能力的前提下，让代码更清晰、更容易维护，也更方便继续迭代测试。

## 具体改动

### 1. 生成链路拆分

- 将查询护栏分类和保守回答提取到 `rag_modules/guardrail.py`
- 将 Prompt 模板和“无上下文时的兜底回答”提取到 `rag_modules/prompts.py`
- 将 `invoke` / `stream` 的安全封装提取到 `rag_modules/stream_handler.py`
- 将基于 Markdown 结构直接拼装回答的逻辑提取到 `rag_modules/structured_generation.py`
- `generation_integration.py` 调整为编排层，负责串联这些子模块，而不是继续承载全部细节

### 2. Web 应用结构整理

- 用 `render_template(...)` 替换了 `render_template_string(...)`
- 将页面模板迁移到 `code/C8/templates/index.html`
- 精简 `web_app.py`，让它更聚焦在路由和 SSE 输出逻辑上
- 对 SSE 事件格式做了收口，使事件输出结构更一致

### 3. 检索与数据层增强

- 将查询元数据过滤提取到 `RetrievalOptimizationModule.extract_filters_from_query(...)`
- 统一复用了 `DataPreparationModule` 中的食材关键词列表
- 为分块缓存增加了版本号和 checksum 校验
- 增加父文档索引映射，避免重复线性扫描
- 对部分菜名匹配分支增加了空值保护，减少脆弱路径

### 4. 小型正确性修复

- 修正会话摘要中“最后一条用户消息”的选取逻辑
- 调整路由规则遍历逻辑，保持优先级顺序更明确
- 改进评测 JSON 解析，使其能兼容被 Markdown 代码块包裹的 LLM 返回结果
- 调整默认索引路径解析方式，使其落在项目代码目录下

### 5. 文档与依赖更新

- 更新根目录 `README.md`，把项目定位从“简单 Demo”扩展为“更完整的 RAG 应用工程实践”
- 补充依赖声明，如 `python-dotenv`、`numpy`、`pytest`

## 影响

- 生成链路的维护成本更低
- 编排逻辑和辅助逻辑边界更清晰
- 后续继续修改 Prompt、护栏、流式输出、结构化回答会更容易
- Flask 入口文件更轻、更好读
- 缓存安全性和检索健壮性有所提升

## 验证

执行：

```bash
pytest code/C8/tests
```

结果：

- `38 passed`

另外检查：

```bash
git diff --check
```

结果：

- 没有阻塞提交的 diff 格式或尾随空白问题

