# 序号引用与自然追问补齐 SPEC

## 背景

2026-07-04 的随机真实链路测试显示，新会话框架已经稳定接管主流程，但仍存在一类没有落地的新框架能力：

- 推荐列表后用户说“第二个怎么做？”时，系统把“第二个”当成菜名，而不是映射到推荐列表第 2 项。
- 用户说“第一个看起来不错，做法说一下”时，系统把“第一个看起来不错”当成菜名。
- 用户说“那蛋炒饭需要哪些食材？”时，系统把“那蛋炒饭”当成菜名，导致父文档过滤失败。
- 用户说“有什么小技巧别粘锅？”时，系统没有继承上一轮可靠菜品，而是把整句误当菜名。
- 用户说“换个清淡一点的菜”时，推荐结果没有稳定体现“清淡”偏好。

这些问题不应该通过恢复旧的独立序号解析模块来解决。旧模块已被新架构替代，正确方向是把这些自然语言现象并入统一的 `Turn Qualification -> Conversation State Builder -> Reference Resolution -> Execution Planning` 链路。

## 目标

1. 在新框架内支持推荐列表的序号引用，例如“第一个”“第二个”“1”“2号”“第 2 个”。
2. 在新框架内处理口语前缀清洗，例如“那蛋炒饭”“这个蛋炒饭”“刚才那个蛋炒饭”。
3. 在新框架内处理省略主语的短追问，例如“有什么小技巧别粘锅？”、“需要哪些食材？”。
4. 在推荐查询里保留自然偏好约束，例如“清淡一点”“下饭”“新手”“简单”“早餐”。
5. 不恢复旧的 `resolve_query_reference()`、旧推荐缓存、旧 `complete_query()` 主路径，也不新增平行语义入口。

## 非目标

- 不重做向量检索、BM25 或 RRF 算法。
- 不重建知识库数据。
- 不改变 Web API 响应格式。
- 不要求模型自由决定序号映射；序号映射必须由结构化状态确定，并由 guard 校验。
- 不把所有模糊追问都强行继承当前菜品；只有低风险短追问可以继承，推荐列表多候选代词仍应澄清。

## 术语

- `ordinal reference`：用户用序号指向推荐列表项，如“第二个怎么做”。
- `explicit dish with discourse prefix`：用户明确说出菜名，但带有口语前缀，如“那蛋炒饭”。
- `implicit single-dish followup`：用户没有说菜名，但在单菜上下文中询问细节，如“有什么技巧别粘锅”。
- `preference constraints`：用户表达的偏好，例如清淡、下饭、简单、新手、早餐。

## 当前缺口

### Turn Qualification 缺口

当前 `qualify_turn()` 只识别：

- smalltalk
- 固定推荐问法
- 代词前缀追问
- correction turn

它没有识别：

- `第一个怎么做`
- `第二个需要什么食材`
- `1 怎么做`
- `第一个看起来不错，做法说一下`
- `需要哪些食材`
- `有什么小技巧别粘锅`

结果是这些输入会落入 `domain_query`，不会运行 reference resolution。

### Conversation State Builder 缺口

当前 snapshot 里虽然有 `priority_order` 的 `"ordinal_recommendation_reference"`，但没有实际字段表达：

```json
{
  "ordinal_reference": {
    "rank": 2,
    "raw_text": "第二个",
    "remaining_query": "怎么做"
  }
}
```

也没有字段表达清洗后的显式菜名：

```json
{
  "cleaned_explicit_dish": {
    "value": "蛋炒饭",
    "removed_prefix": "那"
  }
}
```

### Reference Resolution 缺口

当前 resolver 只处理：

1. correction explicit target
2. 推荐列表里的代词歧义澄清
3. 单候选继承
4. no reference needed

它没有处理：

- ordinal rank -> recent recommendations
- ordinal 越界 -> ask clarification
- cleaned explicit dish -> explicit target
- implicit single-dish followup -> current dish

### Execution Rewrite 缺口

当前 `rewrite_query_for_execution()` 只处理 `apply_correction`。它需要支持：

- `resolve_ordinal_selection`
- `resolve_cleaned_explicit_dish`
- `resolve_current_dish_followup`

例如：

- “第二个怎么做？” + 推荐列表第 2 项“麻婆豆腐” -> “麻婆豆腐怎么做？”
- “那蛋炒饭需要哪些食材？” -> “蛋炒饭需要哪些食材？”
- “有什么小技巧别粘锅？” + 当前菜“蛋炒饭” -> “蛋炒饭有什么小技巧别粘锅？”

## 目标架构补丁

### 1. Turn Qualification 新规则

新增识别：

```json
{
  "turn_type": "followup_query",
  "should_run_reference_resolution": true,
  "reference_trigger": "ordinal_reference | implicit_detail_followup | explicit_dish_with_prefix"
}
```

触发条件：

- 以 `第一个|第二个|第三个|第1个|第2个|1|2|1号|2号` 开头，并带有做法/食材/技巧/介绍/评价意图。
- 短问句包含 `怎么做|做法|食材|材料|技巧|粘锅|难不难|要多久|热量|介绍`，且没有可靠显式菜名。
- 明确菜名带口语前缀，例如 `那蛋炒饭`、`这个蛋炒饭`。

约束：

- `ordinal_reference` 必须进入 reference resolution。
- 推荐列表中的纯代词“它/这个/那个怎么做”仍然应澄清，不能默认第一个。

### 2. Snapshot 新字段

在 `resolution_constraints` 中新增：

```json
{
  "ordinal_reference": {
    "rank": 2,
    "raw_text": "第二个",
    "remaining_query": "怎么做"
  },
  "cleaned_explicit_dish": {
    "value": "蛋炒饭",
    "removed_prefix": "那"
  },
  "implicit_followup": {
    "enabled": true,
    "remaining_query": "有什么小技巧别粘锅",
    "requires_single_active_dish": true
  },
  "preference_constraints": {
    "taste": ["清淡"],
    "meal": ["早餐"],
    "difficulty": ["新手", "简单"],
    "style": ["下饭"]
  }
}
```

字段语义：

- `ordinal_reference.rank` 使用 1-based rank。
- `ordinal_reference.remaining_query` 保留用户真实意图，不保留序号和评价废词。
- `cleaned_explicit_dish.value` 只用于显式菜名清洗，不用于凭空猜菜。
- `implicit_followup.enabled` 只有在问句是低风险细节追问时为 true。
- `preference_constraints` 后续交给 query plan/retrieval filters 或 rerank 使用。

### 3. Reference Resolution 新分支优先级

新的优先级：

1. `explicit_query_target`
2. `cleaned_explicit_dish`
3. `ordinal_recommendation_reference`
4. `last_confirmed_target`
5. `implicit_single_dish_followup`
6. `pronoun_recommendation_reference`
7. `current_dish`

注意：

- `ordinal_recommendation_reference` 是明确引用，rank 有效时可以直接 resolved。
- 推荐列表里的“它/这个/那个”不是明确引用，仍然 ambiguous。
- `implicit_single_dish_followup` 只有在当前没有推荐列表多候选歧义，且存在一个 active current dish 时才能 resolved。

### 4. Resolution 输出新增 next_action

新增或复用：

```json
{
  "next_action": "retrieve_detail | ask_clarification | apply_correction | apply_reference_resolution"
}
```

对于 ordinal 和 implicit followup，使用：

```json
{
  "resolution_status": "resolved",
  "resolved_target": "麻婆豆腐",
  "target_source": "ordinal_recommendation_reference",
  "next_action": "apply_reference_resolution",
  "writeback_eligible": true,
  "decision_basis": "explicit"
}
```

### 5. Rewrite 行为

`rewrite_query_for_execution()` 负责把 resolved target 与用户剩余意图合并。

示例：

| 输入 | 状态 | 输出 |
|---|---|---|
| 第二个怎么做 | recent_recommendations[2] = 麻婆豆腐 | 麻婆豆腐怎么做 |
| 第一个看起来不错，做法说一下 | recent_recommendations[1] = 燕麦鸡蛋饼 | 燕麦鸡蛋饼做法说一下 |
| 那蛋炒饭需要哪些食材 | cleaned_explicit_dish = 蛋炒饭 | 蛋炒饭需要哪些食材 |
| 有什么小技巧别粘锅 | current_dish = 蛋炒饭 | 蛋炒饭有什么小技巧别粘锅 |

### 6. Guard 行为

- ordinal rank 小于 1 或大于推荐列表长度 -> `ask_clarification`
- ordinal 有效但推荐列表为空 -> `ask_clarification`
- resolved target 不在 recent recommendations 且不是 explicit cleaned dish/current dish -> `ask_clarification`
- implicit followup 在 recommendation_list 模式下不允许默认选择 -> `ask_clarification`

## 测试要求

### 单元测试

必须新增：

- `qualify_turn("第二个怎么做？")` 进入 reference resolution。
- `build_conversation_snapshot(..., current_query="第二个怎么做？")` 提取 rank=2。
- `resolve_reference_from_snapshot()` 将 rank=2 映射到推荐列表第 2 项。
- ordinal 越界返回 clarification。
- `rewrite_query_for_execution()` 将 ordinal 结果重写为真实菜名查询。
- “那蛋炒饭需要哪些食材”清洗为“蛋炒饭”。
- “有什么小技巧别粘锅”在 single_dish 模式下继承 current dish。
- “它怎么做”在 recommendation_list 模式下仍然澄清。

### 真实链路慢测

必须覆盖：

1. `有没有适合新手的早餐？ -> 第一个看起来不错，做法说一下`
2. `我晚上想吃点下饭的，有啥推荐？ -> 第二个怎么做？`
3. `家里只有鸡蛋和米饭，能做什么？ -> 那蛋炒饭需要哪些食材？ -> 有什么小技巧别粘锅？`
4. `今天吃什么？ -> 它怎么做？` 仍然澄清
5. `换个清淡一点的菜` 推荐结果不应明显偏离清淡意图

## 验收标准

- “第二个怎么做？”不再把“第二个”当菜名。
- “第一个看起来不错，做法说一下”能映射到推荐列表第 1 项。
- “那蛋炒饭需要哪些食材？”不会把“那蛋炒饭”当菜名。
- 单菜上下文中的“有什么小技巧别粘锅？”能继承当前菜。
- 推荐列表多候选中的“它怎么做？”仍然澄清，不乱猜。
- 所有新增能力都通过新 `Reference Resolution` 链路完成，不恢复旧平行模块。

