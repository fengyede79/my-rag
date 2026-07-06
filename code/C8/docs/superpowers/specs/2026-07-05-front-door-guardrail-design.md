# Front Door Guardrail Design

Date: 2026-07-05

## Goal

Add a thin deterministic front door before the existing conversation RAG chain. The front door decides only whether a user input should enter the main chain.

This design keeps three components alive with clean boundaries:

```text
front_door_guardrail
  -> turn_qualification
  -> reference_resolution
```

The front door is not a semantic planner. It does not understand recipes, produce dish names, classify route types, build filters, rewrite queries, or influence retrieval inputs.

## Current Main Chain To Preserve

The existing main chain is the source of truth for useful recipe interactions:

```text
ask_question(question)
  -> qualify_turn(question)
  -> build_conversation_snapshot(...)
  -> resolve_reference_from_snapshot(...) when needed
  -> build_execution_plan(...)
  -> _build_query_plan(...)
  -> rewrite_query_for_execution(...)
  -> retrieval
  -> generation
  -> writeback_turn_state(...)
```

The front door must not replace this chain. It only decides whether the original question should enter it.

## Target Flow

```text
ask_question(question)
  -> check_front_door(question)
       -> block        # return clarification, no query planning, no retrieval
       -> direct_reply # return polite answer, no query planning, no retrieval
       -> continue     # pass original question unchanged to existing main chain
  -> qualify_turn(question)
  -> build_conversation_snapshot(...)
  -> resolve_reference_from_snapshot(...) when needed
  -> build_execution_plan(...)
  -> _build_query_plan(...)
  -> retrieval
  -> generation
  -> state writeback
```

For `continue`, `question` must be passed through unchanged.

## Component Boundaries

### `front_door_guardrail`

Responsibility: decide if the input should enter the main chain.

Allowed decisions:

```python
{
    "decision": "continue" | "block" | "direct_reply",
    "reason": str,
    "message": str | None,
}
```

Allowed behavior:

- Block structurally useless input.
- Directly reply to exact smalltalk.
- Directly reply to clearly out-of-domain input.
- Default to `continue`.

Forbidden behavior:

- No model calls.
- No vector search.
- No query rewriting.
- No dish extraction.
- No route classification.
- No filters.
- No semantic hints.

The front door must never return, mutate, or imply:

- `dish_name`
- `intent_type`
- `route_type`
- `filters`
- `content_type`
- `semantic_result`
- `rewritten_query`

### `turn_qualification`

Responsibility: classify only inputs that have already passed the front door and are entering the main chain.

It should handle:

- `recommendation_query`
- `followup_query`
- `domain_query`

It should no longer be the primary smalltalk handler. Exact smalltalk belongs to the front door. A compatibility fallback is acceptable during migration, but tests should move smalltalk expectations to `front_door_guardrail`.

### `reference_resolution`

Responsibility: resolve what a follow-up refers to.

It should handle:

- Pronouns: `它`, `这个`, `那道菜`
- Ordinal references: `第二个怎么做？`
- Corrections: `不是这个，是蛋炒饭`
- Short single-dish follow-ups: `有什么小技巧别粘锅？`
- Cleaned explicit dish mentions: `那蛋炒饭需要哪些食材？`

It should not handle smalltalk, clear out-of-domain input, or structural validity checks.

### `_build_query_plan` And Retrieval

Responsibility: own useful semantic planning and retrieval inputs.

Allowed dish-name sources:

1. `reference_resolution`
2. `query_router`
3. `_infer_explicit_dish_topic`

Forbidden dish-name sources:

- `front_door_guardrail`
- local semantic analyzers
- broad fallback rules before query planning

Retrieval continues to consume only query-plan outputs:

```text
question
rewritten_query
filters
dish_name
```

## Front Door Decisions

### 1. `block`

Use `block` only for inputs that are structurally impossible to answer without additional context.

Examples:

```text
""
" "
"？"
"这道"
"那道"
"这个"
"那个"
"它"
```

Expected shape:

```python
{
    "decision": "block",
    "reason": "empty_or_isolated_reference",
    "message": "我还不知道你指的是哪道菜，可以说具体一点吗？",
}
```

Do not block meaningful short food inputs:

```text
"牛排"
"煎蛋"
"凉面"
```

Do not block resolvable follow-ups:

```text
"这道菜怎么做？"
"它需要什么食材？"
```

Those should continue into `turn_qualification` and `reference_resolution`.

### 2. `direct_reply` For Exact Smalltalk

Exact smalltalk belongs to the front door.

Examples:

```text
"你好"
"您好"
"谢谢"
"哈哈"
"你是谁"
"你能做什么"
```

Expected shape:

```python
{
    "decision": "direct_reply",
    "reason": "smalltalk_exact",
    "message": "我是食谱助手，可以帮你查菜谱、做法、食材和推荐。",
}
```

Forbidden broad smalltalk signals:

```text
"怎么样"
"可以吗"
"好不好"
"行不行"
```

Those are common recipe-question suffixes and must continue into the main chain.

### 3. `direct_reply` For Clear Out-Of-Domain

The out-of-domain check must be conservative. A query should stop only when it has a clear out-of-domain signal and no food, recipe, cooking, or kitchen-action signal.

Examples:

```text
"今天天气怎么样"
"Python怎么学"
"股票怎么买"
"手机壳发黄怎么办"
"路由器总断网怎么办"
```

Expected shape:

```python
{
    "decision": "direct_reply",
    "reason": "clear_out_of_domain",
    "message": "我主要帮你处理食谱和做菜相关问题，可以问我菜品做法、食材或推荐。",
}
```

Must continue:

```text
"空气炸锅鸡翅怎么做"
"蛋炒饭里放螺丝椒可以吗"
"厨房刀怎么切肉更安全"
"牛排怎么煎"
"今天吃什么"
"我想吃点清淡的"
```

### 4. `continue`

If no deterministic stop condition matches:

```python
{
    "decision": "continue",
    "reason": "default_continue",
    "message": None,
}
```

This is intentional. Prefer false negatives over false positives. It is acceptable to let weak inputs enter the main chain; it is not acceptable to block useful recipe questions.

## State Writeback

For `block` and `direct_reply`, `ask_question()` may write the interaction to conversation history, but must not update reliable recipe state.

The turn info passed to `writeback_turn_state()` should use:

```python
{
    "turn_type": "front_door_blocked" | "front_door_direct_reply",
    "response_mode": "polite_direct_reply",
    "should_retrieve": False,
    "should_update_topic_state": False,
    "should_update_entity_state": False,
    "should_run_reference_resolution": False,
    "reference_trigger": "none",
}
```

The execution result should be:

```python
{"success": True, "answer": answer}
```

`state_writeback_review` must treat these turns as `message_only`.

## Relationship To Existing `guardrail.py`

The existing `guardrail.py` contains useful phrases and answer text, but it is not currently wired into the main path and mixes several concerns.

This implementation should not simply call `_maybe_handle_guardrail_query()` from `ask_question()`. Instead:

- Create a focused `front_door_guardrail.py`.
- Move only conservative deterministic checks into the front door.
- Keep broad or uncertain food-judgement behavior out of this pass unless covered by tests.
- Leave retrieval and query planning untouched.

`guardrail.py` can remain for existing tests or future cleanup, but it should not be the new front-door API.

## Rejected Approaches

Do not restore or introduce:

- `local_semantic_analyzer.py`
- `strict_guardrail.py`
- local small-model semantic routing before query planning
- any pre-plan module that emits dish names, route types, filters, or rewritten queries

Reason: those layers can pollute retrieval inputs before the main chain has a chance to resolve references and plan queries correctly.

## Test Scope

Add focused regression tests only.

Direct module tests:

- Continue useful recipe-like inputs:
  - `土豆丝怎么样`
  - `蛋炒饭好不好`
  - `青椒肉丝可以吗`
  - `牛排怎么煎`
  - `宫保鸡丁怎么做`
  - `今天吃什么`
  - `我想吃点清淡的`
  - `这道菜怎么做？`
  - `它需要什么食材？`
- Block useless inputs:
  - empty input
  - pure punctuation
  - `这道`
  - `那个`
  - `它`
- Direct reply:
  - `你好`
  - `谢谢`
  - `你是谁`
  - `Python怎么学`
  - `今天天气怎么样`

Integration tests:

- `block` stops before query planning and retrieval.
- `direct_reply` stops before query planning and retrieval.
- `continue` leaves dish extraction to `_build_query_plan`.
- Existing reference-resolution tests still pass.
- Existing focused conversation suite still passes.

## Acceptance Criteria

1. `front_door_guardrail.check_front_door()` exists and returns only `decision`, `reason`, and `message`.
2. Exact smalltalk is handled by the front door.
3. `turn_qualification` no longer owns smalltalk as its primary responsibility.
4. `continue` passes the original question unchanged into the existing main chain.
5. The front door does not output or mutate semantic information.
6. Structurally useless inputs stop before planning and retrieval.
7. Clearly non-retrieval inputs get a polite direct reply.
8. Resolvable follow-ups continue into `reference_resolution`.
9. `local_semantic_analyzer` and `strict_guardrail` remain absent.
10. Focused unit, conversation, and reference-resolution tests pass.
