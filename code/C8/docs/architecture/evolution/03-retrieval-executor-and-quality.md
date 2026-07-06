# 03 Retrieval Executor And Quality

Status: accepted baseline

## Purpose

Move retrieval decisions out of the main chain and into a single Retrieval Executor with evidence quality checks and controlled fallback.

Stage 03 is not primarily a search-quality tuning stage. It is a control-boundary stage. The main chain should stop deciding how retrieval runs, when filters relax, when fallback runs, and whether empty evidence is answerable. Those decisions belong to `RetrievalExecutor`.

The current runtime may not complete every final-architecture retrieval behavior immediately, but it must not drift from the frozen architecture. Once `RetrievalExecutor` owns retrieval execution, old production retrieval branches must be removed or reduced to actively called low-level helpers. Uncalled old retrieval functions, branches, fallback paths, and tests must not remain.

## Current Starting Point

Current retrieval control is split across:

- `main.py::_search_relevant_chunks()`
  - merges query-plan filters with extracted filters;
  - forces `dish_name` into filters;
  - chooses `metadata_filtered_search()` or `hybrid_search()`;
  - performs a tips/content-type fallback by relaxing `content_type`.
- `rag_modules/retrieval_optimization.py`
  - owns vector and BM25 search;
  - owns RRF-like fusion/rerank behavior;
  - owns metadata filtering;
  - owns exact dish special handling;
  - owns internal metadata degradation behavior.

This works for a demo, but the boundary is wrong. The main chain should not know fallback rules, and the low-level retrieval module should not decide business-level evidence policy.

Stage 03 keeps `RetrievalOptimizationModule` as a low-level retrieval provider where useful, but moves orchestration into a new retrieval-facing module.

## First Thing To Do

Introduce a retrieval-facing contract before changing retrieval semantics:

```text
QueryPlan
-> RetrievalExecutor.execute(query_plan)
-> RetrievalResult
```

Then cut production retrieval calls in `main.py` to the executor.

The first implementation should preserve current successful list/detail retrieval behavior while making retrieval quality and fallback explicit.

## Scope

Stage 03 owns:

- `QueryPlan` retrieval-facing structure;
- query-plan normalization from the existing runtime query plan into the retrieval-facing contract;
- `RetrievalExecutor` single production entrypoint;
- primary retrieval orchestration;
- metadata soft weighting boundary;
- fusion boundary;
- evidence quality check;
- controlled fallback;
- low-evidence/no-result result object;
- trace fields needed to explain retrieval behavior;
- deletion of old production retrieval orchestration from `main.py`.

Stage 03 does not own parent expansion, section trimming, context packing, answer-mode specialization, streaming lifecycle, or state version checks.

Stage 03 also does not rewrite upstream intent routing. Existing query planning may still determine `route_type`, `dish_name`, and initial filters. Before retrieval, those fields must be normalized into the retrieval-facing `QueryPlan` contract.

## QueryPlan Contract

`QueryPlan` is retrieval-facing only. It should be created before retrieval and consumed by `RetrievalExecutor`.

Required shape:

```python
{
    "query": "宫保鸡丁 做法",
    "original_query": "第一个怎么做",
    "dish_name": "宫保鸡丁",
    "filters": {
        "content_type": "steps",
        "ingredient": ["鸡肉"],
        "taste": ["不辣"],
    },
    "top_k": 3,
    "fallback_policy": "disabled | relaxed_filters | broad_search",
    "hard_filters": ["dish_name"],
    "soft_filters": ["ingredient", "taste", "difficulty", "time", "health_preference"],
    "answer_mode_hint": "recipe_detail",
    "trace_id": "optional"
}
```

Rules:

- `query` is the actual retrieval query.
- `original_query` is preserved for diagnostics.
- Explicit `dish_name` can be a hard filter.
- `content_type` may be hard or strong preferred depending on intent.
- Ingredient, taste, difficulty, time, and health preference are soft filters by default.
- `fallback_policy="disabled"` is the default for exact dish detail requests unless the caller explicitly allows relaxation.
- Broad search must not run for an exact missing dish unless policy says it may.

## QueryPlan Normalization

The existing runtime query plan is not yet the retrieval-facing `QueryPlan`. Stage 03 must add an explicit normalization step instead of letting `main.py` pass its current ad hoc dict directly into the executor.

Conceptual API:

```python
def build_retrieval_query_plan(
    *,
    original_query: str,
    rewritten_query: str,
    base_query_plan: dict,
    execution_plan: dict,
    resolution: dict | None,
    preference_constraints: dict | None,
    top_k: int,
) -> dict:
    ...
```

Mapping rules:

- `query` comes from the rewritten retrieval query.
- `original_query` keeps the user's original query.
- `dish_name` comes from resolved target first, then base query plan dish name.
- `filters` starts from base query plan filters plus safe preference constraints.
- `answer_mode_hint` comes from Turn Understanding or Execution Plan when available.
- `top_k` comes from runtime config.
- `hard_filters` contains `dish_name` only when the dish is explicit or reliably resolved.
- `content_type` is hard only for exact detail intents where relaxing it would change the answer target; otherwise it is a strong preference.
- `soft_filters` contains sparse metadata preferences such as ingredient, taste, difficulty, time, and health preference.
- `fallback_policy` defaults to `disabled` for hard exact-dish requests, `relaxed_filters` for sparse preference/list queries, and never `broad_search` unless the caller explicitly sets it.

This normalization function is the cut line between upstream planning and retrieval execution.

## RetrievalExecutor Contract

`RetrievalExecutor` is the only layer that decides how retrieval is executed.

Conceptual API:

```python
class RetrievalExecutor:
    def __init__(self, retrieval_module):
        ...

    def execute(self, query_plan: dict) -> dict:
        ...
```

Required output:

```python
{
    "chunks": [],
    "quality": {
        "enough_evidence": True,
        "quality_reason": "exact_dish_and_content_type_matched",
        "fallback_used": False,
        "relaxed_filter": False,
        "candidate_count": 3,
        "selected_dishes": ["宫保鸡丁"],
    },
    "low_evidence": None,
    "trace": {
        "strategy": "primary",
        "query": "宫保鸡丁 做法",
        "filters": {},
        "hard_filters": [],
        "soft_filters": [],
        "fallback_policy": "disabled",
        "primary_count": 3,
        "fallback_count": 0,
    }
}
```

If evidence is insufficient and no answer should be generated from retrieved chunks:

```python
{
    "chunks": [],
    "quality": {
        "enough_evidence": False,
        "quality_reason": "exact_dish_not_found",
        "fallback_used": False,
        "relaxed_filter": False,
        "candidate_count": 0,
        "selected_dishes": [],
    },
    "low_evidence": {
        "answer_type": "no_result",
        "answer": "知识库里没有找到这道菜的可靠做法。",
        "state_diff_policy": "low_evidence",
        "quality_reason": "exact_dish_not_found",
    },
    "trace": {...}
}
```

Rules:

- `chunks` is always present.
- `quality` is always present.
- `low_evidence` is either `None` or a returnable result.
- Generation should not need to infer low evidence from an empty chunk list.
- Fallback chunks must carry metadata markers:

```python
{
    "fallback": True,
    "relaxed_filter": True
}
```

## Retrieval Order

Required order:

```text
QueryPlan
-> primary retrieval
-> fusion
-> quality check
-> optional fallback
-> fallback quality check
-> rerank
-> RetrievalResult
```

Stage 03 may use existing `RetrievalOptimizationModule.hybrid_search()` and `metadata_filtered_search()` internally, but orchestration belongs to `RetrievalExecutor`.

Because existing low-level methods may already perform RRF-like fusion and reranking internally, the first Stage 03 implementation may treat the low-level provider output as `primary_candidates` and record `fusion_strategy="delegated"`. Do not add a second LLM reranker in this stage.

### Primary Retrieval

Primary retrieval should preserve current successful behavior first.

Suggested initial mapping:

- If `QueryPlan` has a hard `dish_name`, retrieve with an exact-dish preference.
- If filters exist, call the low-level filtered retrieval provider.
- If no filters exist, call hybrid retrieval.
- Metadata preferences that are not hard filters should be used as soft weighting or trace-only fields until a real weighting implementation exists.

Important: soft filters must not remove all candidates by default.

### Fusion

Fusion is the combination of vector, BM25, and metadata signals. It is separate from rerank.

Stage 03 can initially delegate fusion to existing low-level retrieval behavior if that behavior already fuses vector and BM25. The executor must still name the step in trace data.

### Evidence Quality Check

Evidence quality is a retrieval-level decision.

Quality checks should consider:

- no candidates;
- exact dish requested but no selected chunk matches that dish;
- selected chunks contain conflicting dish names for an exact dish request;
- requested content type is absent;
- candidate count is below the minimum needed for the answer mode;
- all evidence came from fallback or relaxed filters;
- scores are absent or low when score metadata exists.

Initial acceptable heuristic:

```text
enough if:
  candidate_count > 0
  and exact dish request either has matching dish chunks or no hard dish filter exists
  and selected chunks are not entirely unrelated fallback evidence
```

The exact thresholds can be conservative. The important requirement is that quality is explicit and traceable.

### Optional Fallback

Fallback must not run by default.

Fallback may run only when all are true:

- primary evidence is insufficient;
- `fallback_policy != "disabled"`;
- fallback would not violate an explicit exact-dish request;
- the executor can mark fallback evidence as weaker.

Fallback policies:

| policy | Meaning |
| --- | --- |
| `disabled` | Do not relax retrieval. Return low evidence if primary fails. |
| `relaxed_filters` | Remove or soften non-dish filters such as `content_type`, ingredient, taste, or time. Keep hard `dish_name` constraints. |
| `broad_search` | Search with a broader query only when no hard exact-dish constraint prohibits it. |

Hard-filter rule:

- If `hard_filters` contains `dish_name`, fallback may relax `content_type` or sparse preferences but must keep the dish constraint.
- If the exact dish cannot be found, broad search must return `no_result` unless the query plan explicitly permits substituting similar dishes.
- If fallback returns a different dish under a hard `dish_name` request, Evidence Quality must reject it.

Fallback markers:

```python
doc.metadata["fallback"] = True
doc.metadata["relaxed_filter"] = True
```

If fallback succeeds, `quality.fallback_used=True`.

If fallback still fails, return `low_evidence`.

## Low Evidence Contract

Low-evidence handling is a result-producing path, not a boolean decision.

Required shape:

```python
{
    "answer_type": "no_result | low_confidence | need_clarification",
    "answer": "...",
    "state_diff_policy": "low_evidence",
    "quality_reason": "..."
}
```

Rules:

- `no_result` is used when no reliable evidence exists.
- `low_confidence` is used when weak evidence exists and a conservative answer is acceptable.
- `need_clarification` is used when the user can choose between plausible retrieval targets.
- Low-evidence paths must not update business entities such as `current_dish`.
- The main chain should pass `low_evidence.answer` directly to writeback/generation handling without forcing normal answer generation.
- The execution result passed to writeback must include `answer_type` equal to `no_result`, `low_confidence`, or `need_clarification`, and `state_diff_policy="low_evidence"`.

## Main Chain Cutover

After Stage 03:

```text
main.py
  builds or normalizes QueryPlan
  calls RetrievalExecutor.execute(query_plan)
  consumes RetrievalResult
```

Illegal after cutover:

- the chat path in `main.py` choosing between `metadata_filtered_search()` and `hybrid_search()`;
- the chat path in `main.py` relaxing `content_type` or other filters as fallback;
- the chat path in `main.py` treating `not relevant_chunks` as the only evidence-quality signal;
- the chat generation path deciding that empty retrieval means normal answer generation.

Allowed after cutover:

- `RetrievalOptimizationModule` may remain as the low-level provider for vector/BM25/filter operations.
- Existing query planning can remain in `main.py` for this stage, as long as retrieval execution itself is delegated to the executor.
- Parent-document expansion remains outside Stage 03 and can still be called after `RetrievalResult.chunks`.
- Auxiliary non-chat helpers may keep direct low-level retrieval temporarily only if they are still called and are explicitly listed outside the Stage 03 chat-path cutover. They must not be used by `ask_question()`.

## Observability

Each `RetrievalResult` should expose enough trace data to answer:

- what query was executed;
- which filters were hard vs soft;
- whether fallback was allowed;
- whether fallback was used;
- why evidence was accepted or rejected;
- which dishes were selected;
- whether selected chunks were exact, relaxed, or fallback.

Minimum trace:

```python
{
    "query": "...",
    "original_query": "...",
    "filters": {},
    "hard_filters": [],
    "soft_filters": [],
    "fallback_policy": "disabled",
    "primary_count": 0,
    "fallback_count": 0,
    "selected_dishes": [],
    "quality_reason": "...",
}
```

## Migration Strategy

Recommended migration sequence:

1. Add `QueryPlan`/`RetrievalResult` contract tests.
2. Add query-plan normalization tests.
3. Add `RetrievalExecutor` with primary retrieval behavior that preserves current successful searches.
4. Add evidence quality checks.
5. Add controlled fallback behavior with explicit fallback markers.
6. Add low-evidence/no-result output.
7. Cut `main.py` chat-path retrieval calls to `RetrievalExecutor`.
8. Delete or narrow old retrieval orchestration in `main.py`.
9. Update tests so retrieval acceptance checks the new result contract, not only raw chunk lists.

## Out Of Scope

- Parent expansion.
- Section selection.
- Context trimming.
- Context pack.
- New answer modes.
- Streaming lifecycle.
- State version conflict handling.
- LLM judge reranking.
- Offline index build changes.

## Deliverable

A retrieval layer where:

- production retrieval enters through `RetrievalExecutor`;
- retrieval result includes chunks, quality, low-evidence result, and trace;
- fallback is controlled by policy and never runs silently;
- exact missing dish does not broad-search into another dish unless explicitly allowed;
- sparse metadata does not hard-kill recall;
- main chain no longer owns retrieval fallback rules.

## Acceptance

Stage 03 is accepted when tests prove:

- exact missing dish does not silently broad-search into another dish;
- sparse metadata does not hard-kill all recall;
- fallback only triggers when allowed;
- fallback results are marked with `fallback=true` and `relaxed_filter=true`;
- low-evidence and no-result paths return explicit answer types;
- `ask_question()` no longer chooses between `metadata_filtered_search()` and `hybrid_search()`;
- `ask_question()` no longer contains fallback filter relaxation logic;
- retrieval-facing `QueryPlan` normalization is tested separately from upstream intent routing;
- existing list/detail recipe queries still reach generation when retrieval evidence is sufficient.
