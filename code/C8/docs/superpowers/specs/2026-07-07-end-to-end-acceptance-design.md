# End-To-End Acceptance Design

Status: draft spec
Owner: C8 recipe RAG system
Date: 2026-07-07

## Purpose

Stage 06 proves that the staged migration has reached the frozen runtime architecture through behavior, traces, and cleanup.

This stage must not introduce a new architecture layer. It is the acceptance gate for the architecture that was already defined by Stages 00-05:

```text
basic safety
-> session snapshot + state_version
-> turn understanding
-> reference resolution
-> execution plan
-> query plan
-> retrieval executor + evidence quality
-> context pack
-> answer generation or stream lifecycle
-> StateUpdatePolicy
-> versioned state commit
```

The goal is to show that the new chain is active end to end and that obsolete old paths are not still carrying production behavior.

## Non-Goals

Stage 06 does not add:

- a new planner;
- full async execution;
- distributed locking;
- LLM judge reranking;
- a larger answer-mode taxonomy;
- new product features outside the frozen recipe RAG scope.

If an acceptance test fails, the fix should normally go back to the owner stage:

| Failing area | Fix belongs to |
| --- | --- |
| state pollution or wrong writeback | Stage 01 |
| wrong turn/domain/reference decision | Stage 02 |
| retrieval/fallback/quality issue | Stage 03 |
| parent expansion or context trimming issue | Stage 04 |
| stale state, stream abort, version conflict | Stage 05 |

Stage 06 may contain glue fixes, fixture fixes, cutover tests, stale-path deletion, and acceptance reports. It should not create a parallel replacement architecture.

## Required Acceptance Artifacts

Stage 06 produces four artifacts:

1. An end-to-end scenario test matrix.
2. A trace assertion layer that proves each scenario used the new runtime chain.
3. A stale-path cleanup checklist and source-level cutover tests.
4. A short acceptance report.

These artifacts should live under the existing project structure:

```text
code/C8/tests/test_end_to_end_acceptance.py
code/C8/tests/test_final_cutover.py
code/C8/docs/architecture/evolution/06-end-to-end-acceptance.md
code/C8/docs/architecture/evolution/acceptance-report.md
```

The exact report filename may include a date if useful, but it should stay near the architecture evolution docs.

## Primary Scenario

The primary scenario is the main proof that multi-turn state, reference resolution, retrieval, answer mode, and writeback stay aligned:

```text
推荐三个鸡肉菜
-> 第一个怎么做
-> 这个能不放辣吗
-> 没有豆瓣酱怎么办
-> 给我换个不辣的
-> 谢谢
```

Required behavior:

- The first turn returns a recommendation list and writes `last_recommendation_list`.
- `第一个怎么做` resolves against the reliable recommendation list and may update `current_dish` after a successful detail answer.
- `这个能不放辣吗` stays attached to `current_dish` or the resolved recipe context.
- `没有豆瓣酱怎么办` uses substitution/troubleshooting behavior without clearing the current dish.
- `给我换个不辣的` uses constraint-aware recommendation behavior and should not corrupt the previous detail context.
- `谢谢` is smalltalk and must not clear business state.

At the end of the scenario:

```python
session.last_recommendation_list  # still valid unless intentionally replaced
session.current_dish              # still the last strong detail dish
session.last_answer_type          # smalltalk is allowed
session.pending_clarification     # None unless the last turn asked one
session.state_version             # incremented only by committed state diffs
```

## Required Additional Scenarios

| Scenario | Required outcome |
| --- | --- |
| `Python 怎么学` | `domain_reject`; no retrieval; no business state update. |
| Recommendation followed by `第一个作者是谁` | Does not silently resolve as recipe detail; rejects or clarifies. |
| Exact missing dish | Uses `no_result` or `low_confidence`; no broad substitution unless fallback policy allows it. |
| Sparse metadata preference query | Metadata is soft weighted; recall is preserved; weak evidence is marked. |
| Stream abort after recommendation | Aborted recommendation is not valid for later ordinal reference. |
| Rapid state-dependent requests | Shared replan budget is consumed once; conflict path is reachable; no infinite loop. |
| Low evidence detail answer | Does not update `current_dish`. |
| Final smalltalk after recipe flow | Does not clear `current_dish` or recommendation state. |

## Trace Contract

Each acceptance scenario must assert enough trace data to prove the new chain was used.

Minimum trace fields:

```python
{
    "turn_id": "...",
    "trace_id": "...",
    "read_state_version": 0,
    "action": "retrieve_detail",
    "answer_mode": "recipe_detail",
    "resolution": {
        "resolved_target": "宫保鸡丁",
        "confidence": 0.9,
        "evidence_source": "last_recommendation_list[0]",
    },
    "query_plan": {},
    "retrieval_quality": {},
    "context_pack_trace": {},
    "answer_type": "detail",
    "state_diff": {},
    "commit_result": {},
    "lifecycle": {},
}
```

The tests do not need to assert every token of generated text. They should assert:

- control-flow action;
- answer type or answer mode;
- whether retrieval ran;
- whether fallback ran;
- whether context pack exists when retrieval ran;
- whether state diff wrote only allowed fields;
- whether commit succeeded or conflict/abort was recorded.

## Cutover Requirements

Before Stage 06 is accepted, the following must be true:

- Production code does not call old retrieval branching when `RetrievalExecutor` owns the path.
- Generation helpers do not call `get_parent_documents()` or rebuild context by themselves.
- Generation helpers do not mutate conversation state directly.
- State writes go through `StateUpdatePolicy` and versioned `ConversationManager` commit.
- Stream writeback uses the lifecycle-aware wrapper, not the old stream writeback path.
- Old helper modules may remain only if they are actively called as thin compatibility wrappers with no independent production behavior.

Stage 06 should include source-level tests for these conditions. This is acceptable here because the goal is architecture cutover, not only black-box behavior.

## Fixture Requirements

Acceptance fixtures must resemble the new production chain:

- parent documents should be recipe-shaped Markdown with selectable `##` sections;
- recommendation fixtures should include at least three distinguishable dishes;
- retrieval fixtures should expose enough metadata to test soft weighting and fallback markers;
- stream fixtures should use real generators and explicitly test full consumption vs abort;
- conflict fixtures should mutate state deterministically, not with sleeps or timing assumptions.

Do not bypass the new chain by monkeypatching directly into answer helpers unless the test is explicitly a cutover/source-level test.

## Acceptance Report

The report should be short and factual:

```text
Stage 06 Acceptance Report

Date:
Commit / branch:

Scenario results:
- primary multi-turn chain: pass/fail
- domain reject: pass/fail
- unrelated ordinal: pass/fail
- missing dish: pass/fail
- sparse metadata: pass/fail
- stream abort: pass/fail
- rapid state conflict: pass/fail

Cutover checks:
- old retrieval path removed or inactive:
- old parent expansion in generation removed:
- direct state mutation removed:
- old stream writeback removed:

Known residual risks:
- ...
```

Known residual risks are allowed if they do not contradict the frozen architecture. For example, weak answer quality on a sparse fixture is acceptable; a shadow old retrieval path carrying production behavior is not.

## Final Acceptance Criteria

Stage 06 is accepted when:

- the primary scenario passes;
- every required additional scenario passes or has a documented architecture-consistent reason for failure;
- trace assertions prove the new chain was used;
- stale-path source checks pass;
- no unused old production path remains after the new path covers the behavior;
- the acceptance report is written.

After Stage 06, there should be no Stage 07 by default. Further work should be ordinary bugfixes, answer quality tuning, or product feature specs, not runtime architecture migration.
