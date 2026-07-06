# 01 State Contract And Writeback Policy

Status: accepted baseline

## Purpose

Make state writes explicit, typed, and safe before changing the main request chain. This stage reduces state pollution risk while preserving existing behavior as much as possible.

Stage 01 is not a state-system rewrite. It is a contract layer over the current writeback behavior. The goal is to make later pipeline changes safer by ensuring every turn has an `answer_type`, every state mutation is represented as a `state_diff`, and each `answer_type` can only modify approved fields.

Stage 01 must cut the production writeback path to the new policy. After the new path passes tests, obsolete old-path branches, modules, tests, and helpers that are no longer called must be deleted. Do not leave unused compatibility code behind.

## Current Starting Point

Current writeback already has a useful review layer:

```text
RecipeRAGSystem._write_conversation_turn()
  -> ConversationManager.writeback_turn_state()
       -> state_writeback_review.review_state_writeback()
       -> writeback_mode switch
```

Current writeback modes:

```text
message_only
clarification_pending
recommendation_list
resolved_followup
correction_turn
explicit_single_dish
normal
```

Current session fields:

```python
{
    "current_entity": None,
    "current_intent": "general",
    "topic_mode": "none",
    "recent_recommendations": [],
    "recent_topics": [],
    "last_confirmed_target": None,
    "current_entity_meta": {},
    "pending_clarification": None,
    "messages": []
}
```

Existing tests already cover several safety behaviors:

- message-only writeback does not replace `current_entity`;
- clarification writeback does not set `current_entity`;
- recommendation writeback preserves existing `current_entity`;
- no-result turns do not replace `current_entity`;
- stream-interrupted writeback does not replace `current_entity`.

Stage 01 should build on this instead of discarding it.

## First Thing To Do

Introduce a single policy surface that translates current execution facts into a state diff:

```python
build_state_diff(answer_type, execution_result, old_state) -> dict
```

The policy should be testable without retrieval, generation, Flask, or LLM calls.

No generation or retrieval function should directly mutate business state after this stage. They can produce execution facts; the policy decides state changes.

## Scope

Stage 01 focuses on state write contracts only:

- define canonical target fields;
- map current fields to canonical fields;
- define `answer_type`;
- define `state_diff`;
- define field whitelist rules;
- adapt current writeback modes to the new policy surface;
- add tests that prove unsafe state writes are blocked.

Stage 01 should preserve current user-visible behavior where possible.

## Canonical State Contract

The frozen architecture uses these canonical fields:

```python
{
    "state_version": 0,
    "current_dish": None,
    "current_entities": [],
    "last_recommendation_list": [],
    "last_answer_type": None,
    "pending_clarification": None,
    "resolved_references": [],
    "turn_lifecycle": {},
    "history": []
}
```

During migration, existing field names can remain physically present. Stage 01 does not require a breaking rename of `SessionState`.

Mapping for Stage 01:

| Canonical field | Current field or source | Stage 01 expectation |
| --- | --- | --- |
| `current_dish` | `current_entity` | Keep current storage, expose policy semantics as `current_dish`. |
| `current_entities` | not first-class yet | Only introduce if needed for comparison tests; otherwise keep as future-compatible diff key. |
| `last_recommendation_list` | `recent_recommendations` | Keep current storage shape with ranked dish dicts. |
| `last_answer_type` | new lightweight field | Stage 01 should add this explicitly. Do not reuse `current_intent`; it has different semantics. |
| `pending_clarification` | `pending_clarification` | Keep current field. |
| `resolved_references` | resolution/execution trace | Can remain trace-only until Stage 02. |
| `turn_lifecycle` | stream interruption facts | Can remain trace-only until Stage 05. |
| `history` | `messages` | Keep current message list. |
| `state_version` | none | Do not implement version checks in Stage 01; leave to Stage 05 unless adding a passive field is cheap and tested. |

## Answer Type Contract

Stage 01 should introduce or document these `answer_type` values:

```text
smalltalk
domain_reject
clarification
recommendation
detail
comparison
history_answer
low_confidence
no_result
stream_aborted
normal
```

Current writeback modes should map to answer types:

| Current writeback mode | Stage 01 answer_type |
| --- | --- |
| `message_only` with smalltalk/front-door/out-of-domain turn | `smalltalk` or `domain_reject` |
| `message_only` with failed retrieval | `no_result` |
| `message_only` with interrupted stream | `stream_aborted` |
| `clarification_pending` | `clarification` |
| `recommendation_list` | `recommendation` |
| `resolved_followup` | `detail` |
| `correction_turn` | `detail` |
| `explicit_single_dish` | `detail` |
| `normal` | `normal` |

If a turn cannot be classified, default to `normal` with conservative state writes.

When current mode is `message_only`, classify `answer_type` using this priority:

```text
1. execution_result.stream_interrupted is true -> stream_aborted
2. execution_result.success is false -> no_result
3. turn_info.turn_type is front_door_blocked or out_of_domain -> domain_reject
4. turn_info.turn_type is smalltalk or front_door_direct_reply -> smalltalk
5. otherwise -> normal
```

This priority prevents a failed retrieval or interrupted stream from being treated as ordinary smalltalk.

After cutover, classification should be based directly on execution facts rather than old `writeback_mode` values. Resolution actions must preserve the old successful follow-up semantics:

```text
resolution.next_action == apply_reference_resolution with resolved_target -> detail
resolution.next_action == apply_correction with resolved_target -> detail
```

The detail target source must fall back from `execution_result.resolved_target` to `resolution.resolved_target`.

## State Diff Contract

A state diff is a structured description of intended mutations. It should be created before applying changes.

Recommended shape:

```python
{
    "answer_type": "detail",
    "allowed_fields": ["current_dish", "last_answer_type", "history"],
    "updates": {
        "current_dish": {
            "value": "宫保鸡丁",
            "source": "explicit_query",
            "confidence": 1.0
        },
        "last_answer_type": "detail"
    },
    "clear": ["pending_clarification"],
    "append_history": True,
    "reason": "explicit_single_dish"
}
```

The implementation can use a smaller structure if tests prove the same guarantees. The important requirement is that the policy can be inspected before state mutation.

## Field Whitelist

| answer_type | Allowed business state updates |
| --- | --- |
| `smalltalk` | No business state updates. May append history and set `last_answer_type`. |
| `domain_reject` | No business state updates. May append history and set `last_answer_type`. |
| `clarification` | May set `pending_clarification`; may append history; may set `last_answer_type`. Must not set `current_dish`. |
| `recommendation` | May set `last_recommendation_list`; may clear stale pending clarification; may append history; may set `last_answer_type`. Must not overwrite `current_dish`. |
| `detail` | May set `current_dish` only when target evidence is strong. May clear pending clarification. May append history and set `last_answer_type`. |
| `comparison` | May set `current_entities`; may append history and set `last_answer_type`. |
| `history_answer` | No entity updates by default. May append history and set `last_answer_type`. |
| `low_confidence` | No business entity updates. May append history and set `last_answer_type`. |
| `no_result` | No business entity updates. May append history and set `last_answer_type`. |
| `stream_aborted` | May append history and set `last_answer_type`; must not update business entities. Full stream lifecycle and complete-answer handling are deferred to Stage 05. |
| `normal` | Conservative fallback. May append history and set `last_answer_type`. Must not update business entities. |

## Strong Target Evidence

`detail` may update `current_dish` only when at least one of these is true:

- current writeback mode is `explicit_single_dish` and `query_plan.dish_name` exists;
- current writeback mode is `resolved_followup` and `execution_result.resolved_target` exists;
- current writeback mode is `correction_turn` and corrected target exists;
- later stages provide an equivalent high-confidence resolution result.

Failed retrieval, low-confidence evidence, no-result, interrupted stream, or message-only paths must not update `current_dish`.

When `current_dish` is updated, the history entry should carry `entities={"dish_name": current_dish}` so existing history compression and follow-up context quality do not regress.

## Recommended Implementation Boundary

Stage 01 should avoid scattering policy across `ConversationManager`.

Preferred module boundary:

```text
code/C8/rag_modules/state_update_policy.py
```

Suggested public functions:

```python
def classify_answer_type(review: dict, turn_info: dict, execution_result: dict, query_plan: dict | None, resolution: dict | None) -> str:
    ...

def build_state_diff(answer_type: str, execution_result: dict, old_state, *, query_plan: dict | None = None, resolution: dict | None = None, answer: str = "", question: str = "") -> dict:
    ...
```

`ConversationManager.writeback_turn_state()` should call the new policy directly. It should not keep `review_state_writeback()` as a production adapter after cutover.

`ConversationManager` owns mutation through a manager method such as:

```python
def apply_state_diff(self, session_id: str, state_diff: dict) -> None:
    ...
```

The policy module owns classification and diff construction. The manager owns mutation because it already owns session locking and existing mutation helpers.

Existing mutation helpers such as `set_current_dish()`, `record_recommendations()`, `set_pending_clarification()`, and `clear_pending_clarification()` can remain. In production writeback paths, they should be called by `apply_state_diff()` or an equivalent policy/apply layer, not directly by unrelated generation or retrieval code. Tests may still call them directly for setup.

If `state_writeback_review.py` becomes uncalled after the cutover, delete it and migrate its tests to `state_update_policy.py`. The final Stage 01 implementation should not retain unused old writeback nodes.

## Out Of Scope

- Reordering front-door guardrail.
- Changing `ask_question()` pipeline order.
- Building Retrieval Executor.
- Adding fallback retrieval.
- Implementing evidence quality checks.
- Implementing optimistic concurrency or version retry behavior.
- Implementing full streaming lifecycle.
- Renaming all `SessionState` fields in one migration.

## Deliverable

A state contract and writeback policy that can be tested independently from retrieval and generation.

Expected deliverables:

- expanded Stage 01 spec;
- implementation plan for Stage 01;
- policy module or equivalent centralized policy surface;
- tests for answer-type classification, field whitelist, and state-diff application;
- compatibility tests proving current safe behaviors still hold.

## Acceptance

Stage 01 is accepted when tests prove:

- smalltalk does not clear or overwrite `current_dish` / `current_entity`;
- domain rejection does not update business state;
- clarification only writes pending clarification state and does not set current dish;
- recommendation updates recommendation list without overwriting current dish;
- detail updates current dish only with strong target evidence;
- failed retrieval maps to `no_result` and does not update business entities;
- low-confidence paths, if introduced in this stage, do not update business entities;
- stream-aborted/interrupted paths do not update business entities;
- all reliable business state writes pass through the centralized policy surface.
