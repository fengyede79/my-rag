# Runtime Versioning And Streaming Design

Status: draft design
Owner: C8 recipe RAG system
Date: 2026-07-07

## Purpose

Stage 05 adds runtime consistency to the single-process recipe RAG chain.

The system has async-like risks: users can send rapid consecutive messages, streaming responses can stop mid-answer, and browsers can disconnect while the backend is still producing chunks. Stage 05 does not introduce a full async runtime to solve these risks. Instead, it adds a small runtime contract so these cases are controlled, observable, and unable to silently pollute business state.

The strongest requirement is stale-generation prevention:

```text
For state-dependent turns, pre-commit version check is not enough.
The runtime must not generate an answer from a stale snapshot, stale resolution, stale plan, or stale context pack.
```

## Design Position

This stage explicitly chooses:

```text
No full async architecture.
Single-process runtime is enough.
Use lightweight runtime control with async awareness.
```

Do not implement:

- `async`/`await` across the RAG chain;
- session queues;
- background workers;
- distributed locks;
- message queues;
- resumable stream jobs;
- exactly-once delivery;
- cross-process transactions.

Implement:

- turn-level runtime context;
- state version reads and checks;
- shared replan budget;
- pre-generation stale-state blocking;
- pre-commit conflict detection;
- streaming lifecycle;
- trace events that explain rapid-message conflicts, stream aborts, and client disconnect-like cases.

## Runtime Context Contract

Every request should create one turn runtime context at entry:

```python
{
    "turn_id": "uuid",
    "trace_id": "uuid-or-short-id",
    "session_id": "default",
    "read_state_version": 12,
    "max_replan_count": 1,
    "replan_count": 0,
    "lifecycle": "started",
    "trace_events": []
}
```

Rules:

- `turn_id` identifies this user turn.
- `trace_id` connects diagnostics across safety, understanding, retrieval, generation, and writeback.
- `read_state_version` is captured from the lightweight session snapshot.
- All version mismatches in this turn share the same `replan_count`.
- `max_replan_count` should be small. Stage 05 should use `1` unless tests show a specific need for `2`.
- Retry budget is turn-level, not per-node.

## State Version Contract

The session state must expose a monotonically increasing `state_version`.

Stage 05 may keep existing physical state fields, but it must provide these operations conceptually:

```python
read_state_version(session_id) -> int
check_state_version(session_id, expected_version) -> VersionCheckResult
commit_state_diff(session_id, state_diff, expected_version) -> CommitResult
```

Recommended result shape:

```python
{
    "matched": False,
    "expected_version": 12,
    "current_version": 13,
    "reason": "state_version_mismatch"
}
```

Commit rule:

- successful commit increments `state_version` exactly once;
- failed commit does not partially apply `state_diff`;
- conflict returns a controlled conflict result.

This stage does not require database transactions. Existing in-process locks in `ConversationManager` are sufficient for single-process consistency.

## State Dependency Rule

Not every turn needs the same checks.

State-dependent turns include:

- ordinal reference: `第一个怎么做`, `那第二个呢`;
- pronoun/current-dish reference: `这个能不放辣吗`, `它怎么做`;
- correction: `不是这个`;
- clarification response;
- history answer;
- any turn whose answer depends on `current_dish`, `last_recommendation_list`, `pending_clarification`, or previous assistant output.

State-independent turns include:

- pure smalltalk that does not inspect business state;
- harmless out-of-domain rejection;
- explicit single-turn recipe query with a concrete dish and no reference dependency.

Rules:

- State-dependent turns must pass a version check before generation.
- Pure state-independent direct answers may skip pre-generation checks.
- Pre-commit checks still apply to all turns that write history or state.

## Version Checkpoints

Stage 05 version checks are:

```text
read snapshot
-> reference resolution if needed
-> version check before planning when resolution used state
-> execution/query/retrieval/context pack
-> version check before generation when answer depends on state
-> generation
-> StateUpdatePolicy builds state_diff
-> pre-commit version check
-> commit or conflict
```

Minimum required checkpoints:

1. **Post-resolution / pre-plan check**
   - Required if reference resolution used `current_dish`, `last_recommendation_list`, or `pending_clarification`.
   - If mismatch, re-read snapshot and re-run understanding/resolution.

2. **Pre-generation check**
   - Required for every state-dependent turn.
   - This is the non-negotiable stale-generation barrier.
   - If mismatch, do not generate from the old resolution, execution plan, query plan, retrieval result, or context pack.

3. **Pre-commit check**
   - Required before applying `state_diff`.
   - If mismatch, do not partially apply state updates.

## Stale Generation Rule

For any state-dependent turn:

```text
version mismatch before generation is blocking
```

The runtime must do one of:

- consume shared replan budget, re-read state, and restart the turn from snapshot/understanding;
- return conflict handling if retry budget is exhausted.

The runtime must not:

- keep old reference resolution;
- keep old execution plan;
- keep old query plan;
- keep old context pack;
- generate an answer and rely on pre-commit check to prevent state pollution.

Bad behavior:

```text
read state v1
resolve "第一个" -> 宫保鸡丁
another turn commits state v2
generate answer about 宫保鸡丁 from old context
pre-commit detects mismatch and skips write
return stale answer to user
```

Correct behavior:

```text
read state v1
resolve "第一个" -> 宫保鸡丁
another turn commits state v2
pre-generation check detects mismatch
re-read state and re-run understanding/resolution
or return conflict if retry budget is exhausted
```

## Shared Replan Budget

All version-check mismatches consume the same turn-level budget.

Recommended defaults:

```python
max_replan_count = 1
```

Rules:

- Do not give VC0, VC1, VC2, and pre-commit independent retry counters.
- A turn may re-enter the chain at most `max_replan_count` times.
- Retry must restart from state snapshot, not from a later partial node.
- If budget is exhausted, return a conflict response.

Conflict response behavior:

- no additional business retrieval after conflict is detected;
- no business answer generated from stale context;
- no business state update;
- may append a conservative history entry;
- should ask the user to restate or confirm the target.

Example response:

```text
上下文刚刚更新了，我需要你再确认一下是指哪一道菜。
```

## Rapid Consecutive Messages

Stage 05 does not implement a session queue.

Risk:

```text
turn A reads state_version=3
turn B completes first and commits state_version=4
turn A reaches generation or commit with expected version=3
```

Required behavior:

- detect `state_version_mismatch`;
- treat it as `concurrent_turn_or_rapid_followup`;
- replan once if budget remains;
- otherwise return conflict;
- do not silently generate from stale state.

Trace event:

```python
{
    "event": "state_version_mismatch",
    "reason": "concurrent_turn_or_rapid_followup",
    "turn_id": "...",
    "trace_id": "...",
    "read_state_version": 3,
    "current_state_version": 4,
    "replan_count": 1
}
```

The runtime does not need to distinguish double-click, two browser tabs, or rapid follow-up. The reliable observable cause is version mismatch.

## Streaming Lifecycle

Streaming must branch before generation. It should not be modeled as "generate full answer, then decide to stream".

Lifecycle:

```text
started
-> retrieval_done
-> streaming
-> completed
```

Abort:

```text
started
-> retrieval_done
-> streaming
-> aborted
```

Failure:

```text
started
-> failed
```

Failure can be reached from any lifecycle point. The simplified path above means a failed turn ends in `failed`; it does not require every failure to happen before retrieval.

Lifecycle fields should include:

```python
{
    "turn_id": "...",
    "trace_id": "...",
    "status": "started | retrieval_done | streaming | completed | aborted | failed",
    "partial_answer_length": 0,
    "commit_business_state": False,
    "reason": None
}
```

Rules:

- `started` may be written when the turn begins.
- `retrieval_done` may be recorded after retrieval/context pack succeeds.
- `streaming` starts before yielding chunks.
- `completed` is set only when the stream is fully consumed.
- `aborted` is set when the stream is not fully consumed.
- `failed` is set on exceptions.

## Stream Abort And Browser Disconnect Boundary

Stage 05 does not need perfect browser-disconnect detection.

Backend-visible cases should be classified as:

```text
client_disconnect_or_stream_not_consumed
```

Required behavior:

- do not treat a partially consumed stream as a completed assistant answer;
- do not update `current_dish` from an aborted stream;
- do not create a valid `last_recommendation_list` from an aborted recommendation stream;
- record trace/lifecycle so the failure is explainable.

Trace event:

```python
{
    "event": "stream_aborted",
    "reason": "client_disconnect_or_stream_not_consumed",
    "turn_id": "...",
    "trace_id": "...",
    "partial_answer_length": 37,
    "commit_business_state": False
}
```

This is not a recovery mechanism. It is an isolation and observability mechanism.

## Completed Stream Commit Rule

Completed streams may commit final state.

Rules:

- accumulate emitted chunks into `assistant_answer`;
- when the stream completes, run pre-commit version check;
- if the version matches, commit through `StateUpdatePolicy`;
- if the version mismatches, return/record conflict and do not apply business state.

`_wrap_stream_with_writeback()` may remain only as a thin adapter if it follows this lifecycle contract. It must not independently decide completed/aborted semantics outside the Stage 05 runtime lifecycle.

## Writeback Contract

Stage 05 does not replace `StateUpdatePolicy`. It adds runtime conditions around it.

Required mapping:

| Runtime outcome | State write behavior |
| --- | --- |
| non-stream completed | build and commit normal `state_diff` after pre-commit check |
| stream completed | build and commit normal `state_diff` after full consumption and pre-commit check |
| stream aborted | use `answer_type=stream_aborted`; no business entity or recommendation update |
| stream failed | no business entity or recommendation update |
| conflict before generation | no business answer; no business state update |
| conflict at pre-commit | no partial state update |

## Observability

Every turn should be explainable from trace data.

Minimum trace fields:

```python
{
    "turn_id": "...",
    "trace_id": "...",
    "read_state_version": 12,
    "current_state_version": 13,
    "replan_count": 1,
    "version_check_points": [],
    "lifecycle": "completed",
    "runtime_events": []
}
```

Useful runtime events:

- `turn_started`;
- `state_version_read`;
- `state_version_match`;
- `state_version_mismatch`;
- `replan_started`;
- `replan_exhausted`;
- `stream_started`;
- `stream_completed`;
- `stream_aborted`;
- `stream_failed`;
- `pre_commit_conflict`.

## Cutover Contract

Old responsibility being replaced:

- stream writeback timing that only defers writeback until generator exhaustion;
- implicit assumptions that pre-commit conflict is enough;
- any per-node retry behavior;
- any stream-abort behavior that can still write complete business state.

New ownership:

- a turn runtime context owns `turn_id`, `trace_id`, `read_state_version`, lifecycle, and shared retry budget;
- state-version checks own stale-generation blocking;
- stream lifecycle owns completed/aborted/failed classification;
- `StateUpdatePolicy` remains the only business state mutation policy.

Illegal after cutover:

- generating a state-dependent answer after a pre-generation version mismatch;
- retrying independently at multiple version-check nodes;
- committing business state from an aborted stream;
- treating browser disconnect / incomplete stream consumption as a completed answer;
- writing complete recommendation state before stream completion;
- leaving `_wrap_stream_with_writeback()` as an independent lifecycle owner.

## Test Fixture Rule

Stage 05 tests must model runtime facts explicitly.

Test fixtures should include:

- a session with a controllable `state_version`;
- a way to simulate a second turn committing between snapshot read and generation;
- a stream generator that can be fully consumed;
- a stream generator that can be partially consumed;
- fake `StateUpdatePolicy` or writeback hooks that prove business state was or was not updated;
- trace assertions for lifecycle and version mismatch reasons.

Tests should not rely on timing sleeps to simulate races. They should use deterministic hooks.

## Acceptance

Stage 05 is accepted when tests prove:

1. A state-dependent turn with a pre-generation version mismatch does not generate from stale context.
2. A version mismatch consumes a shared turn-level replan budget.
3. Retry exhaustion returns conflict handling without retrieval/generation from stale state.
4. Pre-commit version mismatch does not partially apply `state_diff`.
5. Completed non-stream turns still commit through `StateUpdatePolicy`.
6. Completed streams commit final answer and allowed business state after full consumption.
7. Aborted streams record lifecycle but do not update `current_dish` or valid recommendation state.
8. Client disconnect-like stream interruption is traceable as `client_disconnect_or_stream_not_consumed`.
9. Pure smalltalk can skip pre-generation version checks but still writes safely.

## Non-Goals

Stage 05 does not implement:

- full async runtime;
- request queues;
- stream resume;
- job cancellation API;
- background indexing;
- distributed concurrency;
- persistent event store.

Those are production-platform concerns. The current project needs a single-process runtime contract with strong stale-generation prevention and clear stream lifecycle semantics.
