# 05 Runtime Versioning And Streaming

Status: expanded baseline

## Purpose

Add runtime consistency around state-dependent generation and streaming without turning the project into a full async agent runtime.

The C8 recipe RAG can face async-like risks:

- users send consecutive messages quickly;
- streaming output is interrupted;
- browsers disconnect before the stream completes;
- a turn generates from a session snapshot that has already become stale.

Stage 05 does not solve these by introducing a queue, worker runtime, or async framework. It solves them with single-process runtime control: turn identifiers, state-version checks, one shared replan budget, streaming lifecycle, and conflict handling.

## Boundary

Decision:

```text
No full async architecture.
Single-process runtime is enough.
Use lightweight runtime control with async awareness.
```

Do not add:

- full-chain `async`/`await`;
- session queues;
- background workers;
- distributed locks;
- message queues;
- resumable stream jobs;
- exactly-once delivery.

Add:

- `turn_id`;
- `trace_id`;
- `read_state_version`;
- shared `max_replan_count`;
- pre-generation state checks;
- pre-commit state checks;
- streaming lifecycle states;
- trace events explaining conflict, abort, and failure.

## First Thing To Do

Introduce a turn runtime context:

```python
{
    "turn_id": "...",
    "trace_id": "...",
    "session_id": "...",
    "read_state_version": 0,
    "max_replan_count": 1,
    "replan_count": 0,
    "lifecycle": "started",
    "trace_events": []
}
```

All version-check mismatches in a turn must consume the same retry budget.

## State Version Rule

The session state must expose a monotonic `state_version`.

Conceptual operations:

```python
read_state_version(session_id) -> int
check_state_version(session_id, expected_version) -> result
commit_state_diff(session_id, state_diff, expected_version) -> result
```

Rules:

- successful state commit increments `state_version`;
- failed pre-commit checks do not partially apply state;
- existing in-process locks are enough for this stage;
- no database transaction system is required.

## Stale Generation Rule

This is the non-negotiable Stage 05 rule:

```text
For any state-dependent turn, a version mismatch before generation is blocking.
```

The runtime must not generate from:

- stale reference resolution;
- stale execution plan;
- stale query plan;
- stale retrieval result;
- stale context pack.

If mismatch occurs before generation:

1. consume shared replan budget;
2. re-read state and restart from snapshot/turn understanding;
3. if budget is exhausted, return conflict handling.

Pre-commit checks are still required, but they are not enough. A system that returns a stale answer and merely avoids state pollution at commit time is still wrong.

## Version Checkpoints

Required checks:

1. After reference resolution and before planning, when reference resolution used session state.
2. Before generation for every state-dependent turn.
3. Before committing the state diff.

Pure state-independent smalltalk may skip the pre-generation check. Any turn that depends on `current_dish`, `last_recommendation_list`, `pending_clarification`, or prior assistant output must not skip it.

## Shared Retry Budget

Use one turn-level budget:

```python
max_replan_count = 1
```

Rules:

- VC0, VC1, VC2, and pre-commit do not get separate counters.
- Replan restarts from session snapshot, not from the failed node.
- Retry exhaustion returns a conflict response.

Conflict response:

```text
上下文刚刚更新了，我需要你再确认一下是指哪一道菜。
```

Conflict handling must not run additional business retrieval after the conflict is detected, generate a business answer from stale context, or update business state.

## Rapid Consecutive Messages

No session queue is required.

Expected behavior:

```text
turn A reads state_version=3
turn B commits state_version=4
turn A detects mismatch before generation or commit
-> replan once or return conflict
```

Trace reason:

```python
{
    "event": "state_version_mismatch",
    "reason": "concurrent_turn_or_rapid_followup",
    "read_state_version": 3,
    "current_state_version": 4
}
```

The runtime does not need to distinguish double-click, two tabs, or rapid follow-up. The reliable cause is version mismatch.

## Streaming Lifecycle

Streaming must branch before generation.

Completed:

```text
started -> retrieval_done -> streaming -> completed
```

Aborted:

```text
started -> retrieval_done -> streaming -> aborted
```

Failed:

```text
started -> failed
```

Failure can be reached from any lifecycle point. The simplified path above means a failed turn ends in `failed`; it does not require every failure to happen before retrieval.

Rules:

- `completed` means the stream was fully consumed.
- `aborted` means the stream was not fully consumed.
- `failed` means generation or streaming raised an exception.
- completed streams may commit final state after pre-commit version check;
- aborted streams must not commit complete business state;
- failed streams must not pollute reference resolution or business state.

## Browser Disconnect And Stream Abort

The backend may not know whether a user clicked stop, closed a tab, lost network, or the client stopped consuming.

Use the observable reason:

```text
client_disconnect_or_stream_not_consumed
```

Trace event:

```python
{
    "event": "stream_aborted",
    "reason": "client_disconnect_or_stream_not_consumed",
    "partial_answer_length": 37,
    "commit_business_state": False
}
```

This is not stream recovery. It is isolation and observability.

## Writeback Rule

Stage 05 wraps Stage 01 `StateUpdatePolicy`; it does not replace it.

| Runtime result | Write behavior |
| --- | --- |
| non-stream completed | commit allowed `state_diff` after pre-commit check |
| stream completed | commit allowed `state_diff` after full stream consumption and pre-commit check |
| stream aborted | `answer_type=stream_aborted`; no business entity or recommendation update |
| stream failed | no business entity or recommendation update |
| pre-generation conflict | no stale business answer; no business state update |
| pre-commit conflict | no partial state update |

## Cutover Rule

Old behavior being replaced:

- stream writeback that only defers until generator exhaustion;
- relying on pre-commit conflict as the only protection;
- implicit or per-node retry behavior;
- stream abort paths that can still write complete business state.

Illegal after cutover:

- generating a state-dependent answer after pre-generation version mismatch;
- committing `current_dish` or `last_recommendation_list` from aborted streams;
- treating incomplete stream consumption as completed;
- letting `_wrap_stream_with_writeback()` independently own lifecycle semantics.

## Acceptance

This stage is accepted when tests prove:

- pre-generation version mismatch blocks stale state-dependent generation;
- version mismatch uses one shared retry budget;
- retry exhaustion reaches conflict handling;
- pre-commit conflict does not partially apply state;
- completed non-stream turns still commit through `StateUpdatePolicy`;
- completed streams commit final answer and allowed state;
- aborted streams do not create valid recommendation lists or current-dish updates;
- client disconnect-like interruption is traceable as `client_disconnect_or_stream_not_consumed`;
- pure smalltalk can skip pre-generation version checks while still writing safely.

## Out Of Scope

- External queues.
- Distributed locks.
- Multi-worker transaction design.
- Full async runtime.
- Stream resume.
- Job cancellation API.
- Complex LLM judge reranking.
