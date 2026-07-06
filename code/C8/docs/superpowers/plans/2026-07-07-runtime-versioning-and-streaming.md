# Runtime Versioning And Streaming Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add single-process runtime control for state-version checks, shared replan budget, and streaming lifecycle without introducing a full async runtime.

**Architecture:** Add a thin `TurnRuntimeContext` boundary around `RecipeRAGSystem.ask_question()`. `ConversationManager` owns monotonic `state_version` reads/checks/commits under its existing lock. `ask_question()` uses runtime helpers to block stale state-dependent generation and to classify streaming turns as completed, aborted, or failed before delegating business mutations to `StateUpdatePolicy`.

**Tech Stack:** Python, pytest, existing `RecipeRAGSystem`, existing `ConversationManager`, existing `StateUpdatePolicy`, existing generator-based streaming/SSE behavior.

---

## Cutover Contract

This plan assumes Stage 01 through Stage 04 are implemented.

Old responsibility being replaced:

- `_wrap_stream_with_writeback()` only defers writeback until the generator is fully consumed.
- There is no `state_version` check before state-dependent generation.
- Pre-commit conflict is not modeled as a first-class runtime result.
- Stream abort/failure is not an explicit lifecycle result.

New ownership:

- `rag_modules.turn_runtime` owns `turn_id`, `trace_id`, `read_state_version`, shared replan budget, lifecycle events, and version-check result helpers.
- `ConversationManager` owns locked `state_version` reads, checks, and expected-version commits.
- `RecipeRAGSystem.ask_question()` owns where runtime checks happen in the main chain.
- `StateUpdatePolicy` remains the only business state mutation policy.

Illegal after cutover:

- generating a state-dependent answer after a pre-generation `state_version` mismatch;
- giving each version checkpoint its own retry counter;
- committing business state from an aborted stream;
- treating incomplete stream consumption as completed;
- leaving `_wrap_stream_with_writeback()` as the independent owner of streaming lifecycle semantics;
- using sleeps/timing races in tests instead of deterministic version-change hooks.

Deletion/narrowing before acceptance:

- `_wrap_stream_with_writeback()` must be narrowed to a lifecycle-aware adapter or replaced by a lifecycle wrapper.
- Any writeback path that bypasses expected-version commit must be removed from production turn writeback.
- Existing stream writeback tests must be migrated from "writeback after full consumption" to "completed stream commits, aborted stream does not commit business state".

---

## File Structure

- Create `code/C8/rag_modules/turn_runtime.py`
  - Defines `TurnRuntimeContext`.
  - Defines version-check result helpers.
  - Defines lifecycle event helpers.
  - Owns shared replan budget semantics.

- Modify `code/C8/rag_modules/conversation_manager.py`
  - Adds `state_version` and `turn_lifecycle` to `SessionState`.
  - Adds `get_state_version(session_id)`.
  - Adds `check_state_version(session_id, expected_version)`.
  - Adds `commit_state_diff(session_id, state_diff, expected_version, lifecycle=None)`.
  - Ensures state mutation increments `state_version` exactly once per successful commit.

- Modify `code/C8/main.py`
  - Creates a runtime context at request entry.
  - Reads `read_state_version` from `ConversationManager`.
  - Performs post-resolution/pre-plan checks where resolution used state.
  - Performs pre-generation checks for state-dependent turns.
  - Performs pre-commit checks through `ConversationManager.commit_state_diff()`.
  - Replaces/narrows `_wrap_stream_with_writeback()` into lifecycle-aware stream handling.

- Modify `code/C8/rag_modules/state_update_policy.py`
  - Keep answer-type whitelist unchanged unless stream lifecycle facts require a field already allowed by `stream_aborted`.
  - Do not add business state fields for aborted streams.

- Create `code/C8/tests/test_turn_runtime.py`
  - Unit tests for runtime context, replan budget, lifecycle helpers, and version-check result shape.

- Modify `code/C8/tests/test_conversation_state.py`
  - Integration tests for stale-generation blocking, shared replan budget, completed stream commit, and aborted stream isolation.

- Modify `code/C8/tests/test_state_hardening.py`
  - Tests for state-version commit behavior and state-dependent stream safety where existing hardening fixtures exercise streaming.

- Create `code/C8/tests/test_runtime_cutover.py`
  - Source-level tests proving old independent stream lifecycle behavior and stale-generation paths are removed.

---

## Task 1: Add Turn Runtime Context

**Files:**
- Create: `code/C8/rag_modules/turn_runtime.py`
- Create: `code/C8/tests/test_turn_runtime.py`

- [ ] **Step 1: Write runtime context tests**

Create `code/C8/tests/test_turn_runtime.py`:

```python
from rag_modules.turn_runtime import (
    TurnRuntimeContext,
    append_runtime_event,
    build_version_mismatch,
    should_replan_after_mismatch,
)


def test_turn_runtime_context_starts_with_ids_and_shared_budget():
    ctx = TurnRuntimeContext.start(session_id="s1", read_state_version=3)

    assert ctx.session_id == "s1"
    assert ctx.read_state_version == 3
    assert ctx.max_replan_count == 1
    assert ctx.replan_count == 0
    assert ctx.lifecycle["status"] == "started"
    assert ctx.turn_id
    assert ctx.trace_id


def test_runtime_events_are_appended_without_replacing_existing_trace():
    ctx = TurnRuntimeContext.start(session_id="s1", read_state_version=3)

    append_runtime_event(ctx, "state_version_read", current_state_version=3)
    append_runtime_event(ctx, "stream_started", partial_answer_length=0)

    assert [event["event"] for event in ctx.trace_events] == [
        "state_version_read",
        "stream_started",
    ]
    assert ctx.trace_events[0]["turn_id"] == ctx.turn_id
    assert ctx.trace_events[0]["trace_id"] == ctx.trace_id


def test_version_mismatch_result_records_expected_current_and_reason():
    mismatch = build_version_mismatch(
        expected_version=3,
        current_version=4,
        reason="concurrent_turn_or_rapid_followup",
    )

    assert mismatch == {
        "matched": False,
        "expected_version": 3,
        "current_version": 4,
        "reason": "concurrent_turn_or_rapid_followup",
    }


def test_replan_budget_is_shared_for_all_mismatches():
    ctx = TurnRuntimeContext.start(session_id="s1", read_state_version=3, max_replan_count=1)

    assert should_replan_after_mismatch(ctx) is True
    assert ctx.replan_count == 1
    assert should_replan_after_mismatch(ctx) is False
    assert ctx.replan_count == 1
```

- [ ] **Step 2: Run runtime context tests and verify they fail**

Run:

```bash
cd code/C8
pytest tests/test_turn_runtime.py -q
```

Expected:

- FAIL because `rag_modules.turn_runtime` does not exist.

- [ ] **Step 3: Implement runtime context**

Create `code/C8/rag_modules/turn_runtime.py`:

```python
"""Single-process turn runtime context for versioning and streaming lifecycle."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any


def _new_id() -> str:
    return uuid.uuid4().hex


@dataclass
class TurnRuntimeContext:
    session_id: str
    read_state_version: int
    turn_id: str
    trace_id: str
    max_replan_count: int = 1
    replan_count: int = 0
    lifecycle: dict[str, Any] = field(default_factory=dict)
    trace_events: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def start(
        cls,
        *,
        session_id: str,
        read_state_version: int,
        max_replan_count: int = 1,
    ) -> "TurnRuntimeContext":
        ctx = cls(
            session_id=session_id,
            read_state_version=read_state_version,
            turn_id=_new_id(),
            trace_id=_new_id(),
            max_replan_count=max_replan_count,
            lifecycle={
                "status": "started",
                "partial_answer_length": 0,
                "commit_business_state": False,
                "reason": None,
            },
        )
        append_runtime_event(ctx, "turn_started", current_state_version=read_state_version)
        return ctx


def append_runtime_event(ctx: TurnRuntimeContext, event: str, **fields: Any) -> None:
    ctx.trace_events.append(
        {
            "event": event,
            "turn_id": ctx.turn_id,
            "trace_id": ctx.trace_id,
            "timestamp": time.time(),
            **fields,
        }
    )


def build_version_mismatch(
    *,
    expected_version: int,
    current_version: int,
    reason: str = "concurrent_turn_or_rapid_followup",
) -> dict[str, Any]:
    return {
        "matched": False,
        "expected_version": expected_version,
        "current_version": current_version,
        "reason": reason,
    }


def should_replan_after_mismatch(ctx: TurnRuntimeContext) -> bool:
    if ctx.replan_count >= ctx.max_replan_count:
        return False
    ctx.replan_count += 1
    append_runtime_event(ctx, "replan_started", replan_count=ctx.replan_count)
    return True


```

- [ ] **Step 4: Run runtime context tests and verify they pass**

Run:

```bash
cd code/C8
pytest tests/test_turn_runtime.py -q
```

Expected:

- PASS.

- [ ] **Step 5: Commit**

```bash
git add code/C8/rag_modules/turn_runtime.py code/C8/tests/test_turn_runtime.py
git commit -m "feat: add turn runtime context"
```

---

## Task 2: Add Session State Version Operations

**Files:**
- Modify: `code/C8/rag_modules/conversation_manager.py`
- Modify: `code/C8/tests/test_state_hardening.py`

- [ ] **Step 1: Write state-version tests**

Append to `code/C8/tests/test_state_hardening.py`:

```python
def test_session_state_version_starts_at_zero_and_increments_on_commit():
    manager = ConversationManager()

    assert manager.get_state_version("version-session") == 0
    result = manager.commit_state_diff(
        "version-session",
        {
            "answer_type": "smalltalk",
            "updates": {"last_answer_type": "smalltalk"},
            "clear": [],
            "append_history": False,
            "history": None,
        },
        expected_version=0,
    )

    assert result["committed"] is True
    assert result["state_version_before"] == 0
    assert result["state_version_after"] == 1
    assert manager.get_state_version("version-session") == 1


def test_commit_state_diff_rejects_mismatched_expected_version_without_mutation():
    manager = ConversationManager()
    manager.commit_state_diff(
        "conflict-session",
        {
            "answer_type": "smalltalk",
            "updates": {"last_answer_type": "smalltalk"},
            "clear": [],
            "append_history": False,
            "history": None,
        },
        expected_version=0,
    )

    result = manager.commit_state_diff(
        "conflict-session",
        {
            "answer_type": "detail",
            "updates": {
                "last_answer_type": "detail",
                "current_dish": {"value": "蛋炒饭", "source": "test", "confidence": 1.0},
            },
            "clear": [],
            "append_history": False,
            "history": None,
        },
        expected_version=0,
    )

    session = manager.get_session("conflict-session")
    assert result["committed"] is False
    assert result["reason"] == "state_version_mismatch"
    assert result["current_version"] == 1
    assert session.current_entity is None
    assert session.last_answer_type == "smalltalk"
    assert manager.get_state_version("conflict-session") == 1
```

- [ ] **Step 2: Run state-version tests and verify they fail**

Run:

```bash
cd code/C8
pytest tests/test_state_hardening.py::test_session_state_version_starts_at_zero_and_increments_on_commit tests/test_state_hardening.py::test_commit_state_diff_rejects_mismatched_expected_version_without_mutation -q
```

Expected:

- FAIL because `ConversationManager.get_state_version()` and `commit_state_diff()` do not exist.

- [ ] **Step 3: Add version fields and methods**

In `code/C8/rag_modules/conversation_manager.py`, add fields to `SessionState`:

```python
    state_version: int = 0
    turn_lifecycle: Dict[str, Any] = field(default_factory=dict)
```

Add methods to `ConversationManager`:

```python
    def get_state_version(self, session_id: str) -> int:
        with self._lock:
            return self.get_session(session_id).state_version

    def check_state_version(self, session_id: str, expected_version: int) -> dict[str, Any]:
        with self._lock:
            current = self.get_session(session_id).state_version
            if current == expected_version:
                return {
                    "matched": True,
                    "expected_version": expected_version,
                    "current_version": current,
                    "reason": "state_version_match",
                }
            return {
                "matched": False,
                "expected_version": expected_version,
                "current_version": current,
                "reason": "state_version_mismatch",
            }

    def _record_turn_lifecycle(
        self,
        session,
        turn_id: str,
        lifecycle: dict[str, Any],
    ) -> None:
        session.turn_lifecycle[turn_id] = dict(lifecycle)
        while len(session.turn_lifecycle) > 20:
            oldest_turn_id = next(iter(session.turn_lifecycle))
            session.turn_lifecycle.pop(oldest_turn_id, None)

    def commit_state_diff(
        self,
        session_id: str,
        state_diff: dict[str, Any],
        *,
        expected_version: int,
        lifecycle: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            session = self.get_session(session_id)
            before = session.state_version
            if before != expected_version:
                return {
                    "committed": False,
                    "reason": "state_version_mismatch",
                    "expected_version": expected_version,
                    "current_version": before,
                }
            self.apply_state_diff(session_id, state_diff)
            if lifecycle is not None:
                self._record_turn_lifecycle(
                    session,
                    state_diff.get("turn_id", "last_turn"),
                    lifecycle,
                )
            session.state_version += 1
            return {
                "committed": True,
                "state_version_before": before,
                "state_version_after": session.state_version,
            }
```

Important implementation note:

- `commit_state_diff()` must increment version exactly once.
- `apply_state_diff()` must not increment version by itself.
- `turn_lifecycle` should not grow without bound. Keep only the most recent 20 lifecycle entries per session.
- Existing tests that call `writeback_turn_state()` directly should still pass after `writeback_turn_state()` is updated in Task 3.

- [ ] **Step 4: Run state-version tests and verify they pass**

Run:

```bash
cd code/C8
pytest tests/test_state_hardening.py::test_session_state_version_starts_at_zero_and_increments_on_commit tests/test_state_hardening.py::test_commit_state_diff_rejects_mismatched_expected_version_without_mutation -q
```

Expected:

- PASS.

- [ ] **Step 5: Commit**

```bash
git add code/C8/rag_modules/conversation_manager.py code/C8/tests/test_state_hardening.py
git commit -m "feat: add session state version commits"
```

---

## Task 3: Route Writeback Through Expected-Version Commit

**Files:**
- Modify: `code/C8/rag_modules/conversation_manager.py`
- Modify: `code/C8/main.py`
- Modify: `code/C8/tests/test_conversation_state.py`

- [ ] **Step 1: Write pre-commit conflict test**

Append to `code/C8/tests/test_conversation_state.py`:

```python
def test_write_conversation_turn_records_pre_commit_conflict_without_business_update():
    system = RecipeRAGSystem.__new__(RecipeRAGSystem)
    manager = ConversationManager()
    system.generation_module = type("Generation", (), {"conversation_manager": manager})()

    manager.commit_state_diff(
        "precommit-conflict",
        {
            "answer_type": "smalltalk",
            "updates": {"last_answer_type": "smalltalk"},
            "clear": [],
            "append_history": False,
            "history": None,
        },
        expected_version=0,
    )

    system._write_conversation_turn(
        session_id="precommit-conflict",
        question="蛋炒饭怎么做",
        answer="蛋炒饭做法",
        turn_info={"turn_type": "domain_query"},
        query_plan={"route_type": "detail", "dish_name": "蛋炒饭"},
        resolution=None,
        execution_result={
            "success": True,
            "resolved_target": "蛋炒饭",
            "runtime": {"read_state_version": 0, "turn_id": "t1", "trace_id": "x1"},
        },
    )

    session = manager.get_session("precommit-conflict")
    assert session.current_entity is None
    assert session.last_answer_type == "smalltalk"
```

- [ ] **Step 2: Run pre-commit conflict test and verify it fails**

Run:

```bash
cd code/C8
pytest tests/test_conversation_state.py::test_write_conversation_turn_records_pre_commit_conflict_without_business_update -q
```

Expected:

- FAIL because `_write_conversation_turn()` still calls `writeback_turn_state()` without expected-version commit semantics.

- [ ] **Step 3: Update `writeback_turn_state()` to accept expected version**

`ConversationManager.writeback_turn_state(...)` already receives `execution_result` in the current codebase. Keep that parameter in the signature and add the new runtime commit parameters after it:

```python
        execution_result: dict | None = None,
        expected_state_version: int | None = None,
        lifecycle: dict[str, Any] | None = None,
```

Replace:

```python
        self.apply_state_diff(session_id, state_diff)
```

with:

```python
        if expected_state_version is None:
            self.apply_state_diff(session_id, state_diff)
            session = self.get_session(session_id)
            session.state_version += 1
            return {
                "committed": True,
                "state_version_after": session.state_version,
                "legacy_expected_version": None,
            }
        runtime = execution_result.get("runtime", {}) if execution_result else {}
        turn_id = runtime.get("turn_id", "last_turn")
        return self.commit_state_diff(
            session_id,
            {**state_diff, "turn_id": turn_id},
            expected_version=expected_state_version,
            lifecycle=lifecycle,
        )
```

This preserves direct unit tests that call `writeback_turn_state()` without expected version while letting production paths use checked commit.

Do not introduce a second source of runtime data here. The `turn_id` must come from `execution_result["runtime"]` when production code calls this method; if tests call the method without `execution_result`, use a local fallback:

```python
        runtime = execution_result.get("runtime", {}) if execution_result else {}
        turn_id = runtime.get("turn_id", "last_turn")
```

- [ ] **Step 4: Update `_write_conversation_turn()` to pass expected version**

In `RecipeRAGSystem._write_conversation_turn()`, extract runtime info:

```python
        runtime = execution_result.get("runtime", {}) if execution_result else {}
        expected_state_version = runtime.get("read_state_version")
        lifecycle = runtime.get("lifecycle")
```

Change the manager call to:

```python
        return conversation_manager.writeback_turn_state(
            session_id=session_id,
            question=question,
            turn_info=turn_info,
            query_plan=query_plan,
            resolution=resolution,
            answer=answer,
            execution_result=execution_result,
            expected_state_version=expected_state_version,
            lifecycle=lifecycle,
        )
```

If `_write_conversation_turn()` previously returned `None`, returning the manager result is allowed and useful for tests.

- [ ] **Step 5: Run pre-commit conflict test and existing writeback tests**

Run:

```bash
cd code/C8
pytest tests/test_conversation_state.py::test_write_conversation_turn_records_pre_commit_conflict_without_business_update tests/test_conversation_state.py::test_message_only_writeback_does_not_replace_current_entity tests/test_conversation_state.py::test_recommendation_list_writeback_updates_recommendation_and_clears_entity tests/test_conversation_state.py::test_resolved_followup_writeback_sets_current_entity_after_successful_retrieval -q
```

Expected:

- PASS.

- [ ] **Step 6: Commit**

```bash
git add code/C8/main.py code/C8/rag_modules/conversation_manager.py code/C8/tests/test_conversation_state.py
git commit -m "feat: commit conversation state with expected version"
```

---

## Task 4: Add Post-Resolution And Pre-Generation Stale-State Blocking

**Files:**
- Modify: `code/C8/main.py`
- Modify: `code/C8/tests/test_conversation_state.py`

- [ ] **Step 1: Add post-resolution / pre-plan stale-state test**

Append to `code/C8/tests/test_conversation_state.py`:

```python
def test_state_dependent_turn_replans_before_planning_after_resolution_mismatch(monkeypatch):
    import main as main_module

    system = _system()
    manager = system.generation_module.conversation_manager
    manager.record_recommendations("stale-resolution", ["蛋炒饭"])

    planned = {"called": False}

    def fail_if_planned(*args, **kwargs):
        planned["called"] = True
        return {"route_type": "detail", "dish_name": "蛋炒饭"}

    def mutate_after_resolution(*args, **kwargs):
        manager.commit_state_diff(
            "stale-resolution",
            {
                "answer_type": "smalltalk",
                "updates": {"last_answer_type": "smalltalk"},
                "clear": [],
                "append_history": False,
                "history": None,
            },
            expected_version=manager.get_state_version("stale-resolution"),
        )
        return {
            "next_action": "apply_reference_resolution",
            "resolved_target": "蛋炒饭",
            "confidence": 0.9,
            "target_source": "last_recommendation_list[0]",
            "writeback_eligible": True,
        }

    monkeypatch.setattr(main_module, "resolve_reference_from_snapshot", mutate_after_resolution)
    monkeypatch.setattr(system, "_build_query_plan", fail_if_planned)

    answer = system.ask_question("第一个怎么做", stream=False, session_id="stale-resolution")

    assert planned["called"] is False
    assert "上下文刚刚更新" in answer
```

- [ ] **Step 2: Add deterministic stale-generation blocking test**

Append to `code/C8/tests/test_conversation_state.py`:

```python
def test_state_dependent_turn_does_not_generate_after_pre_generation_version_mismatch(monkeypatch):
    system = _system()
    manager = system.generation_module.conversation_manager
    manager.record_recommendations("stale-gen", ["蛋炒饭"])

    generated = {"called": False}

    def fail_if_generated(*args, **kwargs):
        generated["called"] = True
        return "不应该生成"

    system.generation_module.generate_step_by_step_answer = fail_if_generated

    original_build_context_pack = system.context_packer.build_context_pack

    def mutate_state_before_generation(**kwargs):
        pack = original_build_context_pack(**kwargs)
        manager.commit_state_diff(
            "stale-gen",
            {
                "answer_type": "smalltalk",
                "updates": {"last_answer_type": "smalltalk"},
                "clear": [],
                "append_history": False,
                "history": None,
            },
            expected_version=manager.get_state_version("stale-gen"),
        )
        return pack

    monkeypatch.setattr(system.context_packer, "build_context_pack", mutate_state_before_generation)

    answer = system.ask_question("第一个怎么做", stream=False, session_id="stale-gen")

    assert generated["called"] is False
    assert "上下文刚刚更新" in answer
```

- [ ] **Step 3: Run stale-state tests and verify they fail**

Run:

```bash
cd code/C8
pytest tests/test_conversation_state.py::test_state_dependent_turn_replans_before_planning_after_resolution_mismatch tests/test_conversation_state.py::test_state_dependent_turn_does_not_generate_after_pre_generation_version_mismatch -q
```

Expected:

- FAIL because `ask_question()` does not yet block planning after reference resolution or generation after context pack when state version changes.

- [ ] **Step 4: Add runtime context creation to `ask_question()`**

In `main.py`, import:

```python
from rag_modules.turn_runtime import (
    TurnRuntimeContext,
    append_runtime_event,
    should_replan_after_mismatch,
)
```

At the start of `ask_question()`, after `conversation_manager` is available:

```python
        conversation_manager = getattr(self.generation_module, "conversation_manager", None)
        read_state_version = (
            conversation_manager.get_state_version(session_id)
            if conversation_manager
            else 0
        )
        runtime_ctx = TurnRuntimeContext.start(
            session_id=session_id,
            read_state_version=read_state_version,
        )
```

Attach runtime data to every `execution_result` before writeback:

```python
execution_result["runtime"] = {
    "turn_id": runtime_ctx.turn_id,
    "trace_id": runtime_ctx.trace_id,
    "read_state_version": runtime_ctx.read_state_version,
    "replan_count": runtime_ctx.replan_count,
    "lifecycle": runtime_ctx.lifecycle,
    "trace_events": runtime_ctx.trace_events,
}
```

- [ ] **Step 5: Add state dependency helper**

In `RecipeRAGSystem`, add:

```python
    def _turn_depends_on_state(self, turn_info: dict, resolution: dict | None, query_plan: dict | None = None) -> bool:
        if turn_info.get("depends_on_state"):
            return True
        if turn_info.get("reference_trigger") not in {None, "none"}:
            return True
        if resolution and resolution.get("resolved_target"):
            return True
        if turn_info.get("action") in {"history_answer", "clarification_response"}:
            return True
        return False
```

- [ ] **Step 6: Add post-resolution / pre-plan version check**

After `resolution` is produced and guarded, but before `build_execution_plan()` or `_build_query_plan()` can run:

```python
        if conversation_manager and self._turn_depends_on_state(turn_info, resolution, None):
            version_check = conversation_manager.check_state_version(
                session_id,
                runtime_ctx.read_state_version,
            )
            if not version_check["matched"]:
                append_runtime_event(
                    runtime_ctx,
                    "state_version_mismatch",
                    reason="concurrent_turn_or_rapid_followup",
                    read_state_version=runtime_ctx.read_state_version,
                    current_state_version=version_check["current_version"],
                    checkpoint="post_resolution_pre_plan",
                )
                answer = "上下文刚刚更新了，我需要你再确认一下是指哪一道菜。"
                runtime_ctx.lifecycle.update({
                    "status": "failed",
                    "reason": "state_version_mismatch_after_resolution",
                    "commit_business_state": False,
                })
                execution_result = {
                    "success": False,
                    "answer": answer,
                    "answer_type": "conflict",
                    "runtime": {
                        "turn_id": runtime_ctx.turn_id,
                        "trace_id": runtime_ctx.trace_id,
                        "read_state_version": runtime_ctx.read_state_version,
                        "replan_count": runtime_ctx.replan_count,
                        "lifecycle": runtime_ctx.lifecycle,
                        "trace_events": runtime_ctx.trace_events,
                    },
                }
                self.last_execution_result = execution_result
                return answer
```

Task 5 will replace this direct conflict return with `{"runtime_action": "replan"}` so the same turn-level retry budget owns every version mismatch.

- [ ] **Step 7: Add pre-generation version check after context pack**

After context pack is built and before `_generate_list_response()` or `_generate_detail_response()`:

```python
        if conversation_manager and self._turn_depends_on_state(turn_info, resolution, query_plan):
            version_check = conversation_manager.check_state_version(
                session_id,
                runtime_ctx.read_state_version,
            )
            if not version_check["matched"]:
                append_runtime_event(
                    runtime_ctx,
                    "state_version_mismatch",
                    reason="concurrent_turn_or_rapid_followup",
                    read_state_version=runtime_ctx.read_state_version,
                    current_state_version=version_check["current_version"],
                    checkpoint="pre_generation",
                )
                answer = "上下文刚刚更新了，我需要你再确认一下是指哪一道菜。"
                runtime_ctx.lifecycle.update({
                    "status": "failed",
                    "reason": "state_version_mismatch_before_generation",
                    "commit_business_state": False,
                })
                execution_result = {
                    "success": False,
                    "answer": answer,
                    "answer_type": "conflict",
                    "runtime": {
                        "turn_id": runtime_ctx.turn_id,
                        "trace_id": runtime_ctx.trace_id,
                        "read_state_version": runtime_ctx.read_state_version,
                        "replan_count": runtime_ctx.replan_count,
                        "lifecycle": runtime_ctx.lifecycle,
                        "trace_events": runtime_ctx.trace_events,
                    },
                }
                self.last_execution_result = execution_result
                return answer
```

- [ ] **Step 8: Run stale-state tests**

Run:

```bash
cd code/C8
pytest tests/test_conversation_state.py::test_state_dependent_turn_replans_before_planning_after_resolution_mismatch tests/test_conversation_state.py::test_state_dependent_turn_does_not_generate_after_pre_generation_version_mismatch -q
```

Expected:

- PASS.

- [ ] **Step 9: Commit**

```bash
git add code/C8/main.py code/C8/tests/test_conversation_state.py
git commit -m "feat: block stale state-dependent generation"
```

---

## Task 5: Make Replan Budget Shared And Non-Recursive

**Files:**
- Modify: `code/C8/main.py`
- Modify: `code/C8/tests/test_conversation_state.py`

- [ ] **Step 1: Add retry exhaustion test**

Append to `code/C8/tests/test_conversation_state.py`:

```python
def test_repeated_version_mismatch_uses_shared_replan_budget_and_returns_conflict(monkeypatch):
    system = _system()
    manager = system.generation_module.conversation_manager
    manager.record_recommendations("shared-budget", ["蛋炒饭"])

    generation_calls = {"count": 0}
    system.generation_module.generate_step_by_step_answer = lambda *args, **kwargs: generation_calls.__setitem__("count", generation_calls["count"] + 1) or "不应生成"

    original_build_context_pack = system.context_packer.build_context_pack

    def always_mutate_before_generation(**kwargs):
        pack = original_build_context_pack(**kwargs)
        manager.commit_state_diff(
            "shared-budget",
            {
                "answer_type": "smalltalk",
                "updates": {"last_answer_type": "smalltalk"},
                "clear": [],
                "append_history": False,
                "history": None,
            },
            expected_version=manager.get_state_version("shared-budget"),
        )
        return pack

    monkeypatch.setattr(system.context_packer, "build_context_pack", always_mutate_before_generation)

    answer = system.ask_question("第一个怎么做", stream=False, session_id="shared-budget")

    assert "上下文刚刚更新" in answer
    assert generation_calls["count"] == 0
    assert system.last_execution_result["runtime"]["replan_count"] == 1
```

- [ ] **Step 2: Run retry exhaustion test and verify it fails or loops incorrectly**

Run:

```bash
cd code/C8
pytest tests/test_conversation_state.py::test_repeated_version_mismatch_uses_shared_replan_budget_and_returns_conflict -q
```

Expected:

- FAIL until `ask_question()` uses one shared runtime context across replans.

- [ ] **Step 3: Refactor `ask_question()` into runtime loop**

Change `ask_question()` to create `runtime_ctx` once, then execute the existing body inside a small loop:

```python
        runtime_ctx = TurnRuntimeContext.start(
            session_id=session_id,
            read_state_version=read_state_version,
        )
        while True:
            result = self._ask_question_once(
                question=question,
                stream=stream,
                session_id=session_id,
                return_diagnostics=return_diagnostics,
                expectation=expectation,
                runtime_ctx=runtime_ctx,
            )
            if not isinstance(result, dict) or result.get("runtime_action") != "replan":
                return result
            if not should_replan_after_mismatch(runtime_ctx):
                return self._build_context_conflict_answer(runtime_ctx)
            runtime_ctx.read_state_version = conversation_manager.get_state_version(session_id)
```

Move the old body into:

```python
    def _ask_question_once(
        self,
        *,
        question: str,
        stream: bool,
        session_id: str,
        return_diagnostics: bool,
        expectation: Dict[str, Any] | None,
        runtime_ctx,
    ):
        ...
```

Refactor boundary:

- The outer `ask_question()` owns module readiness checks, `conversation_manager` lookup, the initial `read_state_version`, `TurnRuntimeContext.start(...)`, and the `while` loop.
- The user-facing `print(f"\n用户问题: {question}")` should stay in outer `ask_question()` before the loop so replans do not duplicate it.
- `_ask_question_once()` owns the previous request body: `basic_safety_gate`, conversation snapshot, turn understanding, reference resolution, direct smalltalk/domain reject/history answers, clarification, retrieval, context packing, generation, diagnostics, and writeback dispatch.
- Direct returns from `_ask_question_once()` remain direct returns unless a version-check checkpoint explicitly returns `{"runtime_action": "replan"}`.

When either post-resolution/pre-plan or pre-generation mismatch happens, return:

```python
return {"runtime_action": "replan"}
```

When budget is exhausted, `_build_context_conflict_answer()` returns the conflict text and sets `last_execution_result`.

Streaming boundary:

- Replan-capable version checks must happen before a stream generator is returned to the caller.
- Once `_wrap_stream_with_lifecycle(...)` has been returned, it must not emit or return `{"runtime_action": "replan"}`.
- If state changes after chunks have already been yielded, the stream wrapper may record a pre-commit conflict and skip business-state mutation, but it must not re-run generation because the client has already seen output.

- [ ] **Step 4: Add conflict answer helper**

Add to `RecipeRAGSystem`:

```python
    def _build_context_conflict_answer(self, runtime_ctx) -> str:
        answer = "上下文刚刚更新了，我需要你再确认一下是指哪一道菜。"
        runtime_ctx.lifecycle.update({
            "status": "failed",
            "reason": "state_version_mismatch_replan_exhausted",
            "commit_business_state": False,
        })
        execution_result = {
            "success": False,
            "answer": answer,
            "answer_type": "conflict",
            "runtime": {
                "turn_id": runtime_ctx.turn_id,
                "trace_id": runtime_ctx.trace_id,
                "read_state_version": runtime_ctx.read_state_version,
                "replan_count": runtime_ctx.replan_count,
                "lifecycle": runtime_ctx.lifecycle,
                "trace_events": runtime_ctx.trace_events,
            },
        }
        self.last_execution_result = execution_result
        return answer
```

- [ ] **Step 5: Run retry exhaustion and stale-state tests**

Run:

```bash
cd code/C8
pytest tests/test_conversation_state.py::test_state_dependent_turn_replans_before_planning_after_resolution_mismatch tests/test_conversation_state.py::test_state_dependent_turn_does_not_generate_after_pre_generation_version_mismatch tests/test_conversation_state.py::test_repeated_version_mismatch_uses_shared_replan_budget_and_returns_conflict -q
```

Expected:

- PASS.

- [ ] **Step 6: Commit**

```bash
git add code/C8/main.py code/C8/tests/test_conversation_state.py
git commit -m "refactor: use shared replan budget for runtime version checks"
```

---

## Task 6: Add Streaming Lifecycle Wrapper

**Files:**
- Modify: `code/C8/main.py`
- Modify: `code/C8/tests/test_conversation_state.py`

- [ ] **Step 1: Add completed stream lifecycle test**

Append to `code/C8/tests/test_conversation_state.py`:

```python
def test_completed_stream_commits_business_state_after_full_consumption():
    system = _system()

    stream = system.ask_question("蛋炒饭怎么做", stream=True, session_id="stream-completed")
    assert list(stream) == ["步骤1"]

    manager = system.generation_module.conversation_manager
    session = manager.get_session("stream-completed")
    assert session.current_entity == "蛋炒饭"
    assert system.last_execution_result["runtime"]["lifecycle"]["status"] == "completed"
    assert system.last_execution_result["runtime"]["lifecycle"]["commit_business_state"] is True
```

- [ ] **Step 2: Add aborted stream lifecycle test**

Append:

```python
def test_aborted_stream_does_not_commit_current_dish_or_recommendations():
    system = _system()

    stream = system.ask_question("蛋炒饭怎么做", stream=True, session_id="stream-aborted")
    first = next(stream)
    assert first == "步骤1"
    stream.close()

    manager = system.generation_module.conversation_manager
    session = manager.get_session("stream-aborted")
    assert session.current_entity is None
    assert session.recent_recommendations == []
    assert system.last_execution_result["runtime"]["lifecycle"]["status"] == "aborted"
    assert system.last_execution_result["runtime"]["lifecycle"]["reason"] == "client_disconnect_or_stream_not_consumed"
```

- [ ] **Step 3: Run stream lifecycle tests and verify they fail**

Run:

```bash
cd code/C8
pytest tests/test_conversation_state.py::test_completed_stream_commits_business_state_after_full_consumption tests/test_conversation_state.py::test_aborted_stream_does_not_commit_current_dish_or_recommendations -q
```

Expected:

- FAIL because `_wrap_stream_with_writeback()` has no explicit lifecycle and does not handle `GeneratorExit` as aborted.

- [ ] **Step 4: Replace `_wrap_stream_with_writeback()` with lifecycle-aware wrapper**

Replace `_wrap_stream_with_writeback()` in `main.py` with:

```python
    def _wrap_stream_with_lifecycle(
        self,
        *,
        answer_stream,
        session_id: str,
        question: str,
        turn_info: dict,
        query_plan: dict | None,
        resolution: dict | None,
        execution_result: dict,
        runtime_ctx,
    ):
        collected = []
        runtime_ctx.lifecycle.update({
            "status": "streaming",
            "partial_answer_length": 0,
            "commit_business_state": False,
            "reason": None,
        })
        append_runtime_event(runtime_ctx, "stream_started")
        try:
            for chunk in answer_stream:
                collected.append(chunk)
                runtime_ctx.lifecycle["partial_answer_length"] = len("".join(collected))
                yield chunk
        except GeneratorExit:
            runtime_ctx.lifecycle.update({
                "status": "aborted",
                "reason": "client_disconnect_or_stream_not_consumed",
                "commit_business_state": False,
            })
            execution_result["stream_interrupted"] = True
            execution_result["answer"] = "".join(collected)
            execution_result["runtime"] = self._runtime_payload(runtime_ctx)
            self.last_execution_result = execution_result
            self._write_conversation_turn(
                session_id=session_id,
                question=question,
                answer="".join(collected),
                turn_info=turn_info,
                query_plan=query_plan,
                resolution=resolution,
                execution_result=execution_result,
            )
            raise
        except Exception:
            runtime_ctx.lifecycle.update({
                "status": "failed",
                "reason": "stream_failed",
                "commit_business_state": False,
            })
            execution_result["stream_interrupted"] = True
            execution_result["success"] = False
            execution_result["runtime"] = self._runtime_payload(runtime_ctx)
            self.last_execution_result = execution_result
            raise
        else:
            full_text = "".join(collected)
            runtime_ctx.lifecycle.update({
                "status": "completed",
                "reason": None,
                "commit_business_state": True,
                "partial_answer_length": len(full_text),
            })
            execution_result["answer"] = full_text
            execution_result["success"] = True
            execution_result["runtime"] = self._runtime_payload(runtime_ctx)
            self.last_execution_result = execution_result
            self._write_conversation_turn(
                session_id=session_id,
                question=question,
                answer=full_text,
                turn_info=turn_info,
                query_plan=query_plan,
                resolution=resolution,
                execution_result=execution_result,
            )
```

Add helper:

```python
    def _runtime_payload(self, runtime_ctx) -> dict:
        return {
            "turn_id": runtime_ctx.turn_id,
            "trace_id": runtime_ctx.trace_id,
            "read_state_version": runtime_ctx.read_state_version,
            "replan_count": runtime_ctx.replan_count,
            "lifecycle": dict(runtime_ctx.lifecycle),
            "trace_events": list(runtime_ctx.trace_events),
        }
```

Replace calls to `_wrap_stream_with_writeback()` with `_wrap_stream_with_lifecycle(..., runtime_ctx=runtime_ctx)`.

Important lifecycle ordering:

- Set `runtime_ctx.lifecycle["status"] = "completed"` before calling `_write_conversation_turn()` on the successful stream path.
- Set `runtime_ctx.lifecycle["status"] = "aborted"` before calling `_write_conversation_turn()` on `GeneratorExit`.
- The stream wrapper must not request a replan after yielding chunks. If `_write_conversation_turn()` rejects the final commit because `expected_state_version` is stale, keep the streamed answer as output, skip business-state mutation, and expose the rejected commit result in `last_execution_result`.

- [ ] **Step 5: Run stream lifecycle tests**

Run:

```bash
cd code/C8
pytest tests/test_conversation_state.py::test_completed_stream_commits_business_state_after_full_consumption tests/test_conversation_state.py::test_aborted_stream_does_not_commit_current_dish_or_recommendations -q
```

Expected:

- PASS.

- [ ] **Step 6: Commit**

```bash
git add code/C8/main.py code/C8/tests/test_conversation_state.py
git commit -m "feat: add stream lifecycle writeback"
```

---

## Task 7: Add Runtime Cutover Tests

**Files:**
- Create: `code/C8/tests/test_runtime_cutover.py`
- Modify: `code/C8/main.py`

- [ ] **Step 1: Add source-level runtime cutover tests**

Create `code/C8/tests/test_runtime_cutover.py`:

```python
import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "main.py"


def _function_source(function_name: str) -> str:
    source = MAIN.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            return ast.get_source_segment(source, node)
    raise AssertionError(f"{function_name} not found")


def test_ask_question_uses_turn_runtime_context():
    source = _function_source("ask_question")

    assert "TurnRuntimeContext.start(" in source
    assert "read_state_version" in source
    assert "max_replan_count" in Path(ROOT / "rag_modules" / "turn_runtime.py").read_text(encoding="utf-8")


def test_generation_path_checks_version_before_generation():
    source = _function_source("_ask_question_once")

    context_pack_index = source.index("context_pack = self.context_packer.build_context_pack(")
    version_check_index = source.index("check_state_version(")
    generation_index = min(
        source.index("self._generate_list_response("),
        source.index("self._generate_detail_response("),
    )

    assert context_pack_index < version_check_index < generation_index


def test_resolution_path_checks_version_before_planning():
    source = _function_source("_ask_question_once")

    resolution_index = source.index("resolution")
    checkpoint_index = source.index("post_resolution_pre_plan")
    planning_index = min(
        source.index("build_execution_plan("),
        source.index("self._build_query_plan("),
    )

    assert resolution_index < checkpoint_index < planning_index


def test_old_stream_writeback_wrapper_is_not_lifecycle_owner():
    source = MAIN.read_text(encoding="utf-8")

    assert "def _wrap_stream_with_writeback" not in source
    assert "def _wrap_stream_with_lifecycle" in source
    lifecycle_source = _function_source("_wrap_stream_with_lifecycle")
    assert "GeneratorExit" in lifecycle_source
    assert "client_disconnect_or_stream_not_consumed" in lifecycle_source
    assert "\"status\": \"completed\"" in lifecycle_source
    assert "\"status\": \"aborted\"" in lifecycle_source


def test_writeback_uses_expected_state_version():
    source = _function_source("_write_conversation_turn")

    assert "expected_state_version" in source
    assert "writeback_turn_state(" in source
```

- [ ] **Step 2: Run cutover tests and verify they pass**

Run:

```bash
cd code/C8
pytest tests/test_runtime_cutover.py -q
```

Expected:

- PASS after Tasks 1-6.

- [ ] **Step 3: Commit**

```bash
git add code/C8/tests/test_runtime_cutover.py code/C8/main.py
git commit -m "test: enforce runtime lifecycle cutover"
```

---

## Task 8: Stage 05 Acceptance

**Files:**
- Verify only unless tests reveal migration gaps.

- [ ] **Step 1: Run focused runtime tests**

Run:

```bash
cd code/C8
pytest tests/test_turn_runtime.py tests/test_runtime_cutover.py -q
```

Expected:

- PASS.

- [ ] **Step 2: Run state and stream tests**

Run:

```bash
cd code/C8
pytest tests/test_conversation_state.py tests/test_state_hardening.py -q
```

Expected:

- PASS.

- [ ] **Step 3: Run adjacent architecture tests**

Run:

```bash
cd code/C8
pytest tests/test_state_update_policy.py tests/test_retrieval_executor.py tests/test_retrieval_executor_cutover.py tests/test_context_packer.py tests/test_context_packer_cutover.py -q
```

Expected:

- PASS.

- [ ] **Step 4: Run web/SSE tests**

Run:

```bash
cd code/C8
pytest tests/test_web_app.py -q
```

Expected:

- PASS.

- [ ] **Step 5: Run placeholder and stale-path scans**

Run:

```bash
cd code/C8
python -c "from pathlib import Path; files=[Path('docs/superpowers/plans/2026-07-07-runtime-versioning-and-streaming.md'),Path('docs/superpowers/specs/2026-07-07-runtime-versioning-and-streaming-design.md'),Path('docs/architecture/evolution/05-runtime-versioning-and-streaming.md')]; bad=['TO'+'DO','TB'+'D','implement'+' later','fill'+' in','Similar'+' to Task','Add'+' appropriate','Write tests'+' for the above']; hits=[(str(p), b) for p in files for b in bad if b in p.read_text(encoding='utf-8')]; assert not hits, hits"
python -c "from pathlib import Path; source=Path('main.py').read_text(encoding='utf-8'); assert 'def _wrap_stream_with_writeback' not in source; assert 'client_disconnect_or_stream_not_consumed' in source; assert 'check_state_version(' in source"
```

Expected:

- First command has no output.
- Second command exits 0.

- [ ] **Step 6: Commit final acceptance fixes if needed**

If previous steps required fixes, run:

```bash
git add code/C8/main.py code/C8/rag_modules/conversation_manager.py code/C8/rag_modules/turn_runtime.py code/C8/tests
git commit -m "test: accept runtime versioning and streaming lifecycle"
```

If no files changed, do not create an empty commit.

---

## Self-Review

Spec coverage:

- Single-process, no full async runtime is preserved.
- `turn_id`, `trace_id`, `read_state_version`, lifecycle, and shared budget are covered by Task 1.
- Monotonic `state_version` and expected-version commit are covered by Task 2 and Task 3.
- Post-resolution/pre-plan and pre-generation stale-state blocking are covered by Task 4.
- Shared replan budget and conflict handling are covered by Task 5.
- Completed/aborted stream lifecycle is covered by Task 6.
- Runtime cutover is covered by Task 7.
- Acceptance verification is covered by Task 8.

Cutover consistency:

- `StateUpdatePolicy` remains the business mutation policy.
- `ConversationManager` owns locked state mutation and version increment.
- `_wrap_stream_with_writeback()` does not remain as independent lifecycle owner.
- Version mismatch after reference resolution blocks stale planning before query/execution planning starts.
- Version mismatch before generation blocks state-dependent generation.
- Replan is only allowed before a stream generator is returned; streamed chunks are never followed by hidden regeneration.
- Aborted stream uses `stream_aborted` behavior and does not update business state.
- `turn_lifecycle` retention is capped at 20 entries per session.

Fixture consistency:

- Tests use deterministic hooks to mutate state before generation.
- Tests do not use sleeps to simulate races.
- Stream abort tests explicitly stop consumption.
- Completed stream tests fully consume the stream before checking state.
