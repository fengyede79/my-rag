# State Contract And Writeback Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Introduce a centralized `answer_type -> state_diff -> apply` policy so conversation state writes are typed, inspectable, and field-whitelisted before the main pipeline is reordered.

**Architecture:** Keep the existing `ConversationManager.writeback_turn_state()` public entrypoint, but cut the production writeback path over to the new `answer_type -> state_diff -> apply` architecture. After tests pass, delete obsolete writeback branches, helpers, imports, and modules that are no longer called. The final Stage 01 state must contain the new world only, not an unused shadow of the old writeback order.

**Tech Stack:** Python dataclasses, pytest, existing C8 conversation manager. The old writeback review module is a removal target, not a dependency of the new architecture.

---

### Task 1: Add Policy Classification Tests

**Files:**
- Create: `code/C8/tests/test_state_update_policy.py`

- [ ] **Step 1: Create failing tests for `classify_answer_type()`**

Create `code/C8/tests/test_state_update_policy.py` with:

```python
from rag_modules.state_update_policy import classify_answer_type


def test_stream_interrupted_becomes_stream_aborted():
    answer_type = classify_answer_type(
        {"turn_type": "domain_query"},
        {"stream_interrupted": True, "success": True},
        query_plan=None,
        resolution=None,
    )

    assert answer_type == "stream_aborted"


def test_failed_retrieval_becomes_no_result():
    answer_type = classify_answer_type(
        {"turn_type": "domain_query"},
        {"success": False},
        query_plan={"route_type": "detail"},
        resolution=None,
    )

    assert answer_type == "no_result"


def test_front_door_blocked_becomes_domain_reject():
    answer_type = classify_answer_type(
        {"turn_type": "front_door_blocked"},
        {"success": True},
        query_plan=None,
        resolution=None,
    )

    assert answer_type == "domain_reject"


def test_smalltalk_turn_becomes_smalltalk():
    answer_type = classify_answer_type(
        {"turn_type": "smalltalk"},
        {"success": True},
        query_plan=None,
        resolution=None,
    )

    assert answer_type == "smalltalk"


def test_clarification_resolution_becomes_clarification():
    answer_type = classify_answer_type(
        {"turn_type": "domain_query"},
        {"success": True},
        query_plan=None,
        resolution={"next_action": "ask_clarification"},
    )

    assert answer_type == "clarification"


def test_recommendation_result_becomes_recommendation():
    answer_type = classify_answer_type(
        {"turn_type": "domain_query"},
        {"success": True, "recommended_dishes": ["回锅肉", "麻婆豆腐"]},
        query_plan={"route_type": "list"},
        resolution=None,
    )

    assert answer_type == "recommendation"


def test_detail_query_becomes_detail():
    answer_type = classify_answer_type(
        {"turn_type": "domain_query"},
        {"success": True, "resolved_target": "蛋炒饭"},
        query_plan={"route_type": "detail"},
        resolution=None,
    )

    assert answer_type == "detail"


def test_apply_reference_resolution_becomes_detail():
    answer_type = classify_answer_type(
        {"turn_type": "domain_query"},
        {"success": True},
        query_plan={"route_type": "basic"},
        resolution={"next_action": "apply_reference_resolution", "resolved_target": "蛋炒饭"},
    )

    assert answer_type == "detail"


def test_apply_correction_becomes_detail():
    answer_type = classify_answer_type(
        {"turn_type": "domain_query"},
        {"success": True},
        query_plan={"route_type": "basic"},
        resolution={"next_action": "apply_correction", "resolved_target": "蛋炒饭"},
    )

    assert answer_type == "detail"
```

- [ ] **Step 2: Run tests and verify they fail because module does not exist**

Run:

```powershell
pytest code\C8\tests\test_state_update_policy.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'rag_modules.state_update_policy'`.

### Task 2: Implement `classify_answer_type()`

**Files:**
- Create: `code/C8/rag_modules/state_update_policy.py`
- Test: `code/C8/tests/test_state_update_policy.py`

- [ ] **Step 1: Add the policy module with classification only**

Create `code/C8/rag_modules/state_update_policy.py`:

```python
"""State update policy for typed, field-limited conversation writeback."""

from __future__ import annotations

from typing import Any


def classify_answer_type(
    turn_info: dict[str, Any],
    execution_result: dict[str, Any],
    query_plan: dict[str, Any] | None,
    resolution: dict[str, Any] | None,
) -> str:
    """Classify answer_type directly from Stage 01 execution facts."""
    turn_type = turn_info.get("turn_type", "domain_query")

    if execution_result.get("stream_interrupted"):
        return "stream_aborted"
    if execution_result.get("success") is False:
        return "no_result"
    if turn_type in {"front_door_blocked", "out_of_domain"}:
        return "domain_reject"
    if turn_type in {"smalltalk", "front_door_direct_reply"}:
        return "smalltalk"
    if resolution and resolution.get("next_action") == "ask_clarification":
        return "clarification"
    if resolution and resolution.get("next_action") in {"apply_correction", "apply_reference_resolution"}:
        if execution_result.get("resolved_target") or resolution.get("resolved_target"):
            return "detail"
    if query_plan and query_plan.get("route_type") == "list":
        return "recommendation" if execution_result.get("recommended_dishes") else "normal"
    if query_plan and query_plan.get("route_type") == "detail":
        return "detail"
    if execution_result.get("resolved_target"):
        return "detail"
    return "normal"
```

- [ ] **Step 2: Run classification tests**

Run:

```powershell
pytest code\C8\tests\test_state_update_policy.py -q
```

Expected: PASS.

### Task 3: Add State Diff Builder Tests

**Files:**
- Modify: `code/C8/tests/test_state_update_policy.py`
- Modify: `code/C8/rag_modules/state_update_policy.py`

- [ ] **Step 1: Add failing tests for field whitelists**

Append to `code/C8/tests/test_state_update_policy.py`:

```python
from rag_modules.state_update_policy import build_state_diff


class DummyState:
    current_entity = "蛋炒饭"
    recent_recommendations = [{"rank": 1, "dish_name": "蛋炒饭"}]
    pending_clarification = None
    last_answer_type = None


def test_smalltalk_diff_does_not_update_business_state():
    diff = build_state_diff(
        "smalltalk",
        {"success": True},
        DummyState(),
        query_plan=None,
        resolution=None,
        answer="不客气",
        question="谢谢",
    )

    assert diff["answer_type"] == "smalltalk"
    assert diff["updates"] == {"last_answer_type": "smalltalk"}
    assert diff["append_history"] is True
    assert "current_dish" not in diff["allowed_fields"]
    assert "last_recommendation_list" not in diff["allowed_fields"]


def test_clarification_diff_sets_pending_only():
    diff = build_state_diff(
        "clarification",
        {"success": True},
        DummyState(),
        query_plan=None,
        resolution={
            "reason": "ambiguous_reference",
            "candidates": ["蛋炒饭", "扬州炒饭"],
            "clarification_question": "你指的是哪一道？",
        },
        answer="你指的是哪一道？",
        question="第一个呢",
    )

    assert diff["answer_type"] == "clarification"
    assert set(diff["allowed_fields"]) == {"pending_clarification", "last_answer_type", "history"}
    assert diff["updates"]["pending_clarification"]["reason"] == "ambiguous_reference"
    assert "current_dish" not in diff["updates"]


def test_recommendation_diff_updates_recommendations_not_current_dish():
    diff = build_state_diff(
        "recommendation",
        {"success": True, "recommended_dishes": ["回锅肉", "麻婆豆腐"]},
        DummyState(),
        query_plan={"route_type": "list"},
        resolution=None,
        answer="推荐回锅肉和麻婆豆腐",
        question="推荐两个菜",
    )

    assert diff["updates"]["last_recommendation_list"] == [
        {"rank": 1, "dish_name": "回锅肉"},
        {"rank": 2, "dish_name": "麻婆豆腐"},
    ]
    assert "current_dish" not in diff["updates"]


def test_detail_diff_requires_strong_target_evidence():
    diff = build_state_diff(
        "detail",
        {"success": True, "resolved_target": "蛋炒饭"},
        DummyState(),
        query_plan={"route_type": "detail"},
        resolution={"target_source": "ordinal_reference", "confidence": 0.8},
        answer="蛋炒饭做法",
        question="第一个怎么做",
    )

    assert diff["updates"]["current_dish"]["value"] == "蛋炒饭"
    assert diff["updates"]["current_dish"]["source"] == "ordinal_reference"
    assert diff["updates"]["current_dish"]["confidence"] == 0.8
    assert diff["history"]["entities"] == {"dish_name": "蛋炒饭"}


def test_detail_diff_uses_resolution_target_fallback():
    diff = build_state_diff(
        "detail",
        {"success": True},
        DummyState(),
        query_plan={"route_type": "basic"},
        resolution={
            "next_action": "apply_reference_resolution",
            "resolved_target": "扬州炒饭",
            "target_source": "ordinal_reference",
            "confidence": 0.8,
        },
        answer="扬州炒饭做法",
        question="第一个怎么做",
    )

    assert diff["updates"]["current_dish"]["value"] == "扬州炒饭"
    assert diff["history"]["entities"] == {"dish_name": "扬州炒饭"}


def test_no_result_diff_does_not_update_business_state():
    diff = build_state_diff(
        "no_result",
        {"success": False, "answer": "没有找到"},
        DummyState(),
        query_plan={"route_type": "detail", "dish_name": "不存在的菜"},
        resolution=None,
        answer="没有找到",
        question="不存在的菜怎么做",
    )

    assert diff["updates"] == {"last_answer_type": "no_result"}
    assert "current_dish" not in diff["allowed_fields"]


def test_stream_aborted_diff_keeps_history_without_business_state():
    diff = build_state_diff(
        "stream_aborted",
        {"stream_interrupted": True, "success": True},
        DummyState(),
        query_plan=None,
        resolution=None,
        answer="半截回答",
        question="推荐几个菜",
    )

    assert diff["updates"] == {"last_answer_type": "stream_aborted"}
    assert diff["append_history"] is True
    assert diff["history"]["question"] == "推荐几个菜"
    assert "current_dish" not in diff["allowed_fields"]
```

- [ ] **Step 2: Run tests and verify missing function failure**

Run:

```powershell
pytest code\C8\tests\test_state_update_policy.py -q
```

Expected: FAIL with `ImportError` or `AttributeError` for `build_state_diff`.

### Task 4: Implement `build_state_diff()`

**Files:**
- Modify: `code/C8/rag_modules/state_update_policy.py`
- Test: `code/C8/tests/test_state_update_policy.py`

- [ ] **Step 1: Add diff builder implementation**

Replace `code/C8/rag_modules/state_update_policy.py` with:

```python
"""State update policy for typed, field-limited conversation writeback."""

from __future__ import annotations

from typing import Any


ANSWER_TYPE_ALLOWED_FIELDS = {
    "smalltalk": {"last_answer_type", "history"},
    "domain_reject": {"last_answer_type", "history"},
    "clarification": {"pending_clarification", "last_answer_type", "history"},
    "recommendation": {"last_recommendation_list", "pending_clarification", "last_answer_type", "history"},
    "detail": {"current_dish", "pending_clarification", "last_answer_type", "history"},
    "comparison": {"current_entities", "last_answer_type", "history"},
    "history_answer": {"last_answer_type", "history"},
    "low_confidence": {"last_answer_type", "history"},
    "no_result": {"last_answer_type", "history"},
    "stream_aborted": {"last_answer_type", "history"},
    "normal": {"last_answer_type", "history"},
}


def classify_answer_type(
    turn_info: dict[str, Any],
    execution_result: dict[str, Any],
    query_plan: dict[str, Any] | None,
    resolution: dict[str, Any] | None,
) -> str:
    """Classify answer_type directly from Stage 01 execution facts."""
    turn_type = turn_info.get("turn_type", "domain_query")

    if execution_result.get("stream_interrupted"):
        return "stream_aborted"
    if execution_result.get("success") is False:
        return "no_result"
    if turn_type in {"front_door_blocked", "out_of_domain"}:
        return "domain_reject"
    if turn_type in {"smalltalk", "front_door_direct_reply"}:
        return "smalltalk"
    if resolution and resolution.get("next_action") == "ask_clarification":
        return "clarification"
    if resolution and resolution.get("next_action") in {"apply_correction", "apply_reference_resolution"}:
        if execution_result.get("resolved_target") or resolution.get("resolved_target"):
            return "detail"
    if query_plan and query_plan.get("route_type") == "list":
        return "recommendation" if execution_result.get("recommended_dishes") else "normal"
    if query_plan and query_plan.get("route_type") == "detail":
        return "detail"
    if execution_result.get("resolved_target"):
        return "detail"
    return "normal"


def _ranked_recommendations(dishes: list[str]) -> list[dict[str, Any]]:
    return [
        {"rank": index + 1, "dish_name": dish}
        for index, dish in enumerate(dishes or [])
    ]


def _detail_target(
    execution_result: dict[str, Any],
    query_plan: dict[str, Any] | None,
    resolution: dict[str, Any] | None,
) -> dict[str, Any] | None:
    query_plan = query_plan or {}
    resolution = resolution or {}

    target = execution_result.get("resolved_target") or resolution.get("resolved_target")
    source = resolution.get("target_source", "state_update_policy")
    confidence = resolution.get("confidence", 0.8)

    if not target and query_plan.get("dish_name"):
        target = query_plan["dish_name"]
        source = "explicit_query"
        confidence = 1.0

    if not target:
        return None

    return {"value": target, "source": source, "confidence": confidence}


def build_state_diff(
    answer_type: str,
    execution_result: dict[str, Any],
    old_state: Any,
    *,
    query_plan: dict[str, Any] | None = None,
    resolution: dict[str, Any] | None = None,
    answer: str = "",
    question: str = "",
) -> dict[str, Any]:
    """Build an inspectable state diff without mutating session state."""
    allowed = ANSWER_TYPE_ALLOWED_FIELDS.get(answer_type, ANSWER_TYPE_ALLOWED_FIELDS["normal"])
    updates: dict[str, Any] = {"last_answer_type": answer_type}
    clear: list[str] = []

    if answer_type == "clarification":
        resolution = resolution or {}
        updates["pending_clarification"] = {
            "reason": resolution.get("reason", "ambiguous_reference"),
            "candidates": resolution.get("candidates", []),
            "original_question": question,
            "clarification_question": resolution.get("clarification_question", answer),
        }

    elif answer_type == "recommendation":
        updates["last_recommendation_list"] = _ranked_recommendations(
            execution_result.get("recommended_dishes", [])
        )
        clear.append("pending_clarification")

    elif answer_type == "detail":
        target = _detail_target(execution_result, query_plan, resolution)
        if target:
            updates["current_dish"] = target
            clear.append("pending_clarification")

    return {
        "answer_type": answer_type,
        "allowed_fields": sorted(allowed),
        "updates": {
            key: value
            for key, value in updates.items()
            if key in allowed
        },
        "clear": [
            field
            for field in clear
            if field in allowed
        ],
        "append_history": "history" in allowed,
        "history": {
            "question": question,
            "answer": answer,
            "intent_type": answer_type,
            "entities": {
                "dish_name": updates["current_dish"]["value"]
            }
            if "current_dish" in updates
            else {},
        }
        if "history" in allowed
        else None,
        "reason": answer_type,
    }
```

- [ ] **Step 2: Run state update policy tests**

Run:

```powershell
pytest code\C8\tests\test_state_update_policy.py -q
```

Expected: PASS.

### Task 5: Add Apply Policy Tests

**Files:**
- Modify: `code/C8/tests/test_state_update_policy.py`
- Modify: `code/C8/rag_modules/conversation_manager.py`

- [ ] **Step 1: Add tests for `apply_state_diff()`**

Append to `code/C8/tests/test_state_update_policy.py`:

```python
from rag_modules.conversation_manager import ConversationManager


def test_apply_recommendation_diff_preserves_current_entity():
    manager = ConversationManager()
    session = manager.get_session("apply-rec")
    manager.set_current_dish("apply-rec", "蛋炒饭", source="setup", confidence=1.0)

    diff = build_state_diff(
        "recommendation",
        {"success": True, "recommended_dishes": ["回锅肉", "麻婆豆腐"]},
        session,
        answer="推荐回锅肉和麻婆豆腐",
        question="推荐两个菜",
    )
    manager.apply_state_diff("apply-rec", diff)

    assert session.current_entity == "蛋炒饭"
    assert [item["dish_name"] for item in session.recent_recommendations] == ["回锅肉", "麻婆豆腐"]
    assert session.last_answer_type == "recommendation"


def test_apply_detail_diff_sets_current_entity():
    manager = ConversationManager()
    session = manager.get_session("apply-detail")

    diff = build_state_diff(
        "detail",
        {"success": True, "resolved_target": "宫保鸡丁"},
        session,
        resolution={"target_source": "ordinal_reference", "confidence": 0.8},
        answer="宫保鸡丁做法",
        question="第一个怎么做",
    )
    manager.apply_state_diff("apply-detail", diff)

    assert session.current_entity == "宫保鸡丁"
    assert session.last_confirmed_target == "宫保鸡丁"
    assert session.current_entity_meta["source"] == "ordinal_reference"
    assert session.last_answer_type == "detail"


def test_apply_no_result_diff_preserves_current_entity():
    manager = ConversationManager()
    session = manager.get_session("apply-no-result")
    manager.set_current_dish("apply-no-result", "蛋炒饭", source="setup", confidence=1.0)

    diff = build_state_diff(
        "no_result",
        {"success": False},
        session,
        answer="没有找到",
        question="不存在的菜怎么做",
    )
    manager.apply_state_diff("apply-no-result", diff)

    assert session.current_entity == "蛋炒饭"
    assert session.last_answer_type == "no_result"
```

- [ ] **Step 2: Run tests and verify missing function failure**

Run:

```powershell
pytest code\C8\tests\test_state_update_policy.py -q
```

Expected: FAIL with `AttributeError` for `ConversationManager.apply_state_diff`.

### Task 6: Implement Manager-Owned `apply_state_diff()` And Add `last_answer_type`

**Files:**
- Modify: `code/C8/rag_modules/conversation_manager.py`
- Test: `code/C8/tests/test_state_update_policy.py`

- [ ] **Step 1: Add `last_answer_type` to `SessionState`**

In `code/C8/rag_modules/conversation_manager.py`, update the `SessionState` dataclass:

```python
    pending_clarification: Optional[Dict[str, Any]] = None
    last_answer_type: Optional[str] = None
```

- [ ] **Step 2: Add manager-owned `apply_state_diff()` implementation**

Add this method to `ConversationManager` in `code/C8/rag_modules/conversation_manager.py`. It must reuse existing manager helpers so the new architecture replaces the old path without losing established behavior:

```python
    def apply_state_diff(self, session_id: str, state_diff: dict[str, Any]) -> None:
        """Apply an approved Stage 01 state diff through existing manager helpers."""
        session = self.get_session(session_id)
        updates = state_diff.get("updates", {})

        if "last_answer_type" in updates:
            session.last_answer_type = updates["last_answer_type"]

        for field in state_diff.get("clear", []):
            if field == "pending_clarification":
                self.clear_pending_clarification(session_id)

        if "pending_clarification" in updates:
            pending = updates["pending_clarification"]
            self.set_pending_clarification(
                session_id,
                reason=pending.get("reason", "ambiguous_reference"),
                candidates=pending.get("candidates", []),
                original_question=pending.get("original_question", ""),
                clarification_question=pending.get("clarification_question", ""),
            )

        if "last_recommendation_list" in updates:
            self.record_recommendations(
                session_id,
                [item["dish_name"] for item in updates["last_recommendation_list"]],
            )

        if "current_dish" in updates:
            target = updates["current_dish"]
            self.set_current_dish(
                session_id,
                target["value"],
                source=target.get("source", "state_update_policy"),
                confidence=target.get("confidence", 0.8),
            )

        history = state_diff.get("history")
        if state_diff.get("append_history") and history:
            self.add_interaction(
                session_id,
                history.get("question", ""),
                history.get("answer", ""),
                intent_type=history.get("intent_type", state_diff.get("answer_type", "general")),
                entities=history.get("entities", {}),
            )
```

Do not add an `apply_state_diff()` function to `state_update_policy.py`. The policy module owns classification and diff construction; `ConversationManager` owns mutation.

- [ ] **Step 3: Run policy tests**

Run:

```powershell
pytest code\C8\tests\test_state_update_policy.py -q
```

Expected: PASS.

### Task 7: Cut `ConversationManager.writeback_turn_state()` Over To Policy

**Files:**
- Modify: `code/C8/rag_modules/conversation_manager.py`
- Test: `code/C8/tests/test_conversation_state.py`
- Test: `code/C8/tests/test_state_update_policy.py`

- [ ] **Step 1: Replace the writeback mode switch with policy classification, diff, and apply**

In `code/C8/rag_modules/conversation_manager.py`, replace the current `review_state_writeback()` call, `mode = review["writeback_mode"]`, branch switch, and all old direct mutation branches with:

```python
        from rag_modules.state_update_policy import (
            build_state_diff,
            classify_answer_type,
        )

        answer_type = classify_answer_type(
            turn_info,
            execution_result,
            query_plan,
            resolution,
        )
        session = self.get_session(session_id)
        state_diff = build_state_diff(
            answer_type,
            execution_result,
            session,
            query_plan=query_plan,
            resolution=resolution,
            answer=answer,
            question=question,
        )
        self.apply_state_diff(session_id, state_diff)
```

After this task, production writeback must not call `review_state_writeback()` and must not mutate state through the old `writeback_mode` branch switch.

- [ ] **Step 2: Run focused state tests**

Run:

```powershell
pytest code\C8\tests\test_state_update_policy.py code\C8\tests\test_conversation_state.py -q
```

Expected: PASS. If tests fail because the new apply path lost old helper behavior, fix `ConversationManager.apply_state_diff()` to reuse or match the existing helper semantics. Do not restore the old writeback mode switch as the production path.

### Task 8: Delete Obsolete Old-Path Code

**Files:**
- Modify: `code/C8/rag_modules/conversation_manager.py`
- Modify or delete: `code/C8/rag_modules/state_writeback_review.py`
- Modify: `code/C8/tests/test_state_writeback_review.py`
- Test: `code/C8/tests/test_state_update_policy.py`
- Test: `code/C8/tests/test_conversation_state.py`

- [ ] **Step 1: Search for old writeback path references**

Run:

```powershell
rg -n "writeback_mode|review_state_writeback|message_only|clarification_pending|recommendation_list|resolved_followup|correction_turn|explicit_single_dish" code\C8
```

Expected: matches identify old-path references that must be migrated to `answer_type/state_diff` or deleted. No old-path reference should remain because the new path calls it.

- [ ] **Step 2: Remove the old branch switch from `ConversationManager`**

Confirm `ConversationManager.writeback_turn_state()` no longer contains direct branches like:

```python
if mode == "message_only":
    ...
if mode == "clarification_pending":
    ...
if mode == "recommendation_list":
    ...
if mode in {"resolved_followup", "correction_turn"}:
    ...
if mode == "explicit_single_dish":
    ...
```

If any remain, delete them and keep only the new flow:

```python
answer_type = classify_answer_type(turn_info, execution_result, query_plan, resolution)
state_diff = build_state_diff(answer_type, execution_result, session, ...)
self.apply_state_diff(session_id, state_diff)
```

The final flow must pass execution facts directly into `classify_answer_type()`.

- [ ] **Step 3: Delete `state_writeback_review.py`**

Run:

```powershell
rg -n "review_state_writeback|state_writeback_review" code\C8
```

Delete `code/C8/rag_modules/state_writeback_review.py`. If `rg` finds any non-test imports after deletion, migrate them to `state_update_policy.py`.

- [ ] **Step 4: Replace old review tests with new policy tests**

Delete `code/C8/tests/test_state_writeback_review.py` after equivalent assertions exist in `code/C8/tests/test_state_update_policy.py`.

The replacement tests should assert `answer_type` and `state_diff`, not `writeback_mode`.

- [ ] **Step 5: Run focused tests after deletion**

Run:

```powershell
pytest code\C8\tests\test_state_update_policy.py code\C8\tests\test_conversation_state.py -q
```

Expected: PASS. There should be no dependency on deleted old-path code.

### Task 9: Add Integration Regression Tests For Stage 01 Acceptance

**Files:**
- Modify: `code/C8/tests/test_conversation_state.py`

- [ ] **Step 1: Add or confirm acceptance tests**

Ensure `code/C8/tests/test_conversation_state.py` includes tests equivalent to:

```python
def test_stage01_writeback_uses_state_diff_policy_for_detail():
    manager = ConversationManager()

    manager.writeback_turn_state(
        session_id="stage01-detail",
        question="蛋炒饭怎么做",
        answer="蛋炒饭做法",
        turn_info={"turn_type": "domain_query"},
        query_plan={"route_type": "detail", "dish_name": "蛋炒饭"},
        resolution=None,
        execution_result={"success": True, "answer": "蛋炒饭做法"},
    )

    session = manager.get_session("stage01-detail")
    assert session.current_entity == "蛋炒饭"
    assert session.last_answer_type == "detail"


def test_stage01_domain_reject_writeback_preserves_current_entity():
    manager = ConversationManager()
    manager.set_current_dish("stage01-domain", "蛋炒饭", source="setup", confidence=1.0)

    manager.writeback_turn_state(
        session_id="stage01-domain",
        question="Python 怎么学",
        answer="我主要处理食谱相关问题。",
        turn_info={"turn_type": "front_door_blocked"},
        query_plan=None,
        resolution=None,
        execution_result={"success": True, "answer": "我主要处理食谱相关问题。"},
    )

    session = manager.get_session("stage01-domain")
    assert session.current_entity == "蛋炒饭"
    assert session.last_answer_type == "domain_reject"


def test_stage01_failed_retrieval_writeback_preserves_current_entity():
    manager = ConversationManager()
    manager.set_current_dish("stage01-no-result", "蛋炒饭", source="setup", confidence=1.0)

    manager.writeback_turn_state(
        session_id="stage01-no-result",
        question="不存在的菜怎么做",
        answer="没有找到",
        turn_info={"turn_type": "domain_query"},
        query_plan={"route_type": "detail", "dish_name": "不存在的菜"},
        resolution=None,
        execution_result={"success": False, "answer": "没有找到"},
    )

    session = manager.get_session("stage01-no-result")
    assert session.current_entity == "蛋炒饭"
    assert session.last_answer_type == "no_result"
```

- [ ] **Step 2: Run acceptance-focused tests**

Run:

```powershell
pytest code\C8\tests\test_conversation_state.py -q
```

Expected: PASS.

### Task 10: Run Regression Suite For Stage 01 Scope

**Files:**
- Test: `code/C8/tests/`

- [ ] **Step 1: Run state and web regression tests**

Run:

```powershell
pytest code\C8\tests\test_state_update_policy.py code\C8\tests\test_conversation_state.py code\C8\tests\test_state_hardening.py code\C8\tests\test_web_app.py -q
```

Expected: PASS.

- [ ] **Step 2: Run full test suite if dependencies are available**

Run:

```powershell
pytest code\C8\tests -q
```

Expected: PASS, or document dependency-related skips/failures that are unrelated to Stage 01.

### Task 11: Update Stage 01 Acceptance Status

**Files:**
- Modify: `code/C8/docs/architecture/evolution/01-state-contract-and-writeback-policy.md`
- Modify: `code/C8/docs/superpowers/plans/2026-07-06-state-contract-and-writeback-policy.md`

- [ ] **Step 1: Mark Stage 01 accepted after tests pass**

In `code/C8/docs/architecture/evolution/01-state-contract-and-writeback-policy.md`, change:

```markdown
Status: expanded baseline
```

to:

```markdown
Status: accepted baseline
```

- [ ] **Step 2: Review final diff**

Run:

```powershell
git diff -- code\C8\rag_modules\state_update_policy.py code\C8\rag_modules\conversation_manager.py code\C8\rag_modules\state_writeback_review.py code\C8\tests\test_state_update_policy.py code\C8\tests\test_state_writeback_review.py code\C8\tests\test_conversation_state.py code\C8\docs\architecture\evolution\01-state-contract-and-writeback-policy.md
```

Expected: diff shows the policy module, controlled writeback integration, old-path deletion, migrated tests, and accepted spec status.

- [ ] **Step 3: Commit Stage 01 changes**

Run:

```powershell
git add code\C8\rag_modules\state_update_policy.py code\C8\rag_modules\conversation_manager.py code\C8\rag_modules\state_writeback_review.py code\C8\tests\test_state_update_policy.py code\C8\tests\test_state_writeback_review.py code\C8\tests\test_conversation_state.py code\C8\docs\architecture\evolution\01-state-contract-and-writeback-policy.md code\C8\docs\superpowers\plans\2026-07-06-state-contract-and-writeback-policy.md
git commit -m "feat: add state update policy"
```

Expected: commit succeeds after review.

---

## Self-Review Checklist

- Stage 01 does not reorder `ask_question()`.
- Stage 01 does not introduce Retrieval Executor or fallback retrieval.
- State writes are cut over to policy classification, diff building, and manager-owned diff application.
- The old writeback mode switch is not the production mutation path after Task 7.
- Uncalled old-path modules, functions, tests, branches, and nodes are deleted after the new path passes tests.
- `normal`, `no_result`, `smalltalk`, and `domain_reject` cannot update business entities.
- `detail` updates current dish only with explicit or resolved target evidence.
- Existing safe writeback behavior remains covered by tests.

