# Conversation Architecture Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the new conversation architecture so recommendation state, reference resolution, retrieval targeting, state writeback, and stream behavior form one verifiable closed loop.

**Architecture:** Keep the new pipeline as the only production semantic path: `Turn Qualification -> Snapshot -> Reference Resolution -> Execution Planning -> Query Plan -> Retrieval -> Generation -> Execution Result -> State Writeback Review`. Remove or disconnect old string-completion logic from production paths, and make tests verify structured state and retrieval targets instead of only checking that answers are non-empty.

**Tech Stack:** Python, pytest, Flask test client, existing `code/C8` RAG modules, DashScope real integration tests, Server-Sent Events stream endpoint.

---

## Reference

- Spec: `code/C8/docs/superpowers/specs/2026-07-04-conversation-architecture-completion-design.md`
- Existing modules:
  - `code/C8/main.py`
  - `code/C8/web_app.py`
  - `code/C8/rag_modules/conversation_manager.py`
  - `code/C8/rag_modules/state_writeback_review.py`
  - `code/C8/rag_modules/reference_resolution.py`
  - `code/C8/rag_modules/conversation_state_builder.py`
  - `code/C8/rag_modules/turn_qualification.py`

## File Map

**Modify:**
- `code/C8/main.py`: force resolved targets into query plan, enrich execution result, expose diagnostics for tests.
- `code/C8/rag_modules/state_writeback_review.py`: make writeback mode depend on query plan plus execution result.
- `code/C8/rag_modules/conversation_manager.py`: remove/disconnect old semantic helpers, add explicit writeback modes, prevent implicit entity pollution.
- `code/C8/rag_modules/conversation_state_builder.py`: include pending clarification in snapshot if not already present.
- `code/C8/rag_modules/reference_resolution.py`: ensure clarification and pending clarification outputs are structured.
- `code/C8/tests/test_state_writeback_review.py`: cover new writeback modes.
- `code/C8/tests/test_conversation_state.py`: cover state pollution and pending clarification.
- `code/C8/tests/test_conversation_integration_real.py`: strengthen real assertions.
- `code/C8/tests/test_web_app.py`: add stream equivalence and stream interruption tests with lightweight fake systems.

**Do not modify unless a test proves the need:**
- `code/C8/rag_modules/retrieval_optimization.py`
- `code/C8/rag_modules/hybrid_router.py`
- knowledge base data under `data/C8`

**Deprecate first, then remove after verification:**
- `ConversationManager.complete_query`
- `ConversationManager._resolve_entity_references`
- `ConversationManager._inherit_intent`
- `RecipeRAGSystem._should_inherit_current_entity` if it directly changes query text outside reference resolution.

---

## Pre-Implementation Fixes (applied before TDD cycle)

These issues were identified by reviewing the plan against the actual codebase and fixed in source before executing the task-by-task plan:

1. **`add_interaction` implicit entity removal**: Removed `if entities and entities.get("dish_name"): session.current_entity = entities["dish_name"]` from `add_interaction()`. Updated existing tests (`test_add_interaction_does_not_implicitly_set_entity`, `test_set_current_dish_updates_entity`, `test_no_result_turn_does_not_replace_current_entity`) to use `set_current_dish()` explicitly.

2. **`complete_query` entity preservation**: Since `add_interaction` no longer sets entity, `complete_query()` now uses `effective_entity = current_entity or session.current_entity` to preserve entity through deprecated path.

3. **DeprecationWarning on old helpers**: Added `warnings.warn(..., DeprecationWarning)` to `complete_query()`, `_resolve_entity_references()`, `_inherit_intent()`. Existing tests wrap calls with `warnings.catch_warnings()`.

4. **`review_state_writeback` signature**: Added `query_plan` parameter and `_route_type()` helper. Recommendation check now uses `route_type == "list"` (from query_plan or execution_result) instead of `turn_type == "recommendation_query"`. Existing tests updated to pass `query_plan={"route_type": "list"}`.

5. **`writeback_turn_state` new modes**: Added `clarification_pending`, `resolved_followup`, `explicit_single_dish` modes. Added `set_pending_clarification()` and `clear_pending_clarification()` helpers.

6. **`clarification` branch execution_result**: Changed from inline `{"success": True, "answer": answer}` to `_build_execution_result()` helper for consistent structure.

7. **Guardrail branch `query_plan` bug**: Fixed `UnboundLocalError` where `query_plan` was referenced before assignment in guardrail branch. Now passes `query_plan={}` to diagnostics.

8. **`last_execution_result` initialization**: Added `self.last_execution_result = {}` in `RecipeRAGSystem.__init__`. Set in all branches (clarification, guardrail, no-result, list, detail, stream).

9. **`_apply_resolved_target_to_query_plan`**: Added method and call in `ask_question()` between query_plan construction and preference propagation.

10. **`_wrap_stream_with_writeback`**: Extracted from inline closure to method on `RecipeRAGSystem`.

11. **`_extract_retrieved_dishes` and `_build_execution_result`**: Added helpers for structured execution result construction. Used in all branches.

---

## Task 1: Strengthen State Writeback Review Contract

**Files:**
- Modify: `code/C8/rag_modules/state_writeback_review.py`
- Modify: `code/C8/tests/test_state_writeback_review.py`

- [ ] **Step 1: Add failing tests for query-plan driven recommendation writeback**

Append to `code/C8/tests/test_state_writeback_review.py`:

```python
from rag_modules.state_writeback_review import review_state_writeback


def test_list_query_with_recommended_dishes_writes_recommendation_list_even_if_turn_type_is_domain():
    review = review_state_writeback(
        turn_info={"turn_type": "domain_query"},
        resolution=None,
        execution_result={
            "success": True,
            "recommended_dishes": ["扬州炒饭", "麻婆豆腐"],
        },
        answer="1. 扬州炒饭\n2. 麻婆豆腐",
        query_plan={"route_type": "list"},
    )

    assert review["should_write_reliable_state"] is True
    assert review["writeback_mode"] == "recommendation_list"


def test_failed_retrieval_is_message_only_even_with_dish_name():
    review = review_state_writeback(
        turn_info={"turn_type": "domain_query"},
        resolution=None,
        execution_result={
            "success": False,
            "resolved_target": "不存在的菜",
        },
        answer="抱歉，没有找到相关信息。",
        query_plan={"route_type": "detail", "dish_name": "不存在的菜"},
    )

    assert review["should_write_reliable_state"] is False
    assert review["writeback_mode"] == "message_only"


def test_resolved_followup_requires_successful_retrieval():
    review = review_state_writeback(
        turn_info={"turn_type": "followup_query"},
        resolution={
            "next_action": "apply_reference_resolution",
            "resolved_target": "蛋炒饭",
            "target_source": "implicit_single_dish_followup",
            "writeback_eligible": True,
        },
        execution_result={
            "success": True,
            "resolved_target": "蛋炒饭",
            "retrieved_dishes": ["蛋炒饭"],
        },
        answer="蛋炒饭的小技巧是先热锅再下油。",
        query_plan={"route_type": "detail", "dish_name": "蛋炒饭"},
    )

    assert review["should_write_reliable_state"] is True
    assert review["writeback_mode"] == "resolved_followup"


def test_ask_clarification_writes_pending_clarification_not_entity():
    review = review_state_writeback(
        turn_info={"turn_type": "followup_query"},
        resolution={
            "next_action": "ask_clarification",
            "clarification_question": "你指的是第几个推荐菜？",
            "reason": "ambiguous_reference",
            "candidates": ["蛋炒饭", "麻婆豆腐"],
        },
        execution_result={"success": True, "answer": "你指的是第几个推荐菜？"},
        answer="你指的是第几个推荐菜？",
        query_plan=None,
    )

    assert review["should_write_reliable_state"] is False
    assert review["writeback_mode"] == "clarification_pending"
```

- [ ] **Step 2: Run tests and confirm failure**

Run:

```powershell
Push-Location code/C8
C:\Users\lenovo\anaconda3\python.exe -m pytest tests/test_state_writeback_review.py -q -s
Pop-Location
```

Expected: FAIL because `review_state_writeback()` does not accept `query_plan` and does not return `resolved_followup` or `clarification_pending`.

- [ ] **Step 3: Update `review_state_writeback` signature and logic**

Replace `review_state_writeback()` in `code/C8/rag_modules/state_writeback_review.py` with:

```python
from __future__ import annotations


def _route_type(query_plan: dict | None, execution_result: dict) -> str | None:
    if query_plan and query_plan.get("route_type"):
        return query_plan["route_type"]
    return execution_result.get("route_type")


def review_state_writeback(
    turn_info: dict,
    resolution: dict | None,
    execution_result: dict,
    answer: str,
    query_plan: dict | None = None,
) -> dict:
    turn_type = turn_info.get("turn_type", "domain_query")

    if turn_type in {"smalltalk", "out_of_domain"}:
        return {"should_write_reliable_state": False, "writeback_mode": "message_only"}

    if resolution and resolution.get("next_action") == "ask_clarification":
        return {"should_write_reliable_state": False, "writeback_mode": "clarification_pending"}

    if execution_result.get("stream_interrupted"):
        return {"should_write_reliable_state": False, "writeback_mode": "message_only"}

    if not execution_result.get("success"):
        return {"should_write_reliable_state": False, "writeback_mode": "message_only"}

    if _route_type(query_plan, execution_result) == "list":
        if execution_result.get("recommended_dishes"):
            return {"should_write_reliable_state": True, "writeback_mode": "recommendation_list"}
        return {"should_write_reliable_state": False, "writeback_mode": "message_only"}

    if resolution and resolution.get("next_action") == "apply_correction":
        return {"should_write_reliable_state": True, "writeback_mode": "correction_turn"}

    if resolution and resolution.get("next_action") == "apply_reference_resolution":
        if execution_result.get("resolved_target"):
            return {"should_write_reliable_state": True, "writeback_mode": "resolved_followup"}
        return {"should_write_reliable_state": False, "writeback_mode": "message_only"}

    if query_plan and query_plan.get("route_type") == "detail" and query_plan.get("dish_name"):
        return {"should_write_reliable_state": True, "writeback_mode": "explicit_single_dish"}

    return {"should_write_reliable_state": True, "writeback_mode": "normal"}
```

- [ ] **Step 4: Run state writeback tests**

Run:

```powershell
Push-Location code/C8
C:\Users\lenovo\anaconda3\python.exe -m pytest tests/test_state_writeback_review.py -q -s
Pop-Location
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add code/C8/rag_modules/state_writeback_review.py code/C8/tests/test_state_writeback_review.py
git commit -m "test: define state writeback review contract"
```

---

## Task 2: Implement Explicit Conversation Writeback Modes

**Files:**
- Modify: `code/C8/rag_modules/conversation_manager.py`
- Modify: `code/C8/tests/test_conversation_state.py`

- [ ] **Step 1: Add failing tests for state pollution prevention**

Append to `code/C8/tests/test_conversation_state.py`:

```python
from rag_modules.conversation_manager import ConversationManager


def test_message_only_writeback_does_not_update_existing_entity():
    manager = ConversationManager()
    manager.set_current_dish("s1", "蛋炒饭", source="explicit_query", confidence=1.0)

    manager.writeback_turn_state(
        session_id="s1",
        question="你好",
        turn_info={"turn_type": "smalltalk"},
        query_plan=None,
        resolution=None,
        answer="你好，我可以帮你查菜谱。",
        execution_result={"success": True, "answer": "你好，我可以帮你查菜谱。"},
    )

    session = manager.get_session("s1")
    assert session.current_entity == "蛋炒饭"
    assert session.last_confirmed_target == "蛋炒饭"


def test_clarification_pending_does_not_update_entity():
    manager = ConversationManager()
    manager.record_recommendations("s2", ["蛋炒饭", "麻婆豆腐"])

    manager.writeback_turn_state(
        session_id="s2",
        question="它怎么做？",
        turn_info={"turn_type": "followup_query"},
        query_plan=None,
        resolution={
            "next_action": "ask_clarification",
            "reason": "ambiguous_reference",
            "candidates": ["蛋炒饭", "麻婆豆腐"],
            "clarification_question": "你指的是第几个推荐菜？",
        },
        answer="你指的是第几个推荐菜？",
        execution_result={"success": True, "answer": "你指的是第几个推荐菜？"},
    )

    session = manager.get_session("s2")
    assert session.topic_mode == "recommendation_list"
    assert session.current_entity is None
    assert session.pending_clarification["reason"] == "ambiguous_reference"
    assert session.pending_clarification["candidates"] == ["蛋炒饭", "麻婆豆腐"]


def test_recommendation_list_writeback_does_not_set_current_entity():
    manager = ConversationManager()
    manager.writeback_turn_state(
        session_id="s3",
        question="我晚上想吃点下饭的，有啥推荐？",
        turn_info={"turn_type": "domain_query"},
        query_plan={"route_type": "list"},
        resolution=None,
        answer="1. 扬州炒饭\n2. 麻婆豆腐",
        execution_result={
            "success": True,
            "recommended_dishes": ["扬州炒饭", "麻婆豆腐"],
            "route_type": "list",
        },
    )

    session = manager.get_session("s3")
    assert session.topic_mode == "recommendation_list"
    assert session.current_entity is None
    assert session.recent_recommendations[1]["dish_name"] == "麻婆豆腐"


def test_resolved_followup_sets_current_dish_after_success():
    manager = ConversationManager()
    manager.record_recommendations("s4", ["扬州炒饭", "麻婆豆腐"])

    manager.writeback_turn_state(
        session_id="s4",
        question="第二个怎么做？",
        turn_info={"turn_type": "followup_query"},
        query_plan={"route_type": "detail", "dish_name": "麻婆豆腐"},
        resolution={
            "next_action": "apply_reference_resolution",
            "resolved_target": "麻婆豆腐",
            "target_source": "ordinal_recommendation_reference",
            "confidence": 1.0,
        },
        answer="麻婆豆腐的做法是先炒豆瓣酱再下豆腐。",
        execution_result={
            "success": True,
            "resolved_target": "麻婆豆腐",
            "target_source": "ordinal_recommendation_reference",
            "retrieved_dishes": ["麻婆豆腐"],
        },
    )

    session = manager.get_session("s4")
    assert session.topic_mode == "single_dish"
    assert session.current_entity == "麻婆豆腐"
    assert session.last_confirmed_target == "麻婆豆腐"
```

- [ ] **Step 2: Run tests and confirm failure**

Run:

```powershell
Push-Location code/C8
C:\Users\lenovo\anaconda3\python.exe -m pytest tests/test_conversation_state.py -k "message_only_writeback or clarification_pending or recommendation_list_writeback or resolved_followup" -q -s
Pop-Location
```

Expected: FAIL because `writeback_turn_state()` does not pass `query_plan` into review and does not implement `clarification_pending`, `resolved_followup`, or `explicit_single_dish`.

- [ ] **Step 3: Add helper methods and explicit writeback modes**

In `code/C8/rag_modules/conversation_manager.py`, add these methods near `record_recommendations()`:

```python
    def set_pending_clarification(
        self,
        session_id: str,
        *,
        reason: str,
        candidates: List[str] | None,
        original_question: str,
        clarification_question: str,
    ):
        with self._lock:
            session = self.get_session(session_id)
            session.pending_clarification = {
                "reason": reason,
                "candidates": candidates or [],
                "original_question": original_question,
                "clarification_question": clarification_question,
                "updated_at": time.time(),
            }

    def clear_pending_clarification(self, session_id: str):
        with self._lock:
            session = self.get_session(session_id)
            session.pending_clarification = None
```

Then replace `writeback_turn_state()` with:

```python
    def writeback_turn_state(
        self,
        *,
        session_id: str,
        question: str,
        turn_info: dict,
        query_plan: dict | None = None,
        resolution: dict | None = None,
        answer: str = "",
        execution_result: dict | None = None,
    ):
        """Write one turn using only reliable structured execution facts."""
        from rag_modules.state_writeback_review import review_state_writeback

        execution_result = execution_result or {}
        review = review_state_writeback(
            turn_info=turn_info,
            resolution=resolution,
            execution_result=execution_result,
            answer=answer,
            query_plan=query_plan,
        )

        mode = review["writeback_mode"]

        if mode == "message_only":
            self.add_interaction(session_id, question, answer, intent_type=turn_info.get("turn_type", "general"), entities={})
            return

        if mode == "clarification_pending":
            self.set_pending_clarification(
                session_id,
                reason=(resolution or {}).get("reason", "ambiguous_reference"),
                candidates=(resolution or {}).get("candidates", []),
                original_question=question,
                clarification_question=(resolution or {}).get("clarification_question", answer),
            )
            self.add_interaction(session_id, question, answer, intent_type="clarification_pending", entities={})
            return

        if mode == "recommendation_list":
            self.clear_pending_clarification(session_id)
            self.record_recommendations(session_id, execution_result.get("recommended_dishes", []))
            self.add_interaction(session_id, question, answer, intent_type="list", entities={})
            return

        if mode in {"resolved_followup", "correction_turn"}:
            resolved = execution_result.get("resolved_target") or (resolution or {}).get("resolved_target")
            if resolved:
                self.clear_pending_clarification(session_id)
                self.set_current_dish(
                    session_id,
                    resolved,
                    source=(resolution or {}).get("target_source", mode),
                    confidence=(resolution or {}).get("confidence", 1.0 if mode == "correction_turn" else 0.8),
                )
                self.add_interaction(session_id, question, answer, intent_type=mode, entities={"dish_name": resolved})
            else:
                self.add_interaction(session_id, question, answer, intent_type=mode, entities={})
            return

        if mode == "explicit_single_dish":
            dish_name = (query_plan or {}).get("dish_name")
            if dish_name:
                self.clear_pending_clarification(session_id)
                self.set_current_dish(session_id, dish_name, source="explicit_query", confidence=1.0)
                self.add_interaction(session_id, question, answer, intent_type="detail", entities={"dish_name": dish_name})
            else:
                self.add_interaction(session_id, question, answer, intent_type="detail", entities={})
            return

        entities = query_plan.get("entities", {}) if query_plan else {}
        self.add_interaction(
            session_id,
            question,
            answer,
            intent_type=query_plan.get("route_type", "general") if query_plan else "general",
            entities=entities,
        )
```

- [ ] **Step 4: Prevent `add_interaction()` from hidden entity mutation**

In `add_interaction()`, remove this block:

```python
            if entities and entities.get("dish_name"):
                session.current_entity = entities["dish_name"]
```

After removal, entity updates happen only through `set_current_dish()`.

- [ ] **Step 5: Run focused conversation state tests**

Run:

```powershell
Push-Location code/C8
C:\Users\lenovo\anaconda3\python.exe -m pytest tests/test_conversation_state.py tests/test_state_writeback_review.py -q -s
Pop-Location
```

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add code/C8/rag_modules/conversation_manager.py code/C8/tests/test_conversation_state.py
git commit -m "feat: make conversation writeback explicit"
```

---

## Task 3: Lock Resolved Target Into Final Query Plan

**Files:**
- Modify: `code/C8/main.py`
- Modify: `code/C8/tests/test_conversation_state.py`

- [ ] **Step 1: Add unit tests for resolved-target query plan enforcement**

Append to `code/C8/tests/test_conversation_state.py`:

```python
from main import RecipeRAGSystem


def test_apply_resolved_target_to_query_plan_sets_dish_and_filter():
    system = RecipeRAGSystem()
    query_plan = {
        "route_type": "detail",
        "filters": {"content_type": "tips"},
        "dish_name": None,
        "entities": {},
    }
    resolution = {
        "next_action": "apply_reference_resolution",
        "resolved_target": "蛋炒饭",
        "target_source": "implicit_single_dish_followup",
    }

    updated = system._apply_resolved_target_to_query_plan(query_plan, resolution)

    assert updated["dish_name"] == "蛋炒饭"
    assert updated["filters"]["dish_name"] == "蛋炒饭"
    assert updated["entities"]["dish_name"] == "蛋炒饭"


def test_apply_resolved_target_to_query_plan_does_nothing_without_resolution():
    system = RecipeRAGSystem()
    query_plan = {
        "route_type": "detail",
        "filters": {"content_type": "tips"},
        "dish_name": None,
        "entities": {},
    }

    updated = system._apply_resolved_target_to_query_plan(query_plan, None)

    assert updated["dish_name"] is None
    assert "dish_name" not in updated["filters"]
```

- [ ] **Step 2: Run tests and confirm failure**

Run:

```powershell
Push-Location code/C8
C:\Users\lenovo\anaconda3\python.exe -m pytest tests/test_conversation_state.py -k "apply_resolved_target_to_query_plan" -q -s
Pop-Location
```

Expected: FAIL because `_apply_resolved_target_to_query_plan()` does not exist.

- [ ] **Step 3: Add query-plan enforcement helper**

In `code/C8/main.py`, add this method inside `RecipeRAGSystem`, near `_write_conversation_turn()`:

```python
    def _apply_resolved_target_to_query_plan(
        self,
        query_plan: Dict[str, Any],
        resolution: Dict[str, Any] | None,
    ) -> Dict[str, Any]:
        """Make guarded reference resolution override route-level dish extraction."""
        if not resolution:
            return query_plan
        if resolution.get("next_action") != "apply_reference_resolution":
            return query_plan
        resolved_target = resolution.get("resolved_target")
        if not resolved_target:
            return query_plan

        query_plan["dish_name"] = resolved_target
        query_plan.setdefault("filters", {})["dish_name"] = resolved_target
        query_plan.setdefault("entities", {})["dish_name"] = resolved_target
        return query_plan
```

- [ ] **Step 4: Call helper before retrieval**

In `ask_question()`, immediately after this block, add the enforcement call:

```python
        query_plan = (
            self._build_query_plan(rewritten_question, session_id)
            if rewritten_question != question
            else base_query_plan
        )
```

```python
        query_plan = self._apply_resolved_target_to_query_plan(query_plan, resolution)
```

This must happen before:

```python
        route_type = query_plan["route_type"]
        filters = query_plan["filters"]
        dish_name = query_plan["dish_name"]
```

- [ ] **Step 5: Run focused tests**

Run:

```powershell
Push-Location code/C8
C:\Users\lenovo\anaconda3\python.exe -m pytest tests/test_conversation_state.py -k "apply_resolved_target_to_query_plan" -q -s
Pop-Location
```

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add code/C8/main.py code/C8/tests/test_conversation_state.py
git commit -m "feat: lock resolved target into query plan"
```

---

## Task 4: Enrich Execution Result For Diagnostics And Tests

**Files:**
- Modify: `code/C8/main.py`
- Modify: `code/C8/tests/test_conversation_state.py`

- [ ] **Step 1: Add helper tests for retrieved dish extraction**

Append to `code/C8/tests/test_conversation_state.py`:

```python
from langchain_core.documents import Document
from main import RecipeRAGSystem


def test_extract_retrieved_dishes_deduplicates_parent_docs():
    system = RecipeRAGSystem()
    docs = [
        Document(page_content="a", metadata={"dish_name": "蛋炒饭"}),
        Document(page_content="b", metadata={"dish_name": "蛋炒饭"}),
        Document(page_content="c", metadata={"dish_name": "麻婆豆腐"}),
    ]

    assert system._extract_retrieved_dishes(docs) == ["蛋炒饭", "麻婆豆腐"]


def test_build_execution_result_includes_resolution_and_query_plan():
    system = RecipeRAGSystem()
    result = system._build_execution_result(
        success=True,
        answer="answer",
        rewritten_question="蛋炒饭有什么小技巧",
        original_question="有什么小技巧？",
        query_plan={
            "route_type": "detail",
            "filters": {"content_type": "tips", "dish_name": "蛋炒饭"},
            "dish_name": "蛋炒饭",
        },
        resolution={
            "resolved_target": "蛋炒饭",
            "target_source": "implicit_single_dish_followup",
        },
        parent_docs=[
            Document(page_content="a", metadata={"dish_name": "蛋炒饭"}),
        ],
    )

    assert result["success"] is True
    assert result["route_type"] == "detail"
    assert result["filters"]["dish_name"] == "蛋炒饭"
    assert result["resolved_target"] == "蛋炒饭"
    assert result["target_source"] == "implicit_single_dish_followup"
    assert result["retrieved_dishes"] == ["蛋炒饭"]
    assert result["query_plan_source"] == "rewritten"
```

- [ ] **Step 2: Run tests and confirm failure**

Run:

```powershell
Push-Location code/C8
C:\Users\lenovo\anaconda3\python.exe -m pytest tests/test_conversation_state.py -k "extract_retrieved_dishes or build_execution_result" -q -s
Pop-Location
```

Expected: FAIL because helper methods do not exist.

- [ ] **Step 3: Add execution result helpers**

In `code/C8/main.py`, add inside `RecipeRAGSystem`:

```python
    def _extract_retrieved_dishes(self, parent_docs: list) -> list[str]:
        dishes: list[str] = []
        for doc in parent_docs or []:
            dish_name = (getattr(doc, "metadata", {}) or {}).get("dish_name")
            if dish_name and dish_name not in dishes:
                dishes.append(dish_name)
        return dishes

    def _build_execution_result(
        self,
        *,
        success: bool,
        answer: Any,
        rewritten_question: str,
        original_question: str,
        query_plan: Dict[str, Any] | None,
        resolution: Dict[str, Any] | None,
        parent_docs: list | None = None,
        recommended_dishes: list[str] | None = None,
    ) -> Dict[str, Any]:
        query_plan = query_plan or {}
        resolution = resolution or {}
        return {
            "success": success,
            "answer": answer,
            "final_query_text": rewritten_question,
            "query_plan_source": "rewritten" if rewritten_question != original_question else "original",
            "route_type": query_plan.get("route_type"),
            "filters": dict(query_plan.get("filters") or {}),
            "dish_name": query_plan.get("dish_name"),
            "resolved_target": resolution.get("resolved_target") or query_plan.get("dish_name"),
            "target_source": resolution.get("target_source"),
            "retrieved_dishes": self._extract_retrieved_dishes(parent_docs or []),
            "recommended_dishes": recommended_dishes or [],
        }
```

- [ ] **Step 4: Use helper in list, detail, and no-result branches**

In `ask_question()`, replace each existing manual `execution_result` dictionary construction in the no-result, list, and detail branches with the helper calls below.

For no-result branch:

```python
            execution_result = self._build_execution_result(
                success=False,
                answer=answer,
                rewritten_question=rewritten_question,
                original_question=question,
                query_plan=query_plan,
                resolution=resolution,
                parent_docs=[],
            )
```

For list branch:

```python
            execution_result = self._build_execution_result(
                success=True,
                answer=answer,
                rewritten_question=rewritten_question,
                original_question=question,
                query_plan=query_plan,
                resolution=resolution,
                parent_docs=list(self._latest_parent_docs),
                recommended_dishes=recommended_dishes,
            )
```

For detail branch:

```python
            execution_result = self._build_execution_result(
                success=True,
                answer=answer,
                rewritten_question=rewritten_question,
                original_question=question,
                query_plan=query_plan,
                resolution=resolution,
                parent_docs=list(self._latest_parent_docs),
            )
```

- [ ] **Step 5: Add diagnostics storage for latest execution result**

At the end of each branch that builds `execution_result`, set:

```python
        self.last_execution_result = execution_result
```

Also initialize in `RecipeRAGSystem.__init__`:

```python
        self.last_execution_result = {}
```

- [ ] **Step 6: Run focused tests**

Run:

```powershell
Push-Location code/C8
C:\Users\lenovo\anaconda3\python.exe -m pytest tests/test_conversation_state.py -k "extract_retrieved_dishes or build_execution_result" -q -s
Pop-Location
```

Expected: PASS.

- [ ] **Step 7: Commit**

```powershell
git add code/C8/main.py code/C8/tests/test_conversation_state.py
git commit -m "feat: expose structured execution results"
```

---

## Task 5: Deprecate And Disconnect Old Production Semantic Path

**Files:**
- Modify: `code/C8/rag_modules/conversation_manager.py`
- Modify: `code/C8/main.py`
- Modify: `code/C8/tests/test_conversation_state.py`

- [ ] **Step 1: Add guard tests that old helpers are deprecated, not deleted yet**

Append to `code/C8/tests/test_conversation_state.py`:

```python
import pytest
import warnings
from rag_modules.conversation_manager import ConversationManager


def test_complete_query_is_deprecated_but_temporarily_available_for_coverage_comparison():
    manager = ConversationManager()

    with pytest.warns(DeprecationWarning, match="complete_query is deprecated"):
        result = manager.complete_query("s1", "它怎么做？")

    assert result == "它怎么做？"


def test_old_string_reference_helpers_emit_deprecation_warnings():
    manager = ConversationManager()

    with pytest.warns(DeprecationWarning, match="_resolve_entity_references is deprecated"):
        assert manager._resolve_entity_references("它怎么做？", "蛋炒饭") == "蛋炒饭怎么做？"

    with pytest.warns(DeprecationWarning, match="_inherit_intent is deprecated"):
        assert manager._inherit_intent("怎么做", "蛋炒饭", "detail") == "蛋炒饭怎么做"
```

- [ ] **Step 2: Run tests and confirm failure**

Run:

```powershell
Push-Location code/C8
C:\Users\lenovo\anaconda3\python.exe -m pytest tests/test_conversation_state.py -k "complete_query_is_deprecated or old_string_reference_helpers" -q -s
Pop-Location
```

Expected: FAIL because methods exist but do not emit `DeprecationWarning` yet.

- [ ] **Step 3: Add explicit deprecation warnings without deleting behavior**

In `code/C8/rag_modules/conversation_manager.py`, add:

```python
import warnings
```

At the start of `complete_query()`, add:

```python
    def complete_query(
        self,
        session_id: str,
        query: str,
        extracted_intent: Dict[str, Any] | None = None,
    ) -> str:
        warnings.warn(
            "ConversationManager.complete_query is deprecated; production flow must use "
            "build_conversation_snapshot + resolve_reference_from_snapshot.",
            DeprecationWarning,
            stacklevel=2,
        )
        session = self.get_session(session_id)
        # Keep the existing body below during this plan until coverage verification passes.
```

- [ ] **Step 4: Add deprecation warnings to old private helpers**

At the start of `_resolve_entity_references()`, add:

```python
        warnings.warn(
            "ConversationManager._resolve_entity_references is deprecated; use reference_resolution.",
            DeprecationWarning,
            stacklevel=2,
        )
```

At the start of `_inherit_intent()`, add:

```python
        warnings.warn(
            "ConversationManager._inherit_intent is deprecated; use rewrite_query_for_execution.",
            DeprecationWarning,
            stacklevel=2,
        )
```

- [ ] **Step 5: Assert no production code calls deprecated helpers**

Run:

```powershell
rg -n "_is_intent_switch|_resolve_entity_references|_inherit_intent|complete_query" code/C8
```

Expected after this task:

- `complete_query` appears in `conversation_manager.py`, tests, docs/plans/specs only.
- `_resolve_entity_references` appears in `conversation_manager.py`, tests, docs/plans/specs only.
- `_inherit_intent` appears in `conversation_manager.py`, tests, docs/plans/specs only.
- No `main.py`, `web_app.py`, or generation module code calls them.

- [ ] **Step 6: Neuter `_should_inherit_current_entity` as a query mutation path**

Search:

```powershell
rg -n "_should_inherit_current_entity" code/C8/main.py
```

If `_build_query_plan()` uses `_should_inherit_current_entity()` to directly inherit `current_entity`, remove only that mutation path. Keep the helper temporarily if other code still uses it for feature detection. Add this comment where the mutation path was removed:

```python
# Do not inherit current_entity here. Reference inheritance must go through
# build_conversation_snapshot -> resolve_reference_from_snapshot.
```

- [ ] **Step 7: Run old-path deprecation tests**

Run:

```powershell
Push-Location code/C8
C:\Users\lenovo\anaconda3\python.exe -m pytest tests/test_conversation_state.py -k "complete_query_is_deprecated or old_string_reference_helpers" -q -s
Pop-Location
```

Expected: PASS.

- [ ] **Step 8: Commit**

```powershell
git add code/C8/rag_modules/conversation_manager.py code/C8/main.py code/C8/tests/test_conversation_state.py
git commit -m "refactor: deprecate old conversation completion path"
```

---

## Task 6: Strengthen Real Integration Tests

**Files:**
- Modify: `code/C8/tests/test_conversation_integration_real.py`

- [ ] **Step 1: Add helper functions for real integration assertions**

At the top of `code/C8/tests/test_conversation_integration_real.py`, after imports, add:

```python
def _require_api_key():
    if not os.getenv("DASHSCOPE_API_KEY"):
        pytest.skip("DASHSCOPE_API_KEY is required for real integration tests")


def _session(app, session_id):
    system = app.config["RAG_SYSTEM"]
    return system.generation_module.conversation_manager.get_session(session_id)


def _last_execution(app):
    system = app.config["RAG_SYSTEM"]
    return getattr(system, "last_execution_result", {})


def _post(client, question, session_id):
    response = client.post("/api/chat", json={"question": question, "session_id": session_id})
    assert response.status_code == 200
    data = response.get_json()
    assert data and "answer" in data
    return data["answer"]
```

- [ ] **Step 2: Replace weak ordinal test with state-based assertions**

Replace `test_real_ordinal_followup_uses_recommendation_rank()` with:

```python
@pytest.mark.real_integration
def test_real_ordinal_followup_uses_recommendation_rank():
    _require_api_key()

    app = create_app()
    client = app.test_client()
    session_id = "real-ordinal-session"

    first_answer = _post(client, "我晚上想吃点下饭的，有啥推荐？", session_id)
    session = _session(app, session_id)
    assert session.topic_mode == "recommendation_list"
    assert len(session.recent_recommendations) >= 2
    second_dish = session.recent_recommendations[1]["dish_name"]
    assert second_dish in first_answer

    second_answer = _post(client, "第二个怎么做？", session_id)
    execution = _last_execution(app)

    assert execution["resolved_target"] == second_dish
    assert execution["target_source"] == "ordinal_recommendation_reference"
    assert execution["retrieved_dishes"] == [second_dish]
    assert "我没找到你说的这个序号" not in second_answer
```

- [ ] **Step 3: Replace weak short follow-up test with retrieval target assertions**

Replace `test_real_single_dish_short_tip_followup_uses_current_dish()` with:

```python
@pytest.mark.real_integration
def test_real_single_dish_short_tip_followup_uses_current_dish():
    _require_api_key()

    app = create_app()
    client = app.test_client()
    session_id = "real-short-followup-session"

    _post(client, "蛋炒饭怎么做？", session_id)
    answer = _post(client, "有什么小技巧别粘锅？", session_id)
    execution = _last_execution(app)
    session = _session(app, session_id)

    assert execution["resolved_target"] == "蛋炒饭"
    assert execution["filters"]["dish_name"] == "蛋炒饭"
    assert execution["retrieved_dishes"] == ["蛋炒饭"]
    assert session.current_entity == "蛋炒饭"
    assert "美式炒蛋" not in answer
    assert "黄油煎虾" not in answer
```

- [ ] **Step 4: Add natural breakfast ordinal test**

Append:

```python
@pytest.mark.real_integration
def test_real_ordinal_with_comment_uses_first_recommendation():
    _require_api_key()

    app = create_app()
    client = app.test_client()
    session_id = "real-breakfast-ordinal-session"

    first_answer = _post(client, "有没有适合新手的早餐？", session_id)
    session = _session(app, session_id)
    assert session.topic_mode == "recommendation_list"
    assert session.recent_recommendations
    first_dish = session.recent_recommendations[0]["dish_name"]
    assert first_dish in first_answer

    answer = _post(client, "第一个看起来不错，做法说一下", session_id)
    execution = _last_execution(app)

    assert execution["resolved_target"] == first_dish
    assert execution["retrieved_dishes"] == [first_dish]
    assert "看起来不错" not in execution["resolved_target"]
    assert "我没找到你说的这个序号" not in answer
```

- [ ] **Step 5: Strengthen recommendation pronoun clarification test**

Update `test_recommendation_followup_requires_clarification_not_wrong_inheritance()` assertions:

```python
    session = _session(app, session_id)
    assert session.topic_mode == "recommendation_list"
    assert session.current_entity is None
    assert session.pending_clarification is not None
    assert session.pending_clarification["reason"] in {"ambiguous_reference", "pronoun_in_recommendation_list"}
```

- [ ] **Step 6: Run real integration tests and confirm failures before implementation tasks are complete**

Run with the real API key from `code/C8/.env`:

```powershell
$line = [string](@(Get-Content code/C8/.env | Where-Object { $_ -match 'DASHSCOPE_API_KEY=' })[0])
$env:DASHSCOPE_API_KEY = (($line -split 'DASHSCOPE_API_KEY=',2)[1]).Trim()
Push-Location code/C8
C:\Users\lenovo\anaconda3\python.exe -m pytest tests/test_conversation_integration_real.py -m real_integration -q -s
Pop-Location
```

Expected before Tasks 1-5 are complete: FAIL on natural recommendation writeback or retrieved target assertions. Expected after Tasks 1-5: PASS.

- [ ] **Step 7: Commit**

```powershell
git add code/C8/tests/test_conversation_integration_real.py
git commit -m "test: assert real conversation state and retrieval targets"
```

---

## Task 7: Add Stream Equivalence And Interruption Tests

**Files:**
- Modify: `code/C8/tests/test_web_app.py`
- Modify: `code/C8/main.py` only if tests expose missing behavior.

- [ ] **Step 1: Add fake system for stream tests**

Append to `code/C8/tests/test_web_app.py`:

```python
class RecordingFakeSystem:
    def __init__(self):
        self.calls = []

    def ask_question(self, question, stream=False, session_id="default"):
        self.calls.append({"question": question, "stream": stream, "session_id": session_id})
        if stream:
            def generate():
                yield "第一段"
                yield "第二段"
            return generate()
        return "第一段第二段"


def _stream_body(response):
    return b"".join(response.response).decode("utf-8")
```

- [ ] **Step 2: Add stream endpoint equivalence test**

Append:

```python
def test_stream_endpoint_uses_same_question_and_session_as_chat():
    fake = RecordingFakeSystem()
    app = create_app(system_factory=lambda: fake)
    client = app.test_client()

    normal = client.post("/api/chat", json={"question": "蛋炒饭怎么做？", "session_id": "s1"})
    assert normal.status_code == 200
    assert normal.get_json()["answer"] == "第一段第二段"

    streamed = client.get("/api/chat/stream", query_string={"question": "蛋炒饭怎么做？", "session_id": "s1"})
    assert streamed.status_code == 200
    body = _stream_body(streamed)

    assert "event: message" in body
    assert "第一段" in body
    assert "第二段" in body
    assert "event: done" in body
    assert fake.calls == [
        {"question": "蛋炒饭怎么做？", "stream": False, "session_id": "s1"},
        {"question": "蛋炒饭怎么做？", "stream": True, "session_id": "s1"},
    ]
```

- [ ] **Step 3: Add stream generator writeback unit test**

Append to `code/C8/tests/test_conversation_state.py`:

```python
def test_stream_writeback_happens_only_after_generator_is_consumed(monkeypatch):
    from main import RecipeRAGSystem

    system = RecipeRAGSystem()
    writes = []

    def fake_write(**kwargs):
        writes.append(kwargs)

    monkeypatch.setattr(system, "_write_conversation_turn", fake_write)

    def chunks():
        yield "第一段"
        yield "第二段"

    wrapped = system._wrap_stream_with_writeback(
        answer_stream=chunks(),
        session_id="s1",
        question="蛋炒饭怎么做？",
        turn_info={"turn_type": "domain_query"},
        query_plan={"route_type": "detail", "dish_name": "蛋炒饭"},
        resolution=None,
        execution_result={"success": False, "answer": ""},
    )

    assert writes == []
    assert next(wrapped) == "第一段"
    assert writes == []
    assert "".join(wrapped) == "第二段"
    assert len(writes) == 1
    assert writes[0]["answer"] == "第一段第二段"
    assert writes[0]["execution_result"]["success"] is True
```

- [ ] **Step 4: Extract stream wrapping helper from inline code**

In `code/C8/main.py`, add:

```python
    def _wrap_stream_with_writeback(
        self,
        *,
        answer_stream,
        session_id: str,
        question: str,
        turn_info: dict,
        query_plan: dict | None,
        resolution: dict | None,
        execution_result: dict,
    ):
        collected = []
        for chunk in answer_stream:
            collected.append(chunk)
            yield chunk
        full_text = "".join(collected)
        execution_result["answer"] = full_text
        execution_result["success"] = True
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

Then replace the inline `_stream_with_writeback()` definition in `ask_question()` with:

```python
            return self._wrap_stream_with_writeback(
                answer_stream=answer,
                session_id=session_id,
                question=question,
                turn_info=turn_info,
                query_plan=query_plan,
                resolution=resolution,
                execution_result=execution_result,
            )
```

- [ ] **Step 5: Run stream tests**

Run:

```powershell
Push-Location code/C8
C:\Users\lenovo\anaconda3\python.exe -m pytest tests/test_web_app.py tests/test_conversation_state.py -k "stream" -q -s
Pop-Location
```

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add code/C8/main.py code/C8/tests/test_web_app.py code/C8/tests/test_conversation_state.py
git commit -m "test: cover stream conversation writeback"
```

---

## Task 8: Full Regression Verification

**Files:**
- No source edits unless a previous task introduced regressions.

- [ ] **Step 1: Run syntax verification**

Run:

```powershell
C:\Users\lenovo\anaconda3\python.exe -m py_compile `
  code/C8/main.py `
  code/C8/web_app.py `
  code/C8/rag_modules/conversation_manager.py `
  code/C8/rag_modules/state_writeback_review.py `
  code/C8/rag_modules/reference_resolution.py `
  code/C8/rag_modules/conversation_state_builder.py `
  code/C8/rag_modules/turn_qualification.py
```

Expected: exit code 0.

- [ ] **Step 2: Run focused unit suite**

Run:

```powershell
Push-Location code/C8
C:\Users\lenovo\anaconda3\python.exe -m pytest `
  tests/test_turn_qualification.py `
  tests/test_reference_resolution.py `
  tests/test_conversation_state.py `
  tests/test_state_writeback_review.py `
  tests/test_web_app.py `
  -q -s
Pop-Location
```

Expected: PASS.

- [ ] **Step 3: Run real integration suite**

Run:

```powershell
$line = [string](@(Get-Content code/C8/.env | Where-Object { $_ -match 'DASHSCOPE_API_KEY=' })[0])
$env:DASHSCOPE_API_KEY = (($line -split 'DASHSCOPE_API_KEY=',2)[1]).Trim()
Push-Location code/C8
C:\Users\lenovo\anaconda3\python.exe -m pytest tests/test_conversation_integration_real.py -m real_integration -q -s
Pop-Location
```

Expected: PASS. If a model response changes wording, assertions should still pass because they verify state and retrieval targets, not prose style.

- [ ] **Step 4: Manually run three real user-like chains**

Run this script:

```powershell
$line = [string](@(Get-Content code/C8/.env | Where-Object { $_ -match 'DASHSCOPE_API_KEY=' })[0])
$env:DASHSCOPE_API_KEY = (($line -split 'DASHSCOPE_API_KEY=',2)[1]).Trim()
Push-Location code/C8
@'
from web_app import create_app

chains = {
    "natural_ordinal": [
        "我晚上想吃点下饭的，有啥推荐？",
        "第二个怎么做？",
    ],
    "single_dish_tip": [
        "蛋炒饭怎么做？",
        "有什么小技巧别粘锅？",
    ],
    "pronoun_clarify": [
        "今天吃什么？",
        "它怎么做？",
    ],
}

app = create_app()
client = app.test_client()
system = app.config["RAG_SYSTEM"]

for session_id, questions in chains.items():
    print("\\n==", session_id, "==")
    for q in questions:
        response = client.post("/api/chat", json={"question": q, "session_id": session_id})
        print("Q:", q)
        print("A:", response.get_json()["answer"])
        print("execution:", getattr(system, "last_execution_result", {}))
    session = system.generation_module.conversation_manager.get_session(session_id)
    print("topic_mode:", session.topic_mode)
    print("current_entity:", session.current_entity)
    print("recent_recommendations:", session.recent_recommendations)
    print("pending_clarification:", session.pending_clarification)
'@ | C:\Users\lenovo\anaconda3\python.exe -
Pop-Location
```

Expected:

- `natural_ordinal`: Q1 writes recommendations; Q2 resolves rank 2 and retrieves only that dish.
- `single_dish_tip`: Q2 resolves `蛋炒饭` and retrieves only `蛋炒饭`.
- `pronoun_clarify`: Q2 asks clarification and does not set `current_entity`.

- [ ] **Step 5: Commit verification-only docs if needed**

If only tests and code changed, no extra commit is needed. If a verification note is added, commit it:

```powershell
git add code/C8/docs/superpowers/plans/2026-07-04-conversation-architecture-completion.md
git commit -m "docs: add conversation architecture completion plan"
```

---

## Task 9: Remove Deprecated Old Helpers After Coverage Verification

**Files:**
- Modify: `code/C8/rag_modules/conversation_manager.py`
- Modify: `code/C8/tests/test_conversation_state.py`

This task must run only after Task 8 passes. It is intentionally placed after full real-chain verification so old behavior is not removed before the new architecture proves coverage.

- [ ] **Step 1: Verify there are no production callers**

Run:

```powershell
rg -n "complete_query\(|_resolve_entity_references\(|_inherit_intent\(" code/C8/main.py code/C8/web_app.py code/C8/rag_modules code/C8/tests
```

Expected before deletion:

- `complete_query`, `_resolve_entity_references`, and `_inherit_intent` appear in `conversation_manager.py`.
- They may appear in deprecation tests.
- They do not appear in `main.py`, `web_app.py`, `generation_integration.py`, or other production callers.

- [ ] **Step 2: Replace deprecation tests with removal tests**

In `code/C8/tests/test_conversation_state.py`, replace the Task 5 deprecation tests with:

```python
from rag_modules.conversation_manager import ConversationManager


def test_old_conversation_completion_helpers_are_removed_after_new_path_coverage():
    manager = ConversationManager()

    assert not hasattr(manager, "complete_query")
    assert not hasattr(manager, "_resolve_entity_references")
    assert not hasattr(manager, "_inherit_intent")
```

- [ ] **Step 3: Run removal test and confirm failure**

Run:

```powershell
Push-Location code/C8
C:\Users\lenovo\anaconda3\python.exe -m pytest tests/test_conversation_state.py -k "old_conversation_completion_helpers_are_removed" -q -s
Pop-Location
```

Expected: FAIL because the deprecated helpers still exist.

- [ ] **Step 4: Physically delete deprecated helpers**

Delete these methods from `code/C8/rag_modules/conversation_manager.py`:

```python
def complete_query(self, session_id: str, query: str, extracted_intent: Dict[str, Any] | None = None) -> str
def _resolve_entity_references(self, query: str, current_entity: Optional[str]) -> str
def _inherit_intent(self, query: str, current_entity: Optional[str], current_intent: str) -> str
```

Also remove `import warnings` if no longer used.

Do not delete:

```python
def reset_session(self, session_id: str)
def _reset_session(self, session_id: str)
def record_recommendations(self, session_id: str, dishes: List[str])
def set_current_dish(self, session_id: str, dish_name: str, source: str, confidence: float, updated_at: float | None = None)
def add_interaction(self, session_id: str, user_query: str, assistant_response: str, intent_type: str = "general", entities: Dict[str, Any] | None = None)
```

- [ ] **Step 5: Run removal and full focused tests**

Run:

```powershell
Push-Location code/C8
C:\Users\lenovo\anaconda3\python.exe -m pytest `
  tests/test_conversation_state.py `
  tests/test_state_writeback_review.py `
  tests/test_reference_resolution.py `
  tests/test_turn_qualification.py `
  -q -s
Pop-Location
```

Expected: PASS.

- [ ] **Step 6: Re-run production caller grep**

Run:

```powershell
rg -n "complete_query\(|_resolve_entity_references\(|_inherit_intent\(" code/C8/main.py code/C8/web_app.py code/C8/rag_modules
```

Expected: no production matches.

- [ ] **Step 7: Commit**

```powershell
git add code/C8/rag_modules/conversation_manager.py code/C8/tests/test_conversation_state.py
git commit -m "refactor: remove deprecated conversation completion helpers"
```

---

## Self-Review Checklist

- [ ] Every SPEC requirement has at least one task:
  - query-plan driven recommendation writeback: Task 1, Task 2, Task 6
  - resolved target locked into query plan: Task 3, Task 6
  - execution result observability: Task 4
  - old logic deprecation/disconnection: Task 5
  - old logic physical removal after coverage verification: Task 9
  - state pollution prevention: Task 2, Task 6
  - stream equivalence and interruption behavior: Task 7
  - real integration hard assertions: Task 6, Task 8
- [ ] No implementation step relies on answer text parsing for entity state.
- [ ] No task restores old ordinal resolver or old recommendation cache.
- [ ] Tests use state and structured diagnostics wherever possible.
- [ ] Stream tests verify writeback timing rather than only SSE formatting.
