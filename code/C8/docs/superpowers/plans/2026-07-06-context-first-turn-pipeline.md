# Context First Turn Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the old pre-context `check_front_door -> qualify_turn -> snapshot` chain with the frozen-architecture order `basic_safety_gate -> snapshot -> understand_turn -> reference resolution -> execution plan`.

**Architecture:** Stage 02 introduces a narrow safety gate and a context-aware turn understanding module. Production calls move to the new contracts; old mixed-responsibility paths are deleted after their behavior is covered by the new modules.

**Tech Stack:** Python, pytest, existing `RecipeRAGSystem`, existing conversation snapshot/reference resolution/execution planning modules.

---

## Cutover Contract

This plan assumes Stage 01 state writeback policy is implemented.

Old responsibility being replaced:

- `front_door_guardrail.check_front_door()` currently mixes basic blocking, smalltalk, and harmless out-of-domain direct replies.
- `turn_qualification.qualify_turn()` classifies turns before snapshot exists.

New ownership:

- `front_door_guardrail.basic_safety_gate()` owns only empty, punctuation-only, unsafe, or invalid-input blocking.
- `turn_understanding.understand_turn(question, snapshot)` owns smalltalk, domain rejection, retrieval action, reference trigger, and answer mode hint.
- `execution_planner.build_execution_plan(turn_info, resolution)` consumes the new action contract and must not re-own domain rejection.

Production cutover:

```text
RecipeRAGSystem.ask_question
  old: check_front_door(question) -> qualify_turn(question) -> build_conversation_snapshot(...)
  new: basic_safety_gate(question) -> build_conversation_snapshot(...) -> understand_turn(question, snapshot)
```

Illegal after cutover:

- `main.py` importing or calling `qualify_turn`.
- `main.py` importing or calling `check_front_door`.
- `front_door_guardrail.py` returning `direct_reply`.
- `front_door_guardrail.py` blocking isolated references like `这个`, `它`, `第一个`.
- `test_turn_qualification.py` and old front-door tests asserting pre-context behavior.

Deletion before acceptance:

- Delete `rag_modules/turn_qualification.py`.
- Delete `tests/test_turn_qualification.py`.
- Replace `tests/test_front_door_guardrail.py` with safety-gate tests, or rewrite it so it imports only `basic_safety_gate`.

---

## File Structure

- Modify `code/C8/rag_modules/front_door_guardrail.py`
  - Keep only the basic safety gate contract.
  - Export `basic_safety_gate(query: str) -> dict`.
  - Do not export `check_front_door` after cutover.

- Create `code/C8/rag_modules/turn_understanding.py`
  - Export `understand_turn(question: str, snapshot: dict) -> dict`.
  - Own context-aware domain, smalltalk, reference trigger, and retrieval decision.

- Modify `code/C8/rag_modules/execution_planner.py`
  - Consume `turn_info["action"]`.
  - Direct `domain_reject` and `smalltalk` to direct-answer execution.
  - Preserve clarification and reference-resolution actions.

- Modify `code/C8/main.py`
  - Replace old order with `basic_safety_gate -> snapshot -> understand_turn`.
  - Keep existing retrieval, query planning, generation, and writeback paths.

- Modify `code/C8/tests/test_front_door_guardrail.py`
  - Rewrite as `basic_safety_gate` tests.

- Create `code/C8/tests/test_turn_understanding.py`
  - Test the new action contract and context-aware decisions.

- Modify `code/C8/tests/test_conversation_state.py`
  - Update integration assertions that currently depend on old front-door or pre-context qualification behavior.

- Create `code/C8/tests/test_context_first_cutover.py`
  - Source-level cutover checks proving production no longer imports/calls old path names.

- Delete `code/C8/rag_modules/turn_qualification.py`
  - Deleted after `main.py` and tests no longer import it.

- Delete `code/C8/tests/test_turn_qualification.py`
  - Replaced by `test_turn_understanding.py`.

---

## Task 1: Narrow The Basic Safety Gate

**Files:**
- Modify: `code/C8/rag_modules/front_door_guardrail.py`
- Modify: `code/C8/tests/test_front_door_guardrail.py`

- [ ] **Step 1: Replace old front-door tests with basic safety gate tests**

Replace `code/C8/tests/test_front_door_guardrail.py` with:

```python
from rag_modules.front_door_guardrail import basic_safety_gate


def test_basic_safety_gate_blocks_empty_and_punctuation_only_inputs():
    for query in ["", " ", "？", "!!!", "..."]:
        result = basic_safety_gate(query)

        assert result == {
            "decision": "block",
            "reason": "empty_or_punctuation",
            "message": "请输入一个具体的食谱或做菜问题。",
        }


def test_basic_safety_gate_continues_isolated_references_for_snapshot_handling():
    for query in ["这个", "它", "第一个", "那道菜"]:
        assert basic_safety_gate(query) == {
            "decision": "continue",
            "reason": "default_continue",
            "message": None,
        }


def test_basic_safety_gate_does_not_classify_smalltalk_or_domain():
    for query in ["你好", "谢谢", "Python怎么学", "股票怎么买", "蛋炒饭怎么做"]:
        assert basic_safety_gate(query) == {
            "decision": "continue",
            "reason": "default_continue",
            "message": None,
        }


def test_basic_safety_gate_result_shape_has_no_semantic_fields():
    forbidden = {
        "dish_name",
        "intent_type",
        "route_type",
        "filters",
        "content_type",
        "semantic_result",
        "rewritten_query",
        "action",
        "answer_mode_hint",
    }

    result = basic_safety_gate("第一个怎么做")

    assert set(result) == {"decision", "reason", "message"}
    assert forbidden.isdisjoint(result)
```

- [ ] **Step 2: Run the safety-gate tests and verify they fail**

Run:

```bash
cd code/C8
pytest tests/test_front_door_guardrail.py -q
```

Expected:

- FAIL because `basic_safety_gate` does not exist yet, or because isolated references are still blocked by the old front-door behavior.

- [ ] **Step 3: Replace `front_door_guardrail.py` with the narrow safety gate**

Replace `code/C8/rag_modules/front_door_guardrail.py` with:

```python
"""Basic safety gate for the context-first turn pipeline.

This module only decides whether a query is structurally safe enough to enter
the context-aware runtime. It does not classify domain, smalltalk, recipes,
references, dishes, filters, routes, answer modes, or rewritten queries.
"""

from __future__ import annotations

import re
from typing import Dict


def _normalize(query: str) -> str:
    return query.strip()


def _is_empty_or_punctuation(text: str) -> bool:
    if not text:
        return True
    return re.fullmatch(r"[\s\W_]+", text) is not None


def _block(reason: str, message: str) -> Dict[str, str | None]:
    return {"decision": "block", "reason": reason, "message": message}


def _continue() -> Dict[str, str | None]:
    return {"decision": "continue", "reason": "default_continue", "message": None}


def basic_safety_gate(query: str) -> Dict[str, str | None]:
    """Return whether a query may enter context-aware turn understanding."""
    normalized = _normalize(query)
    if _is_empty_or_punctuation(normalized):
        return _block("empty_or_punctuation", "请输入一个具体的食谱或做菜问题。")
    return _continue()
```

- [ ] **Step 4: Run the safety-gate tests and verify they pass**

Run:

```bash
cd code/C8
pytest tests/test_front_door_guardrail.py -q
```

Expected:

- PASS.

- [ ] **Step 5: Commit**

```bash
git add code/C8/rag_modules/front_door_guardrail.py code/C8/tests/test_front_door_guardrail.py
git commit -m "refactor: narrow front door to basic safety gate"
```

---

## Task 2: Add Context-Aware Turn Understanding

**Files:**
- Create: `code/C8/rag_modules/turn_understanding.py`
- Create: `code/C8/tests/test_turn_understanding.py`

- [ ] **Step 1: Write the turn understanding tests**

Create `code/C8/tests/test_turn_understanding.py`:

```python
from rag_modules.turn_understanding import understand_turn


def _snapshot(
    current_dish=None,
    recent_recommendations=None,
    pending_clarification=False,
):
    current_meta = {
        "value": current_dish,
        "active": bool(current_dish),
        "source": "confirmed" if current_dish else "none",
        "confidence": 1.0 if current_dish else 0.0,
    }
    return {
        "reference_state": {
            "current_dish": current_meta,
            "recent_recommendations": recent_recommendations or [],
        },
        "resolution_constraints": {
            "allowed_reference_targets": [
                item["dish_name"] for item in (recent_recommendations or [])
            ] or ([current_dish] if current_dish else []),
        },
        "state_health": {
            "has_pending_clarification": pending_clarification,
        },
    }


def test_smalltalk_is_direct_and_non_retrieval():
    result = understand_turn("谢谢", _snapshot(current_dish="蛋炒饭"))

    assert result["action"] == "smalltalk"
    assert result["answer_mode_hint"] == "safe_direct"
    assert result["should_retrieve"] is False
    assert result["needs_reference_resolution"] is False


def test_harmless_out_of_domain_is_domain_reject_after_snapshot_exists():
    result = understand_turn("Python怎么学", _snapshot())

    assert result["action"] == "domain_reject"
    assert result["answer_mode_hint"] == "safe_direct"
    assert result["should_retrieve"] is False
    assert result["needs_reference_resolution"] is False


def test_recipe_list_query_retrieves_list():
    result = understand_turn("今天吃什么", _snapshot())

    assert result["action"] == "retrieve_list"
    assert result["answer_mode_hint"] == "recommendation"
    assert result["should_retrieve"] is True
    assert result["reference_trigger"] == "none"


def test_recipe_detail_query_retrieves_detail_without_reference_resolution():
    result = understand_turn("蛋炒饭怎么做", _snapshot())

    assert result["action"] == "retrieve_detail"
    assert result["answer_mode_hint"] == "recipe_detail"
    assert result["should_retrieve"] is True
    assert result["needs_reference_resolution"] is False


def test_ordinal_recipe_followup_uses_reference_resolution():
    snapshot = _snapshot(
        recent_recommendations=[
            {"dish_name": "蛋炒饭"},
            {"dish_name": "番茄炒蛋"},
        ]
    )

    result = understand_turn("第一个怎么做", snapshot)

    assert result["action"] == "retrieve_detail"
    assert result["reference_trigger"] == "ordinal_reference"
    assert result["needs_reference_resolution"] is True
    assert result["depends_on_state"] is True


def test_ordinal_constraint_followup_is_not_domain_rejected_before_snapshot():
    snapshot = _snapshot(recent_recommendations=[{"dish_name": "鸡胸肉沙拉"}])

    result = understand_turn("第一个适合减脂吗", snapshot)

    assert result["action"] == "retrieve_detail"
    assert result["reference_trigger"] == "ordinal_reference"
    assert result["needs_reference_resolution"] is True


def test_ordinal_non_recipe_intent_does_not_resolve_to_recipe_detail():
    snapshot = _snapshot(recent_recommendations=[{"dish_name": "蛋炒饭"}])

    result = understand_turn("第一个作者是谁", snapshot)

    assert result["action"] == "domain_reject"
    assert result["should_retrieve"] is False
    assert result["needs_reference_resolution"] is False


def test_pronoun_followup_with_current_dish_uses_reference_resolution():
    result = understand_turn("这个能不放辣吗", _snapshot(current_dish="宫保鸡丁"))

    assert result["action"] in {"retrieve_detail", "substitution"}
    assert result["reference_trigger"] == "pronoun"
    assert result["needs_reference_resolution"] is True


def test_pronoun_followup_without_state_requests_reference_resolution_for_clarification():
    result = understand_turn("它呢", _snapshot())

    assert result["action"] == "retrieve_detail"
    assert result["reference_trigger"] == "pronoun"
    assert result["needs_reference_resolution"] is True
```

- [ ] **Step 2: Run the turn understanding tests and verify they fail**

Run:

```bash
cd code/C8
pytest tests/test_turn_understanding.py -q
```

Expected:

- FAIL because `rag_modules.turn_understanding` does not exist.

- [ ] **Step 3: Implement `turn_understanding.py`**

Create `code/C8/rag_modules/turn_understanding.py`:

```python
"""Context-aware turn understanding for the runtime pipeline."""

from __future__ import annotations

from typing import Any, Dict


SMALLTALK_EXACT = {
    "你好",
    "您好",
    "谢谢",
    "多谢",
    "哈哈",
    "你是谁",
    "你能做什么",
}

OUT_OF_DOMAIN_KEYWORDS = {
    "Python",
    "python",
    "Java",
    "C++",
    "代码",
    "编程",
    "算法",
    "股票",
    "房价",
    "天气",
    "新闻",
    "政治",
    "历史",
    "手机壳",
}

RECIPE_ALLOW_SIGNALS = {
    "菜",
    "食谱",
    "做法",
    "怎么做",
    "怎么煮",
    "怎么炒",
    "食材",
    "材料",
    "配料",
    "推荐",
    "吃什么",
    "减脂",
    "热量",
    "不辣",
    "辣",
    "下饭",
    "早餐",
    "午餐",
    "晚餐",
    "蛋",
    "鸡",
    "肉",
    "鱼",
    "虾",
    "饭",
    "面",
    "汤",
}

LIST_SIGNALS = {"吃什么", "推荐", "来几个", "有哪些", "有啥"}
DETAIL_SIGNALS = {"怎么做", "做法", "食材", "材料", "配料", "步骤", "技巧", "热量", "适合", "减脂", "不放", "替换", "为什么"}
ORDINAL_PREFIXES = ("第一个", "第二个", "第三个", "第四个", "第五个", "1号", "2号", "3号", "4号", "5号")
PRONOUN_PREFIXES = ("这个", "这个菜", "这道", "这道菜", "它", "那个", "那道", "那道菜")
CORRECTION_PREFIXES = ("不是这个，是", "不是，是", "不对，是", "不，是")
UNSUPPORTED_ORDINAL_INTENTS = {"作者", "发明", "哪年", "历史", "出处", "谁写"}


def _base_result(
    *,
    action: str,
    answer_mode_hint: str,
    should_retrieve: bool,
    depends_on_state: bool = False,
    needs_reference_resolution: bool = False,
    domain_confidence: float = 1.0,
    reference_trigger: str = "none",
    reason: str,
) -> Dict[str, Any]:
    return {
        "action": action,
        "answer_mode_hint": answer_mode_hint,
        "depends_on_state": depends_on_state,
        "needs_reference_resolution": needs_reference_resolution,
        "domain_confidence": domain_confidence,
        "reference_trigger": reference_trigger,
        "should_retrieve": should_retrieve,
        "reason": reason,
        "turn_type": _legacy_turn_type(action),
        "response_mode": "retrieve_answer" if should_retrieve else "polite_direct_reply",
        "should_update_topic_state": action == "retrieve_list",
        "should_update_entity_state": action == "retrieve_detail",
        "should_run_reference_resolution": needs_reference_resolution,
    }


def _legacy_turn_type(action: str) -> str:
    if action == "retrieve_list":
        return "recommendation_query"
    if action in {"retrieve_detail", "compare", "substitution"}:
        return "followup_query"
    if action in {"smalltalk", "domain_reject"}:
        return "direct_answer"
    return "domain_query"


def _normalize(question: str) -> str:
    return question.strip().rstrip("?!？！。")


def _has_recent_recommendations(snapshot: dict) -> bool:
    return bool(snapshot.get("reference_state", {}).get("recent_recommendations") or [])


def _has_current_dish(snapshot: dict) -> bool:
    current = snapshot.get("reference_state", {}).get("current_dish", {})
    return bool(current.get("active") and current.get("value"))


def _starts_with_any(text: str, prefixes: tuple[str, ...]) -> bool:
    return any(text.startswith(prefix) for prefix in prefixes)


def _has_recipe_signal(text: str) -> bool:
    return any(signal in text for signal in RECIPE_ALLOW_SIGNALS)


def _has_out_of_domain_signal(text: str) -> bool:
    return any(signal in text for signal in OUT_OF_DOMAIN_KEYWORDS)


def _has_detail_signal(text: str) -> bool:
    return any(signal in text for signal in DETAIL_SIGNALS)


def _has_list_signal(text: str) -> bool:
    return any(signal in text for signal in LIST_SIGNALS)


def _has_unsupported_ordinal_intent(text: str) -> bool:
    return any(signal in text for signal in UNSUPPORTED_ORDINAL_INTENTS)


def understand_turn(question: str, snapshot: dict) -> Dict[str, Any]:
    """Classify a turn after a lightweight session snapshot exists."""
    text = _normalize(question)

    if text in SMALLTALK_EXACT:
        return _base_result(
            action="smalltalk",
            answer_mode_hint="safe_direct",
            should_retrieve=False,
            reason="smalltalk_exact",
        )

    if _starts_with_any(text, CORRECTION_PREFIXES):
        return _base_result(
            action="retrieve_detail",
            answer_mode_hint="recipe_detail",
            should_retrieve=True,
            depends_on_state=True,
            needs_reference_resolution=True,
            reference_trigger="correction",
            reason="correction_reference",
        )

    if _starts_with_any(text, ORDINAL_PREFIXES):
        if _has_unsupported_ordinal_intent(text):
            return _base_result(
                action="domain_reject",
                answer_mode_hint="safe_direct",
                should_retrieve=False,
                depends_on_state=_has_recent_recommendations(snapshot),
                reason="ordinal_non_recipe_intent",
            )
        return _base_result(
            action="retrieve_detail",
            answer_mode_hint="recipe_detail",
            should_retrieve=True,
            depends_on_state=True,
            needs_reference_resolution=True,
            reference_trigger="ordinal_reference",
            reason="ordinal_recipe_followup",
        )

    if _starts_with_any(text, PRONOUN_PREFIXES):
        action = "substitution" if any(token in text for token in ("不放", "替换", "换成")) else "retrieve_detail"
        return _base_result(
            action=action,
            answer_mode_hint="substitution" if action == "substitution" else "recipe_detail",
            should_retrieve=True,
            depends_on_state=True,
            needs_reference_resolution=True,
            reference_trigger="pronoun",
            reason="pronoun_followup",
        )

    if _has_out_of_domain_signal(text) and not _has_recipe_signal(text):
        return _base_result(
            action="domain_reject",
            answer_mode_hint="safe_direct",
            should_retrieve=False,
            domain_confidence=0.95,
            reason="harmless_out_of_domain",
        )

    if _has_list_signal(text):
        return _base_result(
            action="retrieve_list",
            answer_mode_hint="recommendation",
            should_retrieve=True,
            reason="recipe_list_query",
        )

    if _has_detail_signal(text) or _has_recipe_signal(text):
        return _base_result(
            action="retrieve_detail",
            answer_mode_hint="recipe_detail",
            should_retrieve=True,
            needs_reference_resolution=False,
            reason="recipe_detail_query",
        )

    if _has_current_dish(snapshot):
        return _base_result(
            action="history_answer",
            answer_mode_hint="history_based",
            should_retrieve=False,
            depends_on_state=True,
            reason="state_contextual_short_query",
        )

    return _base_result(
        action="domain_reject",
        answer_mode_hint="safe_direct",
        should_retrieve=False,
        domain_confidence=0.6,
        reason="no_recipe_signal",
    )
```

- [ ] **Step 4: Run the turn understanding tests and verify they pass**

Run:

```bash
cd code/C8
pytest tests/test_turn_understanding.py -q
```

Expected:

- PASS.

- [ ] **Step 5: Commit**

```bash
git add code/C8/rag_modules/turn_understanding.py code/C8/tests/test_turn_understanding.py
git commit -m "feat: add context-aware turn understanding"
```

---

## Task 3: Make Execution Planning Consume The New Action Contract

**Files:**
- Modify: `code/C8/rag_modules/execution_planner.py`
- Modify: `code/C8/tests/test_conversation_state.py`

- [ ] **Step 1: Add execution planner tests for new actions**

Append these tests to `code/C8/tests/test_conversation_state.py` near existing `build_execution_plan` tests:

```python
def test_execution_plan_directs_domain_reject_without_retrieval():
    plan = build_execution_plan(
        {
            "action": "domain_reject",
            "response_mode": "polite_direct_reply",
            "should_retrieve": False,
        },
        resolution=None,
    )

    assert plan == {"action": "direct_domain_reject", "message": None}


def test_execution_plan_directs_smalltalk_without_retrieval():
    plan = build_execution_plan(
        {
            "action": "smalltalk",
            "response_mode": "polite_direct_reply",
            "should_retrieve": False,
        },
        resolution=None,
    )

    assert plan == {"action": "direct_smalltalk_reply", "message": None}


def test_execution_plan_uses_retrieve_list_action():
    plan = build_execution_plan(
        {
            "action": "retrieve_list",
            "response_mode": "retrieve_answer",
            "should_retrieve": True,
        },
        resolution=None,
    )

    assert plan == {"action": "retrieve_list", "message": None}


def test_execution_plan_uses_retrieve_detail_action():
    plan = build_execution_plan(
        {
            "action": "retrieve_detail",
            "response_mode": "retrieve_answer",
            "should_retrieve": True,
        },
        resolution=None,
    )

    assert plan == {"action": "retrieve_detail", "message": None}
```

- [ ] **Step 2: Run the execution planner tests and verify they fail**

Run:

```bash
cd code/C8
pytest tests/test_conversation_state.py::test_execution_plan_directs_domain_reject_without_retrieval tests/test_conversation_state.py::test_execution_plan_directs_smalltalk_without_retrieval tests/test_conversation_state.py::test_execution_plan_uses_retrieve_list_action tests/test_conversation_state.py::test_execution_plan_uses_retrieve_detail_action -q
```

Expected:

- FAIL because `build_execution_plan()` does not yet read `turn_info["action"]` for direct domain rejection.

- [ ] **Step 3: Replace `build_execution_plan()` with action-first logic**

Replace `code/C8/rag_modules/execution_planner.py` with:

```python
"""Execution planning for context-first turn decisions."""

from __future__ import annotations


def build_execution_plan(turn_info: dict, resolution: dict | None) -> dict:
    """Convert turn understanding and reference resolution into runtime action."""
    if resolution and resolution.get("next_action") == "ask_clarification":
        return {"action": "ask_clarification", "message": resolution["clarification_question"]}
    if resolution and resolution.get("next_action") == "apply_correction":
        return {"action": "apply_correction", "message": None}
    if resolution and resolution.get("next_action") == "apply_reference_resolution":
        return {"action": "apply_reference_resolution", "message": None}

    action = turn_info.get("action")
    if action == "domain_reject":
        return {"action": "direct_domain_reject", "message": None}
    if action == "smalltalk":
        return {"action": "direct_smalltalk_reply", "message": None}
    if action == "retrieve_list":
        return {"action": "retrieve_list", "message": None}
    if action in {"retrieve_detail", "compare", "substitution", "history_answer"}:
        return {"action": "retrieve_detail", "message": None}

    if turn_info.get("response_mode") == "polite_direct_reply":
        return {"action": "direct_smalltalk_reply", "message": None}
    if turn_info.get("turn_type") == "recommendation_query":
        return {"action": "retrieve_list", "message": None}
    return {"action": "retrieve_detail", "message": None}
```

- [ ] **Step 4: Run execution planner tests and verify they pass**

Run:

```bash
cd code/C8
pytest tests/test_conversation_state.py::test_execution_plan_directs_domain_reject_without_retrieval tests/test_conversation_state.py::test_execution_plan_directs_smalltalk_without_retrieval tests/test_conversation_state.py::test_execution_plan_uses_retrieve_list_action tests/test_conversation_state.py::test_execution_plan_uses_retrieve_detail_action -q
```

Expected:

- PASS.

- [ ] **Step 5: Commit**

```bash
git add code/C8/rag_modules/execution_planner.py code/C8/tests/test_conversation_state.py
git commit -m "refactor: make execution planning action-driven"
```

---

## Task 4: Rewire `ask_question()` To Context-First Order

**Files:**
- Modify: `code/C8/main.py`
- Modify: `code/C8/tests/test_conversation_state.py`

- [ ] **Step 1: Add integration tests for context-first behavior**

Append these tests to `code/C8/tests/test_conversation_state.py`:

```python
def test_context_first_pipeline_does_not_block_ordinal_followup_before_snapshot(monkeypatch):
    from main import RecipeRAGSystem

    system = RecipeRAGSystem.__new__(RecipeRAGSystem)
    calls = []

    class FakeConversationManager:
        def get_session(self, session_id):
            class Session:
                current_entity_meta = {}
                recent_recommendations = [{"dish_name": "鸡胸肉沙拉"}]
                recent_topics = []
                last_confirmed_target = None
                messages = []
                topic_mode = None
                current_intent = None
                pending_clarification = None
            return Session()

    class FakeGeneration:
        conversation_manager = FakeConversationManager()
        llm = None

        def generate_smalltalk_answer(self, question):
            return "smalltalk"

    system.retrieval_module = object()
    system.generation_module = FakeGeneration()
    system._latest_parent_docs = []
    system.last_execution_result = None

    monkeypatch.setattr(system, "_build_query_plan", lambda question, session_id: {"route_type": "detail", "dish_name": "鸡胸肉沙拉"})
    monkeypatch.setattr(system, "_apply_resolved_target_to_query_plan", lambda query_plan, resolution: query_plan)
    monkeypatch.setattr(system, "_search_relevant_chunks", lambda question, rewritten_query, filters, dish_name, top_k=5, query_dish=None: [{"content": "鸡胸肉沙拉做法", "metadata": {"dish_name": "鸡胸肉沙拉"}}])
    monkeypatch.setattr(system, "_get_parent_documents", lambda chunks, target_dish_name=None: [])
    monkeypatch.setattr(system, "_generate_detail_response", lambda question, stream, session_id, route_type, filters, entities, dish_name, relevant_chunks: "鸡胸肉沙拉适合减脂。")
    monkeypatch.setattr(system, "_write_conversation_turn", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr("main.resolve_reference_from_snapshot", lambda snapshot, llm: None)
    monkeypatch.setattr("main.guard_resolution_output", lambda resolution, constraints: resolution)

    answer = system.ask_question("第一个适合减脂吗", stream=False, session_id="ctx-first")

    assert answer == "鸡胸肉沙拉适合减脂。"
    assert calls
    assert calls[-1]["turn_info"]["action"] == "retrieve_detail"
    assert calls[-1]["turn_info"]["reference_trigger"] == "ordinal_reference"


def test_context_first_pipeline_routes_domain_reject_without_retrieval(monkeypatch):
    from main import RecipeRAGSystem

    system = RecipeRAGSystem.__new__(RecipeRAGSystem)
    calls = []
    search_calls = []

    class FakeConversationManager:
        def get_session(self, session_id):
            class Session:
                current_entity_meta = {}
                recent_recommendations = []
                recent_topics = []
                last_confirmed_target = None
                messages = []
                topic_mode = None
                current_intent = None
                pending_clarification = None
            return Session()

    class FakeGeneration:
        conversation_manager = FakeConversationManager()
        llm = None

        def generate_smalltalk_answer(self, question):
            return "smalltalk"

    system.retrieval_module = object()
    system.generation_module = FakeGeneration()
    system._latest_parent_docs = []
    system.last_execution_result = None
    monkeypatch.setattr(system, "_search_relevant_chunks", lambda *args, **kwargs: search_calls.append(args) or [])
    monkeypatch.setattr(system, "_write_conversation_turn", lambda **kwargs: calls.append(kwargs))

    answer = system.ask_question("Python怎么学", stream=False, session_id="domain-reject")

    assert "食谱" in answer or "做菜" in answer
    assert search_calls == []
    assert calls[-1]["turn_info"]["action"] == "domain_reject"


def test_context_first_pipeline_routes_smalltalk_without_recipe_state_update(monkeypatch):
    from main import RecipeRAGSystem

    system = RecipeRAGSystem.__new__(RecipeRAGSystem)
    calls = []

    class FakeConversationManager:
        def get_session(self, session_id):
            class Session:
                current_entity_meta = {"value": "蛋炒饭", "active": True, "source": "confirmed", "confidence": 1.0, "updated_at": 0.0}
                recent_recommendations = []
                recent_topics = []
                last_confirmed_target = "蛋炒饭"
                messages = []
                topic_mode = None
                current_intent = None
                pending_clarification = None
            return Session()

    class FakeGeneration:
        conversation_manager = FakeConversationManager()
        llm = None

        def generate_smalltalk_answer(self, question):
            return "不客气。"

    system.retrieval_module = object()
    system.generation_module = FakeGeneration()
    system._latest_parent_docs = []
    system.last_execution_result = None
    monkeypatch.setattr(system, "_write_conversation_turn", lambda **kwargs: calls.append(kwargs))

    answer = system.ask_question("谢谢", stream=False, session_id="smalltalk")

    assert answer == "不客气。"
    assert calls[-1]["turn_info"]["action"] == "smalltalk"
    assert calls[-1]["turn_info"]["should_update_entity_state"] is False
```

- [ ] **Step 2: Run the context-first integration tests and verify they fail**

Run:

```bash
cd code/C8
pytest tests/test_conversation_state.py::test_context_first_pipeline_does_not_block_ordinal_followup_before_snapshot tests/test_conversation_state.py::test_context_first_pipeline_routes_domain_reject_without_retrieval tests/test_conversation_state.py::test_context_first_pipeline_routes_smalltalk_without_recipe_state_update -q
```

Expected:

- FAIL because `ask_question()` still calls the old pre-context front-door and qualification order.

- [ ] **Step 3: Change imports in `main.py`**

In `code/C8/main.py`, replace:

```python
from rag_modules.turn_qualification import qualify_turn
from rag_modules.front_door_guardrail import check_front_door
```

with:

```python
from rag_modules.front_door_guardrail import basic_safety_gate
from rag_modules.turn_understanding import understand_turn
```

- [ ] **Step 4: Replace the early front-door and qualification block**

Inside `RecipeRAGSystem.ask_question()`, replace the block from:

```python
front_door = check_front_door(question)
```

through the end of:

```python
if turn_info["response_mode"] == "polite_direct_reply":
    ...
    return answer
```

with:

```python
safety = basic_safety_gate(question)
logger.info(
    "[BasicSafetyGate] decision=%s reason=%s",
    safety["decision"],
    safety["reason"],
)

if safety["decision"] == "block":
    answer = safety["message"] or "请输入一个具体的食谱或做菜问题。"
    self.last_execution_result = {"success": True, "answer": answer}
    self._write_conversation_turn(
        session_id=session_id,
        question=question,
        answer=answer,
        turn_info={
            "action": "invalid_input",
            "answer_mode_hint": "safe_direct",
            "turn_type": "basic_safety_blocked",
            "response_mode": "polite_direct_reply",
            "should_retrieve": False,
            "should_update_topic_state": False,
            "should_update_entity_state": False,
            "should_run_reference_resolution": False,
            "reference_trigger": "none",
        },
        query_plan=None,
        resolution=None,
        execution_result=self.last_execution_result,
    )
    return answer

conversation_manager = getattr(self.generation_module, "conversation_manager", None)
snapshot = None
resolution = None
if conversation_manager:
    snapshot = build_conversation_snapshot(
        conversation_manager.get_session(session_id),
        current_query=question,
    )
else:
    snapshot = {
        "reference_state": {
            "current_dish": {"value": None, "active": False},
            "recent_recommendations": [],
        },
        "resolution_constraints": {"allowed_reference_targets": []},
        "state_health": {"has_pending_clarification": False},
    }

turn_info = understand_turn(question, snapshot)

if turn_info["action"] == "smalltalk":
    answer = self.generation_module.generate_smalltalk_answer(question)
    execution_result = {"success": True, "answer": answer}
    self.last_execution_result = execution_result
    self._write_conversation_turn(
        session_id=session_id,
        question=question,
        answer=answer,
        turn_info=turn_info,
        query_plan=None,
        resolution=None,
        execution_result=execution_result,
    )
    return answer

if turn_info["action"] == "domain_reject":
    answer = "我主要处理食谱、做菜、食材和菜品推荐相关问题。"
    execution_result = {"success": True, "answer": answer}
    self.last_execution_result = execution_result
    self._write_conversation_turn(
        session_id=session_id,
        question=question,
        answer=answer,
        turn_info=turn_info,
        query_plan=None,
        resolution=None,
        execution_result=execution_result,
    )
    return answer

if turn_info["should_run_reference_resolution"]:
    resolution = resolve_reference_from_snapshot(snapshot, getattr(self.generation_module, "llm", None))
    resolution = guard_resolution_output(
        resolution,
        snapshot["resolution_constraints"],
    )

if resolution and resolution["next_action"] == "ask_clarification":
    answer = resolution["clarification_question"]
    execution_result = self._build_execution_result(
        success=True,
        answer=answer,
        rewritten_question=question,
        original_question=question,
        query_plan=None,
        resolution=resolution,
        parent_docs=[],
    )
    self.last_execution_result = execution_result
    self._write_conversation_turn(
        session_id=session_id,
        question=question,
        answer=answer,
        turn_info=turn_info,
        query_plan=None,
        resolution=resolution,
        execution_result=execution_result,
    )
    return answer
```

Then remove the entire old `# --- Structured Snapshot + Reference Resolution ---` block from `conversation_manager = getattr(...)` through `return answer` (the snapshot build, reference resolution, guard, and clarification early-return). This block starts right after the `qualify_turn` block and ends before `# --- Execution Planning ---`. In the current `main.py` this is approximately lines 693–730. Removing it avoids building the snapshot and resolving references twice.

- [ ] **Step 5: Run the context-first integration tests and verify they pass**

Run:

```bash
cd code/C8
pytest tests/test_conversation_state.py::test_context_first_pipeline_does_not_block_ordinal_followup_before_snapshot tests/test_conversation_state.py::test_context_first_pipeline_routes_domain_reject_without_retrieval tests/test_conversation_state.py::test_context_first_pipeline_routes_smalltalk_without_recipe_state_update -q
```

Expected:

- PASS.

- [ ] **Step 6: Commit**

```bash
git add code/C8/main.py code/C8/tests/test_conversation_state.py
git commit -m "refactor: rewire ask question to context-first pipeline"
```

---

## Task 5: Delete The Old Turn Qualification Path

**Files:**
- Delete: `code/C8/rag_modules/turn_qualification.py`
- Delete: `code/C8/tests/test_turn_qualification.py`
- Create: `code/C8/tests/test_context_first_cutover.py`

- [ ] **Step 1: Add cutover tests**

Create `code/C8/tests/test_context_first_cutover.py`:

```python
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_main_uses_context_first_contracts_not_old_front_door_or_qualification():
    source = (ROOT / "main.py").read_text(encoding="utf-8")

    assert "from rag_modules.front_door_guardrail import check_front_door" not in source
    assert "from rag_modules.turn_qualification import qualify_turn" not in source
    assert "check_front_door(" not in source
    assert "qualify_turn(" not in source
    assert "basic_safety_gate(question)" in source
    assert "understand_turn(question, snapshot)" in source


def test_context_first_order_is_visible_in_main_source():
    source = (ROOT / "main.py").read_text(encoding="utf-8")

    safety_index = source.index("basic_safety_gate(question)")
    snapshot_index = source.index("build_conversation_snapshot(")
    understanding_index = source.index("understand_turn(question, snapshot)")

    assert safety_index < snapshot_index < understanding_index


def test_old_turn_qualification_module_is_removed():
    assert not (ROOT / "rag_modules" / "turn_qualification.py").exists()


def test_front_door_exports_only_basic_safety_gate_contract():
    source = (ROOT / "rag_modules" / "front_door_guardrail.py").read_text(encoding="utf-8")

    assert "def basic_safety_gate(" in source
    assert "def check_front_door(" not in source
    assert "direct_reply" not in source
    assert "smalltalk" not in source.lower()
    assert "out_of_domain" not in source
```

- [ ] **Step 2: Run cutover tests and verify they fail before deletion**

Run:

```bash
cd code/C8
pytest tests/test_context_first_cutover.py -q
```

Expected:

- FAIL while `turn_qualification.py` or old import/call strings still exist.

- [ ] **Step 3: Delete old qualification module and test**

Delete these files:

```text
code/C8/rag_modules/turn_qualification.py
code/C8/tests/test_turn_qualification.py
```

- [ ] **Step 4: Search for forbidden production references**

Run:

```bash
rg -n "check_front_door|qualify_turn|turn_qualification" code/C8 --glob "!docs/**"
```

Expected:

- No output.

- [ ] **Step 5: Run cutover tests and verify they pass**

Run:

```bash
cd code/C8
pytest tests/test_context_first_cutover.py -q
```

Expected:

- PASS.

- [ ] **Step 6: Commit**

```bash
git add code/C8/main.py code/C8/rag_modules/front_door_guardrail.py code/C8/rag_modules/turn_understanding.py code/C8/tests/test_context_first_cutover.py
git rm code/C8/rag_modules/turn_qualification.py code/C8/tests/test_turn_qualification.py
git commit -m "refactor: remove old pre-context turn qualification path"
```

---

## Task 6: Run Stage 02 Acceptance Suite

**Files:**
- Verify only.

- [ ] **Step 1: Run focused Stage 02 tests**

Run:

```bash
cd code/C8
pytest tests/test_front_door_guardrail.py tests/test_turn_understanding.py tests/test_context_first_cutover.py -q
```

Expected:

- PASS.

- [ ] **Step 2: Run conversation-state tests that cover integration**

Run:

```bash
cd code/C8
pytest tests/test_conversation_state.py -q
```

Expected:

- PASS.

- [ ] **Step 3: Run forbidden-reference scan**

Run:

```bash
rg -n "check_front_door|qualify_turn|turn_qualification" code/C8 --glob "!docs/**"
```

Expected:

- No output.

- [ ] **Step 4: Run UTF-8 anchor check on Stage 02 docs and tests**

Run:

```bash
python -c "from pathlib import Path; files=[Path('docs/architecture/evolution/02-context-first-turn-pipeline.md'),Path('docs/superpowers/plans/2026-07-06-context-first-turn-pipeline.md'),Path('tests/test_front_door_guardrail.py'),Path('tests/test_turn_understanding.py'),Path('tests/test_context_first_cutover.py')]; [p.read_text(encoding='utf-8') for p in files]; text=files[1].read_text(encoding='utf-8'); assert '第一个适合减脂吗' in text and 'Python怎么学' in text and '蛋炒饭怎么做' in text"
```

Expected:

- No output and exit code 0.

- [ ] **Step 5: Commit any final test migration edits**

If Step 2 required updates to tests, run:

```bash
git add code/C8/tests/test_conversation_state.py
git commit -m "test: align conversation tests with context-first pipeline"
```

If Step 2 required no edits, do not create an empty commit.

---

## Self-Review

Spec coverage:

- Basic safety gate is covered by Task 1.
- Context-first `ask_question()` order is covered by Task 4.
- `TurnUnderstanding` action contract is covered by Task 2.
- Context-aware domain rejection and smalltalk are covered by Tasks 2 and 4.
- Reference-resolution decision ownership is covered by Tasks 2 and 4.
- Execution Plan action boundary is covered by Task 3.
- Old path cutover and deletion are covered by Task 5.
- Acceptance scans and focused tests are covered by Task 6.

Type consistency:

- The production API names are `basic_safety_gate()` and `understand_turn()`.
- The new turn contract uses `action`, `answer_mode_hint`, `depends_on_state`, `needs_reference_resolution`, `domain_confidence`, `reference_trigger`, `should_retrieve`, and `reason`.
- Legacy compatibility fields remain in `turn_info`: `turn_type`, `response_mode`, `should_update_topic_state`, `should_update_entity_state`, and `should_run_reference_resolution`.

Cutover consistency:

- `check_front_door` is not kept as a compatibility alias.
- `qualify_turn` is not kept as a compatibility alias.
- Tests prove the production path uses snapshot before turn understanding.
- Tests prove old files and old production references are gone.
