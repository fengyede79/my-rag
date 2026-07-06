# Front Door Guardrail Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a thin deterministic front door that coexists with `turn_qualification` and `reference_resolution` while preserving the current main conversation chain.

**Architecture:** `front_door_guardrail.check_front_door()` runs at the start of `RecipeRAGSystem.ask_question()`. It returns only `continue`, `block`, or `direct_reply`; `continue` passes the original question unchanged into the existing chain, while `block` and `direct_reply` return message-only answers before query planning and retrieval. Exact smalltalk moves to the front door; `turn_qualification` remains responsible for recommendation, follow-up, correction, and domain turns inside the main chain.

**Tech Stack:** Python, pytest, existing `RecipeRAGSystem`, existing `rag_modules` package.

---

## Reference

- Spec: `code/C8/docs/superpowers/specs/2026-07-05-front-door-guardrail-design.md`
- Existing main flow:

```text
ask_question
  -> qualify_turn
  -> build_conversation_snapshot
  -> resolve_reference_from_snapshot
  -> build_execution_plan
  -> _build_query_plan
  -> retrieval
  -> generation
  -> writeback_turn_state
```

## File Structure

- Create: `code/C8/rag_modules/front_door_guardrail.py`
  - Owns the conservative front-door checks.
  - Exposes `check_front_door(query: str) -> dict`.
  - Returns only `decision`, `reason`, and `message`.

- Create: `code/C8/tests/test_front_door_guardrail.py`
  - Tests the front-door module directly.

- Modify: `code/C8/main.py`
  - Imports `check_front_door`.
  - Runs it before `qualify_turn`.
  - Stops early for `block` and `direct_reply`.
  - Leaves `continue` path untouched.

- Modify: `code/C8/rag_modules/turn_qualification.py`
  - Removes smalltalk as primary responsibility.
  - Keeps recommendation, follow-up, correction, ordinal, implicit-detail, and domain classification.

- Modify: `code/C8/rag_modules/state_writeback_review.py`
  - Treats front-door direct turns as `message_only`.

- Modify: `code/C8/tests/test_conversation_state.py`
  - Adds regressions for early stop and query-plan ownership.
  - Adjusts smalltalk expectations to front-door behavior.

- Modify: `code/C8/tests/test_turn_qualification.py`
  - Removes or replaces smalltalk ownership assertions.
  - Keeps tests for recommendation/follow-up/correction/ordinal/short-detail classification.

- Do not create or restore:
  - `code/C8/rag_modules/local_semantic_analyzer.py`
  - `code/C8/rag_modules/strict_guardrail.py`

---

## Task 1: Specify The Front-Door Module Behavior

**Files:**
- Create: `code/C8/tests/test_front_door_guardrail.py`
- Test target: `code/C8/rag_modules/front_door_guardrail.py`

- [ ] **Step 1: Write failing module tests**

Create `code/C8/tests/test_front_door_guardrail.py`:

```python
from rag_modules.front_door_guardrail import check_front_door


def test_front_door_continues_recipe_like_inputs():
    for query in [
        "土豆丝怎么样",
        "蛋炒饭好不好",
        "青椒肉丝可以吗",
        "牛排怎么煎",
        "宫保鸡丁怎么做",
        "今天吃什么",
        "我想吃点清淡的",
        "这道菜怎么做？",
        "它需要什么食材？",
    ]:
        assert check_front_door(query) == {
            "decision": "continue",
            "reason": "default_continue",
            "message": None,
        }


def test_front_door_blocks_structurally_useless_inputs():
    for query in ["", " ", "？", "这道", "那道", "这个", "那个", "它"]:
        result = check_front_door(query)

        assert result["decision"] == "block"
        assert result["reason"] == "empty_or_isolated_reference"
        assert result["message"]


def test_front_door_does_not_block_meaningful_short_food_inputs():
    for query in ["牛排", "煎蛋", "凉面"]:
        assert check_front_door(query) == {
            "decision": "continue",
            "reason": "default_continue",
            "message": None,
        }


def test_front_door_direct_replies_to_exact_smalltalk():
    for query in ["你好", "您好", "谢谢", "哈哈", "你是谁", "你能做什么"]:
        result = check_front_door(query)

        assert result["decision"] == "direct_reply"
        assert result["reason"] == "smalltalk_exact"
        assert result["message"]


def test_front_door_does_not_stop_recipe_suffix_questions_as_smalltalk():
    for query in ["土豆丝怎么样", "蛋炒饭可以吗", "青椒肉丝好不好"]:
        assert check_front_door(query)["decision"] == "continue"


def test_front_door_direct_replies_to_clear_out_of_domain():
    for query in ["Python怎么学", "今天天气怎么样", "股票怎么买", "手机壳发黄怎么办"]:
        result = check_front_door(query)

        assert result["decision"] == "direct_reply"
        assert result["reason"] == "clear_out_of_domain"
        assert result["message"]


def test_front_door_does_not_stop_kitchen_or_cooking_queries():
    for query in [
        "空气炸锅鸡翅怎么做",
        "蛋炒饭里放螺丝椒可以吗",
        "厨房刀怎么切肉更安全",
        "牛排怎么煎",
    ]:
        assert check_front_door(query)["decision"] == "continue"


def test_front_door_result_shape_has_no_semantic_fields():
    forbidden = {
        "dish_name",
        "intent_type",
        "route_type",
        "filters",
        "content_type",
        "semantic_result",
        "rewritten_query",
    }

    for query in ["你好", "这道", "蛋炒饭怎么做"]:
        result = check_front_door(query)
        assert set(result) == {"decision", "reason", "message"}
        assert forbidden.isdisjoint(result)
```

- [ ] **Step 2: Run test to verify RED**

Run:

```powershell
Push-Location code/C8
python -m pytest tests/test_front_door_guardrail.py -q
Pop-Location
```

Expected:

- FAIL or ERROR with `ModuleNotFoundError: No module named 'rag_modules.front_door_guardrail'`.

---

## Task 2: Implement The Deterministic Front Door

**Files:**
- Create: `code/C8/rag_modules/front_door_guardrail.py`
- Test: `code/C8/tests/test_front_door_guardrail.py`

- [ ] **Step 1: Create `front_door_guardrail.py`**

Create `code/C8/rag_modules/front_door_guardrail.py`:

```python
"""Conservative front-door guardrail.

This module decides only whether a query should enter the main RAG chain.
It must not produce dish names, intents, route types, filters, content types,
semantic results, or rewritten queries.
"""

from __future__ import annotations

import re
from typing import Dict


ISOLATED_REFERENCES = {
    "这道",
    "那道",
    "这个",
    "那个",
    "它",
    "这",
    "那",
}

SMALLTALK_EXACT = {
    "你好",
    "您好",
    "谢谢",
    "哈哈",
    "你是谁",
    "你能做什么",
}

OUT_OF_DOMAIN_KEYWORDS = {
    "天气",
    "新闻",
    "股票",
    "房价",
    "编程",
    "代码",
    "算法",
    "Python",
    "Java",
    "C++",
    "历史",
    "政治",
    "宗教",
    "医疗",
    "诊断",
    "治疗",
    "手机壳",
    "路由器",
    "断网",
}

DOMAIN_ALLOW_SIGNALS = {
    "菜",
    "食谱",
    "食材",
    "材料",
    "做法",
    "步骤",
    "烹饪",
    "早餐",
    "午饭",
    "晚饭",
    "夜宵",
    "甜品",
    "饮品",
    "汤",
    "饭",
    "面",
    "粥",
    "肉",
    "鱼",
    "虾",
    "蛋",
    "豆腐",
    "牛排",
    "鸡翅",
    "怎么做",
    "怎么煎",
    "怎么炒",
    "怎么炖",
    "怎么切",
    "空气炸锅",
    "厨房",
    "切肉",
    "螺丝椒",
    "吃什么",
}


def _normalize(query: str) -> str:
    return query.strip().rstrip("?!？！。")


def _is_empty_or_punctuation(normalized: str) -> bool:
    if not normalized:
        return True
    return re.fullmatch(r"[\s\W_]+", normalized) is not None


def _has_domain_allow_signal(normalized: str) -> bool:
    return any(signal in normalized for signal in DOMAIN_ALLOW_SIGNALS)


def _has_out_of_domain_signal(normalized: str) -> bool:
    return any(keyword in normalized for keyword in OUT_OF_DOMAIN_KEYWORDS)


def _block() -> Dict[str, str | None]:
    return {
        "decision": "block",
        "reason": "empty_or_isolated_reference",
        "message": "我还不知道你指的是哪道菜，可以说具体一点吗？",
    }


def _direct_reply(message: str, reason: str) -> Dict[str, str | None]:
    return {
        "decision": "direct_reply",
        "reason": reason,
        "message": message,
    }


def _continue() -> Dict[str, str | None]:
    return {
        "decision": "continue",
        "reason": "default_continue",
        "message": None,
    }


def check_front_door(query: str) -> Dict[str, str | None]:
    """Return the front-door decision for a user query."""
    normalized = _normalize(query)

    if _is_empty_or_punctuation(normalized) or normalized in ISOLATED_REFERENCES:
        return _block()

    if normalized in SMALLTALK_EXACT:
        return _direct_reply(
            "我是食谱助手，可以帮你查菜谱、做法、食材和推荐。",
            "smalltalk_exact",
        )

    if _has_out_of_domain_signal(normalized) and not _has_domain_allow_signal(normalized):
        return _direct_reply(
            "我主要帮你处理食谱和做菜相关问题，可以问我菜品做法、食材或推荐。",
            "clear_out_of_domain",
        )

    return _continue()
```

- [ ] **Step 2: Run front-door tests**

Run:

```powershell
Push-Location code/C8
python -m pytest tests/test_front_door_guardrail.py -q
Pop-Location
```

Expected:

- PASS.

---

## Task 3: Move Smalltalk Ownership Out Of `turn_qualification`

**Files:**
- Modify: `code/C8/rag_modules/turn_qualification.py`
- Modify: `code/C8/tests/test_turn_qualification.py`
- Test: `code/C8/tests/test_front_door_guardrail.py`

- [ ] **Step 1: Replace the smalltalk ownership test**

In `code/C8/tests/test_turn_qualification.py`, remove `test_smalltalk_turn_does_not_enter_retrieval` and add:

```python
def test_turn_qualification_treats_passed_inputs_as_main_chain_domain_queries():
    result = qualify_turn("你好")

    assert result["turn_type"] == "domain_query"
    assert result["should_retrieve"] is True
    assert result["should_run_reference_resolution"] is False
```

This test documents that `turn_qualification` is not the primary smalltalk gate. In production, `你好` should be stopped by `front_door_guardrail` before this function is called.

- [ ] **Step 2: Run test to verify RED**

Run:

```powershell
Push-Location code/C8
python -m pytest tests/test_turn_qualification.py::test_turn_qualification_treats_passed_inputs_as_main_chain_domain_queries -q
Pop-Location
```

Expected:

- FAIL because `qualify_turn("你好")` still returns `smalltalk`.

- [ ] **Step 3: Remove exact smalltalk branch from `qualify_turn`**

In `code/C8/rag_modules/turn_qualification.py`, remove:

```python
SMALLTALK_EXACT_PATTERNS = {"你好", "您好", "谢谢", "哈哈", "你是谁", "你怎么做", "你怎么做？"}
```

And remove this branch from `qualify_turn()`:

```python
    if normalized in SMALLTALK_EXACT_PATTERNS:
        return {
            "turn_type": "smalltalk",
            "should_retrieve": False,
            "should_update_topic_state": False,
            "should_update_entity_state": False,
            "should_run_reference_resolution": False,
            "response_mode": "polite_direct_reply",
            "reference_trigger": "none",
        }
```

Leave recommendation, correction, pronoun, ordinal, implicit-detail, and domain branches unchanged.

- [ ] **Step 4: Run turn qualification tests**

Run:

```powershell
Push-Location code/C8
python -m pytest tests/test_turn_qualification.py -q
Pop-Location
```

Expected:

- PASS.

---

## Task 4: Wire The Front Door Into `ask_question`

**Files:**
- Modify: `code/C8/main.py`
- Modify: `code/C8/tests/test_conversation_state.py`
- Test: `code/C8/tests/test_front_door_guardrail.py`
- Test: `code/C8/tests/test_conversation_state.py`

- [ ] **Step 1: Add integration tests for early stop and pass-through**

Append to `code/C8/tests/test_conversation_state.py`:

```python
def test_front_door_block_stops_before_query_planning_and_retrieval():
    system = _system()

    system.generation_module.query_router = lambda query: (_ for _ in ()).throw(
        AssertionError("blocked front-door input should not reach query planning")
    )
    system.retrieval_module.hybrid_search = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("blocked front-door input should not reach retrieval")
    )

    answer = system.ask_question("这道", stream=False, session_id="front-door-block-session")

    assert "哪道菜" in answer


def test_front_door_direct_reply_stops_before_query_planning_and_retrieval():
    system = _system()

    system.generation_module.query_router = lambda query: (_ for _ in ()).throw(
        AssertionError("direct front-door reply should not reach query planning")
    )
    system.retrieval_module.hybrid_search = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("direct front-door reply should not reach retrieval")
    )

    answer = system.ask_question("你好", stream=False, session_id="front-door-smalltalk-session")

    assert "食谱助手" in answer or "推荐菜" in answer


def test_front_door_continue_preserves_original_question_for_query_planning():
    system = _system()
    calls = []

    def fake_query_router(query):
        calls.append(query)
        return {
            "type": "general",
            "filters": {},
            "dish_name": None,
            "confidence": 0.7,
        }

    system.generation_module.query_router = fake_query_router

    plan = system._build_query_plan("土豆丝怎么样", "front-door-plan-session")

    assert calls == ["土豆丝怎么样"]
    assert plan["dish_name"] == "土豆丝"
    assert plan["entities"]["dish_name"] == "土豆丝"
```

- [ ] **Step 2: Run integration tests to verify RED**

Run:

```powershell
Push-Location code/C8
python -m pytest `
  tests/test_conversation_state.py::test_front_door_block_stops_before_query_planning_and_retrieval `
  tests/test_conversation_state.py::test_front_door_direct_reply_stops_before_query_planning_and_retrieval `
  tests/test_conversation_state.py::test_front_door_continue_preserves_original_question_for_query_planning `
  -q
Pop-Location
```

Expected:

- At least the first two tests FAIL because `ask_question()` does not call `check_front_door()` yet.

- [ ] **Step 3: Import `check_front_door`**

In `code/C8/main.py`, add near the other `rag_modules` imports:

```python
from rag_modules.front_door_guardrail import check_front_door
```

- [ ] **Step 4: Add front-door handling at the start of `ask_question`**

In `RecipeRAGSystem.ask_question`, immediately after:

```python
        print(f"\n用户问题: {question}")
        original_question = question
        self._latest_parent_docs = []
```

insert:

```python
        front_door = check_front_door(question)
        logger.info(
            "[FrontDoor] decision=%s reason=%s",
            front_door["decision"],
            front_door["reason"],
        )

        if front_door["decision"] in {"block", "direct_reply"}:
            answer = front_door["message"] or "我主要帮你处理食谱和做菜相关问题。"
            self.last_execution_result = {"success": True, "answer": answer}
            self._write_conversation_turn(
                session_id=session_id,
                question=question,
                answer=answer,
                turn_info={
                    "turn_type": (
                        "front_door_blocked"
                        if front_door["decision"] == "block"
                        else "front_door_direct_reply"
                    ),
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
```

Leave the existing `qualify_turn` and downstream chain unchanged for `continue`.

- [ ] **Step 5: Run focused front-door integration tests**

Run:

```powershell
Push-Location code/C8
python -m pytest `
  tests/test_front_door_guardrail.py `
  tests/test_conversation_state.py::test_front_door_block_stops_before_query_planning_and_retrieval `
  tests/test_conversation_state.py::test_front_door_direct_reply_stops_before_query_planning_and_retrieval `
  tests/test_conversation_state.py::test_front_door_continue_preserves_original_question_for_query_planning `
  -q
Pop-Location
```

Expected:

- PASS.

---

## Task 5: Ensure Front-Door Turns Are Message-Only Writebacks

**Files:**
- Modify: `code/C8/rag_modules/state_writeback_review.py`
- Modify: `code/C8/tests/test_state_writeback_review.py`

- [ ] **Step 1: Add failing writeback tests**

Append to `code/C8/tests/test_state_writeback_review.py`:

```python
def test_front_door_blocked_turn_is_message_only():
    review = review_state_writeback(
        turn_info={"turn_type": "front_door_blocked"},
        resolution=None,
        execution_result={"success": True, "answer": "我还不知道你指的是哪道菜"},
        answer="我还不知道你指的是哪道菜",
        query_plan=None,
    )

    assert review["should_write_reliable_state"] is False
    assert review["writeback_mode"] == "message_only"


def test_front_door_direct_reply_turn_is_message_only():
    review = review_state_writeback(
        turn_info={"turn_type": "front_door_direct_reply"},
        resolution=None,
        execution_result={"success": True, "answer": "我是食谱助手"},
        answer="我是食谱助手",
        query_plan=None,
    )

    assert review["should_write_reliable_state"] is False
    assert review["writeback_mode"] == "message_only"
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```powershell
Push-Location code/C8
python -m pytest tests/test_state_writeback_review.py -q
Pop-Location
```

Expected:

- FAIL because front-door turn types are not treated as message-only yet.

- [ ] **Step 3: Add front-door turn types to message-only branch**

In `code/C8/rag_modules/state_writeback_review.py`, change:

```python
    if turn_type in {"smalltalk", "out_of_domain"}:
        return {"should_write_reliable_state": False, "writeback_mode": "message_only"}
```

to:

```python
    if turn_type in {
        "smalltalk",
        "out_of_domain",
        "front_door_blocked",
        "front_door_direct_reply",
    }:
        return {"should_write_reliable_state": False, "writeback_mode": "message_only"}
```

- [ ] **Step 4: Run state writeback tests**

Run:

```powershell
Push-Location code/C8
python -m pytest tests/test_state_writeback_review.py -q
Pop-Location
```

Expected:

- PASS.

---

## Task 6: Verify Rejected Layers Stay Out

**Files:**
- No production edits unless stale rejected files or references are found.

- [ ] **Step 1: Search for rejected layer references**

Run:

```powershell
rg -n "local_semantic_analyzer|get_local_analyzer|semantic_result|strict_guardrail" code/C8
```

Expected:

- No production code references.
- Mentions in docs/specs/plans are acceptable only when documenting rejected approaches.

- [ ] **Step 2: Confirm front-door does not leak semantic fields**

Run:

```powershell
Push-Location code/C8
python -m pytest tests/test_front_door_guardrail.py::test_front_door_result_shape_has_no_semantic_fields -q
Pop-Location
```

Expected:

- PASS.

- [ ] **Step 3: Confirm old guardrail helper is not wired into main path**

Run:

```powershell
rg -n "_maybe_handle_guardrail_query\\(" code/C8/main.py
```

Expected:

- One match for the method definition is acceptable.
- No call sites are expected in `ask_question()`.

---

## Task 7: Final Verification

**Files:**
- No edits.

- [ ] **Step 1: Run syntax verification**

Run:

```powershell
Push-Location code/C8
python -m py_compile `
  main.py `
  web_app.py `
  rag_modules/front_door_guardrail.py `
  rag_modules/turn_qualification.py `
  rag_modules/conversation_manager.py `
  rag_modules/state_writeback_review.py `
  rag_modules/reference_resolution.py `
  rag_modules/conversation_state_builder.py `
  rag_modules/execution_planner.py
Pop-Location
```

Expected:

- Exit code 0.

- [ ] **Step 2: Run focused unit suite**

Run:

```powershell
Push-Location code/C8
python -m pytest -q `
  tests/test_front_door_guardrail.py `
  tests/test_turn_qualification.py `
  tests/test_reference_resolution.py `
  tests/test_conversation_state.py `
  tests/test_state_writeback_review.py `
  tests/test_web_app.py
Pop-Location
```

Expected:

- PASS.

- [ ] **Step 3: Run existing guardrail and evaluation smoke tests**

Run:

```powershell
Push-Location code/C8
python -m pytest -q `
  tests/test_query_guardrails.py `
  tests/test_evaluation_framework.py `
  tests/test_process_diagnostics.py
Pop-Location
```

Expected:

- PASS.

- [ ] **Step 4: Run real integration tests if key is available**

Run:

```powershell
Push-Location code/C8
$content = Get-Content .env -Raw
if ($content -match 'DASHSCOPE_API_KEY=([^\r\n]+)') {
    $env:DASHSCOPE_API_KEY = $matches[1].Trim()
    python -m pytest -s tests/test_conversation_integration_real.py -m real_integration -q
} else {
    Write-Output "DASHSCOPE_API_KEY not found; real integration skipped"
}
Pop-Location
```

Expected:

- PASS when key is available.
- SKIP only if key is unavailable.

- [ ] **Step 5: Manual boundary review**

Open `code/C8/main.py` and verify:

- `check_front_door(question)` runs before `qualify_turn(question)`.
- `block` and `direct_reply` return before `_build_query_plan()` and retrieval.
- `continue` does not mutate `question`.
- `_build_query_plan()` remains the first owner of explicit dish extraction.
- `_search_relevant_chunks()` receives only query-plan outputs, not front-door hints.

---

## Self-Review Notes

Spec coverage:

- Three-component coexistence is covered by Tasks 3 and 4.
- Smalltalk ownership moves to front door in Tasks 1, 2, and 3.
- Front-door early stop is covered by Task 4.
- Message-only writeback is covered by Task 5.
- Rejected semantic layers staying out is covered by Task 6.
- Existing reference-resolution loop is protected by Task 7.

Placeholder scan:

- No task contains unfinished placeholder markers or shorthand copy instructions.
- Code snippets are complete enough to implement directly.
- Commands and expected outcomes are specified.

Type consistency:

- Front-door function name is consistently `check_front_door`.
- Front-door return keys are consistently `decision`, `reason`, and `message`.
- Front-door turn types are consistently `front_door_blocked` and `front_door_direct_reply`.
