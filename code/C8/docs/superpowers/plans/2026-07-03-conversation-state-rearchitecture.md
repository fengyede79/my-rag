# Conversation State Rearchitecture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the conversation pipeline so turn admission, structured conversation state, model-assisted reference resolution, conflict-prioritized execution, and guarded state writeback replace the current direct `current_entity` inheritance path.

**Architecture:** Introduce a front-door turn qualification layer before retrieval, replace single-value entity memory with a structured conversation snapshot, and route ambiguous follow-ups through a constrained reference-resolution step. Add explicit correction handling, query rewrite, execution-result-based writeback, and a final writeback review so the real runtime path matches the design instead of bypassing it.

**Tech Stack:** Python, Flask, pytest, LangChain/OpenAI-compatible chat model, existing RAG modules, repository-local markdown docs

---

## File Map

**Create:**
- `code/C8/rag_modules/turn_qualification.py`
- `code/C8/rag_modules/conversation_state_builder.py`
- `code/C8/rag_modules/reference_resolution.py`
- `code/C8/rag_modules/execution_planner.py`
- `code/C8/rag_modules/state_writeback_review.py`
- `code/C8/tests/test_turn_qualification.py`
- `code/C8/tests/test_reference_resolution.py`
- `code/C8/tests/test_state_writeback_review.py`
- `code/C8/tests/test_conversation_integration_real.py`

**Modify:**
- `code/C8/rag_modules/conversation_manager.py`
- `code/C8/rag_modules/generation_integration.py`
- `code/C8/main.py`
- `code/C8/web_app.py`
- `code/C8/tests/test_conversation_state.py`
- `code/C8/tests/test_web_app.py`

**Reference:**
- `code/C8/docs/superpowers/specs/2026-07-03-conversation-state-rearchitecture-design.md`

---

## Migration Notes

This plan intentionally keeps the current retrieval and generation stack in place while moving conversation decisions around it.

1. Keep `_resolve_question_reference()` during migration, but classify it as temporary compatibility logic. It must not remain a first-class long-term semantic layer.
2. Insert `qualify_turn()` immediately after the temporary ordinal-resolution step.
3. Run `_maybe_handle_guardrail_query()` only for non-smalltalk turns.
4. Build the structured snapshot before retrieval for every in-domain turn.
5. Run reference resolution only when `turn_info["should_run_reference_resolution"]` is `True`, but allow explicit correction turns to supply new explicit targets even if they are not already the only remembered candidate.
6. Always call `build_execution_plan()` before retrieval so the main path is execution-plan-driven rather than ad hoc branching.
7. Any correction turn that rewrites the user query must rebuild `query_plan` from `rewritten_question`; it must not reuse a stale plan built from the original text.
8. Retrieval and generation must return a structured `execution_result` object. State writeback consumes that object; it must not guess data from `turn_info`.
9. Insert `review_state_writeback()` after answer generation and before any session-state mutation.

The migration pipeline becomes:

1. `_resolve_question_reference()` (temporary compatibility only)
2. `qualify_turn()`
3. early return for `smalltalk`
4. `_maybe_handle_guardrail_query()`
5. early return for guardrail refusals
6. `build_conversation_snapshot()`
7. `resolve_reference_from_snapshot()` and `guard_resolution_output()`
8. `build_execution_plan()`
9. `rewrite_query_for_execution()`
10. rebuild `query_plan` when the execution query changes
11. existing retrieval and generation flow
12. `review_state_writeback()`
13. conditional writeback using `execution_result`

Boundary definition:

- `Turn Qualification` decides whether the turn may enter retrieval.
- `Reference Resolution` decides what the user is referring to, including correction turns and follow-up ambiguity.
- `Execution Planning` decides what action actually runs and what query text should be sent into retrieval.
- `State Writeback Review` decides whether a completed action is allowed to mutate reliable conversation state.

Runtime integration note:

- `RecipeRAGSystem` already has real helper paths for retrieval and answer generation: `_search_relevant_chunks()`, `_generate_list_response()`, and `_generate_detail_response()`.
- `web_app.py` already contains `_configure_file_logging()`. This plan keeps and reuses that helper; it does not assume the helper is missing.

---

### Task 1: Add Turn Qualification And Separate Smalltalk From Guardrail

**Files:**
- Create: `code/C8/rag_modules/turn_qualification.py`
- Modify: `code/C8/main.py`
- Modify: `code/C8/rag_modules/generation_integration.py`
- Test: `code/C8/tests/test_turn_qualification.py`

- [ ] **Step 1: Write the failing tests for turn admission**

```python
from rag_modules.turn_qualification import qualify_turn


def test_smalltalk_turn_does_not_enter_retrieval():
    result = qualify_turn("你好")
    assert result["turn_type"] == "smalltalk"
    assert result["should_retrieve"] is False
    assert result["response_mode"] == "polite_direct_reply"


def test_recommendation_turn_keeps_retrieval_enabled():
    result = qualify_turn("今天吃什么？")
    assert result["turn_type"] == "recommendation_query"
    assert result["should_retrieve"] is True
    assert result["should_update_entity_state"] is False


def test_followup_turn_requests_reference_resolution():
    result = qualify_turn("它怎么做？")
    assert result["turn_type"] == "followup_query"
    assert result["should_run_reference_resolution"] is True
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `C:\Users\lenovo\anaconda3\python.exe -m pytest code/C8/tests/test_turn_qualification.py -q`

Expected: FAIL with `ModuleNotFoundError` or missing `qualify_turn`.

- [ ] **Step 3: Write the minimal qualification module**

```python
SMALLTALK_EXACT_PATTERNS = {"你好", "您好", "谢谢", "哈哈", "你是谁"}
RECOMMENDATION_EXACT_PATTERNS = {
    "今天吃什么",
    "今天吃什么？",
    "晚饭吃什么",
    "晚饭吃什么？",
    "午饭吃什么",
    "午饭吃什么？",
}
FOLLOWUP_PREFIXES = ("它", "这个", "那个", "这道菜", "那道菜", "再说一个", "再讲一个")


def _normalize_for_match(question: str) -> str:
    return question.strip().rstrip("?!？！。")


def qualify_turn(question: str) -> dict:
    normalized = _normalize_for_match(question)
    if normalized in SMALLTALK_EXACT_PATTERNS:
        return {
            "turn_type": "smalltalk",
            "should_retrieve": False,
            "should_update_topic_state": False,
            "should_update_entity_state": False,
            "should_run_reference_resolution": False,
            "response_mode": "polite_direct_reply",
        }
    if normalized in RECOMMENDATION_EXACT_PATTERNS:
        return {
            "turn_type": "recommendation_query",
            "should_retrieve": True,
            "should_update_topic_state": True,
            "should_update_entity_state": False,
            "should_run_reference_resolution": False,
            "response_mode": "retrieve_answer",
        }
    if normalized.startswith(FOLLOWUP_PREFIXES):
        return {
            "turn_type": "followup_query",
            "should_retrieve": True,
            "should_update_topic_state": False,
            "should_update_entity_state": False,
            "should_run_reference_resolution": True,
            "response_mode": "retrieve_answer",
        }
    return {
        "turn_type": "domain_query",
        "should_retrieve": True,
        "should_update_topic_state": True,
        "should_update_entity_state": True,
        "should_run_reference_resolution": False,
        "response_mode": "retrieve_answer",
    }
```

- [ ] **Step 4: Wire qualification into the main request path**

```python
from rag_modules.turn_qualification import qualify_turn


def _write_conversation_turn(
    self,
    *,
    session_id: str,
    question: str,
    answer: str,
    turn_info: dict,
    query_plan: dict | None,
    resolution: dict | None,
    execution_result: dict,
):
    conversation_manager = getattr(self.generation_module, "conversation_manager", None)
    if not conversation_manager:
        return
    conversation_manager.writeback_turn_state(
        session_id=session_id,
        question=question,
        turn_info=turn_info,
        query_plan=query_plan,
        resolution=resolution,
        answer=answer,
        execution_result=execution_result,
    )


turn_info = qualify_turn(question)
if turn_info["response_mode"] == "polite_direct_reply":
    answer = self.generation_module.generate_smalltalk_answer(question)
    self._write_conversation_turn(
        session_id=session_id,
        question=question,
        answer=answer,
        turn_info=turn_info,
        query_plan=None,
        resolution=None,
        execution_result={"success": True, "answer": answer},
    )
    return answer

guardrail_answer = self._maybe_handle_guardrail_query(question)
if guardrail_answer is not None:
    self._write_conversation_turn(
        session_id=session_id,
        question=question,
        answer=guardrail_answer,
        turn_info={**turn_info, "turn_type": "out_of_domain"},
        query_plan=None,
        resolution=None,
        execution_result={"success": True, "answer": guardrail_answer},
    )
    return guardrail_answer
```

- [ ] **Step 4.1: Add the missing smalltalk generator to `GenerationIntegrationModule`**

```python
def generate_smalltalk_answer(self, query: str) -> str:
    normalized = query.strip().rstrip("?!？！。")
    if normalized in {"你好", "您好"}:
        return "你好，我可以帮你推荐菜、查做法，或者继续接着上一道菜聊。"
    if normalized == "谢谢":
        return "不客气，你可以继续问我吃什么、某道菜怎么做，或者食材怎么处理。"
    if normalized == "哈哈":
        return "那我们继续。你想让我推荐菜，还是直接查一道菜的做法？"
    if normalized == "你是谁":
        return "我是你的食谱助手，可以帮你推荐菜、查做法、查食材和烹饪技巧。"
    return "我在。你可以直接问我今天吃什么，或者某道菜怎么做。"
```

- [ ] **Step 5: Run tests and commit**

Run:
- `C:\Users\lenovo\anaconda3\python.exe -m pytest code/C8/tests/test_turn_qualification.py -q`
- `C:\Users\lenovo\anaconda3\python.exe -m pytest code/C8/tests/test_web_app.py -q`

Expected: PASS

Commit:

```bash
git add code/C8/rag_modules/turn_qualification.py code/C8/main.py code/C8/rag_modules/generation_integration.py code/C8/tests/test_turn_qualification.py code/C8/tests/test_web_app.py
git commit -m "feat: add turn qualification layer"
```

---

### Task 2: Replace Single-Entity State With Structured Snapshot And Correction-Aware Constraints

**Files:**
- Create: `code/C8/rag_modules/conversation_state_builder.py`
- Modify: `code/C8/rag_modules/conversation_manager.py`
- Test: `code/C8/tests/test_conversation_state.py`

- [ ] **Step 1: Write the failing structured-state tests**

```python
from rag_modules.conversation_manager import ConversationManager
from rag_modules.conversation_state_builder import build_conversation_snapshot


def test_recommendation_turn_sets_recommendation_mode_not_current_dish():
    manager = ConversationManager()
    manager.record_recommendations("s1", ["蛋炒饭", "麻辣香锅", "扬州炒饭"])
    snapshot = build_conversation_snapshot(manager.get_session("s1"), current_query="今天吃什么？")
    assert snapshot["topic_state"]["mode"] == "recommendation_list"
    assert snapshot["reference_state"]["current_dish"]["active"] is False
    assert snapshot["reference_state"]["recent_recommendations"][0]["dish_name"] == "蛋炒饭"


def test_current_dish_carries_source_and_confidence():
    manager = ConversationManager()
    manager.set_current_dish("s2", "蛋炒饭", source="explicit_query", confidence=1.0)
    snapshot = build_conversation_snapshot(manager.get_session("s2"), current_query="它怎么做？")
    assert snapshot["reference_state"]["current_dish"]["value"] == "蛋炒饭"
    assert snapshot["reference_state"]["current_dish"]["source"] == "explicit_query"
    assert snapshot["reference_state"]["current_dish"]["confidence"] == 1.0


def test_correction_query_adds_explicit_target_to_constraints():
    manager = ConversationManager()
    manager.set_current_dish("s3", "宫保鸡丁", source="inferred", confidence=0.55)
    snapshot = build_conversation_snapshot(manager.get_session("s3"), current_query="不是这个，是蛋炒饭")
    assert "蛋炒饭" in snapshot["resolution_constraints"]["explicit_query_targets"]
    assert snapshot["resolution_constraints"]["allow_external_explicit_target"] is True


def test_correction_query_marks_non_dish_text_as_unverified():
    manager = ConversationManager()
    manager.set_current_dish("s4", "宫保鸡丁", source="inferred", confidence=0.55)
    snapshot = build_conversation_snapshot(manager.get_session("s4"), current_query="不是这个，是那个简单点的")
    assert snapshot["resolution_constraints"]["explicit_query_targets"] == ["那个简单点的"]
    assert snapshot["resolution_constraints"]["explicit_query_target_verified"] is False
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `C:\Users\lenovo\anaconda3\python.exe -m pytest code/C8/tests/test_conversation_state.py -k "recommendation_mode or current_dish_carries_source or correction_query_adds_explicit_target" -q`

Expected: FAIL because the helper methods and snapshot builder do not exist in the new shape.

- [ ] **Step 3: Extend `ConversationManager` with structured state fields**

```python
@dataclass
class SessionState:
    session_id: str
    created_at: float
    last_active: float
    is_active: bool = True
    messages: List[Message] = field(default_factory=list)
    current_entity: Optional[str] = None
    current_intent: str = "general"
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    topic_mode: str = "none"
    recent_recommendations: List[Dict[str, Any]] = field(default_factory=list)
    recent_topics: List[str] = field(default_factory=list)
    last_confirmed_target: Optional[str] = None
    current_entity_meta: Dict[str, Any] = field(default_factory=dict)
    pending_clarification: Optional[Dict[str, Any]] = None


def record_recommendations(self, session_id: str, dishes: List[str]):
    session = self.get_session(session_id)
    session.topic_mode = "recommendation_list"
    session.recent_recommendations = [
        {"rank": index + 1, "dish_name": dish}
        for index, dish in enumerate(dishes)
    ]


def set_current_dish(
    self,
    session_id: str,
    dish_name: str,
    source: str,
    confidence: float,
    updated_at: float | None = None,
):
    session = self.get_session(session_id)
    session.current_entity = dish_name
    session.last_confirmed_target = dish_name
    session.topic_mode = "single_dish"
    session.current_entity_meta = {
        "value": dish_name,
        "source": source,
        "confidence": confidence,
        "updated_at": updated_at or time.time(),
    }
```

- [ ] **Step 4: Add the snapshot builder with current-query-aware constraints**

```python
import re


def _extract_explicit_query_targets(current_query: str) -> list[str]:
    correction_match = re.match(r"^不是这个，是(.+)$", current_query.strip())
    if correction_match:
        return [correction_match.group(1).strip()]
    return []


def _is_verified_explicit_target(target: str) -> bool:
    return len(target) >= 2 and all(ch not in target for ch in "这个那个简单一点一些")


def build_conversation_snapshot(
    session,
    current_query: str,
    now_ts: float | None = None,
) -> dict:
    current_meta = dict(session.current_entity_meta or {})
    now_ts = now_ts or time.time()
    if current_meta:
        age_seconds = now_ts - current_meta["updated_at"]
        current_meta["active"] = not (
            current_meta["source"] == "inferred" and age_seconds > 900
        )
    else:
        current_meta = {
            "value": None,
            "source": "none",
            "confidence": 0.0,
            "updated_at": 0.0,
            "active": False,
        }

    explicit_query_targets = _extract_explicit_query_targets(current_query)
    explicit_query_target_verified = all(
        _is_verified_explicit_target(target) for target in explicit_query_targets
    ) if explicit_query_targets else False

    return {
        "topic_state": {
            "mode": session.topic_mode,
            "current_topic": current_meta["value"] if current_meta["active"] else None,
            "pending_topic": None,
            "last_topic_source": current_meta["source"],
        },
        "reference_state": {
            "current_dish": current_meta,
            "recent_recommendations": list(session.recent_recommendations),
            "recent_topics": list(session.recent_topics),
            "last_confirmed_target": session.last_confirmed_target,
        },
        "conversation_state": {
            "last_user_query": next((m.content for m in reversed(session.messages) if m.role == "user"), ""),
            "current_user_query": current_query,
            "last_system_action": session.current_intent,
            "last_system_response_summary": "",
            "recent_turns": [
                {"role": m.role, "content": m.content, "intent_type": m.intent_type}
                for m in session.messages[-6:]
            ],
        },
        "resolution_constraints": {
            "allowed_reference_targets": [
                item["dish_name"] for item in session.recent_recommendations
            ] or ([current_meta["value"]] if current_meta["active"] and current_meta["value"] else []),
            "explicit_query_targets": explicit_query_targets,
            "allow_external_explicit_target": bool(explicit_query_targets),
            "explicit_query_target_verified": explicit_query_target_verified,
            "allow_default_selection": False,
            "must_clarify_if_ambiguous": True,
            "allow_topic_switch_detection": True,
            "priority_order": [
                "explicit_query_target",
                "last_confirmed_target",
                "ordinal_recommendation_reference",
                "pronoun_recommendation_reference",
                "current_dish",
            ],
        },
        "state_health": {
            "state_version": 1,
            "last_reliable_turn_id": None,
            "has_ambiguous_reference": not current_meta["active"] and bool(current_meta["value"]),
            "has_pending_clarification": bool(session.pending_clarification),
        },
    }
```

- [ ] **Step 5: Run tests and commit**

Run:
- `C:\Users\lenovo\anaconda3\python.exe -m pytest code/C8/tests/test_conversation_state.py -q`

Expected: PASS

Commit:

```bash
git add code/C8/rag_modules/conversation_manager.py code/C8/rag_modules/conversation_state_builder.py code/C8/tests/test_conversation_state.py
git commit -m "feat: add structured conversation snapshot"
```

---

### Task 3: Add Model-Assisted Reference Resolution With Reachable Correction And Query Rewrite

**Files:**
- Create: `code/C8/rag_modules/reference_resolution.py`
- Modify: `code/C8/main.py`
- Test: `code/C8/tests/test_reference_resolution.py`

- [ ] **Step 1: Write the failing reference-resolution tests**

```python
from rag_modules.reference_resolution import (
    guard_resolution_output,
    resolve_reference_from_snapshot,
    rewrite_query_for_execution,
)


def test_ambiguous_recommendation_followup_requires_clarification():
    snapshot = {
        "topic_state": {"mode": "recommendation_list"},
        "reference_state": {
            "current_dish": {"value": None, "source": "none", "confidence": 0.0, "updated_at": 0.0, "active": False},
            "recent_recommendations": [
                {"rank": 1, "dish_name": "蛋炒饭"},
                {"rank": 2, "dish_name": "麻辣香锅"},
            ],
            "recent_topics": [],
            "last_confirmed_target": None,
        },
        "conversation_state": {
            "last_user_query": "今天吃什么？",
            "current_user_query": "它怎么做？",
        },
        "resolution_constraints": {
            "allowed_reference_targets": ["蛋炒饭", "麻辣香锅"],
            "explicit_query_targets": [],
            "allow_external_explicit_target": False,
            "allow_default_selection": False,
            "must_clarify_if_ambiguous": True,
            "allow_topic_switch_detection": True,
            "priority_order": [
                "explicit_query_target",
                "last_confirmed_target",
                "ordinal_recommendation_reference",
                "pronoun_recommendation_reference",
                "current_dish",
            ],
        },
    }
    result = resolve_reference_from_snapshot(snapshot, llm=None)
    assert result["resolution_status"] == "ambiguous"
    assert result["next_action"] == "ask_clarification"
    assert result["writeback_eligible"] is False


def test_explicit_correction_is_not_blocked_by_old_candidates():
    snapshot = {
        "topic_state": {"mode": "single_dish"},
        "reference_state": {
            "current_dish": {"value": "宫保鸡丁", "source": "inferred", "confidence": 0.55, "updated_at": 10.0, "active": True},
            "recent_recommendations": [],
            "recent_topics": [],
            "last_confirmed_target": "宫保鸡丁",
        },
        "conversation_state": {
            "last_user_query": "它怎么做？",
            "current_user_query": "不是这个，是蛋炒饭",
        },
        "resolution_constraints": {
            "allowed_reference_targets": ["宫保鸡丁"],
            "explicit_query_targets": ["蛋炒饭"],
            "allow_external_explicit_target": True,
            "explicit_query_target_verified": True,
            "allow_default_selection": False,
            "must_clarify_if_ambiguous": True,
            "allow_topic_switch_detection": True,
            "priority_order": [
                "explicit_query_target",
                "last_confirmed_target",
                "ordinal_recommendation_reference",
                "pronoun_recommendation_reference",
                "current_dish",
            ],
        },
    }
    result = guard_resolution_output(
        resolve_reference_from_snapshot(snapshot, llm=None),
        snapshot["resolution_constraints"],
    )
    assert result["resolved_target"] == "蛋炒饭"
    assert result["next_action"] == "apply_correction"


def test_unverified_correction_target_requires_clarification():
    snapshot = {
        "topic_state": {"mode": "single_dish"},
        "reference_state": {
            "current_dish": {"value": "宫保鸡丁", "source": "inferred", "confidence": 0.55, "updated_at": 10.0, "active": True},
            "recent_recommendations": [],
            "recent_topics": [],
            "last_confirmed_target": "宫保鸡丁",
        },
        "conversation_state": {
            "last_user_query": "它怎么做？",
            "current_user_query": "不是这个，是那个简单点的",
        },
        "resolution_constraints": {
            "allowed_reference_targets": ["宫保鸡丁"],
            "explicit_query_targets": ["那个简单点的"],
            "allow_external_explicit_target": True,
            "explicit_query_target_verified": False,
            "allow_default_selection": False,
            "must_clarify_if_ambiguous": True,
            "allow_topic_switch_detection": True,
            "priority_order": [
                "explicit_query_target",
                "last_confirmed_target",
                "ordinal_recommendation_reference",
                "pronoun_recommendation_reference",
                "current_dish",
            ],
        },
    }
    result = resolve_reference_from_snapshot(snapshot, llm=None)
    assert result["next_action"] == "ask_clarification"
    assert result["decision_basis"] == "ambiguous"


def test_correction_rewrites_query_for_retrieval():
    execution_plan = {"action": "apply_correction"}
    resolution = {"resolved_target": "蛋炒饭"}
    query_plan = {"route_type": "detail", "content_type": "steps"}
    rewritten = rewrite_query_for_execution("不是这个，是蛋炒饭", execution_plan, resolution, query_plan)
    assert rewritten == "蛋炒饭怎么做"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `C:\Users\lenovo\anaconda3\python.exe -m pytest code/C8/tests/test_reference_resolution.py -q`

Expected: FAIL because the module and helper functions do not exist.

- [ ] **Step 3: Implement resolution, guard, and query rewrite helpers**

```python
def resolve_reference_from_snapshot(snapshot: dict, llm) -> dict:
    constraints = snapshot["resolution_constraints"]
    query = snapshot["conversation_state"]["current_user_query"]
    explicit_targets = constraints["explicit_query_targets"]
    explicit_query_target_verified = constraints["explicit_query_target_verified"]
    candidates = constraints["allowed_reference_targets"]

    if explicit_targets:
        if not explicit_query_target_verified:
            return {
                "resolution_status": "ambiguous",
                "resolved_target": None,
                "target_source": None,
                "confidence": 0.0,
                "reason": "explicit_target_not_verified",
                "next_action": "ask_clarification",
                "clarification_question": "请直接告诉我明确的菜名，我再继续帮你查做法。",
                "writeback_eligible": False,
                "decision_basis": "ambiguous",
            }
        return {
            "resolution_status": "resolved",
            "resolved_target": explicit_targets[0],
            "target_source": "explicit_query_target",
            "confidence": 1.0,
            "reason": "user_correction",
            "next_action": "apply_correction",
            "clarification_question": None,
            "writeback_eligible": True,
            "decision_basis": "explicit",
        }

    if snapshot["topic_state"]["mode"] == "recommendation_list" and query.startswith(("它", "这个", "那个")):
        return {
            "resolution_status": "ambiguous",
            "resolved_target": None,
            "target_source": None,
            "confidence": 0.0,
            "reason": "multiple_candidates_in_recommendation_list",
            "next_action": "ask_clarification",
            "clarification_question": "你是指第几个推荐菜，还是直接告诉我菜名？",
            "writeback_eligible": False,
            "decision_basis": "ambiguous",
        }

    if len(candidates) == 1:
        return {
            "resolution_status": "resolved",
            "resolved_target": candidates[0],
            "target_source": "current_dish",
            "confidence": 1.0,
            "reason": "single_candidate",
            "next_action": "retrieve_detail",
            "clarification_question": None,
            "writeback_eligible": True,
            "decision_basis": "inferred",
        }

    return {
        "resolution_status": "no_reference_needed",
        "resolved_target": None,
        "target_source": None,
        "confidence": 0.0,
        "reason": "no_reference_needed",
        "next_action": "continue_general",
        "clarification_question": None,
        "writeback_eligible": False,
        "decision_basis": "none",
    }


def guard_resolution_output(result: dict, constraints: dict) -> dict:
    allowed_targets = constraints["allowed_reference_targets"]
    explicit_targets = constraints["explicit_query_targets"]
    allow_external_explicit_target = constraints["allow_external_explicit_target"]

    if result.get("decision_basis") == "explicit" and result.get("resolved_target") in explicit_targets:
        return result

    if (
        result.get("decision_basis") == "explicit"
        and allow_external_explicit_target
        and result.get("resolved_target")
    ):
        return result

    if result.get("resolved_target") and result["resolved_target"] not in allowed_targets:
        return {
            "resolution_status": "ambiguous",
            "resolved_target": None,
            "target_source": None,
            "confidence": 0.0,
            "reason": "resolved_target_not_in_allowed_candidates",
            "next_action": "ask_clarification",
            "clarification_question": "请直接告诉我菜名，或者说明是第几个推荐菜。",
            "writeback_eligible": False,
            "decision_basis": "ambiguous",
        }
    return result


def rewrite_query_for_execution(
    original_question: str,
    execution_plan: dict,
    resolution: dict | None,
    query_plan: dict | None,
) -> str:
    if execution_plan["action"] == "apply_correction" and resolution and resolution.get("resolved_target"):
        content_type = (query_plan or {}).get("content_type")
        if content_type == "ingredients":
            return f"{resolution['resolved_target']}需要什么食材"
        return f"{resolution['resolved_target']}怎么做"
    return original_question
```

- [ ] **Step 4: Integrate reference resolution into `ask_question()`**

```python
snapshot = build_conversation_snapshot(
    self.generation_module.conversation_manager.get_session(session_id),
    current_query=question,
)

resolution = None
if turn_info["should_run_reference_resolution"]:
    resolution = resolve_reference_from_snapshot(snapshot, self.generation_module.llm)
    resolution = guard_resolution_output(
        resolution,
        snapshot["resolution_constraints"],
    )

if resolution and resolution["next_action"] == "ask_clarification":
    answer = resolution["clarification_question"]
    self._write_conversation_turn(
        session_id=session_id,
        question=question,
        answer=answer,
        turn_info=turn_info,
        query_plan=None,
        resolution=resolution,
        execution_result={"success": True, "answer": answer},
    )
    return answer
```

- [ ] **Step 5: Run tests and commit**

Run:
- `C:\Users\lenovo\anaconda3\python.exe -m pytest code/C8/tests/test_reference_resolution.py -q`
- `C:\Users\lenovo\anaconda3\python.exe -m pytest code/C8/tests/test_conversation_state.py -q`

Expected: PASS

Commit:

```bash
git add code/C8/rag_modules/reference_resolution.py code/C8/main.py code/C8/tests/test_reference_resolution.py code/C8/tests/test_conversation_state.py
git commit -m "feat: add guarded reference resolution"
```

---

### Task 4: Make Execution Planning Real And Drive Writeback From Execution Results

**Files:**
- Create: `code/C8/rag_modules/execution_planner.py`
- Create: `code/C8/rag_modules/state_writeback_review.py`
- Modify: `code/C8/main.py`
- Modify: `code/C8/rag_modules/conversation_manager.py`
- Test: `code/C8/tests/test_conversation_state.py`
- Test: `code/C8/tests/test_state_writeback_review.py`

- [ ] **Step 1: Write the failing execution-planning and writeback-review tests**

```python
from rag_modules.execution_planner import build_execution_plan
from rag_modules.state_writeback_review import review_state_writeback


def test_recommendation_query_returns_retrieve_list_action():
    plan = build_execution_plan(
        turn_info={"turn_type": "recommendation_query", "response_mode": "retrieve_answer"},
        resolution=None,
    )
    assert plan["action"] == "retrieve_list"


def test_correction_resolution_returns_apply_correction_plan():
    plan = build_execution_plan(
        turn_info={"turn_type": "followup_query", "response_mode": "retrieve_answer"},
        resolution={"next_action": "apply_correction", "resolved_target": "蛋炒饭"},
    )
    assert plan["action"] == "apply_correction"


def test_failed_execution_blocks_reliable_writeback():
    review = review_state_writeback(
        turn_info={"turn_type": "followup_query"},
        resolution={"resolved_target": "蛋炒饭", "writeback_eligible": True, "decision_basis": "inferred"},
        execution_result={"success": False},
        answer="暂时失败",
    )
    assert review["should_write_reliable_state"] is False


def test_recommendation_writeback_reads_from_execution_result():
    review = review_state_writeback(
        turn_info={"turn_type": "recommendation_query"},
        resolution=None,
        execution_result={"success": True, "recommended_dishes": ["蛋炒饭", "麻辣香锅"]},
        answer="我推荐蛋炒饭和麻辣香锅",
    )
    assert review["writeback_mode"] == "recommendation_list"


def test_empty_recommendation_result_does_not_write_recommendation_state():
    review = review_state_writeback(
        turn_info={"turn_type": "recommendation_query"},
        resolution=None,
        execution_result={"success": True, "recommended_dishes": []},
        answer="今天可以吃点清淡的",
    )
    assert review["writeback_mode"] == "message_only"
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
- `C:\Users\lenovo\anaconda3\python.exe -m pytest code/C8/tests/test_conversation_state.py -k "execution_plan" -q`
- `C:\Users\lenovo\anaconda3\python.exe -m pytest code/C8/tests/test_state_writeback_review.py -q`

Expected: FAIL because the planner and reviewer do not exist.

- [ ] **Step 3: Add execution planner and writeback reviewer**

```python
def build_execution_plan(turn_info: dict, resolution: dict | None) -> dict:
    if turn_info["response_mode"] == "polite_direct_reply":
        return {"action": "direct_smalltalk_reply", "message": None}
    if resolution and resolution.get("next_action") == "ask_clarification":
        return {"action": "ask_clarification", "message": resolution["clarification_question"]}
    if resolution and resolution.get("next_action") == "apply_correction":
        return {"action": "apply_correction", "message": None}
    if turn_info.get("turn_type") == "recommendation_query":
        return {"action": "retrieve_list", "message": None}
    return {"action": "retrieve_detail", "message": None}


def review_state_writeback(
    turn_info: dict,
    resolution: dict | None,
    execution_result: dict,
    answer: str,
) -> dict:
    if turn_info["turn_type"] in {"smalltalk", "out_of_domain"}:
        return {"should_write_reliable_state": False, "writeback_mode": "message_only"}
    if not execution_result.get("success"):
        return {"should_write_reliable_state": False, "writeback_mode": "message_only"}
    if turn_info["turn_type"] == "recommendation_query":
        if not execution_result.get("recommended_dishes"):
            return {"should_write_reliable_state": False, "writeback_mode": "message_only"}
        return {"should_write_reliable_state": True, "writeback_mode": "recommendation_list"}
    if resolution and resolution.get("next_action") == "apply_correction":
        return {"should_write_reliable_state": True, "writeback_mode": "correction_turn"}
    if resolution and not resolution.get("writeback_eligible", False):
        return {"should_write_reliable_state": False, "writeback_mode": "message_only"}
    return {"should_write_reliable_state": True, "writeback_mode": "normal"}
```

- [ ] **Step 3.1: Add recommendation extraction helper in `RecipeRAGSystem`**

```python
def _extract_recommended_dishes(self, answer: str, parent_docs: list) -> list[str]:
    dish_names = []
    for doc in parent_docs:
        dish_name = (doc.metadata or {}).get("dish_name")
        if dish_name and dish_name not in dish_names:
            dish_names.append(dish_name)

    if dish_names:
        return dish_names[:5]

    fallback = []
    for line in answer.splitlines():
        cleaned = line.strip().lstrip("-").lstrip("1234567890.、 ").strip()
        if 1 < len(cleaned) <= 20 and cleaned not in fallback:
            fallback.append(cleaned)
    return fallback[:5]
```

- [ ] **Step 4: Make `ask_question()` execution-plan-driven**

```python
execution_plan = build_execution_plan(turn_info, resolution)
base_query_plan = self._build_query_plan(question, session_id)
rewritten_question = rewrite_query_for_execution(question, execution_plan, resolution, base_query_plan)
query_plan = (
    self._build_query_plan(rewritten_question, session_id)
    if rewritten_question != question
    else base_query_plan
)

if execution_plan["action"] == "ask_clarification":
    answer = execution_plan["message"]
    execution_result = {"success": True, "answer": answer}
    self._write_conversation_turn(
        session_id=session_id,
        question=question,
        answer=answer,
        turn_info=turn_info,
        query_plan=query_plan,
        resolution=resolution,
        execution_result=execution_result,
    )
    return answer

filters = query_plan["filters"]
dish_name = query_plan["dish_name"]
route_type = query_plan["route_type"]
entities = query_plan["entities"]
relevant_chunks = self._search_relevant_chunks(
    question,
    rewritten_question,
    filters,
    dish_name,
)

if not relevant_chunks:
    answer = "抱歉，没有找到相关的食谱信息。请尝试其他菜品名称或关键词。"
    execution_result = {
        "success": False,
        "answer": answer,
        "final_query_text": rewritten_question,
        "query_plan_source": "rewritten" if rewritten_question != question else "original",
    }
    self._write_conversation_turn(
        session_id=session_id,
        question=question,
        answer=answer,
        turn_info=turn_info,
        query_plan=query_plan,
        resolution=resolution,
        execution_result=execution_result,
    )
    return answer

if execution_plan["action"] == "retrieve_list":
    answer = self._generate_list_response(rewritten_question, session_id, relevant_chunks)
    recommended_dishes = self._extract_recommended_dishes(answer, list(self._latest_parent_docs))
    execution_result = {
        "success": True,
        "answer": answer,
        "recommended_dishes": recommended_dishes,
        "final_query_text": rewritten_question,
        "query_plan_source": "rewritten" if rewritten_question != question else "original",
    }
else:
    answer = self._generate_detail_response(
        rewritten_question,
        False,
        session_id,
        route_type,
        filters,
        entities,
        dish_name,
        relevant_chunks,
    )
    execution_result = {
        "success": True,
        "answer": answer,
        "resolved_target": resolution.get("resolved_target") if resolution else None,
        "final_query_text": rewritten_question,
        "query_plan_source": "rewritten" if rewritten_question != question else "original",
    }

self._write_conversation_turn(
    session_id=session_id,
    question=question,
    answer=answer,
    turn_info=turn_info,
    query_plan=query_plan,
    resolution=resolution,
    execution_result=execution_result,
)
return answer
```

`query_plan` must never be reused across a correction rewrite. If `rewritten_question != question`, the plan is rebuilt from the rewritten text before retrieval.

- [ ] **Step 4.1: Replace unconditional entity writeback with conditional reviewed writeback**

```python
def writeback_turn_state(
    self,
    session_id: str,
    question: str,
    turn_info: dict,
    query_plan: dict | None,
    resolution: dict | None,
    answer: str,
    execution_result: dict,
):
    review = review_state_writeback(
        turn_info=turn_info,
        resolution=resolution,
        execution_result=execution_result,
        answer=answer,
    )

    if review["writeback_mode"] == "message_only":
        self.add_interaction(session_id, question, answer, intent_type=turn_info["turn_type"], entities={})
        return

    if review["writeback_mode"] == "recommendation_list":
        self.record_recommendations(session_id, execution_result.get("recommended_dishes", []))
        self.add_interaction(session_id, question, answer, intent_type="list", entities={})
        return

    if review["writeback_mode"] == "correction_turn":
        self.set_current_dish(
            session_id,
            execution_result["resolved_target"],
            source="explicit_query",
            confidence=1.0,
        )
        self.add_interaction(session_id, question, answer, intent_type="correction_turn", entities={"dish": execution_result["resolved_target"]})
        return

    if execution_result.get("resolved_target"):
        self.set_current_dish(
            session_id,
            execution_result["resolved_target"],
            source=resolution.get("target_source", "resolved_followup") if resolution else "resolved_followup",
            confidence=resolution.get("confidence", 0.0) if resolution else 0.0,
        )
    entities = query_plan.get("entities", {}) if query_plan else {}
    self.add_interaction(session_id, question, answer, intent_type=query_plan.get("route_type", "general"), entities=entities)
```

- [ ] **Step 5: Run tests and commit**

Run:
- `C:\Users\lenovo\anaconda3\python.exe -m pytest code/C8/tests/test_conversation_state.py -q`
- `C:\Users\lenovo\anaconda3\python.exe -m pytest code/C8/tests/test_state_writeback_review.py -q`
- `C:\Users\lenovo\anaconda3\python.exe -m pytest code/C8/tests/test_web_app.py -q`

Expected: PASS

Commit:

```bash
git add code/C8/rag_modules/execution_planner.py code/C8/rag_modules/state_writeback_review.py code/C8/main.py code/C8/rag_modules/conversation_manager.py code/C8/tests/test_conversation_state.py code/C8/tests/test_state_writeback_review.py code/C8/tests/test_web_app.py
git commit -m "feat: add execution planning and guarded writeback"
```

---

### Task 5: Add Full-Chain Real Integration Coverage For Ambiguity And Correction

**Files:**
- Create: `code/C8/tests/test_conversation_integration_real.py`
- Modify: `code/C8/web_app.py`
- Modify: `code/C8/tests/test_web_app.py`

- [ ] **Step 1: Write the failing real integration tests**

```python
import os
import pytest

from web_app import create_app


@pytest.mark.real_integration
def test_recommendation_followup_requires_clarification_not_wrong_inheritance():
    if not os.getenv("DASHSCOPE_API_KEY"):
        pytest.skip("DASHSCOPE_API_KEY is required for real integration tests")

    app = create_app()
    client = app.test_client()
    session_id = "real-followup-session"

    first = client.post("/api/chat", json={"question": "今天吃什么？", "session_id": session_id})
    assert first.status_code == 200

    second = client.post("/api/chat", json={"question": "它怎么做？", "session_id": session_id})
    assert second.status_code == 200

    answer = second.get_json()["answer"]
    assert "今天吃什么怎么做" not in answer
    assert ("第几个推荐菜" in answer) or ("直接告诉我菜名" in answer)


@pytest.mark.real_integration
def test_user_correction_overrides_previous_inference():
    if not os.getenv("DASHSCOPE_API_KEY"):
        pytest.skip("DASHSCOPE_API_KEY is required for real integration tests")

    app = create_app()
    client = app.test_client()
    session_id = "real-correction-session"

    first = client.post("/api/chat", json={"question": "宫保鸡丁怎么做？", "session_id": session_id})
    assert first.status_code == 200

    second = client.post("/api/chat", json={"question": "不是这个，是蛋炒饭", "session_id": session_id})
    assert second.status_code == 200

    answer = second.get_json()["answer"]
    assert "蛋炒饭" in answer
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `C:\Users\lenovo\anaconda3\python.exe -m pytest code/C8/tests/test_conversation_integration_real.py -m real_integration -q`

Expected: FAIL on the current real-chain behavior, or SKIP if no API key is available.

- [ ] **Step 3: Add real-test-safe app setup and logging**

```python
def create_app(
    system_factory: Optional[Callable[[], RecipeRAGSystem]] = None,
    log_path: Optional[Path] = None,
) -> Flask:
    _configure_file_logging(log_path or Path(__file__).with_name("web_app.runtime.log"))
    app = Flask(__name__)
    app.config["SYSTEM_FACTORY"] = system_factory or _default_system_factory
    app.config["RAG_SYSTEM"] = None
    app.config["RAG_LOCK"] = threading.Lock()
    return app
```

Use the existing helper already present in `code/C8/web_app.py`:

```python
def _configure_file_logging(log_path: Path) -> None:
    root_logger = logging.getLogger()
    resolved_path = log_path.resolve()
    for handler in root_logger.handlers:
        if (
            isinstance(handler, logging.FileHandler)
            and Path(getattr(handler, "baseFilename", "")).resolve() == resolved_path
        ):
            return

    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(resolved_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    root_logger.addHandler(file_handler)
    if root_logger.level > logging.INFO:
        root_logger.setLevel(logging.INFO)
```

Add log assertions when debugging failures:

```python
assert "final_query_text=今天吃什么" not in log_text
assert "writeback_mode=correction_turn" in log_text
```

- [ ] **Step 4: Run the full targeted suite**

Run:
- `C:\Users\lenovo\anaconda3\python.exe -m pytest code/C8/tests/test_turn_qualification.py code/C8/tests/test_reference_resolution.py code/C8/tests/test_state_writeback_review.py code/C8/tests/test_conversation_state.py code/C8/tests/test_web_app.py -q`
- `C:\Users\lenovo\anaconda3\python.exe -m pytest code/C8/tests/test_conversation_integration_real.py -m real_integration -q`

Expected:
- First command PASS
- Second command PASS or SKIP only when credentials are unavailable

- [ ] **Step 5: Commit**

```bash
git add code/C8/tests/test_conversation_integration_real.py code/C8/web_app.py code/C8/tests/test_web_app.py
git commit -m "test: add real conversation integration coverage"
```

---

## Spec Coverage Check

- Turn qualification and separation of smalltalk vs guardrail are covered by Task 1.
- Structured state, source/confidence metadata, state health, invalidation, and correction-aware constraints are covered by Task 2.
- Model-assisted reference resolution, reachable correction handling, and query rewrite are covered by Task 3.
- Execution planning, execution-result-driven writeback, correction override, and failure-gated writeback are covered by Task 4.
- Real-chain regression coverage for ambiguity and user correction is covered by Task 5.

No spec section is left without a corresponding implementation task.

## Placeholder Scan

Checked for:
- `TODO`
- `TBD`
- “appropriate error handling”
- “similar to above”

None remain in the plan.

## Type Consistency Check

The plan uses these stable names across tasks:
- `qualify_turn`
- `build_conversation_snapshot`
- `resolve_reference_from_snapshot`
- `guard_resolution_output`
- `rewrite_query_for_execution`
- `build_execution_plan`
- `review_state_writeback`
- `writeback_turn_state`

The same names are reused consistently in tests and implementation steps.
