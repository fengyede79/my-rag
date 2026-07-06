# Ordinal Reference And Natural Followup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the new conversation framework so natural user follow-ups like “第二个怎么做？”, “那蛋炒饭需要哪些食材？”, and “有什么小技巧别粘锅？” resolve through structured state instead of being misread as dish names.

**Architecture:** Extend the existing new pipeline without restoring the old parallel ordinal resolver. Turn qualification detects reference-triggering natural follow-ups, the snapshot builder extracts structured ordinal/cleaned-dish/implicit-followup facts, reference resolution maps them through guarded state, and execution rewrite sends a concrete dish query into the existing retrieval/generation stack.

**Tech Stack:** Python, pytest, Flask test client, existing `code/C8` RAG modules, DashScope-backed real integration tests

---

## File Map

**Modify:**
- `code/C8/rag_modules/turn_qualification.py`
- `code/C8/rag_modules/conversation_state_builder.py`
- `code/C8/rag_modules/reference_resolution.py`
- `code/C8/rag_modules/execution_planner.py`
- `code/C8/main.py`
- `code/C8/rag_modules/generation_integration.py`
- `code/C8/tests/test_turn_qualification.py`
- `code/C8/tests/test_conversation_state.py`
- `code/C8/tests/test_reference_resolution.py`
- `code/C8/tests/test_conversation_integration_real.py`

**Do not restore or recreate:**
- `GenerationIntegrationModule.resolve_query_reference()`
- `GenerationIntegrationModule.save_recommendations()`
- old recommendation-cache based ordinal resolution
- main-path `ConversationManager.complete_query()` inheritance
- old conversation-aware generation methods that write state internally

**Reference:**
- `code/C8/docs/superpowers/specs/2026-07-04-ordinal-reference-natural-followup.md`

---

## Task 1: Make Natural Reference Turns Enter Reference Resolution

**Files:**
- Modify: `code/C8/rag_modules/turn_qualification.py`
- Test: `code/C8/tests/test_turn_qualification.py`

- [x] **Step 1: Write failing tests for ordinal and short follow-up admission**

Append these tests to `code/C8/tests/test_turn_qualification.py`:

```python
def test_ordinal_turn_requests_reference_resolution():
    result = qualify_turn("第二个怎么做？")
    assert result["turn_type"] == "followup_query"
    assert result["should_run_reference_resolution"] is True
    assert result["reference_trigger"] == "ordinal_reference"


def test_ordinal_with_comment_requests_reference_resolution():
    result = qualify_turn("第一个看起来不错，做法说一下")
    assert result["turn_type"] == "followup_query"
    assert result["should_run_reference_resolution"] is True
    assert result["reference_trigger"] == "ordinal_reference"


def test_short_detail_followup_requests_reference_resolution():
    result = qualify_turn("有什么小技巧别粘锅？")
    assert result["turn_type"] == "followup_query"
    assert result["should_run_reference_resolution"] is True
    assert result["reference_trigger"] == "implicit_detail_followup"
```

- [x] **Step 2: Run tests and confirm they fail**

Run:

```powershell
C:\Users\lenovo\anaconda3\python.exe -m pytest code/C8/tests/test_turn_qualification.py -q
```

Expected: FAIL because `reference_trigger` is absent and ordinal/short follow-ups are still classified as `domain_query`.

- [x] **Step 3: Add ordinal and short follow-up detection**

In `code/C8/rag_modules/turn_qualification.py`, add:

```python
ORDINAL_REFERENCE_PATTERNS = (
    "第一个", "第二个", "第三个", "第四个", "第五个",
    "第1个", "第2个", "第3个", "第4个", "第5个",
    "1", "2", "3", "4", "5",
    "1号", "2号", "3号", "4号", "5号",
)

DETAIL_FOLLOWUP_KEYWORDS = (
    "怎么做", "做法", "食材", "材料", "原料", "配料",
    "技巧", "粘锅", "难不难", "要多久", "热量", "介绍",
)


def _followup_result(reference_trigger: str) -> dict:
    return {
        "turn_type": "followup_query",
        "should_retrieve": True,
        "should_update_topic_state": False,
        "should_update_entity_state": False,
        "should_run_reference_resolution": True,
        "response_mode": "retrieve_answer",
        "reference_trigger": reference_trigger,
    }


def _starts_with_ordinal_reference(normalized: str) -> bool:
    return any(
        normalized == pattern or normalized.startswith(pattern)
        for pattern in ORDINAL_REFERENCE_PATTERNS
    )


def _looks_like_short_detail_followup(normalized: str) -> bool:
    if len(normalized) > 18:
        return False
    return any(keyword in normalized for keyword in DETAIL_FOLLOWUP_KEYWORDS)
```

Then in `qualify_turn()` before the final `domain_query` return:

```python
    if _starts_with_ordinal_reference(normalized):
        return _followup_result("ordinal_reference")

    if _looks_like_short_detail_followup(normalized):
        return _followup_result("implicit_detail_followup")
```

For existing follow-up and correction branches, return `_followup_result("pronoun_or_correction")` or merge `reference_trigger` into the existing dict.

- [x] **Step 4: Run tests and confirm they pass**

Run:

```powershell
C:\Users\lenovo\anaconda3\python.exe -m pytest code/C8/tests/test_turn_qualification.py -q
```

Expected: PASS.

---

## Task 2: Extract Ordinal References, Cleaned Dish Names, And Preference Constraints Into Snapshot

**Files:**
- Modify: `code/C8/rag_modules/conversation_state_builder.py`
- Test: `code/C8/tests/test_conversation_state.py`

- [x] **Step 1: Write failing snapshot tests**

Append these tests to `code/C8/tests/test_conversation_state.py`:

```python
def test_snapshot_extracts_ordinal_reference():
    manager = ConversationManager()
    manager.record_recommendations("ordinal-s1", ["扬州炒饭", "麻婆豆腐", "白灼菜心"])

    snapshot = build_conversation_snapshot(
        manager.get_session("ordinal-s1"),
        current_query="第二个怎么做？",
    )

    ordinal = snapshot["resolution_constraints"]["ordinal_reference"]
    assert ordinal["rank"] == 2
    assert ordinal["raw_text"] == "第二个"
    assert ordinal["remaining_query"] == "怎么做"


def test_snapshot_extracts_ordinal_reference_with_comment():
    manager = ConversationManager()
    manager.record_recommendations("ordinal-s2", ["燕麦鸡蛋饼", "牛奶燕麦"])

    snapshot = build_conversation_snapshot(
        manager.get_session("ordinal-s2"),
        current_query="第一个看起来不错，做法说一下",
    )

    ordinal = snapshot["resolution_constraints"]["ordinal_reference"]
    assert ordinal["rank"] == 1
    assert ordinal["remaining_query"] == "做法说一下"


def test_snapshot_cleans_discourse_prefix_from_explicit_dish():
    manager = ConversationManager()

    snapshot = build_conversation_snapshot(
        manager.get_session("clean-prefix-s1"),
        current_query="那蛋炒饭需要哪些食材？",
    )

    cleaned = snapshot["resolution_constraints"]["cleaned_explicit_dish"]
    assert cleaned["value"] == "蛋炒饭"
    assert cleaned["removed_prefix"] == "那"


def test_snapshot_extracts_preference_constraints():
    manager = ConversationManager()

    snapshot = build_conversation_snapshot(
        manager.get_session("preference-s1"),
        current_query="算了，换个清淡一点的菜",
    )

    preferences = snapshot["resolution_constraints"]["preference_constraints"]
    assert "清淡" in preferences["taste"]
```

- [x] **Step 2: Run tests and confirm they fail**

Run:

```powershell
C:\Users\lenovo\anaconda3\python.exe -m pytest code/C8/tests/test_conversation_state.py -k "ordinal_reference or discourse_prefix or preference_constraints" -q
```

Expected: FAIL because these snapshot fields do not exist.

- [x] **Step 3: Add ordinal extraction helpers**

In `code/C8/rag_modules/conversation_state_builder.py`, add:

```python
CHINESE_ORDINAL_TO_RANK = {
    "一": 1,
    "二": 2,
    "三": 3,
    "四": 4,
    "五": 5,
}

ORDINAL_COMMENT_PATTERNS = (
    "看起来不错",
    "看起来挺好",
    "不错",
    "挺好",
    "可以",
)


def _strip_ordinal_comment(text: str) -> str:
    cleaned = text.strip("，,。！？? ")
    for pattern in ORDINAL_COMMENT_PATTERNS:
        cleaned = cleaned.replace(pattern, "")
    return cleaned.strip("，,。！？? ")


def _extract_ordinal_reference(current_query: str) -> dict | None:
    text = current_query.strip()
    match = re.match(r"^(第\s*(?P<cn>[一二三四五])\s*个|第\s*(?P<num>[1-5])\s*个|(?P<plain>[1-5])\s*号?)(?P<rest>.*)$", text)
    if not match:
        return None

    rank = None
    if match.group("cn"):
        rank = CHINESE_ORDINAL_TO_RANK[match.group("cn")]
    elif match.group("num"):
        rank = int(match.group("num"))
    elif match.group("plain"):
        rank = int(match.group("plain"))

    raw_text = text[: match.end() - len(match.group("rest"))]
    remaining_query = _strip_ordinal_comment(match.group("rest") or "")
    if not remaining_query:
        remaining_query = "怎么做"

    return {
        "rank": rank,
        "raw_text": raw_text.strip(),
        "remaining_query": remaining_query,
    }
```

- [x] **Step 4: Add cleaned explicit dish helper**

In the same file, add:

```python
DISCOURSE_PREFIXES = ("那", "这个", "这道", "刚才那个", "刚才这道")
DETAIL_SUFFIXES = (
    "需要哪些食材", "需要什么食材", "有什么食材",
    "怎么做", "做法", "做法说一下", "有什么小技巧", "有哪些技巧",
)


def _extract_cleaned_explicit_dish(current_query: str) -> dict | None:
    text = current_query.strip().rstrip("?!？！。")
    for prefix in sorted(DISCOURSE_PREFIXES, key=len, reverse=True):
        if not text.startswith(prefix):
            continue
        candidate = text[len(prefix):]
        for suffix in sorted(DETAIL_SUFFIXES, key=len, reverse=True):
            if suffix in candidate:
                dish = candidate.split(suffix, 1)[0].strip("，,。 ")
                if 2 <= len(dish) <= 12:
                    return {"value": dish, "removed_prefix": prefix}
    return None
```

- [x] **Step 5: Add implicit follow-up and preference helpers**

In the same file, add:

```python
IMPLICIT_FOLLOWUP_KEYWORDS = ("怎么做", "做法", "食材", "技巧", "粘锅", "难不难", "要多久", "热量")


def _extract_implicit_followup(current_query: str) -> dict:
    text = current_query.strip().rstrip("?!？！。")
    enabled = len(text) <= 18 and any(keyword in text for keyword in IMPLICIT_FOLLOWUP_KEYWORDS)
    return {
        "enabled": enabled,
        "remaining_query": text,
        "requires_single_active_dish": True,
    }


def _extract_preference_constraints(current_query: str) -> dict:
    text = current_query.strip()
    preferences = {"taste": [], "meal": [], "difficulty": [], "style": []}
    if "清淡" in text:
        preferences["taste"].append("清淡")
    if "下饭" in text:
        preferences["style"].append("下饭")
    if "新手" in text or "简单" in text:
        preferences["difficulty"].append("新手" if "新手" in text else "简单")
    if "早餐" in text:
        preferences["meal"].append("早餐")
    return preferences
```

- [x] **Step 6: Add fields to `resolution_constraints`**

Inside `build_conversation_snapshot()`, compute:

```python
ordinal_reference = _extract_ordinal_reference(current_query)
cleaned_explicit_dish = _extract_cleaned_explicit_dish(current_query)
implicit_followup = _extract_implicit_followup(current_query)
preference_constraints = _extract_preference_constraints(current_query)
```

Then add to `resolution_constraints`:

```python
            "ordinal_reference": ordinal_reference,
            "cleaned_explicit_dish": cleaned_explicit_dish,
            "implicit_followup": implicit_followup,
            "preference_constraints": preference_constraints,
```

- [x] **Step 7: Run tests and confirm they pass**

Run:

```powershell
C:\Users\lenovo\anaconda3\python.exe -m pytest code/C8/tests/test_conversation_state.py -k "ordinal_reference or discourse_prefix or preference_constraints" -q
```

Expected: PASS.

---

## Task 3: Resolve Ordinal, Cleaned Explicit Dish, And Implicit Follow-Up Through Reference Resolution

**Files:**
- Modify: `code/C8/rag_modules/reference_resolution.py`
- Test: `code/C8/tests/test_reference_resolution.py`

- [x] **Step 1: Write failing reference-resolution tests**

Append these tests to `code/C8/tests/test_reference_resolution.py`:

```python
def test_ordinal_reference_resolves_to_recommendation_rank():
    snapshot = {
        "topic_state": {"mode": "recommendation_list"},
        "reference_state": {
            "current_dish": {"value": None, "source": "none", "confidence": 0.0, "updated_at": 0.0, "active": False},
            "recent_recommendations": [
                {"rank": 1, "dish_name": "扬州炒饭"},
                {"rank": 2, "dish_name": "麻婆豆腐"},
            ],
            "recent_topics": [],
            "last_confirmed_target": None,
        },
        "conversation_state": {"current_user_query": "第二个怎么做？"},
        "resolution_constraints": {
            "allowed_reference_targets": ["扬州炒饭", "麻婆豆腐"],
            "explicit_query_targets": [],
            "allow_external_explicit_target": False,
            "ordinal_reference": {"rank": 2, "raw_text": "第二个", "remaining_query": "怎么做"},
            "cleaned_explicit_dish": None,
            "implicit_followup": {"enabled": False, "remaining_query": "", "requires_single_active_dish": True},
            "must_clarify_if_ambiguous": True,
        },
    }

    result = resolve_reference_from_snapshot(snapshot, llm=None)

    assert result["resolution_status"] == "resolved"
    assert result["resolved_target"] == "麻婆豆腐"
    assert result["target_source"] == "ordinal_recommendation_reference"
    assert result["next_action"] == "apply_reference_resolution"
    assert result["writeback_eligible"] is True


def test_ordinal_reference_out_of_range_asks_clarification():
    snapshot = {
        "topic_state": {"mode": "recommendation_list"},
        "reference_state": {
            "current_dish": {"value": None, "source": "none", "confidence": 0.0, "updated_at": 0.0, "active": False},
            "recent_recommendations": [{"rank": 1, "dish_name": "扬州炒饭"}],
            "recent_topics": [],
            "last_confirmed_target": None,
        },
        "conversation_state": {"current_user_query": "第二个怎么做？"},
        "resolution_constraints": {
            "allowed_reference_targets": ["扬州炒饭"],
            "explicit_query_targets": [],
            "allow_external_explicit_target": False,
            "ordinal_reference": {"rank": 2, "raw_text": "第二个", "remaining_query": "怎么做"},
            "cleaned_explicit_dish": None,
            "implicit_followup": {"enabled": False, "remaining_query": "", "requires_single_active_dish": True},
            "must_clarify_if_ambiguous": True,
        },
    }

    result = resolve_reference_from_snapshot(snapshot, llm=None)

    assert result["resolution_status"] == "ambiguous"
    assert result["next_action"] == "ask_clarification"


def test_cleaned_explicit_dish_resolves_without_old_candidates():
    snapshot = {
        "topic_state": {"mode": "single_dish"},
        "reference_state": {
            "current_dish": {"value": "宫保鸡丁", "source": "explicit_query", "confidence": 1.0, "updated_at": 1.0, "active": True},
            "recent_recommendations": [],
            "recent_topics": [],
            "last_confirmed_target": "宫保鸡丁",
        },
        "conversation_state": {"current_user_query": "那蛋炒饭需要哪些食材？"},
        "resolution_constraints": {
            "allowed_reference_targets": ["宫保鸡丁"],
            "explicit_query_targets": [],
            "allow_external_explicit_target": True,
            "ordinal_reference": None,
            "cleaned_explicit_dish": {"value": "蛋炒饭", "removed_prefix": "那"},
            "implicit_followup": {"enabled": False, "remaining_query": "", "requires_single_active_dish": True},
            "must_clarify_if_ambiguous": True,
        },
    }

    result = resolve_reference_from_snapshot(snapshot, llm=None)

    assert result["resolved_target"] == "蛋炒饭"
    assert result["target_source"] == "cleaned_explicit_dish"
    assert result["next_action"] == "apply_reference_resolution"


def test_implicit_followup_uses_single_active_current_dish():
    snapshot = {
        "topic_state": {"mode": "single_dish"},
        "reference_state": {
            "current_dish": {"value": "蛋炒饭", "source": "explicit_query", "confidence": 1.0, "updated_at": 1.0, "active": True},
            "recent_recommendations": [],
            "recent_topics": [],
            "last_confirmed_target": "蛋炒饭",
        },
        "conversation_state": {"current_user_query": "有什么小技巧别粘锅？"},
        "resolution_constraints": {
            "allowed_reference_targets": ["蛋炒饭"],
            "explicit_query_targets": [],
            "allow_external_explicit_target": False,
            "ordinal_reference": None,
            "cleaned_explicit_dish": None,
            "implicit_followup": {"enabled": True, "remaining_query": "有什么小技巧别粘锅", "requires_single_active_dish": True},
            "must_clarify_if_ambiguous": True,
        },
    }

    result = resolve_reference_from_snapshot(snapshot, llm=None)

    assert result["resolved_target"] == "蛋炒饭"
    assert result["target_source"] == "implicit_single_dish_followup"
```

- [x] **Step 2: Run tests and confirm they fail**

Run:

```powershell
C:\Users\lenovo\anaconda3\python.exe -m pytest code/C8/tests/test_reference_resolution.py -q
```

Expected: FAIL because resolver lacks these branches.

- [x] **Step 3: Add helper functions in `reference_resolution.py`**

```python
def _find_recommendation_by_rank(recommendations: list[dict], rank: int) -> str | None:
    for item in recommendations:
        if item.get("rank") == rank:
            return item.get("dish_name")
    return None


def _resolved(target: str, source: str, reason: str, decision_basis: str = "explicit") -> dict:
    return {
        "resolution_status": "resolved",
        "resolved_target": target,
        "target_source": source,
        "confidence": 1.0,
        "reason": reason,
        "next_action": "apply_reference_resolution",
        "clarification_question": None,
        "writeback_eligible": True,
        "decision_basis": decision_basis,
    }


def _clarify(reason: str, question: str) -> dict:
    return {
        "resolution_status": "ambiguous",
        "resolved_target": None,
        "target_source": None,
        "confidence": 0.0,
        "reason": reason,
        "next_action": "ask_clarification",
        "clarification_question": question,
        "writeback_eligible": False,
        "decision_basis": "ambiguous",
    }
```

- [x] **Step 4: Add new resolution branches before pronoun ambiguity**

Inside `resolve_reference_from_snapshot()`, after correction explicit target handling and before pronoun ambiguity:

```python
    cleaned_explicit_dish = constraints.get("cleaned_explicit_dish")
    if cleaned_explicit_dish and cleaned_explicit_dish.get("value"):
        return _resolved(
            cleaned_explicit_dish["value"],
            "cleaned_explicit_dish",
            "explicit_dish_with_discourse_prefix",
            decision_basis="explicit",
        )

    ordinal_reference = constraints.get("ordinal_reference")
    if ordinal_reference:
        rank = ordinal_reference["rank"]
        target = _find_recommendation_by_rank(
            snapshot["reference_state"].get("recent_recommendations", []),
            rank,
        )
        if not target:
            return _clarify(
                "ordinal_rank_out_of_range",
                "我没找到你说的这个序号。你可以说第几个，或者直接告诉我菜名。",
            )
        return _resolved(
            target,
            "ordinal_recommendation_reference",
            "user_selected_recommendation_by_rank",
            decision_basis="explicit",
        )

    implicit_followup = constraints.get("implicit_followup") or {}
    current_dish = snapshot["reference_state"].get("current_dish") or {}
    if implicit_followup.get("enabled"):
        if snapshot["topic_state"].get("mode") == "recommendation_list":
            return _clarify(
                "implicit_followup_in_recommendation_list",
                "你是想问第几个推荐菜？也可以直接告诉我菜名。",
            )
        if current_dish.get("active") and current_dish.get("value"):
            return _resolved(
                current_dish["value"],
                "implicit_single_dish_followup",
                "single_active_dish_followup",
                decision_basis="inferred",
            )
```

- [x] **Step 5: Update guard to allow cleaned explicit dish**

In `guard_resolution_output()`, before checking `allowed_targets`, add:

```python
    if result.get("target_source") == "cleaned_explicit_dish" and result.get("resolved_target"):
        return result
```

- [x] **Step 6: Run tests and confirm they pass**

Run:

```powershell
C:\Users\lenovo\anaconda3\python.exe -m pytest code/C8/tests/test_reference_resolution.py -q
```

Expected: PASS.

---

## Task 4: Rewrite Resolved Natural Follow-Ups Into Concrete Retrieval Queries

**Files:**
- Modify: `code/C8/rag_modules/reference_resolution.py`
- Modify: `code/C8/rag_modules/execution_planner.py`
- Test: `code/C8/tests/test_reference_resolution.py`
- Test: `code/C8/tests/test_conversation_state.py`

- [x] **Step 1: Write failing rewrite tests**

Append these tests to `code/C8/tests/test_reference_resolution.py`:

```python
def test_ordinal_resolution_rewrites_to_target_plus_remaining_query():
    execution_plan = {"action": "apply_reference_resolution"}
    resolution = {
        "resolved_target": "麻婆豆腐",
        "target_source": "ordinal_recommendation_reference",
    }
    query_plan = {
        "route_type": "detail",
        "filters": {"content_type": "steps"},
        "content_type": "steps",
    }

    rewritten = rewrite_query_for_execution("第二个怎么做？", execution_plan, resolution, query_plan)

    assert rewritten == "麻婆豆腐怎么做"


def test_cleaned_dish_resolution_rewrites_without_discourse_prefix():
    execution_plan = {"action": "apply_reference_resolution"}
    resolution = {
        "resolved_target": "蛋炒饭",
        "target_source": "cleaned_explicit_dish",
    }
    query_plan = {
        "route_type": "detail",
        "filters": {"content_type": "ingredients"},
        "content_type": "ingredients",
    }

    rewritten = rewrite_query_for_execution("那蛋炒饭需要哪些食材？", execution_plan, resolution, query_plan)

    assert rewritten == "蛋炒饭需要哪些食材"


def test_implicit_followup_rewrites_to_current_dish_query():
    execution_plan = {"action": "apply_reference_resolution"}
    resolution = {
        "resolved_target": "蛋炒饭",
        "target_source": "implicit_single_dish_followup",
    }
    query_plan = {
        "route_type": "detail",
        "filters": {"content_type": "tips"},
        "content_type": "tips",
    }

    rewritten = rewrite_query_for_execution("有什么小技巧别粘锅？", execution_plan, resolution, query_plan)

    assert rewritten == "蛋炒饭有什么小技巧别粘锅"
```

- [x] **Step 2: Run tests and confirm they fail**

Run:

```powershell
C:\Users\lenovo\anaconda3\python.exe -m pytest code/C8/tests/test_reference_resolution.py -q
```

Expected: FAIL because only `apply_correction` rewrites today.

- [x] **Step 3: Extend execution planner**

In `code/C8/rag_modules/execution_planner.py`, add this branch before recommendation query handling:

```python
    if resolution and resolution.get("next_action") == "apply_reference_resolution":
        return {"action": "apply_reference_resolution", "message": None}
```

- [x] **Step 4: Add rewrite helpers**

In `code/C8/rag_modules/reference_resolution.py`, add:

```python
def _strip_question_punctuation(text: str) -> str:
    return text.strip().rstrip("?!？！。")


def _remove_ordinal_text(original_question: str) -> str:
    text = _strip_question_punctuation(original_question)
    text = re.sub(r"^(第\s*[一二三四五]\s*个|第\s*[1-5]\s*个|[1-5]\s*号?)", "", text)
    for pattern in ("看起来不错", "看起来挺好", "不错", "挺好", "可以"):
        text = text.replace(pattern, "")
    text = text.strip("，,。 ")
    return text or "怎么做"


def _remove_discourse_prefix_and_target(original_question: str, target: str) -> str:
    text = _strip_question_punctuation(original_question)
    prefixes = ("刚才那个", "刚才这道", "这个", "这道", "那")
    for prefix in prefixes:
        if text.startswith(prefix + target):
            return text[len(prefix + target):].strip("，,。 ") or "怎么做"
    if text.startswith(target):
        return text[len(target):].strip("，,。 ") or "怎么做"
    return text
```

- [x] **Step 5: Extend `rewrite_query_for_execution()`**

Modify `rewrite_query_for_execution()`:

```python
    if execution_plan["action"] == "apply_reference_resolution" and resolution and resolution.get("resolved_target"):
        target = resolution["resolved_target"]
        source = resolution.get("target_source")
        if source == "ordinal_recommendation_reference":
            return f"{target}{_remove_ordinal_text(original_question)}"
        if source == "cleaned_explicit_dish":
            return f"{target}{_remove_discourse_prefix_and_target(original_question, target)}"
        if source == "implicit_single_dish_followup":
            return f"{target}{_strip_question_punctuation(original_question)}"
```

Keep the existing `apply_correction` branch.

- [x] **Step 6: Run tests and confirm they pass**

Run:

```powershell
C:\Users\lenovo\anaconda3\python.exe -m pytest code/C8/tests/test_reference_resolution.py code/C8/tests/test_conversation_state.py -q
```

Expected: PASS.

---

## Task 5: Stop Query Router From Treating Reference Phrases As Dish Names

**Files:**
- Modify: `code/C8/main.py`
- Modify: `code/C8/rag_modules/generation_integration.py`
- Test: `code/C8/tests/test_conversation_state.py`

- [x] **Step 1: Write failing main-path guard tests**

Append these tests to `code/C8/tests/test_conversation_state.py`:

```python
def test_query_plan_does_not_treat_ordinal_as_dish_name():
    system = _system()
    system.generation_module.query_router = lambda query: {
        "type": "detail",
        "filters": {"content_type": "steps"},
        "dish_name": "第二个",
        "confidence": 0.95,
    }

    plan = system._build_query_plan("第二个怎么做？", "ordinal-plan-session")

    assert plan["dish_name"] is None
    assert plan["entities"]["dish_name"] is None


def test_query_plan_does_not_treat_full_tip_question_as_dish_name():
    system = _system()
    system.generation_module.query_router = lambda query: {
        "type": "detail",
        "filters": {"content_type": "tips"},
        "dish_name": "有什么小技巧别粘锅",
        "confidence": 0.95,
    }

    plan = system._build_query_plan("有什么小技巧别粘锅？", "tip-plan-session")

    assert plan["dish_name"] is None
    assert plan["entities"]["dish_name"] is None
```

- [x] **Step 2: Run tests and confirm they fail**

Run:

```powershell
C:\Users\lenovo\anaconda3\python.exe -m pytest code/C8/tests/test_conversation_state.py -k "ordinal_as_dish_name or full_tip_question" -q
```

Expected: FAIL because query plan trusts the router-provided bad `dish_name`.

- [x] **Step 3: Add invalid dish-name filter**

In `code/C8/main.py`, add:

```python
def _is_invalid_reference_dish_name(self, dish_name: str | None) -> bool:
    if not dish_name:
        return False
    normalized = dish_name.strip()
    if re.match(r"^(第?[一二三四五1-5]个?|[1-5]号?)$", normalized):
        return True
    invalid_fragments = ("有什么", "哪些", "怎么", "为何", "为什么", "技巧", "粘锅")
    return any(fragment in normalized for fragment in invalid_fragments)
```

In `_build_query_plan()`, after reading `dish_name` from intent:

```python
        if self._is_invalid_reference_dish_name(dish_name):
            logger.info("丢弃引用短语型伪菜名: %s", dish_name)
            dish_name = None
```

- [x] **Step 4: Run tests and confirm they pass**

Run:

```powershell
C:\Users\lenovo\anaconda3\python.exe -m pytest code/C8/tests/test_conversation_state.py -k "ordinal_as_dish_name or full_tip_question" -q
```

Expected: PASS.

---

## Task 6: Add Real-Chain Regression Tests For Human-Like Follow-Ups

**Files:**
- Modify: `code/C8/tests/test_conversation_integration_real.py`

- [x] **Step 1: Add real integration tests**

Append to `code/C8/tests/test_conversation_integration_real.py`:

```python
@pytest.mark.real_integration
def test_real_ordinal_followup_uses_recommendation_rank():
    if not os.getenv("DASHSCOPE_API_KEY"):
        pytest.skip("DASHSCOPE_API_KEY is required for real integration tests")

    app = create_app()
    client = app.test_client()
    session_id = "real-ordinal-session"

    first = client.post("/api/chat", json={"question": "我晚上想吃点下饭的，有啥推荐？", "session_id": session_id})
    assert first.status_code == 200
    first_answer = first.get_json()["answer"]
    assert "2." in first_answer

    second = client.post("/api/chat", json={"question": "第二个怎么做？", "session_id": session_id})
    assert second.status_code == 200
    answer = second.get_json()["answer"]

    assert "第二个" not in answer
    assert "没有足够完整" not in answer


@pytest.mark.real_integration
def test_real_discourse_prefix_dish_name_is_cleaned():
    if not os.getenv("DASHSCOPE_API_KEY"):
        pytest.skip("DASHSCOPE_API_KEY is required for real integration tests")

    app = create_app()
    client = app.test_client()
    session_id = "real-clean-prefix-session"

    first = client.post("/api/chat", json={"question": "家里只有鸡蛋和米饭，能做什么？", "session_id": session_id})
    assert first.status_code == 200

    second = client.post("/api/chat", json={"question": "那蛋炒饭需要哪些食材？", "session_id": session_id})
    assert second.status_code == 200
    answer = second.get_json()["answer"]

    assert "那蛋炒饭" not in answer
    assert "蛋炒饭" in answer


@pytest.mark.real_integration
def test_real_single_dish_short_tip_followup_uses_current_dish():
    if not os.getenv("DASHSCOPE_API_KEY"):
        pytest.skip("DASHSCOPE_API_KEY is required for real integration tests")

    app = create_app()
    client = app.test_client()
    session_id = "real-short-followup-session"

    first = client.post("/api/chat", json={"question": "蛋炒饭怎么做？", "session_id": session_id})
    assert first.status_code == 200

    second = client.post("/api/chat", json={"question": "有什么小技巧别粘锅？", "session_id": session_id})
    assert second.status_code == 200
    answer = second.get_json()["answer"]

    assert "有什么小技巧别粘锅" not in answer
    assert "没有足够完整" not in answer
```

- [x] **Step 2: Run real tests and confirm they fail before implementation**

Run:

```powershell
$line = [string](@(Get-Content code/C8/.env | Where-Object { $_ -match 'DASHSCOPE_API_KEY=' })[0])
$env:DASHSCOPE_API_KEY = (($line -split 'DASHSCOPE_API_KEY=',2)[1]).Trim()
C:\Users\lenovo\anaconda3\python.exe -m pytest code/C8/tests/test_conversation_integration_real.py -m real_integration -q
```

Expected: At least one of the three new tests FAILS on the current behavior. If credentials are unavailable, tests SKIP and implementation must still rely on unit tests.

- [x] **Step 3: Run full targeted verification after implementation**

Run:

```powershell
C:\Users\lenovo\anaconda3\python.exe -m pytest code/C8/tests/test_turn_qualification.py code/C8/tests/test_reference_resolution.py code/C8/tests/test_conversation_state.py code/C8/tests/test_state_writeback_review.py code/C8/tests/test_web_app.py -q
$line = [string](@(Get-Content code/C8/.env | Where-Object { $_ -match 'DASHSCOPE_API_KEY=' })[0])
$env:DASHSCOPE_API_KEY = (($line -split 'DASHSCOPE_API_KEY=',2)[1]).Trim()
C:\Users\lenovo\anaconda3\python.exe -m pytest code/C8/tests/test_conversation_integration_real.py -m real_integration -q
```

Expected:
- First command PASS.
- Second command PASS, or SKIP only when credentials are unavailable.

---

## Task 7: Preference Constraint Smoke Test And Minimal Retrieval Plumbing

**Files:**
- Modify: `code/C8/main.py`
- Modify: `code/C8/rag_modules/conversation_state_builder.py`
- Test: `code/C8/tests/test_conversation_state.py`

- [x] **Step 1: Add a failing preference propagation test**

Append to `code/C8/tests/test_conversation_state.py`:

```python
def test_preference_constraints_are_available_to_query_plan():
    manager = ConversationManager()
    snapshot = build_conversation_snapshot(
        manager.get_session("preference-plan-session"),
        current_query="换个清淡一点的菜",
    )

    preferences = snapshot["resolution_constraints"]["preference_constraints"]

    assert preferences["taste"] == ["清淡"]
```

This test should pass once Task 2 is done. It exists to pin the state contract before retrieval plumbing.

- [x] **Step 2: Add minimal query-plan preference hook**

In `code/C8/main.py`, after snapshot construction and before retrieval, keep the snapshot available:

```python
preference_constraints = (
    snapshot.get("resolution_constraints", {}).get("preference_constraints", {})
    if snapshot
    else {}
)
```

When `query_plan["route_type"] == "list"` and `preference_constraints` has values, add them to query plan for diagnostics and future rerank:

```python
if query_plan["route_type"] == "list" and any(preference_constraints.values()):
    query_plan["preference_constraints"] = preference_constraints
```

Do not force metadata filters for `清淡` yet unless the knowledge base has a stable metadata field for it. This task only ensures the preference survives into the execution layer.

- [x] **Step 3: Add diagnostics assertion**

If an existing diagnostics helper exposes `query_plan`, add:

```python
def test_list_query_plan_keeps_preference_constraints():
    system = _system()
    system.generation_module.query_router = lambda query: {
        "type": "list",
        "filters": {},
        "dish_name": None,
        "confidence": 0.7,
    }

    result = system.ask_question(
        "换个清淡一点的菜",
        stream=False,
        session_id="preference-diagnostics-session",
        return_diagnostics=True,
    )

    assert result["diagnostics"]["query_plan"]["preference_constraints"]["taste"] == ["清淡"]
```

- [x] **Step 4: Run tests**

Run:

```powershell
C:\Users\lenovo\anaconda3\python.exe -m pytest code/C8/tests/test_conversation_state.py -k "preference" -q
```

Expected: PASS.

---

## Plan Review

### Coverage Review

- SPEC requirement “序号引用进入新 Reference Resolution” is covered by Tasks 1, 2, 3, 4, and 6.
- SPEC requirement “不恢复旧独立序号解析模块” is enforced by File Map and Task 3/4 architecture boundaries.
- SPEC requirement “口语前缀清洗” is covered by Tasks 2, 3, 4, and real test in Task 6.
- SPEC requirement “短追问继承当前单菜” is covered by Tasks 1, 2, 3, 4, and real test in Task 6.
- SPEC requirement “推荐列表代词仍澄清” is preserved by Task 3 ordering and existing real integration test.
- SPEC requirement “偏好约束保留” is covered by Tasks 2 and 7.

### Risk Review

- **Risk:** `remaining_query` cleanup may over-strip useful text in “第一个看起来不错，做法说一下”.
  **Mitigation:** Task 2 pins the expected remaining query as `做法说一下`.

- **Risk:** `implicit_detail_followup` may over-inherit on generic questions.
  **Mitigation:** Task 1 limits short follow-ups by length and detail keywords; Task 3 blocks implicit inheritance in recommendation-list mode.

- **Risk:** `cleaned_explicit_dish` may treat non-dish text as dish.
  **Mitigation:** Task 5 rejects obvious question fragments as dish names; future work can add knowledge-base validation if needed.

- **Risk:** Preference constraints do not yet improve retrieval quality.
  **Mitigation:** Task 7 intentionally only preserves preferences into query plan; stronger rerank is left for a separate focused plan.

### Placeholder Review

Checked for unsupported placeholders:

- No `TODO`
- No `TBD`
- No “similar to above”
- No undefined task dependency
- No instruction that says only “add tests” without concrete test code

### Interface Consistency Review

The plan consistently uses these names:

- `qualify_turn`
- `build_conversation_snapshot`
- `resolve_reference_from_snapshot`
- `guard_resolution_output`
- `rewrite_query_for_execution`
- `build_execution_plan`
- `ordinal_reference`
- `cleaned_explicit_dish`
- `implicit_followup`
- `preference_constraints`

No conflict with the existing “new framework only” boundary was found.

### Review Verdict

The plan is implementable as written and should be executed with TDD. The only intentionally deferred area is strong preference-aware reranking for “清淡/下饭” quality; this plan preserves preference state but does not claim to fully solve semantic taste ranking.

