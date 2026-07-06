# Retrieval Executor And Quality Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the chat path's ad hoc retrieval branching with a `RetrievalExecutor` that consumes a normalized retrieval `QueryPlan` and returns chunks, quality, low-evidence output, and trace.

**Architecture:** Add `rag_modules/retrieval_executor.py` as the retrieval orchestration boundary. `main.py` may still build the upstream query plan, but it must normalize that plan before retrieval and call only `RetrievalExecutor.execute()` for chat retrieval.

**Tech Stack:** Python, pytest, LangChain `Document`, existing `RetrievalOptimizationModule`, existing `RecipeRAGSystem.ask_question()`.

---

## Cutover Contract

This plan assumes Stage 02 context-first turn pipeline is implemented.

Old responsibility being replaced:

- `RecipeRAGSystem._search_relevant_chunks()` chooses low-level retrieval strategy, merges filters, forces dish filters, and owns fallback relaxation.
- The chat path in `ask_question()` treats `not relevant_chunks` as the only evidence-quality signal.

New ownership:

- `build_retrieval_query_plan(...)` normalizes the existing runtime query plan into a retrieval-facing contract.
- `RetrievalExecutor.execute(query_plan)` is the only chat-path retrieval entrypoint.
- `RetrievalExecutor` owns primary retrieval, evidence quality, fallback policy, fallback markers, low-evidence result, and retrieval trace.

Production cutover:

```text
ask_question()
  old: _search_relevant_chunks(...) -> if not relevant_chunks -> no-result answer
  new: build_retrieval_query_plan(...) -> RetrievalExecutor.execute(...) -> RetrievalResult
```

Illegal after cutover:

- `ask_question()` calling `_search_relevant_chunks()`.
- `ask_question()` choosing between `metadata_filtered_search()` and `hybrid_search()`.
- `ask_question()` relaxing `content_type` or any other retrieval filters.
- `ask_question()` using `if not relevant_chunks` as its evidence-quality decision.

Deletion before acceptance:

- Delete `RecipeRAGSystem._search_relevant_chunks()`.
- Keep direct low-level retrieval calls only in non-chat helper methods such as category or ingredient helpers if still used.
- Add source-level tests proving `ask_question()` no longer contains the old retrieval branch.

---

## File Structure

- Create `code/C8/rag_modules/retrieval_executor.py`
  - Defines `build_retrieval_query_plan(...)`.
  - Defines `RetrievalExecutor`.
  - Defines helper functions for hard/soft filter normalization, evidence quality, fallback, and low-evidence output.

- Create `code/C8/tests/test_retrieval_executor.py`
  - Unit tests for query-plan normalization, executor primary retrieval, fallback rules, quality checks, and low-evidence output.

- Modify `code/C8/main.py`
  - Imports `RetrievalExecutor` and `build_retrieval_query_plan`.
  - Instantiates `self.retrieval_executor` after `self.retrieval_module` exists.
  - Replaces chat-path retrieval with `RetrievalExecutor.execute(...)`.
  - Deletes `_search_relevant_chunks()`.

- Create `code/C8/tests/test_retrieval_executor_cutover.py`
  - Source-level tests proving `ask_question()` uses the executor and no longer owns fallback/retrieval branching.

- Modify `code/C8/tests/test_conversation_state.py`
  - Update integration tests that assert old no-chunk behavior to assert explicit low-evidence behavior when needed.

---

## Task 1: Add Retrieval QueryPlan Normalization

**Files:**
- Create: `code/C8/rag_modules/retrieval_executor.py`
- Create: `code/C8/tests/test_retrieval_executor.py`

- [ ] **Step 1: Write failing normalization tests**

Create `code/C8/tests/test_retrieval_executor.py`:

```python
from langchain_core.documents import Document

from rag_modules.retrieval_executor import build_retrieval_query_plan


def test_query_plan_normalization_prefers_resolved_target_as_hard_dish_filter():
    result = build_retrieval_query_plan(
        original_query="第一个怎么做",
        rewritten_query="宫保鸡丁 怎么做",
        base_query_plan={
            "route_type": "detail",
            "dish_name": "第一个",
            "filters": {"content_type": "steps"},
            "entities": {"dish_name": "第一个", "filters": {"content_type": "steps"}},
        },
        execution_plan={"action": "apply_reference_resolution", "answer_mode": "recipe_detail"},
        resolution={"resolved_target": "宫保鸡丁", "confidence": 0.95},
        preference_constraints=None,
        top_k=3,
    )

    assert result["query"] == "宫保鸡丁 怎么做"
    assert result["original_query"] == "第一个怎么做"
    assert result["dish_name"] == "宫保鸡丁"
    assert result["filters"]["dish_name"] == "宫保鸡丁"
    assert result["filters"]["content_type"] == "steps"
    assert result["hard_filters"] == ["dish_name"]
    assert "content_type" in result["soft_filters"]
    assert result["fallback_policy"] == "disabled"
    assert result["top_k"] == 3


def test_query_plan_normalization_uses_relaxed_filters_for_sparse_list_preferences():
    result = build_retrieval_query_plan(
        original_query="推荐几个不辣的鸡肉菜",
        rewritten_query="推荐几个不辣的鸡肉菜",
        base_query_plan={
            "route_type": "list",
            "dish_name": None,
            "filters": {},
            "entities": {"dish_name": None, "filters": {}},
            "preference_constraints": {"taste": ["不辣"], "ingredient": ["鸡肉"]},
        },
        execution_plan={"action": "retrieve_list", "answer_mode": "recommendation"},
        resolution=None,
        preference_constraints={"taste": ["不辣"], "ingredient": ["鸡肉"]},
        top_k=5,
    )

    assert result["query"] == "推荐几个不辣的鸡肉菜"
    assert result["dish_name"] is None
    assert result["hard_filters"] == []
    assert "dish_name" not in result["filters"]
    assert result["filters"]["taste"] == ["不辣"]
    assert result["filters"]["ingredient"] == ["鸡肉"]
    assert result["soft_filters"] == ["ingredient", "taste", "difficulty", "time", "health_preference"]
    assert result["fallback_policy"] == "relaxed_filters"
    assert result["answer_mode_hint"] == "recommendation"


def test_query_plan_normalization_keeps_broad_search_disabled_by_default():
    result = build_retrieval_query_plan(
        original_query="西湖醋鱼怎么做",
        rewritten_query="西湖醋鱼怎么做",
        base_query_plan={
            "route_type": "detail",
            "dish_name": "西湖醋鱼",
            "filters": {},
            "entities": {"dish_name": "西湖醋鱼", "filters": {}},
        },
        execution_plan={"action": "retrieve_detail", "answer_mode": "recipe_detail"},
        resolution=None,
        preference_constraints=None,
        top_k=3,
    )

    assert result["dish_name"] == "西湖醋鱼"
    assert result["hard_filters"] == ["dish_name"]
    assert result["fallback_policy"] == "disabled"
```

- [ ] **Step 2: Run normalization tests and verify they fail**

Run:

```bash
cd code/C8
pytest tests/test_retrieval_executor.py::test_query_plan_normalization_prefers_resolved_target_as_hard_dish_filter tests/test_retrieval_executor.py::test_query_plan_normalization_uses_relaxed_filters_for_sparse_list_preferences tests/test_retrieval_executor.py::test_query_plan_normalization_keeps_broad_search_disabled_by_default -q
```

Expected:

- FAIL because `rag_modules.retrieval_executor` does not exist.

- [ ] **Step 3: Implement query-plan normalization**

Create `code/C8/rag_modules/retrieval_executor.py`:

```python
"""Retrieval execution boundary for the runtime chat path."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

from langchain_core.documents import Document


SOFT_FILTER_KEYS = ["ingredient", "taste", "difficulty", "time", "health_preference"]


def _copy_dict(value: dict | None) -> dict:
    return dict(value or {})


def _resolved_dish(base_query_plan: dict, resolution: dict | None) -> str | None:
    if resolution:
        resolved = resolution.get("resolved_target") or resolution.get("resolved_entity")
        if resolved:
            return resolved
    return base_query_plan.get("dish_name")


def _merge_preference_constraints(filters: dict, preference_constraints: dict | None) -> dict:
    merged = dict(filters)
    for key, value in (preference_constraints or {}).items():
        if value:
            merged[key] = value
    return merged


def _answer_mode_hint(execution_plan: dict, base_query_plan: dict) -> str:
    if execution_plan.get("answer_mode"):
        return execution_plan["answer_mode"]
    action = execution_plan.get("action")
    if action == "retrieve_list" or base_query_plan.get("route_type") == "list":
        return "recommendation"
    return "recipe_detail"


def _fallback_policy(route_type: str, hard_filters: list[str], filters: dict) -> str:
    if "dish_name" in hard_filters:
        return "disabled"
    if route_type == "list" or any(key in filters for key in SOFT_FILTER_KEYS):
        return "relaxed_filters"
    return "disabled"


def build_retrieval_query_plan(
    *,
    original_query: str,
    rewritten_query: str,
    base_query_plan: dict,
    execution_plan: dict,
    resolution: dict | None,
    preference_constraints: dict | None,
    top_k: int,
) -> dict:
    """Normalize the runtime query plan into the retrieval-facing contract."""
    route_type = base_query_plan.get("route_type", "detail")
    dish_name = _resolved_dish(base_query_plan, resolution)
    filters = _merge_preference_constraints(
        _copy_dict(base_query_plan.get("filters")),
        preference_constraints,
    )

    hard_filters: list[str] = []
    if dish_name:
        filters["dish_name"] = dish_name
        hard_filters.append("dish_name")

    soft_filters = list(SOFT_FILTER_KEYS)
    if "content_type" in filters:
        soft_filters.append("content_type")

    return {
        "query": rewritten_query,
        "original_query": original_query,
        "dish_name": dish_name,
        "filters": filters,
        "top_k": top_k,
        "fallback_policy": _fallback_policy(route_type, hard_filters, filters),
        "hard_filters": hard_filters,
        "soft_filters": soft_filters,
        "answer_mode_hint": _answer_mode_hint(execution_plan, base_query_plan),
        "route_type": route_type,
    }
```

- [ ] **Step 4: Run normalization tests and verify they pass**

Run:

```bash
cd code/C8
pytest tests/test_retrieval_executor.py::test_query_plan_normalization_prefers_resolved_target_as_hard_dish_filter tests/test_retrieval_executor.py::test_query_plan_normalization_uses_relaxed_filters_for_sparse_list_preferences tests/test_retrieval_executor.py::test_query_plan_normalization_keeps_broad_search_disabled_by_default -q
```

Expected:

- PASS.

- [ ] **Step 5: Commit**

```bash
git add code/C8/rag_modules/retrieval_executor.py code/C8/tests/test_retrieval_executor.py
git commit -m "feat: normalize retrieval query plans"
```

---

## Task 2: Add Primary Retrieval Executor Result Contract

**Files:**
- Modify: `code/C8/rag_modules/retrieval_executor.py`
- Modify: `code/C8/tests/test_retrieval_executor.py`

- [ ] **Step 1: Add primary retrieval result tests**

Append to `code/C8/tests/test_retrieval_executor.py`:

```python
from rag_modules.retrieval_executor import RetrievalExecutor


class FakeRetrievalModule:
    def __init__(self, filtered_docs=None, hybrid_docs=None, extracted_filters=None):
        self.filtered_docs = filtered_docs or []
        self.hybrid_docs = hybrid_docs or []
        self.extracted_filters = extracted_filters or {}
        self.calls = []

    def extract_filters_from_query(self, query):
        self.calls.append(("extract_filters_from_query", query))
        return dict(self.extracted_filters)

    def metadata_filtered_search(self, query, filters, top_k=5, query_dish=None):
        self.calls.append(("metadata_filtered_search", query, dict(filters), top_k, query_dish))
        return list(self.filtered_docs[:top_k])

    def hybrid_search(self, query, top_k=3, query_dish=None):
        self.calls.append(("hybrid_search", query, top_k, query_dish))
        return list(self.hybrid_docs[:top_k])


def _doc(dish_name, content_type="steps", content="content"):
    return Document(page_content=content, metadata={"dish_name": dish_name, "content_type": content_type})


def test_executor_uses_filtered_primary_retrieval_when_filters_exist():
    docs = [_doc("宫保鸡丁", "steps", "宫保鸡丁步骤")]
    retrieval_module = FakeRetrievalModule(filtered_docs=docs)
    executor = RetrievalExecutor(retrieval_module)

    result = executor.execute(
        {
            "query": "宫保鸡丁 怎么做",
            "original_query": "第一个怎么做",
            "dish_name": "宫保鸡丁",
            "filters": {"dish_name": "宫保鸡丁", "content_type": "steps"},
            "top_k": 3,
            "fallback_policy": "disabled",
            "hard_filters": ["dish_name"],
            "soft_filters": ["content_type"],
            "answer_mode_hint": "recipe_detail",
        }
    )

    assert result["chunks"] == docs
    assert result["quality"]["enough_evidence"] is True
    assert result["quality"]["candidate_count"] == 1
    assert result["quality"]["selected_dishes"] == ["宫保鸡丁"]
    assert result["quality"]["fallback_used"] is False
    assert result["quality"]["relaxed_filter"] is False
    assert result["low_evidence"] is None
    assert result["trace"]["strategy"] == "primary"
    assert result["trace"]["fusion_strategy"] == "delegated"
    assert retrieval_module.calls[0][0] == "metadata_filtered_search"


def test_executor_uses_hybrid_primary_retrieval_without_filters():
    docs = [_doc("番茄炒蛋", "steps", "番茄炒蛋步骤")]
    retrieval_module = FakeRetrievalModule(hybrid_docs=docs)
    executor = RetrievalExecutor(retrieval_module)

    result = executor.execute(
        {
            "query": "今天吃什么",
            "original_query": "今天吃什么",
            "dish_name": None,
            "filters": {},
            "top_k": 3,
            "fallback_policy": "relaxed_filters",
            "hard_filters": [],
            "soft_filters": [],
            "answer_mode_hint": "recommendation",
        }
    )

    assert result["chunks"] == docs
    assert result["quality"]["enough_evidence"] is True
    assert result["trace"]["strategy"] == "primary"
    assert retrieval_module.calls[0][0] == "hybrid_search"
```

- [ ] **Step 2: Run primary retrieval tests and verify they fail**

Run:

```bash
cd code/C8
pytest tests/test_retrieval_executor.py::test_executor_uses_filtered_primary_retrieval_when_filters_exist tests/test_retrieval_executor.py::test_executor_uses_hybrid_primary_retrieval_without_filters -q
```

Expected:

- FAIL because `RetrievalExecutor` does not exist yet.

- [ ] **Step 3: Add `RetrievalExecutor` primary execution**

Append this implementation to `code/C8/rag_modules/retrieval_executor.py`:

```python
class RetrievalExecutor:
    """Execute retrieval and return chunks plus explicit evidence quality."""

    def __init__(self, retrieval_module):
        self.retrieval_module = retrieval_module

    def execute(self, query_plan: dict) -> dict:
        primary_chunks = self._primary_retrieval(query_plan)
        quality = self._check_quality(query_plan, primary_chunks, fallback_used=False, relaxed_filter=False)
        trace = self._build_trace(
            query_plan=query_plan,
            strategy="primary",
            primary_count=len(primary_chunks),
            fallback_count=0,
            quality=quality,
        )

        return {
            "chunks": primary_chunks if quality["enough_evidence"] else [],
            "quality": quality,
            "low_evidence": None if quality["enough_evidence"] else self._low_evidence(quality["quality_reason"]),
            "trace": trace,
        }

    def _primary_retrieval(self, query_plan: dict) -> list[Document]:
        query = query_plan["query"]
        filters = dict(query_plan.get("filters") or {})
        top_k = query_plan.get("top_k", 3)
        dish_name = query_plan.get("dish_name")
        if filters:
            return list(
                self.retrieval_module.metadata_filtered_search(
                    query,
                    filters,
                    top_k=top_k,
                    query_dish=dish_name,
                )
            )
        return list(
            self.retrieval_module.hybrid_search(
                query,
                top_k=top_k,
                query_dish=dish_name,
            )
        )

    def _selected_dishes(self, chunks: Iterable[Document]) -> list[str]:
        dishes: list[str] = []
        for chunk in chunks:
            dish_name = (chunk.metadata or {}).get("dish_name")
            if dish_name and dish_name not in dishes:
                dishes.append(dish_name)
        return dishes

    def _check_quality(
        self,
        query_plan: dict,
        chunks: list[Document],
        *,
        fallback_used: bool,
        relaxed_filter: bool,
    ) -> dict:
        selected_dishes = self._selected_dishes(chunks)
        dish_name = query_plan.get("dish_name")
        hard_filters = set(query_plan.get("hard_filters") or [])

        enough = bool(chunks)
        reason = "primary_candidates_found" if enough else "no_candidates"

        if enough and dish_name and "dish_name" in hard_filters:
            if dish_name not in selected_dishes:
                enough = False
                reason = "exact_dish_not_found"
            elif len(selected_dishes) > 1:
                enough = False
                reason = "conflicting_dishes_for_exact_request"
            else:
                reason = "exact_dish_matched"

        return {
            "enough_evidence": enough,
            "quality_reason": reason,
            "fallback_used": fallback_used,
            "relaxed_filter": relaxed_filter,
            "candidate_count": len(chunks),
            "selected_dishes": selected_dishes,
        }

    def _low_evidence(self, quality_reason: str) -> dict:
        return {
            "answer_type": "no_result",
            "answer": "知识库里没有找到可靠的食谱信息。",
            "state_diff_policy": "low_evidence",
            "quality_reason": quality_reason,
        }

    def _build_trace(
        self,
        *,
        query_plan: dict,
        strategy: str,
        primary_count: int,
        fallback_count: int,
        quality: dict,
    ) -> dict:
        return {
            "strategy": strategy,
            "fusion_strategy": "delegated",
            "query": query_plan.get("query"),
            "original_query": query_plan.get("original_query"),
            "filters": dict(query_plan.get("filters") or {}),
            "hard_filters": list(query_plan.get("hard_filters") or []),
            "soft_filters": list(query_plan.get("soft_filters") or []),
            "fallback_policy": query_plan.get("fallback_policy", "disabled"),
            "primary_count": primary_count,
            "fallback_count": fallback_count,
            "selected_dishes": list(quality.get("selected_dishes") or []),
            "quality_reason": quality.get("quality_reason"),
        }
```

- [ ] **Step 4: Run primary retrieval tests and verify they pass**

Run:

```bash
cd code/C8
pytest tests/test_retrieval_executor.py::test_executor_uses_filtered_primary_retrieval_when_filters_exist tests/test_retrieval_executor.py::test_executor_uses_hybrid_primary_retrieval_without_filters -q
```

Expected:

- PASS.

- [ ] **Step 5: Commit**

```bash
git add code/C8/rag_modules/retrieval_executor.py code/C8/tests/test_retrieval_executor.py
git commit -m "feat: add retrieval executor primary path"
```

---

## Task 3: Add Evidence Quality And Low-Evidence Results

**Files:**
- Modify: `code/C8/rag_modules/retrieval_executor.py`
- Modify: `code/C8/tests/test_retrieval_executor.py`

- [ ] **Step 1: Add evidence quality and low-evidence tests**

Append to `code/C8/tests/test_retrieval_executor.py`:

```python
def test_executor_rejects_different_dish_for_hard_exact_dish_request():
    retrieval_module = FakeRetrievalModule(filtered_docs=[_doc("鱼香肉丝", "steps", "鱼香肉丝步骤")])
    executor = RetrievalExecutor(retrieval_module)

    result = executor.execute(
        {
            "query": "西湖醋鱼 怎么做",
            "original_query": "西湖醋鱼怎么做",
            "dish_name": "西湖醋鱼",
            "filters": {"dish_name": "西湖醋鱼"},
            "top_k": 3,
            "fallback_policy": "disabled",
            "hard_filters": ["dish_name"],
            "soft_filters": [],
            "answer_mode_hint": "recipe_detail",
        }
    )

    assert result["chunks"] == []
    assert result["quality"]["enough_evidence"] is False
    assert result["quality"]["quality_reason"] == "exact_dish_not_found"
    assert result["low_evidence"] == {
        "answer_type": "no_result",
        "answer": "知识库里没有找到可靠的食谱信息。",
        "state_diff_policy": "low_evidence",
        "quality_reason": "exact_dish_not_found",
    }


def test_executor_rejects_conflicting_dishes_for_hard_exact_dish_request():
    retrieval_module = FakeRetrievalModule(
        filtered_docs=[
            _doc("宫保鸡丁", "steps", "宫保鸡丁步骤"),
            _doc("鱼香肉丝", "steps", "鱼香肉丝步骤"),
        ]
    )
    executor = RetrievalExecutor(retrieval_module)

    result = executor.execute(
        {
            "query": "宫保鸡丁 怎么做",
            "original_query": "宫保鸡丁怎么做",
            "dish_name": "宫保鸡丁",
            "filters": {"dish_name": "宫保鸡丁"},
            "top_k": 3,
            "fallback_policy": "disabled",
            "hard_filters": ["dish_name"],
            "soft_filters": [],
            "answer_mode_hint": "recipe_detail",
        }
    )

    assert result["quality"]["enough_evidence"] is False
    assert result["quality"]["quality_reason"] == "conflicting_dishes_for_exact_request"
    assert result["low_evidence"]["answer_type"] == "no_result"


def test_executor_returns_low_evidence_when_primary_has_no_candidates():
    retrieval_module = FakeRetrievalModule(filtered_docs=[])
    executor = RetrievalExecutor(retrieval_module)

    result = executor.execute(
        {
            "query": "不存在的菜 怎么做",
            "original_query": "不存在的菜怎么做",
            "dish_name": "不存在的菜",
            "filters": {"dish_name": "不存在的菜"},
            "top_k": 3,
            "fallback_policy": "disabled",
            "hard_filters": ["dish_name"],
            "soft_filters": [],
            "answer_mode_hint": "recipe_detail",
        }
    )

    assert result["chunks"] == []
    assert result["quality"]["quality_reason"] == "no_candidates"
    assert result["low_evidence"]["state_diff_policy"] == "low_evidence"
```

- [ ] **Step 2: Run evidence tests and verify current behavior**

Run:

```bash
cd code/C8
pytest tests/test_retrieval_executor.py::test_executor_rejects_different_dish_for_hard_exact_dish_request tests/test_retrieval_executor.py::test_executor_rejects_conflicting_dishes_for_hard_exact_dish_request tests/test_retrieval_executor.py::test_executor_returns_low_evidence_when_primary_has_no_candidates -q
```

Expected:

- PASS if Task 2 quality helper already covers these cases.
- If a test fails, update only `_check_quality()` or `_low_evidence()` in `code/C8/rag_modules/retrieval_executor.py` so the expected quality reason and low-evidence shape match the tests.

- [ ] **Step 3: Run the full retrieval executor unit tests**

Run:

```bash
cd code/C8
pytest tests/test_retrieval_executor.py -q
```

Expected:

- PASS.

- [ ] **Step 4: Commit**

```bash
git add code/C8/rag_modules/retrieval_executor.py code/C8/tests/test_retrieval_executor.py
git commit -m "feat: add retrieval evidence quality results"
```

---

## Task 4: Add Controlled Fallback

**Files:**
- Modify: `code/C8/rag_modules/retrieval_executor.py`
- Modify: `code/C8/tests/test_retrieval_executor.py`

- [ ] **Step 1: Add fallback policy tests**

Append to `code/C8/tests/test_retrieval_executor.py`:

```python
def test_fallback_does_not_run_when_policy_disabled():
    retrieval_module = FakeRetrievalModule(filtered_docs=[], hybrid_docs=[_doc("鱼香肉丝")])
    executor = RetrievalExecutor(retrieval_module)

    result = executor.execute(
        {
            "query": "西湖醋鱼 怎么做",
            "original_query": "西湖醋鱼怎么做",
            "dish_name": "西湖醋鱼",
            "filters": {"dish_name": "西湖醋鱼", "content_type": "steps"},
            "top_k": 3,
            "fallback_policy": "disabled",
            "hard_filters": ["dish_name"],
            "soft_filters": ["content_type"],
            "answer_mode_hint": "recipe_detail",
        }
    )

    assert result["chunks"] == []
    assert result["quality"]["fallback_used"] is False
    assert result["trace"]["fallback_count"] == 0
    assert [call[0] for call in retrieval_module.calls].count("metadata_filtered_search") == 1


def test_relaxed_filter_fallback_keeps_hard_dish_filter_and_marks_docs():
    fallback_doc = _doc("宫保鸡丁", "introduction", "宫保鸡丁介绍")
    retrieval_module = FakeRetrievalModule(filtered_docs=[])

    def metadata_filtered_search(query, filters, top_k=5, query_dish=None):
        retrieval_module.calls.append(("metadata_filtered_search", query, dict(filters), top_k, query_dish))
        if filters == {"dish_name": "宫保鸡丁"}:
            return [fallback_doc]
        return []

    retrieval_module.metadata_filtered_search = metadata_filtered_search
    executor = RetrievalExecutor(retrieval_module)

    result = executor.execute(
        {
            "query": "宫保鸡丁 技巧",
            "original_query": "宫保鸡丁有什么技巧",
            "dish_name": "宫保鸡丁",
            "filters": {"dish_name": "宫保鸡丁", "content_type": "tips"},
            "top_k": 3,
            "fallback_policy": "relaxed_filters",
            "hard_filters": ["dish_name"],
            "soft_filters": ["content_type"],
            "answer_mode_hint": "recipe_detail",
        }
    )

    assert result["chunks"] == [fallback_doc]
    assert result["quality"]["enough_evidence"] is True
    assert result["quality"]["fallback_used"] is True
    assert result["quality"]["relaxed_filter"] is True
    assert fallback_doc.metadata["fallback"] is True
    assert fallback_doc.metadata["relaxed_filter"] is True
    assert result["trace"]["fallback_count"] == 1
    assert retrieval_module.calls[-1][2] == {"dish_name": "宫保鸡丁"}


def test_broad_search_fallback_rejected_for_hard_exact_dish_request():
    retrieval_module = FakeRetrievalModule(filtered_docs=[], hybrid_docs=[_doc("鱼香肉丝")])
    executor = RetrievalExecutor(retrieval_module)

    result = executor.execute(
        {
            "query": "西湖醋鱼 怎么做",
            "original_query": "西湖醋鱼怎么做",
            "dish_name": "西湖醋鱼",
            "filters": {"dish_name": "西湖醋鱼"},
            "top_k": 3,
            "fallback_policy": "broad_search",
            "hard_filters": ["dish_name"],
            "soft_filters": [],
            "answer_mode_hint": "recipe_detail",
        }
    )

    assert result["chunks"] == []
    assert result["quality"]["enough_evidence"] is False
    assert result["quality"]["fallback_used"] is False
    assert result["low_evidence"]["answer_type"] == "no_result"
    assert all(call[0] != "hybrid_search" for call in retrieval_module.calls)
```

- [ ] **Step 2: Run fallback tests and verify they fail**

Run:

```bash
cd code/C8
pytest tests/test_retrieval_executor.py::test_fallback_does_not_run_when_policy_disabled tests/test_retrieval_executor.py::test_relaxed_filter_fallback_keeps_hard_dish_filter_and_marks_docs tests/test_retrieval_executor.py::test_broad_search_fallback_rejected_for_hard_exact_dish_request -q
```

Expected:

- At least `test_relaxed_filter_fallback_keeps_hard_dish_filter_and_marks_docs` FAILS because fallback is not implemented.

- [ ] **Step 3: Implement controlled fallback**

In `code/C8/rag_modules/retrieval_executor.py`, replace `execute()` with:

```python
    def execute(self, query_plan: dict) -> dict:
        primary_chunks = self._primary_retrieval(query_plan)
        primary_quality = self._check_quality(
            query_plan,
            primary_chunks,
            fallback_used=False,
            relaxed_filter=False,
        )

        if primary_quality["enough_evidence"]:
            return {
                "chunks": primary_chunks,
                "quality": primary_quality,
                "low_evidence": None,
                "trace": self._build_trace(
                    query_plan=query_plan,
                    strategy="primary",
                    primary_count=len(primary_chunks),
                    fallback_count=0,
                    quality=primary_quality,
                ),
            }

        fallback_chunks = self._fallback_retrieval(query_plan)
        if fallback_chunks:
            fallback_quality = self._check_quality(
                query_plan,
                fallback_chunks,
                fallback_used=True,
                relaxed_filter=True,
            )
            if fallback_quality["enough_evidence"]:
                return {
                    "chunks": fallback_chunks,
                    "quality": fallback_quality,
                    "low_evidence": None,
                    "trace": self._build_trace(
                        query_plan=query_plan,
                        strategy="fallback",
                        primary_count=len(primary_chunks),
                        fallback_count=len(fallback_chunks),
                        quality=fallback_quality,
                    ),
                }

        return {
            "chunks": [],
            "quality": primary_quality,
            "low_evidence": self._low_evidence(primary_quality["quality_reason"]),
            "trace": self._build_trace(
                query_plan=query_plan,
                strategy="low_evidence",
                primary_count=len(primary_chunks),
                fallback_count=len(fallback_chunks),
                quality=primary_quality,
            ),
        }
```

Then append these methods to the `RetrievalExecutor` class:

```python
    def _fallback_retrieval(self, query_plan: dict) -> list[Document]:
        policy = query_plan.get("fallback_policy", "disabled")
        if policy == "disabled":
            return []

        hard_filters = set(query_plan.get("hard_filters") or [])
        if policy == "broad_search" and "dish_name" in hard_filters:
            return []

        if policy == "relaxed_filters":
            relaxed_filters = self._relaxed_filters(query_plan)
            if not relaxed_filters and query_plan.get("filters"):
                return []
            chunks = list(
                self.retrieval_module.metadata_filtered_search(
                    query_plan["query"],
                    relaxed_filters,
                    top_k=query_plan.get("top_k", 3),
                    query_dish=query_plan.get("dish_name"),
                )
            )
            return self._mark_fallback(chunks)

        if policy == "broad_search":
            chunks = list(
                self.retrieval_module.hybrid_search(
                    query_plan["query"],
                    top_k=query_plan.get("top_k", 3),
                    query_dish=query_plan.get("dish_name"),
                )
            )
            return self._mark_fallback(chunks)

        return []

    def _relaxed_filters(self, query_plan: dict) -> dict:
        filters = dict(query_plan.get("filters") or {})
        hard_filters = set(query_plan.get("hard_filters") or [])
        return {key: value for key, value in filters.items() if key in hard_filters}

    def _mark_fallback(self, chunks: list[Document]) -> list[Document]:
        for chunk in chunks:
            chunk.metadata["fallback"] = True
            chunk.metadata["relaxed_filter"] = True
        return chunks
```

- [ ] **Step 4: Run fallback tests and verify they pass**

Run:

```bash
cd code/C8
pytest tests/test_retrieval_executor.py::test_fallback_does_not_run_when_policy_disabled tests/test_retrieval_executor.py::test_relaxed_filter_fallback_keeps_hard_dish_filter_and_marks_docs tests/test_retrieval_executor.py::test_broad_search_fallback_rejected_for_hard_exact_dish_request -q
```

Expected:

- PASS.

- [ ] **Step 5: Run all retrieval executor tests**

Run:

```bash
cd code/C8
pytest tests/test_retrieval_executor.py -q
```

Expected:

- PASS.

- [ ] **Step 6: Commit**

```bash
git add code/C8/rag_modules/retrieval_executor.py code/C8/tests/test_retrieval_executor.py
git commit -m "feat: add controlled retrieval fallback"
```

---

## Task 5: Wire The Executor Into `RecipeRAGSystem`

**Files:**
- Modify: `code/C8/main.py`
- Modify: `code/C8/tests/test_conversation_state.py`

- [ ] **Step 1: Add chat-path integration tests**

Append to `code/C8/tests/test_conversation_state.py`:

```python
def test_chat_path_uses_retrieval_executor_result(monkeypatch):
    from main import RecipeRAGSystem
    from langchain_core.documents import Document

    system = RecipeRAGSystem.__new__(RecipeRAGSystem)
    calls = []
    doc = Document(page_content="蛋炒饭步骤", metadata={"dish_name": "蛋炒饭", "content_type": "steps"})

    class FakeGeneration:
        conversation_manager = None

        def query_router(self, question):
            return {
                "type": "detail",
                "filters": {"content_type": "steps"},
                "dish_name": "蛋炒饭",
                "confidence": 1.0,
            }

    class FakeExecutor:
        def execute(self, query_plan):
            calls.append(query_plan)
            return {
                "chunks": [doc],
                "quality": {
                    "enough_evidence": True,
                    "quality_reason": "exact_dish_matched",
                    "fallback_used": False,
                    "relaxed_filter": False,
                    "candidate_count": 1,
                    "selected_dishes": ["蛋炒饭"],
                },
                "low_evidence": None,
                "trace": {"strategy": "primary"},
            }

    system.retrieval_module = object()
    system.retrieval_executor = FakeExecutor()
    system.generation_module = FakeGeneration()
    system.config = type("Config", (), {"top_k": 3})()
    system._latest_parent_docs = []
    system.last_execution_result = None

    monkeypatch.setattr(system, "_apply_resolved_target_to_query_plan", lambda query_plan, resolution: query_plan)
    monkeypatch.setattr(system, "_generate_detail_response", lambda *args, **kwargs: "蛋炒饭做法")
    monkeypatch.setattr(system, "_write_conversation_turn", lambda **kwargs: None)

    answer = system.ask_question("蛋炒饭怎么做", stream=False, session_id="executor-chat")

    assert answer == "蛋炒饭做法"
    assert calls
    assert calls[0]["query"] == "蛋炒饭怎么做"
    assert calls[0]["dish_name"] == "蛋炒饭"
    assert calls[0]["filters"]["content_type"] == "steps"


def test_chat_path_returns_low_evidence_without_generation(monkeypatch):
    from main import RecipeRAGSystem

    system = RecipeRAGSystem.__new__(RecipeRAGSystem)
    writes = []
    generation_calls = []

    class FakeGeneration:
        conversation_manager = None

        def query_router(self, question):
            return {
                "type": "detail",
                "filters": {},
                "dish_name": "西湖醋鱼",
                "confidence": 1.0,
            }

    class FakeExecutor:
        def execute(self, query_plan):
            return {
                "chunks": [],
                "quality": {
                    "enough_evidence": False,
                    "quality_reason": "exact_dish_not_found",
                    "fallback_used": False,
                    "relaxed_filter": False,
                    "candidate_count": 0,
                    "selected_dishes": [],
                },
                "low_evidence": {
                    "answer_type": "no_result",
                    "answer": "知识库里没有找到这道菜的可靠做法。",
                    "state_diff_policy": "low_evidence",
                    "quality_reason": "exact_dish_not_found",
                },
                "trace": {"strategy": "low_evidence"},
            }

    system.retrieval_module = object()
    system.retrieval_executor = FakeExecutor()
    system.generation_module = FakeGeneration()
    system.config = type("Config", (), {"top_k": 3})()
    system._latest_parent_docs = []
    system.last_execution_result = None

    monkeypatch.setattr(system, "_apply_resolved_target_to_query_plan", lambda query_plan, resolution: query_plan)
    monkeypatch.setattr(system, "_generate_detail_response", lambda *args, **kwargs: generation_calls.append(args) or "should not happen")
    monkeypatch.setattr(system, "_write_conversation_turn", lambda **kwargs: writes.append(kwargs))

    answer = system.ask_question("西湖醋鱼怎么做", stream=False, session_id="low-evidence-chat")

    assert answer == "知识库里没有找到这道菜的可靠做法。"
    assert generation_calls == []
    assert writes[-1]["execution_result"]["answer_type"] == "no_result"
    assert writes[-1]["execution_result"]["state_diff_policy"] == "low_evidence"
    assert writes[-1]["execution_result"]["retrieval_quality"]["quality_reason"] == "exact_dish_not_found"
```

- [ ] **Step 2: Run chat-path tests and verify they fail**

Run:

```bash
cd code/C8
pytest tests/test_conversation_state.py::test_chat_path_uses_retrieval_executor_result tests/test_conversation_state.py::test_chat_path_returns_low_evidence_without_generation -q
```

Expected:

- FAIL because `ask_question()` still calls `_search_relevant_chunks()` and does not consume `RetrievalResult`.

- [ ] **Step 3: Import executor APIs in `main.py`**

In `code/C8/main.py`, add:

```python
from rag_modules.retrieval_executor import RetrievalExecutor, build_retrieval_query_plan
```

- [ ] **Step 4: Instantiate the executor after retrieval module creation**

In the knowledge-base setup path where `self.retrieval_module = RetrievalOptimizationModule(vectorstore, chunks)` is assigned, add:

```python
self.retrieval_executor = RetrievalExecutor(self.retrieval_module)
```

Also make `ask_question()` robust for tests that set only `retrieval_module`:

```python
if not hasattr(self, "retrieval_executor") or self.retrieval_executor is None:
    self.retrieval_executor = RetrievalExecutor(self.retrieval_module)
```

Place that guard after the existing `if not all([self.retrieval_module, self.generation_module]):` check.

- [ ] **Step 5: Replace chat-path retrieval block**

In `RecipeRAGSystem.ask_question()`, replace:

```python
route_type = query_plan["route_type"]
filters = query_plan["filters"]
dish_name = query_plan["dish_name"]
entities = query_plan["entities"]
rewritten_query = self._rewrite_question_for_search(rewritten_question, route_type)
relevant_chunks = self._search_relevant_chunks(
    rewritten_question,
    rewritten_query,
    filters,
    dish_name,
)
self._print_relevant_chunk_summary(relevant_chunks)

if not relevant_chunks:
    ...
    return answer
```

with:

```python
route_type = query_plan["route_type"]
filters = query_plan["filters"]
dish_name = query_plan["dish_name"]
entities = query_plan["entities"]
rewritten_query = self._rewrite_question_for_search(rewritten_question, route_type)

# Preserve legacy query-text filter extraction (category, difficulty, ingredient)
extracted_filters = self.retrieval_module.extract_filters_from_query(question)
for key, value in extracted_filters.items():
    if key not in query_plan["filters"]:
        query_plan["filters"][key] = value

retrieval_query_plan = build_retrieval_query_plan(
    original_query=question,
    rewritten_query=rewritten_query,
    base_query_plan=query_plan,
    execution_plan=execution_plan,
    resolution=resolution,
    preference_constraints=preference_constraints,
    top_k=self.config.top_k,
)
retrieval_result = self.retrieval_executor.execute(retrieval_query_plan)
query_plan["retrieval_query_plan"] = retrieval_query_plan
query_plan["retrieval_quality"] = retrieval_result["quality"]
query_plan["retrieval_trace"] = retrieval_result["trace"]
relevant_chunks = retrieval_result["chunks"]
self._print_relevant_chunk_summary(relevant_chunks)

if retrieval_result["low_evidence"]:
    low = retrieval_result["low_evidence"]
    answer = low["answer"]
    execution_result = self._build_execution_result(
        success=False,
        answer=answer,
        rewritten_question=rewritten_question,
        original_question=question,
        query_plan=query_plan,
        resolution=resolution,
        parent_docs=[],
    )
    execution_result["answer_type"] = low["answer_type"]
    execution_result["state_diff_policy"] = low["state_diff_policy"]
    execution_result["retrieval_quality"] = retrieval_result["quality"]
    execution_result["retrieval_trace"] = retrieval_result["trace"]
    self.last_execution_result = execution_result
    self._write_conversation_turn(
        session_id=session_id,
        question=question,
        answer=answer,
        turn_info=turn_info,
        query_plan=query_plan,
        resolution=resolution,
        execution_result=execution_result,
    )
    if return_diagnostics and not stream:
        self.last_query_diagnostics = self._build_turn_diagnostics(
            original_question=original_question,
            resolved_question=question,
            rewritten_query=rewritten_query,
            query_plan=query_plan,
            answer=answer,
            expectation=expectation or {},
            generation_trace={
                "strategy": "low_evidence",
                "retrieval_quality": retrieval_result["quality"],
                "retrieval_trace": retrieval_result["trace"],
            },
        )
        return {"answer": answer, "diagnostics": self.last_query_diagnostics}
    return answer
```

- [ ] **Step 6: Attach retrieval quality to successful execution results**

After each successful `_build_execution_result(...)` for list and detail retrieval, add:

```python
execution_result["retrieval_quality"] = retrieval_result["quality"]
execution_result["retrieval_trace"] = retrieval_result["trace"]
```

- [ ] **Step 7: Run chat-path tests and verify they pass**

Run:

```bash
cd code/C8
pytest tests/test_conversation_state.py::test_chat_path_uses_retrieval_executor_result tests/test_conversation_state.py::test_chat_path_returns_low_evidence_without_generation -q
```

Expected:

- PASS.

- [ ] **Step 8: Update the Stage 02 ordinal integration test for the executor contract**

The existing `test_context_first_pipeline_does_not_block_ordinal_followup_before_snapshot` in `code/C8/tests/test_conversation_state.py` monkeypatches `_search_relevant_chunks`, which is deleted in Task 6. Update it to use a `FakeExecutor` instead.

Replace:

```python
    system.retrieval_module = object()
    system.generation_module = FakeGeneration()
    system._latest_parent_docs = []
    system.last_execution_result = None

    monkeypatch.setattr(system, "_build_query_plan", lambda question, session_id: {"route_type": "detail", "dish_name": "鸡胸肉沙拉", "filters": {}, "entities": []})
    monkeypatch.setattr(system, "_apply_resolved_target_to_query_plan", lambda query_plan, resolution: query_plan)
    monkeypatch.setattr(system, "_search_relevant_chunks", lambda question, rewritten_query, filters, dish_name, top_k=5, query_dish=None: [{"content": "鸡胸肉沙拉做法", "metadata": {"dish_name": "鸡胸肉沙拉"}}])
    monkeypatch.setattr(system, "_print_relevant_chunk_summary", lambda chunks: None)
    monkeypatch.setattr(system, "_rewrite_question_for_search", lambda question, route_type: question)
    monkeypatch.setattr(system, "_generate_detail_response", lambda question, stream, session_id, route_type, filters, entities, dish_name, relevant_chunks: "鸡胸肉沙拉适合减脂。")
    monkeypatch.setattr(system, "_write_conversation_turn", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr("main.resolve_reference_from_snapshot", lambda snapshot, llm: None)
    monkeypatch.setattr("main.guard_resolution_output", lambda resolution, constraints: resolution)
```

with:

```python
    from langchain_core.documents import Document

    class FakeRetrievalModule:
        def extract_filters_from_query(self, query):
            return {}

    doc = Document(page_content="鸡胸肉沙拉做法", metadata={"dish_name": "鸡胸肉沙拉"})

    class FakeExecutor:
        def execute(self, query_plan):
            return {
                "chunks": [doc],
                "quality": {
                    "enough_evidence": True,
                    "quality_reason": "exact_dish_matched",
                    "fallback_used": False,
                    "relaxed_filter": False,
                    "candidate_count": 1,
                    "selected_dishes": ["鸡胸肉沙拉"],
                },
                "low_evidence": None,
                "trace": {"strategy": "primary"},
            }

    system.retrieval_module = FakeRetrievalModule()
    system.retrieval_executor = FakeExecutor()
    system.generation_module = FakeGeneration()
    system.config = type("Config", (), {"top_k": 3})()
    system._latest_parent_docs = []
    system.last_execution_result = None

    monkeypatch.setattr(system, "_build_query_plan", lambda question, session_id: {"route_type": "detail", "dish_name": "鸡胸肉沙拉", "filters": {}, "entities": []})
    monkeypatch.setattr(system, "_apply_resolved_target_to_query_plan", lambda query_plan, resolution: query_plan)
    monkeypatch.setattr(system, "_print_relevant_chunk_summary", lambda chunks: None)
    monkeypatch.setattr(system, "_rewrite_question_for_search", lambda question, route_type: question)
    monkeypatch.setattr(system, "_generate_detail_response", lambda question, stream, session_id, route_type, filters, entities, dish_name, relevant_chunks: "鸡胸肉沙拉适合减脂。")
    monkeypatch.setattr(system, "_write_conversation_turn", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr("main.resolve_reference_from_snapshot", lambda snapshot, llm: None)
    monkeypatch.setattr("main.guard_resolution_output", lambda resolution, constraints: resolution)
    monkeypatch.setattr("main.build_retrieval_query_plan", lambda **kwargs: kwargs)
```

- [ ] **Step 9: Commit**

```bash
git add code/C8/main.py code/C8/tests/test_conversation_state.py
git commit -m "refactor: route chat retrieval through executor"
```

---

## Task 6: Delete Old Chat Retrieval Orchestration

**Files:**
- Modify: `code/C8/main.py`
- Create: `code/C8/tests/test_retrieval_executor_cutover.py`

- [ ] **Step 1: Add cutover source tests**

Create `code/C8/tests/test_retrieval_executor_cutover.py`:

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


def test_old_search_relevant_chunks_method_is_removed():
    source = MAIN.read_text(encoding="utf-8")

    assert "def _search_relevant_chunks(" not in source


def test_ask_question_uses_retrieval_executor_not_old_low_level_branching():
    ask_question_source = _function_source("ask_question")

    assert "build_retrieval_query_plan(" in ask_question_source
    assert "self.retrieval_executor.execute(" in ask_question_source
    assert "_search_relevant_chunks(" not in ask_question_source
    assert "metadata_filtered_search(" not in ask_question_source
    assert "hybrid_search(" not in ask_question_source
    assert "fallback_filters" not in ask_question_source
    assert "pop(\"content_type\"" not in ask_question_source
    assert "pop('content_type'" not in ask_question_source


def test_ask_question_does_not_use_empty_chunks_as_quality_gate():
    ask_question_source = _function_source("ask_question")

    assert "if not relevant_chunks" not in ask_question_source
    assert "retrieval_result[\"low_evidence\"]" in ask_question_source
```

- [ ] **Step 2: Run cutover tests and verify they fail**

Run:

```bash
cd code/C8
pytest tests/test_retrieval_executor_cutover.py -q
```

Expected:

- FAIL because `_search_relevant_chunks()` still exists.

- [ ] **Step 3: Delete `_search_relevant_chunks()`**

In `code/C8/main.py`, delete the entire method:

```python
def _search_relevant_chunks(
    self,
    question: str,
    rewritten_query: str,
    filters: Dict[str, Any],
    dish_name: str,
):
    ...
```

Do not delete `_print_relevant_chunk_summary()`.

- [ ] **Step 4: Search for old chat retrieval calls**

Run:

```bash
rg -n "_search_relevant_chunks|fallback_filters|pop\\([\"']content_type|metadata_filtered_search\\(|hybrid_search\\(" code/C8/main.py
```

Expected:

- No `_search_relevant_chunks`, `fallback_filters`, or `pop("content_type")` matches.
- `metadata_filtered_search(` and `hybrid_search(` may still appear only in auxiliary helper methods outside `ask_question()`.

- [ ] **Step 5: Run cutover tests and verify they pass**

Run:

```bash
cd code/C8
pytest tests/test_retrieval_executor_cutover.py -q
```

Expected:

- PASS.

- [ ] **Step 6: Commit**

```bash
git add code/C8/main.py code/C8/tests/test_retrieval_executor_cutover.py
git commit -m "refactor: remove old chat retrieval orchestration"
```

---

## Task 7: Run Stage 03 Acceptance Suite

**Files:**
- Verify only unless existing tests require migration to the new contract.

- [ ] **Step 1: Run focused retrieval tests**

Run:

```bash
cd code/C8
pytest tests/test_retrieval_executor.py tests/test_retrieval_executor_cutover.py -q
```

Expected:

- PASS.

- [ ] **Step 2: Run conversation-state tests**

Run:

```bash
cd code/C8
pytest tests/test_conversation_state.py -q
```

Expected:

- PASS.

- [ ] **Step 3: Run source scan for forbidden chat-path retrieval ownership**

Run:

```bash
cd code/C8
python - <<'PY'
import ast
from pathlib import Path

source = Path("main.py").read_text(encoding="utf-8")
tree = ast.parse(source)
ask = None
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name == "ask_question":
        ask = ast.get_source_segment(source, node)
        break
assert ask is not None
for forbidden in [
    "_search_relevant_chunks(",
    "metadata_filtered_search(",
    "hybrid_search(",
    "fallback_filters",
    "pop(\"content_type\"",
    "pop('content_type'",
    "if not relevant_chunks",
]:
    assert forbidden not in ask, forbidden
assert "self.retrieval_executor.execute(" in ask
assert "build_retrieval_query_plan(" in ask
PY
```

Expected:

- No output and exit code 0.

- [ ] **Step 4: Run UTF-8 anchor check on Stage 03 docs and tests**

Run:

```bash
cd code/C8
python -c "from pathlib import Path; files=[Path('docs/architecture/evolution/03-retrieval-executor-and-quality.md'),Path('docs/superpowers/plans/2026-07-06-retrieval-executor-and-quality.md'),Path('tests/test_retrieval_executor.py'),Path('tests/test_retrieval_executor_cutover.py')]; [p.read_text(encoding='utf-8') for p in files]; text=files[1].read_text(encoding='utf-8'); assert 'RetrievalExecutor' in text and '西湖醋鱼' in text and 'fallback_policy' in text"
```

Expected:

- No output and exit code 0.

- [ ] **Step 5: Commit any test migration edits**

If Step 2 required updates to existing tests, run:

```bash
git add code/C8/tests/test_conversation_state.py
git commit -m "test: align retrieval tests with executor contract"
```

If Step 2 required no edits, do not create an empty commit.

---

## Self-Review

Spec coverage:

- QueryPlan normalization is covered by Task 1.
- RetrievalExecutor primary retrieval is covered by Task 2.
- Evidence Quality Check is covered by Task 3.
- Controlled fallback and fallback markers are covered by Task 4.
- Low-evidence result shape and writeback-facing fields are covered by Tasks 3 and 5.
- Chat-path production cutover is covered by Task 5.
- Deletion of old chat retrieval orchestration is covered by Task 6.
- Acceptance verification is covered by Task 7.

Type consistency:

- The normalizer function is consistently named `build_retrieval_query_plan`.
- The executor class is consistently named `RetrievalExecutor`.
- Retrieval results consistently use `chunks`, `quality`, `low_evidence`, and `trace`.
- Quality results consistently use `enough_evidence`, `quality_reason`, `fallback_used`, `relaxed_filter`, `candidate_count`, and `selected_dishes`.
- Low-evidence results consistently use `answer_type`, `answer`, `state_diff_policy`, and `quality_reason`.

Cutover consistency:

- `ask_question()` no longer owns low-level retrieval branching after Task 6.
- `_search_relevant_chunks()` is deleted rather than retained unused.
- Low-level retrieval methods may remain only inside `RetrievalOptimizationModule` and non-chat helpers.
- The plan starts with query-plan normalization because that is the boundary between upstream planning and retrieval execution.
