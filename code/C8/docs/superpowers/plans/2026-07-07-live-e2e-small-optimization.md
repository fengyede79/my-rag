# Live E2E Small Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve the first real 50-turn live E2E result from the `38/50` baseline by fixing bounded dish alias retrieval, stateful constraint followups, and live report observability without changing the frozen runtime architecture.

**Architecture:** Keep the existing main chain intact. Add a small alias resolver under the retrieval boundary, extend `Turn Understanding` to classify constraint followups as state-dependent when a current dish exists, and expose optional diagnostics through `/api/chat` so the live runner can report generation/retrieval modes.

**Tech Stack:** Python 3, Flask, pytest, LangChain `Document`, existing C8 `RetrievalExecutor`, existing live E2E runner.

## Global Constraints

- Do not introduce a new RAG chain.
- Do not introduce a new planner architecture.
- Do not introduce async runtime, queues, workers, or concurrency primitives.
- Do not introduce broad LLM judging.
- Do not perform large prompt rewrites.
- Do not create a second retrieval implementation beside `RetrievalExecutor`.
- Do not use fake live E2E shortcuts.
- Do not weaken live assertions to make the report pass.
- Preserve the frozen production chain: `Turn Understanding -> Reference Resolution -> Execution Plan -> Query Plan -> Retrieval Executor -> Evidence Quality Check -> Context Pack -> Answer Generation -> StateUpdatePolicy -> Live Report`.
- Use the `38 PASS / 12 FAIL` run `live-e2e-20260707-223633` as the baseline.
- Target at least `44 PASS / 6 FAIL` on the same primary model and scenario set, with no `INFRA_ERROR`.

---

## File Structure

- Create `rag_modules/dish_aliases.py`
  - Owns the bounded alias map and helper functions.
  - No retrieval calls, no LLM calls, no state access.
- Modify `rag_modules/retrieval_executor.py`
  - Uses aliases only after primary quality is insufficient.
  - Records alias fallback diagnostics in retrieval trace.
- Modify `rag_modules/turn_understanding.py`
  - Recognizes stateful substitution/constraint followups when a snapshot has `current_dish`.
- Modify `web_app.py`
  - Adds optional non-stream diagnostics response for live E2E through `include_diagnostics`.
- Modify `e2e/client.py`
  - Sends `include_diagnostics=true` on chat requests and stores diagnostics in `HTTPResult`.
- Modify `e2e/assertions.py`
  - Adds optional diagnostic fields to `TurnResult`.
- Modify `e2e/reporting.py`
  - Adds generation-mode and retrieval diagnostics summaries.
- Modify `e2e/live_e2e_runner.py`
  - Passes diagnostics from `HTTPResult` into `TurnResult`.
- Tests:
  - `tests/test_dish_aliases.py`
  - `tests/test_retrieval_executor.py`
  - `tests/test_turn_understanding.py`
  - `tests/test_web_app.py`
  - `tests/test_live_e2e_client.py`
  - `tests/test_live_e2e_assertions.py`
  - `tests/test_live_e2e_reporting.py`

---

### Task 1: Bounded Dish Alias Resolver

**Files:**
- Create: `rag_modules/dish_aliases.py`
- Create: `tests/test_dish_aliases.py`

**Interfaces:**
- Produces: `dish_aliases_for(dish_name: str | None) -> list[str]`
- Produces: `is_known_alias_target(original: str | None, candidate: str | None) -> bool`
- Consumed by: `RetrievalExecutor` in Task 2

- [ ] **Step 1: Write the failing tests**

Create `tests/test_dish_aliases.py`:

```python
from rag_modules.dish_aliases import dish_aliases_for, is_known_alias_target


def test_dish_aliases_for_known_live_failure_names():
    assert dish_aliases_for("番茄炒蛋") == ["西红柿炒鸡蛋", "番茄鸡蛋", "番茄炒鸡蛋"]
    assert dish_aliases_for("凉拌黄瓜") == ["拍黄瓜", "黄瓜"]
    assert dish_aliases_for("可乐鸡翅") == ["鸡翅"]
    assert dish_aliases_for("红烧肉") == ["五花肉"]


def test_dish_aliases_strip_whitespace_and_ignore_unknowns():
    assert dish_aliases_for(" 番茄炒蛋 ") == ["西红柿炒鸡蛋", "番茄鸡蛋", "番茄炒鸡蛋"]
    assert dish_aliases_for("不存在的菜") == []
    assert dish_aliases_for(None) == []


def test_known_alias_target_accepts_original_and_aliases():
    assert is_known_alias_target("番茄炒蛋", "番茄炒蛋") is True
    assert is_known_alias_target("番茄炒蛋", "西红柿炒鸡蛋") is True
    assert is_known_alias_target("番茄炒蛋", "番茄鸡蛋") is True
    assert is_known_alias_target("番茄炒蛋", "鱼香肉丝") is False
    assert is_known_alias_target(None, "番茄炒蛋") is False
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
pytest tests/test_dish_aliases.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'rag_modules.dish_aliases'`.

- [ ] **Step 3: Implement the resolver**

Create `rag_modules/dish_aliases.py`:

```python
from __future__ import annotations

"""Small bounded recipe-name alias helpers for retrieval fallback."""


DISH_ALIASES: dict[str, list[str]] = {
    "番茄炒蛋": ["西红柿炒鸡蛋", "番茄鸡蛋", "番茄炒鸡蛋"],
    "西红柿炒鸡蛋": ["番茄炒蛋", "番茄鸡蛋", "番茄炒鸡蛋"],
    "凉拌黄瓜": ["拍黄瓜", "黄瓜"],
    "可乐鸡翅": ["鸡翅"],
    "红烧肉": ["五花肉"],
}


def _normalize_name(dish_name: str | None) -> str:
    return (dish_name or "").strip()


def dish_aliases_for(dish_name: str | None) -> list[str]:
    """Return bounded aliases for a dish name, preserving configured order."""
    normalized = _normalize_name(dish_name)
    if not normalized:
        return []
    return list(DISH_ALIASES.get(normalized, []))


def is_known_alias_target(original: str | None, candidate: str | None) -> bool:
    """Return true when candidate is the original dish or a configured alias."""
    original_name = _normalize_name(original)
    candidate_name = _normalize_name(candidate)
    if not original_name or not candidate_name:
        return False
    return candidate_name == original_name or candidate_name in dish_aliases_for(original_name)
```

- [ ] **Step 4: Run tests to verify pass**

Run:

```bash
pytest tests/test_dish_aliases.py -q
```

Expected: `3 passed`.

- [ ] **Step 5: Commit**

```bash
git add rag_modules/dish_aliases.py tests/test_dish_aliases.py
git commit -m "feat: add bounded dish alias resolver"
```

---

### Task 2: Alias Fallback In RetrievalExecutor

**Files:**
- Modify: `rag_modules/retrieval_executor.py`
- Modify: `tests/test_retrieval_executor.py`

**Interfaces:**
- Consumes: `dish_aliases_for(dish_name: str | None) -> list[str]`
- Produces trace fields:
  - `strategy="alias_fallback"`
  - `dish_alias_used`
  - `fallback_used=True`
  - `relaxed_filter=True`

- [ ] **Step 1: Add failing retrieval executor tests**

Append to `tests/test_retrieval_executor.py`:

```python
def test_alias_fallback_runs_after_exact_dish_primary_fails():
    alias_doc = _doc("西红柿炒鸡蛋", "ingredients", "西红柿 鸡蛋")
    retrieval_module = FakeRetrievalModule(filtered_docs=[])

    def metadata_filtered_search(query, filters, top_k=5, query_dish=None):
        retrieval_module.calls.append(("metadata_filtered_search", query, dict(filters), top_k, query_dish))
        if filters.get("dish_name") == "西红柿炒鸡蛋":
            return [alias_doc]
        return []

    retrieval_module.metadata_filtered_search = metadata_filtered_search
    executor = RetrievalExecutor(retrieval_module)

    result = executor.execute(
        {
            "query": "番茄炒蛋需要什么食材",
            "original_query": "番茄炒蛋需要什么食材？",
            "dish_name": "番茄炒蛋",
            "filters": {"dish_name": "番茄炒蛋", "content_type": "ingredients"},
            "top_k": 3,
            "fallback_policy": "relaxed_filters",
            "hard_filters": ["dish_name"],
            "soft_filters": ["content_type"],
            "answer_mode_hint": "recipe_detail",
        }
    )

    assert result["chunks"] == [alias_doc]
    assert result["low_evidence"] is None
    assert result["quality"]["enough_evidence"] is True
    assert result["quality"]["fallback_used"] is True
    assert result["quality"]["relaxed_filter"] is True
    assert result["trace"]["strategy"] == "alias_fallback"
    assert result["trace"]["dish_alias_used"] == "西红柿炒鸡蛋"
    assert alias_doc.metadata["fallback"] is True
    assert alias_doc.metadata["relaxed_filter"] is True
    assert alias_doc.metadata["dish_alias_used"] == "西红柿炒鸡蛋"


def test_alias_fallback_does_not_run_when_primary_exact_match_succeeds():
    exact_doc = _doc("番茄炒蛋", "ingredients", "番茄 鸡蛋")
    retrieval_module = FakeRetrievalModule(filtered_docs=[exact_doc])
    executor = RetrievalExecutor(retrieval_module)

    result = executor.execute(
        {
            "query": "番茄炒蛋需要什么食材",
            "original_query": "番茄炒蛋需要什么食材？",
            "dish_name": "番茄炒蛋",
            "filters": {"dish_name": "番茄炒蛋", "content_type": "ingredients"},
            "top_k": 3,
            "fallback_policy": "relaxed_filters",
            "hard_filters": ["dish_name"],
            "soft_filters": ["content_type"],
            "answer_mode_hint": "recipe_detail",
        }
    )

    assert result["chunks"] == [exact_doc]
    assert result["trace"]["strategy"] == "primary"
    assert "dish_alias_used" not in result["trace"]
    assert len([call for call in retrieval_module.calls if call[0] == "metadata_filtered_search"]) == 1


def test_alias_fallback_keeps_low_evidence_when_alias_returns_wrong_dish():
    wrong_doc = _doc("鱼香肉丝", "steps", "鱼香肉丝步骤")
    retrieval_module = FakeRetrievalModule(filtered_docs=[])

    def metadata_filtered_search(query, filters, top_k=5, query_dish=None):
        retrieval_module.calls.append(("metadata_filtered_search", query, dict(filters), top_k, query_dish))
        if filters.get("dish_name") == "五花肉":
            return [wrong_doc]
        return []

    retrieval_module.metadata_filtered_search = metadata_filtered_search
    executor = RetrievalExecutor(retrieval_module)

    result = executor.execute(
        {
            "query": "红烧肉怎么做",
            "original_query": "红烧肉怎么做？",
            "dish_name": "红烧肉",
            "filters": {"dish_name": "红烧肉", "content_type": "steps"},
            "top_k": 3,
            "fallback_policy": "relaxed_filters",
            "hard_filters": ["dish_name"],
            "soft_filters": ["content_type"],
            "answer_mode_hint": "recipe_detail",
        }
    )

    assert result["chunks"] == []
    assert result["low_evidence"]["answer_type"] == "no_result"
    assert result["trace"]["strategy"] == "low_evidence"
```

- [ ] **Step 2: Run targeted tests to verify failure**

Run:

```bash
pytest tests/test_retrieval_executor.py::test_alias_fallback_runs_after_exact_dish_primary_fails tests/test_retrieval_executor.py::test_alias_fallback_does_not_run_when_primary_exact_match_succeeds tests/test_retrieval_executor.py::test_alias_fallback_keeps_low_evidence_when_alias_returns_wrong_dish -q
```

Expected: at least `test_alias_fallback_runs_after_exact_dish_primary_fails` FAILS because alias fallback is not implemented.

- [ ] **Step 3: Import alias helpers**

At the top of `rag_modules/retrieval_executor.py`, after `Document` import, add:

```python
from rag_modules.dish_aliases import dish_aliases_for, is_known_alias_target
```

- [ ] **Step 4: Add alias quality mode**

Change `_check_quality` signature in `rag_modules/retrieval_executor.py` to:

```python
    def _check_quality(
        self,
        query_plan: dict,
        chunks: list[Document],
        *,
        fallback_used: bool,
        relaxed_filter: bool,
        allow_alias_match: bool = False,
    ) -> dict:
```

Inside the exact dish block, replace:

```python
            if dish_name not in selected_dishes:
                enough = False
                reason = "exact_dish_not_found"
```

with:

```python
            if dish_name not in selected_dishes:
                alias_matches = [
                    selected for selected in selected_dishes
                    if allow_alias_match and is_known_alias_target(dish_name, selected)
                ]
                if alias_matches and len(selected_dishes) == 1:
                    reason = "alias_dish_matched"
                else:
                    enough = False
                    reason = "exact_dish_not_found"
```

- [ ] **Step 5: Add alias fallback execution**

In `RetrievalExecutor.execute`, immediately after the primary quality early-return block and before existing `fallback_chunks = self._fallback_retrieval(query_plan)`, insert:

```python
        alias_chunks, alias_used = self._alias_fallback_retrieval(query_plan)
        if alias_chunks:
            alias_quality = self._check_quality(
                query_plan,
                alias_chunks,
                fallback_used=True,
                relaxed_filter=True,
                allow_alias_match=True,
            )
            if alias_quality["enough_evidence"]:
                return {
                    "chunks": alias_chunks,
                    "quality": alias_quality,
                    "low_evidence": None,
                    "trace": self._build_trace(
                        query_plan=query_plan,
                        strategy="alias_fallback",
                        primary_count=len(primary_chunks),
                        fallback_count=len(alias_chunks),
                        quality=alias_quality,
                        dish_alias_used=alias_used,
                    ),
                }
```

- [ ] **Step 6: Add alias fallback helper and trace field**

Update `_build_trace` signature:

```python
    def _build_trace(
        self,
        *,
        query_plan: dict,
        strategy: str,
        primary_count: int,
        fallback_count: int,
        quality: dict,
        dish_alias_used: str | None = None,
    ) -> dict:
```

Change the return to assign to a variable and conditionally add alias:

```python
        trace = {
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
            "fallback_used": quality.get("fallback_used"),
            "relaxed_filter": quality.get("relaxed_filter"),
        }
        if dish_alias_used:
            trace["dish_alias_used"] = dish_alias_used
        return trace
```

Add this method before `_fallback_retrieval`:

```python
    def _alias_fallback_retrieval(self, query_plan: dict) -> tuple[list[Document], str | None]:
        dish_name = query_plan.get("dish_name")
        if not dish_name or "dish_name" not in set(query_plan.get("hard_filters") or []):
            return [], None

        aliases = dish_aliases_for(dish_name)
        if not aliases:
            return [], None

        base_filters = dict(query_plan.get("filters") or {})
        top_k = query_plan.get("top_k", 3)
        for alias in aliases:
            alias_filters = dict(base_filters)
            alias_filters["dish_name"] = alias
            chunks = list(
                self.retrieval_module.metadata_filtered_search(
                    query_plan["query"],
                    alias_filters,
                    top_k=top_k,
                    query_dish=alias,
                )
            )
            selected = self._selected_dishes(chunks)
            if len(selected) == 1 and is_known_alias_target(dish_name, selected[0]):
                return self._mark_fallback(chunks, dish_alias_used=alias), alias
        return [], None
```

Change `_mark_fallback` signature and body:

```python
    def _mark_fallback(self, chunks: list[Document], dish_alias_used: str | None = None) -> list[Document]:
        for chunk in chunks:
            chunk.metadata["fallback"] = True
            chunk.metadata["relaxed_filter"] = True
            if dish_alias_used:
                chunk.metadata["dish_alias_used"] = dish_alias_used
        return chunks
```

- [ ] **Step 7: Run retrieval executor tests**

Run:

```bash
pytest tests/test_dish_aliases.py tests/test_retrieval_executor.py -q
```

Expected: all tests pass.

- [ ] **Step 8: Commit**

```bash
git add rag_modules/retrieval_executor.py tests/test_retrieval_executor.py
git commit -m "feat: add alias fallback to retrieval executor"
```

---

### Task 3: Stateful Constraint Followup Understanding

**Files:**
- Modify: `rag_modules/turn_understanding.py`
- Modify: `tests/test_turn_understanding.py`

**Interfaces:**
- Produces existing `understand_turn(question: str, snapshot: dict) -> dict` with improved fields:
  - `action="substitution"` or `action="retrieve_detail"`
  - `depends_on_state=True`
  - `needs_reference_resolution=True`
  - `reference_trigger="constraint_followup"`

- [ ] **Step 1: Add failing turn understanding tests**

Append to `tests/test_turn_understanding.py`:

```python
def test_constraint_followup_without_pronoun_inherits_current_dish():
    result = understand_turn("没有花生可以吗", _snapshot(current_dish="宫保鸡丁"))

    assert result["action"] == "substitution"
    assert result["answer_mode_hint"] == "substitution"
    assert result["should_retrieve"] is True
    assert result["depends_on_state"] is True
    assert result["needs_reference_resolution"] is True
    assert result["reference_trigger"] == "constraint_followup"


def test_low_oil_followup_inherits_current_dish_as_substitution():
    result = understand_turn("能少油一点吗", _snapshot(current_dish="宫保鸡丁"))

    assert result["action"] == "substitution"
    assert result["answer_mode_hint"] == "substitution"
    assert result["reference_trigger"] == "constraint_followup"


def test_takeaway_suitability_followup_inherits_current_dish_as_constraint_check():
    result = understand_turn("适合带饭吗", _snapshot(current_dish="宫保鸡丁"))

    assert result["action"] == "retrieve_detail"
    assert result["answer_mode_hint"] == "constraint_check"
    assert result["depends_on_state"] is True
    assert result["needs_reference_resolution"] is True
    assert result["reference_trigger"] == "constraint_followup"


def test_constraint_language_without_state_remains_normal_recipe_detail():
    result = understand_turn("能少油一点吗", _snapshot())

    assert result["action"] == "retrieve_detail"
    assert result["needs_reference_resolution"] is False
    assert result["reference_trigger"] == "none"


def test_out_of_domain_query_does_not_inherit_current_dish():
    result = understand_turn("Python 可以少油一点吗", _snapshot(current_dish="宫保鸡丁"))

    assert result["action"] == "domain_reject"
    assert result["should_retrieve"] is False
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
pytest tests/test_turn_understanding.py::test_constraint_followup_without_pronoun_inherits_current_dish tests/test_turn_understanding.py::test_low_oil_followup_inherits_current_dish_as_substitution tests/test_turn_understanding.py::test_takeaway_suitability_followup_inherits_current_dish_as_constraint_check tests/test_turn_understanding.py::test_constraint_language_without_state_remains_normal_recipe_detail tests/test_turn_understanding.py::test_out_of_domain_query_does_not_inherit_current_dish -q
```

Expected: first three tests fail because constraint followup detection is missing.

- [ ] **Step 3: Add token sets and helper functions**

In `rag_modules/turn_understanding.py`, after `UNSUPPORTED_ORDINAL_INTENTS`, add:

```python
SUBSTITUTION_FOLLOWUP_SIGNALS = {
    "没有",
    "不放",
    "不要",
    "替代",
    "换成",
    "少油",
    "少盐",
    "少糖",
}

CONSTRAINT_FOLLOWUP_SIGNALS = {
    "适合带饭",
    "适合新手",
    "热量高",
    "减脂",
    "不辣",
}
```

After `_has_unsupported_ordinal_intent`, add:

```python
def _has_substitution_followup_signal(text: str) -> bool:
    return any(signal in text for signal in SUBSTITUTION_FOLLOWUP_SIGNALS)


def _has_constraint_followup_signal(text: str) -> bool:
    return any(signal in text for signal in CONSTRAINT_FOLLOWUP_SIGNALS)
```

- [ ] **Step 4: Insert stateful constraint branch**

In `understand_turn`, after the pronoun branch and before the out-of-domain branch, insert:

```python
    if _has_out_of_domain_signal(text) and not _has_recipe_signal(text):
        return _base_result(
            action="domain_reject",
            answer_mode_hint="safe_direct",
            should_retrieve=False,
            domain_confidence=0.95,
            reason="harmless_out_of_domain",
        )

    if _has_current_dish(snapshot) and _has_substitution_followup_signal(text):
        return _base_result(
            action="substitution",
            answer_mode_hint="substitution",
            should_retrieve=True,
            depends_on_state=True,
            needs_reference_resolution=True,
            reference_trigger="constraint_followup",
            reason="stateful_substitution_followup",
        )

    if _has_current_dish(snapshot) and _has_constraint_followup_signal(text):
        return _base_result(
            action="retrieve_detail",
            answer_mode_hint="constraint_check",
            should_retrieve=True,
            depends_on_state=True,
            needs_reference_resolution=True,
            reference_trigger="constraint_followup",
            reason="stateful_constraint_followup",
        )
```

Remove the old duplicated out-of-domain branch immediately below this inserted code if it remains present.

- [ ] **Step 5: Preserve no-state behavior**

Ensure `DETAIL_SIGNALS` includes enough constraint words for no-state direct detail classification:

```python
DETAIL_SIGNALS = {"怎么做", "做法", "食材", "材料", "配料", "步骤", "技巧", "热量", "适合", "减脂", "不放", "替换", "为什么", "少油", "少盐", "少糖", "不要", "没有"}
```

- [ ] **Step 6: Run turn understanding tests**

Run:

```bash
pytest tests/test_turn_understanding.py -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add rag_modules/turn_understanding.py tests/test_turn_understanding.py
git commit -m "feat: inherit current dish for constraint followups"
```

---

### Task 4: Optional Diagnostics Through Live HTTP Path

**Files:**
- Modify: `web_app.py`
- Modify: `e2e/client.py`
- Modify: `e2e/assertions.py`
- Modify: `e2e/live_e2e_runner.py`
- Modify: `tests/test_web_app.py`
- Modify: `tests/test_live_e2e_client.py`
- Modify: `tests/test_live_e2e_assertions.py`

**Interfaces:**
- `POST /api/chat` accepts optional JSON boolean `include_diagnostics`.
- When `include_diagnostics` is true, response shape is:
  - `{"answer": str, "diagnostics": dict | None}`
- `HTTPResult` gains `diagnostics: dict[str, Any] | None = None`.
- `TurnResult` gains optional diagnostic fields:
  - `model_requested`
  - `generation_mode`
  - `context_doc_count`
  - `retrieval_strategy`
  - `quality_reason`
  - `selected_dishes`
  - `fallback_used`
  - `dish_alias_used`

- [ ] **Step 1: Add failing web app diagnostics test**

Append to `tests/test_web_app.py`:

```python
def test_chat_can_return_diagnostics_when_requested():
    class FakeGeneration:
        model_name = "qwen-plus-2025-07-28"
        last_generation_trace = {"strategy": "structured", "context_doc_count": 1}

    class FakeSystem:
        generation_module = FakeGeneration()
        last_execution_result = {}

        def ask_question(self, question, stream=False, session_id="default"):
            self.last_execution_result = {
                "retrieval_trace": {
                    "strategy": "primary",
                    "quality_reason": "exact_dish_matched",
                    "selected_dishes": ["蛋炒饭"],
                    "fallback_used": False,
                },
                "retrieval_quality": {"quality_reason": "exact_dish_matched"},
                "context_pack_trace": {"context_doc_count": 1},
            }
            return "诊断回答"

    app = create_app(system_factory=lambda: FakeSystem())
    client = app.test_client()

    response = client.post(
        "/api/chat",
        json={"question": "蛋炒饭怎么做？", "session_id": "s1", "include_diagnostics": True},
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["answer"] == "诊断回答"
    assert payload["diagnostics"]["generation"]["strategy"] == "structured"
    assert payload["diagnostics"]["retrieval"]["strategy"] == "primary"
    assert payload["diagnostics"]["model_requested"] == "qwen-plus-2025-07-28"
```

If `tests/test_web_app.py` does not import `create_app`, add:

```python
from web_app import create_app
```

- [ ] **Step 2: Add failing client/assertion tests**

Append to `tests/test_live_e2e_assertions.py`:

```python
def test_turn_result_records_optional_diagnostics():
    result = evaluate_assertions(
        run_id="run",
        model="qwen-plus-2025-07-28",
        scenario_id="s1",
        category="single_recipe_detail",
        session_id="sess",
        turn_index=1,
        endpoint="chat",
        question="蛋炒饭怎么做？",
        http_status=200,
        answer="蛋炒饭需要鸡蛋和米饭。",
        assertions={"http_status": 200, "answer_contains_any": ["蛋炒饭"]},
        latency_ms=10,
        attempt=1,
        sse_done_event=None,
        error=None,
        diagnostics={
            "model_requested": "qwen-plus-2025-07-28",
            "generation": {"strategy": "structured", "context_doc_count": 2},
            "retrieval": {
                "strategy": "alias_fallback",
                "quality_reason": "alias_dish_matched",
                "selected_dishes": ["西红柿炒鸡蛋"],
                "fallback_used": True,
                "dish_alias_used": "西红柿炒鸡蛋",
            },
        },
    )

    assert result.model_requested == "qwen-plus-2025-07-28"
    assert result.generation_mode == "structured"
    assert result.context_doc_count == 2
    assert result.retrieval_strategy == "alias_fallback"
    assert result.quality_reason == "alias_dish_matched"
    assert result.selected_dishes == ["西红柿炒鸡蛋"]
    assert result.fallback_used is True
    assert result.dish_alias_used == "西红柿炒鸡蛋"
```

Append to `tests/test_live_e2e_client.py`:

```python
def test_parse_chat_payload_with_diagnostics():
    from e2e.client import parse_chat_payload

    payload = {
        "answer": "回答",
        "diagnostics": {"generation": {"strategy": "structured"}},
    }

    answer, diagnostics = parse_chat_payload(payload)

    assert answer == "回答"
    assert diagnostics == {"generation": {"strategy": "structured"}}
```

- [ ] **Step 3: Run tests to verify failure**

Run:

```bash
pytest tests/test_web_app.py::test_chat_can_return_diagnostics_when_requested tests/test_live_e2e_assertions.py::test_turn_result_records_optional_diagnostics tests/test_live_e2e_client.py::test_parse_chat_payload_with_diagnostics -q
```

Expected: failures because diagnostics are not wired.

- [ ] **Step 4: Modify web app chat route**

In `web_app.py`, add this helper above `create_app`:

```python
def _build_live_diagnostics(system: RecipeRAGSystem) -> dict:
    execution = getattr(system, "last_execution_result", {}) or {}
    generation_module = getattr(system, "generation_module", None)
    generation_trace = getattr(generation_module, "last_generation_trace", {}) or {}
    retrieval_trace = execution.get("retrieval_trace") or {}
    retrieval_quality = execution.get("retrieval_quality") or {}
    context_trace = execution.get("context_pack_trace") or {}
    model_requested = (
        getattr(generation_module, "model_name", None)
        or getattr(getattr(system, "config", None), "llm_model", None)
    )

    generation_strategy = generation_trace.get("strategy")
    if not generation_strategy and execution.get("answer_type") == "no_result":
        generation_strategy = "no_context"
    if not generation_strategy and execution.get("answer_mode") == "recommendation":
        generation_strategy = "list_template"

    context_doc_count = generation_trace.get("context_doc_count")
    if context_doc_count is None:
        context_doc_count = context_trace.get("context_doc_count")

    def first_present(primary: dict, secondary: dict, key: str):
        if key in primary:
            return primary.get(key)
        return secondary.get(key)

    return {
        "model_requested": model_requested,
        "generation": {
            "strategy": generation_strategy,
            "context_doc_count": context_doc_count,
            "content_type": generation_trace.get("content_type"),
        },
        "retrieval": {
            "strategy": retrieval_trace.get("strategy"),
            "quality_reason": first_present(retrieval_trace, retrieval_quality, "quality_reason"),
            "selected_dishes": first_present(retrieval_trace, retrieval_quality, "selected_dishes"),
            "fallback_used": first_present(retrieval_trace, retrieval_quality, "fallback_used"),
            "relaxed_filter": first_present(retrieval_trace, retrieval_quality, "relaxed_filter"),
            "dish_alias_used": retrieval_trace.get("dish_alias_used"),
        },
    }
```

In `web_app.py`, inside `chat`, after `session_id` assignment, add:

```python
        include_diagnostics = bool(payload.get("include_diagnostics"))
```

Replace:

```python
        answer = system.ask_question(question, stream=False, session_id=session_id)
        if isinstance(answer, dict):
            answer = answer.get("answer", "")
```

with:

```python
        result = system.ask_question(question, stream=False, session_id=session_id)
        if isinstance(result, dict):
            answer = result.get("answer", "")
        else:
            answer = result
        diagnostics = _build_live_diagnostics(system) if include_diagnostics else None
```

Replace:

```python
        return jsonify({"answer": answer})
```

with:

```python
        response_payload = {"answer": answer}
        if include_diagnostics:
            response_payload["diagnostics"] = diagnostics
        return jsonify(response_payload)
```

- [ ] **Step 5: Add chat payload parser and diagnostics request**

In `e2e/client.py`, add `Any` import:

```python
from typing import Any
```

Change `HTTPResult` to:

```python
@dataclass(frozen=True)
class HTTPResult:
    http_status: int | None
    answer: str
    latency_ms: int
    error: str | None = None
    sse_done_event: bool | None = None
    diagnostics: dict[str, Any] | None = None
```

Add after `ParsedSSE`:

```python
def parse_chat_payload(payload: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
    answer = str(payload.get("answer", ""))
    diagnostics = payload.get("diagnostics")
    return answer, diagnostics if isinstance(diagnostics, dict) else None
```

In `LiveE2EClient.chat`, change body to include diagnostics:

```python
        body = json.dumps(
            {"question": question, "session_id": session_id, "include_diagnostics": True},
            ensure_ascii=False,
        ).encode("utf-8")
```

Replace payload handling:

```python
                payload = json.loads(response.read().decode("utf-8"))
                answer, diagnostics = parse_chat_payload(payload)
                return HTTPResult(
                    http_status=response.status,
                    answer=answer,
                    latency_ms=int((time.time() - started) * 1000),
                    diagnostics=diagnostics,
                )
```

Apply the same `diagnostics=None` default implicitly for error returns; no changes are needed there beyond dataclass compatibility.

- [ ] **Step 6: Extend TurnResult diagnostics**

In `e2e/assertions.py`, add these fields to `TurnResult` after `error`:

```python
    model_requested: str | None = None
    generation_mode: str | None = None
    context_doc_count: int | None = None
    retrieval_strategy: str | None = None
    quality_reason: str | None = None
    selected_dishes: list[str] | None = None
    fallback_used: bool | None = None
    dish_alias_used: str | None = None
```

Add them to `to_dict()`:

```python
            "model_requested": self.model_requested,
            "generation_mode": self.generation_mode,
            "context_doc_count": self.context_doc_count,
            "retrieval_strategy": self.retrieval_strategy,
            "quality_reason": self.quality_reason,
            "selected_dishes": self.selected_dishes,
            "fallback_used": self.fallback_used,
            "dish_alias_used": self.dish_alias_used,
```

Add helper before `evaluate_assertions`:

```python
def _diagnostic_fields(diagnostics: dict[str, Any] | None, model: str) -> dict[str, Any]:
    diagnostics = diagnostics or {}
    generation = diagnostics.get("generation") if isinstance(diagnostics.get("generation"), dict) else {}
    retrieval = diagnostics.get("retrieval") if isinstance(diagnostics.get("retrieval"), dict) else {}
    selected = retrieval.get("selected_dishes")
    return {
        "model_requested": diagnostics.get("model_requested") or model,
        "generation_mode": generation.get("strategy"),
        "context_doc_count": generation.get("context_doc_count"),
        "retrieval_strategy": retrieval.get("strategy"),
        "quality_reason": retrieval.get("quality_reason"),
        "selected_dishes": selected if isinstance(selected, list) else None,
        "fallback_used": retrieval.get("fallback_used"),
        "dish_alias_used": retrieval.get("dish_alias_used"),
    }
```

Change `evaluate_assertions` signature to include:

```python
    diagnostics: dict[str, Any] | None = None,
```

At the start of `evaluate_assertions`, add:

```python
    diagnostic_fields = _diagnostic_fields(diagnostics, model)
```

In every `TurnResult(...)` constructor inside `evaluate_assertions`, add:

```python
            **diagnostic_fields,
```

- [ ] **Step 7: Pass diagnostics from runner**

In `e2e/live_e2e_runner.py`, in the `evaluate_assertions(...)` call, add:

```python
                    diagnostics=response.diagnostics,
```

- [ ] **Step 8: Run diagnostics tests**

Run:

```bash
pytest tests/test_web_app.py::test_chat_can_return_diagnostics_when_requested tests/test_live_e2e_assertions.py::test_turn_result_records_optional_diagnostics tests/test_live_e2e_client.py::test_parse_chat_payload_with_diagnostics -q
```

Expected: all pass.

- [ ] **Step 9: Run live e2e unit suite**

Run:

```bash
pytest tests/test_live_e2e_client.py tests/test_live_e2e_assertions.py tests/test_live_e2e_runner.py -q
```

Expected: all pass.

- [ ] **Step 10: Commit**

```bash
git add web_app.py e2e/client.py e2e/assertions.py e2e/live_e2e_runner.py tests/test_web_app.py tests/test_live_e2e_client.py tests/test_live_e2e_assertions.py
git commit -m "feat: expose live e2e diagnostics"
```

---

### Task 5: Report Generation Mode And Retrieval Diagnostics

**Files:**
- Modify: `e2e/reporting.py`
- Modify: `tests/test_live_e2e_reporting.py`

**Interfaces:**
- Consumes: `TurnResult.generation_mode`, `retrieval_strategy`, `quality_reason`, `dish_alias_used`.
- Produces Markdown sections:
  - `## Generation Mode Summary`
  - `## Retrieval Strategy Summary`
  - diagnostics columns in failure table.

- [ ] **Step 1: Add failing reporting test**

Modify helper `_result` in `tests/test_live_e2e_reporting.py` to accept diagnostics:

```python
def _result(
    status: str,
    category: str = "domain_reject",
    *,
    generation_mode: str | None = None,
    retrieval_strategy: str | None = None,
    quality_reason: str | None = None,
    dish_alias_used: str | None = None,
) -> TurnResult:
    return TurnResult(
        run_id="run-1",
        model="qwen-plus-2025-07-28",
        scenario_id="s1",
        category=category,
        turn_index=1,
        session_id="sess",
        endpoint="chat",
        question="Python 怎么学？",
        http_status=200,
        answer="我主要处理食谱问题。",
        status=status,
        failure_class=None if status == "PASS" else status,
        latency_ms=100,
        attempt=1,
        error=None,
        model_requested="qwen-plus-2025-07-28",
        generation_mode=generation_mode,
        context_doc_count=1 if generation_mode else None,
        retrieval_strategy=retrieval_strategy,
        quality_reason=quality_reason,
        selected_dishes=["西红柿炒鸡蛋"] if dish_alias_used else None,
        fallback_used=bool(dish_alias_used) if dish_alias_used else None,
        dish_alias_used=dish_alias_used,
    )
```

Append test:

```python
def test_markdown_report_includes_generation_and_retrieval_diagnostics(tmp_path: Path):
    results = [
        _result("PASS", generation_mode="structured", retrieval_strategy="primary"),
        _result("PASS", generation_mode="llm", retrieval_strategy="alias_fallback", quality_reason="alias_dish_matched", dish_alias_used="西红柿炒鸡蛋"),
        _result("FAIL", generation_mode="no_context", retrieval_strategy="low_evidence", quality_reason="no_candidates"),
    ]
    markdown = tmp_path / "run.md"

    write_markdown_report(
        markdown,
        run_id="run-1",
        models=["qwen-plus-2025-07-28"],
        delay_seconds=5,
        results=results,
    )

    report = markdown.read_text(encoding="utf-8")
    assert "## Generation Mode Summary" in report
    assert "| structured | 1 |" in report
    assert "| llm | 1 |" in report
    assert "| no_context | 1 |" in report
    assert "## Retrieval Strategy Summary" in report
    assert "| alias_fallback | 1 |" in report
    assert "Quality Reason" in report
    assert "no_candidates" in report
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
pytest tests/test_live_e2e_reporting.py::test_markdown_report_includes_generation_and_retrieval_diagnostics -q
```

Expected: FAIL because report sections do not exist.

- [ ] **Step 3: Extend summarize_results**

In `e2e/reporting.py`, change `summarize_results` return to:

```python
def summarize_results(results: list[TurnResult]) -> dict:
    return {
        "total": len(results),
        "by_status": dict(Counter(result.status for result in results)),
        "by_model": dict(Counter(result.model for result in results)),
        "by_category": dict(Counter(result.category for result in results)),
        "by_generation_mode": dict(Counter(result.generation_mode or "unknown" for result in results)),
        "by_retrieval_strategy": dict(Counter(result.retrieval_strategy or "unknown" for result in results)),
    }
```

- [ ] **Step 4: Add report sections**

In `write_markdown_report`, after Category Summary section, insert:

```python
        "## Generation Mode Summary",
        "",
        _table(summary["by_generation_mode"]),
        "",
        "## Retrieval Strategy Summary",
        "",
        _table(summary["by_retrieval_strategy"]),
        "",
```

- [ ] **Step 5: Extend failure table**

Replace failure table header:

```python
        "| Model | Scenario | Turn | Status | Generation | Retrieval | Quality Reason | Error |",
        "| --- | --- | ---: | --- | --- | --- | --- | --- |",
```

Replace failure row append with:

```python
        generation = result.generation_mode or "unknown"
        retrieval = result.retrieval_strategy or "unknown"
        quality = (result.quality_reason or "").replace("|", "\\|")[:120]
        lines.append(
            f"| {result.model} | {result.scenario_id} | {result.turn_index} | {result.status} "
            f"| {generation} | {retrieval} | {quality} | {error} |"
        )
```

- [ ] **Step 6: Run reporting tests**

Run:

```bash
pytest tests/test_live_e2e_reporting.py -q
```

Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add e2e/reporting.py tests/test_live_e2e_reporting.py
git commit -m "test: summarize live e2e diagnostics"
```

---

### Task 6: Deterministic Regression And Live Acceptance

**Files:**
- No production file changes expected.
- Generated live reports under `e2e/results/` are not committed unless explicitly requested.

**Interfaces:**
- Consumes all previous task outputs.
- Produces verification evidence and remaining-failure classification.

- [ ] **Step 1: Run focused unit tests**

Run:

```bash
pytest tests/test_dish_aliases.py tests/test_retrieval_executor.py tests/test_turn_understanding.py tests/test_web_app.py tests/test_live_e2e_client.py tests/test_live_e2e_assertions.py tests/test_live_e2e_reporting.py tests/test_live_e2e_runner.py -q
```

Expected: all pass.

- [ ] **Step 2: Run deterministic architecture regression tests**

Run:

```bash
pytest tests/test_end_to_end_acceptance.py tests/test_final_cutover.py tests/test_conversation_state.py tests/test_state_hardening.py -q
```

Expected: all pass.

- [ ] **Step 3: Run 1-turn live smoke**

Run:

```bash
python e2e/live_e2e_runner.py --models qwen-plus-2025-07-28 --limit-turns 1 --delay-seconds 0 --max-retries 0 --request-timeout-seconds 300 --stream-timeout-seconds 300 --port 5065
```

Expected:

```text
[qwen-plus-2025-07-28] 1/1 single_recipe_detail_001 PASS
```

Also inspect the generated Markdown report and confirm it contains:

```text
## Generation Mode Summary
## Retrieval Strategy Summary
```

- [ ] **Step 4: Run full 50-turn live acceptance**

Run:

```bash
python e2e/live_e2e_runner.py --models qwen-plus-2025-07-28 --limit-turns 50 --delay-seconds 5 --max-retries 1 --request-timeout-seconds 300 --stream-timeout-seconds 300 --port 5066
```

Expected:

- No `INFRA_ERROR`.
- No unbounded timeout.
- No repeated rate-limit stop.
- Target result is at least `44 PASS / 6 FAIL`.

- [ ] **Step 5: Classify remaining failures**

Open the generated Markdown report under `e2e/results/`.

For every remaining FAIL, classify in the final implementation notes as one of:

```text
data coverage
retrieval behavior
answer-generation behavior
assertion mismatch
```

Do not edit scenario assertions unless the failure is clearly an assertion mismatch and the answer is objectively correct.

- [ ] **Step 6: Check worktree**

Run:

```bash
git status --short
```

Expected:

- Source/test changes are committed from prior tasks.
- `web_app.runtime.log` may be modified.
- New live report files under `e2e/results/` may exist and are ignored by `e2e/.gitignore`.

- [ ] **Step 7: Final response**

Report:

- focused unit test result;
- deterministic regression result;
- 1-turn live smoke result;
- 50-turn live result compared to `38/50`;
- path to the generated Markdown report;
- remaining failure categories.

Do not claim the optimization achieved the target unless the 50-turn report proves it.

---

## Self-Review

Spec coverage:

- Alias-aware detail fallback is covered by Tasks 1 and 2.
- Stateful constraint followup inheritance is covered by Task 3.
- Live report generation-mode observability is covered by Tasks 4 and 5.
- Live smoke and full live acceptance are covered by Task 6.
- Frozen architecture constraints are listed in Global Constraints and no task adds a new production node.

Placeholder scan:

- This plan contains no forbidden placeholder terms or unspecified test steps.

Type consistency:

- `dish_aliases_for(dish_name: str | None) -> list[str]` and `is_known_alias_target(original, candidate) -> bool` are introduced in Task 1 and consumed in Task 2.
- `HTTPResult.diagnostics` is introduced in Task 4 and consumed by `evaluate_assertions`.
- `TurnResult` diagnostic fields are introduced in Task 4 and consumed by reporting in Task 5.
