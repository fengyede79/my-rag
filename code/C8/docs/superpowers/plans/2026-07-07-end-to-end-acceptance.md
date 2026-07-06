# End-To-End Acceptance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove the frozen C8 recipe RAG runtime works end to end, then remove or block stale production paths that would let the old architecture survive beside the new one.

**Architecture:** Stage 06 does not add a new runtime layer. It builds deterministic acceptance fixtures around the real `RecipeRAGSystem.ask_question()` chain, asserts trace/state behavior across multi-turn scenarios, adds final source-level cutover tests, and writes an acceptance report.

**Tech Stack:** Python, pytest, LangChain `Document`, AST/source inspection tests, existing C8 modules (`RecipeRAGSystem`, `RetrievalExecutor`, `ContextPacker`, `ConversationManager`, `StateUpdatePolicy`).

---

## Preconditions

Execute this plan after Stage 05 has been implemented.

Required Stage 05 surface before Task 1 begins:

- `rag_modules.turn_runtime.TurnRuntimeContext` exists.
- `RecipeRAGSystem.ask_question()` uses one shared runtime context and retry budget.
- stream writeback is lifecycle-aware.
- `ConversationManager.commit_state_diff()` and `writeback_turn_state(..., expected_state_version=...)` exist.

If any precondition is missing, finish the Stage 05 plan first.

## File Structure

- Create `code/C8/tests/acceptance_fixtures.py`
  - Owns deterministic recipe docs, fake retrieval/data/generation modules, and trace helpers.
  - Uses the real `RecipeRAGSystem`, `RetrievalExecutor`, `ContextPacker`, `ConversationManager`, and state writeback path.
- Create `code/C8/tests/test_end_to_end_acceptance.py`
  - Owns scenario-level acceptance tests.
  - Tests behavior and trace/state outcomes, not exact generated prose.
- Create `code/C8/tests/test_final_cutover.py`
  - Owns source-level final architecture checks.
  - Prevents old production paths from remaining active after cutover.
- Modify `code/C8/main.py`
  - Only for cleanup if final cutover tests reveal obsolete helpers or old stream wrapper still exist.
- Modify `code/C8/docs/architecture/evolution/06-end-to-end-acceptance.md`
  - Records final stage acceptance summary.
- Create `code/C8/docs/architecture/evolution/acceptance-report.md`
  - Records scenario results and known residual risks.

---

## Task 1: Create Acceptance Fixture Harness

**Files:**
- Create: `code/C8/tests/acceptance_fixtures.py`
- Test: `code/C8/tests/test_end_to_end_acceptance.py`

- [ ] **Step 1: Write the failing fixture smoke test**

Create `code/C8/tests/test_end_to_end_acceptance.py`:

```python
from acceptance_fixtures import ask_and_trace, build_acceptance_system


def test_acceptance_fixture_uses_real_runtime_boundaries():
    system = build_acceptance_system()

    answer, trace = ask_and_trace(system, "推荐三个鸡肉菜", session_id="fixture-smoke")

    session = system.generation_module.conversation_manager.get_session("fixture-smoke")
    assert "宫保鸡丁" in answer
    assert session.recent_recommendations
    assert trace["query_plan"]["route_type"] == "list"
    assert trace["retrieval_quality"]["enough_evidence"] is True
    assert trace["context_pack_trace"]["selected_section_count"] >= 1
    assert trace["commit_result"]["committed"] is True
```

- [ ] **Step 2: Run the smoke test and verify it fails**

Run:

```bash
cd code/C8
pytest tests/test_end_to_end_acceptance.py::test_acceptance_fixture_uses_real_runtime_boundaries -q
```

Expected:

- FAIL with `ModuleNotFoundError: No module named 'acceptance_fixtures'`.

- [ ] **Step 3: Create deterministic acceptance fixtures**

Create `code/C8/tests/acceptance_fixtures.py`:

```python
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from langchain_core.documents import Document

from main import RecipeRAGSystem
from rag_modules.context_packer import ContextPacker
from rag_modules.conversation_manager import ConversationManager
from rag_modules.retrieval_executor import RetrievalExecutor


RECIPE_DOCS: dict[str, str] = {
    "宫保鸡丁": (
        "# 宫保鸡丁的做法\n\n"
        "## 必备原料和工具\n\n"
        "- 鸡腿肉\n- 花生\n- 干辣椒\n- 豆瓣酱可选\n\n"
        "## 操作\n\n"
        "1. 鸡肉切丁腌制。\n2. 先炒鸡丁，再加入调味汁。\n3. 最后放花生。\n\n"
        "## 附加内容\n\n"
        "- 不吃辣可以减少干辣椒，用甜椒补香味。\n"
        "- 没有豆瓣酱时可用少量生抽、醋和糖调味。\n"
    ),
    "香菇滑鸡": (
        "# 香菇滑鸡的做法\n\n"
        "## 必备原料和工具\n\n"
        "- 鸡腿肉\n- 香菇\n- 姜片\n\n"
        "## 操作\n\n"
        "1. 鸡肉腌制。\n2. 香菇炒香。\n3. 合炒后小火焖熟。\n\n"
        "## 附加内容\n\n"
        "- 口味温和，不辣。\n"
    ),
    "可乐鸡翅": (
        "# 可乐鸡翅的做法\n\n"
        "## 必备原料和工具\n\n"
        "- 鸡翅\n- 可乐\n- 生抽\n\n"
        "## 操作\n\n"
        "1. 鸡翅煎上色。\n2. 加可乐和生抽焖煮。\n3. 收汁。\n\n"
        "## 附加内容\n\n"
        "- 甜口，不辣。\n"
    ),
    "番茄炒蛋": (
        "# 番茄炒蛋的做法\n\n"
        "## 必备原料和工具\n\n"
        "- 番茄\n- 鸡蛋\n\n"
        "## 操作\n\n"
        "1. 先炒鸡蛋。\n2. 再炒番茄出汁。\n3. 合炒调味。\n"
    ),
}


def _chunk(dish_name: str, content_type: str = "general", **metadata: Any) -> Document:
    return Document(
        page_content=f"{dish_name} {content_type}",
        metadata={
            "dish_name": dish_name,
            "parent_id": f"{dish_name}-parent",
            "content_type": content_type,
            **metadata,
        },
    )


class AcceptanceRetrievalModule:
    def __init__(self):
        self.last_search_trace: dict[str, Any] = {}

    def extract_filters_from_query(self, query: str) -> dict[str, Any]:
        filters: dict[str, Any] = {}
        if "不辣" in query or "不放辣" in query:
            filters["taste"] = "不辣"
        if "鸡" in query:
            filters["ingredient"] = "鸡"
        return filters

    def metadata_filtered_search(
        self,
        query: str,
        filters: dict[str, Any],
        top_k: int = 3,
        query_dish: str | None = None,
    ):
        self.last_search_trace = {
            "method": "metadata_filtered_search",
            "query": query,
            "filters": dict(filters),
            "query_dish": query_dish,
        }
        if "不存在的菜" in query or query_dish == "不存在的菜":
            return []

        dish_name = filters.get("dish_name") or query_dish
        if dish_name:
            if dish_name not in RECIPE_DOCS:
                return []
            return [_chunk(dish_name, filters.get("content_type", "general"))]

        if filters.get("taste") == "不辣":
            return [
                _chunk("香菇滑鸡", "general"),
                _chunk("可乐鸡翅", "general"),
            ][:top_k]

        if filters.get("ingredient") == "鸡" or "鸡" in query:
            return [
                _chunk("宫保鸡丁", "general"),
                _chunk("香菇滑鸡", "general"),
                _chunk("可乐鸡翅", "general"),
            ][:top_k]

        return [_chunk("番茄炒蛋", "general")][:top_k]

    def hybrid_search(self, query: str, top_k: int = 3, query_dish: str | None = None):
        self.last_search_trace = {
            "method": "hybrid_search",
            "query": query,
            "query_dish": query_dish,
        }
        if "不存在的菜" in query or query_dish == "不存在的菜":
            return []
        if query_dish and query_dish in RECIPE_DOCS:
            return [_chunk(query_dish, "general")]
        if "鸡" in query:
            return [
                _chunk("宫保鸡丁", "general"),
                _chunk("香菇滑鸡", "general"),
                _chunk("可乐鸡翅", "general"),
            ][:top_k]
        return [_chunk("番茄炒蛋", "general")][:top_k]


class AcceptanceDataModule:
    def get_parent_documents(self, chunks, target_dish_name: str | None = None):
        dishes: list[str] = []
        if target_dish_name:
            dishes.append(target_dish_name)
        for chunk in chunks:
            dish_name = (chunk.metadata or {}).get("dish_name")
            if dish_name and dish_name not in dishes:
                dishes.append(dish_name)

        return [
            Document(
                page_content=RECIPE_DOCS[dish_name],
                metadata={
                    "dish_name": dish_name,
                    "parent_id": f"{dish_name}-parent",
                    "rrf_score": 1.0,
                },
            )
            for dish_name in dishes
            if dish_name in RECIPE_DOCS
        ]


class AcceptanceGenerationModule:
    def __init__(self):
        self.conversation_manager = ConversationManager()
        self.last_generation_trace: dict[str, Any] = {}
        self.llm = None

    def resolve_query_reference(self, query, session_id):
        return query

    def query_router(self, query: str) -> dict[str, Any]:
        if "推荐" in query or "换个" in query:
            filters = {}
            if "鸡" in query:
                filters["ingredient"] = "鸡"
            if "不辣" in query or "不放辣" in query:
                filters["taste"] = "不辣"
            return {"type": "list", "filters": filters, "dish_name": None, "confidence": 0.95}

        dish_name = None
        for name in RECIPE_DOCS:
            if name in query:
                dish_name = name
                break
        if "不存在的菜" in query:
            dish_name = "不存在的菜"

        content_type = "steps"
        if "豆瓣酱" in query or "替代" in query or "不放辣" in query:
            content_type = "tips"
        if "材料" in query or "食材" in query:
            content_type = "ingredients"

        return {
            "type": "detail",
            "filters": {"content_type": content_type},
            "dish_name": dish_name,
            "confidence": 0.95,
        }

    def get_current_entity(self, session_id):
        return self.conversation_manager.get_current_entity(session_id)

    def _classify_query_guardrail(self, query):
        return None

    def query_rewrite(self, query):
        return query

    def _dish_names(self, context_docs):
        names: list[str] = []
        for doc in context_docs:
            name = (doc.metadata or {}).get("dish_name")
            if name and name not in names:
                names.append(name)
        return names

    def generate_smalltalk_answer(self, query: str) -> str:
        self.last_generation_trace = {"strategy": "smalltalk"}
        return "不客气，继续想做菜也可以问我。"

    def generate_list_answer(self, query, context_docs):
        names = self._dish_names(context_docs)
        self.last_generation_trace = {"strategy": "list", "dishes": names}
        return "为你推荐：\n" + "\n".join(f"{index + 1}. {name}" for index, name in enumerate(names))

    def generate_step_by_step_answer(self, query, context_docs, content_type=None):
        names = self._dish_names(context_docs)
        joined = "\n".join(doc.page_content for doc in context_docs)
        self.last_generation_trace = {
            "strategy": "detail",
            "dishes": names,
            "content_type": content_type,
        }
        if "豆瓣酱" in query:
            return f"{names[0]}可以不用豆瓣酱，改用生抽、醋和糖。" if names else "可以换成生抽、醋和糖。"
        if "不放辣" in query:
            return f"{names[0]}可以少放或不放辣椒。" if names else "可以不放辣椒。"
        if names:
            return f"{names[0]}做法：{joined[:120]}"
        return "知识库里没有找到可靠的食谱信息。"

    def generate_step_by_step_answer_stream(self, query, context_docs, content_type=None):
        text = self.generate_step_by_step_answer(query, context_docs, content_type=content_type)
        for part in [text[:10], text[10:]]:
            if part:
                yield part

    def generate_basic_answer(self, query, context_docs, content_type=None):
        return self.generate_step_by_step_answer(query, context_docs, content_type=content_type)

    def generate_basic_answer_stream(self, query, context_docs, content_type=None):
        yield self.generate_basic_answer(query, context_docs, content_type=content_type)


def build_acceptance_system() -> RecipeRAGSystem:
    system = RecipeRAGSystem.__new__(RecipeRAGSystem)
    system.config = SimpleNamespace(
        top_k=3,
        context_pack_max_chars_total=2400,
        context_pack_max_chars_per_doc=1200,
        context_pack_max_docs=5,
    )
    system.data_module = AcceptanceDataModule()
    system.retrieval_module = AcceptanceRetrievalModule()
    system.retrieval_executor = RetrievalExecutor(system.retrieval_module)
    system.context_packer = ContextPacker(
        max_chars_total=system.config.context_pack_max_chars_total,
        max_chars_per_doc=system.config.context_pack_max_chars_per_doc,
        max_docs=system.config.context_pack_max_docs,
    )
    system.generation_module = AcceptanceGenerationModule()
    system._latest_parent_docs = []
    system.last_query_diagnostics = {}
    system.last_execution_result = {}
    return system


def ask_and_trace(system: RecipeRAGSystem, question: str, *, session_id: str, stream: bool = False):
    answer = system.ask_question(question, stream=stream, session_id=session_id)
    if stream:
        answer = "".join(list(answer))
    trace = dict(system.last_execution_result or {})
    trace.setdefault("query_plan", trace.get("query_plan") or {})
    trace.setdefault("retrieval_quality", trace.get("retrieval_quality") or trace["query_plan"].get("retrieval_quality") or {})
    trace.setdefault("context_pack_trace", trace.get("context_pack_trace") or {})
    trace.setdefault("runtime", trace.get("runtime") or {})
    trace.setdefault("commit_result", trace.get("commit_result") or trace.get("writeback_result") or {})
    return answer, trace
```

- [ ] **Step 4: Run the fixture smoke test**

Run:

```bash
cd code/C8
pytest tests/test_end_to_end_acceptance.py::test_acceptance_fixture_uses_real_runtime_boundaries -q
```

Expected:

- PASS after Stage 05 exposes `commit_result` or equivalent writeback result on `last_execution_result`.
- If it fails only because the commit result is not attached to `last_execution_result`, update `_write_conversation_turn()` to store the returned commit result on `execution_result["commit_result"]`.

- [ ] **Step 5: Commit**

```bash
git add code/C8/tests/acceptance_fixtures.py code/C8/tests/test_end_to_end_acceptance.py code/C8/main.py
git commit -m "test: add end-to-end acceptance fixtures"
```

---

## Task 2: Add Primary Multi-Turn Acceptance Scenario

**Files:**
- Modify: `code/C8/tests/test_end_to_end_acceptance.py`

- [ ] **Step 1: Add primary scenario test**

Append to `code/C8/tests/test_end_to_end_acceptance.py`:

```python

def test_primary_multi_turn_recipe_chain_preserves_state_and_trace():
    system = build_acceptance_system()
    session_id = "primary-chain"

    first_answer, first_trace = ask_and_trace(system, "推荐三个鸡肉菜", session_id=session_id)
    assert "宫保鸡丁" in first_answer
    assert first_trace["query_plan"]["route_type"] == "list"
    assert first_trace["retrieval_quality"]["enough_evidence"] is True

    second_answer, second_trace = ask_and_trace(system, "第一个怎么做", session_id=session_id)
    assert "宫保鸡丁" in second_answer
    assert second_trace["query_plan"]["route_type"] == "detail"
    assert second_trace["retrieval_quality"]["quality_reason"] == "exact_dish_matched"

    third_answer, third_trace = ask_and_trace(system, "这个能不放辣吗", session_id=session_id)
    assert "不放辣" in third_answer or "辣椒" in third_answer
    assert third_trace["query_plan"]["route_type"] == "detail"

    fourth_answer, fourth_trace = ask_and_trace(system, "没有豆瓣酱怎么办", session_id=session_id)
    assert "生抽" in fourth_answer
    assert fourth_trace["query_plan"]["route_type"] == "detail"

    fifth_answer, fifth_trace = ask_and_trace(system, "给我换个不辣的", session_id=session_id)
    assert "香菇滑鸡" in fifth_answer or "可乐鸡翅" in fifth_answer
    assert fifth_trace["query_plan"]["route_type"] == "list"

    sixth_answer, sixth_trace = ask_and_trace(system, "谢谢", session_id=session_id)
    assert "不客气" in sixth_answer
    assert sixth_trace.get("query_plan", {}) == {}

    session = system.generation_module.conversation_manager.get_session(session_id)
    assert session.current_entity == "宫保鸡丁"
    assert session.recent_recommendations
    assert session.last_answer_type == "smalltalk"
    assert session.pending_clarification is None
    assert session.state_version >= 6
```

- [ ] **Step 2: Run the primary scenario and verify it fails if any stage boundary is incomplete**

Run:

```bash
cd code/C8
pytest tests/test_end_to_end_acceptance.py::test_primary_multi_turn_recipe_chain_preserves_state_and_trace -q
```

Expected:

- PASS only when Stages 01-05 are active together.
- A failure should be routed back to the owner stage described in `2026-07-07-end-to-end-acceptance-design.md`.

- [ ] **Step 3: Fix only architecture-consistent gaps**

Allowed fixes:

- expose missing trace fields on `execution_result`;
- attach commit result from `_write_conversation_turn()`;
- adjust test fixtures to better resemble recipe documents;
- remove stale production calls that bypass the new chain.

Disallowed fixes:

- bypassing `RetrievalExecutor`;
- writing state directly from generation;
- letting low-evidence paths update `current_entity`;
- adding a new planner or async runtime.

- [ ] **Step 4: Rerun the primary scenario**

Run:

```bash
cd code/C8
pytest tests/test_end_to_end_acceptance.py::test_primary_multi_turn_recipe_chain_preserves_state_and_trace -q
```

Expected:

- PASS.

- [ ] **Step 5: Commit**

```bash
git add code/C8/tests/test_end_to_end_acceptance.py code/C8/main.py code/C8/rag_modules
git commit -m "test: accept primary multi-turn runtime chain"
```

---

## Task 3: Add Required Additional Acceptance Scenarios

**Files:**
- Modify: `code/C8/tests/test_end_to_end_acceptance.py`

- [ ] **Step 1: Add domain reject and unrelated ordinal tests**

Append to `code/C8/tests/test_end_to_end_acceptance.py`:

```python

def test_harmless_out_of_domain_rejects_without_retrieval_or_business_state():
    system = build_acceptance_system()

    answer, trace = ask_and_trace(system, "Python 怎么学", session_id="domain-reject")

    session = system.generation_module.conversation_manager.get_session("domain-reject")
    assert "食谱" in answer or "做菜" in answer
    assert trace.get("query_plan", {}) == {}
    assert session.current_entity is None
    assert session.recent_recommendations == []


def test_unrelated_ordinal_after_recommendation_does_not_silently_resolve_as_recipe_detail():
    system = build_acceptance_system()
    ask_and_trace(system, "推荐三个鸡肉菜", session_id="unrelated-ordinal")

    answer, trace = ask_and_trace(system, "第一个作者是谁", session_id="unrelated-ordinal")

    session = system.generation_module.conversation_manager.get_session("unrelated-ordinal")
    assert session.current_entity is None
    assert trace.get("answer_type") in {"domain_reject", "clarification", "low_confidence", "no_result", None}
    assert "宫保鸡丁做法" not in answer
```

- [ ] **Step 2: Add missing dish and sparse metadata tests**

Append:

```python

def test_exact_missing_dish_returns_no_result_without_current_dish_update():
    system = build_acceptance_system()

    answer, trace = ask_and_trace(system, "不存在的菜怎么做", session_id="missing-dish")

    session = system.generation_module.conversation_manager.get_session("missing-dish")
    assert "没有找到可靠" in answer
    assert trace["retrieval_quality"]["enough_evidence"] is False
    assert trace.get("answer_type") in {"no_result", "low_confidence"}
    assert session.current_entity is None


def test_sparse_metadata_preference_uses_soft_weighting_or_marked_fallback():
    system = build_acceptance_system()

    answer, trace = ask_and_trace(system, "推荐不辣的鸡肉菜", session_id="sparse-metadata")

    assert "香菇滑鸡" in answer or "可乐鸡翅" in answer
    assert trace["retrieval_quality"]["enough_evidence"] is True
    assert "taste" in trace["query_plan"]["retrieval_query_plan"]["soft_filters"]
    if trace["retrieval_quality"]["fallback_used"]:
        assert trace["retrieval_quality"]["relaxed_filter"] is True
```

- [ ] **Step 3: Add stream abort and rapid conflict tests**

Append:

```python

def test_stream_abort_after_recommendation_does_not_create_valid_ordinal_list():
    system = build_acceptance_system()
    stream = system.ask_question("推荐三个鸡肉菜", stream=True, session_id="stream-abort-acceptance")

    first = next(stream)
    assert first
    stream.close()

    session = system.generation_module.conversation_manager.get_session("stream-abort-acceptance")
    assert session.recent_recommendations == []
    assert system.last_execution_result["runtime"]["lifecycle"]["status"] == "aborted"

    answer, trace = ask_and_trace(system, "第一个怎么做", session_id="stream-abort-acceptance")
    assert trace.get("answer_type") in {"clarification", "no_result", "low_confidence", None}
    assert "宫保鸡丁做法" not in answer


def test_rapid_state_dependent_turn_reaches_shared_conflict_path(monkeypatch):
    system = build_acceptance_system()
    manager = system.generation_module.conversation_manager
    ask_and_trace(system, "推荐三个鸡肉菜", session_id="rapid-conflict")

    original_build_context_pack = system.context_packer.build_context_pack

    def mutate_before_generation(**kwargs):
        pack = original_build_context_pack(**kwargs)
        manager.commit_state_diff(
            "rapid-conflict",
            {
                "answer_type": "smalltalk",
                "updates": {"last_answer_type": "smalltalk"},
                "clear": [],
                "append_history": False,
                "history": None,
            },
            expected_version=manager.get_state_version("rapid-conflict"),
        )
        return pack

    monkeypatch.setattr(system.context_packer, "build_context_pack", mutate_before_generation)

    answer, trace = ask_and_trace(system, "第一个怎么做", session_id="rapid-conflict")

    assert "上下文刚刚更新" in answer
    assert trace["runtime"]["replan_count"] == 1
    assert trace["runtime"]["lifecycle"]["status"] == "failed"
```

- [ ] **Step 4: Add low-evidence and final smalltalk tests**

Append:

```python

def test_low_evidence_detail_does_not_update_current_dish():
    system = build_acceptance_system()

    answer, trace = ask_and_trace(system, "不存在的菜怎么做", session_id="low-evidence-detail")

    session = system.generation_module.conversation_manager.get_session("low-evidence-detail")
    assert "没有找到可靠" in answer
    assert trace.get("answer_type") in {"no_result", "low_confidence"}
    assert session.current_entity is None


def test_final_smalltalk_after_recipe_flow_does_not_clear_business_state():
    system = build_acceptance_system()
    ask_and_trace(system, "推荐三个鸡肉菜", session_id="final-smalltalk")
    ask_and_trace(system, "第一个怎么做", session_id="final-smalltalk")

    before = system.generation_module.conversation_manager.get_session("final-smalltalk")
    assert before.current_entity == "宫保鸡丁"
    assert before.recent_recommendations

    answer, trace = ask_and_trace(system, "谢谢", session_id="final-smalltalk")

    after = system.generation_module.conversation_manager.get_session("final-smalltalk")
    assert "不客气" in answer
    assert trace.get("query_plan", {}) == {}
    assert after.current_entity == "宫保鸡丁"
    assert after.recent_recommendations
```

- [ ] **Step 5: Run all end-to-end acceptance scenarios**

Run:

```bash
cd code/C8
pytest tests/test_end_to_end_acceptance.py -q
```

Expected:

- PASS.

- [ ] **Step 6: Commit**

```bash
git add code/C8/tests/test_end_to_end_acceptance.py code/C8/tests/acceptance_fixtures.py code/C8/main.py code/C8/rag_modules
git commit -m "test: add final runtime acceptance scenarios"
```

---

## Task 4: Add Final Cutover Tests And Remove Stale Production Paths

**Files:**
- Create: `code/C8/tests/test_final_cutover.py`
- Modify: `code/C8/main.py`

- [ ] **Step 1: Add source-level final cutover tests**

Create `code/C8/tests/test_final_cutover.py`:

```python
import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "main.py"


def _source() -> str:
    return MAIN.read_text(encoding="utf-8")


def _function_source(function_name: str) -> str:
    source = _source()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            return ast.get_source_segment(source, node)
    raise AssertionError(f"{function_name} not found")


def test_runtime_chain_uses_shared_context_and_lifecycle_wrapper():
    source = _source()
    ask_source = _function_source("ask_question")

    assert "TurnRuntimeContext.start(" in ask_source
    assert "_ask_question_once(" in ask_source
    assert "should_replan_after_mismatch(" in ask_source
    assert "def _wrap_stream_with_writeback" not in source
    assert "def _wrap_stream_with_lifecycle" in source


def test_chat_path_retrieval_only_goes_through_retrieval_executor():
    once_source = _function_source("_ask_question_once")

    assert "self.retrieval_executor.execute(" in once_source
    assert ".metadata_filtered_search(" not in once_source
    assert ".hybrid_search(" not in once_source


def test_generation_helpers_do_not_expand_parent_docs_or_write_state():
    for function_name in ["_generate_list_response", "_generate_detail_response"]:
        helper_source = _function_source(function_name)
        assert "get_parent_documents(" not in helper_source
        assert "conversation_manager" not in helper_source
        assert "writeback_turn_state(" not in helper_source
        assert "record_recommendations(" not in helper_source
        assert "set_current_dish(" not in helper_source
        assert "add_interaction(" not in helper_source


def test_state_writeback_uses_policy_and_expected_version():
    write_source = _function_source("_write_conversation_turn")
    manager_source = (ROOT / "rag_modules" / "conversation_manager.py").read_text(encoding="utf-8")

    assert "writeback_turn_state(" in write_source
    assert "expected_state_version" in write_source
    assert "build_state_diff(" in manager_source
    assert "commit_state_diff(" in manager_source


def test_legacy_convenience_search_helpers_are_not_left_as_independent_runtime_paths():
    source = _source()

    assert "def search_by_category" not in source
    assert "def get_ingredients_list" not in source
```

- [ ] **Step 2: Run final cutover tests and verify they fail on remaining stale paths**

Run:

```bash
cd code/C8
pytest tests/test_final_cutover.py -q
```

Expected:

- FAIL if any stale production path remains.
- In the current pre-cutover code, expected failures include `_wrap_stream_with_writeback`, `search_by_category`, or `get_ingredients_list` if they still exist.

- [ ] **Step 3: Replace old stream wrapper if Stage 05 left it behind**

In `code/C8/main.py`, delete `def _wrap_stream_with_writeback(...)` and ensure the final lifecycle wrapper is named:

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
        ...
```

Change the stream return inside `_ask_question_once()` to:

```python
            return self._wrap_stream_with_lifecycle(
                answer_stream=answer,
                session_id=session_id,
                question=question,
                turn_info=turn_info,
                query_plan=query_plan,
                resolution=resolution,
                execution_result=execution_result,
                runtime_ctx=runtime_ctx,
            )
```

The lifecycle wrapper body should already be provided by the Stage 05 plan. Do not add a second wrapper.

- [ ] **Step 4: Delete obsolete convenience runtime helpers if unused**

Before deleting, verify they are not used:

```bash
cd code/C8
rg -n "search_by_category|get_ingredients_list" .
```

Expected before deletion:

- only definitions in `main.py`, or definitions plus tests that should be removed or rewritten.

Delete these methods from `RecipeRAGSystem` if they are not actively called by the new runtime:

```python
    def search_by_category(...)
    def get_ingredients_list(...)
```

Reason:

- They perform direct retrieval/generation outside `RetrievalExecutor`, `ContextPacker`, and `StateUpdatePolicy`.
- They are not part of the frozen main chat chain.

- [ ] **Step 5: Run final cutover tests**

Run:

```bash
cd code/C8
pytest tests/test_final_cutover.py -q
```

Expected:

- PASS.

- [ ] **Step 6: Run end-to-end acceptance after cleanup**

Run:

```bash
cd code/C8
pytest tests/test_end_to_end_acceptance.py tests/test_final_cutover.py -q
```

Expected:

- PASS.

- [ ] **Step 7: Commit**

```bash
git add code/C8/main.py code/C8/tests/test_final_cutover.py code/C8/tests/test_end_to_end_acceptance.py
git commit -m "test: enforce final runtime cutover"
```

---

## Task 5: Write Acceptance Report And Run Final Suite

**Files:**
- Create: `code/C8/docs/architecture/evolution/acceptance-report.md`
- Modify: `code/C8/docs/architecture/evolution/06-end-to-end-acceptance.md`

- [ ] **Step 1: Create the acceptance report**

Create `code/C8/docs/architecture/evolution/acceptance-report.md`:

```markdown
# Stage 06 Acceptance Report

Date: 2026-07-07

## Scope

Stage 06 validates the frozen C8 recipe RAG runtime architecture end to end.

The validated chain is:

```text
basic safety
-> session snapshot + state_version
-> turn understanding
-> reference resolution
-> execution plan
-> query plan
-> retrieval executor + evidence quality
-> context pack
-> answer generation or stream lifecycle
-> StateUpdatePolicy
-> versioned state commit
```

## Scenario Results

| Scenario | Result |
| --- | --- |
| Primary multi-turn recipe chain | Pass |
| Harmless out-of-domain rejection | Pass |
| Unrelated ordinal after recommendation | Pass |
| Exact missing dish | Pass |
| Sparse metadata preference query | Pass |
| Stream abort after recommendation | Pass |
| Rapid state-dependent conflict | Pass |
| Low-evidence detail state safety | Pass |
| Final smalltalk state preservation | Pass |

## Cutover Checks

| Check | Result |
| --- | --- |
| Chat retrieval goes through `RetrievalExecutor` | Pass |
| Generation helpers do not expand parent docs | Pass |
| Generation helpers do not write session state | Pass |
| Writeback uses `StateUpdatePolicy` and expected-version commit | Pass |
| Old stream writeback wrapper is removed | Pass |
| Obsolete direct retrieval helpers are removed | Pass |

## Known Residual Risks

- Acceptance fixtures are deterministic and smaller than the full recipe corpus.
- Answer wording quality still depends on prompts and generation behavior.
- Sparse metadata behavior is validated through soft weighting and fallback markers, not large-scale recall metrics.

These risks do not contradict the frozen architecture.

## Final Decision

Stage 06 is accepted. The runtime architecture migration is complete. Future work should be bugfixes, answer quality tuning, data quality improvements, or new feature specs, not a new architecture migration stage.
```

- [ ] **Step 2: Update Stage 06 evolution doc with acceptance status**

Modify `code/C8/docs/architecture/evolution/06-end-to-end-acceptance.md`:

```markdown
# 06 End To End Acceptance

Status: accepted

Detailed spec:

- `../../superpowers/specs/2026-07-07-end-to-end-acceptance-design.md`

Acceptance report:

- `acceptance-report.md`
```

Keep the existing purpose, scenarios, and acceptance sections below this header. Do not remove the scenario matrix.

- [ ] **Step 3: Run focused final acceptance**

Run:

```bash
cd code/C8
pytest tests/test_end_to_end_acceptance.py tests/test_final_cutover.py -q
```

Expected:

- PASS.

- [ ] **Step 4: Run adjacent architecture regression tests**

Run:

```bash
cd code/C8
pytest tests/test_state_update_policy.py tests/test_context_first_cutover.py tests/test_retrieval_executor.py tests/test_retrieval_executor_cutover.py tests/test_context_packer.py tests/test_context_packer_cutover.py tests/test_turn_runtime.py -q
```

Expected:

- PASS.

- [ ] **Step 5: Run web boundary tests**

Run:

```bash
cd code/C8
pytest tests/test_web_app.py -q
```

Expected:

- PASS.

- [ ] **Step 6: Run placeholder and final-path scans**

Run:

```bash
cd code/C8
python -c "from pathlib import Path; files=[Path('docs/superpowers/specs/2026-07-07-end-to-end-acceptance-design.md'),Path('docs/superpowers/plans/2026-07-07-end-to-end-acceptance.md'),Path('docs/architecture/evolution/06-end-to-end-acceptance.md'),Path('docs/architecture/evolution/acceptance-report.md')]; bad=['TO'+'DO','TB'+'D','implement'+' later','fill'+' in','Similar'+' to Task','Add'+' appropriate','Write tests'+' for the above']; hits=[(str(p), b) for p in files for b in bad if b in p.read_text(encoding='utf-8')]; assert not hits, hits"
python -c "from pathlib import Path; source=Path('main.py').read_text(encoding='utf-8'); assert 'def _wrap_stream_with_writeback' not in source; assert 'def _wrap_stream_with_lifecycle' in source; assert 'def search_by_category' not in source; assert 'def get_ingredients_list' not in source"
```

Expected:

- both commands exit 0.

- [ ] **Step 7: Commit**

```bash
git add code/C8/docs/architecture/evolution/06-end-to-end-acceptance.md code/C8/docs/architecture/evolution/acceptance-report.md code/C8/docs/superpowers/plans/2026-07-07-end-to-end-acceptance.md
git commit -m "docs: accept final runtime architecture"
```

---

## Task 6: Final Full Verification

**Files:**
- Verify only unless tests expose a migration gap.

- [ ] **Step 1: Run the complete C8 test suite**

Run:

```bash
cd code/C8
pytest tests -q
```

Expected:

- PASS.

- [ ] **Step 2: Inspect untracked and modified files**

Run:

```bash
git status --short
```

Expected:

- only intended Stage 06 files and any necessary implementation cleanup files are modified.
- unrelated user files are not staged.

- [ ] **Step 3: Commit any final implementation fixes**

If Task 6 required fixes, commit them:

```bash
git add code/C8/main.py code/C8/rag_modules code/C8/tests
git commit -m "fix: close final runtime acceptance gaps"
```

If no files changed, do not create an empty commit.

---

## Self-Review

Spec coverage:

- Primary multi-turn scenario is covered by Task 2.
- Domain reject, unrelated ordinal, missing dish, sparse metadata, stream abort, rapid conflict, low evidence, and final smalltalk are covered by Task 3.
- Trace assertions are covered by Task 1, Task 2, and Task 3.
- Stale-path cleanup is covered by Task 4.
- Acceptance report is covered by Task 5.
- Final verification is covered by Task 6.

Cutover consistency:

- The plan does not add a new runtime architecture.
- The plan keeps `RetrievalExecutor`, `ContextPacker`, `StateUpdatePolicy`, and versioned `ConversationManager` as the owners of their responsibilities.
- The plan treats old wrappers and direct retrieval helpers as illegal once the new path covers the behavior.
- The plan routes failures back to Stages 01-05 rather than creating parallel fixes in Stage 06.

Fixture consistency:

- Fixtures use recipe-shaped Markdown with `##` sections.
- Fixtures run through real `RecipeRAGSystem.ask_question()`.
- Fixtures use real `RetrievalExecutor`, `ContextPacker`, and `ConversationManager`.
- Stream abort tests use a real generator and explicit `close()`.
- Conflict tests mutate state deterministically, without sleeps or timing assumptions.
