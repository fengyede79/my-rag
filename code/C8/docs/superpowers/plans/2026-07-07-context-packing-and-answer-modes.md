# Context Packing And Answer Modes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move parent expansion and context selection out of generation helpers and into a `ContextPacker` that returns packed generation context plus explicit answer mode and trace data.

**Architecture:** Add `rag_modules/context_packer.py` as the context boundary after retrieval. `ask_question()` performs parent expansion once, builds a `ContextPack`, passes packed `Document` objects into generation, and keeps original parent docs only for diagnostics/writeback.

**Tech Stack:** Python, pytest, LangChain `Document`, existing `RecipeRAGSystem`, existing `DataPreparationModule.get_parent_documents()`, existing generation methods.

---

## Cutover Contract

This plan assumes Stage 03 retrieval executor is implemented.

Old responsibility being replaced:

- `_generate_list_response()` expands parent docs and passes full parents to list generation.
- `_generate_detail_response()` expands parent docs and passes full parents to detail generation.
- Generation helpers indirectly decide how much parent-document context reaches prompts.

New ownership:

- `ContextPacker` extracts Markdown sections, selects sections by answer mode/content type, trims context, and returns `ContextPack`.
- `ask_question()` owns parent expansion placement after retrieval and before generation.
- Generation helpers accept already-packed `context_docs` and never call `get_parent_documents()`.

Production cutover:

```text
ask_question()
  old: _generate_*_response(..., relevant_chunks) -> get_parent_documents(...) -> generation(full_parent_docs)
  new: parent_docs = get_parent_documents(...)
       context_pack = ContextPacker.build_context_pack(...)
       _generate_*_response(..., context_pack) -> generation(context_pack["context_docs"])
```

Illegal after cutover:

- `_generate_list_response()` calling `data_module.get_parent_documents()`.
- `_generate_detail_response()` calling `data_module.get_parent_documents()`.
- `ask_question()` passing raw `relevant_chunks` into generation helpers.
- `_generate_detail_response()` keeping a `filters` parameter only to recover `content_type`.
- `_generate_detail_response()` keeping unused `session_id` or `entities` parameters.
- `_generate_list_response()` keeping an unused `session_id` parameter.
- generation helpers assigning `_latest_parent_docs`.
- context-pack trace being duplicated into both `query_plan` and `execution_result`.
- `ContextPacker()` being instantiated with hard-coded defaults instead of `RAGConfig` values.
- generation helper logs describing parent docs as if full parent docs were passed to generation.

Deletion before acceptance:

- Remove parent-expansion code from `_generate_list_response()`.
- Remove parent-expansion code from `_generate_detail_response()`.
- Remove `filters` from `_generate_detail_response(...)`.
- Remove `session_id` from `_generate_list_response(...)`.
- Remove `session_id` and `entities` from `_generate_detail_response(...)`.
- Add source-level tests proving generation helpers no longer call `get_parent_documents()`.
- Add source-level tests proving context-pack trace is written only to `execution_result`.
- Add source-level tests proving `ContextPacker` receives its budgets from `RAGConfig`.

---

## File Structure

- Create `code/C8/rag_modules/context_packer.py`
  - Defines `extract_markdown_sections(parent_doc)`.
  - Defines `finalize_answer_mode(...)`.
  - Defines `ContextPacker`.
  - Owns section selection, trimming, compact fallback context, and context-pack trace.

- Modify `code/C8/rag_modules/__init__.py`
  - Exports `context_packer` for consistency with existing submodule exports.

- Create `code/C8/tests/test_context_packer.py`
  - Unit tests for section extraction, answer mode finalization, section selection, trimming, chunk fallback, and recommendation summaries.

- Modify `code/C8/main.py`
  - Imports `ContextPacker`.
  - Instantiates `self.context_packer`.
  - Performs parent expansion after retrieval and before generation.
  - Passes `context_pack` to generation helper methods.
  - Sets `_latest_parent_docs` once in `ask_question()` from `context_pack["parent_docs"]`.

- Modify: `code/C8/config.py`
  - Adds context-pack budget fields to `RAGConfig`.
  - Loads them from environment variables.
  - Validates they are positive integers.

- Create `code/C8/tests/test_context_packer_cutover.py`
  - Source-level tests proving parent expansion lives in `ask_question()` and not generation helpers.

- Modify `code/C8/tests/test_conversation_state.py`
  - Add integration tests proving packed docs are passed to generation and original parent docs still feed writeback diagnostics.
  - Migrate existing monkeypatched `_generate_detail_response` call sites to the new signature:
    - `test_context_first_pipeline_does_not_block_ordinal_followup_before_snapshot`
    - `test_chat_path_uses_retrieval_executor_result`
  - Confirm `test_chat_path_returns_low_evidence_without_generation` still returns before context packing and does not need `data_module`.
  - Migrate `_system()` and `_tips_system()` fixtures so successful retrieval tests have section-shaped parent docs and a context packer path:
    - `test_stream_detail_turn_persists_conversation_state_after_stream_consumed`
    - `test_list_turn_is_recorded_in_conversation_history`
    - `test_tips_query_falls_back_to_same_dish_when_tips_chunks_are_missing`
    - `test_detail_generation_uses_single_new_writeback_path`
    - `test_recipe_question_preserves_original_for_query_planning`
  - Keep stream-related tests passing after the helper signature change.

- Modify `code/C8/tests/test_state_hardening.py`
  - Migrate `_system_with_generation()` fixture to construct `ContextPacker` and use recipe-shaped parent docs.

---

## Task 1: Add Markdown Section Extraction

**Files:**
- Create: `code/C8/rag_modules/context_packer.py`
- Modify: `code/C8/rag_modules/__init__.py`
- Create: `code/C8/tests/test_context_packer.py`

- [ ] **Step 1: Write section extraction tests**

Create `code/C8/tests/test_context_packer.py`:

```python
from langchain_core.documents import Document

from rag_modules.context_packer import extract_markdown_sections


RECIPE_MARKDOWN = """# 宫保鸡丁的做法

宫保鸡丁是一道家常菜。

## 必备原料和工具

- 鸡腿肉 300g
- 花生米 50g
- 干辣椒 适量

## 计算

两人份。

## 操作

### 腌制鸡肉

- 鸡肉切丁。
- 加入淀粉抓匀。

### 炒制

- 热锅下油。
- 倒入鸡丁翻炒。

## 附加内容

- 火候不要太大。
- 可以减少辣椒。
"""


def _parent_doc(text=RECIPE_MARKDOWN):
    return Document(
        page_content=text,
        metadata={"dish_name": "宫保鸡丁", "parent_id": "parent-1"},
    )


def test_extract_markdown_sections_maps_common_recipe_headings():
    sections = extract_markdown_sections(_parent_doc())

    assert [section["section_type"] for section in sections] == [
        "introduction",
        "ingredients",
        "calculation",
        "steps",
        "tips",
    ]
    assert sections[0]["title"] == "宫保鸡丁的做法"
    assert sections[1]["title"] == "必备原料和工具"
    assert sections[3]["title"] == "操作"
    assert "### 腌制鸡肉" in sections[3]["text"]
    assert sections[3]["dish_name"] == "宫保鸡丁"
    assert sections[3]["parent_id"] == "parent-1"


def test_extract_markdown_sections_keeps_text_before_first_second_level_heading_as_introduction():
    sections = extract_markdown_sections(_parent_doc())

    introduction = sections[0]
    assert introduction["section_type"] == "introduction"
    assert "宫保鸡丁是一道家常菜" in introduction["text"]
    assert "## 必备原料和工具" not in introduction["text"]


def test_extract_markdown_sections_preserves_h3_substeps_inside_steps_section():
    sections = extract_markdown_sections(_parent_doc())

    steps = next(section for section in sections if section["section_type"] == "steps")
    assert steps["title"] == "操作"
    assert "### 腌制鸡肉" in steps["text"]
    assert "### 炒制" in steps["text"]
    assert "腌制鸡肉" not in [section["title"] for section in sections]
```

- [ ] **Step 2: Run section extraction tests and verify they fail**

Run:

```bash
cd code/C8
pytest tests/test_context_packer.py::test_extract_markdown_sections_maps_common_recipe_headings tests/test_context_packer.py::test_extract_markdown_sections_keeps_text_before_first_second_level_heading_as_introduction tests/test_context_packer.py::test_extract_markdown_sections_preserves_h3_substeps_inside_steps_section -q
```

Expected:

- FAIL because `rag_modules.context_packer` does not exist.

- [ ] **Step 3: Implement section extraction**

Create `code/C8/rag_modules/context_packer.py`. Stage 04 intentionally treats only `#` and `##` as section boundaries; `###` and deeper headings stay inside their containing `##` section so ordered substeps remain together.

```python
"""Context packing and answer-mode finalization for recipe generation."""

from __future__ import annotations

import re
from typing import Any, Dict, List

from langchain_core.documents import Document


SECTION_TYPE_ALIASES = {
    "ingredients": ("必备原料和工具", "原料", "食材", "材料", "配料"),
    "steps": ("操作", "步骤", "做法", "制作"),
    "tips": ("附加内容", "注意", "小贴士", "技巧", "提示"),
    "calculation": ("计算", "用量", "份量", "分量"),
}


def _section_type_for_heading(title: str) -> str:
    for section_type, aliases in SECTION_TYPE_ALIASES.items():
        if any(alias in title for alias in aliases):
            return section_type
    return "introduction"


def extract_markdown_sections(parent_doc: Document) -> List[Dict[str, Any]]:
    """Extract coarse recipe sections from a Markdown parent document."""
    text = parent_doc.page_content or ""
    dish_name = (parent_doc.metadata or {}).get("dish_name")
    parent_id = (parent_doc.metadata or {}).get("parent_id")
    sections: list[dict[str, Any]] = []
    current_title: str | None = None
    current_level = 1
    current_lines: list[str] = []

    def flush() -> None:
        if not current_lines:
            return
        section_text = "\n".join(current_lines).strip()
        if not section_text:
            return
        title = current_title or (dish_name or "简介")
        sections.append(
            {
                "title": title,
                "section_type": _section_type_for_heading(title) if current_level >= 2 else "introduction",
                "text": section_text,
                "level": current_level,
                "dish_name": dish_name,
                "parent_id": parent_id,
                "source_metadata": dict(parent_doc.metadata or {}),
            }
        )

    for line in text.splitlines():
        match = re.match(r"^(#{1,2})\s+(.+?)\s*$", line)
        if match:
            flush()
            current_level = len(match.group(1))
            current_title = match.group(2).strip()
            current_lines = [line]
            continue
        current_lines.append(line)

    flush()
    return sections
```

Then update `code/C8/rag_modules/__init__.py`:

```python
from . import context_packer
```

and add `"context_packer"` to `__all__`.

- [ ] **Step 4: Run section extraction tests and verify they pass**

Run:

```bash
cd code/C8
pytest tests/test_context_packer.py::test_extract_markdown_sections_maps_common_recipe_headings tests/test_context_packer.py::test_extract_markdown_sections_keeps_text_before_first_second_level_heading_as_introduction tests/test_context_packer.py::test_extract_markdown_sections_preserves_h3_substeps_inside_steps_section -q
```

Expected:

- PASS.

- [ ] **Step 5: Commit**

```bash
git add code/C8/rag_modules/context_packer.py code/C8/rag_modules/__init__.py code/C8/tests/test_context_packer.py
git commit -m "feat: extract recipe context sections"
```

---

## Task 2: Add Answer Mode Finalization

**Files:**
- Modify: `code/C8/rag_modules/context_packer.py`
- Modify: `code/C8/tests/test_context_packer.py`

- [ ] **Step 1: Add answer-mode finalization tests**

Append to `code/C8/tests/test_context_packer.py`:

```python
from rag_modules.context_packer import finalize_answer_mode


def test_finalize_answer_mode_uses_list_route_as_recommendation():
    result = finalize_answer_mode(
        turn_info={"action": "retrieve_list", "answer_mode_hint": "recommendation"},
        execution_plan={"action": "retrieve_list"},
        query_plan={"route_type": "list"},
        retrieval_quality={"enough_evidence": True},
    )

    assert result == "recommendation"


def test_finalize_answer_mode_keeps_substitution_from_turn_understanding():
    result = finalize_answer_mode(
        turn_info={"action": "substitution", "answer_mode_hint": "substitution"},
        execution_plan={"action": "retrieve_detail"},
        query_plan={"route_type": "detail", "filters": {"content_type": "steps"}},
        retrieval_quality={"enough_evidence": True},
    )

    assert result == "substitution"


def test_finalize_answer_mode_keeps_comparison_from_turn_understanding():
    result = finalize_answer_mode(
        turn_info={"action": "compare", "answer_mode_hint": "comparison"},
        execution_plan={"action": "retrieve_detail"},
        query_plan={"route_type": "detail"},
        retrieval_quality={"enough_evidence": True},
    )

    assert result == "comparison"


def test_finalize_answer_mode_defaults_retrieve_detail_to_recipe_detail():
    result = finalize_answer_mode(
        turn_info={"action": "retrieve_detail", "answer_mode_hint": "recipe_detail"},
        execution_plan={"action": "retrieve_detail"},
        query_plan={"route_type": "detail"},
        retrieval_quality={"enough_evidence": True},
    )

    assert result == "recipe_detail"


def test_finalize_answer_mode_keeps_history_answer_even_without_hint():
    result = finalize_answer_mode(
        turn_info={"action": "history_answer"},
        execution_plan={"action": "history_answer"},
        query_plan={"route_type": "direct"},
        retrieval_quality=None,
    )

    assert result == "history_based"
```

- [ ] **Step 2: Run answer-mode tests and verify they fail**

Run:

```bash
cd code/C8
pytest tests/test_context_packer.py::test_finalize_answer_mode_uses_list_route_as_recommendation tests/test_context_packer.py::test_finalize_answer_mode_keeps_substitution_from_turn_understanding tests/test_context_packer.py::test_finalize_answer_mode_keeps_comparison_from_turn_understanding tests/test_context_packer.py::test_finalize_answer_mode_defaults_retrieve_detail_to_recipe_detail tests/test_context_packer.py::test_finalize_answer_mode_keeps_history_answer_even_without_hint -q
```

Expected:

- FAIL because `finalize_answer_mode` does not exist.

- [ ] **Step 3: Implement answer-mode finalization**

Append to `code/C8/rag_modules/context_packer.py`:

```python
def finalize_answer_mode(
    *,
    turn_info: dict,
    execution_plan: dict,
    query_plan: dict,
    retrieval_quality: dict | None,
) -> str:
    """Finalize answer mode without letting document shape change the user task."""
    action = turn_info.get("action") or execution_plan.get("action")
    hint = turn_info.get("answer_mode_hint") or execution_plan.get("answer_mode")
    route_type = query_plan.get("route_type")

    if action == "smalltalk" or hint == "safe_direct":
        return "safe_direct"
    if action == "history_answer" or hint == "history_based":
        return "history_based"
    if action == "substitution" or hint == "substitution":
        return "substitution"
    if action == "compare" or hint == "comparison":
        return "comparison"
    if hint == "troubleshooting":
        return "troubleshooting"
    if action == "retrieve_list" or route_type == "list":
        return "recommendation"
    return "recipe_detail"
```

- [ ] **Step 4: Run answer-mode tests and verify they pass**

Run:

```bash
cd code/C8
pytest tests/test_context_packer.py::test_finalize_answer_mode_uses_list_route_as_recommendation tests/test_context_packer.py::test_finalize_answer_mode_keeps_substitution_from_turn_understanding tests/test_context_packer.py::test_finalize_answer_mode_keeps_comparison_from_turn_understanding tests/test_context_packer.py::test_finalize_answer_mode_defaults_retrieve_detail_to_recipe_detail tests/test_context_packer.py::test_finalize_answer_mode_keeps_history_answer_even_without_hint -q
```

Expected:

- PASS.

- [ ] **Step 5: Commit**

```bash
git add code/C8/rag_modules/context_packer.py code/C8/tests/test_context_packer.py
git commit -m "feat: finalize answer modes for context packing"
```

---

## Task 3: Add ContextPacker Section Selection And Trimming

**Files:**
- Modify: `code/C8/rag_modules/context_packer.py`
- Modify: `code/C8/tests/test_context_packer.py`

- [ ] **Step 1: Add ContextPacker tests**

Append to `code/C8/tests/test_context_packer.py`:

```python
from rag_modules.context_packer import ContextPacker


def test_context_packer_ingredients_question_selects_ingredient_section_not_full_document():
    parent = _parent_doc()
    packer = ContextPacker(max_chars_total=2000, max_chars_per_doc=800, max_docs=5)

    pack = packer.build_context_pack(
        query="宫保鸡丁需要什么食材",
        retrieval_result={"chunks": [Document(page_content="chunk", metadata={"dish_name": "宫保鸡丁"})]},
        query_plan={"route_type": "detail", "filters": {"content_type": "ingredients"}},
        execution_plan={"action": "retrieve_detail"},
        turn_info={"action": "retrieve_detail", "answer_mode_hint": "recipe_detail"},
        parent_docs=[parent],
    )

    assert pack["answer_mode"] == "recipe_detail"
    assert pack["content_type"] == "ingredients"
    assert len(pack["context_docs"]) == 1
    assert "## 必备原料和工具" in pack["context_docs"][0].page_content
    assert "## 操作" not in pack["context_docs"][0].page_content
    assert pack["selected_sections"][0]["section_type"] == "ingredients"
    assert pack["parent_docs"] == [parent]


def test_context_packer_steps_question_preserves_ordered_procedure_context():
    parent = _parent_doc()
    packer = ContextPacker(max_chars_total=2000, max_chars_per_doc=1000, max_docs=5)

    pack = packer.build_context_pack(
        query="宫保鸡丁怎么做",
        retrieval_result={"chunks": []},
        query_plan={"route_type": "detail", "filters": {"content_type": "steps"}},
        execution_plan={"action": "retrieve_detail"},
        turn_info={"action": "retrieve_detail", "answer_mode_hint": "recipe_detail"},
        parent_docs=[parent],
    )

    text = pack["context_docs"][0].page_content
    assert "## 操作" in text
    assert text.index("### 腌制鸡肉") < text.index("### 炒制")
    assert pack["selected_sections"][0]["section_type"] == "steps"


def test_context_packer_preserves_parent_ranking_metadata_on_section_docs():
    parent = Document(
        page_content=RECIPE_MARKDOWN,
        metadata={"dish_name": "宫保鸡丁", "parent_id": "parent-ranked", "rrf_score": 0.87},
    )
    packer = ContextPacker(max_chars_total=2000, max_chars_per_doc=1000, max_docs=5)

    pack = packer.build_context_pack(
        query="宫保鸡丁怎么做",
        retrieval_result={"chunks": []},
        query_plan={"route_type": "detail", "filters": {"content_type": "steps"}},
        execution_plan={"action": "retrieve_detail"},
        turn_info={"action": "retrieve_detail", "answer_mode_hint": "recipe_detail"},
        parent_docs=[parent],
    )

    assert pack["context_docs"][0].metadata["rrf_score"] == 0.87
    assert pack["context_docs"][0].metadata["parent_id"] == "parent-ranked"
    assert pack["context_docs"][0].metadata["context_pack_mode"] == "section"


def test_context_packer_substitution_selects_ingredients_and_tips():
    parent = _parent_doc()
    packer = ContextPacker(max_chars_total=2000, max_chars_per_doc=800, max_docs=5)

    pack = packer.build_context_pack(
        query="这个能不放辣椒吗",
        retrieval_result={"chunks": []},
        query_plan={"route_type": "detail", "filters": {}},
        execution_plan={"action": "retrieve_detail"},
        turn_info={"action": "substitution", "answer_mode_hint": "substitution"},
        parent_docs=[parent],
    )

    combined = "\n".join(doc.page_content for doc in pack["context_docs"])
    assert pack["answer_mode"] == "substitution"
    assert "## 必备原料和工具" in combined
    assert "## 附加内容" in combined
    assert {section["section_type"] for section in pack["selected_sections"]} == {"ingredients", "tips"}


def test_context_packer_recommendation_uses_compact_parent_summaries():
    parent = _parent_doc()
    packer = ContextPacker(max_chars_total=2000, max_chars_per_doc=300, max_docs=5)

    pack = packer.build_context_pack(
        query="推荐几个鸡肉菜",
        retrieval_result={"chunks": []},
        query_plan={"route_type": "list", "filters": {}},
        execution_plan={"action": "retrieve_list"},
        turn_info={"action": "retrieve_list", "answer_mode_hint": "recommendation"},
        parent_docs=[parent],
    )

    assert pack["answer_mode"] == "recommendation"
    assert len(pack["context_docs"]) == 1
    assert pack["context_docs"][0].metadata["context_pack_mode"] == "summary"
    assert len(pack["context_docs"][0].page_content) <= 300
```

- [ ] **Step 2: Run ContextPacker tests and verify they fail**

Run:

```bash
cd code/C8
pytest tests/test_context_packer.py::test_context_packer_ingredients_question_selects_ingredient_section_not_full_document tests/test_context_packer.py::test_context_packer_steps_question_preserves_ordered_procedure_context tests/test_context_packer.py::test_context_packer_preserves_parent_ranking_metadata_on_section_docs tests/test_context_packer.py::test_context_packer_substitution_selects_ingredients_and_tips tests/test_context_packer.py::test_context_packer_recommendation_uses_compact_parent_summaries -q
```

Expected:

- FAIL because `ContextPacker` does not exist.

- [ ] **Step 3: Implement ContextPacker**

Append to `code/C8/rag_modules/context_packer.py`:

```python
SECTION_PREFERENCES = {
    "ingredients": (["ingredients"], ["calculation", "tips"]),
    "steps": (["steps"], ["ingredients", "tips"]),
    "tips": (["tips"], ["steps"]),
    "introduction": (["introduction"], ["tips"]),
    "calculation": (["calculation"], ["ingredients"]),
    "substitution": (["ingredients", "tips"], ["steps"]),
    "troubleshooting": (["tips", "steps"], ["ingredients"]),
}


class ContextPacker:
    """Build generation-facing packed context from retrieved chunks and parent docs."""

    def __init__(
        self,
        *,
        max_chars_total: int = 2400,
        max_chars_per_doc: int = 1200,
        max_docs: int = 5,
    ):
        self.max_chars_total = max_chars_total
        self.max_chars_per_doc = max_chars_per_doc
        self.max_docs = max_docs

    def build_context_pack(
        self,
        *,
        query: str,
        retrieval_result: dict,
        query_plan: dict,
        execution_plan: dict,
        turn_info: dict,
        parent_docs: list[Document],
    ) -> dict:
        answer_mode = finalize_answer_mode(
            turn_info=turn_info,
            execution_plan=execution_plan,
            query_plan=query_plan,
            retrieval_quality=(retrieval_result or {}).get("quality"),
        )
        content_type = self._content_type(query_plan, answer_mode)

        if answer_mode == "recommendation":
            context_docs, selected_sections, trimmed = self._build_recommendation_context(parent_docs)
        else:
            context_docs, selected_sections, trimmed = self._build_section_context(
                parent_docs=parent_docs,
                content_type=content_type,
                answer_mode=answer_mode,
            )

        fallback_to_chunks = False
        if not context_docs:
            context_docs = self._chunk_fallback((retrieval_result or {}).get("chunks") or [])
            selected_sections = []
            fallback_to_chunks = bool(context_docs)
            trimmed = False

        answer_mode_initial = turn_info.get("answer_mode_hint") or execution_plan.get("answer_mode")
        return {
            "answer_mode": answer_mode,
            "context_docs": context_docs,
            "parent_docs": list(parent_docs or []),
            "selected_sections": selected_sections,
            "content_type": content_type,
            "trace": {
                "input_chunk_count": len((retrieval_result or {}).get("chunks") or []),
                "parent_doc_count": len(parent_docs or []),
                "selected_section_count": len(selected_sections),
                "answer_mode_initial": answer_mode_initial,
                "answer_mode_final": answer_mode,
                "answer_mode_source": "turn_info" if turn_info.get("answer_mode_hint") else "execution_plan",
                "content_type": content_type,
                "trimmed": trimmed,
                "fallback_to_chunks": fallback_to_chunks,
            },
        }

    def _content_type(self, query_plan: dict, answer_mode: str) -> str:
        filters = query_plan.get("filters") or {}
        if answer_mode in {"substitution", "troubleshooting"}:
            return answer_mode
        return filters.get("content_type") or ("steps" if answer_mode == "recipe_detail" else answer_mode)

    def _build_section_context(
        self,
        *,
        parent_docs: list[Document],
        content_type: str,
        answer_mode: str,
    ) -> tuple[list[Document], list[dict], bool]:
        preferred, secondary = SECTION_PREFERENCES.get(
            content_type,
            SECTION_PREFERENCES.get(answer_mode, (["steps"], ["ingredients", "tips"])),
        )
        sections = []
        for parent_doc in parent_docs or []:
            sections.extend(extract_markdown_sections(parent_doc))

        chosen = [section for section in sections if section["section_type"] in preferred]
        if not chosen:
            chosen = [section for section in sections if section["section_type"] in secondary]
        elif answer_mode in {"substitution", "troubleshooting"}:
            chosen.extend(section for section in sections if section["section_type"] in secondary)

        return self._sections_to_docs(chosen)

    def _sections_to_docs(self, sections: list[dict]) -> tuple[list[Document], list[dict], bool]:
        context_docs: list[Document] = []
        selected: list[dict] = []
        total = 0
        trimmed_any = False
        for section in sections:
            if len(context_docs) >= self.max_docs or total >= self.max_chars_total:
                break
            text, trimmed = self._trim_text(section["text"], self.max_chars_per_doc)
            remaining = self.max_chars_total - total
            if len(text) > remaining:
                text, trimmed = self._trim_text(text, remaining)
            if not text:
                continue
            trimmed_any = trimmed_any or trimmed
            total += len(text)
            context_docs.append(
                Document(
                    page_content=text,
                    metadata={
                        **(section.get("source_metadata") or {}),
                        "dish_name": section.get("dish_name"),
                        "parent_id": section.get("parent_id"),
                        "section_title": section.get("title"),
                        "section_type": section.get("section_type"),
                        "context_pack_mode": "section",
                    },
                )
            )
            selected.append(
                {
                    "dish_name": section.get("dish_name"),
                    "section_title": section.get("title"),
                    "section_type": section.get("section_type"),
                    "source_parent_id": section.get("parent_id"),
                    "token_estimate": max(1, len(text) // 4),
                }
            )
        return context_docs, selected, trimmed_any

    def _build_recommendation_context(self, parent_docs: list[Document]) -> tuple[list[Document], list[dict], bool]:
        docs: list[Document] = []
        selected: list[dict] = []
        total = 0
        trimmed_any = False
        for parent_doc in (parent_docs or [])[: self.max_docs]:
            dish_name = (parent_doc.metadata or {}).get("dish_name")
            summary = self._summary_text(parent_doc)
            summary, trimmed = self._trim_text(summary, min(self.max_chars_per_doc, self.max_chars_total - total))
            if not summary:
                continue
            total += len(summary)
            trimmed_any = trimmed_any or trimmed
            docs.append(
                Document(
                    page_content=summary,
                    metadata={**(parent_doc.metadata or {}), "context_pack_mode": "summary"},
                )
            )
            selected.append(
                {
                    "dish_name": dish_name,
                    "section_title": "summary",
                    "section_type": "summary",
                    "source_parent_id": (parent_doc.metadata or {}).get("parent_id"),
                    "token_estimate": max(1, len(summary) // 4),
                }
            )
            if total >= self.max_chars_total:
                break
        return docs, selected, trimmed_any

    def _summary_text(self, parent_doc: Document) -> str:
        dish_name = (parent_doc.metadata or {}).get("dish_name", "未知菜品")
        sections = extract_markdown_sections(parent_doc)
        intro = next((section["text"] for section in sections if section["section_type"] == "introduction"), "")
        ingredients = next((section["text"] for section in sections if section["section_type"] == "ingredients"), "")
        return f"# {dish_name}\n{intro}\n{ingredients}".strip()

    def _chunk_fallback(self, chunks: list[Document]) -> list[Document]:
        docs = []
        for chunk in chunks[: self.max_docs]:
            text, _ = self._trim_text(chunk.page_content or "", self.max_chars_per_doc)
            docs.append(Document(page_content=text, metadata={**(chunk.metadata or {}), "context_pack_mode": "chunk_fallback"}))
        return docs

    def _trim_text(self, text: str, limit: int) -> tuple[str, bool]:
        if limit <= 0:
            return "", True
        if len(text) <= limit:
            return text, False
        cut = text[:limit].rstrip()
        last_newline = cut.rfind("\n")
        if last_newline > limit * 0.6:
            cut = cut[:last_newline].rstrip()
        return cut, True
```

- [ ] **Step 4: Run ContextPacker tests and verify they pass**

Run:

```bash
cd code/C8
pytest tests/test_context_packer.py::test_context_packer_ingredients_question_selects_ingredient_section_not_full_document tests/test_context_packer.py::test_context_packer_steps_question_preserves_ordered_procedure_context tests/test_context_packer.py::test_context_packer_preserves_parent_ranking_metadata_on_section_docs tests/test_context_packer.py::test_context_packer_substitution_selects_ingredients_and_tips tests/test_context_packer.py::test_context_packer_recommendation_uses_compact_parent_summaries -q
```

Expected:

- PASS.

- [ ] **Step 5: Commit**

```bash
git add code/C8/rag_modules/context_packer.py code/C8/tests/test_context_packer.py
git commit -m "feat: pack recipe context by answer mode"
```

---

## Task 4: Add Trimming And Fallback Context Tests

**Files:**
- Modify: `code/C8/tests/test_context_packer.py`
- Modify: `code/C8/rag_modules/context_packer.py`

- [ ] **Step 1: Add trimming and fallback tests**

Append to `code/C8/tests/test_context_packer.py`:

```python
def test_context_packer_trims_selected_sections_with_trace():
    long_steps = "# 长菜的做法\n\n## 操作\n\n" + "\n".join(f"- 第 {i} 步操作说明很长" for i in range(50))
    parent = Document(page_content=long_steps, metadata={"dish_name": "长菜", "parent_id": "p-long"})
    packer = ContextPacker(max_chars_total=180, max_chars_per_doc=120, max_docs=5)

    pack = packer.build_context_pack(
        query="长菜怎么做",
        retrieval_result={"chunks": []},
        query_plan={"route_type": "detail", "filters": {"content_type": "steps"}},
        execution_plan={"action": "retrieve_detail"},
        turn_info={"action": "retrieve_detail", "answer_mode_hint": "recipe_detail"},
        parent_docs=[parent],
    )

    assert len(pack["context_docs"][0].page_content) <= 120
    assert pack["trace"]["trimmed"] is True
    assert pack["context_docs"][0].metadata["section_type"] == "steps"


def test_context_packer_falls_back_to_retrieved_chunks_when_parent_sections_missing():
    chunk = Document(page_content="检索块里的做法摘要", metadata={"dish_name": "未知菜", "content_type": "steps"})
    packer = ContextPacker(max_chars_total=2000, max_chars_per_doc=500, max_docs=5)

    pack = packer.build_context_pack(
        query="未知菜怎么做",
        retrieval_result={"chunks": [chunk]},
        query_plan={"route_type": "detail", "filters": {"content_type": "steps"}},
        execution_plan={"action": "retrieve_detail"},
        turn_info={"action": "retrieve_detail", "answer_mode_hint": "recipe_detail"},
        parent_docs=[],
    )

    assert len(pack["context_docs"]) == 1
    assert pack["context_docs"][0].page_content == "检索块里的做法摘要"
    assert pack["context_docs"][0].metadata["context_pack_mode"] == "chunk_fallback"
    assert pack["context_docs"][0].metadata["dish_name"] == "未知菜"
    assert pack["answer_mode"] == "recipe_detail"
    assert pack["parent_docs"] == []
    assert pack["selected_sections"] == []
    assert pack["trace"]["parent_doc_count"] == 0
    assert pack["trace"]["selected_section_count"] == 0
    assert pack["trace"]["fallback_to_chunks"] is True
```

- [ ] **Step 2: Run trimming and fallback tests**

Run:

```bash
cd code/C8
pytest tests/test_context_packer.py::test_context_packer_trims_selected_sections_with_trace tests/test_context_packer.py::test_context_packer_falls_back_to_retrieved_chunks_when_parent_sections_missing -q
```

Expected:

- PASS with the Task 3 implementation.

- [ ] **Step 3: Run all context packer unit tests**

Run:

```bash
cd code/C8
pytest tests/test_context_packer.py -q
```

Expected:

- PASS.

- [ ] **Step 4: Commit**

```bash
git add code/C8/rag_modules/context_packer.py code/C8/tests/test_context_packer.py
git commit -m "test: cover context trimming and chunk fallback"
```

---

## Task 5: Wire ContextPack Into `ask_question()`

**Files:**
- Modify: `code/C8/config.py`
- Modify: `code/C8/main.py`
- Modify: `code/C8/tests/test_conversation_state.py`
- Modify: `code/C8/tests/test_state_hardening.py`

- [ ] **Step 1: Add context-pack budget fields to `RAGConfig`**

In `code/C8/config.py`, extend `RAGConfig`:

```python
    # Context packing configuration
    context_pack_max_chars_total: int = 2400
    context_pack_max_chars_per_doc: int = 1200
    context_pack_max_docs: int = 5
```

In `validate()`, add:

```python
        if self.context_pack_max_chars_total <= 0:
            raise ValueError("context_pack_max_chars_total 必须大于 0")
        if self.context_pack_max_chars_per_doc <= 0:
            raise ValueError("context_pack_max_chars_per_doc 必须大于 0")
        if self.context_pack_max_docs <= 0:
            raise ValueError("context_pack_max_docs 必须大于 0")
```

In `to_dict()`, add:

```python
            "context_pack_max_chars_total": self.context_pack_max_chars_total,
            "context_pack_max_chars_per_doc": self.context_pack_max_chars_per_doc,
            "context_pack_max_docs": self.context_pack_max_docs,
```

In `from_env()`, add:

```python
            "context_pack_max_chars_total": override_values.get(
                "context_pack_max_chars_total",
                _parse_int(
                    env.get("RAG_CONTEXT_PACK_MAX_CHARS_TOTAL"),
                    defaults.context_pack_max_chars_total,
                ),
            ),
            "context_pack_max_chars_per_doc": override_values.get(
                "context_pack_max_chars_per_doc",
                _parse_int(
                    env.get("RAG_CONTEXT_PACK_MAX_CHARS_PER_DOC"),
                    defaults.context_pack_max_chars_per_doc,
                ),
            ),
            "context_pack_max_docs": override_values.get(
                "context_pack_max_docs",
                _parse_int(
                    env.get("RAG_CONTEXT_PACK_MAX_DOCS"),
                    defaults.context_pack_max_docs,
                ),
            ),
```

- [ ] **Step 2: Add chat-path integration tests**

Append to `code/C8/tests/test_conversation_state.py`:

```python
def test_chat_path_builds_context_pack_before_detail_generation(monkeypatch):
    from main import RecipeRAGSystem
    from langchain_core.documents import Document

    system = RecipeRAGSystem.__new__(RecipeRAGSystem)
    child = Document(page_content="child", metadata={"dish_name": "蛋炒饭", "parent_id": "p1"})
    parent = Document(page_content="# 蛋炒饭的做法\n\n## 必备原料和工具\n\n- 鸡蛋\n\n## 操作\n\n- 炒饭", metadata={"dish_name": "蛋炒饭", "parent_id": "p1"})
    packed = Document(page_content="## 操作\n\n- 炒饭", metadata={"dish_name": "蛋炒饭", "section_type": "steps"})
    generation_calls = []
    writes = []

    class FakeGeneration:
        conversation_manager = None

        def query_router(self, question):
            return {"type": "detail", "filters": {"content_type": "steps"}, "dish_name": "蛋炒饭", "confidence": 1.0}

        def generate_step_by_step_answer(self, question, context_docs, content_type=None):
            generation_calls.append((question, context_docs, content_type))
            return "蛋炒饭做法"

    class FakeRetrievalExecutor:
        def execute(self, query_plan):
            return {
                "chunks": [child],
                "quality": {"enough_evidence": True, "quality_reason": "ok", "fallback_used": False, "relaxed_filter": False, "candidate_count": 1, "selected_dishes": ["蛋炒饭"]},
                "low_evidence": None,
                "trace": {"strategy": "primary"},
            }

    class FakeData:
        def get_parent_documents(self, chunks, target_dish_name=None):
            assert chunks == [child]
            assert target_dish_name == "蛋炒饭"
            return [parent]

    class FakeContextPacker:
        def build_context_pack(self, **kwargs):
            assert kwargs["parent_docs"] == [parent]
            return {
                "answer_mode": "recipe_detail",
                "context_docs": [packed],
                "parent_docs": [parent],
                "selected_sections": [{"section_type": "steps"}],
                "content_type": "steps",
                "trace": {"selected_section_count": 1},
            }

    system.retrieval_module = type("Retrieval", (), {"extract_filters_from_query": lambda self, question: {}})()
    system.retrieval_executor = FakeRetrievalExecutor()
    system.context_packer = FakeContextPacker()
    system.generation_module = FakeGeneration()
    system.data_module = FakeData()
    system.config = type("Config", (), {
        "top_k": 3,
        "context_pack_max_chars_total": 2400,
        "context_pack_max_chars_per_doc": 1200,
        "context_pack_max_docs": 5,
    })()
    system._latest_parent_docs = []
    system.last_execution_result = None

    monkeypatch.setattr(system, "_apply_resolved_target_to_query_plan", lambda query_plan, resolution: query_plan)
    monkeypatch.setattr(system, "_write_conversation_turn", lambda **kwargs: writes.append(kwargs))

    answer = system.ask_question("蛋炒饭怎么做", stream=False, session_id="context-pack-detail")

    assert answer == "蛋炒饭做法"
    assert generation_calls[0][1] == [packed]
    assert generation_calls[0][2] == "steps"
    assert system._latest_parent_docs == [parent]
    assert writes[-1]["execution_result"]["context_pack_trace"] == {"selected_section_count": 1}


def test_chat_path_builds_context_pack_before_list_generation(monkeypatch):
    from main import RecipeRAGSystem
    from langchain_core.documents import Document

    system = RecipeRAGSystem.__new__(RecipeRAGSystem)
    child = Document(page_content="child", metadata={"dish_name": "番茄炒蛋", "parent_id": "p1"})
    parent = Document(page_content="# 番茄炒蛋的做法\n\n## 必备原料和工具\n\n- 番茄\n- 鸡蛋", metadata={"dish_name": "番茄炒蛋", "parent_id": "p1"})
    packed = Document(page_content="# 番茄炒蛋\n- 番茄\n- 鸡蛋", metadata={"dish_name": "番茄炒蛋", "context_pack_mode": "summary"})
    generation_calls = []
    writes = []

    class FakeGeneration:
        conversation_manager = None

        def query_router(self, question):
            return {"type": "list", "filters": {}, "dish_name": None, "confidence": 1.0}

        def generate_list_answer(self, question, context_docs):
            generation_calls.append((question, context_docs))
            return "1. 番茄炒蛋"

    class FakeRetrievalExecutor:
        def execute(self, query_plan):
            return {
                "chunks": [child],
                "quality": {"enough_evidence": True, "quality_reason": "ok", "fallback_used": False, "relaxed_filter": False, "candidate_count": 1, "selected_dishes": ["番茄炒蛋"]},
                "low_evidence": None,
                "trace": {"strategy": "primary"},
            }

    class FakeData:
        def get_parent_documents(self, chunks, target_dish_name=None):
            assert target_dish_name is None
            return [parent]

    class FakeContextPacker:
        def build_context_pack(self, **kwargs):
            return {
                "answer_mode": "recommendation",
                "context_docs": [packed],
                "parent_docs": [parent],
                "selected_sections": [{"section_type": "summary"}],
                "content_type": "recommendation",
                "trace": {"selected_section_count": 1},
            }

    system.retrieval_module = type("Retrieval", (), {"extract_filters_from_query": lambda self, question: {}})()
    system.retrieval_executor = FakeRetrievalExecutor()
    system.context_packer = FakeContextPacker()
    system.generation_module = FakeGeneration()
    system.data_module = FakeData()
    system.config = type("Config", (), {
        "top_k": 3,
        "context_pack_max_chars_total": 2400,
        "context_pack_max_chars_per_doc": 1200,
        "context_pack_max_docs": 5,
    })()
    system._latest_parent_docs = []
    system.last_execution_result = None

    monkeypatch.setattr(system, "_apply_resolved_target_to_query_plan", lambda query_plan, resolution: query_plan)
    monkeypatch.setattr(system, "_write_conversation_turn", lambda **kwargs: writes.append(kwargs))

    answer = system.ask_question("今天吃什么", stream=False, session_id="context-pack-list")

    assert answer == "1. 番茄炒蛋"
    assert generation_calls[0][1] == [packed]
    assert system._latest_parent_docs == [parent]
    assert writes[-1]["execution_result"]["context_pack_trace"] == {"selected_section_count": 1}


def test_chat_path_builds_context_pack_for_streaming_detail_generation(monkeypatch):
    from main import RecipeRAGSystem
    from langchain_core.documents import Document

    system = RecipeRAGSystem.__new__(RecipeRAGSystem)
    child = Document(page_content="child", metadata={"dish_name": "蛋炒饭", "parent_id": "p1"})
    parent = Document(page_content="# 蛋炒饭的做法\n\n## 操作\n\n- 炒饭", metadata={"dish_name": "蛋炒饭", "parent_id": "p1"})
    packed = Document(page_content="## 操作\n\n- 炒饭", metadata={"dish_name": "蛋炒饭", "section_type": "steps"})
    writes = []

    class FakeGeneration:
        conversation_manager = None

        def query_router(self, question):
            return {"type": "detail", "filters": {"content_type": "steps"}, "dish_name": "蛋炒饭", "confidence": 1.0}

        def generate_step_by_step_answer_stream(self, question, context_docs, content_type=None):
            assert context_docs == [packed]
            assert content_type == "steps"
            yield "蛋炒饭"
            yield "做法"

    class FakeRetrievalExecutor:
        def execute(self, query_plan):
            return {
                "chunks": [child],
                "quality": {"enough_evidence": True, "quality_reason": "ok", "fallback_used": False, "relaxed_filter": False, "candidate_count": 1, "selected_dishes": ["蛋炒饭"]},
                "low_evidence": None,
                "trace": {"strategy": "primary"},
            }

    class FakeData:
        def get_parent_documents(self, chunks, target_dish_name=None):
            assert target_dish_name == "蛋炒饭"
            return [parent]

    class FakeContextPacker:
        def build_context_pack(self, **kwargs):
            return {
                "answer_mode": "recipe_detail",
                "context_docs": [packed],
                "parent_docs": [parent],
                "selected_sections": [{"section_type": "steps"}],
                "content_type": "steps",
                "trace": {"selected_section_count": 1},
            }

    system.retrieval_module = type("Retrieval", (), {"extract_filters_from_query": lambda self, question: {}})()
    system.retrieval_executor = FakeRetrievalExecutor()
    system.context_packer = FakeContextPacker()
    system.generation_module = FakeGeneration()
    system.data_module = FakeData()
    system.config = type("Config", (), {
        "top_k": 3,
        "context_pack_max_chars_total": 2400,
        "context_pack_max_chars_per_doc": 1200,
        "context_pack_max_docs": 5,
    })()
    system._latest_parent_docs = []
    system.last_execution_result = None

    monkeypatch.setattr(system, "_apply_resolved_target_to_query_plan", lambda query_plan, resolution: query_plan)
    monkeypatch.setattr(system, "_write_conversation_turn", lambda **kwargs: writes.append(kwargs))

    stream = system.ask_question("蛋炒饭怎么做", stream=True, session_id="context-pack-stream")
    assert "".join(stream) == "蛋炒饭做法"
    assert writes[-1]["execution_result"]["context_pack_trace"] == {"selected_section_count": 1}


def test_chat_path_parent_expansion_allows_none_target_for_detail_without_dish(monkeypatch):
    from main import RecipeRAGSystem
    from langchain_core.documents import Document

    system = RecipeRAGSystem.__new__(RecipeRAGSystem)
    child = Document(page_content="child", metadata={"dish_name": "番茄炒蛋", "parent_id": "p1"})
    parent = Document(page_content="# 番茄炒蛋的做法\n\n## 操作\n\n- 炒蛋", metadata={"dish_name": "番茄炒蛋", "parent_id": "p1"})

    class FakeGeneration:
        conversation_manager = None

        def query_router(self, question):
            return {"type": "detail", "filters": {"content_type": "steps"}, "dish_name": None, "confidence": 0.7}

        def generate_step_by_step_answer(self, question, context_docs, content_type=None):
            return "可以这样做"

    class FakeRetrievalExecutor:
        def execute(self, query_plan):
            return {
                "chunks": [child],
                "quality": {"enough_evidence": True, "quality_reason": "ok", "fallback_used": False, "relaxed_filter": False, "candidate_count": 1, "selected_dishes": ["番茄炒蛋"]},
                "low_evidence": None,
                "trace": {"strategy": "primary"},
            }

    class FakeData:
        def get_parent_documents(self, chunks, target_dish_name=None):
            assert target_dish_name is None
            return [parent]

    class FakeContextPacker:
        def build_context_pack(self, **kwargs):
            return {
                "answer_mode": "recipe_detail",
                "context_docs": [parent],
                "parent_docs": [parent],
                "selected_sections": [{"section_type": "steps"}],
                "content_type": "steps",
                "trace": {"selected_section_count": 1},
            }

    system.retrieval_module = type("Retrieval", (), {"extract_filters_from_query": lambda self, question: {}})()
    system.retrieval_executor = FakeRetrievalExecutor()
    system.context_packer = FakeContextPacker()
    system.generation_module = FakeGeneration()
    system.data_module = FakeData()
    system.config = type("Config", (), {
        "top_k": 3,
        "context_pack_max_chars_total": 2400,
        "context_pack_max_chars_per_doc": 1200,
        "context_pack_max_docs": 5,
    })()
    system._latest_parent_docs = []
    system.last_execution_result = None

    monkeypatch.setattr(
        system,
        "_build_query_plan",
        lambda question, session_id: {
            "route_type": "detail",
            "filters": {"content_type": "steps"},
            "dish_name": None,
            "entities": {"dish_name": None, "filters": {"content_type": "steps"}},
            "confidence": 0.7,
        },
    )
    monkeypatch.setattr(system, "_apply_resolved_target_to_query_plan", lambda query_plan, resolution: query_plan)
    monkeypatch.setattr(system, "_write_conversation_turn", lambda **kwargs: None)

    assert system.ask_question("这个怎么做", stream=False, session_id="none-target-detail") == "可以这样做"
```

- [ ] **Step 3: Run integration tests and verify they fail**

Run:

```bash
cd code/C8
pytest tests/test_conversation_state.py::test_chat_path_builds_context_pack_before_detail_generation tests/test_conversation_state.py::test_chat_path_builds_context_pack_before_list_generation tests/test_conversation_state.py::test_chat_path_builds_context_pack_for_streaming_detail_generation tests/test_conversation_state.py::test_chat_path_parent_expansion_allows_none_target_for_detail_without_dish -q
```

Expected:

- FAIL because `ask_question()` still passes raw retrieved chunks to generation helpers and generation helpers still expand parents internally.

- [ ] **Step 4: Import and instantiate `ContextPacker`**

In `code/C8/main.py`, add:

```python
from rag_modules.context_packer import ContextPacker
```

When `RecipeRAGSystem` initializes runtime modules after retrieval module creation, add:

```python
self.context_packer = ContextPacker(
    max_chars_total=self.config.context_pack_max_chars_total,
    max_chars_per_doc=self.config.context_pack_max_chars_per_doc,
    max_docs=self.config.context_pack_max_docs,
)
```

Inside `ask_question()`, after the retrieval executor guard, add:

```python
if not hasattr(self, "context_packer") or self.context_packer is None:
    self.context_packer = ContextPacker(
        max_chars_total=self.config.context_pack_max_chars_total,
        max_chars_per_doc=self.config.context_pack_max_chars_per_doc,
        max_docs=self.config.context_pack_max_docs,
    )
```

- [ ] **Step 5: Build context pack before generation**

In `RecipeRAGSystem.ask_question()`, after the `low_evidence` return block and before list/detail generation, add:

```python
parent_docs = self.data_module.get_parent_documents(
    relevant_chunks,
    target_dish_name=dish_name if route_type != "list" else None,
)
context_pack = self.context_packer.build_context_pack(
    query=rewritten_question,
    retrieval_result=retrieval_result,
    query_plan=query_plan,
    execution_plan=execution_plan,
    turn_info=turn_info,
    parent_docs=parent_docs,
)
self._latest_parent_docs = list(context_pack["parent_docs"])
```

Contract:

- list routes always pass `target_dish_name=None`;
- detail/basic routes pass the already-resolved `dish_name`;
- if that resolved `dish_name` is `None`, parent expansion must receive `target_dish_name=None`;
- do not infer a new dish name inside the parent-expansion call.

- [ ] **Step 6: Pass context pack into generation helpers**

Replace:

```python
answer = self._generate_list_response(rewritten_question, session_id, relevant_chunks)
```

with:

```python
answer = self._generate_list_response(rewritten_question, context_pack)
```

Replace the detail call:

```python
answer = self._generate_detail_response(
    rewritten_question,
    stream,
    session_id,
    route_type,
    filters,
    entities,
    dish_name,
    relevant_chunks,
)
```

with:

```python
answer = self._generate_detail_response(
    rewritten_question,
    stream,
    route_type,
    dish_name,
    context_pack,
)
```

After each successful `_build_execution_result(...)`, add:

```python
execution_result["context_pack_trace"] = context_pack["trace"]
execution_result["answer_mode"] = context_pack["answer_mode"]
```

- [ ] **Step 7: Change generation helper signatures to consume context packs**

Change `_generate_list_response` in `code/C8/main.py` to. This helper must not assign `_latest_parent_docs`; the main chain already set it from `context_pack["parent_docs"]`. It must log packed context docs because those are what generation actually receives.

```python
    def _generate_list_response(self, question: str, context_pack: dict):
        """生成列表类回答。"""
        print("生成菜品列表...")
        context_docs = list(context_pack["context_docs"])
        doc_names = []
        for doc in context_docs:
            dish_name = doc.metadata.get("dish_name", "未知菜品")
            mode = doc.metadata.get("context_pack_mode") or doc.metadata.get("section_type") or "packed"
            doc_names.append(f"{dish_name}/{mode}")

        if doc_names:
            print(f"传入生成的上下文: {', '.join(doc_names)}")

        return self.generation_module.generate_list_answer(question, context_docs)
```

Change `_generate_detail_response` to remove the dead `filters`, `session_id`, and `entities` parameters and consume only `context_pack` as the content-type source:

```python
    def _generate_detail_response(
        self,
        question: str,
        stream: bool,
        route_type: str,
        dish_name: str,
        context_pack: dict,
    ):
        """生成详细或通用问答结果。"""
        context_docs = list(context_pack["context_docs"])
        doc_names = []
        for doc in context_docs:
            current_dish_name = doc.metadata.get("dish_name", "未知菜品")
            mode = doc.metadata.get("context_pack_mode") or doc.metadata.get("section_type") or "packed"
            doc_names.append(f"{current_dish_name}/{mode}")

        if doc_names:
            print(f"传入生成的上下文: {', '.join(doc_names)}")
        else:
            print("没有可传入生成的上下文文档")

        print("生成详细回答...")
        content_type = context_pack.get("content_type")

        if route_type == "detail":
            if stream:
                return self.generation_module.generate_step_by_step_answer_stream(
                    question,
                    context_docs,
                    content_type=content_type,
                )
            return self.generation_module.generate_step_by_step_answer(
                question,
                context_docs,
                content_type=content_type,
            )

        if stream:
            return self.generation_module.generate_basic_answer_stream(
                question,
                context_docs,
                content_type=content_type,
            )
        return self.generation_module.generate_basic_answer(
            question,
            context_docs,
            content_type=content_type,
        )
```

- [ ] **Step 8: Run integration tests and verify they pass**

Run:

```bash
cd code/C8
pytest tests/test_conversation_state.py::test_chat_path_builds_context_pack_before_detail_generation tests/test_conversation_state.py::test_chat_path_builds_context_pack_before_list_generation tests/test_conversation_state.py::test_chat_path_builds_context_pack_for_streaming_detail_generation tests/test_conversation_state.py::test_chat_path_parent_expansion_allows_none_target_for_detail_without_dish -q
```

Expected:

- PASS.

- [ ] **Step 9: Migrate existing tests and fixtures touched by the new context-pack boundary**

Search:

```bash
rg -n "monkeypatch\.setattr\(system, \"_generate_detail_response\"|_generate_detail_response|def _system|def _tips_system|def _system_with_generation|test_chat_path_uses_retrieval_executor_result|test_chat_path_returns_low_evidence_without_generation" code/C8/tests/test_conversation_state.py code/C8/tests/test_state_hardening.py
```

For monkeypatched `_generate_detail_response` lambdas, update the signature from the old 8-argument helper contract:

```python
lambda question, stream, session_id, route_type, filters, entities, dish_name, relevant_chunks: ...
```

to the new 5-argument helper contract:

```python
lambda question, stream, route_type, dish_name, context_pack: ...
```

Update `test_context_first_pipeline_does_not_block_ordinal_followup_before_snapshot` specifically:

```python
monkeypatch.setattr(
    system,
    "_generate_detail_response",
    lambda question, stream, route_type, dish_name, context_pack: "鸡胸肉沙拉适合减脂。",
)
```

Update `code/C8/tests/test_conversation_state.py` so its stub retrieval/data modules return recipe-shaped Markdown and its fixtures construct a real `ContextPacker`. Replace the existing `_StubRetrievalModule`, `_TipsFallbackRetrievalModule`, `_StubDataModule`, `_TipsFallbackDataModule`, `_system()`, and `_tips_system()` definitions with:

```python
from rag_modules.context_packer import ContextPacker


EGG_FRIED_RICE_MD = (
    "# 蛋炒饭的做法\n\n"
    "## 必备原料和工具\n\n"
    "- 米饭\n"
    "- 鸡蛋\n\n"
    "## 操作\n\n"
    "- 鸡蛋打散。\n"
    "- 下锅炒饭。\n"
)

PAN_FRIED_RICE_MD = (
    "# 煎饭的做法\n\n"
    "## 操作\n\n"
    "1. 先热锅，再下油。\n"
    "2. 煎到底部定型后再加水。\n"
    "3. 收干前不要频繁翻动。\n\n"
    "## 附加内容\n\n"
    "- 火候不要太大。\n"
    "- 热锅后再下油更不容易粘锅。\n"
)


class _StubRetrievalModule:
    last_search_trace = {}

    def extract_filters_from_query(self, query):
        return {}

    def metadata_filtered_search(self, *args, **kwargs):
        return [Document(page_content="蛋炒饭怎么做", metadata={"dish_name": "蛋炒饭", "parent_id": "egg-parent"})]

    def hybrid_search(self, *args, **kwargs):
        return [Document(page_content="蛋炒饭怎么做", metadata={"dish_name": "蛋炒饭", "parent_id": "egg-parent"})]


class _TipsFallbackRetrievalModule:
    last_search_trace = {}

    def extract_filters_from_query(self, query):
        return {}

    def metadata_filtered_search(self, query, filters, top_k=3, query_dish=None):
        if filters.get("content_type") == "tips":
            return []
        return [Document(page_content="煎饭怎么做", metadata={"dish_name": "煎饭", "parent_id": "pan-parent"})]

    def hybrid_search(self, *args, **kwargs):
        return [Document(page_content="煎饭怎么做", metadata={"dish_name": "煎饭", "parent_id": "pan-parent"})]


class _StubDataModule:
    def get_parent_documents(self, chunks, target_dish_name=None):
        return [
            Document(
                page_content=EGG_FRIED_RICE_MD,
                metadata={"dish_name": "蛋炒饭", "parent_id": "egg-parent", "rrf_score": 1.0},
            )
        ]


class _TipsFallbackDataModule:
    def get_parent_documents(self, chunks, target_dish_name=None):
        return [
            Document(
                page_content=PAN_FRIED_RICE_MD,
                metadata={"dish_name": "煎饭", "parent_id": "pan-parent", "rrf_score": 1.0},
            )
        ]


def _system() -> RecipeRAGSystem:
    system = RecipeRAGSystem.__new__(RecipeRAGSystem)
    system.config = SimpleNamespace(
        top_k=3,
        context_pack_max_chars_total=2400,
        context_pack_max_chars_per_doc=1200,
        context_pack_max_docs=5,
    )
    system.data_module = _StubDataModule()
    system.retrieval_module = _StubRetrievalModule()
    system.context_packer = ContextPacker(
        max_chars_total=system.config.context_pack_max_chars_total,
        max_chars_per_doc=system.config.context_pack_max_chars_per_doc,
        max_docs=system.config.context_pack_max_docs,
    )
    system.generation_module = _StubGenerationModule()
    system._latest_parent_docs = []
    system.last_query_diagnostics = {}
    system.last_execution_result = {}
    return system


def _tips_system() -> RecipeRAGSystem:
    system = RecipeRAGSystem.__new__(RecipeRAGSystem)
    system.config = SimpleNamespace(
        top_k=3,
        context_pack_max_chars_total=2400,
        context_pack_max_chars_per_doc=1200,
        context_pack_max_docs=5,
    )
    system.data_module = _TipsFallbackDataModule()
    system.retrieval_module = _TipsFallbackRetrievalModule()
    system.context_packer = ContextPacker(
        max_chars_total=system.config.context_pack_max_chars_total,
        max_chars_per_doc=system.config.context_pack_max_chars_per_doc,
        max_docs=system.config.context_pack_max_docs,
    )
    system.generation_module = _StubGenerationModule()
    system._latest_parent_docs = []
    system.last_query_diagnostics = {}
    system.last_execution_result = {}
    return system
```

Expected existing tests that must still pass after this fixture migration:

- `test_stream_detail_turn_persists_conversation_state_after_stream_consumed`
- `test_list_turn_is_recorded_in_conversation_history`
- `test_tips_query_falls_back_to_same_dish_when_tips_chunks_are_missing`
- `test_detail_generation_uses_single_new_writeback_path`
- `test_recipe_question_preserves_original_for_query_planning`
- `test_stream_detail_turn_uses_conversation_context`

Update `test_chat_path_uses_retrieval_executor_result` so it no longer relies on monkeypatching `_generate_detail_response` to skip parent expansion. Add a minimal `data_module` and `context_packer`:

```python
class FakeData:
    def get_parent_documents(self, chunks, target_dish_name=None):
        return [Document(page_content="# 蛋炒饭的做法\n\n## 操作\n\n- 炒饭", metadata={"dish_name": "蛋炒饭"})]

class FakeContextPacker:
    def build_context_pack(self, **kwargs):
        return {
            "answer_mode": "recipe_detail",
            "context_docs": kwargs["parent_docs"],
            "parent_docs": kwargs["parent_docs"],
            "selected_sections": [{"section_type": "steps"}],
            "content_type": "steps",
            "trace": {"selected_section_count": 1},
        }

system.data_module = FakeData()
system.context_packer = FakeContextPacker()
```

Do not add `data_module` to `test_chat_path_returns_low_evidence_without_generation` just to satisfy the new path. That test must continue proving low-evidence returns before parent expansion and context packing. If it starts requiring `data_module`, the low-evidence order has regressed.

Update `code/C8/tests/test_state_hardening.py` the same way. Add the import:

```python
from rag_modules.context_packer import ContextPacker
```

Replace its `_StubRetrievalModule`, `_StubDataModule`, and `_system_with_generation()` definitions with:

```python
EGG_FRIED_RICE_MD = (
    "# 蛋炒饭的做法\n\n"
    "## 必备原料和工具\n\n"
    "- 米饭\n"
    "- 鸡蛋\n\n"
    "## 操作\n\n"
    "- 鸡蛋打散。\n"
    "- 下锅炒饭。\n"
)


class _StubRetrievalModule:
    last_search_trace = {}

    def extract_filters_from_query(self, query):
        return {}

    def metadata_filtered_search(self, *args, **kwargs):
        return [Document(page_content="蛋炒饭怎么做", metadata={"dish_name": "蛋炒饭", "parent_id": "egg-parent"})]

    def hybrid_search(self, *args, **kwargs):
        return [Document(page_content="蛋炒饭怎么做", metadata={"dish_name": "蛋炒饭", "parent_id": "egg-parent"})]


class _StubDataModule:
    def get_parent_documents(self, chunks, target_dish_name=None):
        return [
            Document(
                page_content=EGG_FRIED_RICE_MD,
                metadata={"dish_name": "蛋炒饭", "parent_id": "egg-parent", "rrf_score": 1.0},
            )
        ]


def _system_with_generation(module):
    system = RecipeRAGSystem.__new__(RecipeRAGSystem)
    system.config = SimpleNamespace(
        top_k=3,
        context_pack_max_chars_total=2400,
        context_pack_max_chars_per_doc=1200,
        context_pack_max_docs=5,
    )
    system.data_module = _StubDataModule()
    system.retrieval_module = _StubRetrievalModule()
    system.context_packer = ContextPacker(
        max_chars_total=system.config.context_pack_max_chars_total,
        max_chars_per_doc=system.config.context_pack_max_chars_per_doc,
        max_docs=system.config.context_pack_max_docs,
    )
    system.generation_module = module
    system._latest_parent_docs = []
    system.last_query_diagnostics = {}
    system.last_execution_result = {}
    return system
```

Run:

```bash
cd code/C8
pytest tests/test_conversation_state.py tests/test_state_hardening.py -q
```

Expected:

- PASS.

- [ ] **Step 10: Commit**

```bash
git add code/C8/config.py code/C8/main.py code/C8/tests/test_conversation_state.py code/C8/tests/test_state_hardening.py
git commit -m "refactor: build context packs before generation"
```

---

## Task 6: Add Context Pack Cutover Tests

**Files:**
- Create: `code/C8/tests/test_context_packer_cutover.py`
- Modify: `code/C8/main.py`
- Modify: `code/C8/rag_modules/__init__.py`

- [ ] **Step 1: Add source-level cutover tests**

Create `code/C8/tests/test_context_packer_cutover.py`:

```python
import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "main.py"


def _function_def(function_name: str) -> ast.FunctionDef:
    source = MAIN.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            return node
    raise AssertionError(f"{function_name} not found")


def _function_source(function_name: str) -> str:
    source = MAIN.read_text(encoding="utf-8")
    return ast.get_source_segment(source, _function_def(function_name))


def _arg_names(function_name: str) -> list[str]:
    return [arg.arg for arg in _function_def(function_name).args.args]


def test_ask_question_builds_context_pack_before_generation_helpers():
    source = _function_source("ask_question")

    parent_index = source.index("parent_docs = self.data_module.get_parent_documents(")
    pack_index = source.index("context_pack = self.context_packer.build_context_pack(")
    list_index = source.index("self._generate_list_response(")
    detail_index = source.index("self._generate_detail_response(")

    assert parent_index < pack_index < list_index
    assert parent_index < pack_index < detail_index


def test_generation_helpers_do_not_expand_parent_documents():
    list_source = _function_source("_generate_list_response")
    detail_source = _function_source("_generate_detail_response")

    assert "get_parent_documents(" not in list_source
    assert "get_parent_documents(" not in detail_source
    assert "context_pack[\"context_docs\"]" in list_source
    assert "context_pack[\"context_docs\"]" in detail_source
    assert "context_pack[\"parent_docs\"]" not in list_source
    assert "context_pack[\"parent_docs\"]" not in detail_source


def test_generation_helper_signatures_match_context_pack_contract():
    assert _arg_names("_generate_list_response") == ["self", "question", "context_pack"]
    assert _arg_names("_generate_detail_response") == [
        "self",
        "question",
        "stream",
        "route_type",
        "dish_name",
        "context_pack",
    ]


def test_generation_helpers_pass_packed_context_docs_to_generation():
    list_source = _function_source("_generate_list_response")
    detail_source = _function_source("_generate_detail_response")

    assert "context_docs = list(context_pack[\"context_docs\"])" in list_source
    assert "generate_list_answer(question, context_docs)" in list_source
    assert "context_docs = list(context_pack[\"context_docs\"])" in detail_source
    assert "generate_step_by_step_answer_stream(" in detail_source
    assert "generate_step_by_step_answer(" in detail_source
    assert "generate_basic_answer_stream(" in detail_source
    assert "generate_basic_answer(" in detail_source


def test_generation_helpers_do_not_assign_latest_parent_docs():
    list_source = _function_source("_generate_list_response")
    detail_source = _function_source("_generate_detail_response")

    assert "_latest_parent_docs" not in list_source
    assert "_latest_parent_docs" not in detail_source


def test_generation_helpers_have_no_dead_parameters():
    detail_source = _function_source("_generate_detail_response")

    assert "session_id" not in _arg_names("_generate_list_response")
    assert "session_id" not in _arg_names("_generate_detail_response")
    assert "filters" not in _arg_names("_generate_detail_response")
    assert "entities" not in _arg_names("_generate_detail_response")
    assert "filters.get" not in detail_source


def test_ask_question_records_context_pack_trace_only_on_execution_result():
    source = _function_source("ask_question")

    assert "query_plan[\"context_pack_trace\"]" not in source
    assert "query_plan[\"answer_mode\"]" not in source
    assert "execution_result[\"context_pack_trace\"]" in source
    assert "execution_result[\"answer_mode\"]" in source


def test_context_packer_is_configured_from_rag_config():
    source = MAIN.read_text(encoding="utf-8")

    assert "max_chars_total=self.config.context_pack_max_chars_total" in source
    assert "max_chars_per_doc=self.config.context_pack_max_chars_per_doc" in source
    assert "max_docs=self.config.context_pack_max_docs" in source


def test_context_packer_submodule_is_exported():
    init_source = (ROOT / "rag_modules" / "__init__.py").read_text(encoding="utf-8")

    assert "from . import context_packer" in init_source
    assert "'context_packer'" in init_source or '"context_packer"' in init_source


def test_generation_logs_describe_packed_context_docs():
    list_source = _function_source("_generate_list_response")
    detail_source = _function_source("_generate_detail_response")

    assert "传入生成的上下文" in list_source
    assert "传入生成的上下文" in detail_source
    assert "找到文档" not in list_source
    assert "找到文档" not in detail_source
```

- [ ] **Step 2: Run cutover tests and verify they pass**

Run:

```bash
cd code/C8
pytest tests/test_context_packer_cutover.py -q
```

Expected:

- PASS after Task 5.

- [ ] **Step 3: Run source scan for forbidden parent expansion in generation helpers**

Run:

```bash
cd code/C8
python -c "import ast; from pathlib import Path; source=Path('main.py').read_text(encoding='utf-8'); tree=ast.parse(source); funcs={n.name: ast.get_source_segment(source,n) for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}; list_sig=funcs['_generate_list_response'].split('):',1)[0]; detail_sig=funcs['_generate_detail_response'].split('):',1)[0]; assert 'get_parent_documents(' not in funcs['_generate_list_response']; assert 'get_parent_documents(' not in funcs['_generate_detail_response']; assert '_latest_parent_docs' not in funcs['_generate_list_response']; assert '_latest_parent_docs' not in funcs['_generate_detail_response']; assert 'session_id' not in list_sig; assert 'session_id' not in detail_sig; assert 'filters' not in detail_sig; assert 'entities' not in detail_sig; assert 'query_plan[\"context_pack_trace\"]' not in funcs['ask_question']; assert 'context_packer.build_context_pack(' in funcs['ask_question']"
```

Expected:

- No output and exit code 0.

- [ ] **Step 4: Commit**

```bash
git add code/C8/main.py code/C8/rag_modules/__init__.py code/C8/tests/test_context_packer_cutover.py
git commit -m "test: prevent parent expansion inside generation helpers"
```

---

## Task 7: Run Stage 04 Acceptance Suite

**Files:**
- Verify only unless existing tests require migration to the new context-pack contract.

- [ ] **Step 1: Run focused context tests**

Run:

```bash
cd code/C8
pytest tests/test_context_packer.py tests/test_context_packer_cutover.py -q
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

- [ ] **Step 3: Run retrieval and context adjacent tests**

Run:

```bash
cd code/C8
pytest tests/test_retrieval_executor.py tests/test_retrieval_executor_cutover.py tests/test_state_hardening.py -q
```

Expected:

- PASS.

- [ ] **Step 4: Run UTF-8 anchor check on Stage 04 docs and tests**

Run:

```bash
cd code/C8
python -c "from pathlib import Path; files=[Path('docs/architecture/evolution/04-context-packing-and-answer-modes.md'),Path('docs/superpowers/plans/2026-07-07-context-packing-and-answer-modes.md'),Path('tests/test_context_packer.py'),Path('tests/test_context_packer_cutover.py')]; [p.read_text(encoding='utf-8') for p in files]; text=files[1].read_text(encoding='utf-8'); anchors=['ContextPacker','\u5bab\u4fdd\u9e21\u4e01','\u5fc5\u5907\u539f\u6599\u548c\u5de5\u5177']; assert all(anchor in text for anchor in anchors), anchors"
```

Expected:

- No output and exit code 0.

- [ ] **Step 5: Run mojibake scan on Stage 04 plan**

Run:

```bash
cd code/C8
python -c "from pathlib import Path; bad=[0x93ac,0x7ed7,0x6769,0x7039,0x748b,0x934b,0x688e,0x7314,0x8e47,0x93c8,0x9422,0x5997,0x7edb]; files=[Path('docs/superpowers/plans/2026-07-07-context-packing-and-answer-modes.md'),Path('docs/architecture/evolution/04-context-packing-and-answer-modes.md')]; hits=[(str(p), ch) for p in files for ch in p.read_text(encoding='utf-8') if ord(ch) in bad]; assert not hits, hits[:10]"
```

Expected:

- No output.

- [ ] **Step 6: Commit any test migration edits**

If Step 2 or Step 3 required updates to existing tests, run:

```bash
git add code/C8/tests/test_conversation_state.py code/C8/tests/test_state_hardening.py
git commit -m "test: align context tests with context pack contract"
```

If no existing tests required edits, do not create an empty commit.

---

## Self-Review

Spec coverage:

- Section extraction is covered by Task 1.
- H3 substeps are intentionally preserved inside the H2 steps section and covered by Task 1.
- Answer mode finalization is covered by Task 2.
- `history_answer` fallback to `history_based` is covered by Task 2.
- Section selection for ingredients, steps, substitution, and recommendation is covered by Task 3.
- Ranking/source metadata preservation on packed section docs is covered by Task 3.
- Context trimming and fallback-to-chunks are covered by Task 4.
- Empty `parent_docs` fallback trace and metadata are covered by Task 4.
- Main-chain context-pack cutover is covered by Task 5.
- Removal of parent expansion from generation helpers is covered by Task 6.
- `rag_modules.context_packer` export is covered by Task 1 and Task 6.
- Stream generation is covered by Task 5.
- `dish_name=None` parent expansion is covered by Task 5.
- Existing `_system()` and `_tips_system()` fixture migration is covered by Task 5.
- Existing retrieval-executor chat-path tests are covered by Task 5.
- `test_state_hardening.py` fixture migration is covered by Task 5.
- Low-evidence no-parent-expansion ordering is preserved by Task 5.
- `dish_name=None` parent expansion is isolated from real query-planning inference by Task 5.
- `RAGConfig` context-pack budgets are covered by Task 5 and Task 6.
- Acceptance verification is covered by Task 7.

Type consistency:

- The context module is consistently named `rag_modules.context_packer`.
- The context module is exported as `rag_modules.context_packer`.
- The class is consistently named `ContextPacker`.
- The pack method is consistently named `build_context_pack`.
- Context pack fields are consistently `answer_mode`, `context_docs`, `parent_docs`, `selected_sections`, `content_type`, and `trace`.
- Generation helpers consistently receive `context_pack`, not raw `relevant_chunks`.
- `_generate_detail_response` does not keep the old `filters` parameter.
- `_generate_detail_response` does not keep dead `session_id` or `entities` parameters.
- `_generate_list_response` does not keep dead `session_id`.
- Context-pack trace is stored on `execution_result`, not duplicated into `query_plan`.
- Test fixtures that exercise successful retrieval use recipe-shaped parent docs with selectable `##` sections.

Cutover consistency:

- Parent expansion moves to `ask_question()`.
- Generation helpers stop calling `get_parent_documents()`.
- Generation helpers stop writing `_latest_parent_docs`.
- Packed section docs are passed to generation.
- Packed section and summary docs preserve source/ranking metadata needed by downstream formatting.
- Original parent docs remain available through `context_pack["parent_docs"]` for `_latest_parent_docs` and writeback diagnostics.
- Generation helper logs describe packed context docs/sections, not full parent documents.
- `ContextPacker` construction is controlled by `RAGConfig` values instead of hard-coded defaults in `main.py`.
