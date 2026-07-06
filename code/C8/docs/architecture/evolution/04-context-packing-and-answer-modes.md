# 04 Context Packing And Answer Modes

Status: expanded baseline

## Purpose

Prevent parent-document expansion from flooding generation context, and make answer mode explicit instead of accidental.

Stage 04 is a context-boundary stage. After Stage 03, retrieval returns evidence-quality-aware chunks. Stage 04 decides how those chunks become generation context. The main chain must stop letting generation helper methods perform parent expansion and pass full parent documents directly into prompts.

The goal is not a large prompt redesign. The goal is to insert a stable contract:

```text
retrieved chunks
-> parent expansion
-> section selection
-> context trimming
-> context pack
-> answer mode finalization
-> generation
```

Once `ContextPacker` owns context construction, old production paths that expand parent docs inside generation helpers must be deleted or narrowed to actively called helpers with no independent context-selection behavior.
Generation helpers should also drop parameters that only existed for the old boundary. After cutover, list generation receives `(question, context_pack)`, while detail/basic generation receives `(question, stream, route_type, dish_name, context_pack)`. `session_id`, `filters`, `entities`, and raw `relevant_chunks` are not generation-helper inputs in Stage 04.

## Current Starting Point

Current chat generation context is built in `main.py` helper methods:

- `_generate_list_response(question, session_id, relevant_chunks)`
  - calls `data_module.get_parent_documents(relevant_chunks)`;
  - stores `_latest_parent_docs`;
  - passes full parent docs to `generation_module.generate_list_answer(...)`.
- `_generate_detail_response(...)`
  - calls `data_module.get_parent_documents(relevant_chunks, target_dish_name=dish_name)`;
  - stores `_latest_parent_docs`;
  - passes full parent docs to `generate_step_by_step_answer(...)` or `generate_basic_answer(...)`.
- `generation_integration._build_context(...)`
  - sorts full docs by `rrf_score`;
  - concatenates page content until a max length is reached.

This creates two boundary problems:

- parent expansion happens inside generation helpers instead of the main runtime context stage;
- full parent documents can enter generation even when the user only asked for ingredients, tips, or substitution.

Stage 04 moves parent expansion and section selection before generation. Generation can still receive `Document` objects, but those documents should represent packed/selected context rather than full parent documents.

## First Thing To Do

Introduce `ContextPacker` as a contract before changing generation prompts:

```text
ContextPacker.build_context_pack(
    query,
    retrieval_result,
    query_plan,
    execution_plan,
    turn_info,
    parent_docs,
)
```

The first implementation should preserve current successful answers by returning LangChain `Document` objects whose `page_content` contains selected sections. This avoids a prompt rewrite while still cutting the context boundary.

## Scope

Stage 04 owns:

- parent expansion placement in the main chain;
- section extraction from Markdown parent documents;
- section selection based on answer mode and content type;
- context trimming;
- `ContextPack` result structure;
- answer mode finalization boundary;
- cutover so generation helpers no longer call `get_parent_documents()`;
- trace data explaining selected sections and final answer mode.

Stage 04 does not own retrieval fallback, evidence quality, state versioning, stream lifecycle, large prompt redesign, or new LLM answer modes.

## Answer Mode Rule

Execution Plan and Turn Understanding own the initial user-facing task. Context packing may finalize or refine answer mode, but it must not unexpectedly change the task because of retrieved document shape.

Initial answer modes:

- `recommendation`
- `recipe_detail`
- `comparison`
- `substitution`
- `troubleshooting`
- `history_based`
- `safe_direct`

Finalization rules:

- `retrieve_list` or route type `list` finalizes to `recommendation`.
- `retrieve_detail` finalizes to `recipe_detail` unless the turn/action already says `substitution`, `comparison`, or `troubleshooting`.
- `substitution` may refine section preferences, but must not become `recommendation`.
- `comparison` may select multiple dish contexts, but must not become a single-dish detail answer.
- `safe_direct` and `history_based` do not require retrieval context packing in Stage 04.

Conceptual API:

```python
def finalize_answer_mode(
    *,
    turn_info: dict,
    execution_plan: dict,
    query_plan: dict,
    retrieval_quality: dict | None,
) -> str:
    ...
```

## ContextPack Contract

`ContextPack` is the generation-facing output of Stage 04.

Required shape:

```python
{
    "answer_mode": "recipe_detail",
    "context_docs": [],          # selected/truncated Document objects for generation
    "parent_docs": [],           # original parent docs used for state/writeback diagnostics
    "selected_sections": [
        {
            "dish_name": "宫保鸡丁",
            "section_title": "操作",
            "section_type": "steps",
            "source_parent_id": "...",
            "token_estimate": 320,
        }
    ],
    "content_type": "steps",
    "trace": {
        "input_chunk_count": 3,
        "parent_doc_count": 1,
        "selected_section_count": 1,
        "trimmed": False,
        "answer_mode_source": "execution_plan",
        "answer_mode_initial": "recipe_detail",
        "answer_mode_final": "recipe_detail",
    }
}
```

Rules:

- `context_docs` is always present.
- `parent_docs` is always present for diagnostics and writeback.
- `context_docs` should contain selected section documents, not full parent documents, when section selection succeeds.
- If section selection fails but parent docs exist, fallback to compact parent excerpts, not unrestricted full documents.
- `answer_mode` must come from finalization rules, not from accidental document shape.
- The main chain should pass `context_pack["context_docs"]` into generation and use `context_pack["parent_docs"]` for `_latest_parent_docs` and writeback diagnostics.
- `trace` should be attached to the final `execution_result`; it should not be duplicated into `query_plan`.

## Section Extraction

Recipe parent documents are Markdown. Common headings include:

- `# 菜名的做法`
- `## 必备原料和工具`
- `## 计算`
- `## 操作`
- `### 子步骤`
- `## 附加内容`

Stage 04 should parse sections structurally enough for reliable selection:

```python
{
    "title": "操作",
    "section_type": "steps",
    "text": "## 操作\n...",
    "level": 2,
    "dish_name": "宫保鸡丁",
    "parent_id": "...",
}
```

Section type mapping:

| headings / signals | section_type |
| --- | --- |
| `必备原料和工具`, `原料`, `食材`, `材料`, `配料` | `ingredients` |
| `操作`, `步骤`, `做法`, `制作` | `steps` |
| `附加内容`, `注意`, `小贴士`, `技巧`, `提示` | `tips` |
| `计算`, `用量`, `份量` | `calculation` |
| top description before first `##` | `introduction` |

The parser does not need a full Markdown AST in Stage 04. A heading-based parser is acceptable if tests cover the common recipe format.
For Stage 04, `###` and deeper headings are preserved inside their containing `##` section rather than promoted into independent sections. This keeps ordered substeps such as `### 腌制鸡肉` and `### 炒制` together under the `steps` context, avoiding accidental reordering or fragmented generation context.

## Section Selection Rule

Select sections based on finalized answer mode, query intent, and requested content type.

Content-type preferences:

| content_type / intent | preferred sections | secondary sections |
| --- | --- | --- |
| `ingredients` | `ingredients` | `calculation`, `tips` |
| `steps` / recipe detail | `steps` | `ingredients`, `tips` |
| `tips` | `tips` | `steps` |
| `introduction` | `introduction` | `tips` |
| `calculation` | `calculation` | `ingredients` |
| `substitution` | `ingredients`, `tips` | `steps` |
| `troubleshooting` | `tips`, `steps` | `ingredients` |
| `recommendation` | compact parent summaries | retrieved chunk snippets |

Rules:

- Ingredient questions should not require full-document context.
- Steps questions must preserve ordered procedure text.
- Tips/substitution questions should include relevant `附加内容`/tips text and enough ingredient context to be useful.
- Recommendation list generation may use compact parent summaries rather than full recipe details.
- If selected sections are empty, include the highest-signal retrieved chunks as compact fallback context.

## Context Trimming

Context trimming keeps selected context within a predictable budget.

Initial budget:

```python
{
    "max_chars_total": 2400,
    "max_chars_per_doc": 1200,
    "max_docs": 5,
}
```

These defaults belong in `RAGConfig` as runtime configuration, not as hard-coded `ContextPacker()` construction inside `main.py`.

Rules:

- Preserve the beginning of ordered `steps` sections.
- Prefer complete bullet/list items when trimming.
- Do not split in the middle of a heading if avoidable.
- Keep dish name and section title in each packed document's metadata.
- Preserve useful ranking/source metadata such as `rrf_score`, `parent_id`, and `dish_name` when creating packed section or summary docs, so downstream prompt formatting keeps deterministic ordering.
- Record whether trimming occurred in `trace.trimmed`.

## Main Chain Cutover

After Stage 04:

```text
ask_question()
  retrieval_result = RetrievalExecutor.execute(...)
  parent_docs = data_module.get_parent_documents(retrieval_result["chunks"], target_dish_name=dish_name)
  context_pack = ContextPacker.build_context_pack(...)
  generation(context_pack["context_docs"])
```

Illegal after cutover:

- `_generate_list_response()` calling `data_module.get_parent_documents()`;
- `_generate_detail_response()` calling `data_module.get_parent_documents()`;
- generation helpers deciding which parent sections are relevant;
- passing unrestricted full parent docs into generation when selected sections exist.
- generation helpers keeping old-boundary dead parameters such as `session_id`, `filters`, or `entities`;
- writing context-pack trace to both `query_plan` and `execution_result`.

Allowed after cutover:

- `data_module.get_parent_documents()` remains the parent-expansion provider.
- `rag_modules.context_packer` should be exported from `rag_modules.__init__` for module discoverability, even though `main.py` may import `ContextPacker` directly.
- `generation_integration._build_context()` may continue formatting `Document` objects into prompt text.
- Structured generation may continue using `content_type`, but it should receive packed docs.
- `_latest_parent_docs` may still store original parent docs from `ContextPack.parent_docs`.
- Generation helper logs may name packed context docs/sections, but should not imply full parent docs were passed to generation.

## Observability

Each context pack should expose:

- input chunk count;
- parent document count;
- selected section count;
- answer mode before and after finalization;
- requested content type;
- selected section titles and types;
- trimming status;
- fallback-to-chunks status.

Minimum trace:

```python
{
    "input_chunk_count": 3,
    "parent_doc_count": 1,
    "selected_section_count": 2,
    "answer_mode_initial": "recipe_detail",
    "answer_mode_final": "recipe_detail",
    "answer_mode_source": "execution_plan",
    "content_type": "steps",
    "trimmed": False,
    "fallback_to_chunks": False,
}
```

## Migration Strategy

Recommended migration sequence:

1. Add section extraction tests using realistic Markdown recipe snippets.
2. Add answer mode finalization tests.
3. Add `ContextPacker` tests for ingredients, steps, tips, substitution, and recommendation.
4. Add context trimming tests.
5. Cut `ask_question()` to perform parent expansion and context packing before generation.
6. Change `_generate_list_response()` and `_generate_detail_response()` so they accept already-packed context docs and no longer call `get_parent_documents()`.
7. Add source-level cutover tests preventing parent expansion inside generation helpers.
8. Run existing conversation tests to ensure list/detail behavior still reaches generation.

## Out Of Scope

- Retrieval fallback policy.
- Evidence quality thresholds.
- State versioning.
- Stream lifecycle.
- Large prompt redesign.
- LLM judge section selection.
- New UI or API response shape.

## Deliverable

A context layer where:

- parent expansion happens once in the main chain after retrieval;
- `ContextPacker` produces selected/truncated generation context;
- answer mode is finalized explicitly;
- generation receives packed docs instead of full parent docs when section selection succeeds;
- parent docs remain available for state/writeback diagnostics;
- old generation helper parent-expansion paths are removed.

## Acceptance

Stage 04 is accepted when tests prove:

- ingredient questions do not require full-document context;
- steps questions preserve ordered procedure context;
- `###` substeps remain ordered inside the selected `steps` section;
- tips/substitution questions include relevant tips and ingredient context;
- recommendation mode uses compact parent summaries;
- packed section docs preserve useful ranking/source metadata such as `rrf_score`;
- empty `parent_docs` falls back to retrieved chunks with explicit `fallback_to_chunks` trace;
- `history_answer` finalizes to `history_based` even when no answer-mode hint is present;
- answer mode remains controlled by planning and cannot flip because of retrieved document shape;
- `_generate_list_response()` and `_generate_detail_response()` no longer call `get_parent_documents()`;
- generation helper signatures contain no old-boundary dead parameters;
- `ask_question()` builds a `ContextPack` before calling generation;
- `rag_modules.context_packer` is exported from `rag_modules.__init__`;
- existing list/detail recipe queries still reach generation with sufficient packed context.
