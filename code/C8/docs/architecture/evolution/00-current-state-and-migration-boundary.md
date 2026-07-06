# 00 Current State And Migration Boundary

Status: accepted baseline

## Purpose

Create a shared map of the current implementation before changing architecture. This stage prevents the migration from becoming a broad rewrite.

This stage does not implement the target architecture. It explains where the current code already has useful pieces, where the order differs from the frozen architecture, and where later stages must tighten contracts.

## Current Runtime Flow

Current request entry points:

```text
web_app.py
  POST /api/chat
    -> get_system()
    -> RecipeRAGSystem.ask_question(stream=False)

  GET /api/chat/stream
    -> get_system()
    -> RecipeRAGSystem.ask_question(stream=True)
    -> SSE message/done/error events

main.py interactive CLI
  -> RecipeRAGSystem.ask_question(stream=True/False)
```

Current system startup and knowledge-base path:

```text
web_app.py
  create_app()
    -> app config stores SYSTEM_FACTORY, RAG_SYSTEM, RAG_LOCK
    -> get_system() lazily creates one shared RecipeRAGSystem under RAG_LOCK
    -> _default_system_factory()
         -> RecipeRAGSystem()
         -> initialize_system()
         -> build_knowledge_base()

main.py interactive CLI
  -> RecipeRAGSystem()
  -> run_interactive()
       -> initialize_system()
       -> build_knowledge_base()
```

Current `initialize_system()` creates:

```text
DataPreparationModule
IndexConstructionModule
GenerationIntegrationModule(enable_conversation=True)
```

Current `build_knowledge_base()` does:

```text
try load existing FAISS index
  -> try load cached chunks
  -> otherwise reload documents and chunks
if index missing
  -> load markdown documents
  -> chunk documents
  -> build vector index
  -> save index and chunks
initialize RetrievalOptimizationModule(vectorstore, chunks)
```

Current `RecipeRAGSystem.ask_question()` flow:

```text
ask_question(question, stream, session_id)
  -> check_front_door(question)
       -> block/direct_reply paths write conversation turn and return

  -> qualify_turn(question)
       -> polite_direct_reply path generates smalltalk, writes turn, returns

  -> build_conversation_snapshot(session, current_query)
       -> optional resolve_reference_from_snapshot()
       -> guard_resolution_output()
       -> ask_clarification path writes turn and returns

  -> build_execution_plan(turn_info, resolution)
  -> _build_query_plan(question, session_id)
       -> generation_module.query_router()
       -> route_type / filters / dish_name / confidence

  -> rewrite_query_for_execution()
  -> optionally rebuild query plan for rewritten question
  -> _apply_resolved_target_to_query_plan()
  -> preference constraint propagation

  -> execution_plan ask_clarification path writes turn and returns

  -> _rewrite_question_for_search()
  -> _search_relevant_chunks()
       -> retrieval_module.extract_filters_from_query()
       -> metadata_filtered_search() or hybrid_search()
       -> limited fallback behavior for tips queries

  -> no result path writes turn and returns

  -> list path:
       -> _generate_list_response()
       -> data_module.get_parent_documents()
       -> generation_module.generate_list_answer()
       -> _extract_recommended_dishes()

  -> detail/basic path:
       -> _generate_detail_response()
       -> data_module.get_parent_documents()
       -> generation_module.generate_step_by_step_answer[_stream]()
          or generate_basic_answer[_stream]()

  -> stream generator path:
       -> _wrap_stream_with_writeback()
       -> write conversation turn after stream is fully consumed

  -> non-stream path:
       -> _write_conversation_turn()
       -> optional diagnostics
       -> return answer
```

Important current ordering facts:

- `check_front_door(question)` runs before session snapshot construction.
- `qualify_turn(question)` also runs before session snapshot construction.
- Reference resolution exists, but it only participates after front-door and turn qualification.
- Retrieval execution is embedded in `RecipeRAGSystem._search_relevant_chunks()`, not isolated behind a Retrieval Executor contract.
- State writeback is centralized through `ConversationManager.writeback_turn_state()`, but there is not yet an explicit `StateUpdatePolicy` field whitelist.
- Streaming writeback is deferred until the stream is fully consumed, but lifecycle states such as `started`, `streaming`, `completed`, and `aborted` are not yet explicit runtime concepts.
- Web mode uses one lazily initialized in-process `RecipeRAGSystem` guarded only during construction. Per-session conversation state is managed inside `ConversationManager`.
- `RecipeRAGSystem.ask_question()` requires `retrieval_module` and `generation_module` to be initialized. Initialization failures surface at request time in lazy web mode.

## Current State And Writeback Shape

Current `SessionState` fields:

```python
{
    "session_id": "...",
    "created_at": 0.0,
    "last_active": 0.0,
    "is_active": True,
    "messages": [],
    "current_entity": None,
    "current_intent": "general",
    "user_preferences": {},
    "topic_mode": "none",
    "recent_recommendations": [],
    "recent_topics": [],
    "last_confirmed_target": None,
    "current_entity_meta": {},
    "pending_clarification": None
}
```

Current writeback path:

```text
RecipeRAGSystem._write_conversation_turn()
  -> ConversationManager.writeback_turn_state()
       -> state_writeback_review.review_state_writeback()
       -> writeback_mode switch
            message_only
            clarification_pending
            recommendation_list
            resolved_followup
            correction_turn
            explicit_single_dish
            normal fallback
```

Current state-write strengths:

- all turn writeback already funnels through `ConversationManager.writeback_turn_state()`;
- `state_writeback_review.review_state_writeback()` already provides a review step before mutating session state;
- recommendation lists, pending clarification, and current dish updates already have separate helper methods.

Current state-write gaps:

- field names do not yet match the frozen target contract (`current_entity` vs `current_dish`, `recent_recommendations` vs `last_recommendation_list`);
- there is no `state_version`;
- there is no `turn_id` or `trace_id` in session writeback;
- writeback uses modes, but not yet a formal `answer_type -> state_diff` whitelist;
- session locking exists inside `ConversationManager`, but there is no optimistic version check for stale reads.

## Existing Capabilities

The current system already contains many pieces needed by the frozen architecture.

| Capability | Current location | Current role |
| --- | --- | --- |
| Web entry | `code/C8/web_app.py` | Provides normal and SSE chat endpoints. |
| Lazy app initialization | `code/C8/web_app.py` | Creates one shared RAG system under `RAG_LOCK`. |
| Main orchestration | `code/C8/main.py` | Owns `RecipeRAGSystem.ask_question()` and most runtime sequencing. |
| Configuration | `code/C8/config.py` | Provides data path, model, retrieval, and index settings. |
| Data preparation | `code/C8/rag_modules/data_preparation.py` | Loads markdown recipes, chunks documents, extracts metadata, and expands parent docs. |
| Index construction | `code/C8/rag_modules/index_construction.py` | Loads or builds FAISS vector index. |
| Front-door guardrail | `code/C8/rag_modules/front_door_guardrail.py` | Performs early block/direct-reply decisions. |
| Turn qualification | `code/C8/rag_modules/turn_qualification.py` | Classifies turn behavior before snapshot. |
| Session state | `code/C8/rag_modules/conversation_manager.py` | Stores session history, current dish, recommendations, and writeback behavior. |
| Writeback review | `code/C8/rag_modules/state_writeback_review.py` | Chooses current writeback mode before session mutation. |
| Snapshot builder | `code/C8/rag_modules/conversation_state_builder.py` | Builds structured state for reference resolution. |
| Reference resolution | `code/C8/rag_modules/reference_resolution.py` | Resolves ordinal and implicit references using snapshot data. |
| Execution planning | `code/C8/rag_modules/execution_planner.py` | Decides high-level execution action from turn info and resolution. |
| Query routing | `code/C8/rag_modules/generation_integration.py` | Produces route type, filters, dish name, and generation mode helpers. |
| Retrieval | `code/C8/rag_modules/retrieval_optimization.py` | Implements hybrid search, metadata filtered search, RRF-style behavior, and filter extraction. |
| Parent document expansion | `code/C8/rag_modules/data_preparation.py` | Maps child chunks back to parent recipe documents. |
| Structured generation | `code/C8/rag_modules/structured_generation.py` | Builds structured answers when document structure is sufficient. |
| Diagnostics | `code/C8/evaluation/process_diagnostics.py` | Builds turn-level diagnostic reports when requested. |
| Evaluation | `code/C8/evaluation/` | Provides evaluation dataset, scoring, and run scripts. |
| State hardening tests | `code/C8/tests/` | Existing tests cover many conversation and guardrail behaviors. |

## Gap Map To Frozen Architecture

| Frozen architecture requirement | Current state | Migration stage |
| --- | --- | --- |
| Context participates before domain rejection and follow-up classification. | Snapshot is built after front-door guardrail and turn qualification. | `02-context-first-turn-pipeline.md` |
| Turn Understanding emits explicit `action` and `answer_mode_hint`. | `qualify_turn()` emits useful fields, but action ownership is not yet the frozen contract. | `02-context-first-turn-pipeline.md` |
| Reference Resolution returns structured confidence/evidence/ambiguity. | Resolution exists and is guarded, but the final contract needs to be tightened and made consistently consumed. | `02-context-first-turn-pipeline.md` |
| State writes go through `StateUpdatePolicy` and `state_diff`. | Writeback is centralized, but policy and field whitelist are implicit. | `01-state-contract-and-writeback-policy.md` |
| Frozen state names are canonical and versioned. | Current state uses `current_entity`, `recent_recommendations`, and no `state_version`. | `01-state-contract-and-writeback-policy.md`, then `05-runtime-versioning-and-streaming.md` |
| Low-evidence/no-result is a typed result-producing node. | No-result path exists; low-confidence policy is not yet a first-class contract. | `03-retrieval-executor-and-quality.md` |
| Retrieval is owned by `RetrievalExecutor`. | Retrieval decision logic lives in `RecipeRAGSystem._search_relevant_chunks()`. | `03-retrieval-executor-and-quality.md` |
| Metadata preferences use soft weighting where appropriate. | `metadata_filtered_search()` can still behave like a hard filter depending on filters. | `03-retrieval-executor-and-quality.md` |
| Evidence Quality Check controls fallback and generation. | Retrieval can return empty results, but quality checking is not a dedicated node. | `03-retrieval-executor-and-quality.md` |
| Fallback is optional and marked as weaker evidence. | Some fallback exists for tips queries; general fallback policy is not formalized. | `03-retrieval-executor-and-quality.md` |
| Parent expansion is followed by section selection and context trimming. | Parent expansion exists; section selection/context packing is not yet explicit. | `04-context-packing-and-answer-modes.md` |
| Execution Plan owns initial answer mode. | Route type and generation mode are split across query routing and downstream generation choices. | `04-context-packing-and-answer-modes.md` |
| Turn runtime has `turn_id`, `trace_id`, `read_state_version`, and shared retry budget. | Diagnostics exist, but runtime versioning is not a chain-level contract. | `05-runtime-versioning-and-streaming.md` |
| Streaming has explicit lifecycle states. | Streaming wrapper defers writeback until consumption, but abort/fail lifecycle is not explicit. | `05-runtime-versioning-and-streaming.md` |
| Web runtime has clear initialization behavior. | Web mode lazily initializes on first request; construction is locked, but startup failures happen inside request handling. | Keep as known behavior unless a later deployment-focused spec changes it. |
| End-to-end acceptance proves staged behavior. | Unit tests exist; the frozen architecture needs scenario-level acceptance. | `06-end-to-end-acceptance.md` |

## Current Observability

Current trace and diagnostics surfaces:

- `RecipeRAGSystem.last_query_diagnostics`;
- `RecipeRAGSystem.last_execution_result`;
- `RetrievalOptimizationModule.last_search_trace`;
- `GenerationIntegrationModule.last_generation_trace`;
- optional `return_diagnostics=True` path for non-streaming calls;
- Flask file logging through `web_app.runtime.log`;
- evaluation helpers under `code/C8/evaluation/`.

Current observability gaps:

- no stable `turn_id` or `trace_id` joins all logs, retrieval traces, generation traces, and state writes;
- stream paths do not have complete lifecycle traces;
- state writeback does not yet emit a formal `state_diff`;
- diagnostics exist but are not yet the single acceptance artifact for all staged scenarios.

## Migration Principles

- Preserve current behavior while introducing contracts.
- Move responsibilities gradually; do not rewrite `ask_question()` in one step.
- Prefer adapter layers before deeper behavior changes.
- Keep each stage independently testable.
- Use existing modules where possible.
- Do not expand the system into a general agent runtime.
- Every stage must state what it changes and what it deliberately leaves alone.

## What We Will Not Change Yet

The early migration stages must not change:

- LLM provider or model selection;
- embedding model;
- FAISS index format;
- Web API paths or response shape;
- front-end UI;
- recipe source data format;
- prompt strategy except where required by a specific stage;
- evaluation framework structure;
- production queueing, distributed locks, or multi-worker transaction design.

The early migration stages should preserve:

- current Web API compatibility;
- current CLI interaction behavior;
- current index loading and rebuilding behavior;
- current evaluation entry points;
- current environment variable requirements such as `DASHSCOPE_API_KEY`.

The early migration stages should also avoid:

- large-scale `main.py` decomposition before contracts exist;
- implementing every answer mode at once;
- broad fallback retrieval without quality policy;
- making streaming lifecycle changes before state writeback policy is stable.
- changing lazy web initialization behavior unless the relevant stage explicitly scopes it.

## Stage Dependency Map

```text
00 current-state map
  -> 01 state contract and writeback policy
      -> 02 context-first turn pipeline
          -> 03 retrieval executor and quality
              -> 04 context packing and answer modes
                  -> 05 runtime versioning and streaming
                      -> 06 end-to-end acceptance
```

Reasoning:

- `01` comes before major pipeline changes because safe state writes reduce migration risk.
- `02` comes before retrieval changes because query and reference behavior define retrieval inputs.
- `03` comes before context packing because context packing depends on retrieval result shape and quality.
- `04` comes before streaming/runtime finalization because generation context and answer mode should be stable first.
- `05` comes after the core chain is stable so version and stream lifecycle checks wrap known behavior.
- `06` validates the whole staged migration.

## Acceptance For Stage 00

Stage 00 is accepted when:

- the current `ask_question()` runtime flow is documented;
- existing capabilities are mapped to files;
- gaps are mapped to the correct future stages;
- early out-of-scope boundaries are explicit;
- later stage plans can reference this file instead of rediscovering current behavior.
