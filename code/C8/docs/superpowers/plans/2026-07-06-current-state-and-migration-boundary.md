# Current State And Migration Boundary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Validate and finalize the Stage 00 current-state map so later migration stages can rely on it without rediscovering the codebase.

**Architecture:** This stage is documentation-only. It maps the current `RecipeRAGSystem` runtime, startup path, session state, writeback behavior, observability, and migration gaps to the frozen architecture without changing runtime code.

**Tech Stack:** Markdown documentation, PowerShell, ripgrep, existing Python/Flask RAG codebase.

---

### Task 1: Verify Current Runtime Flow Against Code

**Files:**
- Read: `code/C8/main.py`
- Read: `code/C8/web_app.py`
- Modify: `code/C8/docs/architecture/evolution/00-current-state-and-migration-boundary.md`

- [ ] **Step 1: Inspect Web request entry points**

Run:

```powershell
Select-String -Path code\C8\web_app.py -Pattern 'def _default_system_factory|def get_system|@app.post|@app.get\("/api/chat/stream|ask_question|sse_event' -Context 2,3
```

Expected: output shows `_default_system_factory()`, `get_system()`, `/api/chat`, `/api/chat/stream`, and both calls to `system.ask_question(...)`.

- [ ] **Step 2: Inspect `ask_question()` control flow**

Run:

```powershell
Select-String -Path code\C8\main.py -Pattern 'def ask_question|check_front_door|qualify_turn|build_conversation_snapshot|resolve_reference_from_snapshot|build_execution_plan|_build_query_plan|rewrite_query_for_execution|_search_relevant_chunks|_generate_list_response|_generate_detail_response|_write_conversation_turn|_wrap_stream_with_writeback' -Context 0,1
```

Expected: output shows the current order:

```text
check_front_door
qualify_turn
build_conversation_snapshot
resolve_reference_from_snapshot
build_execution_plan
_build_query_plan
rewrite_query_for_execution
_search_relevant_chunks
generation
writeback
stream wrapper
```

- [ ] **Step 3: Confirm Stage 00 documents this order**

Open `code/C8/docs/architecture/evolution/00-current-state-and-migration-boundary.md` and confirm the `Current Runtime Flow` section includes at least:

```text
check_front_door(question)
qualify_turn(question)
build_conversation_snapshot(session, current_query)
  -> optional resolve_reference_from_snapshot()
  -> guard_resolution_output()
  -> ask_clarification path writes turn and returns
build_execution_plan(turn_info, resolution)
_build_query_plan(question, session_id)
rewrite_query_for_execution()
_apply_resolved_target_to_query_plan()
preference constraint propagation
_rewrite_question_for_search()
_search_relevant_chunks()
_generate_list_response() / _generate_detail_response()
_wrap_stream_with_writeback()
_write_conversation_turn()
```

This should match the existing spec content.

- [ ] **Step 4: Patch Stage 00 if any runtime step is missing**

If Step 3 finds a missing runtime step, add it to the `Current Runtime Flow` block. Keep the description factual and current-state-only. Do not add target architecture behavior to this section.

### Task 2: Verify Startup And Knowledge-Base Mapping

**Files:**
- Read: `code/C8/main.py`
- Read: `code/C8/web_app.py`
- Modify: `code/C8/docs/architecture/evolution/00-current-state-and-migration-boundary.md`

- [ ] **Step 1: Inspect startup functions**

Run:

```powershell
Select-String -Path code\C8\main.py,code\C8\web_app.py -Pattern 'def _default_system_factory|def initialize_system|def build_knowledge_base|_try_load_existing_knowledge_base|_rebuild_knowledge_base|RetrievalOptimizationModule' -Context 1,2
```

Expected: output shows web lazy initialization and main initialization/build paths.

- [ ] **Step 2: Confirm Stage 00 documents startup and build behavior**

Open `code/C8/docs/architecture/evolution/00-current-state-and-migration-boundary.md` and confirm the `Current system startup and knowledge-base path` section includes:

```text
create_app()
get_system()
_default_system_factory()
initialize_system()
build_knowledge_base()
try load existing FAISS index
try load cached chunks
build vector index if missing
initialize RetrievalOptimizationModule(vectorstore, chunks)
```

- [ ] **Step 3: Patch only factual omissions**

If startup behavior is missing, patch only the current behavior. Do not propose eager loading or deployment changes in Stage 00.

### Task 3: Verify Session State And Writeback Mapping

**Files:**
- Read: `code/C8/rag_modules/conversation_manager.py`
- Read: `code/C8/rag_modules/state_writeback_review.py`
- Modify: `code/C8/docs/architecture/evolution/00-current-state-and-migration-boundary.md`

- [ ] **Step 1: Inspect current session fields**

Run:

```powershell
Select-String -Path code\C8\rag_modules\conversation_manager.py -Pattern 'class SessionState|current_entity|recent_recommendations|pending_clarification|writeback_turn_state|writeback_mode' -Context 2,4
```

Expected: output shows the current `SessionState` fields and the writeback switch.

- [ ] **Step 2: Inspect writeback review modes**

Run:

```powershell
Select-String -Path code\C8\rag_modules\state_writeback_review.py -Pattern 'writeback_mode|message_only|clarification_pending|recommendation_list|resolved_followup|correction_turn|explicit_single_dish' -Context 1,2
```

Expected: output shows review modes used by `ConversationManager.writeback_turn_state()`.

- [ ] **Step 3: Confirm Stage 00 documents current fields**

Confirm `Current State And Writeback Shape` includes this current-state shape:

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

- [ ] **Step 4: Confirm Stage 00 documents current writeback modes**

Confirm the same section lists:

```text
message_only
clarification_pending
recommendation_list
resolved_followup
correction_turn
explicit_single_dish
normal fallback
```

- [ ] **Step 5: Patch gaps without changing policy**

If fields or modes are missing, update Stage 00 as current-state documentation only. Do not introduce `StateUpdatePolicy` implementation details here beyond identifying it as a future Stage 01 gap.

### Task 4: Verify Gap Map And Migration Boundaries

**Files:**
- Read: `code/C8/docs/architecture/main-runtime-architecture-spec.md`
- Modify: `code/C8/docs/architecture/evolution/00-current-state-and-migration-boundary.md`

- [ ] **Step 1: Compare Stage 00 against frozen architecture**

Run:

```powershell
Select-String -Path code\C8\docs\architecture\main-runtime-architecture-spec.md -Pattern 'Turn Understanding|Reference Resolution|Retrieval Executor|Evidence Quality Check|StateUpdatePolicy|Version Checks|Streaming Lifecycle|Observability' -Context 1,2
```

Expected: output shows the major frozen architecture contracts.

- [ ] **Step 2: Confirm Stage 00 has a migration-stage row for each major contract**

Confirm `Gap Map To Frozen Architecture` maps these contracts:

```text
context-first domain/follow-up understanding -> Stage 02
Turn Understanding action/answer_mode_hint -> Stage 02
Reference Resolution confidence/evidence/ambiguity -> Stage 02
StateUpdatePolicy/state_diff -> Stage 01
state names and versioning -> Stage 01 and Stage 05
RetrievalExecutor -> Stage 03
Evidence Quality Check -> Stage 03
fallback policy -> Stage 03
section selection/context trimming -> Stage 04
answer mode ownership -> Stage 04
turn_id/trace_id/read_state_version/retry budget -> Stage 05
stream lifecycle -> Stage 05
end-to-end acceptance -> Stage 06
```

- [ ] **Step 3: Confirm early boundaries are explicit**

Confirm `What We Will Not Change Yet` includes:

```text
LLM provider or model selection
embedding model
FAISS index format
Web API paths or response shape
front-end UI
recipe source data format
evaluation framework structure
production queueing/distributed locks
current Web API compatibility
current CLI behavior
current index loading and rebuilding behavior
current environment variable requirements
```

- [ ] **Step 4: Patch incorrect stage mappings**

If a frozen contract is missing or mapped to the wrong stage, patch the table. Keep mappings consistent with `code/C8/docs/architecture/evolution/README.md`.

### Task 5: Validate Stage 00 Documentation Hygiene

**Files:**
- Read: `code/C8/docs/architecture/evolution/00-current-state-and-migration-boundary.md`

- [ ] **Step 1: Scan for placeholder language**

Run:

```powershell
$terms = @('TB' + 'D', 'TO' + 'DO', 'implement' + ' later', 'fill' + ' in')
rg -n ($terms -join '|') code\C8\docs\architecture\evolution\00-current-state-and-migration-boundary.md
```

Expected: no output. Exit code `1` from `rg` is acceptable here because it means no matches.

- [ ] **Step 2: Confirm expected sections exist**

Run:

```powershell
Select-String -Path code\C8\docs\architecture\evolution\00-current-state-and-migration-boundary.md -Pattern 'Current Runtime Flow|Current State And Writeback Shape|Existing Capabilities|Gap Map To Frozen Architecture|Current Observability|Migration Principles|What We Will Not Change Yet|Stage Dependency Map|Acceptance For Stage 00'
```

Expected: one match for each listed section.

- [ ] **Step 3: Confirm file size is still reviewable**

Run:

```powershell
Get-Content -Path code\C8\docs\architecture\evolution\00-current-state-and-migration-boundary.md | Measure-Object -Line -Word
```

Expected: document remains concise enough for review. A line count around 250-350 lines is acceptable for this construction map.

### Task 6: Stage 00 Acceptance Record

**Files:**
- Modify: `code/C8/docs/architecture/evolution/00-current-state-and-migration-boundary.md`

- [ ] **Step 1: Add an acceptance note if the team wants the file marked accepted**

If Stage 00 has been reviewed and accepted, change the status line from:

```markdown
Status: expanded baseline
```

to:

```markdown
Status: accepted baseline
```

- [ ] **Step 2: Review final diff**

Run:

```powershell
git diff -- code\C8\docs\architecture\evolution\00-current-state-and-migration-boundary.md
```

Expected: diff contains documentation-only changes.

- [ ] **Step 3: Commit Stage 00 docs when approved**

Run:

```powershell
git add code\C8\docs\architecture\evolution\00-current-state-and-migration-boundary.md code\C8\docs\superpowers\plans\2026-07-06-current-state-and-migration-boundary.md
git commit -m "docs: add current-state migration boundary"
```

Expected: commit succeeds after review. Do not include runtime code changes in this commit. Only include files that were actually modified during this stage.

---

## Self-Review Checklist

- Stage 00 remains documentation-only.
- Current behavior is clearly separated from target behavior.
- Every major frozen architecture gap maps to a later stage.
- Early migration boundaries prevent accidental broad rewrite.
- No runtime code changes are required to accept this stage.
