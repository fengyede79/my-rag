# 02 Context First Turn Pipeline

Status: accepted baseline

## Purpose

Move the main request chain toward context-first understanding without changing every downstream behavior at once.

Stage 02 cuts the order of the turn pipeline. The old order lets `check_front_door(question)` and `qualify_turn(question)` decide too much before session context exists. The new order must read a lightweight snapshot first, then decide whether the turn is smalltalk, harmless out-of-domain, a retrievable recipe query, or a state-dependent follow-up.

This stage should preserve retrieval and generation behavior as much as possible. The point is not to improve search quality yet; it is to ensure the right query reaches the existing search/generation path.

Stage 02 may leave some advanced user flows conservatively routed, but it must not introduce a side path that contradicts the frozen architecture. Once a new module owns a responsibility, the old module path for that responsibility must be deleted or reduced to a thin compatibility wrapper that is still actively called by the new module. Uncalled old modules, branches, and tests must not remain.

## Current Starting Point

Current `ask_question()` order:

```text
check_front_door(question)
-> qualify_turn(question)
-> build_conversation_snapshot(session, current_query)
-> optional resolve_reference_from_snapshot()
-> build_execution_plan()
-> query plan / retrieval / generation / writeback
```

Current problems:

- front-door domain rejection cannot see previous recommendation lists or current dish;
- `qualify_turn()` cannot use snapshot evidence for ordinal, pronoun, or implicit follow-up;
- smalltalk and out-of-domain handling are split between front-door and turn qualification;
- reference resolution already exists, but it runs after early decisions that may have blocked the turn.

Stage 02 should treat the old front-door and turn-qualification modules as sources of behavior to split apart, not as permanent architecture boundaries. Their mixed responsibilities should move into narrower contracts:

```text
front_door_guardrail.py
  old: safety + smalltalk + harmless out-of-domain
  new: basic safety only

turn_qualification.py
  old: current-query-only follow-up classification
  new: context-aware turn understanding, or replaced by a new module with that responsibility
```

Existing modules should be reused where possible:

- `front_door_guardrail.py` can be narrowed into a basic safety gate;
- `turn_qualification.py` can evolve into context-aware turn understanding;
- `conversation_state_builder.py` already provides the snapshot shape;
- `reference_resolution.py` can remain the reference-resolution engine, with a stricter output adapter if needed;
- `execution_planner.py` can consume the new action contract.

## First Thing To Do

Split the existing front-door behavior into:

1. `basic_safety_gate(query)`: only handles empty input, punctuation-only input, dangerous/malicious input, or obviously invalid input;
2. `build_conversation_snapshot(session, current_query)`;
3. `understand_turn(query, snapshot)`: context-aware action decision.

The first code change should establish this order without changing retrieval internals:

```text
basic_safety_gate
-> read session snapshot
-> turn understanding
-> reference resolution
-> execution plan
```

## Scope

Stage 02 owns:

- basic safety gate contract;
- context-first order in `ask_question()`;
- `TurnUnderstanding` action contract;
- context-aware domain rejection;
- context-aware smalltalk handling;
- deciding when reference resolution should run;
- adapting existing `qualify_turn()` behavior into the new contract;
- ensuring blocking ambiguity is the only direct clarification trigger.

Stage 02 should not rewrite retrieval, generation prompts, or state write policy.

## Basic Safety Gate Contract

The basic safety gate is narrower than the old front door.

Allowed outputs:

```python
{
    "decision": "continue | block",
    "reason": "empty | punctuation_only | unsafe | invalid_input | default_continue",
    "message": None,
}
```

Required behavior:

- It may block empty or punctuation-only input.
- It may block unsafe or malicious input if such rules exist.
- It must not classify harmless out-of-domain questions.
- It must not block isolated references such as `这个`, `它`, or `第一个`; these require snapshot-aware handling.
- It must not produce dish names, filters, route types, answer modes, or rewritten queries.

Harmless out-of-domain questions move to Turn Understanding, not the basic gate.

## Turn Understanding Contract

Turn Understanding must produce:

```python
{
    "action": "domain_reject | smalltalk | history_answer | retrieve_list | retrieve_detail | compare | substitution | clarification_response",
    "answer_mode_hint": "safe_direct | recommendation | recipe_detail | comparison | substitution | troubleshooting | history_based",
    "depends_on_state": True,
    "needs_reference_resolution": True,
    "domain_confidence": 0.0,
    "reference_trigger": "none | ordinal_reference | pronoun | implicit_detail_followup | correction",
    "should_retrieve": True,
    "reason": "short diagnostic reason"
}
```

Action rules:

| action | Meaning |
| --- | --- |
| `domain_reject` | Harmless but out-of-domain; answer directly without retrieval. |
| `smalltalk` | Greeting, thanks, assistant identity, or capability question; answer directly. |
| `history_answer` | Terminal architecture action for reliable history-only answers. Stage 02 may classify it, but should not implement a new history-answer engine. |
| `retrieve_list` | Recommendation/list-style recipe query. |
| `retrieve_detail` | Recipe detail query, including ingredients, steps, tips, intro, or constraints. |
| `compare` | Compare multiple dishes or options. Stage 02 may classify this but downstream can still route conservatively. |
| `substitution` | Ingredient or constraint modification. Stage 02 may classify this but downstream can still route conservatively. |
| `clarification_response` | User appears to answer a pending clarification. |

Initial Stage 02 implementation can map unsupported advanced actions (`compare`, `substitution`, `history_answer`) to existing direct-answer or retrieval behavior after classification. The action label should still be explicit so later stages can specialize it. Stage 02 must not add answer-mode-specific generation for these advanced labels.

Direct-answer rules:

```text
domain_reject -> direct answer, no retrieval, write back through Stage 01 answer_type=domain_reject
smalltalk -> direct answer, no retrieval, write back through Stage 01 answer_type=smalltalk
history_answer -> do not implement a new engine in Stage 02; downgrade to direct answer or conservative retrieval
```

## Context-Aware Domain Rules

Turn Understanding must use snapshot evidence before rejecting a query as out-of-domain.

Examples:

| query | context | expected action |
| --- | --- | --- |
| `Python 怎么学` | no relevant recipe context | `domain_reject` |
| `谢谢` | any context | `smalltalk` |
| `第一个怎么做` | recent recommendations exist | `retrieve_detail`, reference trigger `ordinal_reference` |
| `第一个作者是谁` | recent recommendations exist | `domain_reject` or clarification; must not silently resolve to recipe detail |
| `第一个适合减脂吗` | recent recommendations exist | state-dependent follow-up; must not be rejected before snapshot |
| `这个能不放辣吗` | active current dish exists | `retrieve_detail` or `substitution`, reference trigger `pronoun` |
| `它呢` | no active current dish and no recommendations | blocking ambiguity via reference resolution or safety gate |

Ordinal references may resolve only when the remaining query is a recipe-supported intent.

```text
第一个怎么做 -> can resolve
第一个需要什么食材 -> can resolve
第一个适合减脂吗 -> can resolve or route as constraint check
第一个作者是谁 -> must not resolve as recipe detail
第一个是哪年发明的 -> must not resolve as recipe detail
```

## Reference Resolution Contract

Reference resolution should return or be adapted to:

```python
{
    "resolved_target": None,
    "confidence": 0.0,
    "evidence_source": None,
    "ambiguity_reason": None,
    "blocking_ambiguity": False,
    "next_action": "apply_reference_resolution | apply_correction | ask_clarification | none",
}
```

Rules:

- Only `blocking_ambiguity=True` or `next_action == "ask_clarification"` should directly produce a clarification question.
- Ordinal references require a valid `recent_recommendations` list.
- Pronoun/implicit references require an active `current_dish` or other reliable allowed target.
- Unrelated ordinal wording must not resolve just because a recommendation list exists.
- Corrections can switch target only when the corrected target is explicit enough.

## Execution Plan Boundary

After Stage 02, Execution Plan should consume the new `action` and resolution output. It should decide whether to:

- direct-answer;
- clarify;
- retrieve list;
- retrieve detail;
- use existing conservative retrieval for unsupported advanced action labels.

Execution Plan should not re-own domain rejection if Turn Understanding already produced `domain_reject`.

`domain_reject` and `smalltalk` should bypass retrieval. Execution Plan may pass them through to direct-answer handling, but must not convert them back into retrievable turns.

## Migration Strategy

Recommended migration sequence:

1. Add tests for context-first behavior using `ConversationManager` and existing snapshot builder.
2. Create or adapt `basic_safety_gate()` so it no longer owns harmless out-of-domain direct replies.
3. Add `understand_turn(question, snapshot)` as the new contract surface.
4. Rewire `ask_question()` order to build snapshot before turn understanding.
5. Adapt `turn_info` compatibility fields so existing downstream code still runs:

```python
{
    "turn_type": "...",
    "response_mode": "...",
    "should_retrieve": ...,
    "should_run_reference_resolution": ...,
    "reference_trigger": "...",
    "action": "...",
    "answer_mode_hint": "...",
}
```

6. Keep existing query routing and retrieval behavior unchanged.
7. Update tests to assert no early domain rejection for valid context-dependent follow-ups.
8. Remove or narrow old module paths after the new contract owns the responsibility. Do not keep uncalled old front-door or turn-qualification branches around as alternate production paths.

## Out Of Scope

- Retrieval Executor.
- Metadata soft weighting.
- Evidence Quality Check.
- Context trimming.
- Full answer-mode specialization.
- Streaming lifecycle.
- Optimistic concurrency and retry budget.
- LLM judge or advanced planner.

## Deliverable

A context-first turn pipeline where:

- session snapshot exists before domain rejection and follow-up classification;
- harmless out-of-domain is handled by Turn Understanding;
- smalltalk is handled without retrieval;
- follow-up detection can use recommendation/current-dish state;
- reference resolution runs only when the turn contract says it should;
- existing retrieval/generation path continues to serve retrievable turns.

## Acceptance

Stage 02 is accepted when tests prove:

- harmless out-of-domain requests become `domain_reject` without retrieval;
- smalltalk does not mutate recipe state;
- ordinal follow-up can use the previous recommendation list;
- pronoun/current-dish follow-up can use active current dish;
- unrelated ordinal wording does not silently resolve to a recipe entity;
- no snapshot-dependent query is blocked before snapshot is read;
- `第一个适合减脂吗` after a recommendation list is not blocked by the basic gate before snapshot;
- existing list/detail recipe queries still reach the existing retrieval and generation path;
- old smalltalk/out-of-domain/follow-up production branches are either deleted or routed through the new `basic_safety_gate` / `understand_turn` contracts.
