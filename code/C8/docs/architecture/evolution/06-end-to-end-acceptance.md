# 06 End To End Acceptance

Status: draft baseline

Detailed spec:

- `../../superpowers/specs/2026-07-07-end-to-end-acceptance-design.md`

## Purpose

Validate that the staged migration reaches the frozen architecture through observable behavior, not just through module creation.

This is the final numbered architecture migration stage. Later work should be ordinary bugfixes, quality tuning, or feature specs unless a new architecture review explicitly reopens the runtime design.

## First Thing To Do

Build an end-to-end scenario matrix that exercises:

- multi-turn recommendation;
- ordinal reference;
- current-dish follow-up;
- harmless out-of-domain rejection;
- low evidence;
- fallback;
- streaming abort;
- state conflict or rapid follow-up.

## Primary Scenario

The main acceptance chain is:

```text
推荐三个鸡肉菜
-> 第一个怎么做
-> 这个能不放辣吗
-> 没有豆瓣酱怎么办
-> 给我换个不辣的
-> 谢谢
```

Expected behavior:

- recommendation list remains available across smalltalk;
- ordinal reference resolves to the intended dish;
- current dish is not overwritten by low-confidence turns;
- substitution and constraint follow-ups stay attached to the correct recipe context;
- final smalltalk does not clear business state.

## Additional Scenarios

| Scenario | Expected behavior |
| --- | --- |
| `Python 怎么学` | Domain reject, no retrieval, no business state update. |
| `第一个作者是谁` after recommendations | Does not silently resolve as recipe detail. |
| Missing exact dish | No broad substitution unless fallback policy allows it. |
| Sparse metadata preference query | Soft weighting preserves recall. |
| Stream abort after recommendation | Aborted recommendation list is not valid for ordinal reference. |
| Rapid state-dependent requests | Retry budget prevents infinite replan; conflict handling is reachable. |

## Deliverable

An acceptance test set and a short result report showing whether the final architecture behavior is satisfied.

## Out Of Scope

- New feature expansion beyond the frozen architecture.
- General agent planning.
- Production deployment hardening.

## Acceptance

This stage is accepted when the primary scenario and additional scenarios pass, and the trace output can explain action, resolution, retrieval quality, answer type, and state diff for each turn.
