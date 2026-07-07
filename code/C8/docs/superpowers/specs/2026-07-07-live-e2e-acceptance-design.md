# Live E2E Acceptance Design

Status: draft spec
Owner: C8 recipe RAG system
Date: 2026-07-07

## Purpose

This spec defines the real end-to-end acceptance test for the C8 recipe RAG system.

The existing deterministic acceptance tests prove the frozen runtime architecture under controlled fixtures. They are necessary but not sufficient. This live E2E test must prove that the system can run against real data, a real Flask service, real HTTP/SSE requests, real indexes, and real external LLM calls.

This is not a new runtime architecture stage. It is a production-like acceptance harness for the frozen architecture.

## Non-Goals

The live E2E harness must not:

- replace the normal pytest unit/integration suite;
- run automatically in fast CI by default;
- use fake generation, fake retrieval, monkeypatches, or test-client shortcuts;
- introduce a new RAG chain or alternate production path;
- silently treat API/network/rate-limit failures as architecture failures;
- require concurrent API calls.

## Definition Of Real E2E

A run qualifies as live E2E only when all of the following are true:

- The Flask app is started as an external local service, not via `app.test_client()`.
- Requests go through HTTP endpoints such as `/api/chat` and `/api/chat/stream`.
- The system loads the real C8 data path and real vector index/chunks.
- If the index is missing, the runner may build it once before scenarios begin.
- The LLM calls use a real `DASHSCOPE_API_KEY`.
- The tested model is set through the same configuration path production uses, such as `RAG_LLM_MODEL`.
- Scenarios use real session IDs and preserve session state across turns.
- Results are written to durable JSONL and Markdown reports.

## Model Pool

The runner must support this model pool:

```text
qwen3-vl-235b-a22b-thinking
qwen3-vl-32b-thinking
qwen-plus-2025-07-28
deepseek-r1-distill-qwen-7b
qwen-max
glm-5
```

The default model list should be configurable. The runner should not hard-code all models as mandatory for every run.

Recommended defaults:

```text
primary smoke model: qwen-plus-2025-07-28
secondary confidence model: qwen-max
optional broad model sweep: all supported models
```

Rationale:

- A full 50-scenario run across every model can be expensive and slow.
- The harness should make it easy to run all models intentionally, but should not surprise the user by doing so.

## Rate Limiting

The runner must be conservative by default.

Required behavior:

- Run scenarios serially by default.
- No concurrent LLM calls.
- Enforce a delay between requests. Default: 5 seconds.
- Enforce a per-model cooldown after rate-limit responses. Default: 60 seconds after repeated 429-like failures.
- Retry transient failures with bounded exponential backoff.
- Stop a model run when repeated rate limits exceed the configured threshold.
- Record rate-limit failures separately from functional failures.

Suggested retry policy:

```text
attempt 1: immediate
attempt 2: wait 10 seconds
attempt 3: wait 30 seconds
attempt 4: wait 60 seconds
then mark RATE_LIMITED or INFRA_ERROR
```

The runner should expose these settings:

```bash
--delay-seconds 5
--max-retries 3
--rate-limit-cooldown-seconds 60
--per-model-scenario-limit 50
--stop-model-after-rate-limits 3
```

## Service Lifecycle

The runner owns the live service lifecycle unless told to use an existing server.

Default flow:

```text
load .env
choose model
start Flask service in a subprocess
wait for service readiness
run scenarios over HTTP/SSE
stop service
write reports
repeat for next model
```

The service should bind to localhost only.

Recommended default:

```text
host: 127.0.0.1
port: 5058
```

The runner must handle port conflicts clearly:

- If the port is busy and `--reuse-server` is not set, fail with an actionable message.
- If `--reuse-server` is set, verify the server responds before running scenarios.

Readiness should be based on a real HTTP request. If no health endpoint exists, the runner may request `/` and accept a successful response. A later implementation may add `/api/health`, but the runner must not require a new production endpoint unless the plan explicitly adds it.

## Scenario Matrix

The baseline live run must contain at least 50 scenario turns.

A "scenario turn" means one user request sent to the service. Multi-turn scenarios count each turn.

Minimum coverage:

| Category | Minimum turns | Purpose |
| --- | ---: | --- |
| Single recipe detail | 10 | Verify detail answers and real retrieval support. |
| Recommendation list | 8 | Verify list routing and recommendation extraction. |
| Multi-turn ordinal/reference | 10 | Verify session state, recommendation references, and current dish. |
| Substitution / constraint follow-up | 6 | Verify answer modes and context attachment. |
| Low evidence / missing dish | 5 | Verify no-result and state safety. |
| Harmless out-of-domain | 4 | Verify domain reject without retrieval pollution. |
| Streaming SSE | 4 | Verify streaming completion and lifecycle. |
| Rapid follow-up / conflict awareness | 3 | Verify state-version conflict path is observable. |

Total minimum: 50 turns.

The exact scenario set should live in a data file so it can be reviewed and expanded without editing runner code.

Recommended path:

```text
code/C8/e2e/scenarios/live_e2e_scenarios.yaml
```

Each scenario should include:

```yaml
- id: primary_chain_001
  category: multi_turn_reference
  session_id: live-primary-001
  turns:
    - question: 推荐三个鸡肉菜
      endpoint: chat
      assertions:
        answer_contains_any: ["宫保鸡丁", "鸡"]
        status: pass
    - question: 第一个怎么做
      endpoint: chat
      assertions:
        answer_not_contains: ["第一个怎么做"]
        state_current_dish_expected: true
```

The implementation may use JSON instead of YAML if project dependencies make YAML inconvenient.

## Assertion Model

Live E2E assertions should be robust to wording differences.

Do assert:

- HTTP status;
- non-empty answer;
- answer contains at least one expected domain term;
- answer does not contain obvious failure strings;
- answer does not echo unresolved references as the main target;
- SSE emits at least one message event and a done event;
- session state effects when state inspection is available;
- trace fields when trace inspection is available.

Do not assert:

- exact full answer text;
- exact sentence order;
- exact model phrasing;
- exact token count.

The runner should support these assertion types:

```text
answer_contains_all
answer_contains_any
answer_not_contains
answer_regex
min_answer_chars
http_status
sse_done_event
state_current_dish_expected
state_recommendations_expected
state_business_state_unchanged
trace_has_keys
retrieval_quality_not_low
```

If the current HTTP API does not expose trace/session state, the first implementation may record only HTTP-level assertions and server logs. A follow-up implementation may add a debug-only trace endpoint guarded by an environment variable. The spec allows that endpoint only for test/diagnostic mode.

## Failure Classification

Every failed turn must be classified into one of these statuses:

```text
PASS
FAIL
FLAKY
RATE_LIMITED
INFRA_ERROR
MODEL_ERROR
DATA_ERROR
SKIPPED
```

Definitions:

- `PASS`: Assertions passed.
- `FAIL`: System returned a valid response but violated functional assertions.
- `FLAKY`: Retry passed after an initial functional failure.
- `RATE_LIMITED`: API rate limit prevented a valid result.
- `INFRA_ERROR`: Service start, network, port, process, or timeout failure.
- `MODEL_ERROR`: Provider returned a model-specific error unrelated to rate limits.
- `DATA_ERROR`: Real data/index is missing, malformed, or insufficient for a scenario.
- `SKIPPED`: Scenario intentionally skipped by filters or model capability rules.

The report must not collapse these into one generic failure bucket.

## Runner Interface

Recommended command:

```bash
cd code/C8
python e2e/live_e2e_runner.py --models qwen-plus-2025-07-28,qwen-max --limit-turns 50 --delay-seconds 5
```

Required options:

```text
--models
--limit-turns
--delay-seconds
--max-retries
--rate-limit-cooldown-seconds
--results-dir
--host
--port
--reuse-server
--stream-timeout-seconds
--request-timeout-seconds
--fail-fast
```

Recommended defaults:

```text
models: qwen-plus-2025-07-28
limit-turns: 50
delay-seconds: 5
max-retries: 3
request-timeout-seconds: 120
stream-timeout-seconds: 180
results-dir: code/C8/e2e/results
host: 127.0.0.1
port: 5058
fail-fast: false
```

## Reports

Every run must create:

```text
code/C8/e2e/results/live-e2e-YYYYMMDD-HHMMSS.jsonl
code/C8/e2e/results/live-e2e-YYYYMMDD-HHMMSS.md
```

JSONL records should include:

```json
{
  "run_id": "live-e2e-20260707-153000",
  "model": "qwen-plus-2025-07-28",
  "scenario_id": "primary_chain_001",
  "turn_index": 2,
  "session_id": "live-primary-001",
  "endpoint": "chat",
  "question": "第一个怎么做",
  "http_status": 200,
  "answer": "...",
  "status": "PASS",
  "failure_class": null,
  "latency_ms": 12345,
  "attempt": 1,
  "error": null
}
```

Markdown report should include:

- run metadata;
- model list;
- rate-limit configuration;
- scenario summary by category;
- summary by model;
- failure table;
- slowest turns;
- skipped/rate-limited turns;
- recommended follow-up actions.

## Security And Secrets

The runner must never print or persist `DASHSCOPE_API_KEY`.

Rules:

- Load the key from environment or existing `.env`.
- Mask secrets in logs if environment is dumped.
- Do not write request headers containing secrets to result files.
- Do not commit result files by default unless explicitly requested.

## Data And Index Requirements

Before running scenarios, the runner should verify:

- `RAG_DATA_PATH` or default data path exists;
- the recipe corpus contains Markdown files;
- `RAG_INDEX_PATH` or default `vector_index` exists, or index rebuilding is allowed;
- Flask service can initialize the RAG system.

If the index build fails, classify the run as `DATA_ERROR` or `INFRA_ERROR` depending on the root cause.

## Acceptance Criteria

The first live E2E implementation is accepted when:

- the runner starts a real Flask service;
- at least one configured real model completes at least 50 live turns;
- all requests use HTTP/SSE, not Flask `test_client`;
- no fake retrieval or fake generation is used;
- rate limiting and bounded retry behavior are implemented;
- JSONL and Markdown reports are generated;
- failures are classified into the required buckets;
- at least the primary model run achieves an agreed pass threshold.

Default pass threshold:

```text
PASS + FLAKY >= 80% of executed functional turns
RATE_LIMITED / INFRA_ERROR are reported separately and do not count as functional pass
```

This threshold is intentionally not 100%. Live LLM behavior and data coverage can vary. The purpose is to expose real issues, not to hide them behind brittle exact-match assertions.

## Open Implementation Notes

These are implementation choices for the later plan:

- Whether to add a debug-only trace endpoint, such as `/api/debug/session/<session_id>`.
- Whether scenarios are stored as JSON or YAML.
- Whether to run one Flask process per model or reuse one process and restart with a new `RAG_LLM_MODEL`.

The recommended initial implementation is:

```text
JSON scenario file
one Flask subprocess per model
no debug endpoint in the first pass unless state assertions require it
```

Restarting per model is slower but cleaner because the current model is configured during system initialization.
