# Live E2E Expanded Scenario Set Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand the live E2E acceptance set from the current Core 50 to a Core+Extended 85-turn evaluation, with suite filtering and suite-level reporting.

**Architecture:** Keep the frozen RAG runtime untouched. The change lives entirely in the live E2E harness: scenario schema gains explicit suite membership, the loader/runner can filter by suite, reports show Core/Extended/Total separately, and the scenario JSON adds 35 harder Extended turns.

**Tech Stack:** Python 3.11, pytest, Flask live service runner, JSON scenario file, Markdown/JSONL reports, DashScope-backed live model calls.

## Global Constraints

- Do not change the RAG main chain.
- Do not change prompts, retrieval, state writeback, or runtime versioning.
- Do not hide failures by loosening assertions.
- Do not inflate the score with many trivial smalltalk or domain-reject turns.
- Do not remove or rewrite the current 50-turn Core Set.
- Do not require concurrent live requests.
- Do not require all supported models for every validation run.
- Preserve the existing `e2e/scenarios/live_e2e_scenarios.json` as the single default scenario file.
- Core Set target: `>= 45/50`.
- Expanded Total target: `>= 80/90`, or equivalent `>= 88%` when the total is not exactly 90.
- First implementation target size: `core: 50`, `extended: 35`, `total: 85`.
- Primary live model: `qwen-plus-2025-07-28`.
- Default live execution remains serial with `--delay-seconds 5`.

---

## File Structure

- Modify `code/C8/e2e/scenarios.py`
  - Add `suite: str` to `Scenario`.
  - Add `filter_scenarios_by_suite(scenarios, suite)`.
  - Keep `flatten_turns()` behavior unchanged after filtering.

- Modify `code/C8/e2e/assertions.py`
  - Add `suite: str` to `TurnResult`.
  - Add `suite` argument to `evaluate_assertions()`.
  - Include `suite` in `TurnResult.to_dict()`.

- Modify `code/C8/e2e/live_e2e_runner.py`
  - Add `--suite core|extended|all`.
  - Filter scenarios before `--limit-turns`.
  - Pass `scenario.suite` into `evaluate_assertions()`.

- Modify `code/C8/e2e/reporting.py`
  - Add suite summary to `summarize_results()`.
  - Add a dedicated `## Suite Summary` table to Markdown reports.
  - Add `Suite` column to the failure table.

- Modify `code/C8/e2e/scenarios/live_e2e_scenarios.json`
  - Add `"suite": "core"` to all current scenarios.
  - Add 35 Extended turns using `"suite": "extended"`.

- Modify `code/C8/tests/test_live_e2e_scenarios.py`
  - Test suite loading, filtering, counts, and category allocation.

- Modify `code/C8/tests/test_live_e2e_runner.py`
  - Test CLI suite argument and filtering-before-limit behavior.

- Modify `code/C8/tests/test_live_e2e_assertions.py`
  - Test `suite` is stored on `TurnResult` and serialized.

- Modify `code/C8/tests/test_live_e2e_reporting.py`
  - Test suite summary and failure table suite column.

---

### Task 1: Scenario Suite Contract

**Files:**
- Modify: `code/C8/e2e/scenarios.py`
- Modify: `code/C8/tests/test_live_e2e_scenarios.py`

**Interfaces:**
- Consumes: `load_scenarios(path: Path) -> list[Scenario]`
- Produces: `Scenario.suite: str`
- Produces: `filter_scenarios_by_suite(scenarios: list[Scenario], suite: str) -> list[Scenario]`

- [ ] **Step 1: Write failing tests for suite loading and filtering**

Add these tests to `code/C8/tests/test_live_e2e_scenarios.py`:

```python
def test_live_e2e_scenarios_have_explicit_suite_membership():
    scenarios = load_scenarios(SCENARIO_FILE)

    assert {scenario.suite for scenario in scenarios}.issubset({"core", "extended"})
    assert all(scenario.suite for scenario in scenarios)


def test_filter_scenarios_by_suite_supports_core_extended_and_all():
    scenarios = load_scenarios(SCENARIO_FILE)

    core = filter_scenarios_by_suite(scenarios, "core")
    extended = filter_scenarios_by_suite(scenarios, "extended")
    all_scenarios = filter_scenarios_by_suite(scenarios, "all")

    assert len(all_scenarios) == len(scenarios)
    assert all(scenario.suite == "core" for scenario in core)
    assert all(scenario.suite == "extended" for scenario in extended)
    assert len(core) + len(extended) == len(all_scenarios)


def test_filter_scenarios_by_suite_rejects_unknown_suite():
    scenarios = load_scenarios(SCENARIO_FILE)

    try:
        filter_scenarios_by_suite(scenarios, "shadow")
    except ValueError as exc:
        assert "suite must be one of" in str(exc)
    else:
        raise AssertionError("expected ValueError")
```

Update the import at the top of the file:

```python
from e2e.scenarios import filter_scenarios_by_suite, flatten_turns, load_scenarios
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
cd code/C8
pytest tests/test_live_e2e_scenarios.py -q
```

Expected: FAIL because `Scenario` has no `suite` attribute and `filter_scenarios_by_suite` does not exist.

- [ ] **Step 3: Implement suite contract**

Modify `code/C8/e2e/scenarios.py`:

```python
@dataclass(frozen=True)
class Scenario:
    id: str
    category: str
    session_id: str
    turns: list[ScenarioTurn]
    suite: str = "core"
```

In `load_scenarios()`, change the `Scenario(...)` construction to:

```python
        scenarios.append(
            Scenario(
                id=str(raw["id"]),
                category=str(raw["category"]),
                session_id=str(raw["session_id"]),
                turns=turns,
                suite=str(raw.get("suite", "core")),
            )
        )
```

Add this function below `load_scenarios()`:

```python
def filter_scenarios_by_suite(scenarios: list[Scenario], suite: str) -> list[Scenario]:
    if suite == "all":
        return list(scenarios)
    if suite not in {"core", "extended"}:
        raise ValueError("suite must be one of: core, extended, all")
    return [scenario for scenario in scenarios if scenario.suite == suite]
```

- [ ] **Step 4: Run scenario tests**

Run:

```bash
cd code/C8
pytest tests/test_live_e2e_scenarios.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add code/C8/e2e/scenarios.py code/C8/tests/test_live_e2e_scenarios.py
git commit -m "test: add live e2e suite contract"
```

---

### Task 2: Runner Suite Filtering

**Files:**
- Modify: `code/C8/e2e/live_e2e_runner.py`
- Modify: `code/C8/tests/test_live_e2e_runner.py`

**Interfaces:**
- Consumes: `filter_scenarios_by_suite(scenarios, suite)`
- Produces: `--suite` CLI argument with choices `core`, `extended`, `all`
- Produces: filtering before `flatten_turns(..., limit_turns=...)`

- [ ] **Step 1: Write failing runner tests**

Add these tests to `code/C8/tests/test_live_e2e_runner.py`:

```python
from e2e.live_e2e_runner import build_arg_parser, select_turns_for_run


def test_runner_accepts_suite_argument():
    args = build_arg_parser().parse_args(["--suite", "extended", "--limit-turns", "3"])

    assert args.suite == "extended"
    assert args.limit_turns == 3


def test_select_turns_filters_suite_before_limit():
    scenarios = [
        Scenario(
            id="core-1",
            category="domain_reject",
            session_id="core-session",
            suite="core",
            turns=[ScenarioTurn(question="Python 怎么学？", endpoint="chat", assertions={})],
        ),
        Scenario(
            id="extended-1",
            category="single_recipe_detail",
            session_id="extended-session",
            suite="extended",
            turns=[
                ScenarioTurn(question="拍黄瓜怎么做？", endpoint="chat", assertions={}),
                ScenarioTurn(question="鱼香肉丝怎么做？", endpoint="chat", assertions={}),
            ],
        ),
    ]

    selected = select_turns_for_run(scenarios, suite="extended", limit_turns=1)

    assert len(selected) == 1
    assert selected[0][0].id == "extended-1"
    assert selected[0][1].question == "拍黄瓜怎么做？"
```

Ensure the imports include:

```python
from e2e.scenarios import Scenario, ScenarioTurn
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
cd code/C8
pytest tests/test_live_e2e_runner.py -q
```

Expected: FAIL because `--suite` and `select_turns_for_run()` do not exist.

- [ ] **Step 3: Implement runner filtering**

In `code/C8/e2e/live_e2e_runner.py`, update the import:

```python
from e2e.scenarios import filter_scenarios_by_suite, flatten_turns, load_scenarios
```

Add this parser argument in `build_arg_parser()`:

```python
    parser.add_argument("--suite", choices=["core", "extended", "all"], default="all")
```

Add this helper near `result_paths()`:

```python
def select_turns_for_run(scenarios, *, suite: str, limit_turns: int | None):
    selected_scenarios = filter_scenarios_by_suite(scenarios, suite)
    return flatten_turns(selected_scenarios, limit_turns=limit_turns)
```

In `run_model()`, replace:

```python
    scenarios = load_scenarios(args.scenario_file)
    turns = flatten_turns(scenarios, limit_turns=args.limit_turns)
```

with:

```python
    scenarios = load_scenarios(args.scenario_file)
    turns = select_turns_for_run(scenarios, suite=args.suite, limit_turns=args.limit_turns)
```

- [ ] **Step 4: Run runner tests**

Run:

```bash
cd code/C8
pytest tests/test_live_e2e_runner.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add code/C8/e2e/live_e2e_runner.py code/C8/tests/test_live_e2e_runner.py
git commit -m "feat: filter live e2e runs by suite"
```

---

### Task 3: Suite Field In Turn Results

**Files:**
- Modify: `code/C8/e2e/assertions.py`
- Modify: `code/C8/e2e/live_e2e_runner.py`
- Modify: `code/C8/tests/test_live_e2e_assertions.py`

**Interfaces:**
- Consumes: `scenario.suite`
- Produces: `TurnResult.suite: str`
- Produces: JSONL rows with `"suite": "core"` or `"suite": "extended"`

- [ ] **Step 1: Write failing assertion test**

Add this test to `code/C8/tests/test_live_e2e_assertions.py`:

```python
def test_evaluate_assertions_records_suite_in_result_and_dict():
    result = evaluate_assertions(
        run_id="run-1",
        model="qwen-plus-2025-07-28",
        scenario_id="single_recipe_detail_001",
        suite="extended",
        category="single_recipe_detail",
        session_id="session-1",
        turn_index=1,
        endpoint="chat",
        question="拍黄瓜怎么调味？",
        http_status=200,
        answer="拍黄瓜可以用蒜、醋、生抽调味。",
        assertions={"http_status": 200, "answer_contains_any": ["黄瓜"]},
        latency_ms=100,
        attempt=1,
        sse_done_event=None,
        error=None,
        diagnostics=None,
    )

    assert result.suite == "extended"
    assert result.to_dict()["suite"] == "extended"
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
cd code/C8
pytest tests/test_live_e2e_assertions.py -q
```

Expected: FAIL because `evaluate_assertions()` does not accept `suite`.

- [ ] **Step 3: Add suite to TurnResult and evaluate_assertions**

In `code/C8/e2e/assertions.py`, add `suite` after `scenario_id`:

```python
    scenario_id: str
    suite: str
    category: str
```

In `to_dict()`, add:

```python
            "suite": self.suite,
```

In `evaluate_assertions()` parameters, add:

```python
    suite: str,
```

In both `TurnResult(...)` constructors inside `evaluate_assertions()`, add:

```python
            suite=suite,
```

- [ ] **Step 4: Pass suite from runner**

In `code/C8/e2e/live_e2e_runner.py`, update the `evaluate_assertions(...)` call:

```python
                    suite=scenario.suite,
```

Place it after `scenario_id=scenario.id`.

- [ ] **Step 5: Update existing test helpers**

Any test helper that constructs `TurnResult(...)` directly must add:

```python
        suite="core",
```

The known helper is `_result()` in `code/C8/tests/test_live_e2e_reporting.py`.

Existing direct `TurnResult(...)` construction in `code/C8/tests/test_live_e2e_assertions.py` must also add:

```python
        suite="core",
```

Existing `evaluate_assertions(...)` calls in `code/C8/tests/test_live_e2e_assertions.py` must add:

```python
        suite="core",
```

- [ ] **Step 6: Run assertion and reporting tests**

Run:

```bash
cd code/C8
pytest tests/test_live_e2e_assertions.py tests/test_live_e2e_reporting.py tests/test_live_e2e_runner.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add code/C8/e2e/assertions.py code/C8/e2e/live_e2e_runner.py code/C8/tests/test_live_e2e_assertions.py code/C8/tests/test_live_e2e_reporting.py
git commit -m "feat: include suite in live e2e results"
```

---

### Task 4: Suite Reporting

**Files:**
- Modify: `code/C8/e2e/reporting.py`
- Modify: `code/C8/tests/test_live_e2e_reporting.py`

**Interfaces:**
- Consumes: `TurnResult.suite`
- Produces: `summary["by_suite_status"]`
- Produces: Markdown `## Suite Summary`
- Produces: failure table with `Suite` column

- [ ] **Step 1: Write failing reporting tests**

Add these tests to `code/C8/tests/test_live_e2e_reporting.py`:

```python
def test_summarize_results_counts_suite_status():
    results = [
        _result("PASS", suite="core"),
        _result("FAIL", suite="core"),
        _result("PASS", suite="extended"),
    ]

    summary = summarize_results(results)

    assert summary["by_suite_status"]["core"]["total"] == 2
    assert summary["by_suite_status"]["core"]["PASS"] == 1
    assert summary["by_suite_status"]["core"]["FAIL"] == 1
    assert summary["by_suite_status"]["extended"]["total"] == 1
    assert summary["by_suite_status"]["extended"]["PASS"] == 1


def test_markdown_report_includes_suite_summary_and_failure_suite_column(tmp_path: Path):
    markdown = tmp_path / "run.md"
    results = [
        _result("PASS", suite="core"),
        _result("FAIL", suite="extended", retrieval_strategy="low_evidence", quality_reason="no_candidates"),
    ]

    write_markdown_report(
        markdown,
        run_id="run-1",
        models=["qwen-plus-2025-07-28"],
        delay_seconds=5,
        results=results,
    )

    report = markdown.read_text(encoding="utf-8")
    assert "## Suite Summary" in report
    assert "| core | 1 | 1 | 0 | 100.0% |" in report
    assert "| extended | 1 | 0 | 1 | 0.0% |" in report
    assert "| total | 2 | 1 | 1 | 50.0% |" in report
    assert "| Suite | Model | Scenario | Turn | Status | Generation | Retrieval | Quality Reason | Error |" in report
    assert "| extended | qwen-plus-2025-07-28 |" in report
```

Update `_result()` in the same file to accept suite:

```python
    suite: str = "core",
```

and pass it into `TurnResult(...)`:

```python
        suite=suite,
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
cd code/C8
pytest tests/test_live_e2e_reporting.py -q
```

Expected: FAIL because suite summary does not exist.

- [ ] **Step 3: Implement suite status summary**

In `code/C8/e2e/reporting.py`, add:

```python
def _suite_status_summary(results: list[TurnResult]) -> dict[str, dict[str, int]]:
    suites = sorted({result.suite for result in results})
    summary: dict[str, dict[str, int]] = {}
    for suite in suites:
        suite_results = [result for result in results if result.suite == suite]
        counts = Counter(result.status for result in suite_results)
        summary[suite] = {"total": len(suite_results), **dict(counts)}

    total_counts = Counter(result.status for result in results)
    summary["total"] = {"total": len(results), **dict(total_counts)}
    return summary
```

In `summarize_results()`, add:

```python
        "by_suite_status": _suite_status_summary(results),
```

- [ ] **Step 4: Implement suite table**

Add this helper to `code/C8/e2e/reporting.py`:

```python
def _suite_table(summary: dict[str, dict[str, int]]) -> str:
    lines = ["| Suite | Total | PASS | FAIL | Pass Rate |", "| --- | ---: | ---: | ---: | ---: |"]
    for suite in [item for item in ["core", "extended", "total"] if item in summary]:
        row = summary[suite]
        total = row.get("total", 0)
        passed = row.get("PASS", 0)
        failed = total - passed
        pass_rate = (passed / total * 100) if total else 0.0
        lines.append(f"| {suite} | {total} | {passed} | {failed} | {pass_rate:.1f}% |")
    return "\n".join(lines)
```

In `write_markdown_report()`, insert this section before `## Status Summary`:

```python
        "## Suite Summary",
        "",
        _suite_table(summary["by_suite_status"]),
        "",
```

- [ ] **Step 5: Add Suite column to failure table**

Replace the failure table header:

```python
        "| Suite | Model | Scenario | Turn | Status | Generation | Retrieval | Quality Reason | Error |",
        "| --- | --- | --- | ---: | --- | --- | --- | --- | --- |",
```

Replace the failure row append with:

```python
        lines.append(
            f"| {result.suite} | {result.model} | {result.scenario_id} | {result.turn_index} | {result.status} "
            f"| {generation} | {retrieval} | {quality} | {error} |"
        )
```

- [ ] **Step 6: Run reporting tests**

Run:

```bash
cd code/C8
pytest tests/test_live_e2e_reporting.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add code/C8/e2e/reporting.py code/C8/tests/test_live_e2e_reporting.py
git commit -m "feat: report live e2e suite summaries"
```

---

### Task 5: Scenario Matrix Expansion

**Files:**
- Modify: `code/C8/e2e/scenarios/live_e2e_scenarios.json`
- Modify: `code/C8/tests/test_live_e2e_scenarios.py`

**Interfaces:**
- Consumes: `Scenario.suite`
- Produces: Core 50 and Extended 35 in the default scenario file

- [ ] **Step 1: Write failing matrix tests**

Add these tests to `code/C8/tests/test_live_e2e_scenarios.py`:

```python
def test_live_e2e_core_and_extended_turn_counts_are_fixed():
    scenarios = load_scenarios(SCENARIO_FILE)
    core_turns = flatten_turns(filter_scenarios_by_suite(scenarios, "core"))
    extended_turns = flatten_turns(filter_scenarios_by_suite(scenarios, "extended"))
    all_turns = flatten_turns(filter_scenarios_by_suite(scenarios, "all"))

    assert len(core_turns) == 50
    assert len(extended_turns) == 35
    assert len(all_turns) == 85


def test_live_e2e_extended_category_allocation_matches_spec():
    scenarios = filter_scenarios_by_suite(load_scenarios(SCENARIO_FILE), "extended")
    counts: dict[str, int] = {}
    for scenario, _turn in flatten_turns(scenarios):
        counts[scenario.category] = counts.get(scenario.category, 0) + 1

    assert counts == {
        "single_recipe_detail": 7,
        "recommendation_list": 7,
        "multi_turn_reference": 7,
        "substitution_constraint": 7,
        "low_evidence": 3,
        "domain_reject": 2,
        "streaming_sse": 1,
        "rapid_followup_conflict": 1,
    }
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
cd code/C8
pytest tests/test_live_e2e_scenarios.py -q
```

Expected: FAIL because all existing scenarios default to core and no Extended Set exists.

- [ ] **Step 3: Mark existing scenarios as core**

In `code/C8/e2e/scenarios/live_e2e_scenarios.json`, add:

```json
"suite": "core"
```

to every existing scenario object:

```json
{
  "id": "single_recipe_detail_001",
  "suite": "core",
  "category": "single_recipe_detail",
  "session_id": "live-detail-001",
  "turns": [...]
}
```

Repeat for all existing scenario IDs:

```text
single_recipe_detail_001
recommendation_list_001
multi_turn_reference_001
substitution_constraint_001
low_evidence_001
domain_reject_001
streaming_sse_001
rapid_followup_conflict_001
```

- [ ] **Step 4: Add Extended single recipe detail scenario**

Append this scenario object inside the top-level `scenarios` array:

```json
{
  "id": "single_recipe_detail_ext_001",
  "suite": "extended",
  "category": "single_recipe_detail",
  "session_id": "live-detail-ext-001",
  "turns": [
    {"question": "西红柿炒鸡蛋需要什么食材？", "endpoint": "chat", "assertions": {"http_status": 200, "min_answer_chars": 20, "answer_contains_any": ["西红柿", "番茄", "鸡蛋"], "answer_not_contains": ["知识库里没有找到可靠"]}},
    {"question": "简易红烧肉怎么做？", "endpoint": "chat", "assertions": {"http_status": 200, "min_answer_chars": 20, "answer_contains_any": ["红烧肉", "五花肉", "做法"], "answer_not_contains": ["简易红烧肉怎么做"]}},
    {"question": "南派红烧肉怎么做？", "endpoint": "chat", "assertions": {"http_status": 200, "min_answer_chars": 20, "answer_contains_any": ["红烧肉", "五花肉", "糖"], "answer_not_contains": ["南派红烧肉怎么做"]}},
    {"question": "拍黄瓜怎么调味？", "endpoint": "chat", "assertions": {"http_status": 200, "min_answer_chars": 20, "answer_contains_any": ["黄瓜", "蒜", "醋"], "answer_not_contains": ["拍黄瓜怎么调味"]}},
    {"question": "鸡翅有哪些家常做法？", "endpoint": "chat", "assertions": {"http_status": 200, "min_answer_chars": 20, "answer_contains_any": ["鸡翅", "可乐", "做法"], "answer_not_contains": ["鸡翅有哪些家常做法"]}},
    {"question": "麻婆豆腐怎么不容易碎？", "endpoint": "chat", "assertions": {"http_status": 200, "min_answer_chars": 20, "answer_contains_any": ["麻婆豆腐", "豆腐", "轻"], "answer_not_contains": ["麻婆豆腐怎么不容易碎"]}},
    {"question": "鱼香肉丝需要准备哪些配菜？", "endpoint": "chat", "assertions": {"http_status": 200, "min_answer_chars": 20, "answer_contains_any": ["鱼香肉丝", "肉丝", "配菜"], "answer_not_contains": ["鱼香肉丝需要准备哪些配菜"]}}
  ]
}
```

- [ ] **Step 5: Add Extended recommendation scenario**

Append this scenario object:

```json
{
  "id": "recommendation_list_ext_001",
  "suite": "extended",
  "category": "recommendation_list",
  "session_id": "live-rec-ext-001",
  "turns": [
    {"question": "推荐三个适合新手的家常菜", "endpoint": "chat", "assertions": {"http_status": 200, "min_answer_chars": 20, "answer_contains_any": ["推荐", "新手", "简单"], "answer_not_contains": ["知识库里没有找到可靠"]}},
    {"question": "推荐几个下饭但不太辣的菜", "endpoint": "chat", "assertions": {"http_status": 200, "min_answer_chars": 20, "answer_contains_any": ["下饭", "不辣", "推荐"], "answer_not_contains": ["知识库里没有找到可靠"]}},
    {"question": "有什么适合带饭的鸡肉菜？", "endpoint": "chat", "assertions": {"http_status": 200, "min_answer_chars": 20, "answer_contains_any": ["带饭", "鸡", "推荐"], "answer_not_contains": ["知识库里没有找到可靠"]}},
    {"question": "家里有土豆和鸡蛋，推荐两个菜", "endpoint": "chat", "assertions": {"http_status": 200, "min_answer_chars": 20, "answer_contains_any": ["土豆", "鸡蛋", "推荐"], "answer_not_contains": ["知识库里没有找到可靠"]}},
    {"question": "推荐几个少油的晚饭菜", "endpoint": "chat", "assertions": {"http_status": 200, "min_answer_chars": 20, "answer_contains_any": ["少油", "晚饭", "推荐"], "answer_not_contains": ["知识库里没有找到可靠"]}},
    {"question": "推荐三个适合孩子吃的不辣菜", "endpoint": "chat", "assertions": {"http_status": 200, "min_answer_chars": 20, "answer_contains_any": ["不辣", "孩子", "推荐"], "answer_not_contains": ["知识库里没有找到可靠"]}},
    {"question": "冰箱里只有豆腐和鸡蛋，可以做什么？", "endpoint": "chat", "assertions": {"http_status": 200, "min_answer_chars": 20, "answer_contains_any": ["豆腐", "鸡蛋", "可以"], "answer_not_contains": ["知识库里没有找到可靠"]}}
  ]
}
```

- [ ] **Step 6: Add Extended multi-turn reference scenario**

Append this scenario object:

```json
{
  "id": "multi_turn_reference_ext_001",
  "suite": "extended",
  "category": "multi_turn_reference",
  "session_id": "live-ref-ext-001",
  "turns": [
    {"question": "推荐三个豆腐菜", "endpoint": "chat", "assertions": {"http_status": 200, "min_answer_chars": 20, "answer_contains_any": ["豆腐", "1."], "answer_not_contains": ["知识库里没有找到可靠"]}},
    {"question": "第二个怎么做？", "endpoint": "chat", "assertions": {"http_status": 200, "min_answer_chars": 20, "answer_not_contains": ["第二个怎么做", "知识库里没有找到可靠"]}},
    {"question": "这个适合新手吗？", "endpoint": "chat", "assertions": {"http_status": 200, "min_answer_chars": 20, "answer_contains_any": ["新手", "适合", "可以"], "answer_not_contains": ["这个适合新手吗"]}},
    {"question": "谢谢", "endpoint": "chat", "assertions": {"http_status": 200, "min_answer_chars": 4, "answer_contains_any": ["不客气", "继续", "可以"]}},
    {"question": "刚才那个需要什么食材？", "endpoint": "chat", "assertions": {"http_status": 200, "min_answer_chars": 20, "answer_contains_any": ["食材", "材料", "豆腐"], "answer_not_contains": ["刚才那个需要什么食材"]}},
    {"question": "换成不辣的可以吗？", "endpoint": "chat", "assertions": {"http_status": 200, "min_answer_chars": 20, "answer_contains_any": ["不辣", "可以", "换"], "answer_not_contains": ["换成不辣的可以吗"]}},
    {"question": "那第一个和第二个哪个更快？", "endpoint": "chat", "assertions": {"http_status": 200, "min_answer_chars": 20, "answer_contains_any": ["第一个", "第二个", "更"], "answer_not_contains": ["那第一个和第二个哪个更快"]}}
  ]
}
```

- [ ] **Step 7: Add Extended substitution and constraint scenario**

Append this scenario object:

```json
{
  "id": "substitution_constraint_ext_001",
  "suite": "extended",
  "category": "substitution_constraint",
  "session_id": "live-sub-ext-001",
  "turns": [
    {"question": "麻婆豆腐怎么做？", "endpoint": "chat", "assertions": {"http_status": 200, "min_answer_chars": 20, "answer_contains_any": ["麻婆豆腐", "豆腐"], "answer_not_contains": ["麻婆豆腐怎么做"]}},
    {"question": "不能吃辣怎么办？", "endpoint": "chat", "assertions": {"http_status": 200, "min_answer_chars": 20, "answer_contains_any": ["辣", "少放", "不放"], "answer_not_contains": ["不能吃辣怎么办"]}},
    {"question": "没有豆瓣酱可以吗？", "endpoint": "chat", "assertions": {"http_status": 200, "min_answer_chars": 20, "answer_contains_any": ["豆瓣酱", "可以", "替代"], "answer_not_contains": ["没有豆瓣酱可以吗"]}},
    {"question": "能少油一点吗？", "endpoint": "chat", "assertions": {"http_status": 200, "min_answer_chars": 20, "answer_contains_any": ["少油", "油", "可以"], "answer_not_contains": ["能少油一点吗"]}},
    {"question": "这个适合带饭吗？", "endpoint": "chat", "assertions": {"http_status": 200, "min_answer_chars": 20, "answer_contains_any": ["带饭", "适合", "可以"], "answer_not_contains": ["这个适合带饭吗"]}},
    {"question": "能不能少盐？", "endpoint": "chat", "assertions": {"http_status": 200, "min_answer_chars": 20, "answer_contains_any": ["少盐", "盐", "可以"], "answer_not_contains": ["能不能少盐"]}},
    {"question": "换个不辣的豆腐菜", "endpoint": "chat", "assertions": {"http_status": 200, "min_answer_chars": 20, "answer_contains_any": ["不辣", "豆腐", "推荐"], "answer_not_contains": ["知识库里没有找到可靠"]}}
  ]
}
```

- [ ] **Step 8: Add remaining Extended low-evidence, domain, streaming, and rapid scenarios**

Append these scenario objects:

```json
{
  "id": "low_evidence_ext_001",
  "suite": "extended",
  "category": "low_evidence",
  "session_id": "live-low-ext-001",
  "turns": [
    {"question": "银河火锅鸡怎么做？", "endpoint": "chat", "assertions": {"http_status": 200, "min_answer_chars": 8, "answer_contains_any": ["没有找到", "不确定", "知识库"], "answer_not_contains": ["银河火锅鸡做法"]}},
    {"question": "空气炸彩虹豆腐需要什么？", "endpoint": "chat", "assertions": {"http_status": 200, "min_answer_chars": 8, "answer_contains_any": ["没有找到", "不确定", "知识库"], "answer_not_contains": ["空气炸彩虹豆腐需要"]}},
    {"question": "不存在的月亮鸡翅能不能少盐？", "endpoint": "chat", "assertions": {"http_status": 200, "min_answer_chars": 8, "answer_contains_any": ["没有找到", "不确定", "知识库"], "answer_not_contains": ["月亮鸡翅能不能少盐"]}}
  ]
},
{
  "id": "domain_reject_ext_001",
  "suite": "extended",
  "category": "domain_reject",
  "session_id": "live-domain-ext-001",
  "turns": [
    {"question": "怎么选机械键盘？", "endpoint": "chat", "assertions": {"http_status": 200, "min_answer_chars": 8, "answer_contains_any": ["食谱", "做菜", "菜"], "answer_not_contains": ["机械键盘"]}},
    {"question": "怎么学摄影？", "endpoint": "chat", "assertions": {"http_status": 200, "min_answer_chars": 8, "answer_contains_any": ["食谱", "做菜", "菜"], "answer_not_contains": ["摄影"]}}
  ]
},
{
  "id": "streaming_sse_ext_001",
  "suite": "extended",
  "category": "streaming_sse",
  "session_id": "live-stream-ext-001",
  "turns": [
    {"question": "麻婆豆腐怎么做？", "endpoint": "stream", "assertions": {"http_status": 200, "sse_done_event": true, "min_answer_chars": 20, "answer_contains_any": ["麻婆豆腐", "豆腐", "辣"]}}
  ]
},
{
  "id": "rapid_followup_conflict_ext_001",
  "suite": "extended",
  "category": "rapid_followup_conflict",
  "session_id": "live-rapid-ext-001",
  "turns": [
    {"question": "推荐三个不辣的鸡肉菜", "endpoint": "chat", "assertions": {"http_status": 200, "min_answer_chars": 20, "answer_contains_any": ["不辣", "鸡", "推荐"], "answer_not_contains": ["知识库里没有找到可靠"]}}
  ]
}
```

- [ ] **Step 9: Validate JSON and scenario tests**

Run:

```bash
cd code/C8
python -m json.tool e2e/scenarios/live_e2e_scenarios.json > NUL
pytest tests/test_live_e2e_scenarios.py -q
```

Expected: JSON command exits `0`; tests PASS with Core 50, Extended 35, Total 85.

- [ ] **Step 10: Commit**

```bash
git add code/C8/e2e/scenarios/live_e2e_scenarios.json code/C8/tests/test_live_e2e_scenarios.py
git commit -m "test: expand live e2e scenario matrix"
```

---

### Task 6: Runner And Report Integration Tests

**Files:**
- Modify: `code/C8/tests/test_live_e2e_runner.py`
- Modify: `code/C8/tests/test_live_e2e_reporting.py`

**Interfaces:**
- Consumes: `--suite`, `TurnResult.suite`, suite summary reporting
- Produces: regression coverage for default `all` behavior and suite-specific output

- [ ] **Step 1: Add default suite parser test**

Add to `code/C8/tests/test_live_e2e_runner.py`:

```python
def test_runner_defaults_to_all_suite():
    args = build_arg_parser().parse_args([])

    assert args.suite == "all"
```

- [ ] **Step 2: Add all-suite limit test**

Add to `code/C8/tests/test_live_e2e_runner.py`:

```python
def test_select_turns_all_suite_keeps_global_order_with_limit():
    scenarios = [
        Scenario(
            id="core-1",
            category="domain_reject",
            session_id="core-session",
            suite="core",
            turns=[ScenarioTurn(question="Python 怎么学？", endpoint="chat", assertions={})],
        ),
        Scenario(
            id="extended-1",
            category="single_recipe_detail",
            session_id="extended-session",
            suite="extended",
            turns=[ScenarioTurn(question="拍黄瓜怎么做？", endpoint="chat", assertions={})],
        ),
    ]

    selected = select_turns_for_run(scenarios, suite="all", limit_turns=2)

    assert [scenario.id for scenario, _turn in selected] == ["core-1", "extended-1"]
```

- [ ] **Step 3: Add JSONL suite serialization test**

Add to `code/C8/tests/test_live_e2e_reporting.py`:

```python
def test_jsonl_report_includes_suite(tmp_path: Path):
    jsonl = tmp_path / "run.jsonl"

    write_jsonl_report(jsonl, [_result("PASS", suite="extended")])

    assert '"suite": "extended"' in jsonl.read_text(encoding="utf-8")
```

- [ ] **Step 4: Run integration tests**

Run:

```bash
cd code/C8
pytest tests/test_live_e2e_runner.py tests/test_live_e2e_reporting.py tests/test_live_e2e_scenarios.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add code/C8/tests/test_live_e2e_runner.py code/C8/tests/test_live_e2e_reporting.py
git commit -m "test: cover live e2e suite integration"
```

---

### Task 7: Full Deterministic Verification

**Files:**
- No production changes expected
- No test file changes expected

**Interfaces:**
- Consumes: all prior tasks
- Produces: local deterministic confidence before any live model call

- [ ] **Step 1: Run focused live E2E harness tests**

Run:

```bash
cd code/C8
pytest tests/test_live_e2e_scenarios.py tests/test_live_e2e_runner.py tests/test_live_e2e_reporting.py tests/test_live_e2e_assertions.py tests/test_live_e2e_client.py tests/test_live_e2e_rate_limit.py tests/test_live_e2e_service.py -q
```

Expected: PASS.

- [ ] **Step 2: Run deterministic acceptance regression**

Run:

```bash
cd code/C8
pytest tests/test_conversation_state.py tests/test_reference_resolution.py tests/test_web_app.py -q
```

Expected: PASS. This confirms the test harness change did not move the runtime chain.

- [ ] **Step 3: Check scenario counts from Python**

Run:

```bash
cd code/C8
python -c "from pathlib import Path; from e2e.scenarios import load_scenarios, filter_scenarios_by_suite, flatten_turns; p=Path('e2e/scenarios/live_e2e_scenarios.json'); s=load_scenarios(p); print(len(flatten_turns(filter_scenarios_by_suite(s,'core'))), len(flatten_turns(filter_scenarios_by_suite(s,'extended'))), len(flatten_turns(filter_scenarios_by_suite(s,'all'))))"
```

Expected output:

```text
50 35 85
```

- [ ] **Step 4: Commit if any verification-only fixes were needed**

If no files changed, skip this step. If a small deterministic test fix was needed, commit it:

```bash
git add code/C8/e2e code/C8/tests
git commit -m "test: stabilize expanded live e2e suite"
```

---

### Task 8: Live Smoke And Expanded Acceptance Run

**Files:**
- Generated only: `code/C8/e2e/results/*.jsonl`
- Generated only: `code/C8/e2e/results/*.md`

**Interfaces:**
- Consumes: live DashScope credentials from `.env`
- Produces: real smoke, Core 50 report, Extended 35 report, All 85 report

- [ ] **Step 1: Confirm API key is available**

Run:

```bash
cd code/C8
python -c "import os; from dotenv import load_dotenv; load_dotenv('.env'); print('DASHSCOPE_API_KEY=present' if os.getenv('DASHSCOPE_API_KEY') else 'DASHSCOPE_API_KEY=missing')"
```

Expected:

```text
DASHSCOPE_API_KEY=present
```

- [ ] **Step 2: Run one-turn smoke**

Run:

```bash
cd code/C8
python e2e/live_e2e_runner.py --models qwen-plus-2025-07-28 --suite core --limit-turns 1 --delay-seconds 0 --max-retries 0
```

Expected: command exits `0`, report contains `Total turns: 1`, status summary contains PASS.

- [ ] **Step 3: Run Core 50**

Run:

```bash
cd code/C8
python e2e/live_e2e_runner.py --models qwen-plus-2025-07-28 --suite core --limit-turns 50 --delay-seconds 5 --max-retries 1
```

Expected: command exits `0`, report `## Suite Summary` shows `core | 50`.

- [ ] **Step 4: Run Extended 35**

Run:

```bash
cd code/C8
python e2e/live_e2e_runner.py --models qwen-plus-2025-07-28 --suite extended --delay-seconds 5 --max-retries 1
```

Expected: command exits `0`, report `## Suite Summary` shows `extended | 35`.

- [ ] **Step 5: Run All 85**

Run:

```bash
cd code/C8
python e2e/live_e2e_runner.py --models qwen-plus-2025-07-28 --suite all --limit-turns 85 --delay-seconds 5 --max-retries 1
```

Expected: command exits `0`, report `## Suite Summary` shows `core | 50`, `extended | 35`, and `total | 85`.

- [ ] **Step 6: Summarize acceptance**

Summarize the newest JSONL report with a deterministic command:

```bash
cd code/C8
python -c "import json; from pathlib import Path; from collections import Counter,defaultdict; files=sorted(Path('e2e/results').glob('live-e2e-*.jsonl'), key=lambda p:p.stat().st_mtime); rows=[json.loads(line) for line in files[-1].read_text(encoding='utf-8').splitlines() if line.strip()]; suites=defaultdict(list); cats=defaultdict(list); [suites[r.get('suite','core')].append(r) or cats[r['category']].append(r) for r in rows]; print('report', files[-1]); [print(f\"suite {k}: {sum(1 for r in v if r['status']=='PASS')}/{len(v)}\") for k,v in sorted(suites.items())]; print(f\"total: {sum(1 for r in rows if r['status']=='PASS')}/{len(rows)}\"); [print(f\"category {k}: {sum(1 for r in cats[k] if r['status']=='PASS')}/{len(cats[k])}\") for k in ['single_recipe_detail','recommendation_list','multi_turn_reference','substitution_constraint']]; status=Counter(r['status'] for r in rows); print('INFRA_ERROR', status.get('INFRA_ERROR',0)); print('RATE_LIMITED', status.get('RATE_LIMITED',0))"
```

The implementation is accepted when:

```text
Core >= 45/50
Total >= 75/85
single_recipe_detail >= 75%
recommendation_list >= 75%
multi_turn_reference >= 75%
substitution_constraint >= 75%
INFRA_ERROR == 0
RATE_LIMITED == 0, unless provider-side throttling is visible
```

Rationale: `75/85` is the nearest whole-turn threshold for the spec's `>= 88%` target.

- [ ] **Step 7: Commit reports only if the project already tracks live reports**

Check:

```bash
cd code/C8
git ls-files e2e/results
```

If the command prints tracked report files, commit the new report pair:

```bash
git add code/C8/e2e/results
git commit -m "test: record expanded live e2e acceptance run"
```

If the command prints nothing, leave reports untracked and paste the report paths into the final implementation summary.

---

## Self-Review Notes

- Spec coverage:
  - Core 50 preservation is covered by Task 5.
  - Extended 35 addition is covered by Task 5.
  - Suite filtering is covered by Tasks 1 and 2.
  - Suite reporting is covered by Task 4.
  - JSONL suite visibility is covered by Tasks 3 and 6.
  - Live smoke/Core/Extended/All runs are covered by Task 8.
  - No runtime architecture changes are required by any task.

- Type consistency:
  - `Scenario.suite` feeds `evaluate_assertions(suite=...)`.
  - `TurnResult.suite` feeds `summarize_results()` and Markdown failure rows.
  - `select_turns_for_run()` filters before `flatten_turns()`, preserving the spec's `--suite` semantics.

- Implementation boundary:
  - This plan deliberately avoids edits to `main.py`, `web_app.py`, `rag_modules/*`, prompts, retrieval, and state writeback.
  - The only live behavior change is which test turns are selected and how their results are reported.
