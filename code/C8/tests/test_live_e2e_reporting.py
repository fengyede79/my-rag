from pathlib import Path

from e2e.assertions import TurnResult
from e2e.reporting import summarize_results, write_jsonl_report, write_markdown_report


def _result(
    status: str,
    category: str = "domain_reject",
    *,
    suite: str = "core",
    generation_mode: str | None = None,
    retrieval_strategy: str | None = None,
    quality_reason: str | None = None,
    dish_alias_used: str | None = None,
) -> TurnResult:
    return TurnResult(
        run_id="run-1",
        model="qwen-plus-2025-07-28",
        scenario_id="s1",
        suite=suite,
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


def test_summarize_results_counts_by_status_model_and_category():
    summary = summarize_results([_result("PASS"), _result("FAIL"), _result("RATE_LIMITED", "streaming_sse")])

    assert summary["by_status"]["PASS"] == 1
    assert summary["by_status"]["FAIL"] == 1
    assert summary["by_status"]["RATE_LIMITED"] == 1
    assert summary["by_model"]["qwen-plus-2025-07-28"] == 3
    assert summary["by_category"]["domain_reject"] == 2


def test_report_writers_create_jsonl_and_markdown(tmp_path: Path):
    results = [_result("PASS"), _result("FAIL")]
    jsonl = tmp_path / "run.jsonl"
    markdown = tmp_path / "run.md"

    write_jsonl_report(jsonl, results)
    write_markdown_report(
        markdown,
        run_id="run-1",
        models=["qwen-plus-2025-07-28"],
        delay_seconds=5,
        results=results,
    )

    assert jsonl.read_text(encoding="utf-8").count("\n") == 2
    report = markdown.read_text(encoding="utf-8")
    assert "Live E2E Report" in report
    assert "qwen-plus-2025-07-28" in report
    assert "| PASS | 1 |" in report


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
