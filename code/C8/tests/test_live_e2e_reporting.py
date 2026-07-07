from pathlib import Path

from e2e.assertions import TurnResult
from e2e.reporting import summarize_results, write_jsonl_report, write_markdown_report


def _result(status: str, category: str = "domain_reject") -> TurnResult:
    return TurnResult(
        run_id="run-1",
        model="qwen-plus-2025-07-28",
        scenario_id="s1",
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
