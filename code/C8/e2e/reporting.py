from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from e2e.assertions import TurnResult


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


def summarize_results(results: list[TurnResult]) -> dict:
    return {
        "total": len(results),
        "by_status": dict(Counter(result.status for result in results)),
        "by_model": dict(Counter(result.model for result in results)),
        "by_category": dict(Counter(result.category for result in results)),
        "by_generation_mode": dict(Counter(result.generation_mode or "unknown" for result in results)),
        "by_retrieval_strategy": dict(Counter(result.retrieval_strategy or "unknown" for result in results)),
        "by_suite_status": _suite_status_summary(results),
    }


def write_jsonl_report(path: Path, results: list[TurnResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")


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


def _table(counter: dict[str, int]) -> str:
    lines = ["| Key | Count |", "| --- | ---: |"]
    for key, count in sorted(counter.items()):
        lines.append(f"| {key} | {count} |")
    return "\n".join(lines)


def write_markdown_report(
    path: Path,
    *,
    run_id: str,
    models: list[str],
    delay_seconds: float,
    results: list[TurnResult],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = summarize_results(results)
    failures = [result for result in results if result.status not in {"PASS", "FLAKY"}]
    slowest = sorted(results, key=lambda result: result.latency_ms, reverse=True)[:10]

    lines = [
        "# Live E2E Report",
        "",
        f"Run ID: `{run_id}`",
        f"Models: `{', '.join(models)}`",
        f"Delay seconds: `{delay_seconds}`",
        f"Total turns: `{summary['total']}`",
        "",
        "## Suite Summary",
        "",
        _suite_table(summary["by_suite_status"]),
        "",
        "## Status Summary",
        "",
        _table(summary["by_status"]),
        "",
        "## Model Summary",
        "",
        _table(summary["by_model"]),
        "",
        "## Category Summary",
        "",
        _table(summary["by_category"]),
        "",
        "## Generation Mode Summary",
        "",
        _table(summary["by_generation_mode"]),
        "",
        "## Retrieval Strategy Summary",
        "",
        _table(summary["by_retrieval_strategy"]),
        "",
        "## Failure Table",
        "",
        "| Suite | Model | Scenario | Turn | Status | Generation | Retrieval | Quality Reason | Error |",
        "| --- | --- | --- | ---: | --- | --- | --- | --- | --- |",
    ]
    for result in failures:
        error = (result.error or "").replace("|", "\\|").replace("\n", " ")[:240]
        generation = result.generation_mode or "unknown"
        retrieval = result.retrieval_strategy or "unknown"
        quality = (result.quality_reason or "").replace("|", "\\|")[:120]
        lines.append(
            f"| {result.suite} | {result.model} | {result.scenario_id} | {result.turn_index} | {result.status} "
            f"| {generation} | {retrieval} | {quality} | {error} |"
        )

    lines.extend([
        "",
        "## Slowest Turns",
        "",
        "| Model | Scenario | Turn | Latency ms | Question |",
        "| --- | --- | ---: | ---: | --- |",
    ])
    for result in slowest:
        question = result.question.replace("|", "\\|")[:120]
        lines.append(f"| {result.model} | {result.scenario_id} | {result.turn_index} | {result.latency_ms} | {question} |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
