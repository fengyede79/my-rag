from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from e2e.assertions import TurnResult


def summarize_results(results: list[TurnResult]) -> dict:
    return {
        "total": len(results),
        "by_status": dict(Counter(result.status for result in results)),
        "by_model": dict(Counter(result.model for result in results)),
        "by_category": dict(Counter(result.category for result in results)),
    }


def write_jsonl_report(path: Path, results: list[TurnResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")


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
        "## Failure Table",
        "",
        "| Model | Scenario | Turn | Status | Error |",
        "| --- | --- | ---: | --- | --- |",
    ]
    for result in failures:
        error = (result.error or "").replace("|", "\\|").replace("\n", " ")[:240]
        lines.append(f"| {result.model} | {result.scenario_id} | {result.turn_index} | {result.status} | {error} |")

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
