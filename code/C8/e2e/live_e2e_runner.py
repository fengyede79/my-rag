from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e2e.assertions import INFRA_ERROR, RATE_LIMITED, TurnResult, evaluate_assertions
from e2e.client import LiveE2EClient
from e2e.rate_limit import RateLimiter
from e2e.reporting import write_jsonl_report, write_markdown_report
from e2e.scenarios import filter_scenarios_by_suite, flatten_turns, load_scenarios
from e2e.service import LiveServiceProcess


DEFAULT_SCENARIO_FILE = ROOT / "e2e" / "scenarios" / "live_e2e_scenarios.json"
DEFAULT_RESULTS_DIR = ROOT / "e2e" / "results"


def parse_models(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def result_paths(results_dir: Path, run_id: str) -> tuple[Path, Path]:
    return results_dir / f"{run_id}.jsonl", results_dir / f"{run_id}.md"


def select_turns_for_run(scenarios, *, suite: str, limit_turns: int | None):
    selected_scenarios = filter_scenarios_by_suite(scenarios, suite)
    return flatten_turns(selected_scenarios, limit_turns=limit_turns)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run live E2E acceptance against the real C8 Flask service.")
    parser.add_argument("--models", default="qwen-plus-2025-07-28")
    parser.add_argument("--limit-turns", type=int, default=50)
    parser.add_argument("--delay-seconds", type=float, default=5)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--rate-limit-cooldown-seconds", type=float, default=60)
    parser.add_argument("--stop-model-after-rate-limits", type=int, default=3)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--scenario-file", type=Path, default=DEFAULT_SCENARIO_FILE)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5058)
    parser.add_argument("--reuse-server", action="store_true")
    parser.add_argument("--stream-timeout-seconds", type=int, default=300)
    parser.add_argument("--request-timeout-seconds", type=int, default=300)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--suite", choices=["core", "extended", "all"], default="all")
    return parser


def _call_turn(client: LiveE2EClient, *, endpoint: str, question: str, session_id: str):
    if endpoint == "stream":
        return client.stream(question=question, session_id=session_id)
    return client.chat(question=question, session_id=session_id)


def run_model(
    *,
    run_id: str,
    model: str,
    args,
    project_dir: Path,
) -> list[TurnResult]:
    scenarios = load_scenarios(args.scenario_file)
    turns = select_turns_for_run(scenarios, suite=args.suite, limit_turns=args.limit_turns)
    service = LiveServiceProcess(
        project_dir=project_dir,
        host=args.host,
        port=args.port,
        model=model,
        reuse_server=args.reuse_server,
    )
    client = LiveE2EClient(
        base_url=f"http://{args.host}:{args.port}",
        request_timeout_seconds=args.request_timeout_seconds,
        stream_timeout_seconds=args.stream_timeout_seconds,
    )
    limiter = RateLimiter(
        delay_seconds=args.delay_seconds,
        rate_limit_cooldown_seconds=args.rate_limit_cooldown_seconds,
    )
    results: list[TurnResult] = []
    rate_limit_count = 0

    service.start()
    try:
        client.wait_until_ready(timeout_seconds=240)
        for turn_number, (scenario, turn) in enumerate(turns, start=1):
            final_result: TurnResult | None = None
            for attempt in range(1, args.max_retries + 2):
                limiter.wait_before_retry(attempt)
                response = _call_turn(
                    client,
                    endpoint=turn.endpoint,
                    question=turn.question,
                    session_id=scenario.session_id,
                )
                final_result = evaluate_assertions(
                    run_id=run_id,
                    model=model,
                    scenario_id=scenario.id,
                    suite=scenario.suite,
                    category=scenario.category,
                    session_id=scenario.session_id,
                    turn_index=turn_number,
                    endpoint=turn.endpoint,
                    question=turn.question,
                    http_status=response.http_status,
                    answer=response.answer,
                    assertions=turn.assertions,
                    latency_ms=response.latency_ms,
                    attempt=attempt,
                    sse_done_event=response.sse_done_event,
                    error=response.error,
                    diagnostics=response.diagnostics,
                )
                if final_result.status != RATE_LIMITED:
                    break
                rate_limit_count += 1
                limiter.wait_after_rate_limit()
                if rate_limit_count >= args.stop_model_after_rate_limits:
                    break

            assert final_result is not None
            results.append(final_result)
            print(
                f"[{model}] {turn_number}/{len(turns)} {scenario.id} "
                f"{final_result.status} {final_result.latency_ms}ms"
            )
            if args.fail_fast and final_result.status not in {"PASS", "FLAKY"}:
                break
            if final_result.status == INFRA_ERROR:
                break
            if rate_limit_count >= args.stop_model_after_rate_limits:
                break
            limiter.wait_after_turn()
    finally:
        if not args.reuse_server:
            service.stop()
    return results


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    load_dotenv(ROOT / ".env")
    if not os.getenv("DASHSCOPE_API_KEY"):
        print("DASHSCOPE_API_KEY is required for live E2E", file=sys.stderr)
        return 2

    run_id = f"live-e2e-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    models = parse_models(args.models)
    all_results: list[TurnResult] = []
    for model in models:
        all_results.extend(run_model(run_id=run_id, model=model, args=args, project_dir=ROOT))

    jsonl_path, markdown_path = result_paths(args.results_dir, run_id)
    write_jsonl_report(jsonl_path, all_results)
    write_markdown_report(
        markdown_path,
        run_id=run_id,
        models=models,
        delay_seconds=args.delay_seconds,
        results=all_results,
    )
    print(f"Wrote {jsonl_path}")
    print(f"Wrote {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
