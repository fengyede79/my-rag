from pathlib import Path

from e2e.client import HTTPResult
from e2e.live_e2e_runner import build_arg_parser, parse_models, result_paths, run_model, select_turns_for_run
from e2e.scenarios import Scenario, ScenarioTurn


def test_parse_models_splits_and_strips_values():
    assert parse_models("qwen-max, qwen-plus-2025-07-28") == ["qwen-max", "qwen-plus-2025-07-28"]


def test_arg_parser_defaults_match_spec():
    args = build_arg_parser().parse_args([])

    assert args.models == "qwen-plus-2025-07-28"
    assert args.limit_turns == 50
    assert args.delay_seconds == 5
    assert args.max_retries == 3
    assert args.rate_limit_cooldown_seconds == 60
    assert args.host == "127.0.0.1"
    assert args.port == 5058
    assert args.request_timeout_seconds == 300
    assert args.stream_timeout_seconds == 300


def test_result_paths_use_run_id_and_results_dir(tmp_path: Path):
    jsonl, markdown = result_paths(tmp_path, "live-e2e-20260707-153000")

    assert jsonl.name == "live-e2e-20260707-153000.jsonl"
    assert markdown.name == "live-e2e-20260707-153000.md"


def test_run_model_stops_after_infra_error(monkeypatch):
    class FakeService:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def start(self):
            return None

        def stop(self):
            return None

    class FakeClient:
        calls = 0

        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def wait_until_ready(self, timeout_seconds):
            return None

        def chat(self, *, question, session_id):
            FakeClient.calls += 1
            return HTTPResult(http_status=None, answer="", latency_ms=300000, error="timed out")

        def stream(self, *, question, session_id):
            raise AssertionError("stream should not be called")

    monkeypatch.setattr("e2e.live_e2e_runner.LiveServiceProcess", FakeService)
    monkeypatch.setattr("e2e.live_e2e_runner.LiveE2EClient", FakeClient)

    args = build_arg_parser().parse_args(["--limit-turns", "2", "--delay-seconds", "0", "--max-retries", "0"])
    results = run_model(run_id="run", model="qwen-plus-2025-07-28", args=args, project_dir=Path.cwd())

    assert len(results) == 1
    assert results[0].status == "INFRA_ERROR"
    assert FakeClient.calls == 1


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
