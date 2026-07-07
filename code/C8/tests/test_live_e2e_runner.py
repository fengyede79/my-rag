from pathlib import Path

from e2e.live_e2e_runner import build_arg_parser, parse_models, result_paths


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


def test_result_paths_use_run_id_and_results_dir(tmp_path: Path):
    jsonl, markdown = result_paths(tmp_path, "live-e2e-20260707-153000")

    assert jsonl.name == "live-e2e-20260707-153000.jsonl"
    assert markdown.name == "live-e2e-20260707-153000.md"
