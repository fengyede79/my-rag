from pathlib import Path

from e2e.scenarios import filter_scenarios_by_suite, flatten_turns, load_scenarios


SCENARIO_FILE = Path(__file__).resolve().parents[1] / "e2e" / "scenarios" / "live_e2e_scenarios.json"


def test_live_e2e_scenario_file_has_at_least_fifty_turns_and_required_categories():
    scenarios = load_scenarios(SCENARIO_FILE)
    turns = flatten_turns(scenarios)
    categories = {scenario.category for scenario in scenarios}

    assert len(turns) >= 50
    assert {
        "single_recipe_detail",
        "recommendation_list",
        "multi_turn_reference",
        "substitution_constraint",
        "low_evidence",
        "domain_reject",
        "streaming_sse",
        "rapid_followup_conflict",
    }.issubset(categories)


def test_flatten_turns_preserves_scenario_order_and_limit():
    scenarios = load_scenarios(SCENARIO_FILE)
    limited = flatten_turns(scenarios, limit_turns=3)

    assert len(limited) == 3
    assert [turn.question for _, turn in limited] == [
        scenarios[0].turns[0].question,
        scenarios[0].turns[1].question,
        scenarios[0].turns[2].question,
    ]


def test_each_turn_has_http_status_and_min_answer_assertion():
    scenarios = load_scenarios(SCENARIO_FILE)
    for scenario, turn in flatten_turns(scenarios):
        assert turn.endpoint in {"chat", "stream"}
        assert "http_status" in turn.assertions
        assert "min_answer_chars" in turn.assertions or turn.endpoint == "stream"
        assert scenario.session_id


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
