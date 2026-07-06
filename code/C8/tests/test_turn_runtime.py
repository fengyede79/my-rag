from rag_modules.turn_runtime import (
    TurnRuntimeContext,
    append_runtime_event,
    build_version_mismatch,
    should_replan_after_mismatch,
)


def test_turn_runtime_context_starts_with_ids_and_shared_budget():
    ctx = TurnRuntimeContext.start(session_id="s1", read_state_version=3)

    assert ctx.session_id == "s1"
    assert ctx.read_state_version == 3
    assert ctx.max_replan_count == 1
    assert ctx.replan_count == 0
    assert ctx.lifecycle["status"] == "started"
    assert ctx.turn_id
    assert ctx.trace_id


def test_runtime_events_are_appended_without_replacing_existing_trace():
    ctx = TurnRuntimeContext.start(session_id="s1", read_state_version=3)

    append_runtime_event(ctx, "state_version_read", current_state_version=3)
    append_runtime_event(ctx, "stream_started", partial_answer_length=0)

    # turn_started is the first event from start()
    assert [event["event"] for event in ctx.trace_events] == [
        "turn_started",
        "state_version_read",
        "stream_started",
    ]
    # trace fields are attached to every event
    assert ctx.trace_events[1]["turn_id"] == ctx.turn_id
    assert ctx.trace_events[1]["trace_id"] == ctx.trace_id


def test_version_mismatch_result_records_expected_current_and_reason():
    mismatch = build_version_mismatch(
        expected_version=3,
        current_version=4,
        reason="concurrent_turn_or_rapid_followup",
    )

    assert mismatch == {
        "matched": False,
        "expected_version": 3,
        "current_version": 4,
        "reason": "concurrent_turn_or_rapid_followup",
    }


def test_replan_budget_is_shared_for_all_mismatches():
    ctx = TurnRuntimeContext.start(session_id="s1", read_state_version=3, max_replan_count=1)

    assert should_replan_after_mismatch(ctx) is True
    assert ctx.replan_count == 1
    assert should_replan_after_mismatch(ctx) is False
    assert ctx.replan_count == 1
