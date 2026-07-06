"""State update policy for typed, field-limited conversation writeback."""

from __future__ import annotations

from typing import Any


ANSWER_TYPE_ALLOWED_FIELDS = {
    "smalltalk": {"last_answer_type", "history"},
    "domain_reject": {"last_answer_type", "history"},
    "clarification": {"pending_clarification", "last_answer_type", "history"},
    "recommendation": {"last_recommendation_list", "pending_clarification", "last_answer_type", "history"},
    "detail": {"current_dish", "pending_clarification", "last_answer_type", "history"},
    "comparison": {"current_entities", "last_answer_type", "history"},
    "history_answer": {"last_answer_type", "history"},
    "low_confidence": {"last_answer_type", "history"},
    "no_result": {"last_answer_type", "history"},
    "stream_aborted": {"last_answer_type", "history"},
    "normal": {"last_answer_type", "history"},
}


def classify_answer_type(
    turn_info: dict[str, Any],
    execution_result: dict[str, Any],
    query_plan: dict[str, Any] | None,
    resolution: dict[str, Any] | None,
) -> str:
    """Classify answer_type directly from Stage 01 execution facts."""
    turn_type = turn_info.get("turn_type", "domain_query")

    if execution_result.get("stream_interrupted"):
        return "stream_aborted"
    if execution_result.get("success") is False:
        return "no_result"
    if turn_type in {"front_door_blocked", "out_of_domain"}:
        return "domain_reject"
    if turn_type in {"smalltalk", "front_door_direct_reply"}:
        return "smalltalk"
    if resolution and resolution.get("next_action") == "ask_clarification":
        return "clarification"
    if resolution and resolution.get("next_action") in {"apply_correction", "apply_reference_resolution"}:
        if execution_result.get("resolved_target") or resolution.get("resolved_target"):
            return "detail"
    if query_plan and query_plan.get("route_type") == "list":
        return "recommendation" if execution_result.get("recommended_dishes") else "normal"
    if query_plan and query_plan.get("route_type") == "detail":
        return "detail"
    if execution_result.get("resolved_target"):
        return "detail"
    return "normal"


def _ranked_recommendations(dishes: list[str]) -> list[dict[str, Any]]:
    return [
        {"rank": index + 1, "dish_name": dish}
        for index, dish in enumerate(dishes or [])
    ]


def _detail_target(
    execution_result: dict[str, Any],
    query_plan: dict[str, Any] | None,
    resolution: dict[str, Any] | None,
) -> dict[str, Any] | None:
    query_plan = query_plan or {}
    resolution = resolution or {}

    target = execution_result.get("resolved_target") or resolution.get("resolved_target")
    source = resolution.get("target_source", "state_update_policy")
    confidence = resolution.get("confidence", 0.8)

    if not target and query_plan.get("dish_name"):
        target = query_plan["dish_name"]
        source = "explicit_query"
        confidence = 1.0

    if not target:
        return None

    return {"value": target, "source": source, "confidence": confidence}


def build_state_diff(
    answer_type: str,
    execution_result: dict[str, Any],
    old_state: Any,
    *,
    query_plan: dict[str, Any] | None = None,
    resolution: dict[str, Any] | None = None,
    answer: str = "",
    question: str = "",
) -> dict[str, Any]:
    """Build an inspectable state diff without mutating session state."""
    allowed = ANSWER_TYPE_ALLOWED_FIELDS.get(answer_type, ANSWER_TYPE_ALLOWED_FIELDS["normal"])
    updates: dict[str, Any] = {"last_answer_type": answer_type}
    clear: list[str] = []

    if answer_type == "clarification":
        resolution = resolution or {}
        updates["pending_clarification"] = {
            "reason": resolution.get("reason", "ambiguous_reference"),
            "candidates": resolution.get("candidates", []),
            "original_question": question,
            "clarification_question": resolution.get("clarification_question", answer),
        }

    elif answer_type == "recommendation":
        updates["last_recommendation_list"] = _ranked_recommendations(
            execution_result.get("recommended_dishes", [])
        )
        clear.append("pending_clarification")

    elif answer_type == "detail":
        target = _detail_target(execution_result, query_plan, resolution)
        if target:
            updates["current_dish"] = target
            clear.append("pending_clarification")

    return {
        "answer_type": answer_type,
        "allowed_fields": sorted(allowed),
        "updates": {
            key: value
            for key, value in updates.items()
            if key in allowed
        },
        "clear": [
            field
            for field in clear
            if field in allowed
        ],
        "append_history": "history" in allowed,
        "history": {
            "question": question,
            "answer": answer,
            "intent_type": answer_type,
            "entities": {
                "dish_name": updates["current_dish"]["value"]
            }
            if "current_dish" in updates
            else {},
        }
        if "history" in allowed
        else None,
        "reason": answer_type,
    }
