from __future__ import annotations

"""Execution planning for context-first turn decisions."""


def build_execution_plan(turn_info: dict, resolution: dict | None) -> dict:
    """Convert turn understanding and reference resolution into runtime action."""
    if resolution and resolution.get("next_action") == "ask_clarification":
        return {"action": "ask_clarification", "message": resolution["clarification_question"]}
    if resolution and resolution.get("next_action") == "apply_correction":
        return {"action": "apply_correction", "message": None}
    if resolution and resolution.get("next_action") == "apply_reference_resolution":
        return {"action": "apply_reference_resolution", "message": None}

    action = turn_info.get("action")
    if action == "domain_reject":
        return {"action": "direct_domain_reject", "message": None}
    if action == "smalltalk":
        return {"action": "direct_smalltalk_reply", "message": None}
    if action == "retrieve_list":
        return {"action": "retrieve_list", "message": None}
    if action in {"retrieve_detail", "compare", "substitution", "history_answer"}:
        return {"action": "retrieve_detail", "message": None}

    if turn_info.get("response_mode") == "polite_direct_reply":
        return {"action": "direct_smalltalk_reply", "message": None}
    if turn_info.get("turn_type") == "recommendation_query":
        return {"action": "retrieve_list", "message": None}
    return {"action": "retrieve_detail", "message": None}
