from rag_modules.reference_resolution import (
    guard_resolution_output,
    resolve_reference_from_snapshot,
    rewrite_query_for_execution,
)


def test_ambiguous_recommendation_followup_requires_clarification():
    snapshot = {
        "topic_state": {"mode": "recommendation_list"},
        "reference_state": {
            "current_dish": {"value": None, "source": "none", "confidence": 0.0, "updated_at": 0.0, "active": False},
            "recent_recommendations": [
                {"rank": 1, "dish_name": "蛋炒饭"},
                {"rank": 2, "dish_name": "麻辣香锅"},
            ],
            "recent_topics": [],
            "last_confirmed_target": None,
        },
        "conversation_state": {
            "last_user_query": "今天吃什么？",
            "current_user_query": "它怎么做？",
        },
        "resolution_constraints": {
            "allowed_reference_targets": ["蛋炒饭", "麻辣香锅"],
            "explicit_query_targets": [],
            "allow_external_explicit_target": False,
            "allow_default_selection": False,
            "must_clarify_if_ambiguous": True,
            "allow_topic_switch_detection": True,
            "priority_order": [
                "explicit_query_target",
                "last_confirmed_target",
                "ordinal_recommendation_reference",
                "pronoun_recommendation_reference",
                "current_dish",
            ],
        },
    }
    result = resolve_reference_from_snapshot(snapshot, llm=None)
    assert result["resolution_status"] == "ambiguous"
    assert result["next_action"] == "ask_clarification"
    assert result["writeback_eligible"] is False


def test_explicit_correction_is_not_blocked_by_old_candidates():
    snapshot = {
        "topic_state": {"mode": "single_dish"},
        "reference_state": {
            "current_dish": {"value": "宫保鸡丁", "source": "inferred", "confidence": 0.55, "updated_at": 10.0, "active": True},
            "recent_recommendations": [],
            "recent_topics": [],
            "last_confirmed_target": "宫保鸡丁",
        },
        "conversation_state": {
            "last_user_query": "它怎么做？",
            "current_user_query": "不是这个，是蛋炒饭",
        },
        "resolution_constraints": {
            "allowed_reference_targets": ["宫保鸡丁"],
            "explicit_query_targets": ["蛋炒饭"],
            "allow_external_explicit_target": True,
            "explicit_query_target_verified": True,
            "allow_default_selection": False,
            "must_clarify_if_ambiguous": True,
            "allow_topic_switch_detection": True,
            "priority_order": [
                "explicit_query_target",
                "last_confirmed_target",
                "ordinal_recommendation_reference",
                "pronoun_recommendation_reference",
                "current_dish",
            ],
        },
    }
    result = guard_resolution_output(
        resolve_reference_from_snapshot(snapshot, llm=None),
        snapshot["resolution_constraints"],
    )
    assert result["resolved_target"] == "蛋炒饭"
    assert result["next_action"] == "apply_correction"


def test_unverified_correction_target_requires_clarification():
    snapshot = {
        "topic_state": {"mode": "single_dish"},
        "reference_state": {
            "current_dish": {"value": "宫保鸡丁", "source": "inferred", "confidence": 0.55, "updated_at": 10.0, "active": True},
            "recent_recommendations": [],
            "recent_topics": [],
            "last_confirmed_target": "宫保鸡丁",
        },
        "conversation_state": {
            "last_user_query": "它怎么做？",
            "current_user_query": "不是这个，是那个简单点的",
        },
        "resolution_constraints": {
            "allowed_reference_targets": ["宫保鸡丁"],
            "explicit_query_targets": ["那个简单点的"],
            "allow_external_explicit_target": True,
            "explicit_query_target_verified": False,
            "allow_default_selection": False,
            "must_clarify_if_ambiguous": True,
            "allow_topic_switch_detection": True,
            "priority_order": [
                "explicit_query_target",
                "last_confirmed_target",
                "ordinal_recommendation_reference",
                "pronoun_recommendation_reference",
                "current_dish",
            ],
        },
    }
    result = resolve_reference_from_snapshot(snapshot, llm=None)
    assert result["next_action"] == "ask_clarification"
    assert result["decision_basis"] == "ambiguous"


def test_correction_rewrites_query_for_retrieval():
    execution_plan = {"action": "apply_correction"}
    resolution = {"resolved_target": "蛋炒饭"}
    query_plan = {"route_type": "detail", "content_type": "steps"}
    rewritten = rewrite_query_for_execution("不是这个，是蛋炒饭", execution_plan, resolution, query_plan)
    assert rewritten == "蛋炒饭怎么做"


# ---- Task 3: Ordinal / cleaned dish / implicit followup resolution tests ----


def test_ordinal_reference_resolves_to_recommendation_rank():
    snapshot = {
        "topic_state": {"mode": "recommendation_list"},
        "reference_state": {
            "current_dish": {"value": None, "source": "none", "confidence": 0.0, "updated_at": 0.0, "active": False},
            "recent_recommendations": [
                {"rank": 1, "dish_name": "扬州炒饭"},
                {"rank": 2, "dish_name": "麻婆豆腐"},
            ],
            "recent_topics": [],
            "last_confirmed_target": None,
        },
        "conversation_state": {"current_user_query": "第二个怎么做？"},
        "resolution_constraints": {
            "allowed_reference_targets": ["扬州炒饭", "麻婆豆腐"],
            "explicit_query_targets": [],
            "allow_external_explicit_target": False,
            "ordinal_reference": {"rank": 2, "raw_text": "第二个", "remaining_query": "怎么做"},
            "cleaned_explicit_dish": None,
            "implicit_followup": {"enabled": False, "remaining_query": "", "requires_single_active_dish": True},
            "must_clarify_if_ambiguous": True,
        },
    }

    result = resolve_reference_from_snapshot(snapshot, llm=None)

    assert result["resolution_status"] == "resolved"
    assert result["resolved_target"] == "麻婆豆腐"
    assert result["target_source"] == "ordinal_recommendation_reference"
    assert result["next_action"] == "apply_reference_resolution"
    assert result["writeback_eligible"] is True


def test_ordinal_reference_out_of_range_asks_clarification():
    snapshot = {
        "topic_state": {"mode": "recommendation_list"},
        "reference_state": {
            "current_dish": {"value": None, "source": "none", "confidence": 0.0, "updated_at": 0.0, "active": False},
            "recent_recommendations": [{"rank": 1, "dish_name": "扬州炒饭"}],
            "recent_topics": [],
            "last_confirmed_target": None,
        },
        "conversation_state": {"current_user_query": "第二个怎么做？"},
        "resolution_constraints": {
            "allowed_reference_targets": ["扬州炒饭"],
            "explicit_query_targets": [],
            "allow_external_explicit_target": False,
            "ordinal_reference": {"rank": 2, "raw_text": "第二个", "remaining_query": "怎么做"},
            "cleaned_explicit_dish": None,
            "implicit_followup": {"enabled": False, "remaining_query": "", "requires_single_active_dish": True},
            "must_clarify_if_ambiguous": True,
        },
    }

    result = resolve_reference_from_snapshot(snapshot, llm=None)

    assert result["resolution_status"] == "ambiguous"
    assert result["next_action"] == "ask_clarification"


def test_cleaned_explicit_dish_resolves_without_old_candidates():
    snapshot = {
        "topic_state": {"mode": "single_dish"},
        "reference_state": {
            "current_dish": {"value": "宫保鸡丁", "source": "explicit_query", "confidence": 1.0, "updated_at": 1.0, "active": True},
            "recent_recommendations": [],
            "recent_topics": [],
            "last_confirmed_target": "宫保鸡丁",
        },
        "conversation_state": {"current_user_query": "那蛋炒饭需要哪些食材？"},
        "resolution_constraints": {
            "allowed_reference_targets": ["宫保鸡丁"],
            "explicit_query_targets": [],
            "allow_external_explicit_target": True,
            "ordinal_reference": None,
            "cleaned_explicit_dish": {"value": "蛋炒饭", "removed_prefix": "那"},
            "implicit_followup": {"enabled": False, "remaining_query": "", "requires_single_active_dish": True},
            "must_clarify_if_ambiguous": True,
        },
    }

    result = resolve_reference_from_snapshot(snapshot, llm=None)

    assert result["resolved_target"] == "蛋炒饭"
    assert result["target_source"] == "cleaned_explicit_dish"
    assert result["next_action"] == "apply_reference_resolution"


def test_implicit_followup_uses_single_active_current_dish():
    snapshot = {
        "topic_state": {"mode": "single_dish"},
        "reference_state": {
            "current_dish": {"value": "蛋炒饭", "source": "explicit_query", "confidence": 1.0, "updated_at": 1.0, "active": True},
            "recent_recommendations": [],
            "recent_topics": [],
            "last_confirmed_target": "蛋炒饭",
        },
        "conversation_state": {"current_user_query": "有什么小技巧别粘锅？"},
        "resolution_constraints": {
            "allowed_reference_targets": ["蛋炒饭"],
            "explicit_query_targets": [],
            "allow_external_explicit_target": False,
            "ordinal_reference": None,
            "cleaned_explicit_dish": None,
            "implicit_followup": {"enabled": True, "remaining_query": "有什么小技巧别粘锅", "requires_single_active_dish": True},
            "must_clarify_if_ambiguous": True,
        },
    }

    result = resolve_reference_from_snapshot(snapshot, llm=None)

    assert result["resolved_target"] == "蛋炒饭"
    assert result["target_source"] == "implicit_single_dish_followup"


# ---- Task 4: Rewrite tests for ordinal / cleaned dish / implicit followup ----


def test_ordinal_resolution_rewrites_to_target_plus_remaining_query():
    execution_plan = {"action": "apply_reference_resolution"}
    resolution = {
        "resolved_target": "麻婆豆腐",
        "target_source": "ordinal_recommendation_reference",
    }
    query_plan = {
        "route_type": "detail",
        "filters": {"content_type": "steps"},
        "content_type": "steps",
    }

    rewritten = rewrite_query_for_execution("第二个怎么做？", execution_plan, resolution, query_plan)

    assert rewritten == "麻婆豆腐怎么做"


def test_cleaned_dish_resolution_rewrites_without_discourse_prefix():
    execution_plan = {"action": "apply_reference_resolution"}
    resolution = {
        "resolved_target": "蛋炒饭",
        "target_source": "cleaned_explicit_dish",
    }
    query_plan = {
        "route_type": "detail",
        "filters": {"content_type": "ingredients"},
        "content_type": "ingredients",
    }

    rewritten = rewrite_query_for_execution("那蛋炒饭需要哪些食材？", execution_plan, resolution, query_plan)

    assert rewritten == "蛋炒饭需要哪些食材"


def test_implicit_followup_rewrites_to_current_dish_query():
    execution_plan = {"action": "apply_reference_resolution"}
    resolution = {
        "resolved_target": "蛋炒饭",
        "target_source": "implicit_single_dish_followup",
    }
    query_plan = {
        "route_type": "detail",
        "filters": {"content_type": "tips"},
        "content_type": "tips",
    }

    rewritten = rewrite_query_for_execution("有什么小技巧别粘锅？", execution_plan, resolution, query_plan)

    assert rewritten == "蛋炒饭有什么小技巧别粘锅"
