from rag_modules.turn_understanding import understand_turn


def _snapshot(
    current_dish=None,
    recent_recommendations=None,
    pending_clarification=False,
):
    current_meta = {
        "value": current_dish,
        "active": bool(current_dish),
        "source": "confirmed" if current_dish else "none",
        "confidence": 1.0 if current_dish else 0.0,
    }
    return {
        "reference_state": {
            "current_dish": current_meta,
            "recent_recommendations": recent_recommendations or [],
        },
        "resolution_constraints": {
            "allowed_reference_targets": [
                item["dish_name"] for item in (recent_recommendations or [])
            ] or ([current_dish] if current_dish else []),
        },
        "state_health": {
            "has_pending_clarification": pending_clarification,
        },
    }


def test_smalltalk_is_direct_and_non_retrieval():
    result = understand_turn("谢谢", _snapshot(current_dish="蛋炒饭"))

    assert result["action"] == "smalltalk"
    assert result["answer_mode_hint"] == "safe_direct"
    assert result["should_retrieve"] is False
    assert result["needs_reference_resolution"] is False


def test_harmless_out_of_domain_is_domain_reject_after_snapshot_exists():
    result = understand_turn("Python怎么学", _snapshot())

    assert result["action"] == "domain_reject"
    assert result["answer_mode_hint"] == "safe_direct"
    assert result["should_retrieve"] is False
    assert result["needs_reference_resolution"] is False


def test_recipe_list_query_retrieves_list():
    result = understand_turn("今天吃什么", _snapshot())

    assert result["action"] == "retrieve_list"
    assert result["answer_mode_hint"] == "recommendation"
    assert result["should_retrieve"] is True
    assert result["reference_trigger"] == "none"


def test_recipe_detail_query_retrieves_detail_without_reference_resolution():
    result = understand_turn("蛋炒饭怎么做", _snapshot())

    assert result["action"] == "retrieve_detail"
    assert result["answer_mode_hint"] == "recipe_detail"
    assert result["should_retrieve"] is True
    assert result["needs_reference_resolution"] is False


def test_ordinal_recipe_followup_uses_reference_resolution():
    snapshot = _snapshot(
        recent_recommendations=[
            {"dish_name": "蛋炒饭"},
            {"dish_name": "番茄炒蛋"},
        ]
    )

    result = understand_turn("第一个怎么做", snapshot)

    assert result["action"] == "retrieve_detail"
    assert result["reference_trigger"] == "ordinal_reference"
    assert result["needs_reference_resolution"] is True
    assert result["depends_on_state"] is True


def test_ordinal_constraint_followup_is_not_domain_rejected_before_snapshot():
    snapshot = _snapshot(recent_recommendations=[{"dish_name": "鸡胸肉沙拉"}])

    result = understand_turn("第一个适合减脂吗", snapshot)

    assert result["action"] == "retrieve_detail"
    assert result["reference_trigger"] == "ordinal_reference"
    assert result["needs_reference_resolution"] is True


def test_ordinal_non_recipe_intent_does_not_resolve_to_recipe_detail():
    snapshot = _snapshot(recent_recommendations=[{"dish_name": "蛋炒饭"}])

    result = understand_turn("第一个作者是谁", snapshot)

    assert result["action"] == "domain_reject"
    assert result["should_retrieve"] is False
    assert result["needs_reference_resolution"] is False


def test_pronoun_followup_with_current_dish_uses_reference_resolution():
    result = understand_turn("这个能不放辣吗", _snapshot(current_dish="宫保鸡丁"))

    assert result["action"] in {"retrieve_detail", "substitution"}
    assert result["reference_trigger"] == "pronoun"
    assert result["needs_reference_resolution"] is True


def test_pronoun_followup_without_state_requests_reference_resolution_for_clarification():
    result = understand_turn("它呢", _snapshot())

    assert result["action"] == "retrieve_detail"
    assert result["reference_trigger"] == "pronoun"
    assert result["needs_reference_resolution"] is True
