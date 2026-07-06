from rag_modules.state_update_policy import build_state_diff, classify_answer_type
from rag_modules.conversation_manager import ConversationManager


def test_stream_interrupted_becomes_stream_aborted():
    answer_type = classify_answer_type(
        {"turn_type": "domain_query"},
        {"stream_interrupted": True, "success": True},
        query_plan=None,
        resolution=None,
    )

    assert answer_type == "stream_aborted"


def test_failed_retrieval_becomes_no_result():
    answer_type = classify_answer_type(
        {"turn_type": "domain_query"},
        {"success": False},
        query_plan={"route_type": "detail"},
        resolution=None,
    )

    assert answer_type == "no_result"


def test_front_door_blocked_becomes_domain_reject():
    answer_type = classify_answer_type(
        {"turn_type": "front_door_blocked"},
        {"success": True},
        query_plan=None,
        resolution=None,
    )

    assert answer_type == "domain_reject"


def test_smalltalk_turn_becomes_smalltalk():
    answer_type = classify_answer_type(
        {"turn_type": "smalltalk"},
        {"success": True},
        query_plan=None,
        resolution=None,
    )

    assert answer_type == "smalltalk"


def test_clarification_resolution_becomes_clarification():
    answer_type = classify_answer_type(
        {"turn_type": "domain_query"},
        {"success": True},
        query_plan=None,
        resolution={"next_action": "ask_clarification"},
    )

    assert answer_type == "clarification"


def test_recommendation_result_becomes_recommendation():
    answer_type = classify_answer_type(
        {"turn_type": "domain_query"},
        {"success": True, "recommended_dishes": ["回锅肉", "麻婆豆腐"]},
        query_plan={"route_type": "list"},
        resolution=None,
    )

    assert answer_type == "recommendation"


def test_detail_query_becomes_detail():
    answer_type = classify_answer_type(
        {"turn_type": "domain_query"},
        {"success": True, "resolved_target": "蛋炒饭"},
        query_plan={"route_type": "detail"},
        resolution=None,
    )

    assert answer_type == "detail"


def test_apply_reference_resolution_becomes_detail():
    answer_type = classify_answer_type(
        {"turn_type": "domain_query"},
        {"success": True},
        query_plan={"route_type": "basic"},
        resolution={"next_action": "apply_reference_resolution", "resolved_target": "蛋炒饭"},
    )

    assert answer_type == "detail"


class DummyState:
    current_entity = "蛋炒饭"
    recent_recommendations = [{"rank": 1, "dish_name": "蛋炒饭"}]
    pending_clarification = None
    last_answer_type = None


def test_smalltalk_diff_does_not_update_business_state():
    diff = build_state_diff(
        "smalltalk",
        {"success": True},
        DummyState(),
        query_plan=None,
        resolution=None,
        answer="不客气",
        question="谢谢",
    )

    assert diff["answer_type"] == "smalltalk"
    assert diff["updates"] == {"last_answer_type": "smalltalk"}
    assert diff["append_history"] is True
    assert "current_dish" not in diff["allowed_fields"]
    assert "last_recommendation_list" not in diff["allowed_fields"]


def test_clarification_diff_sets_pending_only():
    diff = build_state_diff(
        "clarification",
        {"success": True},
        DummyState(),
        query_plan=None,
        resolution={
            "reason": "ambiguous_reference",
            "candidates": ["蛋炒饭", "扬州炒饭"],
            "clarification_question": "你指的是哪一道？",
        },
        answer="你指的是哪一道？",
        question="第一个呢",
    )

    assert diff["answer_type"] == "clarification"
    assert set(diff["allowed_fields"]) == {"pending_clarification", "last_answer_type", "history"}
    assert diff["updates"]["pending_clarification"]["reason"] == "ambiguous_reference"
    assert "current_dish" not in diff["updates"]


def test_recommendation_diff_updates_recommendations_not_current_dish():
    diff = build_state_diff(
        "recommendation",
        {"success": True, "recommended_dishes": ["回锅肉", "麻婆豆腐"]},
        DummyState(),
        query_plan={"route_type": "list"},
        resolution=None,
        answer="推荐回锅肉和麻婆豆腐",
        question="推荐两个菜",
    )

    assert diff["updates"]["last_recommendation_list"] == [
        {"rank": 1, "dish_name": "回锅肉"},
        {"rank": 2, "dish_name": "麻婆豆腐"},
    ]
    assert "current_dish" not in diff["updates"]


def test_detail_diff_requires_strong_target_evidence():
    diff = build_state_diff(
        "detail",
        {"success": True, "resolved_target": "蛋炒饭"},
        DummyState(),
        query_plan={"route_type": "detail"},
        resolution={"target_source": "ordinal_reference", "confidence": 0.8},
        answer="蛋炒饭做法",
        question="第一个怎么做",
    )

    assert diff["updates"]["current_dish"]["value"] == "蛋炒饭"
    assert diff["updates"]["current_dish"]["source"] == "ordinal_reference"
    assert diff["updates"]["current_dish"]["confidence"] == 0.8
    assert diff["history"]["entities"] == {"dish_name": "蛋炒饭"}


def test_detail_diff_uses_resolution_target_fallback():
    diff = build_state_diff(
        "detail",
        {"success": True},
        DummyState(),
        query_plan={"route_type": "basic"},
        resolution={
            "next_action": "apply_reference_resolution",
            "resolved_target": "扬州炒饭",
            "target_source": "ordinal_reference",
            "confidence": 0.8,
        },
        answer="扬州炒饭做法",
        question="第一个怎么做",
    )

    assert diff["updates"]["current_dish"]["value"] == "扬州炒饭"
    assert diff["history"]["entities"] == {"dish_name": "扬州炒饭"}


def test_no_result_diff_does_not_update_business_state():
    diff = build_state_diff(
        "no_result",
        {"success": False, "answer": "没有找到"},
        DummyState(),
        query_plan={"route_type": "detail", "dish_name": "不存在的菜"},
        resolution=None,
        answer="没有找到",
        question="不存在的菜怎么做",
    )

    assert diff["updates"] == {"last_answer_type": "no_result"}
    assert "current_dish" not in diff["allowed_fields"]


def test_stream_aborted_diff_keeps_history_without_business_state():
    diff = build_state_diff(
        "stream_aborted",
        {"stream_interrupted": True, "success": True},
        DummyState(),
        query_plan=None,
        resolution=None,
        answer="半截回答",
        question="推荐几个菜",
    )

    assert diff["updates"] == {"last_answer_type": "stream_aborted"}
    assert diff["append_history"] is True
    assert diff["history"]["question"] == "推荐几个菜"
    assert "current_dish" not in diff["allowed_fields"]


def test_apply_correction_becomes_detail():
    answer_type = classify_answer_type(
        {"turn_type": "domain_query"},
        {"success": True},
        query_plan={"route_type": "basic"},
        resolution={"next_action": "apply_correction", "resolved_target": "蛋炒饭"},
    )

    assert answer_type == "detail"


def test_apply_recommendation_diff_preserves_current_entity():
    manager = ConversationManager()
    session = manager.get_session("apply-rec")
    manager.set_current_dish("apply-rec", "蛋炒饭", source="setup", confidence=1.0)

    diff = build_state_diff(
        "recommendation",
        {"success": True, "recommended_dishes": ["回锅肉", "麻婆豆腐"]},
        session,
        answer="推荐回锅肉和麻婆豆腐",
        question="推荐两个菜",
    )
    manager.apply_state_diff("apply-rec", diff)

    assert session.current_entity == "蛋炒饭"
    assert [item["dish_name"] for item in session.recent_recommendations] == ["回锅肉", "麻婆豆腐"]
    assert session.last_answer_type == "recommendation"


def test_apply_detail_diff_sets_current_entity():
    manager = ConversationManager()
    session = manager.get_session("apply-detail")

    diff = build_state_diff(
        "detail",
        {"success": True, "resolved_target": "宫保鸡丁"},
        session,
        resolution={"target_source": "ordinal_reference", "confidence": 0.8},
        answer="宫保鸡丁做法",
        question="第一个怎么做",
    )
    manager.apply_state_diff("apply-detail", diff)

    assert session.current_entity == "宫保鸡丁"
    assert session.last_confirmed_target == "宫保鸡丁"
    assert session.current_entity_meta["source"] == "ordinal_reference"
    assert session.last_answer_type == "detail"


def test_apply_no_result_diff_preserves_current_entity():
    manager = ConversationManager()
    session = manager.get_session("apply-no-result")
    manager.set_current_dish("apply-no-result", "蛋炒饭", source="setup", confidence=1.0)

    diff = build_state_diff(
        "no_result",
        {"success": False},
        session,
        answer="没有找到",
        question="不存在的菜怎么做",
    )
    manager.apply_state_diff("apply-no-result", diff)

    assert session.current_entity == "蛋炒饭"
    assert session.last_answer_type == "no_result"
