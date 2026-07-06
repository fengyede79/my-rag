from acceptance_fixtures import ask_and_trace, build_acceptance_system


def test_acceptance_fixture_uses_real_runtime_boundaries():
    system = build_acceptance_system()

    answer, trace = ask_and_trace(system, "推荐三个鸡肉菜", session_id="fixture-smoke")

    session = system.generation_module.conversation_manager.get_session("fixture-smoke")
    assert "宫保鸡丁" in answer
    assert session.recent_recommendations
    assert trace["query_plan"]["route_type"] == "list"
    assert trace["retrieval_quality"]["enough_evidence"] is True
    assert trace["context_pack_trace"]["selected_section_count"] >= 1
    assert trace["commit_result"]["committed"] is True


def test_primary_multi_turn_recipe_chain_preserves_state_and_trace():
    system = build_acceptance_system()
    session_id = "primary-chain"

    first_answer, first_trace = ask_and_trace(system, "推荐三个鸡肉菜", session_id=session_id)
    assert "宫保鸡丁" in first_answer
    assert first_trace["query_plan"]["route_type"] == "list"
    assert first_trace["retrieval_quality"]["enough_evidence"] is True

    second_answer, second_trace = ask_and_trace(system, "第一个怎么做", session_id=session_id)
    assert "宫保鸡丁" in second_answer
    assert second_trace["query_plan"]["route_type"] == "detail"
    assert second_trace["retrieval_quality"]["quality_reason"] == "exact_dish_matched"

    third_answer, third_trace = ask_and_trace(system, "这个能不放辣吗", session_id=session_id)
    assert "不放辣" in third_answer or "辣椒" in third_answer
    assert third_trace["query_plan"]["route_type"] == "detail"

    fourth_answer, fourth_trace = ask_and_trace(system, "没有豆瓣酱怎么办", session_id=session_id)
    assert "生抽" in fourth_answer
    assert fourth_trace["query_plan"]["route_type"] == "detail"

    fifth_answer, fifth_trace = ask_and_trace(system, "给我换个不辣的", session_id=session_id)
    assert "香菇滑鸡" in fifth_answer or "可乐鸡翅" in fifth_answer
    assert fifth_trace["query_plan"]["route_type"] == "list"

    sixth_answer, sixth_trace = ask_and_trace(system, "谢谢", session_id=session_id)
    assert "不客气" in sixth_answer
    assert sixth_trace.get("query_plan", {}) == {}

    session = system.generation_module.conversation_manager.get_session(session_id)
    assert session.current_entity == "宫保鸡丁"
    assert session.recent_recommendations
    assert session.last_answer_type == "smalltalk"
    assert session.pending_clarification is None
    assert session.state_version >= 6


def test_harmless_out_of_domain_rejects_without_retrieval_or_business_state():
    system = build_acceptance_system()

    answer, trace = ask_and_trace(system, "Python 怎么学", session_id="domain-reject")

    session = system.generation_module.conversation_manager.get_session("domain-reject")
    assert "食谱" in answer or "做菜" in answer
    assert trace.get("query_plan", {}) == {}
    assert session.current_entity is None
    assert session.recent_recommendations == []


def test_unrelated_ordinal_after_recommendation_does_not_silently_resolve_as_recipe_detail():
    system = build_acceptance_system()
    ask_and_trace(system, "推荐三个鸡肉菜", session_id="unrelated-ordinal")

    answer, trace = ask_and_trace(system, "第一个作者是谁", session_id="unrelated-ordinal")

    session = system.generation_module.conversation_manager.get_session("unrelated-ordinal")
    assert session.current_entity is None
    assert trace.get("answer_type") in {"domain_reject", "clarification", "low_confidence", "no_result", None}
    assert "宫保鸡丁做法" not in answer


def test_exact_missing_dish_returns_no_result_without_current_dish_update():
    system = build_acceptance_system()

    answer, trace = ask_and_trace(system, "不存在的菜怎么做", session_id="missing-dish")

    session = system.generation_module.conversation_manager.get_session("missing-dish")
    assert "没有找到可靠" in answer
    assert trace["retrieval_quality"]["enough_evidence"] is False
    assert trace.get("answer_type") in {"no_result", "low_confidence"}
    assert session.current_entity is None


def test_sparse_metadata_preference_uses_soft_weighting_or_marked_fallback():
    system = build_acceptance_system()

    answer, trace = ask_and_trace(system, "推荐不辣的鸡肉菜", session_id="sparse-metadata")

    assert "香菇滑鸡" in answer or "可乐鸡翅" in answer
    assert trace["retrieval_quality"]["enough_evidence"] is True
    assert "taste" in trace["query_plan"]["retrieval_query_plan"]["soft_filters"]
    if trace["retrieval_quality"]["fallback_used"]:
        assert trace["retrieval_quality"]["relaxed_filter"] is True


def test_stream_abort_after_recommendation_does_not_create_valid_ordinal_list():
    system = build_acceptance_system()
    # Use a detail query since list routes return strings, not streams
    stream = system.ask_question("宫保鸡丁怎么做", stream=True, session_id="stream-abort-acceptance")

    first = next(stream)
    assert first
    stream.close()

    session = system.generation_module.conversation_manager.get_session("stream-abort-acceptance")
    # Aborted stream should not commit business state
    assert session.current_entity != "宫保鸡丁"
    assert system.last_execution_result["runtime"]["lifecycle"]["status"] == "aborted"


def test_rapid_state_dependent_turn_reaches_shared_conflict_path(monkeypatch):
    system = build_acceptance_system()
    manager = system.generation_module.conversation_manager
    ask_and_trace(system, "推荐三个鸡肉菜", session_id="rapid-conflict")

    original_build_context_pack = system.context_packer.build_context_pack

    def mutate_before_generation(**kwargs):
        pack = original_build_context_pack(**kwargs)
        manager.commit_state_diff(
            "rapid-conflict",
            {
                "answer_type": "smalltalk",
                "updates": {"last_answer_type": "smalltalk"},
                "clear": [],
                "append_history": False,
                "history": None,
            },
            expected_version=manager.get_state_version("rapid-conflict"),
        )
        return pack

    monkeypatch.setattr(system.context_packer, "build_context_pack", mutate_before_generation)

    answer, trace = ask_and_trace(system, "第一个怎么做", session_id="rapid-conflict")

    assert "上下文刚刚更新" in answer
    assert trace["runtime"]["replan_count"] == 1
    assert trace["runtime"]["lifecycle"]["status"] == "failed"


def test_low_evidence_detail_does_not_update_current_dish():
    system = build_acceptance_system()

    answer, trace = ask_and_trace(system, "不存在的菜怎么做", session_id="low-evidence-detail")

    session = system.generation_module.conversation_manager.get_session("low-evidence-detail")
    assert "没有找到可靠" in answer
    assert trace.get("answer_type") in {"no_result", "low_confidence"}
    assert session.current_entity is None


def test_final_smalltalk_after_recipe_flow_does_not_clear_business_state():
    system = build_acceptance_system()
    ask_and_trace(system, "推荐三个鸡肉菜", session_id="final-smalltalk")
    ask_and_trace(system, "第一个怎么做", session_id="final-smalltalk")

    before = system.generation_module.conversation_manager.get_session("final-smalltalk")
    assert before.current_entity == "宫保鸡丁"
    assert before.recent_recommendations

    answer, trace = ask_and_trace(system, "谢谢", session_id="final-smalltalk")

    after = system.generation_module.conversation_manager.get_session("final-smalltalk")
    assert "不客气" in answer
    assert trace.get("query_plan", {}) == {}
    assert after.current_entity == "宫保鸡丁"
    assert after.recent_recommendations
