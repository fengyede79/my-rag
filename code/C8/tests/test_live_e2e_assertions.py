from e2e.assertions import TurnResult, classify_error_text, evaluate_assertions


def test_evaluate_assertions_passes_contains_and_not_contains():
    result = evaluate_assertions(
        run_id="r1",
        model="qwen-plus-2025-07-28",
        scenario_id="s1",
        suite="core",
        category="single_recipe_detail",
        session_id="sess",
        turn_index=1,
        endpoint="chat",
        question="蛋炒饭怎么做？",
        http_status=200,
        answer="蛋炒饭需要鸡蛋和米饭，先炒鸡蛋再炒饭。",
        assertions={
            "http_status": 200,
            "min_answer_chars": 10,
            "answer_contains_any": ["蛋炒饭", "番茄炒蛋"],
            "answer_contains_all": ["鸡蛋", "米饭"],
            "answer_not_contains": ["知识库里没有找到可靠"],
        },
        latency_ms=123,
        attempt=1,
        sse_done_event=None,
        error=None,
    )

    assert result.status == "PASS"
    assert result.failure_class is None


def test_evaluate_assertions_returns_fail_with_reasons():
    result = evaluate_assertions(
        run_id="r1",
        model="qwen-plus-2025-07-28",
        scenario_id="s1",
        suite="core",
        category="single_recipe_detail",
        session_id="sess",
        turn_index=1,
        endpoint="chat",
        question="蛋炒饭怎么做？",
        http_status=200,
        answer="不知道。",
        assertions={
            "http_status": 200,
            "min_answer_chars": 10,
            "answer_contains_any": ["蛋炒饭"],
        },
        latency_ms=123,
        attempt=1,
        sse_done_event=None,
        error=None,
    )

    assert result.status == "FAIL"
    assert result.failure_class == "FAIL"
    assert "min_answer_chars" in result.error
    assert "answer_contains_any" in result.error


def test_classify_rate_limit_and_model_errors():
    assert classify_error_text("HTTP 429: rate limit exceeded") == "RATE_LIMITED"
    assert classify_error_text("timeout waiting for service") == "INFRA_ERROR"
    assert classify_error_text("model not found") == "MODEL_ERROR"


def test_turn_result_json_line_masks_none_values_consistently():
    result = TurnResult(
        run_id="r1",
        model="qwen-max",
        scenario_id="s1",
        suite="core",
        category="domain_reject",
        turn_index=1,
        session_id="sess",
        endpoint="chat",
        question="Python 怎么学？",
        http_status=200,
        answer="我主要处理食谱问题。",
        status="PASS",
        failure_class=None,
        latency_ms=42,
        attempt=1,
        error=None,
    )

    payload = result.to_dict()
    assert payload["model"] == "qwen-max"
    assert payload["failure_class"] is None
    assert payload["answer"] == "我主要处理食谱问题。"


def test_turn_result_records_optional_diagnostics():
    result = evaluate_assertions(
        run_id="run",
        model="qwen-plus-2025-07-28",
        scenario_id="s1",
        suite="core",
        category="single_recipe_detail",
        session_id="sess",
        turn_index=1,
        endpoint="chat",
        question="蛋炒饭怎么做？",
        http_status=200,
        answer="蛋炒饭需要鸡蛋和米饭。",
        assertions={"http_status": 200, "answer_contains_any": ["蛋炒饭"]},
        latency_ms=10,
        attempt=1,
        sse_done_event=None,
        error=None,
        diagnostics={
            "model_requested": "qwen-plus-2025-07-28",
            "generation": {"strategy": "structured", "context_doc_count": 2},
            "retrieval": {
                "strategy": "alias_fallback",
                "quality_reason": "alias_dish_matched",
                "selected_dishes": ["西红柿炒鸡蛋"],
                "fallback_used": True,
                "dish_alias_used": "西红柿炒鸡蛋",
            },
        },
    )

    assert result.model_requested == "qwen-plus-2025-07-28"
    assert result.generation_mode == "structured"
    assert result.context_doc_count == 2
    assert result.retrieval_strategy == "alias_fallback"
    assert result.quality_reason == "alias_dish_matched"
    assert result.selected_dishes == ["西红柿炒鸡蛋"]
    assert result.fallback_used is True
    assert result.dish_alias_used == "西红柿炒鸡蛋"


def test_evaluate_assertions_records_suite_in_result_and_dict():
    result = evaluate_assertions(
        run_id="run-1",
        model="qwen-plus-2025-07-28",
        scenario_id="single_recipe_detail_001",
        suite="extended",
        category="single_recipe_detail",
        session_id="session-1",
        turn_index=1,
        endpoint="chat",
        question="拍黄瓜怎么调味？",
        http_status=200,
        answer="拍黄瓜可以用蒜、醋、生抽调味。",
        assertions={"http_status": 200, "answer_contains_any": ["黄瓜"]},
        latency_ms=100,
        attempt=1,
        sse_done_event=None,
        error=None,
        diagnostics=None,
    )

    assert result.suite == "extended"
    assert result.to_dict()["suite"] == "extended"
