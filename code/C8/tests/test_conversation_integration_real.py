import os
import pytest

from web_app import create_app


@pytest.mark.real_integration
def test_recommendation_followup_requires_clarification_not_wrong_inheritance():
    if not os.getenv("DASHSCOPE_API_KEY"):
        pytest.skip("DASHSCOPE_API_KEY is required for real integration tests")

    app = create_app()
    client = app.test_client()
    session_id = "real-followup-session"

    first = client.post("/api/chat", json={"question": "今天吃什么？", "session_id": session_id})
    assert first.status_code == 200

    second = client.post("/api/chat", json={"question": "它怎么做？", "session_id": session_id})
    assert second.status_code == 200

    answer = second.get_json()["answer"]
    assert "今天吃什么怎么做" not in answer
    assert ("第几个推荐菜" in answer) or ("直接告诉我菜名" in answer)


@pytest.mark.real_integration
def test_user_correction_overrides_previous_inference():
    if not os.getenv("DASHSCOPE_API_KEY"):
        pytest.skip("DASHSCOPE_API_KEY is required for real integration tests")

    app = create_app()
    client = app.test_client()
    session_id = "real-correction-session"

    first = client.post("/api/chat", json={"question": "宫保鸡丁怎么做？", "session_id": session_id})
    assert first.status_code == 200

    second = client.post("/api/chat", json={"question": "不是这个，是蛋炒饭", "session_id": session_id})
    assert second.status_code == 200

    answer = second.get_json()["answer"]
    assert "蛋炒饭" in answer


@pytest.mark.real_integration
def test_real_ordinal_followup_uses_recommendation_rank():
    if not os.getenv("DASHSCOPE_API_KEY"):
        pytest.skip("DASHSCOPE_API_KEY is required for real integration tests")

    app = create_app()
    client = app.test_client()
    session_id = "real-ordinal-session"

    first = client.post("/api/chat", json={"question": "我晚上想吃点下饭的，有啥推荐？", "session_id": session_id})
    assert first.status_code == 200
    first_answer = first.get_json()["answer"]
    assert "2." in first_answer

    second = client.post("/api/chat", json={"question": "第二个怎么做？", "session_id": session_id})
    assert second.status_code == 200
    answer = second.get_json()["answer"]

    assert "第二个" not in answer
    assert "没有足够完整" not in answer


@pytest.mark.real_integration
def test_real_discourse_prefix_dish_name_is_cleaned():
    if not os.getenv("DASHSCOPE_API_KEY"):
        pytest.skip("DASHSCOPE_API_KEY is required for real integration tests")

    app = create_app()
    client = app.test_client()
    session_id = "real-clean-prefix-session"

    first = client.post("/api/chat", json={"question": "家里只有鸡蛋和米饭，能做什么？", "session_id": session_id})
    assert first.status_code == 200

    second = client.post("/api/chat", json={"question": "那蛋炒饭需要哪些食材？", "session_id": session_id})
    assert second.status_code == 200
    answer = second.get_json()["answer"]

    assert "那蛋炒饭" not in answer
    assert "蛋炒饭" in answer


@pytest.mark.real_integration
def test_real_single_dish_short_tip_followup_uses_current_dish():
    if not os.getenv("DASHSCOPE_API_KEY"):
        pytest.skip("DASHSCOPE_API_KEY is required for real integration tests")

    app = create_app()
    client = app.test_client()
    session_id = "real-short-followup-session"

    first = client.post("/api/chat", json={"question": "蛋炒饭怎么做？", "session_id": session_id})
    assert first.status_code == 200

    second = client.post("/api/chat", json={"question": "有什么小技巧别粘锅？", "session_id": session_id})
    assert second.status_code == 200
    answer = second.get_json()["answer"]

    assert "有什么小技巧别粘锅" not in answer
    assert "没有足够完整" not in answer
