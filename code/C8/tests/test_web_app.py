import json
from pathlib import Path

from web_app import create_app


class _DummyRAGSystem:
    def ask_question(self, question, stream=False, session_id="default"):
        print("查询类型: detail")
        print("检索相关文档...")

        if stream:
            def generate():
                print("生成详细回答...")
                yield "你好，"
                yield "这是流式回答。"

            return generate()

        print("生成详细回答...")
        return "这是完整回答。"


def test_chat_stream_endpoint_only_emits_answer_chunks():
    app = create_app(system_factory=lambda: _DummyRAGSystem())
    client = app.test_client()

    response = client.get("/api/chat/stream?question=测试问题&session_id=test-session")
    payload = b"".join(response.response).decode("utf-8")

    assert response.status_code == 200
    assert response.headers["Content-Type"].startswith("text/event-stream")
    assert "data: 你好，" in payload
    assert "data: 这是流式回答。" in payload
    assert "event: done" in payload
    assert "查询类型" not in payload
    assert "检索相关文档" not in payload


def test_chat_endpoint_returns_clean_answer_json():
    app = create_app(system_factory=lambda: _DummyRAGSystem())
    client = app.test_client()

    response = client.post(
        "/api/chat",
        data=json.dumps({"question": "测试问题", "session_id": "test-session"}),
        content_type="application/json",
    )

    assert response.status_code == 200
    assert response.get_json() == {"answer": "这是完整回答。"}


def test_chat_endpoint_writes_utf8_request_logs(tmp_path: Path):
    log_path = tmp_path / "web_app.test.log"
    app = create_app(system_factory=lambda: _DummyRAGSystem(), log_path=log_path)
    client = app.test_client()

    response = client.post(
        "/api/chat",
        data=json.dumps({"question": "西湖醋鱼怎么样？", "session_id": "log-session"}),
        content_type="application/json",
    )

    assert response.status_code == 200
    assert log_path.exists()
    log_text = log_path.read_text(encoding="utf-8")
    assert "收到 /api/chat 请求" in log_text
    assert "西湖醋鱼怎么样？" in log_text
    assert "log-session" in log_text
    assert "完成 /api/chat 请求" in log_text


# ---- Task 7: stream equivalence ----


def test_stream_endpoint_is_equivalent_to_non_stream_endpoint():
    app = create_app(system_factory=lambda: _DummyRAGSystem())
    client = app.test_client()

    response = client.get("/api/chat/stream?question=测试问题&session_id=test-equiv")
    payload = b"".join(response.response).decode("utf-8")

    assert response.status_code == 200
    assert "data: 你好，" in payload
    assert "data: 这是流式回答。" in payload
    assert "event: done" in payload


def test_chat_can_return_diagnostics_when_requested():
    class FakeGeneration:
        model_name = "qwen-plus-2025-07-28"
        last_generation_trace = {"strategy": "structured", "context_doc_count": 1}

    class FakeSystem:
        generation_module = FakeGeneration()
        last_execution_result = {}

        def ask_question(self, question, stream=False, session_id="default"):
            self.last_execution_result = {
                "retrieval_trace": {
                    "strategy": "primary",
                    "quality_reason": "exact_dish_matched",
                    "selected_dishes": ["蛋炒饭"],
                    "fallback_used": False,
                },
                "retrieval_quality": {"quality_reason": "exact_dish_matched"},
                "context_pack_trace": {"context_doc_count": 1},
            }
            return "诊断回答"

    app = create_app(system_factory=lambda: FakeSystem())
    client = app.test_client()

    response = client.post(
        "/api/chat",
        json={"question": "蛋炒饭怎么做？", "session_id": "s1", "include_diagnostics": True},
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["answer"] == "诊断回答"
    assert payload["diagnostics"]["generation"]["strategy"] == "structured"
    assert payload["diagnostics"]["retrieval"]["strategy"] == "primary"
    assert payload["diagnostics"]["model_requested"] == "qwen-plus-2025-07-28"


def test_diagnostics_force_no_context_when_low_evidence():
    """When answer_type is no_result, generation_mode must be no_context
    even if last_generation_trace contains a stale strategy."""

    class FakeGeneration:
        model_name = "qwen-plus-2025-07-28"
        # Stale trace from a previous successful request
        last_generation_trace = {"strategy": "structured", "context_doc_count": 3}

    class FakeSystem:
        generation_module = FakeGeneration()
        last_execution_result = {}

        def ask_question(self, question, stream=False, session_id="default"):
            self.last_execution_result = {
                "answer_type": "no_result",
                "retrieval_trace": {
                    "strategy": "low_evidence",
                    "quality_reason": "no_candidates",
                },
                "retrieval_quality": {"quality_reason": "no_candidates"},
                "context_pack_trace": {},
            }
            return "抱歉，没有找到相关食谱。"

    app = create_app(system_factory=lambda: FakeSystem())
    client = app.test_client()

    response = client.post(
        "/api/chat",
        json={"question": "外星人怎么做？", "session_id": "s2", "include_diagnostics": True},
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["diagnostics"]["generation"]["strategy"] == "no_context"
    assert payload["diagnostics"]["generation"]["context_doc_count"] == 0
    assert payload["diagnostics"]["retrieval"]["strategy"] == "low_evidence"
