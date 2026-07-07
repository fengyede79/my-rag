from __future__ import annotations

import json
import logging
import threading
import uuid
from pathlib import Path
from typing import Callable, Optional

from flask import Flask, Response, jsonify, render_template, request, stream_with_context

from main import RecipeRAGSystem

logger = logging.getLogger(__name__)


def _configure_file_logging(log_path: Path) -> None:
    """为 Web 应用补充稳定的 UTF-8 文件日志。"""
    root_logger = logging.getLogger()
    resolved_path = log_path.resolve()

    for handler in root_logger.handlers:
        if (
            isinstance(handler, logging.FileHandler)
            and Path(getattr(handler, "baseFilename", "")).resolve() == resolved_path
        ):
            return

    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(resolved_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    root_logger.addHandler(file_handler)
    if root_logger.level > logging.INFO:
        root_logger.setLevel(logging.INFO)


def _default_system_factory() -> RecipeRAGSystem:
    system = RecipeRAGSystem()
    system.initialize_system()
    system.build_knowledge_base()
    return system


def _build_live_diagnostics(system: RecipeRAGSystem) -> dict:
    execution = getattr(system, "last_execution_result", {}) or {}
    generation_module = getattr(system, "generation_module", None)
    generation_trace = getattr(generation_module, "last_generation_trace", {}) or {}
    retrieval_trace = execution.get("retrieval_trace") or {}
    retrieval_quality = execution.get("retrieval_quality") or {}
    context_trace = execution.get("context_pack_trace") or {}
    model_requested = (
        getattr(generation_module, "model_name", None)
        or getattr(getattr(system, "config", None), "llm_model", None)
    )

    generation_strategy = generation_trace.get("strategy")
    context_doc_count = generation_trace.get("context_doc_count")

    # When retrieval ended in low_evidence / no_result the generation module
    # may not have run at all, so its last_generation_trace can be stale from
    # a previous request.  Force the truth here.
    if execution.get("answer_type") == "no_result":
        generation_strategy = "no_context"
        context_doc_count = 0
    elif not generation_strategy and execution.get("answer_mode") == "recommendation":
        generation_strategy = "list_template"

    if context_doc_count is None:
        context_doc_count = context_trace.get("context_doc_count")

    def first_present(primary: dict, secondary: dict, key: str):
        if key in primary:
            return primary.get(key)
        return secondary.get(key)

    return {
        "model_requested": model_requested,
        "generation": {
            "strategy": generation_strategy,
            "context_doc_count": context_doc_count,
            "content_type": generation_trace.get("content_type"),
        },
        "retrieval": {
            "strategy": retrieval_trace.get("strategy"),
            "quality_reason": first_present(retrieval_trace, retrieval_quality, "quality_reason"),
            "selected_dishes": first_present(retrieval_trace, retrieval_quality, "selected_dishes"),
            "fallback_used": first_present(retrieval_trace, retrieval_quality, "fallback_used"),
            "relaxed_filter": first_present(retrieval_trace, retrieval_quality, "relaxed_filter"),
            "dish_alias_used": retrieval_trace.get("dish_alias_used"),
        },
    }


def create_app(
    system_factory: Optional[Callable[[], RecipeRAGSystem]] = None,
    log_path: Optional[Path] = None,
) -> Flask:
    _configure_file_logging(log_path or Path(__file__).with_name("web_app.runtime.log"))
    app = Flask(__name__)
    app.config["SYSTEM_FACTORY"] = system_factory or _default_system_factory
    app.config["RAG_SYSTEM"] = None
    app.config["RAG_LOCK"] = threading.Lock()

    def get_system() -> RecipeRAGSystem:
        if app.config["RAG_SYSTEM"] is None:
            with app.config["RAG_LOCK"]:
                if app.config["RAG_SYSTEM"] is None:
                    app.config["RAG_SYSTEM"] = app.config["SYSTEM_FACTORY"]()
        return app.config["RAG_SYSTEM"]

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.post("/api/chat")
    def chat():
        payload = request.get_json(silent=True) or {}
        question = (payload.get("question") or "").strip()
        session_id = (payload.get("session_id") or "default").strip() or "default"
        include_diagnostics = bool(payload.get("include_diagnostics"))
        if not question:
            return jsonify({"error": "question is required"}), 400

        logger.info("收到 /api/chat 请求: session_id=%s question=%s", session_id, question)
        system = get_system()
        result = system.ask_question(question, stream=False, session_id=session_id)
        if isinstance(result, dict):
            answer = result.get("answer", "")
        else:
            answer = result
        diagnostics = _build_live_diagnostics(system) if include_diagnostics else None
        logger.info(
            "完成 /api/chat 请求: session_id=%s answer_length=%s",
            session_id,
            len(answer or ""),
        )
        response_payload = {"answer": answer}
        if include_diagnostics:
            response_payload["diagnostics"] = diagnostics
        return jsonify(response_payload)

    @app.get("/api/chat/stream")
    def chat_stream():
        question = (request.args.get("question") or "").strip()
        session_id = (request.args.get("session_id") or "default").strip() or "default"
        if not question:
            return jsonify({"error": "question is required"}), 400

        logger.info("收到 /api/chat/stream 请求: session_id=%s question=%s", session_id, question)
        system = get_system()

        def sse_event(name: str, data: str) -> str:
            lines = data.splitlines() or [""]
            payload = "".join(f"data: {line}\n" for line in lines)
            return f"event: {name}\n{payload}\n"

        def generate():
            try:
                answer_stream = system.ask_question(question, stream=True, session_id=session_id)
                if isinstance(answer_stream, str):
                    logger.info(
                        "完成 /api/chat/stream 请求: session_id=%s mode=single_chunk answer_length=%s",
                        session_id,
                        len(answer_stream),
                    )
                    yield sse_event("message", answer_stream)
                else:
                    chunk_count = 0
                    for chunk in answer_stream:
                        if chunk:
                            chunk_count += 1
                            yield sse_event("message", str(chunk))
                    logger.info(
                        "完成 /api/chat/stream 请求: session_id=%s mode=stream chunk_count=%s",
                        session_id,
                        chunk_count,
                    )
                yield sse_event("done", "[DONE]")
            except Exception as exc:  # pragma: no cover - surfaced in UI
                logger.exception("处理 /api/chat/stream 请求失败: session_id=%s", session_id)
                yield sse_event("error", json.dumps({"message": str(exc)}, ensure_ascii=False))

        return Response(
            stream_with_context(generate()),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
