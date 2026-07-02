from __future__ import annotations

import json
import threading
import uuid
from pathlib import Path
from typing import Callable, Optional

from flask import Flask, Response, jsonify, render_template, request, stream_with_context

from main import RecipeRAGSystem



def _default_system_factory() -> RecipeRAGSystem:
    system = RecipeRAGSystem()
    system.initialize_system()
    system.build_knowledge_base()
    return system


def create_app(system_factory: Optional[Callable[[], RecipeRAGSystem]] = None) -> Flask:
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
        if not question:
            return jsonify({"error": "question is required"}), 400

        system = get_system()
        answer = system.ask_question(question, stream=False, session_id=session_id)
        if isinstance(answer, dict):
            answer = answer.get("answer", "")
        return jsonify({"answer": answer})

    @app.get("/api/chat/stream")
    def chat_stream():
        question = (request.args.get("question") or "").strip()
        session_id = (request.args.get("session_id") or "default").strip() or "default"
        if not question:
            return jsonify({"error": "question is required"}), 400

        system = get_system()

        def sse_event(name: str, data: str) -> str:
            lines = data.splitlines() or [""]
            payload = "".join(f"data: {line}\n" for line in lines)
            return f"event: {name}\n{payload}\n"

        def generate():
            try:
                answer_stream = system.ask_question(question, stream=True, session_id=session_id)
                if isinstance(answer_stream, str):
                    yield sse_event("message", answer_stream)
                else:
                    for chunk in answer_stream:
                        if chunk:
                            yield sse_event("message", str(chunk))
                yield sse_event("done", "[DONE]")
            except Exception as exc:  # pragma: no cover - surfaced in UI
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
