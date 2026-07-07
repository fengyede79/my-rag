from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


PASS = "PASS"
FAIL = "FAIL"
FLAKY = "FLAKY"
RATE_LIMITED = "RATE_LIMITED"
INFRA_ERROR = "INFRA_ERROR"
MODEL_ERROR = "MODEL_ERROR"
DATA_ERROR = "DATA_ERROR"
SKIPPED = "SKIPPED"


@dataclass(frozen=True)
class TurnResult:
    run_id: str
    model: str
    scenario_id: str
    category: str
    turn_index: int
    session_id: str
    endpoint: str
    question: str
    http_status: int | None
    answer: str
    status: str
    failure_class: str | None
    latency_ms: int
    attempt: int
    error: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "model": self.model,
            "scenario_id": self.scenario_id,
            "category": self.category,
            "turn_index": self.turn_index,
            "session_id": self.session_id,
            "endpoint": self.endpoint,
            "question": self.question,
            "http_status": self.http_status,
            "answer": self.answer,
            "status": self.status,
            "failure_class": self.failure_class,
            "latency_ms": self.latency_ms,
            "attempt": self.attempt,
            "error": self.error,
        }


def classify_error_text(error: str) -> str:
    text = (error or "").lower()
    if "429" in text or "rate limit" in text or "too many requests" in text:
        return RATE_LIMITED
    if "timeout" in text or "connection" in text or "port" in text or "service" in text:
        return INFRA_ERROR
    if "model" in text or "invalid_request" in text or "400" in text:
        return MODEL_ERROR
    if "data" in text or "index" in text or "vector" in text:
        return DATA_ERROR
    return INFRA_ERROR


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _failure(reasons: list[str]) -> tuple[str, str | None, str | None]:
    if reasons:
        return FAIL, FAIL, "; ".join(reasons)
    return PASS, None, None


def evaluate_assertions(
    *,
    run_id: str,
    model: str,
    scenario_id: str,
    category: str,
    session_id: str,
    turn_index: int,
    endpoint: str,
    question: str,
    http_status: int | None,
    answer: str,
    assertions: dict[str, Any],
    latency_ms: int,
    attempt: int,
    sse_done_event: bool | None,
    error: str | None,
) -> TurnResult:
    if error:
        failure_class = classify_error_text(error)
        return TurnResult(
            run_id=run_id,
            model=model,
            scenario_id=scenario_id,
            category=category,
            turn_index=turn_index,
            session_id=session_id,
            endpoint=endpoint,
            question=question,
            http_status=http_status,
            answer=answer,
            status=failure_class,
            failure_class=failure_class,
            latency_ms=latency_ms,
            attempt=attempt,
            error=error,
        )

    reasons: list[str] = []
    expected_status = assertions.get("http_status")
    if expected_status is not None and http_status != int(expected_status):
        reasons.append(f"http_status expected {expected_status} got {http_status}")

    min_chars = assertions.get("min_answer_chars")
    if min_chars is not None and len(answer.strip()) < int(min_chars):
        reasons.append(f"min_answer_chars expected {min_chars} got {len(answer.strip())}")

    contains_all = _as_list(assertions.get("answer_contains_all"))
    missing_all = [item for item in contains_all if item not in answer]
    if missing_all:
        reasons.append(f"answer_contains_all missing {missing_all}")

    contains_any = _as_list(assertions.get("answer_contains_any"))
    if contains_any and not any(item in answer for item in contains_any):
        reasons.append(f"answer_contains_any missing all {contains_any}")

    forbidden = [item for item in _as_list(assertions.get("answer_not_contains")) if item in answer]
    if forbidden:
        reasons.append(f"answer_not_contains found {forbidden}")

    regex = assertions.get("answer_regex")
    if regex and not re.search(str(regex), answer):
        reasons.append(f"answer_regex did not match {regex}")

    if assertions.get("sse_done_event") is True and sse_done_event is not True:
        reasons.append("sse_done_event expected true")

    status, failure_class, reason_text = _failure(reasons)
    return TurnResult(
        run_id=run_id,
        model=model,
        scenario_id=scenario_id,
        category=category,
        turn_index=turn_index,
        session_id=session_id,
        endpoint=endpoint,
        question=question,
        http_status=http_status,
        answer=answer,
        status=status,
        failure_class=failure_class,
        latency_ms=latency_ms,
        attempt=attempt,
        error=reason_text,
    )
