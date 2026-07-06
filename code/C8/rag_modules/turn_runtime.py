"""Single-process turn runtime context for versioning and streaming lifecycle."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any


def _new_id() -> str:
    return uuid.uuid4().hex


@dataclass
class TurnRuntimeContext:
    session_id: str
    read_state_version: int
    turn_id: str
    trace_id: str
    max_replan_count: int = 1
    replan_count: int = 0
    lifecycle: dict[str, Any] = field(default_factory=dict)
    trace_events: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def start(
        cls,
        *,
        session_id: str,
        read_state_version: int,
        max_replan_count: int = 1,
    ) -> "TurnRuntimeContext":
        ctx = cls(
            session_id=session_id,
            read_state_version=read_state_version,
            turn_id=_new_id(),
            trace_id=_new_id(),
            max_replan_count=max_replan_count,
            lifecycle={
                "status": "started",
                "partial_answer_length": 0,
                "commit_business_state": False,
                "reason": None,
            },
        )
        append_runtime_event(ctx, "turn_started", current_state_version=read_state_version)
        return ctx


def append_runtime_event(ctx: TurnRuntimeContext, event: str, **fields: Any) -> None:
    ctx.trace_events.append(
        {
            "event": event,
            "turn_id": ctx.turn_id,
            "trace_id": ctx.trace_id,
            "timestamp": time.time(),
            **fields,
        }
    )


def build_version_mismatch(
    *,
    expected_version: int,
    current_version: int,
    reason: str = "concurrent_turn_or_rapid_followup",
) -> dict[str, Any]:
    return {
        "matched": False,
        "expected_version": expected_version,
        "current_version": current_version,
        "reason": reason,
    }


def should_replan_after_mismatch(ctx: TurnRuntimeContext) -> bool:
    if ctx.replan_count >= ctx.max_replan_count:
        return False
    ctx.replan_count += 1
    append_runtime_event(ctx, "replan_started", replan_count=ctx.replan_count)
    return True
