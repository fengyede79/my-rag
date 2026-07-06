"""
会话管理模块。
负责多轮问答中的会话状态、实体继承、指代消解与历史压缩。
"""

from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Message:
    role: str
    content: str
    timestamp: float
    entities: Dict[str, Any] = field(default_factory=dict)
    intent_type: str = "general"


@dataclass
class SessionState:
    session_id: str
    created_at: float
    last_active: float
    is_active: bool = True
    messages: List[Message] = field(default_factory=list)
    current_entity: Optional[str] = None
    current_intent: str = "general"
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    topic_mode: str = "none"
    recent_recommendations: List[Dict[str, Any]] = field(default_factory=list)
    recent_topics: List[str] = field(default_factory=list)
    last_confirmed_target: Optional[str] = None
    current_entity_meta: Dict[str, Any] = field(default_factory=dict)
    pending_clarification: Optional[Dict[str, Any]] = None
    last_answer_type: Optional[str] = None
    state_version: int = 0
    turn_lifecycle: Dict[str, Any] = field(default_factory=dict)


class ConversationManager:
    """线程安全的会话状态管理器。"""

    def __init__(self, max_history_turns: int = 10, expire_seconds: int = 3600):
        self.sessions: Dict[str, SessionState] = {}
        self.max_history_turns = max_history_turns
        self.expire_seconds = expire_seconds
        self._lock = threading.RLock()

    def _new_session_state(self, session_id: str) -> SessionState:
        now = time.time()
        return SessionState(
            session_id=session_id,
            created_at=now,
            last_active=now,
        )

    def get_session(self, session_id: str) -> SessionState:
        with self._lock:
            self._cleanup_expired_sessions_locked()
            session = self.sessions.get(session_id)
            if session is None:
                session = self._new_session_state(session_id)
                self.sessions[session_id] = session
                logger.info(f"创建新会话: {session_id}")
            else:
                session.last_active = time.time()
            return session

    def _cleanup_expired_sessions_locked(self):
        current_time = time.time()
        expired = [
            sid
            for sid, state in self.sessions.items()
            if current_time - state.last_active > self.expire_seconds
        ]
        for sid in expired:
            del self.sessions[sid]
            logger.info(f"清理过期会话: {sid}")

    def _reset_session(self, session_id: str):
        with self._lock:
            if session_id in self.sessions:
                self.sessions[session_id] = self._new_session_state(session_id)

    def reset_session(self, session_id: str):
        with self._lock:
            if session_id in self.sessions:
                del self.sessions[session_id]
                logger.info(f"手动重置会话: {session_id}")

    def add_interaction(
        self,
        session_id: str,
        user_query: str,
        assistant_response: str,
        intent_type: str = "general",
        entities: Dict[str, Any] | None = None,
    ):
        with self._lock:
            session = self.get_session(session_id)
            current_time = time.time()

            session.messages.append(
                Message(
                    role="user",
                    content=user_query,
                    timestamp=current_time,
                    intent_type=intent_type,
                    entities=entities or {},
                )
            )
            session.messages.append(
                Message(
                    role="assistant",
                    content=assistant_response,
                    timestamp=current_time,
                )
            )

            session.current_intent = intent_type

            self._compress_history(session)
            logger.info(f"添加对话到会话 {session_id}，当前历史: {len(session.messages)} 条")

    def _compress_history(self, session: SessionState):
        max_messages = self.max_history_turns * 2
        if len(session.messages) <= max_messages:
            return

        keep_recent = 10
        older_messages = session.messages[:-keep_recent]

        summary_parts = []
        if older_messages:
            first_user = older_messages[0]
            summary_parts.append(f"讨论过 {first_user.entities.get('dish_name', '一些菜品')}")
            last_user = older_messages[-2] if len(older_messages) >= 2 else older_messages[0]  # user/assistant 成对，[-2] 取最后一条用户消息
            summary_parts.append(f"最后问过 {last_user.entities.get('dish_name', '相关内容')}")

        summary = "；".join(summary_parts) if summary_parts else "早期对话摘要"
        summary_msg = Message(
            role="system",
            content=f"[早期对话摘要] {summary}",
            timestamp=older_messages[-1].timestamp if older_messages else time.time(),
        )
        session.messages = [summary_msg] + session.messages[-keep_recent:]
        logger.info(f"历史压缩完成，保留 {len(session.messages)} 条消息")

    def get_conversation_context(self, session_id: str, max_turns: int = 2) -> str:
        with self._lock:
            session = self.get_session(session_id)
            if not session.messages:
                return ""

            context_parts = []
            user_messages = [m for m in session.messages if m.role == "user"]

            relevant_indices = set()
            for i in range(len(user_messages)):
                if i >= len(user_messages) - max_turns:
                    for j in range(len(session.messages)):
                        if session.messages[j] == user_messages[i]:
                            relevant_indices.add(j)
                            if j + 1 < len(session.messages):
                                relevant_indices.add(j + 1)
                            break

            for idx in sorted(relevant_indices):
                msg = session.messages[idx]
                role = "用户" if msg.role == "user" else "助手"
                context_parts.append(f"{role}: {msg.content[:200]}")

            return "\n".join(context_parts)

    def get_current_entity(self, session_id: str) -> Optional[str]:
        with self._lock:
            session = self.get_session(session_id)
            return session.current_entity

    def record_recommendations(self, session_id: str, dishes: List[str]):
        """记录推荐列表，切换 topic_mode 为 recommendation_list。"""
        with self._lock:
            session = self.get_session(session_id)
            session.topic_mode = "recommendation_list"
            session.recent_recommendations = [
                {"rank": index + 1, "dish_name": dish}
                for index, dish in enumerate(dishes)
            ]

    def set_current_dish(
        self,
        session_id: str,
        dish_name: str,
        source: str,
        confidence: float,
        updated_at: float | None = None,
    ):
        """设置当前菜品及其结构化元信息。"""
        with self._lock:
            session = self.get_session(session_id)
            session.current_entity = dish_name
            session.last_confirmed_target = dish_name
            session.topic_mode = "single_dish"
            session.current_entity_meta = {
                "value": dish_name,
                "source": source,
                "confidence": confidence,
                "updated_at": updated_at or time.time(),
            }

    def set_pending_clarification(
        self,
        session_id: str,
        *,
        reason: str,
        candidates: List[str] | None,
        original_question: str,
        clarification_question: str,
    ):
        with self._lock:
            session = self.get_session(session_id)
            session.pending_clarification = {
                "reason": reason,
                "candidates": candidates or [],
                "original_question": original_question,
                "clarification_question": clarification_question,
                "updated_at": time.time(),
            }

    def clear_pending_clarification(self, session_id: str):
        with self._lock:
            session = self.get_session(session_id)
            session.pending_clarification = None

    def apply_state_diff(self, session_id: str, state_diff: dict[str, Any]) -> None:
        """Apply an approved Stage 01 state diff through existing manager helpers."""
        session = self.get_session(session_id)
        updates = state_diff.get("updates", {})

        if "last_answer_type" in updates:
            session.last_answer_type = updates["last_answer_type"]

        for field in state_diff.get("clear", []):
            if field == "pending_clarification":
                self.clear_pending_clarification(session_id)

        if "pending_clarification" in updates:
            pending = updates["pending_clarification"]
            self.set_pending_clarification(
                session_id,
                reason=pending.get("reason", "ambiguous_reference"),
                candidates=pending.get("candidates", []),
                original_question=pending.get("original_question", ""),
                clarification_question=pending.get("clarification_question", ""),
            )

        if "last_recommendation_list" in updates:
            self.record_recommendations(
                session_id,
                [item["dish_name"] for item in updates["last_recommendation_list"]],
            )

        if "current_dish" in updates:
            target = updates["current_dish"]
            self.set_current_dish(
                session_id,
                target["value"],
                source=target.get("source", "state_update_policy"),
                confidence=target.get("confidence", 0.8),
            )

        history = state_diff.get("history")
        if state_diff.get("append_history") and history:
            self.add_interaction(
                session_id,
                history.get("question", ""),
                history.get("answer", ""),
                intent_type=history.get("intent_type", state_diff.get("answer_type", "general")),
                entities=history.get("entities", {}),
            )

    def get_state_version(self, session_id: str) -> int:
        with self._lock:
            return self.get_session(session_id).state_version

    def check_state_version(self, session_id: str, expected_version: int) -> dict[str, Any]:
        with self._lock:
            current = self.get_session(session_id).state_version
            if current == expected_version:
                return {
                    "matched": True,
                    "expected_version": expected_version,
                    "current_version": current,
                    "reason": "state_version_match",
                }
            return {
                "matched": False,
                "expected_version": expected_version,
                "current_version": current,
                "reason": "state_version_mismatch",
            }

    def _record_turn_lifecycle(
        self,
        session: SessionState,
        turn_id: str,
        lifecycle: dict[str, Any],
    ) -> None:
        session.turn_lifecycle[turn_id] = dict(lifecycle)
        while len(session.turn_lifecycle) > 20:
            oldest_turn_id = next(iter(session.turn_lifecycle))
            session.turn_lifecycle.pop(oldest_turn_id, None)

    def commit_state_diff(
        self,
        session_id: str,
        state_diff: dict[str, Any],
        *,
        expected_version: int,
        lifecycle: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            session = self.get_session(session_id)
            before = session.state_version
            if before != expected_version:
                return {
                    "committed": False,
                    "reason": "state_version_mismatch",
                    "expected_version": expected_version,
                    "current_version": before,
                }
            self.apply_state_diff(session_id, state_diff)
            if lifecycle is not None:
                self._record_turn_lifecycle(
                    session,
                    state_diff.get("turn_id", "last_turn"),
                    lifecycle,
                )
            session.state_version += 1
            return {
                "committed": True,
                "state_version_before": before,
                "state_version_after": session.state_version,
            }

    def writeback_turn_state(
        self,
        *,
        session_id: str,
        question: str,
        turn_info: dict,
        query_plan: dict | None = None,
        resolution: dict | None = None,
        answer: str = "",
        execution_result: dict | None = None,
        expected_state_version: int | None = None,
        lifecycle: dict[str, Any] | None = None,
    ):
        """Write one turn using the centralized state update policy."""
        from rag_modules.state_update_policy import (
            build_state_diff,
            classify_answer_type,
        )

        execution_result = execution_result or {}

        answer_type = classify_answer_type(
            turn_info,
            execution_result,
            query_plan,
            resolution,
        )
        session = self.get_session(session_id)
        state_diff = build_state_diff(
            answer_type,
            execution_result,
            session,
            query_plan=query_plan,
            resolution=resolution,
            answer=answer,
            question=question,
        )
        if expected_state_version is None:
            self.apply_state_diff(session_id, state_diff)
            session = self.get_session(session_id)
            session.state_version += 1
            return {
                "committed": True,
                "state_version_after": session.state_version,
                "legacy_expected_version": None,
            }
        runtime = execution_result.get("runtime", {}) if execution_result else {}
        turn_id = runtime.get("turn_id", "last_turn")
        return self.commit_state_diff(
            session_id,
            {**state_diff, "turn_id": turn_id},
            expected_version=expected_state_version,
            lifecycle=lifecycle,
        )
