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

    def complete_query(
        self,
        session_id: str,
        query: str,
        extracted_intent: Dict[str, Any] | None = None,
    ) -> str:
        session = self.get_session(session_id)

        if not session.messages:
            logger.info(f"新会话，无需补全: {query}")
            return query

        if self._is_intent_switch(query):
            logger.info(f"检测到意图切换，重置会话: {query}")
            self._reset_session(session_id)
            return query

        current_entity = None
        if extracted_intent and extracted_intent.get("dish_name"):
            current_entity = extracted_intent["dish_name"]

        completed_query = self._resolve_entity_references(query, session.current_entity)
        completed_query = self._inherit_intent(
            completed_query,
            session.current_entity,
            session.current_intent,
        )

        if current_entity:
            with self._lock:
                session = self.sessions.get(session_id)
                if session:
                    session.current_entity = current_entity

        logger.info(f"查询补全: '{query}' -> '{completed_query}'")
        return completed_query

    def _is_intent_switch(self, query: str) -> bool:
        switch_keywords = [
            "换个话题",
            "推荐别的",
            "其他的",
            "不一样",
            "换一道",
            "另外",
            "重新",
            "换一个",
            "新的话题",
        ]
        continue_keywords = [
            "怎么做",
            "步骤",
            "做法",
            "食材",
            "原料",
            "配料",
            "材料",
            "技巧",
            "注意",
            "调味",
            "火候",
            "时间",
        ]

        has_switch = any(kw in query for kw in switch_keywords)
        has_continue = any(kw in query for kw in continue_keywords)

        if has_switch and has_continue:
            if len(query) < 10:
                return True
            continue_count = sum(1 for kw in continue_keywords if kw in query)
            switch_count = sum(1 for kw in switch_keywords if kw in query)
            return switch_count > continue_count

        return has_switch

    def _resolve_entity_references(self, query: str, current_entity: Optional[str]) -> str:
        if not current_entity:
            return query

        reference_patterns = [
            r"^它",
            r"^这个",
            r"^那个",
            r"^这道菜",
            r"^那道菜",
            r"^刚才那个",
            r"^之前那个",
            r"^前面说的",
        ]

        for pattern in reference_patterns:
            if re.search(pattern, query):
                resolved_query = re.sub(pattern, current_entity, query, count=1)
                logger.info(f"检测到指代消解: '{query}' -> '{resolved_query}'")
                return resolved_query

        return query

    def _inherit_intent(
        self,
        query: str,
        current_entity: Optional[str],
        current_intent: str,
    ) -> str:
        if not current_entity:
            return query

        if current_entity in query:
            return query

        if len(query) > 5 and any(category in query for category in ["荤菜", "素菜", "汤", "主食"]):
            return query

        intent_suffixes = {
            "detail": ["怎么做", "制作方法", "做法"],
            "list": ["推荐", "菜谱"],
            "general": ["是什么", "介绍"],
        }

        if len(query) <= 5:
            suffix = intent_suffixes.get(current_intent, ["做法"])[0]
            completed = f"{current_entity}{suffix}"
            logger.info(f"意图继承补全: '{query}' -> '{completed}'")
            return completed

        followup_keywords = [
            "怎么做",
            "做法",
            "步骤",
            "食材",
            "材料",
            "原料",
            "配料",
            "技巧",
            "介绍",
            "再说",
            "再讲",
        ]
        if any(keyword in query for keyword in followup_keywords):
            normalized_query = re.sub(r"^(再说一下|再讲一下|再说说|再讲讲|再说|再讲)", "", query).strip()
            completed = f"{current_entity}{normalized_query or query}"
            logger.info(f"多轮实体继承: '{query}' -> '{completed}'")
            return completed

        return query

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
            if entities and entities.get("dish_name"):
                session.current_entity = entities["dish_name"]

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
