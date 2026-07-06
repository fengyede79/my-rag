"""
结构化会话快照构建模块。
将 SessionState 转换为统一的 snapshot dict，供后续 reference_resolution 和 execution_planner 消费。
"""

from __future__ import annotations

import re
import time

CHINESE_ORDINAL_TO_RANK = {
    "一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
}

ORDINAL_COMMENT_PATTERNS = (
    "看起来不错",
    "看起来挺好",
    "不错",
    "挺好",
    "可以",
)

DISCOURSE_PREFIXES = ("那", "这个", "这道", "刚才那个", "刚才这道")
DETAIL_SUFFIXES = (
    "需要哪些食材", "需要什么食材", "有什么食材",
    "怎么做", "做法说一下", "做法", "有什么小技巧", "有哪些技巧",
)

IMPLICIT_FOLLOWUP_KEYWORDS = ("怎么做", "做法", "食材", "技巧", "粘锅", "难不难", "要多久", "热量")


def _strip_ordinal_comment(text: str) -> str:
    cleaned = text.strip("，,。！？? ")
    for pattern in sorted(ORDINAL_COMMENT_PATTERNS, key=len, reverse=True):
        cleaned = cleaned.replace(pattern, "")
    return cleaned.strip("，,。！？? ")


def _extract_ordinal_reference(current_query: str) -> dict | None:
    text = current_query.strip()
    match = re.match(
        r"^(第\s*(?P<cn>[一二三四五])\s*个|第\s*(?P<num>[1-5])\s*个|(?P<plain>[1-5])\s*号)(?P<rest>.*)$",
        text,
    )
    if not match:
        return None

    rank = None
    if match.group("cn"):
        rank = CHINESE_ORDINAL_TO_RANK[match.group("cn")]
    elif match.group("num"):
        rank = int(match.group("num"))
    elif match.group("plain"):
        rank = int(match.group("plain"))

    raw_text = text[: match.end() - len(match.group("rest"))]
    remaining_query = _strip_ordinal_comment(match.group("rest") or "")
    if not remaining_query:
        remaining_query = "怎么做"

    return {
        "rank": rank,
        "raw_text": raw_text.strip(),
        "remaining_query": remaining_query,
    }


def _extract_cleaned_explicit_dish(current_query: str) -> dict | None:
    text = current_query.strip().rstrip("?!？！。")
    for prefix in sorted(DISCOURSE_PREFIXES, key=len, reverse=True):
        if not text.startswith(prefix):
            continue
        candidate = text[len(prefix):]
        for suffix in sorted(DETAIL_SUFFIXES, key=len, reverse=True):
            if suffix in candidate:
                dish = candidate.split(suffix, 1)[0].strip("，,。 ")
                if 2 <= len(dish) <= 12:
                    return {"value": dish, "removed_prefix": prefix}
    return None


def _extract_implicit_followup(current_query: str) -> dict:
    text = current_query.strip().rstrip("?!？！。")
    enabled = len(text) <= 18 and any(keyword in text for keyword in IMPLICIT_FOLLOWUP_KEYWORDS)
    return {
        "enabled": enabled,
        "remaining_query": text,
        "requires_single_active_dish": True,
    }


def _extract_preference_constraints(current_query: str) -> dict:
    text = current_query.strip()
    preferences = {"taste": [], "meal": [], "difficulty": [], "style": []}
    if "清淡" in text:
        preferences["taste"].append("清淡")
    if "下饭" in text:
        preferences["style"].append("下饭")
    if "新手" in text or "简单" in text:
        preferences["difficulty"].append("新手" if "新手" in text else "简单")
    if "早餐" in text:
        preferences["meal"].append("早餐")
    return preferences


def _extract_explicit_query_targets(current_query: str) -> list[str]:
    correction_match = re.match(r"^不是这个，是(.+)$", current_query.strip())
    if correction_match:
        return [correction_match.group(1).strip()]
    return []


def _is_verified_explicit_target(target: str) -> bool:
    return len(target) >= 2 and all(ch not in target for ch in "这个那个简单一点一些")


def build_conversation_snapshot(
    session,
    current_query: str,
    now_ts: float | None = None,
) -> dict:
    current_meta = dict(session.current_entity_meta or {})
    now_ts = now_ts or time.time()
    if current_meta:
        age_seconds = now_ts - current_meta["updated_at"]
        current_meta["active"] = not (
            current_meta["source"] == "inferred" and age_seconds > 900
        )
    else:
        current_meta = {
            "value": None,
            "source": "none",
            "confidence": 0.0,
            "updated_at": 0.0,
            "active": False,
        }

    explicit_query_targets = _extract_explicit_query_targets(current_query)
    explicit_query_target_verified = all(
        _is_verified_explicit_target(target) for target in explicit_query_targets
    ) if explicit_query_targets else False

    ordinal_reference = _extract_ordinal_reference(current_query)
    cleaned_explicit_dish = _extract_cleaned_explicit_dish(current_query)
    implicit_followup = _extract_implicit_followup(current_query)
    preference_constraints = _extract_preference_constraints(current_query)

    return {
        "topic_state": {
            "mode": session.topic_mode,
            "current_topic": current_meta["value"] if current_meta["active"] else None,
            "pending_topic": None,
            "last_topic_source": current_meta["source"],
        },
        "reference_state": {
            "current_dish": current_meta,
            "recent_recommendations": list(session.recent_recommendations),
            "recent_topics": list(session.recent_topics),
            "last_confirmed_target": session.last_confirmed_target,
        },
        "conversation_state": {
            "last_user_query": next((m.content for m in reversed(session.messages) if m.role == "user"), ""),
            "current_user_query": current_query,
            "last_system_action": session.current_intent,
            "last_system_response_summary": "",
            "recent_turns": [
                {"role": m.role, "content": m.content, "intent_type": m.intent_type}
                for m in session.messages[-6:]
            ],
        },
        "resolution_constraints": {
            "allowed_reference_targets": [
                item["dish_name"] for item in session.recent_recommendations
            ] or ([current_meta["value"]] if current_meta["active"] and current_meta["value"] else []),
            "explicit_query_targets": explicit_query_targets,
            "allow_external_explicit_target": bool(explicit_query_targets),
            "explicit_query_target_verified": explicit_query_target_verified,
            "allow_default_selection": False,
            "must_clarify_if_ambiguous": True,
            "allow_topic_switch_detection": True,
            "ordinal_reference": ordinal_reference,
            "cleaned_explicit_dish": cleaned_explicit_dish,
            "implicit_followup": implicit_followup,
            "preference_constraints": preference_constraints,
            "priority_order": [
                "explicit_query_target",
                "cleaned_explicit_dish",
                "ordinal_recommendation_reference",
                "last_confirmed_target",
                "implicit_single_dish_followup",
                "pronoun_recommendation_reference",
                "current_dish",
            ],
        },
        "state_health": {
            "state_version": 1,
            "last_reliable_turn_id": None,
            "has_ambiguous_reference": not current_meta["active"] and bool(current_meta["value"]),
            "has_pending_clarification": bool(session.pending_clarification),
        },
    }
