"""Context-aware turn understanding for the runtime pipeline."""

from __future__ import annotations

from typing import Any, Dict


SMALLTALK_EXACT = {
    "你好",
    "您好",
    "谢谢",
    "多谢",
    "哈哈",
    "你是谁",
    "你能做什么",
}

OUT_OF_DOMAIN_KEYWORDS = {
    "Python",
    "python",
    "Java",
    "C++",
    "代码",
    "编程",
    "算法",
    "股票",
    "房价",
    "天气",
    "新闻",
    "政治",
    "历史",
    "手机壳",
}

RECIPE_ALLOW_SIGNALS = {
    "菜",
    "食谱",
    "做法",
    "怎么做",
    "怎么煮",
    "怎么炒",
    "食材",
    "材料",
    "配料",
    "推荐",
    "吃什么",
    "减脂",
    "热量",
    "不辣",
    "辣",
    "下饭",
    "早餐",
    "午餐",
    "晚餐",
    "蛋",
    "鸡",
    "肉",
    "鱼",
    "虾",
    "饭",
    "面",
    "汤",
}

LIST_SIGNALS = {"吃什么", "推荐", "来几个", "有哪些", "有啥"}
DETAIL_SIGNALS = {"怎么做", "做法", "食材", "材料", "配料", "步骤", "技巧", "热量", "适合", "减脂", "不放", "替换", "为什么"}
ORDINAL_PREFIXES = ("第一个", "第二个", "第三个", "第四个", "第五个", "1号", "2号", "3号", "4号", "5号")
PRONOUN_PREFIXES = ("这个", "这个菜", "这道", "这道菜", "它", "那个", "那道", "那道菜")
CORRECTION_PREFIXES = ("不是这个，是", "不是，是", "不对，是", "不，是")
UNSUPPORTED_ORDINAL_INTENTS = {"作者", "发明", "哪年", "历史", "出处", "谁写"}


def _base_result(
    *,
    action: str,
    answer_mode_hint: str,
    should_retrieve: bool,
    depends_on_state: bool = False,
    needs_reference_resolution: bool = False,
    domain_confidence: float = 1.0,
    reference_trigger: str = "none",
    reason: str,
) -> Dict[str, Any]:
    return {
        "action": action,
        "answer_mode_hint": answer_mode_hint,
        "depends_on_state": depends_on_state,
        "needs_reference_resolution": needs_reference_resolution,
        "domain_confidence": domain_confidence,
        "reference_trigger": reference_trigger,
        "should_retrieve": should_retrieve,
        "reason": reason,
        "turn_type": _legacy_turn_type(action),
        "response_mode": "retrieve_answer" if should_retrieve else "polite_direct_reply",
        "should_update_topic_state": action == "retrieve_list",
        "should_update_entity_state": action == "retrieve_detail",
        "should_run_reference_resolution": needs_reference_resolution,
    }


def _legacy_turn_type(action: str) -> str:
    if action == "retrieve_list":
        return "recommendation_query"
    if action in {"retrieve_detail", "compare", "substitution"}:
        return "followup_query"
    if action in {"smalltalk", "domain_reject"}:
        return "direct_answer"
    return "domain_query"


def _normalize(question: str) -> str:
    return question.strip().rstrip("?!？！。")


def _has_recent_recommendations(snapshot: dict) -> bool:
    return bool(snapshot.get("reference_state", {}).get("recent_recommendations") or [])


def _has_current_dish(snapshot: dict) -> bool:
    current = snapshot.get("reference_state", {}).get("current_dish", {})
    return bool(current.get("active") and current.get("value"))


def _starts_with_any(text: str, prefixes: tuple[str, ...]) -> bool:
    return any(text.startswith(prefix) for prefix in prefixes)


def _has_recipe_signal(text: str) -> bool:
    return any(signal in text for signal in RECIPE_ALLOW_SIGNALS)


def _has_out_of_domain_signal(text: str) -> bool:
    return any(signal in text for signal in OUT_OF_DOMAIN_KEYWORDS)


def _has_detail_signal(text: str) -> bool:
    return any(signal in text for signal in DETAIL_SIGNALS)


def _has_list_signal(text: str) -> bool:
    return any(signal in text for signal in LIST_SIGNALS)


def _has_unsupported_ordinal_intent(text: str) -> bool:
    return any(signal in text for signal in UNSUPPORTED_ORDINAL_INTENTS)


def understand_turn(question: str, snapshot: dict) -> Dict[str, Any]:
    """Classify a turn after a lightweight session snapshot exists."""
    text = _normalize(question)

    if text in SMALLTALK_EXACT:
        return _base_result(
            action="smalltalk",
            answer_mode_hint="safe_direct",
            should_retrieve=False,
            reason="smalltalk_exact",
        )

    if _starts_with_any(text, CORRECTION_PREFIXES):
        return _base_result(
            action="retrieve_detail",
            answer_mode_hint="recipe_detail",
            should_retrieve=True,
            depends_on_state=True,
            needs_reference_resolution=True,
            reference_trigger="correction",
            reason="correction_reference",
        )

    if _starts_with_any(text, ORDINAL_PREFIXES):
        if _has_unsupported_ordinal_intent(text):
            return _base_result(
                action="domain_reject",
                answer_mode_hint="safe_direct",
                should_retrieve=False,
                depends_on_state=_has_recent_recommendations(snapshot),
                reason="ordinal_non_recipe_intent",
            )
        return _base_result(
            action="retrieve_detail",
            answer_mode_hint="recipe_detail",
            should_retrieve=True,
            depends_on_state=True,
            needs_reference_resolution=True,
            reference_trigger="ordinal_reference",
            reason="ordinal_recipe_followup",
        )

    if _starts_with_any(text, PRONOUN_PREFIXES):
        action = "substitution" if any(token in text for token in ("不放", "替换", "换成")) else "retrieve_detail"
        return _base_result(
            action=action,
            answer_mode_hint="substitution" if action == "substitution" else "recipe_detail",
            should_retrieve=True,
            depends_on_state=True,
            needs_reference_resolution=True,
            reference_trigger="pronoun",
            reason="pronoun_followup",
        )

    if _has_out_of_domain_signal(text) and not _has_recipe_signal(text):
        return _base_result(
            action="domain_reject",
            answer_mode_hint="safe_direct",
            should_retrieve=False,
            domain_confidence=0.95,
            reason="harmless_out_of_domain",
        )

    if _has_list_signal(text):
        return _base_result(
            action="retrieve_list",
            answer_mode_hint="recommendation",
            should_retrieve=True,
            reason="recipe_list_query",
        )

    if _has_detail_signal(text) or _has_recipe_signal(text):
        return _base_result(
            action="retrieve_detail",
            answer_mode_hint="recipe_detail",
            should_retrieve=True,
            needs_reference_resolution=False,
            reason="recipe_detail_query",
        )

    if _has_current_dish(snapshot):
        return _base_result(
            action="history_answer",
            answer_mode_hint="history_based",
            should_retrieve=False,
            depends_on_state=True,
            reason="state_contextual_short_query",
        )

    return _base_result(
        action="domain_reject",
        answer_mode_hint="safe_direct",
        should_retrieve=False,
        domain_confidence=0.6,
        reason="no_recipe_signal",
    )
