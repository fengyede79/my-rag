"""
指代消解模块。
基于结构化快照判定用户指代目标，处理纠正、歧义、序号、口语前缀清洗和短追问。
"""

from __future__ import annotations

import re


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolved(target: str, source: str, reason: str, decision_basis: str = "explicit") -> dict:
    return {
        "resolution_status": "resolved",
        "resolved_target": target,
        "target_source": source,
        "confidence": 1.0,
        "reason": reason,
        "next_action": "apply_reference_resolution",
        "clarification_question": None,
        "writeback_eligible": True,
        "decision_basis": decision_basis,
    }


def _clarify(reason: str, question: str) -> dict:
    return {
        "resolution_status": "ambiguous",
        "resolved_target": None,
        "target_source": None,
        "confidence": 0.0,
        "reason": reason,
        "next_action": "ask_clarification",
        "clarification_question": question,
        "writeback_eligible": False,
        "decision_basis": "ambiguous",
    }


def _find_recommendation_by_rank(recommendations: list[dict], rank: int) -> str | None:
    for item in recommendations:
        if item.get("rank") == rank:
            return item.get("dish_name")
    return None


def _strip_question_punctuation(text: str) -> str:
    return text.strip().rstrip("?!？！。")


def _remove_ordinal_text(original_question: str) -> str:
    text = _strip_question_punctuation(original_question)
    text = re.sub(r"^(第\s*[一二三四五]\s*个|第\s*[1-5]\s*个|[1-5]\s*号)", "", text)
    for pattern in sorted(("看起来不错", "看起来挺好", "不错", "挺好", "可以"), key=len, reverse=True):
        text = text.replace(pattern, "")
    text = text.strip("，,。 ")
    return text or "怎么做"


def _remove_discourse_prefix_and_target(original_question: str, target: str) -> str:
    text = _strip_question_punctuation(original_question)
    prefixes = ("刚才那个", "刚才这道", "这个", "这道", "那")
    for prefix in prefixes:
        if text.startswith(prefix + target):
            return text[len(prefix + target):].strip("，,。 ") or "怎么做"
    if text.startswith(target):
        return text[len(target):].strip("，,。 ") or "怎么做"
    return text


def _has_explicit_dish_before_detail_keyword(question: str) -> bool:
    """Detect explicit dish prefixes like '蛋炒饭有什么小技巧'."""
    text = _strip_question_punctuation(question)
    generic_starts = (
        "有什么",
        "有哪些",
        "怎么",
        "做法",
        "需要什么",
        "需要哪些",
        "食材",
        "材料",
        "原料",
        "配料",
        "技巧",
    )
    if text.startswith(generic_starts):
        return False

    detail_keywords = (
        "需要哪些食材",
        "需要什么食材",
        "有什么食材",
        "有什么小技巧",
        "有哪些技巧",
        "怎么做",
        "做法",
        "食材",
        "材料",
        "原料",
        "配料",
        "技巧",
        "粘锅",
        "难不难",
        "要多久",
        "热量",
    )
    positions = [
        text.find(keyword)
        for keyword in detail_keywords
        if text.find(keyword) > 0
    ]
    if not positions:
        return False

    candidate = text[: min(positions)].strip("，,。 的")
    if not (2 <= len(candidate) <= 12):
        return False
    return all("\u4e00" <= ch <= "\u9fff" for ch in candidate)


# ---------------------------------------------------------------------------
# Core resolution
# ---------------------------------------------------------------------------

def resolve_reference_from_snapshot(snapshot: dict, llm) -> dict:
    constraints = snapshot["resolution_constraints"]
    query = snapshot["conversation_state"]["current_user_query"]
    explicit_targets = constraints["explicit_query_targets"]
    explicit_query_target_verified = constraints.get("explicit_query_target_verified", False)
    candidates = constraints["allowed_reference_targets"]

    # 1. Explicit correction target (highest priority)
    if explicit_targets:
        if not explicit_query_target_verified:
            return _clarify(
                "explicit_target_not_verified",
                "请直接告诉我明确的菜名，我再继续帮你查做法。",
            )
        return {
            "resolution_status": "resolved",
            "resolved_target": explicit_targets[0],
            "target_source": "explicit_query_target",
            "confidence": 1.0,
            "reason": "user_correction",
            "next_action": "apply_correction",
            "clarification_question": None,
            "writeback_eligible": True,
            "decision_basis": "explicit",
        }

    # 2. Cleaned explicit dish (e.g. "那蛋炒饭" → "蛋炒饭")
    cleaned_explicit_dish = constraints.get("cleaned_explicit_dish")
    if cleaned_explicit_dish and cleaned_explicit_dish.get("value"):
        return _resolved(
            cleaned_explicit_dish["value"],
            "cleaned_explicit_dish",
            "explicit_dish_with_discourse_prefix",
            decision_basis="explicit",
        )

    # 3. Ordinal reference (e.g. "第二个" → rank 2 in recommendations)
    ordinal_reference = constraints.get("ordinal_reference")
    if ordinal_reference:
        rank = ordinal_reference["rank"]
        target = _find_recommendation_by_rank(
            snapshot["reference_state"].get("recent_recommendations", []),
            rank,
        )
        if not target:
            return _clarify(
                "ordinal_rank_out_of_range",
                "我没找到你说的这个序号。你可以说第几个，或者直接告诉我菜名。",
            )
        return _resolved(
            target,
            "ordinal_recommendation_reference",
            "user_selected_recommendation_by_rank",
            decision_basis="explicit",
        )

    # 4. Pronoun in recommendation list → ambiguous
    if snapshot["topic_state"]["mode"] == "recommendation_list" and query.startswith(("它", "这个", "那个")):
        return _clarify(
            "multiple_candidates_in_recommendation_list",
            "你是指第几个推荐菜，还是直接告诉我菜名？",
        )

    # 5. Implicit single-dish followup (e.g. "有什么小技巧别粘锅？")
    implicit_followup = constraints.get("implicit_followup") or {}
    current_dish = snapshot["reference_state"].get("current_dish") or {}
    if implicit_followup.get("enabled"):
        if _has_explicit_dish_before_detail_keyword(query):
            # 原问题已有明确菜品，跳过隐式追问解析
            pass
        elif snapshot["topic_state"].get("mode") == "recommendation_list":
            return _clarify(
                "implicit_followup_in_recommendation_list",
                "你是想问第几个推荐菜？也可以直接告诉我菜名。",
            )
        elif current_dish.get("active") and current_dish.get("value"):
            return _resolved(
                current_dish["value"],
                "implicit_single_dish_followup",
                "single_active_dish_followup",
                decision_basis="inferred",
            )

    # 6. Single candidate → resolved
    if len(candidates) == 1:
        return {
            "resolution_status": "resolved",
            "resolved_target": candidates[0],
            "target_source": "current_dish",
            "confidence": 1.0,
            "reason": "single_candidate",
            "next_action": "retrieve_detail",
            "clarification_question": None,
            "writeback_eligible": True,
            "decision_basis": "inferred",
        }

    # 7. No reference needed
    return {
        "resolution_status": "no_reference_needed",
        "resolved_target": None,
        "target_source": None,
        "confidence": 0.0,
        "reason": "no_reference_needed",
        "next_action": "continue_general",
        "clarification_question": None,
        "writeback_eligible": False,
        "decision_basis": "none",
    }


# ---------------------------------------------------------------------------
# Guard
# ---------------------------------------------------------------------------

def guard_resolution_output(result: dict, constraints: dict) -> dict:
    allowed_targets = constraints["allowed_reference_targets"]
    explicit_targets = constraints["explicit_query_targets"]
    allow_external_explicit_target = constraints["allow_external_explicit_target"]

    # Explicit correction always passes
    if result.get("decision_basis") == "explicit" and result.get("resolved_target") in explicit_targets:
        return result

    if (
        result.get("decision_basis") == "explicit"
        and allow_external_explicit_target
        and result.get("resolved_target")
    ):
        return result

    # Cleaned explicit dish passes (it was extracted from the query itself)
    if result.get("target_source") == "cleaned_explicit_dish" and result.get("resolved_target"):
        return result

    # Ordinal recommendation reference passes (resolved from ranked list)
    if result.get("target_source") == "ordinal_recommendation_reference" and result.get("resolved_target"):
        return result

    # Implicit single-dish followup passes
    if result.get("target_source") == "implicit_single_dish_followup" and result.get("resolved_target"):
        return result

    # Non-explicit resolved target must be in allowed candidates
    if result.get("resolved_target") and result["resolved_target"] not in allowed_targets:
        return _clarify(
            "resolved_target_not_in_allowed_candidates",
            "请直接告诉我菜名，或者说明是第几个推荐菜。",
        )
    return result


# ---------------------------------------------------------------------------
# Query rewrite
# ---------------------------------------------------------------------------

def rewrite_query_for_execution(
    original_question: str,
    execution_plan: dict,
    resolution: dict | None,
    query_plan: dict | None,
) -> str:
    # Correction rewrite
    if execution_plan["action"] == "apply_correction" and resolution and resolution.get("resolved_target"):
        filters = (query_plan or {}).get("filters", {})
        content_type = filters.get("content_type") or (query_plan or {}).get("content_type")
        if content_type == "ingredients":
            return f"{resolution['resolved_target']}需要什么食材"
        return f"{resolution['resolved_target']}怎么做"

    # Reference resolution rewrite (ordinal, cleaned dish, implicit followup)
    if execution_plan["action"] == "apply_reference_resolution" and resolution and resolution.get("resolved_target"):
        target = resolution["resolved_target"]
        source = resolution.get("target_source")
        if source == "ordinal_recommendation_reference":
            return f"{target}{_remove_ordinal_text(original_question)}"
        if source == "cleaned_explicit_dish":
            return f"{target}{_remove_discourse_prefix_and_target(original_question, target)}"
        if source == "implicit_single_dish_followup":
            if _has_explicit_dish_before_detail_keyword(original_question):
                # 原问题已有明确菜品，直接返回原问题（去掉标点）
                return _strip_question_punctuation(original_question)
            return f"{target}{_strip_question_punctuation(original_question)}"

    return original_question
