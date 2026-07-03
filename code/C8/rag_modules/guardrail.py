"""
查询护栏模块。
负责在检索前识别超出知识库边界的问题，并生成保守回答。
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def classify_query_guardrail(query: str) -> Optional[str]:
    """识别应在检索前直接保守兜底的问题。

    Args:
        query: 用户查询

    Returns:
        None 表示通过护栏检查，str 表示拦截原因：
        - "temporal_personal": 时间性个人记忆查询
        - "out_of_domain": 超域问题
        - "unsupported_food_judgement": 不支持的食物判断
    """
    normalized_query = query.strip()
    if not normalized_query:
        return None

    temporal_markers = [
        "昨天",
        "前天",
        "上周",
        "上次",
        "之前",
        "刚才",
        "前几天",
        "明天",
        "昨晚",
    ]
    personal_markers = ["我", "我之前", "我上次", "你记得我", "记得我"]
    memory_actions = [
        "吃了什么",
        "吃了啥",
        "吃过什么",
        "做过什么",
        "做了什么",
        "喝了什么",
        "点了什么",
        "哪道菜",
        "吃什么",
        "哪一种",
    ]
    if (
        any(marker in normalized_query for marker in temporal_markers)
        and any(marker in normalized_query for marker in personal_markers)
        and any(action in normalized_query for action in memory_actions)
    ):
        return "temporal_personal"

    food_terms = [
        "菜",
        "食谱",
        "食材",
        "做法",
        "步骤",
        "制作",
        "烹饪",
        "早餐",
        "午饭",
        "午餐",
        "晚饭",
        "晚餐",
        "夜宵",
        "甜品",
        "饮品",
        "汤",
        "面",
        "饭",
        "粥",
        "空气炸锅",
        "电饭煲",
        "烤箱",
        "煮",
        "炒",
        "蒸",
        "炸",
        "炖",
        "烤",
        "推荐",
        "吃什么",
    ]
    out_of_domain_objects = [
        "路由器",
        "手机壳",
        "羽绒服",
        "电脑",
        "书桌",
        "绿植",
        "窗帘",
        "玻璃",
        "不锈钢",
        "天气",
    ]
    out_of_domain_actions = [
        "清洗",
        "处理",
        "修复",
        "断网",
        "发黄",
        "换盆",
        "发霉",
        "噪音",
        "怎么办",
        "保养",
        "怎么洗",
        "洗",
    ]
    smalltalk_terms = [
        "你怎么回答这么快",
        "你怎么反应这么快",
        "为什么这么快",
        "谢谢",
        "厉害",
        "真快",
    ]
    has_out_of_domain_action = any(action in normalized_query for action in out_of_domain_actions)
    has_out_of_domain_object = any(obj in normalized_query for obj in out_of_domain_objects)
    has_food_term = any(term in normalized_query for term in food_terms)

    unsupported_comparison_patterns = [
        "是一个菜吗",
        "是不是一个菜",
        "同一个菜吗",
        "一样吗",
    ]
    if any(pattern in normalized_query for pattern in unsupported_comparison_patterns):
        return "unsupported_food_judgement"

    if "需要" in normalized_query and "吗" in normalized_query and not any(
        marker in normalized_query for marker in ["什么", "哪些", "多少", "怎么", "如何"]
    ):
        return "unsupported_food_judgement"

    beverage_conflict_terms = ["奶茶", "长岛冰茶"]
    savory_cooking_terms = ["红烧", "麻婆", "鱼", "肉", "豆腐", "鸡", "虾"]
    if (
        any(term in normalized_query for term in beverage_conflict_terms)
        and any(term in normalized_query for term in savory_cooking_terms)
        and any(term in normalized_query for term in ["做法", "步骤", "推荐", "一起说"])
    ):
        return "unsupported_food_judgement"

    if has_food_term:
        return None

    if has_out_of_domain_action and has_out_of_domain_object:
        return "out_of_domain"

    if has_out_of_domain_object or any(term in normalized_query for term in smalltalk_terms):
        return "out_of_domain"

    return None


def build_guardrail_answer(query: str, reason: str) -> str:
    """统一生成边界问题的保守回答。

    Args:
        query: 用户查询
        reason: 拦截原因（由 classify_query_guardrail 返回）

    Returns:
        保守回答文本
    """
    if reason == "temporal_personal":
        return (
            "我不知道你之前具体吃了什么或做过哪道菜，因为知识库不会记录你的个人经历。"
            "如果你愿意，我可以推荐几道合适的菜，再根据你现在想吃的口味、食材或做法继续细化。"
        )

    if reason == "out_of_domain":
        return (
            "这个问题不属于当前食谱知识库能够可靠回答的范围，所以我不清楚该怎么直接判断。"
            "如果你愿意，我可以继续帮你处理做菜、食材、步骤或菜品推荐相关的问题。"
        )

    if reason == "unsupported_food_judgement":
        return (
            "我不知道该怎么可靠判断这个混合问题，因为它超出了当前食谱知识库擅长的问答范围。"
            "如果你愿意，我可以改为单独回答某道菜的做法、食材，或者重新给你推荐菜品。"
        )

    return (
        "这个问题超出了当前食谱知识库能可靠回答的范围。"
        "如果你愿意，我可以继续帮你回答菜谱和做饭相关的问题。"
    )
