from __future__ import annotations

"""Small bounded recipe-name alias helpers for retrieval fallback."""


DISH_ALIASES: dict[str, list[str]] = {
    "番茄炒蛋": ["西红柿炒鸡蛋", "番茄鸡蛋", "番茄炒鸡蛋"],
    "西红柿炒鸡蛋": ["番茄炒蛋", "番茄鸡蛋", "番茄炒鸡蛋"],
    "凉拌黄瓜": ["拍黄瓜", "黄瓜"],
    "可乐鸡翅": ["鸡翅"],
    "红烧肉": ["五花肉"],
}


def _normalize_name(dish_name: str | None) -> str:
    return (dish_name or "").strip()


def dish_aliases_for(dish_name: str | None) -> list[str]:
    """Return bounded aliases for a dish name, preserving configured order."""
    normalized = _normalize_name(dish_name)
    if not normalized:
        return []
    return list(DISH_ALIASES.get(normalized, []))


def is_known_alias_target(original: str | None, candidate: str | None) -> bool:
    """Return true when candidate is the original dish or a configured alias."""
    original_name = _normalize_name(original)
    candidate_name = _normalize_name(candidate)
    if not original_name or not candidate_name:
        return False
    return candidate_name == original_name or candidate_name in dish_aliases_for(original_name)
