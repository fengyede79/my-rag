from __future__ import annotations

"""Retrieval execution boundary for the runtime chat path."""

from typing import Any, Dict, Iterable, List

from langchain_core.documents import Document

from rag_modules.dish_aliases import dish_aliases_for, is_known_alias_target


SOFT_FILTER_KEYS = ["ingredient", "taste", "difficulty", "time", "health_preference"]

INGREDIENT_FAMILY_MAP: dict[str, set[str]] = {
    "鸡": {"鸡", "鸡翅", "鸡胸肉", "鸡腿", "鸡丁", "鸡丝", "鸡蛋"},
    "鱼": {"鱼", "鲈鱼", "鲤鱼", "鱼头", "鱼片"},
    "虾": {"虾", "大虾", "红虾", "小龙虾"},
    "肉": {"肉", "猪肉", "五花肉", "排骨", "里脊"},
    "牛": {"牛", "牛肉", "牛腩", "肥牛"},
    "豆腐": {"豆腐", "豆腐", "豆干"},
    "土豆": {"土豆", "马铃薯"},
    "蛋": {"蛋", "鸡蛋", "鸭蛋", "鹌鹑蛋"},
}


def classify_evidence_identity(dish_name: str | None, selected_dishes: list[str]) -> str:
    if not dish_name or not selected_dishes:
        return "no_identity"
    if dish_name in selected_dishes:
        return "exact_identity"
    if any(is_known_alias_target(dish_name, s) for s in selected_dishes):
        return "alias_identity"
    for _family, members in INGREDIENT_FAMILY_MAP.items():
        dish_in_family = any(dish_name in m or m in dish_name for m in members)
        if dish_in_family:
            sel_in_family = any(any(s in m or m in s for m in members) for s in selected_dishes)
            if sel_in_family:
                return "similar_reference"
    return "no_identity"


def _copy_dict(value: dict | None) -> dict:
    return dict(value or {})


def _resolved_dish(base_query_plan: dict, resolution: dict | None) -> str | None:
    if resolution:
        resolved = resolution.get("resolved_target") or resolution.get("resolved_entity")
        if resolved:
            return resolved
    return base_query_plan.get("dish_name")


def _merge_preference_constraints(filters: dict, preference_constraints: dict | None) -> dict:
    merged = dict(filters)
    for key, value in (preference_constraints or {}).items():
        if value:
            merged[key] = value
    return merged


def _answer_mode_hint(execution_plan: dict, base_query_plan: dict) -> str:
    if execution_plan.get("answer_mode"):
        return execution_plan["answer_mode"]
    action = execution_plan.get("action")
    if action == "retrieve_list" or base_query_plan.get("route_type") == "list":
        return "recommendation"
    return "recipe_detail"


def _fallback_policy(route_type: str, hard_filters: list[str], filters: dict) -> str:
    droppable_keys = set(SOFT_FILTER_KEYS) | {"content_type"}
    has_droppable = any(key in filters for key in droppable_keys)
    if has_droppable or route_type == "list":
        return "relaxed_filters"
    return "disabled"


def build_retrieval_query_plan(
    *,
    original_query: str,
    rewritten_query: str,
    base_query_plan: dict,
    execution_plan: dict,
    resolution: dict | None,
    preference_constraints: dict | None,
    top_k: int,
) -> dict:
    """Normalize the runtime query plan into the retrieval-facing contract."""
    route_type = base_query_plan.get("route_type", "detail")
    dish_name = _resolved_dish(base_query_plan, resolution)
    filters = _merge_preference_constraints(
        _copy_dict(base_query_plan.get("filters")),
        preference_constraints,
    )

    hard_filters: list[str] = []
    if dish_name and route_type != "list":
        filters["dish_name"] = dish_name
        hard_filters.append("dish_name")

    soft_filters = list(SOFT_FILTER_KEYS)
    if "content_type" in filters:
        soft_filters.append("content_type")

    return {
        "query": rewritten_query,
        "original_query": original_query,
        "dish_name": dish_name,
        "filters": filters,
        "top_k": top_k,
        "fallback_policy": _fallback_policy(route_type, hard_filters, filters),
        "hard_filters": hard_filters,
        "soft_filters": soft_filters,
        "answer_mode_hint": _answer_mode_hint(execution_plan, base_query_plan),
        "route_type": route_type,
    }


class RetrievalExecutor:
    """Execute retrieval and return chunks plus explicit evidence quality."""

    def __init__(self, retrieval_module):
        self.retrieval_module = retrieval_module

    def execute(self, query_plan: dict) -> dict:
        primary_chunks = self._primary_retrieval(query_plan)
        primary_quality = self._check_quality(
            query_plan,
            primary_chunks,
            fallback_used=False,
            relaxed_filter=False,
        )

        if primary_quality["enough_evidence"]:
            return {
                "chunks": primary_chunks,
                "quality": primary_quality,
                "low_evidence": None,
                "trace": self._build_trace(
                    query_plan=query_plan,
                    strategy="primary",
                    primary_count=len(primary_chunks),
                    fallback_count=0,
                    quality=primary_quality,
                ),
            }

        alias_chunks, alias_used = self._alias_fallback_retrieval(query_plan)
        if alias_chunks:
            alias_quality = self._check_quality(
                query_plan,
                alias_chunks,
                fallback_used=True,
                relaxed_filter=True,
                allow_alias_match=True,
            )
            if alias_quality["enough_evidence"]:
                return {
                    "chunks": alias_chunks,
                    "quality": alias_quality,
                    "low_evidence": None,
                    "trace": self._build_trace(
                        query_plan=query_plan,
                        strategy="alias_fallback",
                        primary_count=len(primary_chunks),
                        fallback_count=len(alias_chunks),
                        quality=alias_quality,
                        dish_alias_used=alias_used,
                    ),
                }

        fallback_chunks = self._fallback_retrieval(query_plan)
        if fallback_chunks:
            fallback_quality = self._check_quality(
                query_plan,
                fallback_chunks,
                fallback_used=True,
                relaxed_filter=True,
            )
            if fallback_quality["enough_evidence"]:
                return {
                    "chunks": fallback_chunks,
                    "quality": fallback_quality,
                    "low_evidence": None,
                    "trace": self._build_trace(
                        query_plan=query_plan,
                        strategy="fallback",
                        primary_count=len(primary_chunks),
                        fallback_count=len(fallback_chunks),
                        quality=fallback_quality,
                    ),
                }

        list_broad_chunks = self._list_broad_fallback(query_plan)
        if list_broad_chunks:
            list_quality = self._check_list_quality(query_plan, list_broad_chunks)
            return {
                "chunks": list_broad_chunks,
                "quality": list_quality,
                "low_evidence": None,
                "trace": self._build_trace(
                    query_plan=query_plan,
                    strategy="list_broad_hybrid",
                    primary_count=len(primary_chunks),
                    fallback_count=len(list_broad_chunks),
                    quality=list_quality,
                ),
            }

        return {
            "chunks": [],
            "quality": primary_quality,
            "low_evidence": self._low_evidence(primary_quality["quality_reason"]),
            "trace": self._build_trace(
                query_plan=query_plan,
                strategy="low_evidence",
                primary_count=len(primary_chunks),
                fallback_count=len(fallback_chunks),
                quality=primary_quality,
            ),
        }

    def _primary_retrieval(self, query_plan: dict) -> list[Document]:
        query = query_plan["query"]
        filters = dict(query_plan.get("filters") or {})
        top_k = query_plan.get("top_k", 3)
        dish_name = query_plan.get("dish_name")
        if filters:
            return list(
                self.retrieval_module.metadata_filtered_search(
                    query,
                    filters,
                    top_k=top_k,
                    query_dish=dish_name,
                )
            )
        return list(
            self.retrieval_module.hybrid_search(
                query,
                top_k=top_k,
                query_dish=dish_name,
            )
        )

    def _selected_dishes(self, chunks: Iterable[Document]) -> list[str]:
        dishes: list[str] = []
        for chunk in chunks:
            dish_name = (chunk.metadata or {}).get("dish_name")
            if dish_name and dish_name not in dishes:
                dishes.append(dish_name)
        return dishes

    def _check_quality(
        self,
        query_plan: dict,
        chunks: list[Document],
        *,
        fallback_used: bool,
        relaxed_filter: bool,
        allow_alias_match: bool = False,
    ) -> dict:
        selected_dishes = self._selected_dishes(chunks)
        dish_name = query_plan.get("dish_name")
        hard_filters = set(query_plan.get("hard_filters") or [])

        enough = bool(chunks)
        reason = "primary_candidates_found" if enough else "no_candidates"

        if enough and dish_name and "dish_name" in hard_filters:
            if dish_name not in selected_dishes:
                # 子串匹配：dish_name 是某个 selected_dish 的子串（≥2字且占比≥50%）
                substring_matches = [
                    s for s in selected_dishes
                    if len(dish_name) >= 2
                    and (dish_name in s or s in dish_name)
                    and min(len(dish_name), len(s)) / max(len(dish_name), len(s)) >= 0.5
                ]
                if substring_matches:
                    reason = "substring_dish_matched"
                else:
                    alias_matches = [
                        selected for selected in selected_dishes
                        if allow_alias_match and is_known_alias_target(dish_name, selected)
                    ]
                    if alias_matches and len(selected_dishes) == 1:
                        reason = "alias_dish_matched"
                    else:
                        enough = False
                        reason = "exact_dish_not_found"
            elif len(selected_dishes) > 1:
                enough = False
                reason = "conflicting_dishes_for_exact_request"
            else:
                reason = "exact_dish_matched"

        identity = classify_evidence_identity(dish_name, selected_dishes)

        return {
            "enough_evidence": enough,
            "quality_reason": reason,
            "fallback_used": fallback_used,
            "relaxed_filter": relaxed_filter,
            "candidate_count": len(chunks),
            "selected_dishes": selected_dishes,
            "evidence_identity": identity,
        }

    def _low_evidence(self, quality_reason: str) -> dict:
        return {
            "answer_type": "no_result",
            "answer": "知识库里没有找到可靠的食谱信息。",
            "state_diff_policy": "low_evidence",
            "quality_reason": quality_reason,
        }

    def _build_trace(
        self,
        *,
        query_plan: dict,
        strategy: str,
        primary_count: int,
        fallback_count: int,
        quality: dict,
        dish_alias_used: str | None = None,
    ) -> dict:
        trace = {
            "strategy": strategy,
            "fusion_strategy": "delegated",
            "query": query_plan.get("query"),
            "original_query": query_plan.get("original_query"),
            "filters": dict(query_plan.get("filters") or {}),
            "hard_filters": list(query_plan.get("hard_filters") or []),
            "soft_filters": list(query_plan.get("soft_filters") or []),
            "fallback_policy": query_plan.get("fallback_policy", "disabled"),
            "primary_count": primary_count,
            "fallback_count": fallback_count,
            "selected_dishes": list(quality.get("selected_dishes") or []),
            "quality_reason": quality.get("quality_reason"),
            "fallback_used": quality.get("fallback_used"),
            "relaxed_filter": quality.get("relaxed_filter"),
        }
        if dish_alias_used:
            trace["dish_alias_used"] = dish_alias_used
        return trace

    def _alias_fallback_retrieval(self, query_plan: dict) -> tuple[list[Document], str | None]:
        dish_name = query_plan.get("dish_name")
        if not dish_name or "dish_name" not in set(query_plan.get("hard_filters") or []):
            return [], None

        aliases = dish_aliases_for(dish_name)
        if not aliases:
            return [], None

        base_filters = dict(query_plan.get("filters") or {})
        top_k = query_plan.get("top_k", 3)
        for alias in aliases:
            alias_filters = dict(base_filters)
            alias_filters["dish_name"] = alias
            chunks = list(
                self.retrieval_module.metadata_filtered_search(
                    query_plan["query"],
                    alias_filters,
                    top_k=top_k,
                    query_dish=alias,
                )
            )
            selected = self._selected_dishes(chunks)
            if len(selected) == 1 and is_known_alias_target(dish_name, selected[0]):
                return self._mark_fallback(chunks, dish_alias_used=alias), alias
        return [], None

    def _fallback_retrieval(self, query_plan: dict) -> list[Document]:
        policy = query_plan.get("fallback_policy", "disabled")
        if policy == "disabled":
            return []

        hard_filters = set(query_plan.get("hard_filters") or [])

        if policy == "relaxed_filters":
            relaxed_filters = self._relaxed_filters(query_plan)
            if not relaxed_filters and query_plan.get("filters"):
                return []
            chunks = list(
                self.retrieval_module.metadata_filtered_search(
                    query_plan["query"],
                    relaxed_filters,
                    top_k=query_plan.get("top_k", 3),
                    query_dish=query_plan.get("dish_name"),
                )
            )
            return self._mark_fallback(chunks)

        if policy == "broad_search":
            if "dish_name" in hard_filters:
                return []
            chunks = list(
                self.retrieval_module.hybrid_search(
                    query_plan["query"],
                    top_k=query_plan.get("top_k", 3),
                    query_dish=query_plan.get("dish_name"),
                )
            )
            return self._mark_fallback(chunks)

        return []

    def _relaxed_filters(self, query_plan: dict) -> dict:
        filters = dict(query_plan.get("filters") or {})
        hard_filters = set(query_plan.get("hard_filters") or [])
        droppable_keys = set(SOFT_FILTER_KEYS) | {"content_type"}
        return {key: value for key, value in filters.items() if key in hard_filters or key not in droppable_keys}

    def _list_broad_fallback(self, query_plan: dict) -> list[Document]:
        route_type = query_plan.get("route_type", "detail")
        answer_mode = query_plan.get("answer_mode_hint", "")
        if route_type != "list" and answer_mode != "recommendation":
            return []
        chunks = list(
            self.retrieval_module.hybrid_search(
                query_plan["query"],
                top_k=query_plan.get("top_k", 3),
                query_dish=None,
            )
        )
        for chunk in chunks:
            chunk.metadata["fallback"] = True
            chunk.metadata["relaxed_filter"] = True
        return chunks

    def _check_list_quality(self, query_plan: dict, chunks: list[Document]) -> dict:
        selected_dishes = self._selected_dishes(chunks)
        enough = len(chunks) > 0 and len(selected_dishes) > 0
        return {
            "enough_evidence": enough,
            "quality_reason": "list_candidates_found" if enough else "no_candidates",
            "fallback_used": True,
            "relaxed_filter": True,
            "candidate_count": len(chunks),
            "selected_dishes": selected_dishes,
        }

    def _mark_fallback(self, chunks: list[Document], dish_alias_used: str | None = None) -> list[Document]:
        for chunk in chunks:
            chunk.metadata["fallback"] = True
            chunk.metadata["relaxed_filter"] = True
            if dish_alias_used:
                chunk.metadata["dish_alias_used"] = dish_alias_used
        return chunks
