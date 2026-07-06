from langchain_core.documents import Document

from rag_modules.retrieval_executor import RetrievalExecutor, build_retrieval_query_plan


class FakeRetrievalModule:
    def __init__(self, filtered_docs=None, hybrid_docs=None, extracted_filters=None):
        self.filtered_docs = filtered_docs or []
        self.hybrid_docs = hybrid_docs or []
        self.extracted_filters = extracted_filters or {}
        self.calls = []

    def extract_filters_from_query(self, query):
        self.calls.append(("extract_filters_from_query", query))
        return dict(self.extracted_filters)

    def metadata_filtered_search(self, query, filters, top_k=5, query_dish=None):
        self.calls.append(("metadata_filtered_search", query, dict(filters), top_k, query_dish))
        return list(self.filtered_docs[:top_k])

    def hybrid_search(self, query, top_k=3, query_dish=None):
        self.calls.append(("hybrid_search", query, top_k, query_dish))
        return list(self.hybrid_docs[:top_k])


def _doc(dish_name, content_type="steps", content="content"):
    return Document(page_content=content, metadata={"dish_name": dish_name, "content_type": content_type})


def test_executor_uses_filtered_primary_retrieval_when_filters_exist():
    docs = [_doc("宫保鸡丁", "steps", "宫保鸡丁步骤")]
    retrieval_module = FakeRetrievalModule(filtered_docs=docs)
    executor = RetrievalExecutor(retrieval_module)

    result = executor.execute(
        {
            "query": "宫保鸡丁 怎么做",
            "original_query": "第一个怎么做",
            "dish_name": "宫保鸡丁",
            "filters": {"dish_name": "宫保鸡丁", "content_type": "steps"},
            "top_k": 3,
            "fallback_policy": "disabled",
            "hard_filters": ["dish_name"],
            "soft_filters": ["content_type"],
            "answer_mode_hint": "recipe_detail",
        }
    )

    assert result["chunks"] == docs
    assert result["quality"]["enough_evidence"] is True
    assert result["quality"]["candidate_count"] == 1
    assert result["quality"]["selected_dishes"] == ["宫保鸡丁"]
    assert result["quality"]["fallback_used"] is False
    assert result["quality"]["relaxed_filter"] is False
    assert result["low_evidence"] is None
    assert result["trace"]["strategy"] == "primary"
    assert result["trace"]["fusion_strategy"] == "delegated"
    assert retrieval_module.calls[0][0] == "metadata_filtered_search"


def test_executor_uses_hybrid_primary_retrieval_without_filters():
    docs = [_doc("番茄炒蛋", "steps", "番茄炒蛋步骤")]
    retrieval_module = FakeRetrievalModule(hybrid_docs=docs)
    executor = RetrievalExecutor(retrieval_module)

    result = executor.execute(
        {
            "query": "今天吃什么",
            "original_query": "今天吃什么",
            "dish_name": None,
            "filters": {},
            "top_k": 3,
            "fallback_policy": "relaxed_filters",
            "hard_filters": [],
            "soft_filters": [],
            "answer_mode_hint": "recommendation",
        }
    )

    assert result["chunks"] == docs
    assert result["quality"]["enough_evidence"] is True
    assert result["trace"]["strategy"] == "primary"
    assert retrieval_module.calls[0][0] == "hybrid_search"


def test_query_plan_normalization_prefers_resolved_target_as_hard_dish_filter():
    result = build_retrieval_query_plan(
        original_query="第一个怎么做",
        rewritten_query="宫保鸡丁 怎么做",
        base_query_plan={
            "route_type": "detail",
            "dish_name": "第一个",
            "filters": {"content_type": "steps"},
            "entities": {"dish_name": "第一个", "filters": {"content_type": "steps"}},
        },
        execution_plan={"action": "apply_reference_resolution", "answer_mode": "recipe_detail"},
        resolution={"resolved_target": "宫保鸡丁", "confidence": 0.95},
        preference_constraints=None,
        top_k=3,
    )

    assert result["query"] == "宫保鸡丁 怎么做"
    assert result["original_query"] == "第一个怎么做"
    assert result["dish_name"] == "宫保鸡丁"
    assert result["filters"]["dish_name"] == "宫保鸡丁"
    assert result["filters"]["content_type"] == "steps"
    assert result["hard_filters"] == ["dish_name"]
    assert "content_type" in result["soft_filters"]
    assert result["fallback_policy"] == "disabled"
    assert result["top_k"] == 3


def test_query_plan_normalization_uses_relaxed_filters_for_sparse_list_preferences():
    result = build_retrieval_query_plan(
        original_query="推荐几个不辣的鸡肉菜",
        rewritten_query="推荐几个不辣的鸡肉菜",
        base_query_plan={
            "route_type": "list",
            "dish_name": None,
            "filters": {},
            "entities": {"dish_name": None, "filters": {}},
            "preference_constraints": {"taste": ["不辣"], "ingredient": ["鸡肉"]},
        },
        execution_plan={"action": "retrieve_list", "answer_mode": "recommendation"},
        resolution=None,
        preference_constraints={"taste": ["不辣"], "ingredient": ["鸡肉"]},
        top_k=5,
    )

    assert result["query"] == "推荐几个不辣的鸡肉菜"
    assert result["dish_name"] is None
    assert result["hard_filters"] == []
    assert "dish_name" not in result["filters"]
    assert result["filters"]["taste"] == ["不辣"]
    assert result["filters"]["ingredient"] == ["鸡肉"]
    assert result["soft_filters"] == ["ingredient", "taste", "difficulty", "time", "health_preference"]
    assert result["fallback_policy"] == "relaxed_filters"
    assert result["answer_mode_hint"] == "recommendation"


def test_query_plan_normalization_keeps_broad_search_disabled_by_default():
    result = build_retrieval_query_plan(
        original_query="西湖醋鱼怎么做",
        rewritten_query="西湖醋鱼怎么做",
        base_query_plan={
            "route_type": "detail",
            "dish_name": "西湖醋鱼",
            "filters": {},
            "entities": {"dish_name": "西湖醋鱼", "filters": {}},
        },
        execution_plan={"action": "retrieve_detail", "answer_mode": "recipe_detail"},
        resolution=None,
        preference_constraints=None,
        top_k=3,
    )

    assert result["dish_name"] == "西湖醋鱼"
    assert result["hard_filters"] == ["dish_name"]
    assert result["fallback_policy"] == "disabled"


def test_executor_rejects_different_dish_for_hard_exact_dish_request():
    retrieval_module = FakeRetrievalModule(filtered_docs=[_doc("鱼香肉丝", "steps", "鱼香肉丝步骤")])
    executor = RetrievalExecutor(retrieval_module)

    result = executor.execute(
        {
            "query": "西湖醋鱼 怎么做",
            "original_query": "西湖醋鱼怎么做",
            "dish_name": "西湖醋鱼",
            "filters": {"dish_name": "西湖醋鱼"},
            "top_k": 3,
            "fallback_policy": "disabled",
            "hard_filters": ["dish_name"],
            "soft_filters": [],
            "answer_mode_hint": "recipe_detail",
        }
    )

    assert result["chunks"] == []
    assert result["quality"]["enough_evidence"] is False
    assert result["quality"]["quality_reason"] == "exact_dish_not_found"
    assert result["low_evidence"] == {
        "answer_type": "no_result",
        "answer": "知识库里没有找到可靠的食谱信息。",
        "state_diff_policy": "low_evidence",
        "quality_reason": "exact_dish_not_found",
    }


def test_executor_rejects_conflicting_dishes_for_hard_exact_dish_request():
    retrieval_module = FakeRetrievalModule(
        filtered_docs=[
            _doc("宫保鸡丁", "steps", "宫保鸡丁步骤"),
            _doc("鱼香肉丝", "steps", "鱼香肉丝步骤"),
        ]
    )
    executor = RetrievalExecutor(retrieval_module)

    result = executor.execute(
        {
            "query": "宫保鸡丁 怎么做",
            "original_query": "宫保鸡丁怎么做",
            "dish_name": "宫保鸡丁",
            "filters": {"dish_name": "宫保鸡丁"},
            "top_k": 3,
            "fallback_policy": "disabled",
            "hard_filters": ["dish_name"],
            "soft_filters": [],
            "answer_mode_hint": "recipe_detail",
        }
    )

    assert result["quality"]["enough_evidence"] is False
    assert result["quality"]["quality_reason"] == "conflicting_dishes_for_exact_request"
    assert result["low_evidence"]["answer_type"] == "no_result"


def test_executor_returns_low_evidence_when_primary_has_no_candidates():
    retrieval_module = FakeRetrievalModule(filtered_docs=[])
    executor = RetrievalExecutor(retrieval_module)

    result = executor.execute(
        {
            "query": "不存在的菜 怎么做",
            "original_query": "不存在的菜怎么做",
            "dish_name": "不存在的菜",
            "filters": {"dish_name": "不存在的菜"},
            "top_k": 3,
            "fallback_policy": "disabled",
            "hard_filters": ["dish_name"],
            "soft_filters": [],
            "answer_mode_hint": "recipe_detail",
        }
    )

    assert result["chunks"] == []
    assert result["quality"]["quality_reason"] == "no_candidates"
    assert result["low_evidence"]["state_diff_policy"] == "low_evidence"


def test_fallback_does_not_run_when_policy_disabled():
    retrieval_module = FakeRetrievalModule(filtered_docs=[], hybrid_docs=[_doc("鱼香肉丝")])
    executor = RetrievalExecutor(retrieval_module)

    result = executor.execute(
        {
            "query": "西湖醋鱼 怎么做",
            "original_query": "西湖醋鱼怎么做",
            "dish_name": "西湖醋鱼",
            "filters": {"dish_name": "西湖醋鱼", "content_type": "steps"},
            "top_k": 3,
            "fallback_policy": "disabled",
            "hard_filters": ["dish_name"],
            "soft_filters": ["content_type"],
            "answer_mode_hint": "recipe_detail",
        }
    )

    assert result["chunks"] == []
    assert result["quality"]["fallback_used"] is False
    assert result["trace"]["fallback_count"] == 0
    assert [call[0] for call in retrieval_module.calls].count("metadata_filtered_search") == 1


def test_relaxed_filter_fallback_keeps_hard_dish_filter_and_marks_docs():
    fallback_doc = _doc("宫保鸡丁", "introduction", "宫保鸡丁介绍")
    retrieval_module = FakeRetrievalModule(filtered_docs=[])

    def metadata_filtered_search(query, filters, top_k=5, query_dish=None):
        retrieval_module.calls.append(("metadata_filtered_search", query, dict(filters), top_k, query_dish))
        if filters == {"dish_name": "宫保鸡丁"}:
            return [fallback_doc]
        return []

    retrieval_module.metadata_filtered_search = metadata_filtered_search
    executor = RetrievalExecutor(retrieval_module)

    result = executor.execute(
        {
            "query": "宫保鸡丁 技巧",
            "original_query": "宫保鸡丁有什么技巧",
            "dish_name": "宫保鸡丁",
            "filters": {"dish_name": "宫保鸡丁", "content_type": "tips"},
            "top_k": 3,
            "fallback_policy": "relaxed_filters",
            "hard_filters": ["dish_name"],
            "soft_filters": ["content_type"],
            "answer_mode_hint": "recipe_detail",
        }
    )

    assert result["chunks"] == [fallback_doc]
    assert result["quality"]["enough_evidence"] is True
    assert result["quality"]["fallback_used"] is True
    assert result["quality"]["relaxed_filter"] is True
    assert fallback_doc.metadata["fallback"] is True
    assert fallback_doc.metadata["relaxed_filter"] is True
    assert result["trace"]["fallback_count"] == 1
    assert retrieval_module.calls[-1][2] == {"dish_name": "宫保鸡丁"}


def test_broad_search_fallback_rejected_for_hard_exact_dish_request():
    retrieval_module = FakeRetrievalModule(filtered_docs=[], hybrid_docs=[_doc("鱼香肉丝")])
    executor = RetrievalExecutor(retrieval_module)

    result = executor.execute(
        {
            "query": "西湖醋鱼 怎么做",
            "original_query": "西湖醋鱼怎么做",
            "dish_name": "西湖醋鱼",
            "filters": {"dish_name": "西湖醋鱼"},
            "top_k": 3,
            "fallback_policy": "broad_search",
            "hard_filters": ["dish_name"],
            "soft_filters": [],
            "answer_mode_hint": "recipe_detail",
        }
    )

    assert result["chunks"] == []
    assert result["quality"]["enough_evidence"] is False
    assert result["quality"]["fallback_used"] is False
    assert result["low_evidence"]["answer_type"] == "no_result"
    assert all(call[0] != "hybrid_search" for call in retrieval_module.calls)
