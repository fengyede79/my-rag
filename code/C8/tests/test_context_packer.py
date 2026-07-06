from langchain_core.documents import Document

from rag_modules.context_packer import extract_markdown_sections


RECIPE_MARKDOWN = """# 宫保鸡丁的做法

宫保鸡丁是一道家常菜。

## 必备原料和工具

- 鸡腿肉 300g
- 花生米 50g
- 干辣椒 适量

## 计算

两人份。

## 操作

### 腌制鸡肉

- 鸡肉切丁。
- 加入淀粉抓匀。

### 炒制

- 热锅下油。
- 倒入鸡丁翻炒。

## 附加内容

- 火候不要太大。
- 可以减少辣椒。
"""


def _parent_doc(text=RECIPE_MARKDOWN):
    return Document(
        page_content=text,
        metadata={"dish_name": "宫保鸡丁", "parent_id": "parent-1"},
    )


def test_extract_markdown_sections_maps_common_recipe_headings():
    sections = extract_markdown_sections(_parent_doc())

    assert [section["section_type"] for section in sections] == [
        "introduction",
        "ingredients",
        "calculation",
        "steps",
        "tips",
    ]
    assert sections[0]["title"] == "宫保鸡丁的做法"
    assert sections[1]["title"] == "必备原料和工具"
    assert sections[3]["title"] == "操作"
    assert "### 腌制鸡肉" in sections[3]["text"]
    assert sections[3]["dish_name"] == "宫保鸡丁"
    assert sections[3]["parent_id"] == "parent-1"


def test_extract_markdown_sections_keeps_text_before_first_second_level_heading_as_introduction():
    sections = extract_markdown_sections(_parent_doc())

    introduction = sections[0]
    assert introduction["section_type"] == "introduction"
    assert "宫保鸡丁是一道家常菜" in introduction["text"]
    assert "## 必备原料和工具" not in introduction["text"]


def test_extract_markdown_sections_preserves_h3_substeps_inside_steps_section():
    sections = extract_markdown_sections(_parent_doc())

    steps = next(section for section in sections if section["section_type"] == "steps")
    assert steps["title"] == "操作"
    assert "### 腌制鸡肉" in steps["text"]
    assert "### 炒制" in steps["text"]
    assert "腌制鸡肉" not in [section["title"] for section in sections]


from rag_modules.context_packer import finalize_answer_mode


def test_finalize_answer_mode_uses_list_route_as_recommendation():
    result = finalize_answer_mode(
        turn_info={"action": "retrieve_list", "answer_mode_hint": "recommendation"},
        execution_plan={"action": "retrieve_list"},
        query_plan={"route_type": "list"},
        retrieval_quality={"enough_evidence": True},
    )

    assert result == "recommendation"


def test_finalize_answer_mode_keeps_substitution_from_turn_understanding():
    result = finalize_answer_mode(
        turn_info={"action": "substitution", "answer_mode_hint": "substitution"},
        execution_plan={"action": "retrieve_detail"},
        query_plan={"route_type": "detail", "filters": {"content_type": "steps"}},
        retrieval_quality={"enough_evidence": True},
    )

    assert result == "substitution"


def test_finalize_answer_mode_keeps_comparison_from_turn_understanding():
    result = finalize_answer_mode(
        turn_info={"action": "compare", "answer_mode_hint": "comparison"},
        execution_plan={"action": "retrieve_detail"},
        query_plan={"route_type": "detail"},
        retrieval_quality={"enough_evidence": True},
    )

    assert result == "comparison"


def test_finalize_answer_mode_defaults_retrieve_detail_to_recipe_detail():
    result = finalize_answer_mode(
        turn_info={"action": "retrieve_detail", "answer_mode_hint": "recipe_detail"},
        execution_plan={"action": "retrieve_detail"},
        query_plan={"route_type": "detail"},
        retrieval_quality={"enough_evidence": True},
    )

    assert result == "recipe_detail"


def test_finalize_answer_mode_keeps_history_answer_even_without_hint():
    result = finalize_answer_mode(
        turn_info={"action": "history_answer"},
        execution_plan={"action": "history_answer"},
        query_plan={"route_type": "direct"},
        retrieval_quality=None,
    )

    assert result == "history_based"


from rag_modules.context_packer import ContextPacker


def test_context_packer_ingredients_question_selects_ingredient_section_not_full_document():
    parent = _parent_doc()
    packer = ContextPacker(max_chars_total=2000, max_chars_per_doc=800, max_docs=5)

    pack = packer.build_context_pack(
        query="宫保鸡丁需要什么食材",
        retrieval_result={"chunks": [Document(page_content="chunk", metadata={"dish_name": "宫保鸡丁"})]},
        query_plan={"route_type": "detail", "filters": {"content_type": "ingredients"}},
        execution_plan={"action": "retrieve_detail"},
        turn_info={"action": "retrieve_detail", "answer_mode_hint": "recipe_detail"},
        parent_docs=[parent],
    )

    assert pack["answer_mode"] == "recipe_detail"
    assert pack["content_type"] == "ingredients"
    assert len(pack["context_docs"]) == 1
    assert "## 必备原料和工具" in pack["context_docs"][0].page_content
    assert "## 操作" not in pack["context_docs"][0].page_content
    assert pack["selected_sections"][0]["section_type"] == "ingredients"
    assert pack["parent_docs"] == [parent]


def test_context_packer_steps_question_preserves_ordered_procedure_context():
    parent = _parent_doc()
    packer = ContextPacker(max_chars_total=2000, max_chars_per_doc=1000, max_docs=5)

    pack = packer.build_context_pack(
        query="宫保鸡丁怎么做",
        retrieval_result={"chunks": []},
        query_plan={"route_type": "detail", "filters": {"content_type": "steps"}},
        execution_plan={"action": "retrieve_detail"},
        turn_info={"action": "retrieve_detail", "answer_mode_hint": "recipe_detail"},
        parent_docs=[parent],
    )

    text = pack["context_docs"][0].page_content
    assert "## 操作" in text
    assert text.index("### 腌制鸡肉") < text.index("### 炒制")
    assert pack["selected_sections"][0]["section_type"] == "steps"


def test_context_packer_preserves_parent_ranking_metadata_on_section_docs():
    parent = Document(
        page_content=RECIPE_MARKDOWN,
        metadata={"dish_name": "宫保鸡丁", "parent_id": "parent-ranked", "rrf_score": 0.87},
    )
    packer = ContextPacker(max_chars_total=2000, max_chars_per_doc=1000, max_docs=5)

    pack = packer.build_context_pack(
        query="宫保鸡丁怎么做",
        retrieval_result={"chunks": []},
        query_plan={"route_type": "detail", "filters": {"content_type": "steps"}},
        execution_plan={"action": "retrieve_detail"},
        turn_info={"action": "retrieve_detail", "answer_mode_hint": "recipe_detail"},
        parent_docs=[parent],
    )

    assert pack["context_docs"][0].metadata["rrf_score"] == 0.87
    assert pack["context_docs"][0].metadata["parent_id"] == "parent-ranked"
    assert pack["context_docs"][0].metadata["context_pack_mode"] == "section"


def test_context_packer_substitution_selects_ingredients_and_tips():
    parent = _parent_doc()
    packer = ContextPacker(max_chars_total=2000, max_chars_per_doc=800, max_docs=5)

    pack = packer.build_context_pack(
        query="这个能不放辣椒吗",
        retrieval_result={"chunks": []},
        query_plan={"route_type": "detail", "filters": {}},
        execution_plan={"action": "retrieve_detail"},
        turn_info={"action": "substitution", "answer_mode_hint": "substitution"},
        parent_docs=[parent],
    )

    combined = "\n".join(doc.page_content for doc in pack["context_docs"])
    assert pack["answer_mode"] == "substitution"
    assert "## 必备原料和工具" in combined
    assert "## 附加内容" in combined
    section_types = {section["section_type"] for section in pack["selected_sections"]}
    assert "ingredients" in section_types
    assert "tips" in section_types


def test_context_packer_recommendation_uses_compact_parent_summaries():
    parent = _parent_doc()
    packer = ContextPacker(max_chars_total=2000, max_chars_per_doc=300, max_docs=5)

    pack = packer.build_context_pack(
        query="推荐几个鸡肉菜",
        retrieval_result={"chunks": []},
        query_plan={"route_type": "list", "filters": {}},
        execution_plan={"action": "retrieve_list"},
        turn_info={"action": "retrieve_list", "answer_mode_hint": "recommendation"},
        parent_docs=[parent],
    )

    assert pack["answer_mode"] == "recommendation"
    assert len(pack["context_docs"]) == 1
    assert pack["context_docs"][0].metadata["context_pack_mode"] == "summary"
    assert len(pack["context_docs"][0].page_content) <= 300


def test_context_packer_trims_selected_sections_with_trace():
    long_steps = "# 长菜的做法\n\n## 操作\n\n" + "\n".join(f"- 第 {i} 步操作说明很长" for i in range(50))
    parent = Document(page_content=long_steps, metadata={"dish_name": "长菜", "parent_id": "p-long"})
    packer = ContextPacker(max_chars_total=180, max_chars_per_doc=120, max_docs=5)

    pack = packer.build_context_pack(
        query="长菜怎么做",
        retrieval_result={"chunks": []},
        query_plan={"route_type": "detail", "filters": {"content_type": "steps"}},
        execution_plan={"action": "retrieve_detail"},
        turn_info={"action": "retrieve_detail", "answer_mode_hint": "recipe_detail"},
        parent_docs=[parent],
    )

    assert len(pack["context_docs"][0].page_content) <= 120
    assert pack["trace"]["trimmed"] is True
    assert pack["context_docs"][0].metadata["section_type"] == "steps"


def test_context_packer_falls_back_to_retrieved_chunks_when_parent_sections_missing():
    chunk = Document(page_content="检索块里的做法摘要", metadata={"dish_name": "未知菜", "content_type": "steps"})
    packer = ContextPacker(max_chars_total=2000, max_chars_per_doc=500, max_docs=5)

    pack = packer.build_context_pack(
        query="未知菜怎么做",
        retrieval_result={"chunks": [chunk]},
        query_plan={"route_type": "detail", "filters": {"content_type": "steps"}},
        execution_plan={"action": "retrieve_detail"},
        turn_info={"action": "retrieve_detail", "answer_mode_hint": "recipe_detail"},
        parent_docs=[],
    )

    assert len(pack["context_docs"]) == 1
    assert pack["context_docs"][0].page_content == "检索块里的做法摘要"
    assert pack["context_docs"][0].metadata["context_pack_mode"] == "chunk_fallback"
    assert pack["context_docs"][0].metadata["dish_name"] == "未知菜"
    assert pack["answer_mode"] == "recipe_detail"
    assert pack["parent_docs"] == []
    assert pack["selected_sections"] == []
    assert pack["trace"]["parent_doc_count"] == 0
    assert pack["trace"]["selected_section_count"] == 0
    assert pack["trace"]["fallback_to_chunks"] is True
