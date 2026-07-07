from langchain_core.documents import Document

from rag_modules.generation_integration import GenerationIntegrationModule
from rag_modules.structured_generation import try_build_constraint_answer


def _module() -> GenerationIntegrationModule:
    return GenerationIntegrationModule.__new__(GenerationIntegrationModule)


def test_tips_can_fall_back_to_steps_when_no_tips_section_exists():
    module = _module()
    doc = Document(
        page_content=(
            "# 老干妈拌面的做法\n"
            "这是一道快手主食。\n"
            "## 操作\n"
            "1. 将水倒入锅中并煮沸\n"
            "2. 将面放入锅中，过程中注意搅拌，避免面粘成一坨\n"
        ),
        metadata={"dish_name": "老干妈拌面"},
    )

    answer = module._try_build_structured_answer(
        "老干妈拌面有什么制作技巧",
        [doc],
        "tips",
    )

    assert answer is not None
    assert "老干妈拌面" in answer
    assert "搅拌" in answer


def test_placeholder_only_section_is_not_returned_as_structured_answer():
    module = _module()
    doc = Document(
        page_content=(
            "# 示例菜的做法\n"
            "## 必备原料和工具\n"
            "<!-- 在这里列出必需原料。 -->\n"
            "<!-- 注意：这里不要输出模板注释。 -->\n"
        ),
        metadata={"dish_name": "示例菜"},
    )

    answer = module._try_build_structured_answer(
        "示例菜需要什么食材",
        [doc],
        "ingredients",
    )

    assert answer is None


def test_rule_router_keeps_dish_name_for_tips_queries():
    module = _module()

    intent = module._rule_based_routing("老干妈拌面有什么制作技巧")

    assert intent["type"] == "detail"
    assert intent["filters"]["content_type"] == "tips"
    assert intent["dish_name"] == "老干妈拌面"


def test_constraint_answer_returns_suitable_verdict_for_bento_query():
    doc = Document(
        page_content=(
            "# 可乐鸡翅\n"
            "## 操作\n"
            "1. 鸡翅焯水\n"
            "2. 加入可乐炖煮20分钟\n"
            "## 附加内容\n"
            "- 适合带饭，加热后风味更佳\n"
            "- 可以提前一天做好\n"
        ),
        metadata={"dish_name": "可乐鸡翅"},
    )

    answer = try_build_constraint_answer("这个适合带饭吗", [doc], "constraint_check")

    assert answer is not None
    assert "适合" in answer
    assert "可乐鸡翅" in answer
    assert "带饭" in answer


def test_constraint_answer_returns_uncertain_when_no_matching_tips():
    doc = Document(
        page_content=(
            "# 蛋炒饭\n"
            "## 操作\n"
            "1. 米饭放凉\n"
            "2. 热锅下油炒蛋\n"
        ),
        metadata={"dish_name": "蛋炒饭"},
    )

    answer = try_build_constraint_answer("这个适合减脂吗", [doc], "constraint_check")

    assert answer is not None
    assert "不确定" in answer


def test_substitution_answer_includes_ingredient_reference():
    doc = Document(
        page_content=(
            "# 麻婆豆腐\n"
            "## 必备原料和工具\n"
            "- 豆腐 400g\n"
            "- 肉末 100g\n"
            "- 豆瓣酱 2勺\n"
            "## 附加内容\n"
            "- 没有豆瓣酱可以用辣酱替代\n"
        ),
        metadata={"dish_name": "麻婆豆腐"},
    )

    answer = try_build_constraint_answer("没有豆瓣酱怎么办", [doc], "substitution")

    assert answer is not None
    assert "麻婆豆腐" in answer
    assert "替代" in answer or "食材" in answer


def test_constraint_answer_returns_none_for_non_constraint_mode():
    doc = Document(page_content="# 蛋炒饭", metadata={"dish_name": "蛋炒饭"})
    assert try_build_constraint_answer("蛋炒饭怎么做", [doc], "recipe_detail") is None


def test_constraint_answer_returns_none_without_context():
    assert try_build_constraint_answer("这个适合带饭吗", [], "constraint_check") is None
