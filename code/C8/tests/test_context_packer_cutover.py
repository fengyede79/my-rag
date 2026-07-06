import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "main.py"


def _function_def(function_name: str) -> ast.FunctionDef:
    source = MAIN.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            return node
    raise AssertionError(f"{function_name} not found")


def _function_source(function_name: str) -> str:
    source = MAIN.read_text(encoding="utf-8")
    return ast.get_source_segment(source, _function_def(function_name))


def _arg_names(function_name: str) -> list[str]:
    return [arg.arg for arg in _function_def(function_name).args.args]


def test_ask_question_builds_context_pack_before_generation_helpers():
    source = _function_source("ask_question")

    parent_index = source.index("parent_docs = self.data_module.get_parent_documents(")
    pack_index = source.index("context_pack = self.context_packer.build_context_pack(")
    list_index = source.index("self._generate_list_response(")
    detail_index = source.index("self._generate_detail_response(")

    assert parent_index < pack_index < list_index
    assert parent_index < pack_index < detail_index


def test_generation_helpers_do_not_expand_parent_documents():
    list_source = _function_source("_generate_list_response")
    detail_source = _function_source("_generate_detail_response")

    assert "get_parent_documents(" not in list_source
    assert "get_parent_documents(" not in detail_source
    assert "context_pack[\"context_docs\"]" in list_source
    assert "context_pack[\"context_docs\"]" in detail_source
    assert "context_pack[\"parent_docs\"]" not in list_source
    assert "context_pack[\"parent_docs\"]" not in detail_source


def test_generation_helper_signatures_match_context_pack_contract():
    assert _arg_names("_generate_list_response") == ["self", "question", "context_pack"]
    assert _arg_names("_generate_detail_response") == [
        "self",
        "question",
        "stream",
        "route_type",
        "dish_name",
        "context_pack",
    ]


def test_generation_helpers_pass_packed_context_docs_to_generation():
    list_source = _function_source("_generate_list_response")
    detail_source = _function_source("_generate_detail_response")

    assert "context_docs = list(context_pack[\"context_docs\"])" in list_source
    assert "generate_list_answer(question, context_docs)" in list_source
    assert "context_docs = list(context_pack[\"context_docs\"])" in detail_source
    assert "generate_step_by_step_answer_stream(" in detail_source
    assert "generate_step_by_step_answer(" in detail_source
    assert "generate_basic_answer_stream(" in detail_source
    assert "generate_basic_answer(" in detail_source


def test_generation_helpers_do_not_assign_latest_parent_docs():
    list_source = _function_source("_generate_list_response")
    detail_source = _function_source("_generate_detail_response")

    assert "_latest_parent_docs" not in list_source
    assert "_latest_parent_docs" not in detail_source


def test_generation_helpers_have_no_dead_parameters():
    detail_source = _function_source("_generate_detail_response")

    assert "session_id" not in _arg_names("_generate_list_response")
    assert "session_id" not in _arg_names("_generate_detail_response")
    assert "filters" not in _arg_names("_generate_detail_response")
    assert "entities" not in _arg_names("_generate_detail_response")
    assert "filters.get" not in detail_source


def test_ask_question_records_context_pack_trace_only_on_execution_result():
    source = _function_source("ask_question")

    assert "query_plan[\"context_pack_trace\"]" not in source
    assert "query_plan[\"answer_mode\"]" not in source
    assert "execution_result[\"context_pack_trace\"]" in source
    assert "execution_result[\"answer_mode\"]" in source


def test_context_packer_is_configured_from_rag_config():
    source = MAIN.read_text(encoding="utf-8")

    assert "max_chars_total=self.config.context_pack_max_chars_total" in source
    assert "max_chars_per_doc=self.config.context_pack_max_chars_per_doc" in source
    assert "max_docs=self.config.context_pack_max_docs" in source


def test_context_packer_submodule_is_exported():
    init_source = (ROOT / "rag_modules" / "__init__.py").read_text(encoding="utf-8")

    assert "from . import context_packer" in init_source
    assert "'context_packer'" in init_source or '"context_packer"' in init_source


def test_generation_logs_describe_packed_context_docs():
    list_source = _function_source("_generate_list_response")
    detail_source = _function_source("_generate_detail_response")

    assert "传入生成的上下文" in list_source
    assert "传入生成的上下文" in detail_source
    assert "找到文档" not in list_source
    assert "找到文档" not in detail_source
