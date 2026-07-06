import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "main.py"


def _source() -> str:
    return MAIN.read_text(encoding="utf-8")


def _function_source(function_name: str) -> str:
    source = _source()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            return ast.get_source_segment(source, node)
    raise AssertionError(f"{function_name} not found")


def test_runtime_chain_uses_shared_context_and_lifecycle_wrapper():
    source = _source()
    ask_source = _function_source("ask_question")

    assert "TurnRuntimeContext.start(" in ask_source
    assert "_ask_question_once(" in ask_source
    assert "should_replan_after_mismatch(" in ask_source
    assert "def _wrap_stream_with_writeback" not in source
    assert "def _wrap_stream_with_lifecycle" in source


def test_chat_path_retrieval_only_goes_through_retrieval_executor():
    once_source = _function_source("_ask_question_once")

    assert "self.retrieval_executor.execute(" in once_source
    assert ".metadata_filtered_search(" not in once_source
    assert ".hybrid_search(" not in once_source


def test_generation_helpers_do_not_expand_parent_docs_or_write_state():
    for function_name in ["_generate_list_response", "_generate_detail_response"]:
        helper_source = _function_source(function_name)
        assert "get_parent_documents(" not in helper_source
        assert "conversation_manager" not in helper_source
        assert "writeback_turn_state(" not in helper_source
        assert "record_recommendations(" not in helper_source
        assert "set_current_dish(" not in helper_source
        assert "add_interaction(" not in helper_source


def test_state_writeback_uses_policy_and_expected_version():
    write_source = _function_source("_write_conversation_turn")
    manager_source = (ROOT / "rag_modules" / "conversation_manager.py").read_text(encoding="utf-8")

    assert "writeback_turn_state(" in write_source
    assert "expected_state_version" in write_source
    assert "build_state_diff(" in manager_source
    assert "commit_state_diff(" in manager_source


def test_legacy_convenience_search_helpers_are_not_left_as_independent_runtime_paths():
    source = _source()

    assert "def search_by_category" not in source
    assert "def get_ingredients_list" not in source
