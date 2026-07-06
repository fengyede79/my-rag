import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "main.py"


def _function_source(function_name: str) -> str:
    source = MAIN.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            return ast.get_source_segment(source, node)
    raise AssertionError(f"{function_name} not found")


def test_old_search_relevant_chunks_method_is_removed():
    source = MAIN.read_text(encoding="utf-8")

    assert "def _search_relevant_chunks(" not in source


def test_ask_question_uses_retrieval_executor_not_old_low_level_branching():
    ask_question_source = _function_source("_ask_question_once")

    assert "build_retrieval_query_plan(" in ask_question_source
    assert "self.retrieval_executor.execute(" in ask_question_source
    assert "_search_relevant_chunks(" not in ask_question_source
    assert "metadata_filtered_search(" not in ask_question_source
    assert "hybrid_search(" not in ask_question_source
    assert "fallback_filters" not in ask_question_source
    assert "pop(\"content_type\"" not in ask_question_source
    assert "pop('content_type'" not in ask_question_source


def test_ask_question_does_not_use_empty_chunks_as_quality_gate():
    ask_question_source = _function_source("_ask_question_once")

    assert "if not relevant_chunks" not in ask_question_source
    assert "retrieval_result[\"low_evidence\"]" in ask_question_source
