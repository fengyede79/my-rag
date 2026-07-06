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


def test_ask_question_uses_turn_runtime_context():
    source = _function_source("ask_question")

    assert "TurnRuntimeContext.start(" in source
    assert "read_state_version" in source
    assert "max_replan_count" in Path(ROOT / "rag_modules" / "turn_runtime.py").read_text(encoding="utf-8")


def test_generation_path_checks_version_before_generation():
    source = _function_source("_ask_question_once")

    context_pack_index = source.index("context_pack = self.context_packer.build_context_pack(")
    # Find the pre-generation check (second occurrence of check_state_version)
    first_check = source.index("check_state_version(")
    version_check_index = source.index("check_state_version(", first_check + 1)
    generation_index = min(
        source.index("self._generate_list_response("),
        source.index("self._generate_detail_response("),
    )

    assert context_pack_index < version_check_index < generation_index


def test_resolution_path_checks_version_before_planning():
    source = _function_source("_ask_question_once")

    resolution_index = source.index("resolution")
    checkpoint_index = source.index("post_resolution_pre_plan")
    planning_index = min(
        source.index("build_execution_plan("),
        source.index("self._build_query_plan("),
    )

    assert resolution_index < checkpoint_index < planning_index


def test_old_stream_writeback_wrapper_is_not_lifecycle_owner():
    source = MAIN.read_text(encoding="utf-8")

    assert "def _wrap_stream_with_writeback" not in source
    assert "def _wrap_stream_with_lifecycle" in source
    lifecycle_source = _function_source("_wrap_stream_with_lifecycle")
    assert "GeneratorExit" in lifecycle_source
    assert "client_disconnect_or_stream_not_consumed" in lifecycle_source
    assert "\"status\": \"completed\"" in lifecycle_source
    assert "\"status\": \"aborted\"" in lifecycle_source


def test_writeback_uses_expected_state_version():
    source = _function_source("_write_conversation_turn")

    assert "expected_state_version" in source
    assert "writeback_turn_state(" in source
