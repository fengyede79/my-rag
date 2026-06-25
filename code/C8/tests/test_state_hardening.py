from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

from langchain_core.documents import Document

from main import RecipeRAGSystem
from rag_modules.conversation_manager import ConversationManager
from rag_modules.generation_integration import GenerationIntegrationModule


class _StubGenerationModule(GenerationIntegrationModule):
    def __init__(self):
        self.conversation_manager = ConversationManager()
        self.last_generation_trace = {}

    def resolve_query_reference(self, query, session_id):
        return query

    def query_router(self, query):
        if "技巧" in query:
            return {
                "type": "detail",
                "filters": {"content_type": "tips"},
                "dish_name": "煎饭" if "煎饭" in query else None,
                "confidence": 0.95,
            }
        return {
            "type": "detail",
            "filters": {"content_type": "steps" if "怎么做" in query else "ingredients"},
            "dish_name": "蛋炒饭" if "蛋炒饭" in query else None,
            "confidence": 0.95,
        }

    def get_current_entity(self, session_id):
        return self.conversation_manager.get_current_entity(session_id)

    def _classify_query_guardrail(self, query):
        return None

    def query_rewrite(self, query):
        return query

    def generate_step_by_step_answer_stream(self, query, context_docs, content_type=None):
        yield "步骤1"

    def generate_step_by_step_with_conversation(
        self,
        query,
        context_docs,
        session_id,
        intent_type="detail",
        entities=None,
        content_type=None,
    ):
        return "步骤1"

    def generate_list_answer(self, query, context_docs):
        return "1. 蛋炒饭"

    def save_recommendations(self, *args, **kwargs):
        return None

    def _record_generation_trace(self, *args, **kwargs):
        self.last_generation_trace = {"strategy": args[0] if args else "unknown"}


class _StreamingConversationGenerationModule(_StubGenerationModule):
    def __init__(self):
        super().__init__()
        self.stream_calls = []

    def generate_step_by_step_answer_stream_with_conversation(
        self,
        query,
        context_docs,
        session_id,
        intent_type="detail",
        entities=None,
        content_type=None,
    ):
        conversation_context = self.conversation_manager.get_conversation_context(session_id)
        self.stream_calls.append(conversation_context)
        yield f"CTX:{conversation_context or 'EMPTY'}"


class _StubRetrievalModule:
    last_search_trace = {}

    def metadata_filtered_search(self, *args, **kwargs):
        return [Document(page_content="# 蛋炒饭", metadata={"dish_name": "蛋炒饭"})]

    def hybrid_search(self, *args, **kwargs):
        return [Document(page_content="# 蛋炒饭", metadata={"dish_name": "蛋炒饭"})]


class _StubDataModule:
    def get_parent_documents(self, chunks, target_dish_name=None):
        return [Document(page_content="# 蛋炒饭", metadata={"dish_name": "蛋炒饭"})]


def _system_with_generation(module):
    system = RecipeRAGSystem.__new__(RecipeRAGSystem)
    system.config = SimpleNamespace(top_k=3)
    system.data_module = _StubDataModule()
    system.retrieval_module = _StubRetrievalModule()
    system.generation_module = module
    system._latest_parent_docs = []
    system.last_query_diagnostics = {}
    return system


def test_intent_switch_clears_conversation_history_context():
    manager = ConversationManager()
    manager.add_interaction(
        "switch-session",
        "我们聊聊蛋炒饭",
        "好的",
        intent_type="general",
        entities={"dish_name": "蛋炒饭"},
    )

    switched = manager.complete_query("switch-session", "换个话题")

    assert switched == "换个话题"
    assert manager.get_current_entity("switch-session") is None
    assert manager.get_conversation_context("switch-session") == ""


def test_stream_detail_turn_uses_conversation_context():
    module = _StreamingConversationGenerationModule()
    module.conversation_manager.add_interaction(
        "stream-followup",
        "我们聊聊蛋炒饭",
        "好的",
        intent_type="general",
        entities={"dish_name": "蛋炒饭"},
    )
    system = _system_with_generation(module)

    response = system.ask_question("再说一下怎么做", stream=True, session_id="stream-followup")

    assert list(response) == ["CTX:用户: 我们聊聊蛋炒饭\n助手: 好的"]
    assert module.stream_calls == ["用户: 我们聊聊蛋炒饭\n助手: 好的"]


def test_conversation_manager_is_safe_under_parallel_access():
    manager = ConversationManager()

    def worker(index: int):
        session_id = f"parallel-{index % 4}"
        manager.add_interaction(
            session_id,
            f"query-{index}",
            f"answer-{index}",
            intent_type="general",
            entities={"dish_name": f"dish-{index % 3}"},
        )
        manager.get_conversation_context(session_id)
        manager.complete_query(session_id, "它怎么做")
        if index % 5 == 0:
            manager.reset_session(session_id)

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(worker, i) for i in range(40)]
        for future in futures:
            future.result()
