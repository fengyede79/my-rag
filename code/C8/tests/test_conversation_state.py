from types import SimpleNamespace

from langchain_core.documents import Document

from main import RecipeRAGSystem
from rag_modules.conversation_manager import ConversationManager
from rag_modules.generation_integration import GenerationIntegrationModule


def _module() -> GenerationIntegrationModule:
    module = GenerationIntegrationModule.__new__(GenerationIntegrationModule)
    module.conversation_manager = ConversationManager()
    return module


def test_add_interaction_updates_current_entity_from_entities():
    manager = ConversationManager()

    manager.add_interaction(
        "session-a",
        "我们聊聊老干妈拌面",
        "好的，我们来聊聊这道菜。",
        intent_type="general",
        entities={"dish_name": "老干妈拌面"},
    )

    assert manager.get_current_entity("session-a") == "老干妈拌面"


def test_complete_query_inherits_current_entity_for_followup_detail_question():
    manager = ConversationManager()
    manager.add_interaction(
        "session-b",
        "我们聊聊煮泡面加蛋",
        "可以。",
        intent_type="general",
        entities={"dish_name": "煮泡面加蛋"},
    )

    completed = manager.complete_query("session-b", "再说一下怎么做")

    assert completed == "煮泡面加蛋怎么做"


def test_complete_query_resolves_pronoun_without_duplicate_dish_name():
    manager = ConversationManager()
    manager.add_interaction(
        "session-c",
        "我们聊聊老干妈拌面",
        "可以。",
        intent_type="general",
        entities={"dish_name": "老干妈拌面"},
    )

    completed = manager.complete_query("session-c", "它需要什么食材")

    assert completed == "老干妈拌面需要什么食材"


def test_query_router_keeps_conversational_dish_topic():
    module = _module()

    intent = module._rule_based_routing("我们聊聊电饭煲三文鱼焖饭")

    assert intent["type"] == "general"
    assert intent["dish_name"] == "电饭煲三文鱼焖饭"
    assert intent["confidence"] >= 0.9


def test_query_router_prefers_conversational_dish_intent_over_list_fallback():
    module = _module()
    module._hybrid_route = lambda query: ("list", "llm", {"confidence": 0.7})

    intent = module.query_router("我们聊聊老干妈拌面")

    assert intent["type"] == "general"
    assert intent["dish_name"] == "老干妈拌面"


class _StubGenerationModule(GenerationIntegrationModule):
    def __init__(self):
        self.conversation_manager = ConversationManager()
        self.last_generation_trace = {}

    def resolve_query_reference(self, query, session_id):
        return query

    def query_router(self, query):
        if "今天吃什么" in query:
            return {"type": "list", "filters": {}, "dish_name": None, "confidence": 0.95}
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

    def generate_step_by_step_answer_stream_with_conversation(
        self,
        query,
        context_docs,
        session_id,
        intent_type="detail",
        entities=None,
        content_type=None,
    ):
        response = self.generate_step_by_step_with_conversation(
            query,
            context_docs,
            session_id,
            intent_type=intent_type,
            entities=entities,
            content_type=content_type,
        )
        yield response

    def generate_step_by_step_with_conversation(
        self,
        query,
        context_docs,
        session_id,
        intent_type="detail",
        entities=None,
        content_type=None,
    ):
        response = self._try_build_structured_answer(query, context_docs, content_type) or "步骤1"
        self.conversation_manager.add_interaction(
            session_id,
            query,
            response,
            intent_type=intent_type,
            entities=entities or {},
        )
        return response

    def generate_list_answer(self, query, context_docs):
        return "为您推荐以下菜品：\n1. 蛋炒饭"

    def save_recommendations(self, *args, **kwargs):
        return None

    def _record_generation_trace(self, *args, **kwargs):
        self.last_generation_trace = {"strategy": args[0] if args else "unknown"}


class _StubRetrievalModule:
    last_search_trace = {}

    def extract_filters_from_query(self, query):
        return {}

    def metadata_filtered_search(self, *args, **kwargs):
        return [Document(page_content="# 蛋炒饭", metadata={"dish_name": "蛋炒饭"})]

    def hybrid_search(self, *args, **kwargs):
        return [Document(page_content="# 蛋炒饭", metadata={"dish_name": "蛋炒饭"})]


class _TipsFallbackRetrievalModule:
    last_search_trace = {}

    def extract_filters_from_query(self, query):
        return {}

    def metadata_filtered_search(self, query, filters, top_k=3, query_dish=None):
        if filters.get("content_type") == "tips":
            return []
        return [Document(page_content="# 煎饭的做法", metadata={"dish_name": "煎饭"})]

    def hybrid_search(self, *args, **kwargs):
        return [Document(page_content="# 煎饭的做法", metadata={"dish_name": "煎饭"})]


class _StubDataModule:
    def get_parent_documents(self, chunks, target_dish_name=None):
        return [Document(page_content="# 蛋炒饭", metadata={"dish_name": "蛋炒饭"})]


class _TipsFallbackDataModule:
    def get_parent_documents(self, chunks, target_dish_name=None):
        return [
            Document(
                page_content=(
                    "# 煎饭的做法\n"
                    "## 操作\n"
                    "1. 先热锅，再下油。\n"
                    "2. 煎到底部定型后再加水。\n"
                    "3. 收干前不要频繁翻动。\n"
                ),
                metadata={"dish_name": "煎饭"},
            )
        ]


def _system() -> RecipeRAGSystem:
    system = RecipeRAGSystem.__new__(RecipeRAGSystem)
    system.config = SimpleNamespace(top_k=3)
    system.data_module = _StubDataModule()
    system.retrieval_module = _StubRetrievalModule()
    system.generation_module = _StubGenerationModule()
    system._latest_parent_docs = []
    system.last_query_diagnostics = {}
    return system


def _tips_system() -> RecipeRAGSystem:
    system = RecipeRAGSystem.__new__(RecipeRAGSystem)
    system.config = SimpleNamespace(top_k=3)
    system.data_module = _TipsFallbackDataModule()
    system.retrieval_module = _TipsFallbackRetrievalModule()
    system.generation_module = _StubGenerationModule()
    system._latest_parent_docs = []
    system.last_query_diagnostics = {}
    return system


def test_stream_detail_turn_persists_conversation_state_after_stream_consumed():
    system = _system()

    response = system.ask_question("蛋炒饭怎么做", stream=True, session_id="stream-session")

    assert list(response) == ["步骤1"]
    assert system.generation_module.conversation_manager.get_current_entity("stream-session") == "蛋炒饭"


def test_list_turn_is_recorded_in_conversation_history():
    system = _system()

    answer = system.ask_question("今天吃什么", stream=False, session_id="list-session")

    assert "蛋炒饭" in answer
    session = system.generation_module.conversation_manager.get_session("list-session")
    assert len(session.messages) == 2


def test_tips_query_falls_back_to_same_dish_when_tips_chunks_are_missing():
    system = _tips_system()

    answer = system.ask_question("煎饭有什么制作技巧", stream=False, session_id="tips-session")

    assert "煎饭" in answer
    assert "热锅" in answer
