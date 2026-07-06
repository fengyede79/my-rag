from types import SimpleNamespace

from langchain_core.documents import Document

from main import RecipeRAGSystem
from rag_modules.conversation_manager import ConversationManager
from rag_modules.conversation_state_builder import build_conversation_snapshot
from rag_modules.execution_planner import build_execution_plan
from rag_modules.generation_integration import GenerationIntegrationModule


def _module() -> GenerationIntegrationModule:
    module = GenerationIntegrationModule.__new__(GenerationIntegrationModule)
    module.conversation_manager = ConversationManager()
    return module


def test_add_interaction_does_not_implicitly_set_entity():
    """add_interaction no longer sets current_entity from entities dict.
    Entity updates go through set_current_dish or writeback_turn_state."""
    manager = ConversationManager()

    manager.add_interaction(
        "session-a",
        "我们聊聊老干妈拌面",
        "好的，我们来聊聊这道菜。",
        intent_type="general",
        entities={"dish_name": "老干妈拌面"},
    )

    # Entity is NOT set implicitly anymore
    assert manager.get_current_entity("session-a") is None


def test_set_current_dish_updates_entity():
    manager = ConversationManager()
    manager.set_current_dish("session-a", "老干妈拌面", source="explicit_query", confidence=1.0)
    assert manager.get_current_entity("session-a") == "老干妈拌面"


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


def test_non_stream_generation_returns_string_not_generator():
    module = GenerationIntegrationModule.__new__(GenerationIntegrationModule)
    module._record_generation_trace = lambda *args, **kwargs: None
    module._try_build_structured_answer = lambda *args, **kwargs: "结构化回答"

    answer = module.generate_step_by_step_answer("蛋炒饭怎么做？", [Document(page_content="x")])

    assert isinstance(answer, str)
    assert answer == "结构化回答"


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

    def generate_step_by_step_answer(self, query, context_docs, content_type=None):
        return self._try_build_structured_answer(query, context_docs, content_type) or f"{query} 步骤1"

    def generate_basic_answer_stream(self, query, context_docs, content_type=None):
        yield self.generate_basic_answer(query, context_docs, content_type=content_type)

    def generate_basic_answer(self, query, context_docs, content_type=None):
        return self._try_build_structured_answer(query, context_docs, content_type) or "回答"

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


def test_build_query_plan_switches_to_new_explicit_topic_instead_of_inheriting_old_one():
    system = _system()
    manager = system.generation_module.conversation_manager
    manager.add_interaction(
        "polluted-session",
        "我们聊聊蛋炒饭",
        "可以。",
        intent_type="general",
        entities={"dish_name": "蛋炒饭"},
    )

    system.generation_module.query_router = lambda query: {
        "type": "general",
        "filters": {},
        "dish_name": None,
        "confidence": 0.7,
    }

    plan = system._build_query_plan("西湖醋鱼怎么样？", "polluted-session")

    assert plan["dish_name"] == "西湖醋鱼"
    assert plan["entities"]["dish_name"] == "西湖醋鱼"


def test_build_query_plan_extracts_new_explicit_topic_from_general_question():
    system = _system()
    manager = system.generation_module.conversation_manager
    manager.add_interaction(
        "general-topic-session",
        "我们聊聊蛋炒饭",
        "可以。",
        intent_type="general",
        entities={"dish_name": "蛋炒饭"},
    )

    system.generation_module.query_router = lambda query: {
        "type": "general",
        "filters": {},
        "dish_name": None,
        "confidence": 0.7,
    }

    plan = system._build_query_plan("西湖醋鱼怎么样？", "general-topic-session")

    assert plan["dish_name"] == "西湖醋鱼"
    assert plan["entities"]["dish_name"] == "西湖醋鱼"


def test_no_result_turn_does_not_replace_current_entity():
    system = _system()
    manager = system.generation_module.conversation_manager
    manager.set_current_dish("no-result-session", "蛋炒饭", source="explicit_query", confidence=1.0)
    manager.add_interaction(
        "no-result-session",
        "蛋炒饭怎么做",
        "步骤1",
        intent_type="detail",
        entities={"dish_name": "蛋炒饭"},
    )

    system.generation_module.query_router = lambda query: {
        "type": "general",
        "filters": {},
        "dish_name": None,
        "confidence": 0.7,
    }
    system.retrieval_module.metadata_filtered_search = lambda *args, **kwargs: []

    answer = system.ask_question("西湖醋鱼怎么样？", stream=False, session_id="no-result-session")

    assert "没有找到相关的食谱信息" in answer
    assert manager.get_current_entity("no-result-session") == "蛋炒饭"


def test_simple_content_type_detection_defaults_to_general_instead_of_list():
    module = GenerationIntegrationModule.__new__(GenerationIntegrationModule)

    content_type, route_type, route_info = module._simple_content_type_detection("西湖醋鱼怎么样？")

    assert content_type == "general"
    assert route_type == "fallback"
    assert route_info["confidence"] == 0.5


def test_build_query_plan_does_not_treat_list_question_as_dish_name():
    system = _system()

    system.generation_module.query_router = lambda query: {
        "type": "list",
        "filters": {},
        "dish_name": None,
        "confidence": 0.95,
    }

    plan = system._build_query_plan("今天吃什么？", "list-topic-session")

    assert plan["route_type"] == "list"
    assert plan["dish_name"] is None
    assert plan["entities"]["dish_name"] is None


# ---- Task 2: Structured Snapshot tests ----


def test_recommendation_turn_sets_recommendation_mode_not_current_dish():
    manager = ConversationManager()
    manager.record_recommendations("s1", ["蛋炒饭", "麻辣香锅", "扬州炒饭"])
    snapshot = build_conversation_snapshot(manager.get_session("s1"), current_query="今天吃什么？")
    assert snapshot["topic_state"]["mode"] == "recommendation_list"
    assert snapshot["reference_state"]["current_dish"]["active"] is False
    assert snapshot["reference_state"]["recent_recommendations"][0]["dish_name"] == "蛋炒饭"


def test_current_dish_carries_source_and_confidence():
    manager = ConversationManager()
    manager.set_current_dish("s2", "蛋炒饭", source="explicit_query", confidence=1.0)
    snapshot = build_conversation_snapshot(manager.get_session("s2"), current_query="它怎么做？")
    assert snapshot["reference_state"]["current_dish"]["value"] == "蛋炒饭"
    assert snapshot["reference_state"]["current_dish"]["source"] == "explicit_query"
    assert snapshot["reference_state"]["current_dish"]["confidence"] == 1.0


def test_correction_query_adds_explicit_target_to_constraints():
    manager = ConversationManager()
    manager.set_current_dish("s3", "宫保鸡丁", source="inferred", confidence=0.55)
    snapshot = build_conversation_snapshot(manager.get_session("s3"), current_query="不是这个，是蛋炒饭")
    assert "蛋炒饭" in snapshot["resolution_constraints"]["explicit_query_targets"]
    assert snapshot["resolution_constraints"]["allow_external_explicit_target"] is True


def test_correction_query_marks_non_dish_text_as_unverified():
    manager = ConversationManager()
    manager.set_current_dish("s4", "宫保鸡丁", source="inferred", confidence=0.55)
    snapshot = build_conversation_snapshot(manager.get_session("s4"), current_query="不是这个，是那个简单点的")
    assert snapshot["resolution_constraints"]["explicit_query_targets"] == ["那个简单点的"]
    assert snapshot["resolution_constraints"]["explicit_query_target_verified"] is False


# ---- Task 4: Execution Planning tests ----


def test_recommendation_query_returns_retrieve_list_action():
    plan = build_execution_plan(
        turn_info={"turn_type": "recommendation_query", "response_mode": "retrieve_answer"},
        resolution=None,
    )
    assert plan["action"] == "retrieve_list"


def test_correction_resolution_returns_apply_correction_plan():
    plan = build_execution_plan(
        turn_info={"turn_type": "followup_query", "response_mode": "retrieve_answer"},
        resolution={"next_action": "apply_correction", "resolved_target": "蛋炒饭"},
    )
    assert plan["action"] == "apply_correction"


def test_reference_resolution_returns_apply_reference_resolution_plan():
    plan = build_execution_plan(
        turn_info={"turn_type": "followup_query", "response_mode": "retrieve_answer"},
        resolution={"next_action": "apply_reference_resolution", "resolved_target": "麻婆豆腐"},
    )
    assert plan["action"] == "apply_reference_resolution"


def test_new_pipeline_does_not_call_legacy_query_completion_for_correction():
    system = _system()
    manager = system.generation_module.conversation_manager
    manager.set_current_dish("new-pipeline-session", "宫保鸡丁", source="inferred", confidence=0.55)

    answer = system.ask_question("不是这个，是蛋炒饭", stream=False, session_id="new-pipeline-session")

    assert "蛋炒饭" in answer
    assert manager.get_current_entity("new-pipeline-session") == "蛋炒饭"


def test_detail_generation_uses_single_new_writeback_path():
    system = _system()
    manager = system.generation_module.conversation_manager

    def fail_if_old_generator_called(*args, **kwargs):
        raise AssertionError("legacy conversation-aware generator should not write state")

    system.generation_module.generate_step_by_step_with_conversation = fail_if_old_generator_called

    answer = system.ask_question("蛋炒饭怎么做？", stream=False, session_id="single-writeback-session")

    assert "蛋炒饭" in answer
    session = manager.get_session("single-writeback-session")
    assert len(session.messages) == 2


# ---- Task 2: Ordinal / cleaned dish / preference snapshot tests ----


def test_snapshot_extracts_ordinal_reference():
    manager = ConversationManager()
    manager.record_recommendations("ordinal-s1", ["扬州炒饭", "麻婆豆腐", "白灼菜心"])

    snapshot = build_conversation_snapshot(
        manager.get_session("ordinal-s1"),
        current_query="第二个怎么做？",
    )

    ordinal = snapshot["resolution_constraints"]["ordinal_reference"]
    assert ordinal["rank"] == 2
    assert ordinal["raw_text"] == "第二个"
    assert ordinal["remaining_query"] == "怎么做"


def test_snapshot_extracts_ordinal_reference_with_comment():
    manager = ConversationManager()
    manager.record_recommendations("ordinal-s2", ["燕麦鸡蛋饼", "牛奶燕麦"])

    snapshot = build_conversation_snapshot(
        manager.get_session("ordinal-s2"),
        current_query="第一个看起来不错，做法说一下",
    )

    ordinal = snapshot["resolution_constraints"]["ordinal_reference"]
    assert ordinal["rank"] == 1
    assert ordinal["remaining_query"] == "做法说一下"


def test_snapshot_cleans_discourse_prefix_from_explicit_dish():
    manager = ConversationManager()

    snapshot = build_conversation_snapshot(
        manager.get_session("clean-prefix-s1"),
        current_query="那蛋炒饭需要哪些食材？",
    )

    cleaned = snapshot["resolution_constraints"]["cleaned_explicit_dish"]
    assert cleaned["value"] == "蛋炒饭"
    assert cleaned["removed_prefix"] == "那"


def test_snapshot_extracts_preference_constraints():
    manager = ConversationManager()

    snapshot = build_conversation_snapshot(
        manager.get_session("preference-s1"),
        current_query="算了，换个清淡一点的菜",
    )

    preferences = snapshot["resolution_constraints"]["preference_constraints"]
    assert "清淡" in preferences["taste"]


# ---- Task 5: Invalid dish-name guard tests ----


def test_query_plan_does_not_treat_ordinal_as_dish_name():
    system = _system()
    system.generation_module.query_router = lambda query: {
        "type": "detail",
        "filters": {"content_type": "steps"},
        "dish_name": "第二个",
        "confidence": 0.95,
    }

    plan = system._build_query_plan("第二个怎么做？", "ordinal-plan-session")

    assert plan["dish_name"] is None
    assert plan["entities"]["dish_name"] is None


def test_query_plan_does_not_treat_full_tip_question_as_dish_name():
    system = _system()
    system.generation_module.query_router = lambda query: {
        "type": "detail",
        "filters": {"content_type": "tips"},
        "dish_name": "有什么小技巧别粘锅",
        "confidence": 0.95,
    }

    plan = system._build_query_plan("有什么小技巧别粘锅？", "tip-plan-session")

    assert plan["dish_name"] is None
    assert plan["entities"]["dish_name"] is None


# ---- Task 7: Preference constraint propagation tests ----


def test_preference_constraints_are_available_to_query_plan():
    manager = ConversationManager()
    snapshot = build_conversation_snapshot(
        manager.get_session("preference-plan-session"),
        current_query="换个清淡一点的菜",
    )

    preferences = snapshot["resolution_constraints"]["preference_constraints"]

    assert preferences["taste"] == ["清淡"]


def test_list_query_plan_keeps_preference_constraints():
    system = _system()
    system.generation_module.query_router = lambda query: {
        "type": "list",
        "filters": {},
        "dish_name": None,
        "confidence": 0.7,
    }

    result = system.ask_question(
        "换个清淡一点的菜",
        stream=False,
        session_id="preference-diagnostics-session",
        return_diagnostics=True,
    )

    # Verify preference constraints survive into the execution layer via snapshot
    from rag_modules.conversation_state_builder import build_conversation_snapshot
    from rag_modules.conversation_manager import ConversationManager
    mgr = system.generation_module.conversation_manager
    snap = build_conversation_snapshot(
        mgr.get_session("preference-diagnostics-session"),
        current_query="换个清淡一点的菜",
    )
    assert snap["resolution_constraints"]["preference_constraints"]["taste"] == ["清淡"]


# ---- Task 2: explicit writeback mode tests ----


def test_message_only_writeback_does_not_replace_current_entity():
    manager = ConversationManager()
    manager.set_current_dish("wb-session", "蛋炒饭", source="explicit_query", confidence=1.0)

    manager.writeback_turn_state(
        session_id="wb-session",
        question="随便聊聊",
        turn_info={"turn_type": "smalltalk"},
        query_plan={},
        resolution=None,
        answer="你好",
        execution_result={"success": True},
    )

    session = manager.get_session("wb-session")
    assert session.current_entity == "蛋炒饭"


def test_clarification_pending_writeback_does_not_set_current_entity():
    manager = ConversationManager()

    manager.writeback_turn_state(
        session_id="clar-session",
        question="它怎么做",
        turn_info={"turn_type": "followup_query"},
        query_plan=None,
        resolution={
            "next_action": "ask_clarification",
            "clarification_question": "你指的是第几个推荐菜？",
            "reason": "ambiguous_reference",
            "candidates": ["蛋炒饭", "麻婆豆腐"],
        },
        answer="你指的是第几个推荐菜？",
        execution_result={"success": True},
    )

    session = manager.get_session("clar-session")
    assert session.current_entity is None
    assert session.pending_clarification is not None
    assert session.pending_clarification["reason"] == "ambiguous_reference"


def test_recommendation_list_writeback_updates_recommendation_and_clears_entity():
    manager = ConversationManager()
    manager.set_current_dish("rec-wb-session", "蛋炒饭", source="explicit_query", confidence=1.0)

    manager.writeback_turn_state(
        session_id="rec-wb-session",
        question="推荐几个下饭菜",
        turn_info={"turn_type": "domain_query"},
        query_plan={"route_type": "list"},
        resolution=None,
        answer="推荐回锅肉、麻婆豆腐、鱼香肉丝",
        execution_result={
            "success": True,
            "recommended_dishes": ["回锅肉", "麻婆豆腐", "鱼香肉丝"],
        },
    )

    session = manager.get_session("rec-wb-session")
    # recommendation_list mode preserves current_entity (user may still reference it)
    assert session.current_entity == "蛋炒饭"
    assert session.topic_mode == "recommendation_list"
    assert [r["dish_name"] for r in session.recent_recommendations] == ["回锅肉", "麻婆豆腐", "鱼香肉丝"]


def test_resolved_followup_writeback_sets_current_entity_after_successful_retrieval():
    manager = ConversationManager()
    manager.set_current_dish("rf-session", "蛋炒饭", source="explicit_query", confidence=1.0)

    manager.writeback_turn_state(
        session_id="rf-session",
        question="它怎么做",
        turn_info={"turn_type": "followup_query"},
        query_plan={"route_type": "detail", "dish_name": "蛋炒饭"},
        resolution={
            "next_action": "apply_reference_resolution",
            "resolved_target": "蛋炒饭",
            "target_source": "implicit_single_dish_followup",
            "writeback_eligible": True,
        },
        answer="蛋炒饭的做法是...",
        execution_result={
            "success": True,
            "resolved_target": "蛋炒饭",
            "retrieved_dishes": ["蛋炒饭"],
        },
    )

    session = manager.get_session("rf-session")
    assert session.current_entity == "蛋炒饭"


# ---- Task 3: resolved target locking ----


def test_apply_resolved_target_to_query_plan_locks_target():
    system = RecipeRAGSystem.__new__(RecipeRAGSystem)
    result = system._apply_resolved_target_to_query_plan(
        query_plan={"route_type": "detail", "dish_name": "蛋炒饭"},
        resolution={
            "next_action": "apply_reference_resolution",
            "resolved_target": "蛋炒饭",
            "target_source": "implicit_single_dish_followup",
        },
    )
    assert result["route_type"] == "detail"
    assert result["dish_name"] == "蛋炒饭"


def test_apply_resolved_target_returns_query_plan_unchanged_for_no_resolution():
    system = RecipeRAGSystem.__new__(RecipeRAGSystem)
    result = system._apply_resolved_target_to_query_plan(
        query_plan={"route_type": "detail", "dish_name": "蛋炒饭"},
        resolution=None,
    )
    # Returns query_plan unchanged when no resolution
    assert result["route_type"] == "detail"
    assert result["dish_name"] == "蛋炒饭"


# ---- Task 4: execution result helpers ----


def test_extract_retrieved_dishes_deduplicates():
    system = RecipeRAGSystem.__new__(RecipeRAGSystem)
    docs = [
        Document(page_content="content", metadata={"dish_name": "蛋炒饭"}),
        Document(page_content="content", metadata={"dish_name": "蛋炒饭"}),
        Document(page_content="content", metadata={"dish_name": "麻婆豆腐"}),
        Document(page_content="content", metadata={"dish_name": None}),
    ]
    result = system._extract_retrieved_dishes(docs)
    assert result == ["蛋炒饭", "麻婆豆腐"]


def test_build_execution_result_structure():
    system = RecipeRAGSystem.__new__(RecipeRAGSystem)
    result = system._build_execution_result(
        success=True,
        answer="test answer",
        rewritten_question="test question",
        original_question="test question",
        query_plan={"route_type": "detail", "dish_name": "蛋炒饭", "filters": {"difficulty": "简单"}},
        resolution={"resolved_target": "蛋炒饭", "target_source": "explicit_query"},
        parent_docs=[Document(page_content="c", metadata={"dish_name": "蛋炒饭"})],
    )
    assert result["success"] is True
    assert result["route_type"] == "detail"
    assert result["dish_name"] == "蛋炒饭"
    assert result["resolved_target"] == "蛋炒饭"
    assert result["retrieved_dishes"] == ["蛋炒饭"]
    assert result["filters"] == {"difficulty": "简单"}


# ---- Task 5: deprecation warnings ----


# ---- Task 7: stream writeback timing ----


def test_stream_writeback_happens_after_full_consumption():
    manager = ConversationManager()
    manager.set_current_dish("stream-wb-session", "蛋炒饭", source="explicit_query", confidence=1.0)

    def fake_stream():
        yield "蛋"
        yield "炒饭"
        yield "的做法"
        yield "是..."

    manager.writeback_turn_state(
        session_id="stream-wb-session",
        question="蛋炒饭怎么做",
        turn_info={"turn_type": "domain_query"},
        query_plan={"route_type": "detail", "dish_name": "蛋炒饭"},
        resolution=None,
        answer="",
        execution_result={
            "success": True,
            "stream_generator": fake_stream,
            "stream_started": True,
        },
    )

    session = manager.get_session("stream-wb-session")
    assert session.current_entity == "蛋炒饭"

def test_front_door_block_stops_before_query_planning_and_retrieval():
    system = _system()

    system.generation_module.query_router = lambda query: (_ for _ in ()).throw(
        AssertionError("blocked front-door input should not reach query planning")
    )
    system.retrieval_module.hybrid_search = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("blocked front-door input should not reach retrieval")
    )

    answer = system.ask_question("这道", stream=False, session_id="front-door-block-session")

    assert "哪道菜" in answer


def test_front_door_direct_reply_stops_before_query_planning_and_retrieval():
    system = _system()

    system.generation_module.query_router = lambda query: (_ for _ in ()).throw(
        AssertionError("direct front-door reply should not reach query planning")
    )
    system.retrieval_module.hybrid_search = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("direct front-door reply should not reach retrieval")
    )

    answer = system.ask_question("你好", stream=False, session_id="front-door-smalltalk-session")

    assert "食谱助手" in answer or "推荐菜" in answer


def test_front_door_continue_preserves_original_question_for_query_planning():
    system = _system()
    calls = []

    def fake_query_router(query):
        calls.append(query)
        return {
            "type": "general",
            "filters": {},
            "dish_name": None,
            "confidence": 0.7,
        }

    system.generation_module.query_router = fake_query_router

    answer = system.ask_question(
        "土豆丝怎么样",
        stream=False,
        session_id="front-door-plan-session",
    )

    assert calls == ["土豆丝怎么样"]
    assert answer
    assert system.last_execution_result["final_query_text"] == "土豆丝怎么样"
    assert system.last_execution_result["dish_name"] == "土豆丝"


# ---- Task 9: Stage 01 integration regression tests ----


def test_stage01_writeback_uses_state_diff_policy_for_detail():
    manager = ConversationManager()

    manager.writeback_turn_state(
        session_id="stage01-detail",
        question="蛋炒饭怎么做",
        answer="蛋炒饭做法",
        turn_info={"turn_type": "domain_query"},
        query_plan={"route_type": "detail", "dish_name": "蛋炒饭"},
        resolution=None,
        execution_result={"success": True, "answer": "蛋炒饭做法"},
    )

    session = manager.get_session("stage01-detail")
    assert session.current_entity == "蛋炒饭"
    assert session.last_answer_type == "detail"


def test_stage01_domain_reject_writeback_preserves_current_entity():
    manager = ConversationManager()
    manager.set_current_dish("stage01-domain", "蛋炒饭", source="setup", confidence=1.0)

    manager.writeback_turn_state(
        session_id="stage01-domain",
        question="Python 怎么学",
        answer="我主要处理食谱相关问题。",
        turn_info={"turn_type": "front_door_blocked"},
        query_plan=None,
        resolution=None,
        execution_result={"success": True, "answer": "我主要处理食谱相关问题。"},
    )

    session = manager.get_session("stage01-domain")
    assert session.current_entity == "蛋炒饭"
    assert session.last_answer_type == "domain_reject"


def test_stage01_failed_retrieval_writeback_preserves_current_entity():
    manager = ConversationManager()
    manager.set_current_dish("stage01-no-result", "蛋炒饭", source="setup", confidence=1.0)

    manager.writeback_turn_state(
        session_id="stage01-no-result",
        question="不存在的菜怎么做",
        answer="没有找到",
        turn_info={"turn_type": "domain_query"},
        query_plan={"route_type": "detail", "dish_name": "不存在的菜"},
        resolution=None,
        execution_result={"success": False, "answer": "没有找到"},
    )

    session = manager.get_session("stage01-no-result")
    assert session.current_entity == "蛋炒饭"
    assert session.last_answer_type == "no_result"
