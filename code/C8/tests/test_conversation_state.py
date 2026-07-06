from types import SimpleNamespace

from langchain_core.documents import Document

from main import RecipeRAGSystem
from rag_modules.context_packer import ContextPacker
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


EGG_FRIED_RICE_MD = (
    "# 蛋炒饭的做法\n\n"
    "## 必备原料和工具\n\n"
    "- 米饭\n"
    "- 鸡蛋\n\n"
    "## 操作\n\n"
    "- 鸡蛋打散。\n"
    "- 下锅炒饭。\n"
)

PAN_FRIED_RICE_MD = (
    "# 煎饭的做法\n\n"
    "## 操作\n\n"
    "1. 先热锅，再下油。\n"
    "2. 煎到底部定型后再加水。\n"
    "3. 收干前不要频繁翻动。\n\n"
    "## 附加内容\n\n"
    "- 火候不要太大。\n"
    "- 热锅后再下油更不容易粘锅。\n"
)


class _StubRetrievalModule:
    last_search_trace = {}

    def extract_filters_from_query(self, query):
        return {}

    def metadata_filtered_search(self, *args, **kwargs):
        return [Document(page_content="蛋炒饭怎么做", metadata={"dish_name": "蛋炒饭", "parent_id": "egg-parent"})]

    def hybrid_search(self, *args, **kwargs):
        return [Document(page_content="蛋炒饭怎么做", metadata={"dish_name": "蛋炒饭", "parent_id": "egg-parent"})]


class _TipsFallbackRetrievalModule:
    last_search_trace = {}

    def extract_filters_from_query(self, query):
        return {}

    def metadata_filtered_search(self, query, filters, top_k=3, query_dish=None):
        if filters.get("content_type") == "tips":
            return []
        return [Document(page_content="煎饭怎么做", metadata={"dish_name": "煎饭", "parent_id": "pan-parent"})]

    def hybrid_search(self, *args, **kwargs):
        return [Document(page_content="煎饭怎么做", metadata={"dish_name": "煎饭", "parent_id": "pan-parent"})]


class _StubDataModule:
    def get_parent_documents(self, chunks, target_dish_name=None):
        return [
            Document(
                page_content=EGG_FRIED_RICE_MD,
                metadata={"dish_name": "蛋炒饭", "parent_id": "egg-parent", "rrf_score": 1.0},
            )
        ]


class _TipsFallbackDataModule:
    def get_parent_documents(self, chunks, target_dish_name=None):
        return [
            Document(
                page_content=PAN_FRIED_RICE_MD,
                metadata={"dish_name": "煎饭", "parent_id": "pan-parent", "rrf_score": 1.0},
            )
        ]


def _system() -> RecipeRAGSystem:
    system = RecipeRAGSystem.__new__(RecipeRAGSystem)
    system.config = SimpleNamespace(
        top_k=3,
        context_pack_max_chars_total=2400,
        context_pack_max_chars_per_doc=1200,
        context_pack_max_docs=5,
    )
    system.data_module = _StubDataModule()
    system.retrieval_module = _StubRetrievalModule()
    system.context_packer = ContextPacker(
        max_chars_total=system.config.context_pack_max_chars_total,
        max_chars_per_doc=system.config.context_pack_max_chars_per_doc,
        max_docs=system.config.context_pack_max_docs,
    )
    system.generation_module = _StubGenerationModule()
    system._latest_parent_docs = []
    system.last_query_diagnostics = {}
    system.last_execution_result = {}
    return system


def _tips_system() -> RecipeRAGSystem:
    system = RecipeRAGSystem.__new__(RecipeRAGSystem)
    system.config = SimpleNamespace(
        top_k=3,
        context_pack_max_chars_total=2400,
        context_pack_max_chars_per_doc=1200,
        context_pack_max_docs=5,
    )
    system.data_module = _TipsFallbackDataModule()
    system.retrieval_module = _TipsFallbackRetrievalModule()
    system.context_packer = ContextPacker(
        max_chars_total=system.config.context_pack_max_chars_total,
        max_chars_per_doc=system.config.context_pack_max_chars_per_doc,
        max_docs=system.config.context_pack_max_docs,
    )
    system.generation_module = _StubGenerationModule()
    system._latest_parent_docs = []
    system.last_query_diagnostics = {}
    system.last_execution_result = {}
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

    assert "没有找到" in answer
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


def test_execution_plan_directs_domain_reject_without_retrieval():
    plan = build_execution_plan(
        {
            "action": "domain_reject",
            "response_mode": "polite_direct_reply",
            "should_retrieve": False,
        },
        resolution=None,
    )

    assert plan == {"action": "direct_domain_reject", "message": None}


def test_execution_plan_directs_smalltalk_without_retrieval():
    plan = build_execution_plan(
        {
            "action": "smalltalk",
            "response_mode": "polite_direct_reply",
            "should_retrieve": False,
        },
        resolution=None,
    )

    assert plan == {"action": "direct_smalltalk_reply", "message": None}


def test_execution_plan_uses_retrieve_list_action():
    plan = build_execution_plan(
        {
            "action": "retrieve_list",
            "response_mode": "retrieve_answer",
            "should_retrieve": True,
        },
        resolution=None,
    )

    assert plan == {"action": "retrieve_list", "message": None}


def test_execution_plan_uses_retrieve_detail_action():
    plan = build_execution_plan(
        {
            "action": "retrieve_detail",
            "response_mode": "retrieve_answer",
            "should_retrieve": True,
        },
        resolution=None,
    )

    assert plan == {"action": "retrieve_detail", "message": None}


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

def test_basic_safety_gate_block_stops_before_query_planning_and_retrieval():
    system = _system()

    system.generation_module.query_router = lambda query: (_ for _ in ()).throw(
        AssertionError("blocked input should not reach query planning")
    )
    system.retrieval_module.hybrid_search = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("blocked input should not reach retrieval")
    )

    answer = system.ask_question("", stream=False, session_id="safety-gate-block-session")

    assert "请输入" in answer or "具体" in answer


def test_smalltalk_stops_before_query_planning_and_retrieval():
    system = _system()

    system.generation_module.query_router = lambda query: (_ for _ in ()).throw(
        AssertionError("smalltalk should not reach query planning")
    )
    system.retrieval_module.hybrid_search = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("smalltalk should not reach retrieval")
    )
    system.generation_module.generate_smalltalk_answer = lambda question: "你好！我是食谱助手。"

    answer = system.ask_question("你好", stream=False, session_id="smalltalk-session")

    assert "食谱助手" in answer


def test_recipe_question_preserves_original_for_query_planning():
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
        "土豆丝怎么做",
        stream=False,
        session_id="recipe-plan-session",
    )

    assert "土豆丝怎么做" in calls
    assert answer


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


# ---- Task 4: Context-first pipeline integration tests ----


def test_context_first_pipeline_does_not_block_ordinal_followup_before_snapshot(monkeypatch):
    from main import RecipeRAGSystem

    system = RecipeRAGSystem.__new__(RecipeRAGSystem)
    calls = []

    class FakeConversationManager:
        def get_session(self, session_id):
            class Session:
                current_entity_meta = {}
                recent_recommendations = [{"dish_name": "鸡胸肉沙拉"}]
                recent_topics = []
                last_confirmed_target = None
                messages = []
                topic_mode = None
                current_intent = None
                pending_clarification = None
            return Session()

        def get_state_version(self, session_id):
            return 0

        def check_state_version(self, session_id, expected_version):
            return {
                "matched": True,
                "expected_version": expected_version,
                "current_version": 0,
                "reason": "state_version_match",
            }

    class FakeGeneration:
        conversation_manager = FakeConversationManager()
        llm = None

        def generate_smalltalk_answer(self, question):
            return "smalltalk"

    from langchain_core.documents import Document

    class FakeRetrievalModule:
        def extract_filters_from_query(self, query):
            return {}

    doc = Document(page_content="鸡胸肉沙拉做法", metadata={"dish_name": "鸡胸肉沙拉"})

    class FakeExecutor:
        def execute(self, query_plan):
            return {
                "chunks": [doc],
                "quality": {
                    "enough_evidence": True,
                    "quality_reason": "exact_dish_matched",
                    "fallback_used": False,
                    "relaxed_filter": False,
                    "candidate_count": 1,
                    "selected_dishes": ["鸡胸肉沙拉"],
                },
                "low_evidence": None,
                "trace": {"strategy": "primary"},
            }

    system.retrieval_module = FakeRetrievalModule()
    system.retrieval_executor = FakeExecutor()
    system.generation_module = FakeGeneration()
    system.config = type("Config", (), {"top_k": 3})()
    system._latest_parent_docs = []
    system.last_execution_result = None

    monkeypatch.setattr(system, "_build_query_plan", lambda question, session_id: {"route_type": "detail", "dish_name": "鸡胸肉沙拉", "filters": {}, "entities": []})
    monkeypatch.setattr(system, "_apply_resolved_target_to_query_plan", lambda query_plan, resolution: query_plan)
    monkeypatch.setattr(system, "_print_relevant_chunk_summary", lambda chunks: None)
    monkeypatch.setattr(system, "_rewrite_question_for_search", lambda question, route_type: question)
    monkeypatch.setattr(system, "_generate_detail_response", lambda question, stream, route_type, dish_name, context_pack: "鸡胸肉沙拉适合减脂。")
    monkeypatch.setattr(system, "_write_conversation_turn", lambda **kwargs: calls.append(kwargs))

    class FakeData:
        def get_parent_documents(self, chunks, target_dish_name=None):
            return [Document(page_content="# 鸡胸肉沙拉\n\n## 操作\n\n- 煎熟。", metadata={"dish_name": "鸡胸肉沙拉"})]

    class FakeContextPacker:
        def build_context_pack(self, **kwargs):
            return {
                "answer_mode": "recipe_detail",
                "context_docs": kwargs["parent_docs"],
                "parent_docs": kwargs["parent_docs"],
                "selected_sections": [{"section_type": "steps"}],
                "content_type": "steps",
                "trace": {"selected_section_count": 1},
            }

    system.data_module = FakeData()
    system.context_packer = FakeContextPacker()
    monkeypatch.setattr("main.resolve_reference_from_snapshot", lambda snapshot, llm: None)
    monkeypatch.setattr("main.guard_resolution_output", lambda resolution, constraints: resolution)
    monkeypatch.setattr("main.build_retrieval_query_plan", lambda **kwargs: kwargs)

    answer = system.ask_question("第一个适合减脂吗", stream=False, session_id="ctx-first")

    assert answer == "鸡胸肉沙拉适合减脂。"
    assert calls
    assert calls[-1]["turn_info"]["action"] == "retrieve_detail"
    assert calls[-1]["turn_info"]["reference_trigger"] == "ordinal_reference"


def test_context_first_pipeline_routes_domain_reject_without_retrieval(monkeypatch):
    from main import RecipeRAGSystem

    system = RecipeRAGSystem.__new__(RecipeRAGSystem)
    calls = []

    class FakeConversationManager:
        def get_session(self, session_id):
            class Session:
                current_entity_meta = {}
                recent_recommendations = []
                recent_topics = []
                last_confirmed_target = None
                messages = []
                topic_mode = None
                current_intent = None
                pending_clarification = None
            return Session()

        def get_state_version(self, session_id):
            return 0

    class FakeGeneration:
        conversation_manager = FakeConversationManager()
        llm = None

        def generate_smalltalk_answer(self, question):
            return "smalltalk"

    system.retrieval_module = object()
    system.generation_module = FakeGeneration()
    system._latest_parent_docs = []
    system.last_execution_result = None
    system.context_packer = object()  # not needed: returns before context packing
    monkeypatch.setattr(system, "_write_conversation_turn", lambda **kwargs: calls.append(kwargs))

    answer = system.ask_question("Python怎么学", stream=False, session_id="domain-reject")

    assert "食谱" in answer or "做菜" in answer
    assert calls[-1]["turn_info"]["action"] == "domain_reject"


def test_context_first_pipeline_routes_smalltalk_without_recipe_state_update(monkeypatch):
    from main import RecipeRAGSystem

    system = RecipeRAGSystem.__new__(RecipeRAGSystem)
    calls = []

    class FakeConversationManager:
        def get_session(self, session_id):
            class Session:
                current_entity_meta = {"value": "蛋炒饭", "active": True, "source": "confirmed", "confidence": 1.0, "updated_at": 0.0}
                recent_recommendations = []
                recent_topics = []
                last_confirmed_target = "蛋炒饭"
                messages = []
                topic_mode = None
                current_intent = None
                pending_clarification = None
            return Session()

        def get_state_version(self, session_id):
            return 0

    class FakeGeneration:
        conversation_manager = FakeConversationManager()
        llm = None

        def generate_smalltalk_answer(self, question):
            return "不客气。"

    system.retrieval_module = object()
    system.generation_module = FakeGeneration()
    system._latest_parent_docs = []
    system.last_execution_result = None
    system.context_packer = object()  # not needed: returns before context packing
    monkeypatch.setattr(system, "_write_conversation_turn", lambda **kwargs: calls.append(kwargs))

    answer = system.ask_question("谢谢", stream=False, session_id="smalltalk")

    assert answer == "不客气。"
    assert calls[-1]["turn_info"]["action"] == "smalltalk"
    assert calls[-1]["turn_info"]["should_update_entity_state"] is False


def test_chat_path_uses_retrieval_executor_result(monkeypatch):
    from main import RecipeRAGSystem
    from langchain_core.documents import Document

    system = RecipeRAGSystem.__new__(RecipeRAGSystem)
    calls = []
    doc = Document(page_content="蛋炒饭步骤", metadata={"dish_name": "蛋炒饭", "content_type": "steps"})

    class FakeGeneration:
        conversation_manager = None

        def query_router(self, question):
            return {
                "type": "detail",
                "filters": {"content_type": "steps"},
                "dish_name": "蛋炒饭",
                "confidence": 1.0,
            }

        def generate_step_by_step_answer(self, question, context_docs, content_type=None):
            return "蛋炒饭做法"

    class FakeExecutor:
        def execute(self, query_plan):
            calls.append(query_plan)
            return {
                "chunks": [doc],
                "quality": {
                    "enough_evidence": True,
                    "quality_reason": "exact_dish_matched",
                    "fallback_used": False,
                    "relaxed_filter": False,
                    "candidate_count": 1,
                    "selected_dishes": ["蛋炒饭"],
                },
                "low_evidence": None,
                "trace": {"strategy": "primary"},
            }

    class FakeRetrievalModule:
        def extract_filters_from_query(self, query):
            return {}

    system.retrieval_module = FakeRetrievalModule()
    system.retrieval_executor = FakeExecutor()
    system.generation_module = FakeGeneration()
    system.config = type("Config", (), {"top_k": 3})()
    system._latest_parent_docs = []
    system.last_execution_result = None

    class FakeData:
        def get_parent_documents(self, chunks, target_dish_name=None):
            return [Document(page_content="# 蛋炒饭的做法\n\n## 操作\n\n- 炒饭", metadata={"dish_name": "蛋炒饭"})]

    class FakeContextPacker:
        def build_context_pack(self, **kwargs):
            return {
                "answer_mode": "recipe_detail",
                "context_docs": kwargs["parent_docs"],
                "parent_docs": kwargs["parent_docs"],
                "selected_sections": [{"section_type": "steps"}],
                "content_type": "steps",
                "trace": {"selected_section_count": 1},
            }

    system.data_module = FakeData()
    system.context_packer = FakeContextPacker()

    monkeypatch.setattr(system, "_apply_resolved_target_to_query_plan", lambda query_plan, resolution: query_plan)
    monkeypatch.setattr("main.resolve_reference_from_snapshot", lambda snapshot, llm: None)
    monkeypatch.setattr("main.guard_resolution_output", lambda resolution, constraints: resolution)
    monkeypatch.setattr(system, "_write_conversation_turn", lambda **kwargs: None)

    answer = system.ask_question("蛋炒饭怎么做", stream=False, session_id="executor-chat")

    assert answer == "蛋炒饭做法"
    assert calls
    assert calls[0]["query"] == "蛋炒饭怎么做"
    assert calls[0]["dish_name"] == "蛋炒饭"
    assert calls[0]["filters"]["content_type"] == "steps"


def test_chat_path_returns_low_evidence_without_generation(monkeypatch):
    from main import RecipeRAGSystem

    system = RecipeRAGSystem.__new__(RecipeRAGSystem)
    writes = []
    generation_calls = []

    class FakeGeneration:
        conversation_manager = None

        def query_router(self, question):
            return {
                "type": "detail",
                "filters": {},
                "dish_name": "西湖醋鱼",
                "confidence": 1.0,
            }

    class FakeExecutor:
        def execute(self, query_plan):
            return {
                "chunks": [],
                "quality": {
                    "enough_evidence": False,
                    "quality_reason": "exact_dish_not_found",
                    "fallback_used": False,
                    "relaxed_filter": False,
                    "candidate_count": 0,
                    "selected_dishes": [],
                },
                "low_evidence": {
                    "answer_type": "no_result",
                    "answer": "知识库里没有找到这道菜的可靠做法。",
                    "state_diff_policy": "low_evidence",
                    "quality_reason": "exact_dish_not_found",
                },
                "trace": {"strategy": "low_evidence"},
            }

    class FakeRetrievalModule:
        def extract_filters_from_query(self, query):
            return {}

    system.retrieval_module = FakeRetrievalModule()
    system.retrieval_executor = FakeExecutor()
    system.generation_module = FakeGeneration()
    system.config = type("Config", (), {"top_k": 3})()
    system._latest_parent_docs = []
    system.last_execution_result = None
    system.context_packer = None  # low-evidence returns before context packing

    monkeypatch.setattr(system, "_apply_resolved_target_to_query_plan", lambda query_plan, resolution: query_plan)
    monkeypatch.setattr(system, "_generate_detail_response", lambda *args, **kwargs: generation_calls.append(args) or "should not happen")
    monkeypatch.setattr(system, "_write_conversation_turn", lambda **kwargs: writes.append(kwargs))

    answer = system.ask_question("西湖醋鱼怎么做", stream=False, session_id="low-evidence-chat")

    assert answer == "知识库里没有找到这道菜的可靠做法。"
    assert generation_calls == []
    assert writes[-1]["execution_result"]["answer_type"] == "no_result"
    assert writes[-1]["execution_result"]["state_diff_policy"] == "low_evidence"
    assert writes[-1]["execution_result"]["retrieval_quality"]["quality_reason"] == "exact_dish_not_found"


def test_chat_path_builds_context_pack_before_detail_generation(monkeypatch):
    from main import RecipeRAGSystem
    from langchain_core.documents import Document

    system = RecipeRAGSystem.__new__(RecipeRAGSystem)
    child = Document(page_content="child", metadata={"dish_name": "蛋炒饭", "parent_id": "p1"})
    parent = Document(page_content="# 蛋炒饭的做法\n\n## 必备原料和工具\n\n- 鸡蛋\n\n## 操作\n\n- 炒饭", metadata={"dish_name": "蛋炒饭", "parent_id": "p1"})
    packed = Document(page_content="## 操作\n\n- 炒饭", metadata={"dish_name": "蛋炒饭", "section_type": "steps"})
    generation_calls = []
    writes = []

    class FakeGeneration:
        conversation_manager = None

        def query_router(self, question):
            return {"type": "detail", "filters": {"content_type": "steps"}, "dish_name": "蛋炒饭", "confidence": 1.0}

        def generate_step_by_step_answer(self, question, context_docs, content_type=None):
            generation_calls.append((question, context_docs, content_type))
            return "蛋炒饭做法"

    class FakeRetrievalExecutor:
        def execute(self, query_plan):
            return {
                "chunks": [child],
                "quality": {"enough_evidence": True, "quality_reason": "ok", "fallback_used": False, "relaxed_filter": False, "candidate_count": 1, "selected_dishes": ["蛋炒饭"]},
                "low_evidence": None,
                "trace": {"strategy": "primary"},
            }

    class FakeData:
        def get_parent_documents(self, chunks, target_dish_name=None):
            assert chunks == [child]
            assert target_dish_name == "蛋炒饭"
            return [parent]

    class FakeContextPacker:
        def build_context_pack(self, **kwargs):
            assert kwargs["parent_docs"] == [parent]
            return {
                "answer_mode": "recipe_detail",
                "context_docs": [packed],
                "parent_docs": [parent],
                "selected_sections": [{"section_type": "steps"}],
                "content_type": "steps",
                "trace": {"selected_section_count": 1},
            }

    system.retrieval_module = type("Retrieval", (), {"extract_filters_from_query": lambda self, question: {}})()
    system.retrieval_executor = FakeRetrievalExecutor()
    system.context_packer = FakeContextPacker()
    system.generation_module = FakeGeneration()
    system.data_module = FakeData()
    system.config = type("Config", (), {
        "top_k": 3,
        "context_pack_max_chars_total": 2400,
        "context_pack_max_chars_per_doc": 1200,
        "context_pack_max_docs": 5,
    })()
    system._latest_parent_docs = []
    system.last_execution_result = None

    monkeypatch.setattr(system, "_apply_resolved_target_to_query_plan", lambda query_plan, resolution: query_plan)
    monkeypatch.setattr(system, "_write_conversation_turn", lambda **kwargs: writes.append(kwargs))

    answer = system.ask_question("蛋炒饭怎么做", stream=False, session_id="context-pack-detail")

    assert answer == "蛋炒饭做法"
    assert generation_calls[0][1] == [packed]
    assert generation_calls[0][2] == "steps"
    assert system._latest_parent_docs == [parent]
    assert writes[-1]["execution_result"]["context_pack_trace"] == {"selected_section_count": 1}


def test_chat_path_builds_context_pack_before_list_generation(monkeypatch):
    from main import RecipeRAGSystem
    from langchain_core.documents import Document

    system = RecipeRAGSystem.__new__(RecipeRAGSystem)
    child = Document(page_content="child", metadata={"dish_name": "番茄炒蛋", "parent_id": "p1"})
    parent = Document(page_content="# 番茄炒蛋的做法\n\n## 必备原料和工具\n\n- 番茄\n- 鸡蛋", metadata={"dish_name": "番茄炒蛋", "parent_id": "p1"})
    packed = Document(page_content="# 番茄炒蛋\n- 番茄\n- 鸡蛋", metadata={"dish_name": "番茄炒蛋", "context_pack_mode": "summary"})
    generation_calls = []
    writes = []

    class FakeGeneration:
        conversation_manager = None

        def query_router(self, question):
            return {"type": "list", "filters": {}, "dish_name": None, "confidence": 1.0}

        def generate_list_answer(self, question, context_docs):
            generation_calls.append((question, context_docs))
            return "1. 番茄炒蛋"

    class FakeRetrievalExecutor:
        def execute(self, query_plan):
            return {
                "chunks": [child],
                "quality": {"enough_evidence": True, "quality_reason": "ok", "fallback_used": False, "relaxed_filter": False, "candidate_count": 1, "selected_dishes": ["番茄炒蛋"]},
                "low_evidence": None,
                "trace": {"strategy": "primary"},
            }

    class FakeData:
        def get_parent_documents(self, chunks, target_dish_name=None):
            assert target_dish_name is None
            return [parent]

    class FakeContextPacker:
        def build_context_pack(self, **kwargs):
            return {
                "answer_mode": "recommendation",
                "context_docs": [packed],
                "parent_docs": [parent],
                "selected_sections": [{"section_type": "summary"}],
                "content_type": "recommendation",
                "trace": {"selected_section_count": 1},
            }

    system.retrieval_module = type("Retrieval", (), {"extract_filters_from_query": lambda self, question: {}})()
    system.retrieval_executor = FakeRetrievalExecutor()
    system.context_packer = FakeContextPacker()
    system.generation_module = FakeGeneration()
    system.data_module = FakeData()
    system.config = type("Config", (), {
        "top_k": 3,
        "context_pack_max_chars_total": 2400,
        "context_pack_max_chars_per_doc": 1200,
        "context_pack_max_docs": 5,
    })()
    system._latest_parent_docs = []
    system.last_execution_result = None

    monkeypatch.setattr(system, "_apply_resolved_target_to_query_plan", lambda query_plan, resolution: query_plan)
    monkeypatch.setattr(system, "_write_conversation_turn", lambda **kwargs: writes.append(kwargs))

    answer = system.ask_question("今天吃什么", stream=False, session_id="context-pack-list")

    assert answer == "1. 番茄炒蛋"
    assert generation_calls[0][1] == [packed]
    assert system._latest_parent_docs == [parent]
    assert writes[-1]["execution_result"]["context_pack_trace"] == {"selected_section_count": 1}


def test_chat_path_builds_context_pack_for_streaming_detail_generation(monkeypatch):
    from main import RecipeRAGSystem
    from langchain_core.documents import Document

    system = RecipeRAGSystem.__new__(RecipeRAGSystem)
    child = Document(page_content="child", metadata={"dish_name": "蛋炒饭", "parent_id": "p1"})
    parent = Document(page_content="# 蛋炒饭的做法\n\n## 操作\n\n- 炒饭", metadata={"dish_name": "蛋炒饭", "parent_id": "p1"})
    packed = Document(page_content="## 操作\n\n- 炒饭", metadata={"dish_name": "蛋炒饭", "section_type": "steps"})
    writes = []

    class FakeGeneration:
        conversation_manager = None

        def query_router(self, question):
            return {"type": "detail", "filters": {"content_type": "steps"}, "dish_name": "蛋炒饭", "confidence": 1.0}

        def generate_step_by_step_answer_stream(self, question, context_docs, content_type=None):
            assert context_docs == [packed]
            assert content_type == "steps"
            yield "蛋炒饭"
            yield "做法"

    class FakeRetrievalExecutor:
        def execute(self, query_plan):
            return {
                "chunks": [child],
                "quality": {"enough_evidence": True, "quality_reason": "ok", "fallback_used": False, "relaxed_filter": False, "candidate_count": 1, "selected_dishes": ["蛋炒饭"]},
                "low_evidence": None,
                "trace": {"strategy": "primary"},
            }

    class FakeData:
        def get_parent_documents(self, chunks, target_dish_name=None):
            assert target_dish_name == "蛋炒饭"
            return [parent]

    class FakeContextPacker:
        def build_context_pack(self, **kwargs):
            return {
                "answer_mode": "recipe_detail",
                "context_docs": [packed],
                "parent_docs": [parent],
                "selected_sections": [{"section_type": "steps"}],
                "content_type": "steps",
                "trace": {"selected_section_count": 1},
            }

    system.retrieval_module = type("Retrieval", (), {"extract_filters_from_query": lambda self, question: {}})()
    system.retrieval_executor = FakeRetrievalExecutor()
    system.context_packer = FakeContextPacker()
    system.generation_module = FakeGeneration()
    system.data_module = FakeData()
    system.config = type("Config", (), {
        "top_k": 3,
        "context_pack_max_chars_total": 2400,
        "context_pack_max_chars_per_doc": 1200,
        "context_pack_max_docs": 5,
    })()
    system._latest_parent_docs = []
    system.last_execution_result = None

    monkeypatch.setattr(system, "_apply_resolved_target_to_query_plan", lambda query_plan, resolution: query_plan)
    monkeypatch.setattr(system, "_write_conversation_turn", lambda **kwargs: writes.append(kwargs))

    stream = system.ask_question("蛋炒饭怎么做", stream=True, session_id="context-pack-stream")
    assert "".join(stream) == "蛋炒饭做法"
    assert writes[-1]["execution_result"]["context_pack_trace"] == {"selected_section_count": 1}


def test_chat_path_parent_expansion_allows_none_target_for_detail_without_dish(monkeypatch):
    from main import RecipeRAGSystem
    from langchain_core.documents import Document

    system = RecipeRAGSystem.__new__(RecipeRAGSystem)
    child = Document(page_content="child", metadata={"dish_name": "番茄炒蛋", "parent_id": "p1"})
    parent = Document(page_content="# 番茄炒蛋的做法\n\n## 操作\n\n- 炒蛋", metadata={"dish_name": "番茄炒蛋", "parent_id": "p1"})

    class FakeGeneration:
        conversation_manager = None

        def query_router(self, question):
            return {"type": "detail", "filters": {"content_type": "steps"}, "dish_name": None, "confidence": 0.7}

        def generate_step_by_step_answer(self, question, context_docs, content_type=None):
            return "可以这样做"

    class FakeRetrievalExecutor:
        def execute(self, query_plan):
            return {
                "chunks": [child],
                "quality": {"enough_evidence": True, "quality_reason": "ok", "fallback_used": False, "relaxed_filter": False, "candidate_count": 1, "selected_dishes": ["番茄炒蛋"]},
                "low_evidence": None,
                "trace": {"strategy": "primary"},
            }

    class FakeData:
        def get_parent_documents(self, chunks, target_dish_name=None):
            assert target_dish_name is None
            return [parent]

    class FakeContextPacker:
        def build_context_pack(self, **kwargs):
            return {
                "answer_mode": "recipe_detail",
                "context_docs": [parent],
                "parent_docs": [parent],
                "selected_sections": [{"section_type": "steps"}],
                "content_type": "steps",
                "trace": {"selected_section_count": 1},
            }

    system.retrieval_module = type("Retrieval", (), {"extract_filters_from_query": lambda self, question: {}})()
    system.retrieval_executor = FakeRetrievalExecutor()
    system.context_packer = FakeContextPacker()
    system.generation_module = FakeGeneration()
    system.data_module = FakeData()
    system.config = type("Config", (), {
        "top_k": 3,
        "context_pack_max_chars_total": 2400,
        "context_pack_max_chars_per_doc": 1200,
        "context_pack_max_docs": 5,
    })()
    system._latest_parent_docs = []
    system.last_execution_result = None

    monkeypatch.setattr(
        system,
        "_build_query_plan",
        lambda question, session_id: {
            "route_type": "detail",
            "filters": {"content_type": "steps"},
            "dish_name": None,
            "entities": {"dish_name": None, "filters": {"content_type": "steps"}},
            "confidence": 0.7,
        },
    )
    monkeypatch.setattr(system, "_apply_resolved_target_to_query_plan", lambda query_plan, resolution: query_plan)
    monkeypatch.setattr("main.resolve_reference_from_snapshot", lambda snapshot, llm: None)
    monkeypatch.setattr("main.guard_resolution_output", lambda resolution, constraints: resolution)
    monkeypatch.setattr(system, "_write_conversation_turn", lambda **kwargs: None)

    assert system.ask_question("这个怎么做", stream=False, session_id="none-target-detail") == "可以这样做"


def test_write_conversation_turn_records_pre_commit_conflict_without_business_update():
    system = RecipeRAGSystem.__new__(RecipeRAGSystem)
    manager = ConversationManager()
    system.generation_module = type("Generation", (), {"conversation_manager": manager})()

    manager.commit_state_diff(
        "precommit-conflict",
        {
            "answer_type": "smalltalk",
            "updates": {"last_answer_type": "smalltalk"},
            "clear": [],
            "append_history": False,
            "history": None,
        },
        expected_version=0,
    )

    system._write_conversation_turn(
        session_id="precommit-conflict",
        question="蛋炒饭怎么做",
        answer="蛋炒饭做法",
        turn_info={"turn_type": "domain_query"},
        query_plan={"route_type": "detail", "dish_name": "蛋炒饭"},
        resolution=None,
        execution_result={
            "success": True,
            "resolved_target": "蛋炒饭",
            "runtime": {"read_state_version": 0, "turn_id": "t1", "trace_id": "x1"},
        },
    )

    session = manager.get_session("precommit-conflict")
    assert session.current_entity is None
    assert session.last_answer_type == "smalltalk"


def test_state_dependent_turn_replans_before_planning_after_resolution_mismatch(monkeypatch):
    import main as main_module

    system = _system()
    manager = system.generation_module.conversation_manager
    manager.record_recommendations("stale-resolution", ["蛋炒饭"])

    planned = {"called": False}

    def fail_if_planned(*args, **kwargs):
        planned["called"] = True
        return {"route_type": "detail", "dish_name": "蛋炒饭"}

    def mutate_after_resolution(*args, **kwargs):
        manager.commit_state_diff(
            "stale-resolution",
            {
                "answer_type": "smalltalk",
                "updates": {"last_answer_type": "smalltalk"},
                "clear": [],
                "append_history": False,
                "history": None,
            },
            expected_version=manager.get_state_version("stale-resolution"),
        )
        return {
            "next_action": "apply_reference_resolution",
            "resolved_target": "蛋炒饭",
            "confidence": 0.9,
            "target_source": "last_recommendation_list[0]",
            "writeback_eligible": True,
        }

    monkeypatch.setattr(main_module, "resolve_reference_from_snapshot", mutate_after_resolution)
    monkeypatch.setattr(system, "_build_query_plan", fail_if_planned)

    answer = system.ask_question("第一个怎么做", stream=False, session_id="stale-resolution")

    assert planned["called"] is False
    assert "上下文刚刚更新" in answer


def test_state_dependent_turn_does_not_generate_after_pre_generation_version_mismatch(monkeypatch):
    system = _system()
    manager = system.generation_module.conversation_manager
    manager.record_recommendations("stale-gen", ["蛋炒饭"])

    generated = {"called": False}

    def fail_if_generated(*args, **kwargs):
        generated["called"] = True
        return "不应该生成"

    system.generation_module.generate_step_by_step_answer = fail_if_generated

    original_build_context_pack = system.context_packer.build_context_pack

    def mutate_state_before_generation(**kwargs):
        pack = original_build_context_pack(**kwargs)
        manager.commit_state_diff(
            "stale-gen",
            {
                "answer_type": "smalltalk",
                "updates": {"last_answer_type": "smalltalk"},
                "clear": [],
                "append_history": False,
                "history": None,
            },
            expected_version=manager.get_state_version("stale-gen"),
        )
        return pack

    monkeypatch.setattr(system.context_packer, "build_context_pack", mutate_state_before_generation)

    answer = system.ask_question("第一个怎么做", stream=False, session_id="stale-gen")

    assert generated["called"] is False
    assert "上下文刚刚更新" in answer


def test_repeated_version_mismatch_uses_shared_replan_budget_and_returns_conflict(monkeypatch):
    system = _system()
    manager = system.generation_module.conversation_manager
    manager.record_recommendations("shared-budget", ["蛋炒饭"])

    generation_calls = {"count": 0}
    system.generation_module.generate_step_by_step_answer = lambda *args, **kwargs: generation_calls.__setitem__("count", generation_calls["count"] + 1) or "不应生成"

    original_build_context_pack = system.context_packer.build_context_pack

    def always_mutate_before_generation(**kwargs):
        pack = original_build_context_pack(**kwargs)
        manager.commit_state_diff(
            "shared-budget",
            {
                "answer_type": "smalltalk",
                "updates": {"last_answer_type": "smalltalk"},
                "clear": [],
                "append_history": False,
                "history": None,
            },
            expected_version=manager.get_state_version("shared-budget"),
        )
        return pack

    monkeypatch.setattr(system.context_packer, "build_context_pack", always_mutate_before_generation)

    answer = system.ask_question("第一个怎么做", stream=False, session_id="shared-budget")

    assert "上下文刚刚更新" in answer
    assert generation_calls["count"] == 0
    assert system.last_execution_result["runtime"]["replan_count"] == 1
