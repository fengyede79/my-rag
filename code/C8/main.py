"""
RAG系统主程序
"""

import os
import sys
import logging
import io
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

# 设置 stdout 为 UTF-8 编码（避免 Windows 控制台中文输出异常）
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# 添加模块路径
sys.path.append(str(Path(__file__).parent))

from config import RAGConfig, load_config
from rag_modules import (
    DataPreparationModule,
    IndexConstructionModule,
    RetrievalOptimizationModule,
    GenerationIntegrationModule
)
from rag_modules.front_door_guardrail import basic_safety_gate
from rag_modules.turn_understanding import understand_turn
from rag_modules.conversation_state_builder import build_conversation_snapshot
from rag_modules.reference_resolution import (
    resolve_reference_from_snapshot,
    guard_resolution_output,
    rewrite_query_for_execution,
)
from rag_modules.execution_planner import build_execution_plan

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RecipeRAGSystem:
    """食谱RAG系统主类"""

    def __init__(self, config: RAGConfig = None):
        """
        初始化RAG系统

        Args:
            config: RAG系统配置，默认使用DEFAULT_CONFIG
        """
        self.config = config or load_config()
        self.data_module = None
        self.index_module = None
        self.retrieval_module = None
        self.generation_module = None
        self._latest_parent_docs = []
        self.last_query_diagnostics = {}
        self.last_execution_result = {}

        # 检查数据路径
        if not Path(self.config.data_path).exists():
            raise FileNotFoundError(f"数据路径不存在: {self.config.data_path}")

        # 检查API密钥
        if not os.getenv("DASHSCOPE_API_KEY"):
            raise ValueError("请设置 DASHSCOPE_API_KEY 环境变量")

    def reset_session(self, session_id: str):
        """重置指定会话的对话状态和推荐缓存"""
        if self.generation_module:
            self.generation_module.reset_session_state(session_id)

    def print_startup_check(self):
        """打印最基本的启动自检信息"""
        print("\n启动自检:")
        print(f"   Python解释器: {sys.executable}")
        print(f"   数据目录: {self.config.data_path}")
        print(f"   索引目录: {self.config.index_save_path}")
        print(f"   索引目录是否存在: {'是' if self.config.index_exists() else '否'}")
        print(
            "   索引缺失时处理方式: "
            + ("自动重建" if self.config.rebuild_index_if_missing else "直接报错")
        )
        print(f"   向量模型: {self.config.embedding_model}")
        print(f"   对话模型: {self.config.llm_model}")
        print(f"   检索条数 top_k: {self.config.top_k}")
        print(f"   生成温度: {self.config.temperature}")
    
    def initialize_system(self):
        """初始化所有模块"""
        print("正在初始化RAG系统...")

        # 1. 初始化数据准备模块
        print("初始化数据准备模块...")
        self.data_module = DataPreparationModule(
            data_path=self.config.data_path,
            cache_path=self.config.index_save_path
        )

        # 2. 初始化索引构建模块
        print("初始化索引构建模块...")
        self.index_module = IndexConstructionModule(
            model_name=self.config.embedding_model,
            index_save_path=self.config.index_save_path
        )

        # 3. 初始化生成集成模块（启用多轮对话支持）
        print("初始化生成集成模块（多轮对话已启用）...")
        self.generation_module = GenerationIntegrationModule(
            model_name=self.config.llm_model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            enable_conversation=True  # 启用多轮对话支持
        )

        print("系统初始化完成！")

    def _load_documents_and_chunks(self):
        """加载文档并完成分块。"""
        print("加载食谱文档...")
        self.data_module.load_documents()
        print("进行文本分块...")
        chunks = self.data_module.chunk_documents()
        return chunks

    def _try_load_existing_knowledge_base(self):
        """尝试加载已有的索引和分块缓存。"""
        vectorstore = self.index_module.load_index()
        if vectorstore is None:
            return None, None

        print("成功加载已保存的向量索引！")
        print("尝试加载分块缓存...")
        if self.data_module.load_chunks():
            print("成功加载分块缓存！")
            return vectorstore, self.data_module.chunks

        print("分块缓存不存在，重新加载文档...")
        chunks = self._load_documents_and_chunks()
        print("保存分块缓存...")
        self.data_module.save_chunks()
        return vectorstore, chunks

    def _rebuild_knowledge_base(self):
        """从原始文档重建索引和分块缓存。"""
        if not self.config.rebuild_index_if_missing:
            raise FileNotFoundError(
                "未找到已保存的向量索引，且当前配置禁止自动重建。"
                f"请检查索引目录: {self.config.index_save_path}"
            )

        print("未找到已保存的索引，开始构建新索引...")
        chunks = self._load_documents_and_chunks()
        print("构建向量索引...")
        vectorstore = self.index_module.build_vector_index(chunks)
        print("保存向量索引...")
        self.index_module.save_index()
        print("保存分块缓存...")
        self.data_module.save_chunks()
        return vectorstore, chunks

    def _print_knowledge_base_stats(self):
        """打印知识库统计信息。"""
        stats = self.data_module.get_statistics()
        print("\n知识库统计:")
        print(f"   文档总数: {stats['total_documents']}")
        print(f"   文本块数: {stats['total_chunks']}")
        print(f"   菜品分类: {list(stats['categories'].keys())}")
        print(f"   难度分布: {stats['difficulties']}")

    def _write_conversation_turn(
        self,
        *,
        session_id: str,
        question: str,
        answer: str,
        turn_info: dict,
        query_plan: dict | None,
        resolution: dict | None,
        execution_result: dict,
    ):
        """委托给 ConversationManager.writeback_turn_state 记录轮次。"""
        conversation_manager = getattr(self.generation_module, "conversation_manager", None)
        if not conversation_manager:
            return
        conversation_manager.writeback_turn_state(
            session_id=session_id,
            question=question,
            turn_info=turn_info,
            query_plan=query_plan,
            resolution=resolution,
            answer=answer,
            execution_result=execution_result,
        )

    def _apply_resolved_target_to_query_plan(
        self,
        query_plan: Dict[str, Any],
        resolution: Dict[str, Any] | None,
    ) -> Dict[str, Any]:
        """Make guarded reference resolution override route-level dish extraction."""
        if not resolution:
            return query_plan
        if resolution.get("next_action") != "apply_reference_resolution":
            return query_plan
        resolved_target = resolution.get("resolved_target")
        if not resolved_target:
            return query_plan

        query_plan["dish_name"] = resolved_target
        query_plan.setdefault("filters", {})["dish_name"] = resolved_target
        query_plan.setdefault("entities", {})["dish_name"] = resolved_target
        return query_plan

    def _wrap_stream_with_writeback(
        self,
        *,
        answer_stream,
        session_id: str,
        question: str,
        turn_info: dict,
        query_plan: dict | None,
        resolution: dict | None,
        execution_result: dict,
    ):
        """Wrap stream generator to defer writeback until fully consumed."""
        collected = []
        for chunk in answer_stream:
            collected.append(chunk)
            yield chunk
        full_text = "".join(collected)
        execution_result["answer"] = full_text
        execution_result["success"] = True
        self.last_execution_result = execution_result
        self._write_conversation_turn(
            session_id=session_id,
            question=question,
            answer=full_text,
            turn_info=turn_info,
            query_plan=query_plan,
            resolution=resolution,
            execution_result=execution_result,
        )

    def _extract_retrieved_dishes(self, parent_docs: list) -> list[str]:
        """从检索文档中提取去重的菜品名列表。"""
        dishes: list[str] = []
        for doc in parent_docs or []:
            dish_name = (getattr(doc, "metadata", {}) or {}).get("dish_name")
            if dish_name and dish_name not in dishes:
                dishes.append(dish_name)
        return dishes

    def _build_execution_result(
        self,
        *,
        success: bool,
        answer: Any,
        rewritten_question: str,
        original_question: str,
        query_plan: Dict[str, Any] | None,
        resolution: Dict[str, Any] | None,
        parent_docs: list | None = None,
        recommended_dishes: list[str] | None = None,
    ) -> Dict[str, Any]:
        """构建结构化的执行结果，供回写和诊断使用。"""
        query_plan = query_plan or {}
        resolution = resolution or {}
        return {
            "success": success,
            "answer": answer,
            "final_query_text": rewritten_question,
            "query_plan_source": "rewritten" if rewritten_question != original_question else "original",
            "route_type": query_plan.get("route_type"),
            "filters": dict(query_plan.get("filters") or {}),
            "dish_name": query_plan.get("dish_name"),
            "resolved_target": resolution.get("resolved_target") or query_plan.get("dish_name"),
            "target_source": resolution.get("target_source"),
            "retrieved_dishes": self._extract_retrieved_dishes(parent_docs or []),
            "recommended_dishes": recommended_dishes or [],
        }

    def _extract_recommended_dishes(self, answer: str, parent_docs: list) -> list[str]:
        """从检索结果中提取推荐菜品列表。"""
        dish_names = []
        for doc in parent_docs:
            dish_name = (doc.metadata or {}).get("dish_name")
            if dish_name and dish_name not in dish_names:
                dish_names.append(dish_name)

        if dish_names:
            return dish_names[:5]

        fallback = []
        for line in answer.splitlines():
            cleaned = line.strip().lstrip("-").lstrip("1234567890.、 ").strip()
            if 1 < len(cleaned) <= 20 and cleaned not in fallback:
                fallback.append(cleaned)
        return fallback[:5]

    def _is_invalid_reference_dish_name(self, dish_name: str | None) -> bool:
        """检测伪菜名：序号引用短语或追问片段不应被当作菜品名。"""
        if not dish_name:
            return False
        normalized = dish_name.strip()
        if re.match(r"^(第?[一二三四五1-5]个?|[1-5]号?)$", normalized):
            return True
        invalid_fragments = ("有什么", "哪些", "怎么", "为何", "为什么", "技巧", "粘锅")
        return any(fragment in normalized for fragment in invalid_fragments)

    def _build_query_plan(self, question: str, session_id: str) -> Dict[str, Any]:
        """构建问答执行所需的查询计划。"""
        intent = self.generation_module.query_router(question)
        route_type = intent["type"]
        filters = intent.get("filters", {})
        dish_name = intent.get("dish_name")
        confidence = intent.get("confidence", 0)

        if not dish_name and route_type != "list":
            dish_name = self._infer_explicit_dish_topic(question)

        if self._is_invalid_reference_dish_name(dish_name):
            logger.info("丢弃引用短语型伪菜名: %s", dish_name)
            dish_name = None

        logger.info("查询计划初判: route_type=%s confidence=%.2f question=%s", route_type, confidence, question)
        print(f"查询类型: {route_type} (置信度: {confidence:.2f})")
        if filters:
            logger.info("查询计划过滤条件: %s", filters)
            print(f"提取过滤条件: {filters}")
        if dish_name:
            logger.info("查询计划菜品名: %s", dish_name)
            print(f"提取菜品名称: {dish_name}")

        entities = {"dish_name": dish_name, "filters": filters}

        return {
            "route_type": route_type,
            "filters": filters,
            "dish_name": dish_name,
            "entities": entities,
            "confidence": confidence,
        }

    def _should_inherit_current_entity(self, question: str) -> bool:
        """判断当前问题是否像是对上一道菜的追问。"""
        normalized = question.strip()
        if not normalized:
            return False

        reference_prefixes = [
            "它",
            "这个",
            "那个",
            "这道菜",
            "那道菜",
            "刚才那个",
            "之前那个",
            "前面说的",
            "再说一下",
            "再讲一下",
            "再说说",
            "再讲讲",
        ]
        if any(normalized.startswith(prefix) for prefix in reference_prefixes):
            return True

        followup_keywords = [
            "怎么做",
            "做法",
            "步骤",
            "食材",
            "材料",
            "原料",
            "配料",
            "技巧",
            "介绍",
            "需要什么",
        ]
        if len(normalized) <= 6 and any(keyword in normalized for keyword in followup_keywords):
            return True

        return False

    def _infer_explicit_dish_topic(self, question: str) -> str | None:
        """从通用问句中补提取显式菜品名，避免新话题被旧会话污染。"""
        normalized = question.strip()
        if not normalized:
            return None

        if self._should_inherit_current_entity(normalized):
            return None

        candidate = normalized
        suffix_patterns = [
            r"(怎么样|咋样|如何|好不好吃|好吃吗|值得做吗|值得试吗)[？?！!。,\s]*$",
            r"(介绍一下|说说|讲讲|聊聊|聊一聊)[？?！!。,\s]*$",
            r"(是什么)[？?！!。,\s]*$",
        ]
        for pattern in suffix_patterns:
            updated = re.sub(pattern, "", candidate)
            if updated != candidate:
                candidate = updated
                break

        candidate = candidate.strip("，。！？?!.、:： ")
        if len(candidate) < 2 or len(candidate) > 12:
            return None

        if not all("\u4e00" <= ch <= "\u9fff" for ch in candidate):
            return None

        return candidate

    def _rewrite_question_for_search(self, question: str, route_type: str) -> str:
        """根据问题类型决定是否重写查询。"""
        if route_type == "list":
            print(f"列表查询保持原样: {question}")
            return question

        print(f"使用新框架确定的查询文本: {question}")
        return question

    def _maybe_handle_guardrail_query(self, question: str):
        """对超出知识库边界的问题直接给出保守回答。"""
        reason = self.generation_module._classify_query_guardrail(question)
        if not reason:
            return None

        print(f"触发边界护栏: {reason}")
        answer = self.generation_module.build_guardrail_answer(question, reason)
        self._latest_parent_docs = []
        self.generation_module._record_generation_trace(
            f"guardrail_{reason}",
            context_doc_count=0,
            reason=reason,
        )
        return answer

    def _search_relevant_chunks(
        self,
        question: str,
        rewritten_query: str,
        filters: Dict[str, Any],
        dish_name: str,
    ):
        """执行检索并返回相关文档块。"""
        print("检索相关文档...")
        extra_filters = self.retrieval_module.extract_filters_from_query(question)
        combined_filters = {**filters, **extra_filters}

        if dish_name and len(dish_name) > 2:
            combined_filters["dish_name"] = dish_name
            logger.info("强制菜品过滤: %s", dish_name)
            print(f"强制菜品名过滤: {dish_name}")

        search_query = rewritten_query
        if dish_name and dish_name not in rewritten_query:
            search_query = f"{dish_name} {rewritten_query}"
            logger.info("增强检索查询: %s", search_query)
            print(f"增强检索查询: {search_query}")

        if combined_filters:
            logger.info("应用检索过滤: %s", combined_filters)
            print(f"应用过滤条件: {combined_filters}")
            relevant_chunks = self.retrieval_module.metadata_filtered_search(
                search_query,
                combined_filters,
                top_k=self.config.top_k,
                query_dish=dish_name,
            )
        else:
            relevant_chunks = self.retrieval_module.hybrid_search(
                search_query,
                top_k=self.config.top_k,
                query_dish=dish_name,
            )

        if (
            not relevant_chunks
            and combined_filters.get("content_type") == "tips"
            and dish_name
        ):
            fallback_filters = dict(combined_filters)
            fallback_filters.pop("content_type", None)
            print(f"技巧检索未命中，回退到同菜品全量内容: {dish_name}")
            if fallback_filters:
                print(f"回退过滤条件: {fallback_filters}")
                relevant_chunks = self.retrieval_module.metadata_filtered_search(
                    search_query,
                    fallback_filters,
                    top_k=self.config.top_k,
                    query_dish=dish_name,
                )
            else:
                relevant_chunks = self.retrieval_module.hybrid_search(
                    search_query,
                    top_k=self.config.top_k,
                    query_dish=dish_name,
                )

        return relevant_chunks

    def _print_relevant_chunk_summary(self, relevant_chunks):
        """打印检索到的文档块摘要。"""
        if not relevant_chunks:
            print("找到 0 个相关文档块")
            return

        chunk_info = []
        content_type_labels = {
            "ingredients": "食材",
            "steps": "步骤",
            "calculation": "计算",
            "tips": "技巧",
            "introduction": "介绍",
            "nutrition": "营养",
            "general": "综合",
        }
        for chunk in relevant_chunks:
            dish_name = chunk.metadata.get("dish_name", "未知菜品")
            content_type = chunk.metadata.get("content_type", "general")
            content_type_label = content_type_labels.get(content_type, content_type)
            content_preview = chunk.page_content[:100].strip()
            if content_preview.startswith("#"):
                title_end = content_preview.find("\n") if "\n" in content_preview else len(content_preview)
                section_title = content_preview[:title_end].replace("#", "").strip()
                chunk_info.append(f"{dish_name}[{content_type_label}]({section_title})")
            else:
                chunk_info.append(f"{dish_name}[{content_type_label}]")

        print(f"找到 {len(relevant_chunks)} 个相关文档块: {', '.join(chunk_info)}")

    def _generate_list_response(self, question: str, session_id: str, relevant_chunks):
        """生成列表类回答。"""
        print("生成菜品列表...")
        relevant_docs = self.data_module.get_parent_documents(relevant_chunks)
        doc_names = []
        for doc in relevant_docs:
            dish_name = doc.metadata.get("dish_name", "未知菜品")
            doc_names.append(dish_name)

        if doc_names:
            print(f"找到文档: {', '.join(doc_names)}")
        self._latest_parent_docs = list(relevant_docs)

        return self.generation_module.generate_list_answer(question, relevant_docs)

    def _generate_detail_response(
        self,
        question: str,
        stream: bool,
        session_id: str,
        route_type: str,
        filters: Dict[str, Any],
        entities: Dict[str, Any],
        dish_name: str,
        relevant_chunks,
    ):
        """生成详细或通用问答结果。"""
        print("获取完整文档...")
        relevant_docs = self.data_module.get_parent_documents(
            relevant_chunks,
            target_dish_name=dish_name,
        )

        doc_names = []
        for doc in relevant_docs:
            current_dish_name = doc.metadata.get("dish_name", "未知菜品")
            doc_names.append(current_dish_name)

        if doc_names:
            print(f"找到文档: {', '.join(doc_names)}")
        else:
            print(f"对应 {len(relevant_docs)} 个完整文档")
        self._latest_parent_docs = list(relevant_docs)

        print("生成详细回答...")
        content_type = filters.get("content_type")
        if route_type == "detail":
            if stream:
                return self.generation_module.generate_step_by_step_answer_stream(
                    question,
                    relevant_docs,
                    content_type=content_type,
                )
            return self.generation_module.generate_step_by_step_answer(
                question,
                relevant_docs,
                content_type=content_type,
            )

        if stream:
            return self.generation_module.generate_basic_answer_stream(
                question,
                relevant_docs,
                content_type=content_type,
            )
        return self.generation_module.generate_basic_answer(
            question,
            relevant_docs,
            content_type=content_type,
        )
    
    def build_knowledge_base(self):
        """构建知识库"""
        print("\n正在构建知识库...")

        vectorstore, chunks = self._try_load_existing_knowledge_base()
        if vectorstore is None:
            vectorstore, chunks = self._rebuild_knowledge_base()

        print("初始化检索优化...")
        self.retrieval_module = RetrievalOptimizationModule(vectorstore, chunks)
        self._print_knowledge_base_stats()
        print("知识库构建完成！")
    
    def ask_question(
        self,
        question: str,
        stream: bool = False,
        session_id: str = "default",
        return_diagnostics: bool = False,
        expectation: Dict[str, Any] = None,
    ):
        """
        回答用户问题

        Args:
            question: 用户问题
            stream: 是否使用流式输出
            session_id: 会话ID（用于多轮对话）

        Returns:
            生成的回答或生成器
        """
        if not all([self.retrieval_module, self.generation_module]):
            raise ValueError("请先构建知识库")

        print(f"\n用户问题: {question}")
        original_question = question
        self._latest_parent_docs = []
        safety = basic_safety_gate(question)
        logger.info(
            "[BasicSafetyGate] decision=%s reason=%s",
            safety["decision"],
            safety["reason"],
        )

        if safety["decision"] == "block":
            answer = safety["message"] or "请输入一个具体的食谱或做菜问题。"
            self.last_execution_result = {"success": True, "answer": answer}
            self._write_conversation_turn(
                session_id=session_id,
                question=question,
                answer=answer,
                turn_info={
                    "action": "invalid_input",
                    "answer_mode_hint": "safe_direct",
                    "turn_type": "basic_safety_blocked",
                    "response_mode": "polite_direct_reply",
                    "should_retrieve": False,
                    "should_update_topic_state": False,
                    "should_update_entity_state": False,
                    "should_run_reference_resolution": False,
                    "reference_trigger": "none",
                },
                query_plan=None,
                resolution=None,
                execution_result=self.last_execution_result,
            )
            return answer

        conversation_manager = getattr(self.generation_module, "conversation_manager", None)
        snapshot = None
        resolution = None
        if conversation_manager:
            snapshot = build_conversation_snapshot(
                conversation_manager.get_session(session_id),
                current_query=question,
            )
        else:
            snapshot = {
                "reference_state": {
                    "current_dish": {"value": None, "active": False},
                    "recent_recommendations": [],
                },
                "resolution_constraints": {"allowed_reference_targets": []},
                "state_health": {"has_pending_clarification": False},
            }

        turn_info = understand_turn(question, snapshot)

        if turn_info["action"] == "smalltalk":
            answer = self.generation_module.generate_smalltalk_answer(question)
            execution_result = {"success": True, "answer": answer}
            self.last_execution_result = execution_result
            self._write_conversation_turn(
                session_id=session_id,
                question=question,
                answer=answer,
                turn_info=turn_info,
                query_plan=None,
                resolution=None,
                execution_result=execution_result,
            )
            return answer

        if turn_info["action"] == "domain_reject":
            answer = "我主要处理食谱、做菜、食材和菜品推荐相关问题。"
            execution_result = {"success": True, "answer": answer}
            self.last_execution_result = execution_result
            self._write_conversation_turn(
                session_id=session_id,
                question=question,
                answer=answer,
                turn_info=turn_info,
                query_plan=None,
                resolution=None,
                execution_result=execution_result,
            )
            return answer

        if turn_info["should_run_reference_resolution"]:
            resolution = resolve_reference_from_snapshot(snapshot, getattr(self.generation_module, "llm", None))
            resolution = guard_resolution_output(
                resolution,
                snapshot["resolution_constraints"],
            )

        if resolution and resolution["next_action"] == "ask_clarification":
            answer = resolution["clarification_question"]
            execution_result = self._build_execution_result(
                success=True,
                answer=answer,
                rewritten_question=question,
                original_question=question,
                query_plan=None,
                resolution=resolution,
                parent_docs=[],
            )
            self.last_execution_result = execution_result
            self._write_conversation_turn(
                session_id=session_id,
                question=question,
                answer=answer,
                turn_info=turn_info,
                query_plan=None,
                resolution=resolution,
                execution_result=execution_result,
            )
            return answer

        # --- Execution Planning ---
        execution_plan = build_execution_plan(turn_info, resolution)
        base_query_plan = self._build_query_plan(question, session_id)
        rewritten_question = rewrite_query_for_execution(question, execution_plan, resolution, base_query_plan)
        query_plan = (
            self._build_query_plan(rewritten_question, session_id)
            if rewritten_question != question
            else base_query_plan
        )

        # --- Lock resolved target into query plan ---
        query_plan = self._apply_resolved_target_to_query_plan(query_plan, resolution)

        # --- Ensure query-plan dish enables state writeback ---
        if resolution and query_plan.get("dish_name") and not resolution.get("resolved_target"):
            resolution["writeback_eligible"] = True

        # --- Preference constraint propagation ---
        preference_constraints = (
            snapshot.get("resolution_constraints", {}).get("preference_constraints", {})
            if snapshot
            else {}
        )
        if query_plan["route_type"] == "list" and any(preference_constraints.values()):
            query_plan["preference_constraints"] = preference_constraints

        if execution_plan["action"] == "ask_clarification":
            answer = execution_plan["message"]
            execution_result = self._build_execution_result(
                success=True,
                answer=answer,
                rewritten_question=rewritten_question,
                original_question=question,
                query_plan=query_plan,
                resolution=resolution,
                parent_docs=[],
            )
            self.last_execution_result = execution_result
            self._write_conversation_turn(
                session_id=session_id,
                question=question,
                answer=answer,
                turn_info=turn_info,
                query_plan=query_plan,
                resolution=resolution,
                execution_result=execution_result,
            )
            return answer

        route_type = query_plan["route_type"]
        filters = query_plan["filters"]
        dish_name = query_plan["dish_name"]
        entities = query_plan["entities"]
        rewritten_query = self._rewrite_question_for_search(rewritten_question, route_type)
        relevant_chunks = self._search_relevant_chunks(
            rewritten_question,
            rewritten_query,
            filters,
            dish_name,
        )
        self._print_relevant_chunk_summary(relevant_chunks)

        if not relevant_chunks:
            logger.info("检索结果为空: question=%s session_id=%s entities=%s", rewritten_question, session_id, entities)
            answer = "抱歉，没有找到相关的食谱信息。请尝试其他菜品名称或关键词。"
            execution_result = self._build_execution_result(
                success=False,
                answer=answer,
                rewritten_question=rewritten_question,
                original_question=question,
                query_plan=query_plan,
                resolution=resolution,
                parent_docs=[],
            )
            self.last_execution_result = execution_result
            self._write_conversation_turn(
                session_id=session_id,
                question=question,
                answer=answer,
                turn_info=turn_info,
                query_plan=query_plan,
                resolution=resolution,
                execution_result=execution_result,
            )
            if return_diagnostics and not stream:
                self.last_query_diagnostics = self._build_turn_diagnostics(
                    original_question=original_question,
                    resolved_question=question,
                    rewritten_query=rewritten_query,
                    query_plan=query_plan,
                    answer=answer,
                    expectation=expectation or {},
                    generation_trace={
                        "strategy": "no_retrieval_result",
                        "content_type": filters.get("content_type"),
                        "context_doc_count": 0,
                        "reason": "no_relevant_chunks",
                    },
                )
                return {"answer": answer, "diagnostics": self.last_query_diagnostics}
            return answer

        if execution_plan["action"] == "retrieve_list" or route_type == "list":
            answer = self._generate_list_response(rewritten_question, session_id, relevant_chunks)
            recommended_dishes = self._extract_recommended_dishes(answer, list(self._latest_parent_docs))
            execution_result = self._build_execution_result(
                success=True,
                answer=answer,
                rewritten_question=rewritten_question,
                original_question=question,
                query_plan=query_plan,
                resolution=resolution,
                parent_docs=list(self._latest_parent_docs),
                recommended_dishes=recommended_dishes,
            )
        else:
            answer = self._generate_detail_response(
                rewritten_question,
                stream,
                session_id,
                route_type,
                filters,
                entities,
                dish_name,
                relevant_chunks,
            )
            execution_result = self._build_execution_result(
                success=True,
                answer=answer,
                rewritten_question=rewritten_question,
                original_question=question,
                query_plan=query_plan,
                resolution=resolution,
                parent_docs=list(self._latest_parent_docs),
            )
        self.last_execution_result = execution_result

        # For stream mode, wrap generator to defer writeback until consumption
        if stream and not isinstance(answer, str):
            return self._wrap_stream_with_writeback(
                answer_stream=answer,
                session_id=session_id,
                question=question,
                turn_info=turn_info,
                query_plan=query_plan,
                resolution=resolution,
                execution_result=execution_result,
            )

        self._write_conversation_turn(
            session_id=session_id,
            question=question,
            answer=answer,
            turn_info=turn_info,
            query_plan=query_plan,
            resolution=resolution,
            execution_result=execution_result,
        )

        if return_diagnostics and not stream:
            self.last_query_diagnostics = self._build_turn_diagnostics(
                original_question=original_question,
                resolved_question=question,
                rewritten_query=rewritten_query,
                query_plan=query_plan,
                answer=answer,
                expectation=expectation or {},
                generation_trace=getattr(self.generation_module, "last_generation_trace", {}),
            )
            return {"answer": answer, "diagnostics": self.last_query_diagnostics}

        return answer

    def _build_turn_diagnostics(
        self,
        original_question: str,
        resolved_question: str,
        rewritten_query: str,
        query_plan: Dict[str, Any],
        answer: str,
        expectation: Dict[str, Any],
        generation_trace: Dict[str, Any],
    ) -> Dict[str, Any]:
        """构建单轮分层诊断报告。"""
        from evaluation.process_diagnostics import build_turn_diagnostic_report

        retrieval_trace = getattr(self.retrieval_module, "last_search_trace", {})
        report = build_turn_diagnostic_report(
            question=resolved_question,
            answer=answer,
            query_plan=query_plan,
            rewritten_query=rewritten_query,
            retrieval_trace=retrieval_trace,
            relevant_docs=list(self._latest_parent_docs),
            generation_trace=generation_trace,
            expectation=expectation,
        )
        report["question_original"] = original_question
        return report
    
    def search_by_category(self, category: str, query: str = "") -> List[str]:
        """
        按分类搜索菜品
        
        Args:
            category: 菜品分类
            query: 可选的额外查询条件
            
        Returns:
            菜品名称列表
        """
        if not self.retrieval_module:
            raise ValueError("请先构建知识库")
        
        # 使用元数据过滤搜索
        search_query = query if query else category
        filters = {"category": category}
        
        docs = self.retrieval_module.metadata_filtered_search(search_query, filters, top_k=10)
        
        # 提取菜品名称
        dish_names = []
        for doc in docs:
            dish_name = doc.metadata.get('dish_name', '未知菜品')
            if dish_name not in dish_names:
                dish_names.append(dish_name)
        
        return dish_names
    
    def get_ingredients_list(self, dish_name: str) -> str:
        """
        获取指定菜品的食材信息

        Args:
            dish_name: 菜品名称

        Returns:
            食材信息
        """
        if not all([self.retrieval_module, self.generation_module]):
            raise ValueError("请先构建知识库")

        # 搜索相关文档
        docs = self.retrieval_module.hybrid_search(dish_name, top_k=3)

        # 生成食材信息
        answer = self.generation_module.generate_basic_answer(
            f"{dish_name}需要什么食材？",
            docs,
            content_type="ingredients",
        )

        return answer
    
    def run_interactive(self):
        """运行交互式问答"""
        print("=" * 60)
        print("尝尝咸淡RAG系统 - 交互式问答")
        print("=" * 60)
        print("解决您的选择困难症，告别'今天吃什么'的世纪难题！")
        self.print_startup_check()

        # 初始化系统
        self.initialize_system()

        # 构建知识库
        self.build_knowledge_base()

        print("\n交互式问答 (输入'退出'结束, 输入's'切换流式模式, 输入'r'重置会话):")

        # 默认使用流式输出（优化用户体验）
        use_stream = True
        print(f"当前模式: 流式输出 {'(开启)' if use_stream else '(关闭)'}")

        # 固定会话ID（用于多轮对话）
        session_id = "user_session_1"

        while True:
            try:
                user_input = input("\n您的问题: ").strip()
                if user_input.lower() in ['退出', 'quit', 'exit', '']:
                    break

                # 支持动态切换流式模式
                if user_input.lower() == 's':
                    use_stream = not use_stream
                    print(f"已切换为: 流式输出 {'(开启)' if use_stream else '(关闭)'}")
                    continue

                # 支持重置会话（切换话题时）
                if user_input.lower() == 'r':
                    self.reset_session(session_id)
                    print("会话已重置，开始新的话题...")
                    continue

                print("\n回答:")
                if use_stream:
                    # 流式输出（默认方式，实时响应）
                    for chunk in self.ask_question(user_input, stream=True, session_id=session_id):
                        print(chunk, end="", flush=True)
                    print("\n")
                else:
                    # 普通输出（等待完整回答）
                    answer = self.ask_question(user_input, stream=False, session_id=session_id)
                    print(f"{answer}\n")

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"处理问题时出错: {e}")

        print("\n感谢使用尝尝咸淡RAG系统！")



def main():
    """主函数"""
    try:
        # 创建RAG系统
        rag_system = RecipeRAGSystem()
        
        # 运行交互式问答
        rag_system.run_interactive()
        
    except Exception as e:
        logger.error(f"系统运行出错: {e}")
        print(f"系统错误: {e}")

if __name__ == "__main__":
    main()
