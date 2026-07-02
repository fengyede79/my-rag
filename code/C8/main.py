"""
RAG系统主程序
"""

import os
import sys
import logging
import io
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

    def _resolve_question_reference(self, question: str, session_id: str) -> str:
        """解析推荐列表中的序号引用。"""
        original_question = question
        resolved_question = self.generation_module.resolve_query_reference(question, session_id)
        if resolved_question != question:
            print(f"序号引用解析: '{question}' -> '{resolved_question}'")
        return resolved_question

    def _build_query_plan(self, question: str, session_id: str) -> Dict[str, Any]:
        """构建问答执行所需的查询计划。"""
        intent = self.generation_module.query_router(question)
        route_type = intent["type"]
        filters = intent.get("filters", {})
        dish_name = intent.get("dish_name")
        confidence = intent.get("confidence", 0)

        print(f"查询类型: {route_type} (置信度: {confidence:.2f})")
        if filters:
            print(f"提取过滤条件: {filters}")
        if dish_name:
            print(f"提取菜品名称: {dish_name}")

        entities = {"dish_name": dish_name, "filters": filters}
        if not dish_name and self.generation_module.conversation_manager:
            current_entity = self.generation_module.get_current_entity(session_id)
            if current_entity:
                dish_name = current_entity
                entities["dish_name"] = dish_name
                print(f"继承会话菜品: {dish_name}")

        return {
            "route_type": route_type,
            "filters": filters,
            "dish_name": dish_name,
            "entities": entities,
            "confidence": confidence,
        }

    def _rewrite_question_for_search(self, question: str, route_type: str) -> str:
        """根据问题类型决定是否重写查询。"""
        if route_type == "list":
            print(f"列表查询保持原样: {question}")
            return question

        print("智能分析查询...")
        return self.generation_module.query_rewrite(question)

    def _complete_question_with_conversation(
        self,
        question: str,
        session_id: str,
        query_plan: Dict[str, Any],
    ) -> Tuple[str, Dict[str, Any]]:
        """在检索前完成多轮问题补全，避免召回阶段丢失当前菜品。"""
        if not getattr(self.generation_module, "conversation_manager", None):
            return question, query_plan

        completed_question = self.generation_module.conversation_manager.complete_query(
            session_id,
            question,
            extracted_intent={"dish_name": query_plan.get("dish_name")},
        )
        if completed_question == question:
            return question, query_plan

        print(f"多轮问题补全: '{question}' -> '{completed_question}'")
        updated_query_plan = self._build_query_plan(completed_question, session_id)
        return completed_question, updated_query_plan

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
            print(f"强制菜品名过滤: {dish_name}")

        search_query = rewritten_query
        if dish_name and dish_name not in rewritten_query:
            search_query = f"{dish_name} {rewritten_query}"
            print(f"增强检索查询: {search_query}")

        if combined_filters:
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
            self.generation_module.save_recommendations(session_id, question, doc_names)
        self._latest_parent_docs = list(relevant_docs)

        return self.generation_module.generate_list_answer(question, relevant_docs)

    def _with_conversation_tracking_stream(
        self,
        stream_iterable,
        *,
        session_id: str,
        question: str,
        intent_type: str,
        entities: Dict[str, Any],
    ):
        """在流式输出结束后补写会话状态，避免流式模式丢失上下文。"""
        conversation_manager = getattr(self.generation_module, "conversation_manager", None)
        if not conversation_manager:
            return stream_iterable

        def tracked_stream():
            chunks = []
            for chunk in stream_iterable:
                chunks.append(chunk)
                yield chunk

            conversation_manager.add_interaction(
                session_id,
                question,
                "".join(chunks),
                intent_type=intent_type,
                entities=entities or {},
            )

        return tracked_stream()

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
                return self.generation_module.generate_step_by_step_answer_stream_with_conversation(
                    question,
                    relevant_docs,
                    session_id=session_id,
                    intent_type=route_type,
                    entities=entities,
                    content_type=content_type,
                )
            return self.generation_module.generate_step_by_step_with_conversation(
                question,
                relevant_docs,
                session_id=session_id,
                intent_type=route_type,
                entities=entities,
                content_type=content_type,
            )

        if stream:
            return self.generation_module.generate_basic_answer_stream_with_conversation(
                question,
                relevant_docs,
                session_id=session_id,
                intent_type=route_type,
                entities=entities,
                content_type=content_type,
            )
        return self.generation_module.generate_with_conversation(
            question,
            relevant_docs,
            session_id=session_id,
            intent_type=route_type,
            entities=entities,
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
        question = self._resolve_question_reference(question, session_id)
        query_plan = self._build_query_plan(question, session_id)
        question, query_plan = self._complete_question_with_conversation(
            question,
            session_id,
            query_plan,
        )
        guardrail_answer = self._maybe_handle_guardrail_query(question)
        if guardrail_answer is not None:
            if return_diagnostics and not stream:
                self.last_query_diagnostics = self._build_turn_diagnostics(
                    original_question=original_question,
                    resolved_question=question,
                    rewritten_query=question,
                    query_plan=query_plan,
                    answer=guardrail_answer,
                    expectation=expectation or {},
                    generation_trace=getattr(self.generation_module, "last_generation_trace", {}),
                )
                return {"answer": guardrail_answer, "diagnostics": self.last_query_diagnostics}
            return guardrail_answer
        route_type = query_plan["route_type"]
        filters = query_plan["filters"]
        dish_name = query_plan["dish_name"]
        entities = query_plan["entities"]
        rewritten_query = self._rewrite_question_for_search(question, route_type)
        relevant_chunks = self._search_relevant_chunks(
            question,
            rewritten_query,
            filters,
            dish_name,
        )
        self._print_relevant_chunk_summary(relevant_chunks)

        if not relevant_chunks:
            answer = "抱歉，没有找到相关的食谱信息。请尝试其他菜品名称或关键词。"
            if getattr(self.generation_module, "conversation_manager", None):
                self.generation_module.conversation_manager.add_interaction(
                    session_id,
                    question,
                    answer,
                    intent_type=route_type,
                    entities=entities or {},
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

        if route_type == "list":
            answer = self._generate_list_response(question, session_id, relevant_chunks)
            if getattr(self.generation_module, "conversation_manager", None):
                self.generation_module.conversation_manager.add_interaction(
                    session_id,
                    question,
                    answer,
                    intent_type=route_type,
                    entities=entities or {},
                )
        else:
            answer = self._generate_detail_response(
                question,
                stream,
                session_id,
                route_type,
                filters,
                entities,
                dish_name,
                relevant_chunks,
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
