"""
生成集成模块 - 编排查询路由、上下文构建和回答生成。
护栏、结构化回答、Prompt 模板、安全调用已拆分到独立模块。
"""

import os
import re
import logging
import threading
from typing import List, Dict, Optional

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from . import guardrail as _guardrail_module
from . import structured_generation as _structured_generation
from . import stream_handler as _stream_handler
from .prompts import (
    build_targeted_prompt,
    build_no_context_answer as _build_no_context_answer_fn,
    BASIC_ANSWER_PROMPT_TEMPLATE,
    STEP_BY_STEP_PROMPT_TEMPLATE,
)

logger = logging.getLogger(__name__)

class GenerationIntegrationModule:
    """生成集成模块 - 负责LLM集成和回答生成"""

    def __init__(self, model_name: str = "qwen-turbo", temperature: float = 0.1,
                 max_tokens: int = 2048, enable_conversation: bool = False):
        """
        初始化生成集成模块

        Args:
            model_name: 模型名称
            temperature: 生成温度
            max_tokens: 最大token数
            enable_conversation: 是否启用多轮对话支持
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.llm = None
        self.conversation_manager = None
        self.hybrid_router = None
        self.last_generation_trace = {}
        self._state_lock = threading.RLock()
        
        # 推荐列表缓存，用于多轮对话
        self.last_recommendations = {}  # {session_id: {"query": "...", "dishes": [...]}}
        
        # 初始化会话管理器（可选）
        if enable_conversation:
            from .conversation_manager import ConversationManager
            self.conversation_manager = ConversationManager()
            logger.info("多轮对话支持已启用")
        
        self.setup_llm()
        self.setup_hybrid_router()

    def _record_generation_trace(
        self,
        strategy: str,
        content_type: str = None,
        context_doc_count: int = 0,
        reason: str = None,
    ):
        """记录最近一次生成路径，供过程级诊断使用。"""
        self.last_generation_trace = {
            "strategy": strategy,
            "content_type": content_type,
            "context_doc_count": context_doc_count,
            "reason": reason,
        }
    
    def setup_llm(self):
        """初始化大语言模型"""
        logger.info(f"正在初始化LLM: {self.model_name}")

        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError("请设置 DASHSCOPE_API_KEY 环境变量")

        # 使用 OpenAI 兼容方式调用通义千问
        # 添加超时和重试机制，提升稳定性
        self.llm = ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            timeout=30,        # 请求超时时间（秒）
            max_retries=2      # 最大重试次数
        )

        logger.info("LLM初始化完成（超时30秒，最大重试2次）")
    
    def setup_hybrid_router(self):
        """初始化混合路由模块"""
        try:
            from .hybrid_router import HybridRouter
            self.hybrid_router = HybridRouter(self.llm)
            logger.info("混合路由模块初始化完成")
        except Exception as e:
            logger.warning(f"初始化混合路由失败: {e}")
            self.hybrid_router = None

    def _safe_chain_invoke(self, chain, input_data: dict, fallback_message: str = None) -> str:
        """安全调用 chain.invoke（委托到 stream_handler）。"""
        return _stream_handler.safe_chain_invoke(chain, input_data, fallback_message)

    def _safe_chain_stream(self, chain, input_data: dict, fallback_message: str = None):
        """安全调用 chain.stream（委托到 stream_handler）。"""
        return _stream_handler.safe_chain_stream(chain, input_data, fallback_message)

    def _check_dish_consistency(self, query: str, context_docs: List[Document]) -> Optional[str]:
        """
        生成前菜品一致性校验（最后防线）

        Args:
            query: 用户查询
            context_docs: 上下文文档列表

        Returns:
            None 表示一致性检查通过，
            str 表示错误提示信息（检查失败）
        """
        if not context_docs:
            return None

        # 从查询中提取菜品名（使用规则路由同样的关键词）
        detail_keywords = sorted(
            ["怎么做", "怎么制作", "制作方法", "需要什么食材", "步骤", "做法", "食材", "材料"],
            key=len, reverse=True
        )
        query_dish = self._extract_dish_name(query, detail_keywords)
        if not query_dish:
            return None  # 无法提取菜品名，跳过检查

        # 从上下文文档中提取所有菜品名
        doc_dishes = {doc.metadata.get('dish_name', '') for doc in context_docs if doc.metadata.get('dish_name')}
        if not doc_dishes:
            return None  # 文档无菜品名，跳过检查

        # 严格匹配：query_dish 必须精确或包含匹配文档菜品名
        # 这样"日式肥牛丼饭"可以匹配"日式肥牛丼饭"，但不匹配"咖喱肥牛"
        match_found = False
        for doc_dish in doc_dishes:
            # 精确匹配或互相包含（长度差异不超过30%）
            if query_dish == doc_dish:
                match_found = True
                break
            # 单向包含（query包含doc或doc包含query）
            if query_dish in doc_dish or doc_dish in query_dish:
                len_ratio = max(len(query_dish), len(doc_dish)) / min(len(query_dish), len(doc_dish))
                if len_ratio <= 1.3:
                    match_found = True
                    break

        if not match_found:
            logger.warning(f"[ConsistencyCheck] 查询菜品'{query_dish}'与文档菜品{list(doc_dishes)}不匹配，拦截生成")
            return f"抱歉，知识库中未找到「{query_dish}」的相关食谱信息。当前检索到的内容属于 {list(doc_dishes)[0]}，请您确认菜品名称或尝试其他问法。"
        return None

    def _classify_query_guardrail(self, query: str) -> Optional[str]:
        """识别应在检索前直接保守兜底的问题（委托到 guardrail）。"""
        return _guardrail_module.classify_query_guardrail(query)

    def build_guardrail_answer(self, query: str, reason: str) -> str:
        """统一生成边界问题的保守回答（委托到 guardrail）。"""
        return _guardrail_module.build_guardrail_answer(query, reason)

    def _build_no_context_answer(self, query: str, content_type: str = None) -> str:
        """在上下文不足时给出保守回答（委托到 prompts）。"""
        detail_keywords = sorted(
            ["怎么做", "怎么制作", "制作方法", "需要什么食材", "步骤", "做法", "食材", "材料"],
            key=len, reverse=True,
        )
        dish_name = self._extract_dish_name(query, detail_keywords)
        return _build_no_context_answer_fn(query, content_type, dish_name=dish_name)

    def _try_build_structured_answer(self, query: str, context_docs: List[Document], content_type: str = None) -> Optional[str]:
        """文档结构明确时优先直接回答（委托到 structured_generation）。"""
        return _structured_generation.try_build_structured_answer(query, context_docs, content_type)

    def _generate_answer(self, query, context_docs, *, stream=False, step_by_step=False, content_type=None):
        """统一的非会话生成核心逻辑。"""
        effective_ct = (content_type or "steps") if step_by_step else content_type

        if not context_docs:
            self._record_generation_trace("no_context", content_type=effective_ct, context_doc_count=0, reason="missing_context_docs")
            response = self._build_no_context_answer(query, effective_ct)
            if stream:
                yield response
                return
            return response

        structured_answer = self._try_build_structured_answer(query, context_docs, effective_ct)
        if structured_answer:
            self._record_generation_trace("structured", content_type=effective_ct, context_doc_count=len(context_docs))
            if stream:
                yield structured_answer
                return
            return structured_answer

        error_msg = self._check_dish_consistency(query, context_docs)
        if error_msg:
            self._record_generation_trace("consistency_blocked", content_type=effective_ct, context_doc_count=len(context_docs), reason="dish_mismatch")
            if stream:
                yield error_msg
                return
            return error_msg

        context = self._build_context(context_docs)
        template = STEP_BY_STEP_PROMPT_TEMPLATE if step_by_step else BASIC_ANSWER_PROMPT_TEMPLATE
        prompt = ChatPromptTemplate.from_template(template)

        chain = (
            {"question": RunnablePassthrough(), "context": lambda _: context}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        if stream:
            self._record_generation_trace("llm_stream", content_type=effective_ct, context_doc_count=len(context_docs))
            yield from self._safe_chain_stream(chain, query)
        else:
            response = self._safe_chain_invoke(chain, query)
            self._record_generation_trace("llm", content_type=effective_ct, context_doc_count=len(context_docs))
            return response

    def generate_basic_answer(self, query, context_docs, content_type=None):
        """生成基础回答（带引用溯源）。"""
        return self._generate_answer(query, context_docs, stream=False, step_by_step=False, content_type=content_type)

    def generate_step_by_step_answer(self, query, context_docs, content_type=None):
        """生成分步骤回答（带引用溯源）。"""
        return self._generate_answer(query, context_docs, stream=False, step_by_step=True, content_type=content_type)

    def generate_basic_answer_stream(self, query, context_docs, content_type=None):
        """生成基础回答 - 流式输出（带引用溯源）。"""
        yield from self._generate_answer(query, context_docs, stream=True, step_by_step=False, content_type=content_type)

    def generate_step_by_step_answer_stream(self, query, context_docs, content_type=None):
        """生成详细步骤回答 - 流式输出（带引用溯源）。"""
        yield from self._generate_answer(query, context_docs, stream=True, step_by_step=True, content_type=content_type)
    
    def save_recommendations(self, session_id: str, query: str, dishes: List[str]):
        """
        保存推荐列表，用于后续引用解析
        
        Args:
            session_id: 会话ID
            query: 原始查询
            dishes: 推荐的菜品列表
        """
        with self._state_lock:
            self.last_recommendations[session_id] = {
                "query": query,
                "dishes": dishes,
                "dish_count": len(dishes)  # 保存列表长度，用于验证引用
            }
        logger.debug(f"保存推荐列表到会话 {session_id}: {dishes}")

    def reset_session_state(self, session_id: str):
        """
        清理当前会话的推荐缓存和对话状态

        Args:
            session_id: 会话ID
        """
        if session_id in self.last_recommendations:
            del self.last_recommendations[session_id]
            logger.info(f"清理推荐列表缓存: {session_id}")

        if self.conversation_manager:
            self.conversation_manager.reset_session(session_id)
    
    def resolve_query_reference(self, query: str, session_id: str) -> str:
        """
        解析查询中的序号引用（如"3怎么做" -> "鸡蛋三明治怎么做"，"3需要什么食材" -> "日式肥牛丼饭需要什么食材"）
        
        Args:
            query: 用户查询
            session_id: 会话ID
            
        Returns:
            解析后的查询
        """
        # 检查是否以序号开头（如 "3..." 或 "3 ..."）
        # 支持的模式：
        # - "3怎么做" -> "菜品名怎么做"
        # - "3看起来不错" -> "菜品名需要什么" (保留核心意图)
        # - "3需要什么食材" -> "菜品名需要什么食材"
        number_pattern = r'^(\d+)[\s，。、,.、]*(.*)$'
        match = re.match(number_pattern, query)
        
        if not match:
            return query
        
        # 提取序号
        number = int(match.group(1))
        rest_query = match.group(2) if match.group(2) else ""
        
        # 获取上一次的推荐列表（加锁保护，避免并发写入时读到不一致状态）
        with self._state_lock:
            if session_id not in self.last_recommendations:
                logger.warning(f"会话 {session_id} 没有推荐列表缓存")
                return query

            rec_info = self.last_recommendations[session_id]
            dishes = rec_info.get("dishes", [])
        
        # 检查序号是否有效
        if number <= 0 or number > len(dishes):
            logger.warning(f"序号 {number} 超出推荐列表范围 (1-{len(dishes)})")
            return query
        
        # 获取对应序号的菜品
        dish_name = dishes[number - 1]  # 1-based to 0-based
        
        # 分析 rest_query，提取核心意图
        # 移除"看起来不错"、"看起来很好"等评价性词汇
        evaluation_patterns = [
            r'看起来不错',
            r'看起来很好', 
            r'看起来很棒',
            r'不错',
            r'挺好',
            r'不错啊',
            r'看起来行',
            r'可以',
            r'可以啊',
            r'好呀',
            r'好嘞'
        ]
        
        core_intent = rest_query
        for pattern in evaluation_patterns:
            # 移除评价性词汇，保留核心查询
            core_intent = re.sub(pattern, '', core_intent)
        
        # 清理多余的标点符号
        core_intent = re.sub(r'^[，。、,\s]+|[，。、,\s]+$', '', core_intent)
        
        # 构建新查询：菜品名 + 核心意图
        if core_intent:
            resolved_query = f"{dish_name}{core_intent}"
        else:
            # 如果只有序号和评价性词汇，添加一个通用后缀
            resolved_query = f"{dish_name}需要什么食材"
        
        logger.info(f"解析序号引用: '{query}' -> '{resolved_query}'")
        
        return resolved_query
    
    def query_rewrite(self, query: str) -> str:
        """
        智能查询重写 - 规则优先，减少 LLM 调用

        Args:
            query: 原始查询

        Returns:
            重写后的查询或原查询
        """
        # 1. 规则优先（不调用 LLM）
        rewritten = self._rule_based_rewrite(query)
        if rewritten:
            logger.info(f"规则重写: '{query}' → '{rewritten}'")
            return rewritten
        
        # 2. 具体查询直接返回（不调用 LLM）
        if self._is_specific_query(query):
            logger.info(f"查询无需重写: '{query}'")
            return query
        
        # 3. LLM fallback（仅模糊查询）
        logger.info(f"使用 LLM 重写模糊查询: '{query}'")
        return self._llm_based_rewrite(query)
    
    def _rule_based_rewrite(self, query: str) -> Optional[str]:
        """
        规则优先的查询重写（不调用 LLM）

        Args:
            query: 原始查询

        Returns:
            重写后的查询，如果规则不匹配则返回 None
        """
        # 定义常见模糊查询的重写规则
        rewrite_rules = {
            "做菜": "简单易做的家常菜谱",
            "做饭": "简单家常菜制作方法",
            "有什么好吃的": "推荐好吃的家常菜",
            "推荐个菜": "简单家常菜推荐",
            "推荐一道菜": "简单家常菜推荐",
            "想吃点什么": "简单家常菜推荐",
            "有什么菜": "家常菜菜谱推荐",
            "川菜": "经典川菜菜谱",
            "湘菜": "经典湘菜菜谱",
            "粤菜": "经典粤菜菜谱",
            "鲁菜": "经典鲁菜菜谱",
            "素菜": "素食菜谱推荐",
            "荤菜": "荤菜菜谱推荐",
            "简单的": "简单易做的菜谱",
            "容易做的": "简单易做的菜谱",
            "新手": "适合新手的家常菜",
            "入门": "适合新手的家常菜",
            "有饮品推荐吗": "简单饮品制作方法",
            "饮料": "简单饮品制作方法",
            "甜品": "简单甜品制作方法",
            "早餐": "简单早餐制作方法",
            "汤": "简单汤品制作方法",
        }
        
        # 检查是否匹配规则
        for pattern, rewritten in rewrite_rules.items():
            if pattern in query.lower() or query.lower() == pattern:
                return rewritten
        
        return None
    
    def _is_specific_query(self, query: str) -> bool:
        """
        判断是否为具体查询（不需要重写）

        Args:
            query: 原始查询

        Returns:
            True 表示具体查询，False 表示模糊查询
        """
        # 具体菜品名称关键词
        dish_keywords = ["怎么做", "怎么制作", "制作方法", "做法", "步骤",
                         "需要什么食材", "食材", "原料", "配料"]
        
        # 如果包含具体菜品名称 + 制作关键词，认为是具体查询
        if any(kw in query for kw in dish_keywords):
            # 检查是否有菜品名称（长度>2且不是纯关键词）
            for kw in dish_keywords:
                if kw in query:
                    parts = query.split(kw)
                    if parts[0].strip() and len(parts[0].strip()) > 2:
                        return True
                    if len(parts) > 1 and parts[1].strip() and len(parts[1].strip()) > 2:
                        return True
        
        # 包含具体烹饪技巧关键词
        technique_keywords = ["不粘锅", "调味", "腌制", "焯水", "爆炒", "炖煮"]
        if any(kw in query for kw in technique_keywords):
            return True
        
        return False
    
    def _llm_based_rewrite(self, query: str) -> str:
        """
        LLM 查询重写（仅用于模糊查询）

        Args:
            query: 原始查询

        Returns:
            重写后的查询
        """
        prompt = PromptTemplate(
            template="""
你是一个智能查询分析助手。请将用户的模糊查询重写为更具体的食谱搜索查询。

原始查询: {query}

重写原则：
- 保持原意不变
- 增加相关烹饪术语
- 优先推荐简单易做的
- 保持简洁性

只输出重写后的查询，不要其他内容:""",
            input_variables=["query"]
        )

        chain = (
            {"query": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        response = self._safe_chain_invoke(chain, {"query": query})
        return response.strip() if response else query



    def query_router(self, query: str) -> Dict:
        """
        查询路由 - 根据查询类型选择不同的处理方式，并提取意图详情

        Args:
            query: 用户查询

        Returns:
            结构化意图信息:
            {
                "type": "list" | "detail" | "general",
                "filters": {"category": "荤菜", "difficulty": "简单", "content_type": "ingredients"},
                "dish_name": "宫保鸡丁",  # 提取的菜品名称（如果有）
                "confidence": 0.9  # 置信度
            }
        """
        # 先做规则路由，明确的 detail 问题优先，避免被 list 兜底覆盖
        intent = self._rule_based_routing(query)
        if self._is_explicit_detail_intent(intent):
            logger.info(f"规则路由优先命中 detail: {intent}")
            return intent
        if self._is_conversational_dish_intent(intent):
            logger.info(f"规则路由优先命中会话菜品主题: {intent}")
            return intent

        # 再使用混合路由获取内容类型
        content_type, route_type, route_info = self._hybrid_route(query)
        confidence = route_info.get("confidence", 0.5)

        logger.info(f"混合路由结果: content_type={content_type}, route_type={route_type}, confidence={confidence}")

        dish_name = intent.get("dish_name")

        if content_type in ["ingredients", "steps", "tips", "introduction"]:
            merged_filters = dict(intent.get("filters", {}))
            merged_filters["content_type"] = content_type
            return {
                "type": "detail",
                "filters": merged_filters,
                "dish_name": dish_name,
                "confidence": max(confidence, intent.get("confidence", 0.0)),
            }

        # 如果混合路由落到 list/general，但规则已经识别出明确菜名或细粒度过滤，则仍走规则结果
        if self._should_prefer_rule_intent(intent, content_type):
            logger.info(f"规则路由优先于混合路由兜底: {intent}")
            return intent

        if content_type == "list":
            return {
                "type": "list",
                "filters": intent.get("filters", {}) if intent.get("type") == "list" else {},
                "dish_name": None,
                "confidence": confidence
            }

        return intent

    def _is_explicit_detail_intent(self, intent: Dict) -> bool:
        """判断规则路由结果是否为明确的 detail 问题。"""
        if intent.get("type") != "detail":
            return False
        if intent.get("dish_name"):
            return True
        return "content_type" in intent.get("filters", {})

    def _is_conversational_dish_intent(self, intent: Dict) -> bool:
        """判断是否为带明确菜名的会话主题。"""
        return intent.get("type") == "general" and bool(intent.get("dish_name"))

    def _should_prefer_rule_intent(self, intent: Dict, hybrid_content_type: str) -> bool:
        """当混合路由结果过于宽泛时，判断是否应优先采用规则路由。"""
        if not intent:
            return False
        if hybrid_content_type not in {"list", "general"}:
            return False
        if intent.get("type") == "general" and intent.get("dish_name"):
            return True
        if intent.get("type") == "detail" and (
            intent.get("dish_name") or intent.get("filters", {}).get("content_type")
        ):
            return True
        return False
    
    def _hybrid_route(self, query: str) -> tuple:
        """
        执行三层混合路由
        
        Returns:
            (content_type, route_type, route_info)
        """
        # 如果混合路由可用，使用它
        if self.hybrid_router:
            return self.hybrid_router.route(query)
        
        # 否则回退到简单规则
        return self._simple_content_type_detection(query)
    
    def _simple_content_type_detection(self, query: str) -> tuple:
        """
        简单的内容类型检测（混合路由不可用时的回退）
        
        Returns:
            (content_type, route_type, route_info)
        """
        # 明确 detail 问题优先，避免被推荐类问题吞掉
        if any(kw in query for kw in ["需要什么食材", "需要什么材料", "需要什么原料", "需要什么配料"]):
            return "ingredients", "rule", {"confidence": 0.98}
        # 食材相关
        if any(kw in query for kw in ["食材", "材料", "配料", "原料"]):
            return "ingredients", "rule", {"confidence": 0.9}
        # 步骤相关
        if any(kw in query for kw in ["步骤", "流程", "做法", "怎么做", "制作方法"]):
            return "steps", "rule", {"confidence": 0.9}
        # 技巧相关
        if any(kw in query for kw in ["技巧", "窍门", "注意", "小贴士"]):
            return "tips", "rule", {"confidence": 0.9}
        # 介绍相关
        if any(kw in query for kw in ["介绍", "简介", "特点", "特色"]):
            return "introduction", "rule", {"confidence": 0.9}
        # 推荐相关
        if any(kw in query for kw in ["推荐", "有什么", "有哪些"]):
            return "list", "rule", {"confidence": 0.9}
        
        # 默认返回list（推荐查询）
        return "list", "fallback", {"confidence": 0.5}

    def _extract_dish_name(self, raw_query: str, detail_keywords: list) -> Optional[str]:
        """
        从原始查询中提取并清洗菜品名称

        Args:
            raw_query: 原始查询，如 "日式肥牛丼饭不错，怎么做？" 或 "日式肥牛丼饭需要的食材是什么？"
            detail_keywords: 已排序的关键词列表（按长度降序）

        Returns:
            清洗后的菜品名，如 "日式肥牛丼饭"，无效返回 None
        """
        # 0. 预处理：移除句首评价类短语（如 "很好吃的"、"很好喝的"）
        raw_query = re.sub(r'^(很好吃的|很好喝的|很好做的|看起来很好|听起来很好|闻起来很香)[，,\s]?', '', raw_query)

        # 1. 先按关键词切割，找到菜品名所在位置
        dish_name = None
        keyword_found = None
        for kw in detail_keywords:
            if kw in raw_query:
                parts = raw_query.split(kw, 1)
                if parts[0].strip():
                    dish_name = parts[0].strip()
                    keyword_found = kw
                elif len(parts) > 1 and parts[1].strip():
                    dish_name = parts[1].strip()
                    keyword_found = kw
                break

        if not dish_name:
            return None

        # 2. 移除可能的前缀（序号、"第X个"等）
        dish_name = re.sub(r'^[第\d一二三四五六七八九十]+[个号.\s]*', '', dish_name)

        # 3. 移除句尾标点
        dish_name = re.sub(r'[，。？！,.!?\s]+$', '', dish_name)

        # 3.1 去掉常见连接词与尾部功能短语
        dish_name = re.sub(
            r'(的|做法|步骤|制作方法|制作|烹饪|食材|材料|原料|配料|技巧|需要|什么)+$',
            '',
            dish_name
        )

        # 4. 找到关键词前的边界：向前扫描，第一个非中文字符（标点/空格）即为 dish_name 的结束
        #    这能干净地截断 "韭菜盒子听起来很好，我想知道我需要什么食材"
        #    → 从"需"往左扫，"，"是非中文字符 → 边界在"，" → dish_name="韭菜盒子听起来很好我"
        if keyword_found:
            keyword_idx = raw_query.find(keyword_found)
            before_keyword = raw_query[:keyword_idx]
            # 从关键词前向前找第一个非中文字符的位置
            boundary_pos = len(before_keyword)
            for i in range(len(before_keyword) - 1, -1, -1):
                if not self._is_chinese(before_keyword[i]):
                    boundary_pos = i
                    break
            dish_name = before_keyword[:boundary_pos].strip()

        # 5. 再次清理首尾标点和空格
        dish_name = dish_name.strip('，。、,.。!?　 ')
        dish_name = re.sub(r'的$', '', dish_name)
        dish_name = re.sub(r'(有什么|有哪些|有啥|怎么|如何|需要.*)$', '', dish_name).strip()

        # 6. 合法性校验：长度过短或包含非菜品名词性成分
        if len(dish_name) < 2:
            return None
        invalid_chars = ['怎么', '需要', '什么', '哪些', '多少']
        if any(k in dish_name for k in invalid_chars):
            return None

        return dish_name if dish_name else None

    def _is_chinese(self, char: str) -> bool:
        """判断一个字符是否为中文"""
        return '\u4e00' <= char <= '\u9fff'

    def _rule_based_routing(self, query: str) -> Dict:
        """
        规则优先的路由判断（不调用 LLM）

        Args:
            query: 用户查询

        Returns:
            结构化意图信息
        """
        intent = {
            "type": "general",
            "filters": {},
            "dish_name": None,
            "confidence": 0.0
        }

        conversational_prefixes = ["我们聊聊", "聊聊", "说说", "讲讲"]
        for prefix in conversational_prefixes:
            if query.startswith(prefix):
                candidate = query[len(prefix):].strip(" ：:，,。！？? ")
                if candidate and len(candidate) >= 2:
                    intent["type"] = "general"
                    intent["dish_name"] = candidate
                    intent["confidence"] = 0.95
                    return intent

        # 列表查询规则
        list_keywords = ["推荐", "有什么", "有哪些", "给我", "列出", "几个"]
        
        # 通用饮食询问模式 - 优先匹配
        general_food_patterns = ["今天吃什么", "明天吃什么", "后天吃什么", "晚饭吃什么", "午饭吃什么", "早餐吃什么", 
                                  "午餐吃什么", "夜宵吃什么", "想吃点", "该吃啥", "吃啥好",
                                  "吃什么好", "吃点啥", "推荐菜", "做啥吃", "吃啥"]
        for pattern in general_food_patterns:
            if pattern in query:
                intent["type"] = "list"
                intent["confidence"] = 0.95
                # 不设置category过滤，让检索返回最相关的家常菜
                logger.info(f"通用饮食询问路由: '{query}' -> list (推荐家常菜)")
                return intent
        
        detail_keywords = sorted(
            [
                "需要什么食材", "需要什么材料", "需要什么原料", "需要什么配料",
                "怎么做", "怎么制作", "制作方法", "制作技巧", "制作步骤",
                "步骤", "做法", "食材", "材料", "原料", "配料", "技巧"
            ],
            key=len,
            reverse=True
        )

        if any(kw in query for kw in detail_keywords):
            intent["type"] = "detail"
            intent["confidence"] = 0.9

            content_type_mapping_items = sorted([
                ("需要什么食材", "ingredients"),
                ("需要什么材料", "ingredients"),
                ("需要什么原料", "ingredients"),
                ("需要什么配料", "ingredients"),
                ("需要什么", "ingredients"),
                ("食材", "ingredients"),
                ("原料", "ingredients"),
                ("配料", "ingredients"),
                ("材料", "ingredients"),
                ("制作步骤", "steps"),
                ("步骤", "steps"),
                ("做法", "steps"),
                ("怎么做", "steps"),
                ("怎么制作", "steps"),
                ("制作方法", "steps"),
                ("制作", "steps"),
                ("烹饪", "steps"),
                ("制作技巧", "tips"),
                ("技巧", "tips"),
                ("小贴士", "tips"),
                ("注意", "tips")
            ], key=lambda x: len(x[0]), reverse=True)

            content_type_mapping = dict(content_type_mapping_items)
            for kw, content_type in content_type_mapping.items():
                if kw in query:
                    intent["filters"]["content_type"] = content_type
                    intent["confidence"] = 0.95
                    break

            intent["dish_name"] = self._extract_dish_name(query, detail_keywords)
            return intent

        if any(kw in query for kw in list_keywords):
            intent["type"] = "list"
            intent["confidence"] = 0.9

            # 提取分类过滤条件
            category_mapping = {
                "荤菜": "荤菜", "素菜": "素菜", "汤": "汤品", "甜品": "甜品",
                "早餐": "早餐", "主食": "主食", "水产": "水产", "调料": "调料", "饮品": "饮品"
            }
            for cat in category_mapping:
                if cat in query:
                    intent["filters"]["category"] = cat
                    intent["confidence"] = 0.95

            # 提取难度过滤条件
            difficulty_keywords = {
                "非常简单": "非常简单", "简单": "简单", "中等": "中等",
                "困难": "困难", "非常困难": "非常困难"
            }
            for diff in sorted(difficulty_keywords.keys(), key=len, reverse=True):
                if diff in query:
                    intent["filters"]["difficulty"] = diff
                    intent["confidence"] = 0.95

        return intent

    def generate_list_answer(self, query: str, context_docs: List[Document]) -> str:
        """
        生成列表式回答 - 适用于推荐类查询

        Args:
            query: 用户查询
            context_docs: 上下文文档列表

        Returns:
            列表式回答
        """
        if not context_docs:
            return "抱歉，没有找到相关的菜品信息。"

        # 提取菜品名称
        dish_names = []
        for doc in context_docs:
            dish_name = doc.metadata.get('dish_name', '未知菜品')
            if dish_name not in dish_names:
                dish_names.append(dish_name)

        # 构建简洁的列表回答
        if len(dish_names) == 1:
            return f"为您推荐：{dish_names[0]}"
        elif len(dish_names) <= 3:
            return f"为您推荐以下菜品：\n" + "\n".join([f"{i+1}. {name}" for i, name in enumerate(dish_names)])
        else:
            return f"为您推荐以下菜品：\n" + "\n".join([f"{i+1}. {name}" for i, name in enumerate(dish_names[:3])]) + f"\n\n还有其他 {len(dish_names)-3} 道菜品可供选择。"

    def _build_context(self, docs: List[Document], max_length: int = 2000) -> str:
        """
        构建上下文字符串

        Args:
            docs: 文档列表
            max_length: 最大长度

        Returns:
            格式化的上下文字符串
        """
        if not docs:
            return "暂无相关食谱信息。"

        # 优化：按相关性分数排序（优先保留高相关性文档）
        # rrf_score 由检索模块的 RRF 重排计算得出
        sorted_docs = sorted(
            docs,
            key=lambda d: d.metadata.get('rrf_score', 0),
            reverse=True
        )

        context_parts = []
        current_length = 0

        for i, doc in enumerate(sorted_docs, 1):
            # 添加元数据信息
            metadata_info = f"【食谱 {i}】"
            if 'dish_name' in doc.metadata:
                metadata_info += f" {doc.metadata['dish_name']}"
            if 'category' in doc.metadata:
                metadata_info += f" | 分类: {doc.metadata['category']}"
            if 'difficulty' in doc.metadata:
                metadata_info += f" | 难度: {doc.metadata['difficulty']}"
            # 显示相关性分数（便于调试和评估）
            if 'rrf_score' in doc.metadata:
                metadata_info += f" | 相关性: {doc.metadata['rrf_score']:.3f}"

            # 构建文档文本
            doc_text = f"{metadata_info}\n{doc.page_content}\n"

            # 检查长度限制
            if current_length + len(doc_text) > max_length:
                break

            context_parts.append(doc_text)
            current_length += len(doc_text)

        divider = "\n" + "="*50 + "\n"
        return divider + divider.join(context_parts)

    # ============================================================
    # 多轮对话支持方法
    # ============================================================

    def _stream_text(self, text: str):
        """将文本作为单块 yield（委托到 stream_handler）。"""
        return _stream_handler.stream_text(text)

    def _run_conversation_pipeline(self, query, context_docs, session_id, *, effective_content_type, intent_type, entities):
        """
        会话生成的共享前置逻辑：校验 → 补全查询 → 构建上下文 → 组装 chain。
        返回 (chain, chain_input, completed_query, conversation_context) 或 (None, early_response, None, None)。
        """
        # 无上下文兜底
        if not context_docs:
            self._record_generation_trace("no_context", content_type=effective_content_type, context_doc_count=0, reason="missing_context_docs")
            response = self._build_no_context_answer(query, effective_content_type)
            self.conversation_manager.add_interaction(session_id, query, response, intent_type=intent_type, entities=entities or {})
            return None, response, None, None

        # 尝试结构化回答
        structured_answer = self._try_build_structured_answer(query, context_docs, effective_content_type)
        if structured_answer:
            self._record_generation_trace("structured", content_type=effective_content_type, context_doc_count=len(context_docs))
            self.conversation_manager.add_interaction(session_id, query, structured_answer, intent_type=intent_type, entities=entities or {})
            return None, structured_answer, None, None

        # 一致性校验
        error_msg = self._check_dish_consistency(query, context_docs)
        if error_msg:
            self._record_generation_trace("consistency_blocked", content_type=effective_content_type, context_doc_count=len(context_docs), reason="dish_mismatch")
            return None, error_msg, None, None

        # 补全查询 + 构建对话上下文
        completed_query = self.conversation_manager.complete_query(
            session_id, query, extracted_intent={"dish_name": entities.get("dish_name") if entities else None}
        )
        conversation_context = self.conversation_manager.get_conversation_context(session_id)
        context = self._build_context_with_conversation(context_docs, conversation_context)
        prompt = self._build_targeted_prompt(effective_content_type)

        chain = (
            {"question": RunnablePassthrough(), "food_context": RunnablePassthrough(), "conversation_context": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        chain_input = {
            "question": completed_query,
            "food_context": context,
            "conversation_context": conversation_context or "（暂无历史对话）"
        }
        return chain, chain_input, completed_query, conversation_context

    def _generate_conversation_answer_stream(self, query, context_docs, session_id, *, step_by_step, intent_type, entities, content_type):
        """会话生成的流式内部实现。"""
        effective_ct = (content_type or "steps") if step_by_step else content_type
        if not self.conversation_manager:
            yield from self._generate_answer(query, context_docs, stream=True, step_by_step=step_by_step, content_type=content_type)
            return

        chain, chain_input, completed_query, _ = self._run_conversation_pipeline(
            query, context_docs, session_id,
            effective_content_type=effective_ct, intent_type=intent_type, entities=entities,
        )
        if chain is None:
            yield chain_input
            return

        self._record_generation_trace("llm_stream", content_type=effective_ct, context_doc_count=len(context_docs))
        collected_chunks = []
        for chunk in self._safe_chain_stream(chain, chain_input):
            collected_chunks.append(chunk)
            yield chunk

        self.conversation_manager.add_interaction(
            session_id, completed_query, "".join(collected_chunks),
            intent_type=intent_type, entities=entities or {}
        )

    def _generate_conversation_answer(self, query, context_docs, session_id, *, step_by_step, intent_type, entities, content_type):
        """会话生成的非流式内部实现。"""
        effective_ct = (content_type or "steps") if step_by_step else content_type
        if not self.conversation_manager:
            return self._generate_answer(query, context_docs, stream=False, step_by_step=step_by_step, content_type=content_type)

        chain, chain_input, completed_query, _ = self._run_conversation_pipeline(
            query, context_docs, session_id,
            effective_content_type=effective_ct, intent_type=intent_type, entities=entities,
        )
        if chain is None:
            return chain_input

        response = self._safe_chain_invoke(chain, chain_input)
        self._record_generation_trace("llm", content_type=effective_ct, context_doc_count=len(context_docs))

        self.conversation_manager.add_interaction(
            session_id, completed_query, response,
            intent_type=intent_type, entities=entities or {}
        )
        return response

    def generate_basic_answer_stream_with_conversation(self, query, context_docs, session_id, intent_type="general", entities=None, content_type=None):
        """支持多轮对话的真流式生成方法（逐 token 输出）。"""
        yield from self._generate_conversation_answer_stream(query, context_docs, session_id, step_by_step=False, intent_type=intent_type, entities=entities, content_type=content_type)

    def generate_step_by_step_answer_stream_with_conversation(self, query, context_docs, session_id, intent_type="detail", entities=None, content_type=None):
        """支持多轮对话的真流式分步骤生成方法（逐 token 输出）。"""
        yield from self._generate_conversation_answer_stream(query, context_docs, session_id, step_by_step=True, intent_type=intent_type, entities=entities, content_type=content_type)

    def _build_context_with_conversation(self, docs: List[Document],
                                        conversation_context: str,
                                        max_length: int = 2500) -> str:
        """
        构建带多轮对话的上下文

        Args:
            docs: 文档列表
            conversation_context: 多轮对话历史
            max_length: 最大长度

        Returns:
            格式化的上下文字符串
        """
        # 先构建食谱上下文
        food_context = self._build_context(docs, max_length=1800)

        # 如果没有对话历史，直接返回食谱上下文
        if not conversation_context:
            return food_context
        
        # 拼接对话历史和食谱上下文
        full_context = f"""【对话历史】
{conversation_context}

【相关食谱信息】
{food_context}"""

        return full_context

    def generate_with_conversation(self, query, context_docs, session_id, intent_type="general", entities=None, content_type=None):
        """支持多轮对话的生成方法。"""
        return self._generate_conversation_answer(query, context_docs, session_id, step_by_step=False, intent_type=intent_type, entities=entities, content_type=content_type)

    def generate_step_by_step_with_conversation(self, query, context_docs, session_id, intent_type="detail", entities=None, content_type=None):
        """支持多轮对话的分步骤回答。"""
        return self._generate_conversation_answer(query, context_docs, session_id, step_by_step=True, intent_type=intent_type, entities=entities, content_type=content_type)

    def _build_targeted_prompt(self, content_type: str = None) -> ChatPromptTemplate:
        """根据内容类型构建针对性的 prompt（委托到 prompts）。"""
        return build_targeted_prompt(content_type)

    def get_conversation_context(self, session_id: str) -> str:
        """获取对话上下文（用于调试）"""
        if not self.conversation_manager:
            return ""
        return self.conversation_manager.get_conversation_context(session_id)

    def get_current_entity(self, session_id: str) -> str:
        """获取当前讨论的实体（用于调试）"""
        if not self.conversation_manager:
            return None
        return self.conversation_manager.get_current_entity(session_id)
