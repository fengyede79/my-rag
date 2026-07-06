"""
生成集成模块 - 编排查询路由、上下文构建和回答生成。
护栏、结构化回答、Prompt 模板、安全调用已拆分到独立模块。
"""

import os
import re
import logging
import threading
from typing import List, Dict, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from . import guardrail as _guardrail_module
from . import structured_generation as _structured_generation
from . import stream_handler as _stream_handler
from .prompts import (
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

    def generate_smalltalk_answer(self, query: str) -> str:
        """为闲聊轮次生成直接回复，不进入检索。"""
        normalized = query.strip().rstrip("?!？！。")
        if normalized in {"你好", "您好"}:
            return "你好，我可以帮你推荐菜、查做法，或者继续接着上一道菜聊。"
        if normalized == "谢谢":
            return "不客气，你可以继续问我吃什么、某道菜怎么做，或者食材怎么处理。"
        if normalized == "哈哈":
            return "那我们继续。你想让我推荐菜，还是直接查一道菜的做法？"
        if normalized == "你是谁":
            return "我是你的食谱助手，可以帮你推荐菜、查做法、查食材和烹饪技巧。"
        return "我在。你可以直接问我今天吃什么，或者某道菜怎么做。"

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
                return iter([response])
            return response

        structured_answer = self._try_build_structured_answer(query, context_docs, effective_ct)
        if structured_answer:
            self._record_generation_trace("structured", content_type=effective_ct, context_doc_count=len(context_docs))
            if stream:
                return iter([structured_answer])
            return structured_answer

        error_msg = self._check_dish_consistency(query, context_docs)
        if error_msg:
            self._record_generation_trace("consistency_blocked", content_type=effective_ct, context_doc_count=len(context_docs), reason="dish_mismatch")
            if stream:
                return iter([error_msg])
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
            return self._safe_chain_stream(chain, query)

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
    
    def reset_session_state(self, session_id: str):
        """
        清理当前会话的推荐缓存和对话状态

        Args:
            session_id: 会话ID
        """
        if self.conversation_manager:
            self.conversation_manager.reset_session(session_id)
    
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
        
        # 默认返回 general，避免把不确定问题误判成推荐类查询
        return "general", "fallback", {"confidence": 0.5}

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
        # 3.2 处理"XX怎么"、"XX怎么样"这类追问模式，保留完整菜名
        dish_name = re.sub(r'怎么(样)?$', '', dish_name)

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
