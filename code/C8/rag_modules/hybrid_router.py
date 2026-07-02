"""
混合路由模块 - 实现三层路由架构：规则路由 -> 语义路由 -> LLM路由
"""

import logging
import numpy as np
from typing import Dict, Tuple, List, Any, Callable

logger = logging.getLogger(__name__)

class HybridRouter:
    """
    三层混合路由：
    1. 规则路由（微秒级）- 处理高频、确定的查询
    2. 语义路由（毫秒级）- 处理规则覆盖不到的语义变体
    3. LLM路由（秒级）- 兜底处理复杂情况
    """
    
    def __init__(self, llm_model):
        """
        初始化混合路由
        
        Args:
            llm_model: LLM模型实例，用于最后的兜底路由
        """
        self.llm_model = llm_model
        self.sentence_model = None
        
        # 初始化各层路由
        self._init_rules()
        self._init_semantic_intents()
    
    def _init_rules(self):
        """初始化规则路由（优先级从高到低）"""
        # 规则格式: (优先级, 匹配函数, 目标content_type, 描述)
        self.rules = [
            # 食材相关
            (15, lambda q: "食材" in q or "材料" in q or "配料" in q or "原料" in q, "ingredients", "食材查询"),
            # 步骤相关
            (14, lambda q: "步骤" in q or "流程" in q, "steps", "步骤查询"),
            # 做法相关
            (13, lambda q: "做法" in q or "怎么做" in q or "制作方法" in q or "怎么弄" in q, "steps", "做法查询"),
            # 技巧相关
            (12, lambda q: "技巧" in q or "窍门" in q or "注意" in q or "小贴士" in q, "tips", "技巧查询"),
            # 介绍相关
            (11, lambda q: "介绍" in q or "简介" in q or "特点" in q or "特色" in q, "introduction", "介绍查询"),
            # 推荐相关（列表查询）
            (10, lambda q: "推荐" in q or "有什么" in q or "有哪些" in q or "给我" in q or "列出" in q, "list", "推荐查询"),
        ]
    
    def _init_semantic_intents(self):
        """初始化语义路由意图"""
        # 定义各content_type的语义变体示例
        self.intents = {
            "ingredients": [
                "需要什么材料", "要用什么", "准备什么", "用什么做",
                "材料清单", "需要准备", "买什么", "配料表",
                "需要哪些材料", "准备哪些东西", "需要准备什么"
            ],
            "steps": [
                "操作步骤", "操作流程", "制作过程", "如何制作",
                "怎么做的", "制作步骤", "烹饪步骤", "步骤是什么",
                "生产步骤", "加工步骤", "一步步教我", "详细步骤"
            ],
            "tips": [
                "有什么技巧", "注意事项", "温馨提示", "小窍门",
                "怎么做更好", "如何做得更好", "关键步骤", "秘诀",
                "有什么要注意的", "制作要点", "技巧分享"
            ],
            "introduction": [
                "这是什么菜", "菜品特点", "由来", "历史",
                "特色是什么", "风味特点", "口感如何", "菜品介绍"
            ],
            "list": [
                "帮我推荐", "推荐几个", "有什么好吃的", "吃什么好",
                "什么菜合适", "有什么推荐", "介绍几个菜", "今天吃什么"
            ]
        }
        
        # 延迟加载SentenceTransformer（仅在需要语义路由时加载）
        self._semantic_ready = False
    
    def _ensure_semantic_ready(self):
        """确保语义路由所需的模型已加载"""
        if not self._semantic_ready:
            try:
                from sentence_transformers import SentenceTransformer
                self.sentence_model = SentenceTransformer('BAAI/bge-small-zh-v1.5')
                
                # 预计算意图向量
                self.intent_embeddings = {}
                for intent, examples in self.intents.items():
                    self.intent_embeddings[intent] = self.sentence_model.encode(
                        examples, normalize_embeddings=True
                    )
                
                self._semantic_ready = True
                logger.info("语义路由模型加载完成")
            except Exception as e:
                logger.warning(f"加载语义路由模型失败: {e}，将跳过语义路由")
                self._semantic_ready = None
    
    def route(self, query: str) -> Tuple[str, str, Dict[str, Any]]:
        """
        执行三层路由
        
        Args:
            query: 用户查询
            
        Returns:
            (content_type, route_type, intent_info)
            - content_type: 匹配到的内容类型
            - route_type: 路由类型 ("rule", "semantic", "llm")
            - intent_info: 额外的意图信息
        """
        # 第一步：规则路由（最快，微秒级）
        result = self._rule_route(query)
        if result:
            content_type, route_type = result
            return content_type, route_type, {"confidence": 1.0}
        
        # 第二步：语义路由（次快，毫秒级）
        result = self._semantic_route(query)
        if result:
            content_type, route_type, similarity = result
            return content_type, route_type, {"confidence": similarity}
        
        # 第三步：LLM路由（最慢，秒级）- 只有当前面都匹配不到时才调用
        result = self._llm_route(query)
        return result, "llm", {"confidence": 0.7}
    
    def _rule_route(self, query: str) -> Tuple[str, str]:
        """
        规则路由 - 处理高频、确定的查询
        
        Returns:
            (content_type, "rule") 或 None
        """
        # 按优先级从高到低匹配
        for priority, matcher, content_type, desc in self.rules:
            if matcher(query):
                logger.debug(f"规则路由匹配成功: {desc} -> {content_type}")
                return content_type, "rule"
        return None
    
    def _semantic_route(self, query: str, threshold: float = 0.7) -> Tuple[str, str, float]:
        """
        语义路由 - 处理规则覆盖不到的语义变体
        
        Args:
            query: 用户查询
            threshold: 相似度阈值
            
        Returns:
            (content_type, "semantic", similarity) 或 None
        """
        # 确保语义模型已加载
        self._ensure_semantic_ready()
        if self._semantic_ready is None:
            return None
        
        try:
            # 编码查询
            query_embedding = self.sentence_model.encode(query, normalize_embeddings=True)
            
            # 计算与各意图的相似度
            max_similarity = 0.0
            best_intent = None
            
            for intent, embeddings in self.intent_embeddings.items():
                similarity = np.max(query_embedding @ embeddings.T)
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_intent = intent
            
            # 判断是否超过阈值
            if best_intent and max_similarity > threshold:
                logger.debug(f"语义路由匹配成功: {best_intent} (相似度: {max_similarity:.4f})")
                return best_intent, "semantic", float(max_similarity)
            
            logger.debug(f"语义路由未匹配（最高相似度: {max_similarity:.4f} < {threshold}）")
            return None
        except Exception as e:
            logger.warning(f"语义路由处理失败: {e}")
            return None
    
    def _llm_route(self, query: str) -> str:
        """
        LLM路由 - 兜底处理复杂情况
        
        Returns:
            content_type
        """
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        
        prompt = ChatPromptTemplate.from_template("""
请分析用户查询，判断用户需要的内容类型：

可选的内容类型：
- ingredients: 用户询问食材、材料、配料、原料等
- steps: 用户询问步骤、做法、制作方法、操作流程等
- tips: 用户询问技巧、窍门、注意事项、小贴士等
- introduction: 用户询问介绍、简介、特点、特色等
- list: 用户想要菜品推荐、列表等

请只返回上述内容类型之一，不要其他内容。

用户查询：{query}

内容类型：""")
        
        chain = (
            {"query": lambda x: x}
            | prompt
            | self.llm_model
            | StrOutputParser()
        )
        
        try:
            result = chain.invoke(query).strip()
            
            # 验证结果是否有效
            valid_types = ["ingredients", "steps", "tips", "introduction", "list"]
            if result in valid_types:
                logger.debug(f"LLM路由匹配成功: {result}")
                return result
            else:
                logger.warning(f"LLM路由返回无效结果: {result}")
                return "list"
        except Exception as e:
            logger.error(f"LLM路由调用失败: {e}")
            return "list"
