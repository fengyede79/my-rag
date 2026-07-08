"""
检索优化模块
"""

import logging
import hashlib
from typing import List, Dict, Any

from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from .data_preparation import DataPreparationModule

logger = logging.getLogger(__name__)

class RetrievalOptimizationModule:
    """检索优化模块 - 负责混合检索和过滤"""
    
    def __init__(self, vectorstore: FAISS, chunks: List[Document]):
        """
        初始化检索优化模块
        
        Args:
            vectorstore: FAISS向量存储
            chunks: 文档块列表
        """
        self.vectorstore = vectorstore
        self.chunks = chunks
        self.last_search_trace = {}
        self.setup_retrievers()

    def _record_search_trace(
        self,
        query: str,
        filters: Dict[str, Any] = None,
        vector_candidates: List[Document] = None,
        bm25_candidates: List[Document] = None,
        reranked_candidates: List[Document] = None,
        final_candidates: List[Document] = None,
        strategy: str = None,
    ):
        """记录最近一次检索轨迹，供过程级诊断使用。"""
        self.last_search_trace = {
            "query": query,
            "filters": filters or {},
            "strategy": strategy,
            "vector_candidates": list(vector_candidates or []),
            "bm25_candidates": list(bm25_candidates or []),
            "reranked_candidates": list(reranked_candidates or []),
            "final_candidates": list(final_candidates or []),
        }

    def setup_retrievers(self):
        """设置向量检索器和BM25检索器"""
        logger.info("正在设置检索器...")

        # 向量检索器
        self.vector_retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )

        # BM25检索器
        self.bm25_retriever = BM25Retriever.from_documents(
            self.chunks,
            k=5
        )



        logger.info("检索器设置完成")
    
    def hybrid_search(self, query: str, top_k: int = 3, content_types: List[str] = None,
                      query_dish: str = None) -> List[Document]:
        """
        混合检索 - 结合向量检索和BM25检索，使用RRF重排

        Args:
            query: 查询文本
            top_k: 返回结果数量
            content_types: 内容类型过滤（可选）
            query_dish: 查询目标菜品名（可选，用于RRF加权boost）

        Returns:
            检索到的文档列表
        """
        # 分别获取向量检索和BM25检索结果
        vector_docs = self.vector_retriever.invoke(query)
        bm25_docs = self.bm25_retriever.invoke(query)

        # 使用RRF重排
        reranked_docs = self._rrf_rerank(vector_docs, bm25_docs, query_dish=query_dish)

        # 应用内容类型过滤
        if content_types:
            reranked_docs = [doc for doc in reranked_docs
                            if doc.metadata.get('content_type') in content_types]

        final_docs = reranked_docs[:top_k]
        self._record_search_trace(
            query=query,
            filters={"content_type": content_types} if content_types else {},
            vector_candidates=vector_docs,
            bm25_candidates=bm25_docs,
            reranked_candidates=reranked_docs,
            final_candidates=final_docs,
            strategy="hybrid_search",
        )
        return final_docs
    
    def metadata_filtered_search(self, query: str, filters: Dict[str, Any], top_k: int = 5,
                                 query_dish: str = None) -> List[Document]:
        """
        带元数据过滤的检索（支持降级策略）

        Args:
            query: 查询文本
            filters: 元数据过滤条件，支持以下关键字:
                - category: 菜品分类
                - difficulty: 难度等级
                - content_type: 内容类型（ingredients/steps/tips等）
                - contains_ingredient: 包含的食材（精确匹配）
                - exclude_ingredient: 排除的食材
                - dish_name: 菜品名称（精确匹配，失败后降级为模糊包含匹配）
            top_k: 返回结果数量
            query_dish: 查询目标菜品名（用于RRF加权boost，可与dish_name不同）

        Returns:
            过滤后的文档列表
        """
        # 避免修改调用方传入的过滤条件
        safe_filters = filters.copy() if isinstance(filters, dict) else {}

        # 提取 content_type 过滤（单独处理）
        content_types = safe_filters.pop('content_type', None)

        # 提取 dish_name 用于特殊处理
        dish_name_filter = safe_filters.get('dish_name')

        # 对明确菜品名查询优先在同名菜块中检索，避免被宽泛关键词带偏
        if dish_name_filter and isinstance(dish_name_filter, str):
            exact_dish_name = dish_name_filter.strip().replace(" ", "").replace("　", "")
            exact_hit_docs = self._search_exact_dish_chunks(
                query=query,
                exact_dish_name=exact_dish_name,
                content_types=content_types,
                top_k=top_k,
            )
            if exact_hit_docs:
                logger.info(f"[MetadataFilter] 命中同名菜品定向检索: {dish_name_filter}")
                self._record_search_trace(
                    query=query,
                    filters=filters,
                    vector_candidates=[],
                    bm25_candidates=[],
                    reranked_candidates=exact_hit_docs,
                    final_candidates=exact_hit_docs[:top_k],
                    strategy="exact_dish_chunks",
                )
                return exact_hit_docs

        # 第一阶段：精确过滤（将query_dish传给hybrid_search用于RRF boost）
        exact_filters = safe_filters.copy()
        docs = self.hybrid_search(query, top_k * 3, content_types, query_dish=query_dish or dish_name_filter)

        if dish_name_filter and isinstance(dish_name_filter, str):
            exact_dish_name = dish_name_filter.strip().replace(" ", "").replace("　", "")
            # 先做精确过滤
            filtered_docs = self._apply_metadata_filters(docs, exact_filters, require_dish_name_exact=True, exact_dish_name=exact_dish_name)
            
            if not filtered_docs:
                # 第二阶段：降级为模糊包含匹配
                logger.info(f"[MetadataFilter] dish_name精确过滤无结果，降级为模糊包含匹配: {dish_name_filter}")
                filtered_docs = self._apply_metadata_filters(docs, exact_filters, require_dish_name_exact=False, exact_dish_name=exact_dish_name)
                
                if not filtered_docs:
                    # 第三阶段：退化为无 dish_name 过滤（同时移除 content_type，避免双重约束导致空结果）
                    logger.warning(f"[MetadataFilter] dish_name模糊匹配也无结果，退化为无 dish_name/content_type 过滤")
                    filters_relaxed = {k: v for k, v in exact_filters.items()
                                       if k not in ('dish_name', 'content_type')}
                    filtered_docs = self._apply_metadata_filters(docs, filters_relaxed, require_dish_name_exact=False, exact_dish_name=None)
        else:
            filtered_docs = self._apply_metadata_filters(
                docs,
                safe_filters,
                require_dish_name_exact=False,
                exact_dish_name=None
            )

        final_docs = filtered_docs[:top_k]
        self._record_search_trace(
            query=query,
            filters=filters,
            vector_candidates=docs,
            bm25_candidates=[],
            reranked_candidates=docs,
            final_candidates=final_docs,
            strategy="metadata_filtered_search",
        )
        return final_docs

    def _search_exact_dish_chunks(self, query: str, exact_dish_name: str,
                                  content_types: List[str] = None, top_k: int = 5) -> List[Document]:
        """
        在完全同名的菜品块中做一次定向检索。

        这样“鸡蛋三明治需要什么食材”这类问题会优先命中同菜名文档，
        不会被“鸡”“蛋”等高频词带到别的菜上。
        """
        exact_candidates = []
        for doc in self.chunks:
            doc_dish_name = doc.metadata.get('dish_name', '').strip().replace(" ", "").replace("　", "")
            if doc_dish_name != exact_dish_name:
                continue
            if content_types and doc.metadata.get('content_type') not in content_types:
                continue
            exact_candidates.append(doc)

        if not exact_candidates:
            return []

        query_terms = [term for term in query.strip().split() if term]
        scored_candidates = []
        for doc in exact_candidates:
            score = 0
            content = doc.page_content
            for term in query_terms:
                if term in content:
                    score += 3
            if doc.metadata.get('content_type') == 'ingredients' and any(kw in query for kw in ['食材', '材料', '原料', '配料']):
                score += 5
            if doc.metadata.get('content_type') == 'steps' and any(kw in query for kw in ['步骤', '做法', '怎么做', '制作方法']):
                score += 5
            if doc.metadata.get('content_type') == 'tips' and any(kw in query for kw in ['技巧', '小贴士', '注意']):
                score += 5
            score += max(0, 3 - doc.metadata.get('chunk_index', 0)) * 0.1
            doc.metadata['rrf_score'] = float(score)
            scored_candidates.append((score, doc))

        scored_candidates.sort(key=lambda item: item[0], reverse=True)
        return [doc for _, doc in scored_candidates[:top_k]]

    def _apply_metadata_filters(self, docs: List[Document], filters: Dict[str, Any], require_dish_name_exact: bool = False, exact_dish_name: str = None) -> List[Document]:
        """
        应用元数据过滤的统一逻辑

        Args:
            docs: 待过滤文档列表
            filters: 过滤条件字典
            require_dish_name_exact: 是否要求 dish_name 精确匹配
            exact_dish_name: 精确匹配的菜品名（已去除空格）

        Returns:
            过滤后的文档列表
        """
        filtered_docs = []
        for doc in docs:
            match = True
            for key, value in filters.items():
                if key == 'contains_ingredient':
                    entities = doc.metadata.get('entities', {})
                    ingredients = entities.get('ingredients', [])
                    dish_name = doc.metadata.get('dish_name', '')
                    if value not in ingredients and value not in dish_name:
                        match = False
                        break
                elif key == 'exclude_ingredient':
                    entities = doc.metadata.get('entities', {})
                    ingredients = entities.get('ingredients', [])
                    dish_name = doc.metadata.get('dish_name', '')
                    if value in ingredients or value in dish_name:
                        match = False
                        break
                elif key == 'dish_name':
                    # 菜品名过滤：精确模式或模糊模式
                    doc_dish_name = doc.metadata.get('dish_name', '').strip().replace(" ", "").replace("　", "")
                    if require_dish_name_exact:
                        if doc_dish_name != exact_dish_name:
                            match = False
                            break
                    else:
                        # 模糊模式：子串包含匹配，带安全边界
                        if not doc_dish_name:
                            match = False
                            break
                        if exact_dish_name:
                            is_substring = exact_dish_name in doc_dish_name or doc_dish_name in exact_dish_name
                            shorter = min(len(exact_dish_name), len(doc_dish_name))
                            longer = max(len(exact_dish_name), len(doc_dish_name))
                            if not is_substring:
                                match = False
                                break
                            elif shorter < 2:
                                # 安全边界：单字不做子串匹配，防止 "鱼" 误匹配 "红烧鱼"
                                match = False
                                break
                            elif shorter / longer < 0.5:
                                # 子串占比不足 50%，拒绝（如 "红烧" vs "湖南家常红烧肉" 2/7=28%）
                                match = False
                                break
                            # else: 子串占比 ≥ 50% 且 ≥ 2字，放行
                elif key in doc.metadata:
                    if isinstance(value, list):
                        if doc.metadata[key] not in value:
                            match = False
                            break
                    else:
                        if doc.metadata[key] != value:
                            match = False
                            break
                else:
                    if key != 'entities':
                        match = False
                        break

            if match:
                filtered_docs.append(doc)

        return filtered_docs

    def _rrf_rerank(self, vector_docs: List[Document], bm25_docs: List[Document],
                    query_dish: str = None, k: int = 60) -> List[Document]:
        """
        使用RRF (Reciprocal Rank Fusion) 算法重排文档，支持菜品名加权与去重

        Args:
            vector_docs: 向量检索结果
            bm25_docs: BM25检索结果
            query_dish: 查询目标菜品名（可选，用于加权boost）
            k: RRF参数，用于平滑排名

        Returns:
            重排后的文档列表
        """
        doc_scores = {}
        doc_objects = {}

        # 向量检索权重（BM25权重的1.5倍）
        vector_weight = 1.5
        bm25_weight = 1.0

        # 计算向量检索结果的RRF分数
        for rank, doc in enumerate(vector_docs):
            doc_id = hashlib.md5(doc.page_content.encode('utf-8')).hexdigest()
            doc_objects[doc_id] = doc

            rrf_score = (1.0 / (k + rank + 1)) * vector_weight
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score

            logger.debug(f"向量检索 - 文档{rank+1}: RRF分数 = {rrf_score:.4f}")

        # 计算BM25检索结果的RRF分数
        for rank, doc in enumerate(bm25_docs):
            doc_id = hashlib.md5(doc.page_content.encode('utf-8')).hexdigest()
            doc_objects[doc_id] = doc

            rrf_score = (1.0 / (k + rank + 1)) * bm25_weight
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score

            logger.debug(f"BM25检索 - 文档{rank+1}: RRF分数 = {rrf_score:.4f}")

        # 菜品名匹配 boost（在校分数计算完成后进行）
        if query_dish:
            query_dish_clean = query_dish.strip().replace(" ", "").replace("　", "")
            for doc_id, doc in doc_objects.items():
                doc_dish = doc.metadata.get('dish_name', '').strip().replace(" ", "").replace("　", "")
                if doc_dish == query_dish_clean:
                    doc_scores[doc_id] *= 2.0
                    logger.debug(f"菜品名精确匹配 boost ×2.0: {doc_dish}")
                elif doc_dish and (query_dish_clean in doc_dish or doc_dish in query_dish_clean):
                    len_ratio = max(len(query_dish_clean), len(doc_dish)) / min(len(query_dish_clean), len(doc_dish))
                    if len_ratio <= 1.3:
                        doc_scores[doc_id] *= 1.5
                        logger.debug(f"菜品名包含匹配 boost ×1.5: {doc_dish}")

        # 按最终RRF分数排序
        sorted_doc_ids = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

        # 菜品级去重：同一道菜最多保留3个子块
        dish_seen_count = {}  # dish_name -> 已保留子块数
        reranked_docs = []
        for doc_id, final_score in sorted_doc_ids:
            if doc_id not in doc_objects:
                continue
            doc = doc_objects[doc_id]
            dish_name = doc.metadata.get('dish_name', '')

            # 检查是否超过同菜品子块上限
            if dish_name in dish_seen_count and dish_seen_count[dish_name] >= 3:
                logger.debug(f"跳过同菜品第{dish_seen_count[dish_name]+1}个子块: {doc.page_content[:30]}...")
                continue

            doc.metadata['rrf_score'] = final_score
            reranked_docs.append(doc)
            dish_seen_count[dish_name] = dish_seen_count.get(dish_name, 0) + 1

            logger.debug(f"最终排序 - 文档: {doc.page_content[:50]}... 最终RRF分数: {final_score:.4f}")

        logger.info(f"RRF重排完成: 向量检索{len(vector_docs)}个文档, BM25检索{len(bm25_docs)}个文档, "
                    f"合并后{len(reranked_docs)}个文档（去重后），共涉及{len(dish_seen_count)}道菜品")

        return reranked_docs

    def extract_filters_from_query(self, query: str) -> dict:
        """从用户问题中提取元数据过滤条件。

        Args:
            query: 用户查询

        Returns:
            过滤条件字典
        """
        filters = {}
        # 分类关键词
        category_keywords = DataPreparationModule.get_supported_categories()
        for cat in category_keywords:
            if cat in query:
                filters['category'] = cat
                break

        # 难度关键词
        difficulty_keywords = DataPreparationModule.get_supported_difficulties()
        for diff in sorted(difficulty_keywords, key=len, reverse=True):
            if diff in query:
                filters['difficulty'] = diff
                break

        # 如果已经是明显的“具体菜品详情”问题，就不要再额外加泛食材过滤，
        # 否则“鸡蛋三明治需要什么食材”会被“鸡”带偏到其他菜。
        detail_markers = ["怎么做", "怎么制作", "制作方法", "步骤", "做法", "食材", "材料", "原料", "配料", "技巧"]
        if any(marker in query for marker in detail_markers):
            return filters

        # 食材关键词 - 使用统一的食材关键词列表
        for ingredient in DataPreparationModule.INGREDIENT_KEYWORDS:
            if ingredient in query:
                filters['contains_ingredient'] = ingredient
                break

        return filters

