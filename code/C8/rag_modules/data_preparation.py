"""
数据准备模块
"""

import logging
import hashlib
import re
import pickle
from pathlib import Path
from typing import List, Dict, Any

from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_core.documents import Document
import uuid

logger = logging.getLogger(__name__)

class DataPreparationModule:
    """数据准备模块 - 负责数据加载、清洗和预处理"""
    # 缓存版本号，当数据结构变更时递增，避免加载不兼容的旧缓存
    CACHE_VERSION = 1
    # 统一维护的分类与难度配置，供外部复用，避免关键词重复定义
    CATEGORY_MAPPING = {
        'meat_dish': '荤菜',
        'vegetable_dish': '素菜',
        'soup': '汤品',
        'dessert': '甜品',
        'breakfast': '早餐',
        'staple': '主食',
        'aquatic': '水产',
        'condiment': '调料',
        'drink': '饮品'
    }
    CATEGORY_LABELS = ['荤菜', '素菜', '汤品', '甜品', '早餐', '主食', '水产', '调料', '饮品']
    DIFFICULTY_LABELS = ['非常简单', '简单', '中等', '困难', '非常困难']
    # 统一的食材关键词列表，供实体提取和查询过滤共用
    INGREDIENT_KEYWORDS = [
        '鱼', '蟹', '虾', '五花肉', '鸡肉', '牛肉', '鱼肉', '鸡蛋', '豆腐',
        '青菜', '白菜', '番茄', '土豆', '胡萝卜', '洋葱',
        '青椒', '红椒', '大蒜', '生姜', '葱', '料酒',
        '生抽', '老抽', '盐', '糖', '油', '醋', '辣椒',
        '花椒', '八角', '桂皮', '香叶', '豆瓣酱',
    ]
    
    def __init__(self, data_path: str, cache_path: str = "./vector_index"):
        """
        初始化数据准备模块
        
        Args:
            data_path: 数据文件夹路径
            cache_path: 缓存路径，用于保存/加载分块结果
        """
        self.data_path = data_path
        self.cache_path = cache_path
        self.documents: List[Document] = []  # 父文档（完整食谱）
        self.chunks: List[Document] = []     # 子文档（按标题分割的小块）
        self.parent_child_map: Dict[str, str] = {}  # 子块ID -> 父文档ID的映射
        self._parent_docs_by_id: Dict[str, Document] = {}  # parent_id -> 父文档的快速索引
    
    def load_documents(self) -> List[Document]:
        """
        加载文档数据
        
        Returns:
            加载的文档列表
        """
        logger.info(f"正在从 {self.data_path} 加载文档...")
        
        # 直接读取Markdown文件以保持原始格式
        documents = []
        data_path_obj = Path(self.data_path)

        for md_file in data_path_obj.rglob("*.md"):
            try:
                # 直接读取文件内容，保持Markdown格式
                with open(md_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                # 为每个父文档分配确定性的唯一ID（基于数据根目录的相对路径）
                try:
                    data_root = Path(self.data_path).resolve()
                    relative_path = Path(md_file).resolve().relative_to(data_root).as_posix()
                except Exception:
                    logger.warning(f"路径解析失败，使用文件名作为相对路径: {md_file}", exc_info=True)
                    relative_path = Path(md_file).as_posix()
                parent_id = hashlib.md5(relative_path.encode("utf-8")).hexdigest()

                # 创建Document对象
                doc = Document(
                    page_content=content,
                    metadata={
                        "source": str(md_file),
                        "parent_id": parent_id,
                        "doc_type": "parent"  # 标记为父文档
                    }
                )
                documents.append(doc)

            except Exception as e:
                logger.warning(f"读取文件 {md_file} 失败: {e}")
        
        # 增强文档元数据
        for doc in documents:
            self._enhance_metadata(doc)
        
        self.documents = documents
        # 构建 parent_id -> Document 的快速索引，避免后续线性扫描
        self._parent_docs_by_id = {
            doc.metadata.get("parent_id"): doc
            for doc in documents
            if doc.metadata.get("parent_id")
        }
        logger.info(f"成功加载 {len(documents)} 个文档")
        return documents
    
    def _enhance_metadata(self, doc: Document):
        """
        增强文档元数据
        
        Args:
            doc: 需要增强元数据的文档
        """
        file_path = Path(doc.metadata.get('source', ''))
        path_parts = file_path.parts
        
        # 提取菜品分类
        doc.metadata['category'] = '其他'
        for key, value in self.CATEGORY_MAPPING.items():
            if key in path_parts:
                doc.metadata['category'] = value
                break
        
        # 提取菜品名称
        doc.metadata['dish_name'] = file_path.stem

        # 分析难度等级
        content = doc.page_content
        if '★★★★★' in content:
            doc.metadata['difficulty'] = '非常困难'
        elif '★★★★' in content:
            doc.metadata['difficulty'] = '困难'
        elif '★★★' in content:
            doc.metadata['difficulty'] = '中等'
        elif '★★' in content:
            doc.metadata['difficulty'] = '简单'
        elif '★' in content:
            doc.metadata['difficulty'] = '非常简单'
        else:
            doc.metadata['difficulty'] = '未知'

    @classmethod
    def get_supported_categories(cls) -> List[str]:
        """对外提供支持的分类标签列表"""
        return cls.CATEGORY_LABELS

    @classmethod
    def get_supported_difficulties(cls) -> List[str]:
        """对外提供支持的难度标签列表"""
        return cls.DIFFICULTY_LABELS
    
    def chunk_documents(self) -> List[Document]:
        """
        Markdown结构感知分块

        Returns:
            分块后的文档列表
        """
        logger.info("正在进行Markdown结构感知分块...")

        if not self.documents:
            raise ValueError("请先加载文档")

        # 使用Markdown标题分割器
        chunks = self._markdown_header_split()

        # 为每个chunk添加基础元数据
        for i, chunk in enumerate(chunks):
            if 'chunk_id' not in chunk.metadata:
                # 如果没有chunk_id（比如分割失败的情况），则生成一个
                chunk.metadata['chunk_id'] = str(uuid.uuid4())
            chunk.metadata['batch_index'] = i  # 在当前批次中的索引
            chunk.metadata['chunk_size'] = len(chunk.page_content)

        self.chunks = chunks
        logger.info(f"Markdown分块完成，共生成 {len(chunks)} 个chunk")
        return chunks

    def _markdown_header_split(self) -> List[Document]:
        """
        使用Markdown标题分割器进行结构化分割

        Returns:
            按标题结构分割的文档列表
        """
        # 定义要分割的标题层级
        headers_to_split_on = [
            ("#", "主标题"),      # 菜品名称
            ("##", "二级标题"),   # 必备原料、计算、操作等
            ("###", "三级标题")   # 简易版本、复杂版本等
        ]

        # 创建Markdown分割器
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False  # 保留标题，便于理解上下文
        )

        all_chunks = []

        for doc in self.documents:
            try:
                # 检查文档内容是否包含Markdown标题
                content_preview = doc.page_content[:200]
                has_headers = any(line.strip().startswith('#') for line in content_preview.split('\n'))

                if not has_headers:
                    logger.warning(f"文档 {doc.metadata.get('dish_name', '未知')} 内容中没有发现Markdown标题")
                    logger.debug(f"内容预览: {content_preview}")

                # 对每个文档进行Markdown分割
                md_chunks = markdown_splitter.split_text(doc.page_content)

                logger.debug(f"文档 {doc.metadata.get('dish_name', '未知')} 分割成 {len(md_chunks)} 个chunk")

                # 如果没有分割成功，说明文档可能没有标题结构
                if len(md_chunks) <= 1:
                    logger.warning(f"文档 {doc.metadata.get('dish_name', '未知')} 未能按标题分割，可能缺少标题结构")

                # 为每个子块建立与父文档的关系
                parent_id = doc.metadata["parent_id"]

                for i, chunk in enumerate(md_chunks):
                    # 为子块分配唯一ID
                    child_id = str(uuid.uuid4())

                    # 合并原文档元数据和新的标题元数据
                    chunk.metadata.update(doc.metadata)
                    chunk.metadata.update({
                        "chunk_id": child_id,
                        "parent_id": parent_id,
                        "doc_type": "child",  # 标记为子文档
                        "chunk_index": i      # 在父文档中的位置
                    })

                    # 增强：提取内容类型
                    self._extract_content_type(chunk)

                    # 增强：提取关键实体
                    self._extract_key_entities(chunk)

                    # 建立父子映射关系
                    self.parent_child_map[child_id] = parent_id

                all_chunks.extend(md_chunks)

            except Exception as e:
                logger.warning(f"文档 {doc.metadata.get('source', '未知')} Markdown分割失败: {e}")
                # 如果Markdown分割失败，将整个文档作为一个chunk
                all_chunks.append(doc)

        logger.info(f"Markdown结构分割完成，生成 {len(all_chunks)} 个结构化块")
        return all_chunks

    def _extract_content_type(self, chunk: Document):
        """
        提取chunk的内容类型
        
        Args:
            chunk: 文档块
        """
        content = chunk.page_content
        headers = chunk.metadata.get('Header', '')
        
        # 定义内容类型关键词映射
        content_type_mapping = {
            'ingredients': ['必备原料', '原料', '食材', '材料', '配料', '主料', '辅料'],
            'steps': ['操作', '步骤', '做法', '制作', '烹饪', '流程'],
            'calculation': ['计算', '用量', '份量', '配比', '比例'],
            'tips': ['附加内容', '小技巧', '小贴士', '注意事项', '温馨提示'],
            'introduction': ['介绍', '简介', '特点', '特色', '历史'],
            'nutrition': ['营养', '热量', '营养价值', '卡路里']
        }
        
        # 检查标题
        if headers:
            for content_type, keywords in content_type_mapping.items():
                if any(keyword in headers for keyword in keywords):
                    chunk.metadata['content_type'] = content_type
                    return
        
        # 检查内容
        for content_type, keywords in content_type_mapping.items():
            if any(keyword in content for keyword in keywords):
                chunk.metadata['content_type'] = content_type
                return
        
        # 默认类型
        chunk.metadata['content_type'] = 'general'

    def _extract_key_entities(self, chunk: Document):
        """
        提取chunk中的关键实体
        
        Args:
            chunk: 文档块
        """
        content = chunk.page_content
        entities = {}
        
        # 提取食材（使用统一的食材关键词列表）
        found_ingredients = [kw for kw in self.INGREDIENT_KEYWORDS if kw in content]
        if found_ingredients:
            entities['ingredients'] = found_ingredients
        
        # 提取烹饪时间（匹配数字+时间单位）
        time_matches = re.findall(r'(\d+)\s*(分钟|小时|秒)', content)
        if time_matches:
            cooking_time = f"{time_matches[0][0]}{time_matches[0][1]}"
            entities['cooking_time'] = cooking_time
        
        # 提取份量（匹配数字+人份）
        serving_matches = re.findall(r'(\d+)\s*(人份|人)', content)
        if serving_matches:
            entities['servings'] = f"{serving_matches[0][0]}人份"
        
        # 添加到metadata
        if entities:
            chunk.metadata['entities'] = entities

    def filter_documents_by_category(self, category: str) -> List[Document]:
        """
        按分类过滤文档
        
        Args:
            category: 菜品分类
            
        Returns:
            过滤后的文档列表
        """
        return [doc for doc in self.documents if doc.metadata.get('category') == category]
    
    def filter_documents_by_difficulty(self, difficulty: str) -> List[Document]:
        """
        按难度过滤文档
        
        Args:
            difficulty: 难度等级
            
        Returns:
            过滤后的文档列表
        """
        return [doc for doc in self.documents if doc.metadata.get('difficulty') == difficulty]
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取数据统计信息

        Returns:
            统计信息字典
        """
        if not self.documents:
            return {}

        categories = {}
        difficulties = {}

        for doc in self.documents:
            # 统计分类
            category = doc.metadata.get('category', '未知')
            categories[category] = categories.get(category, 0) + 1

            # 统计难度
            difficulty = doc.metadata.get('difficulty', '未知')
            difficulties[difficulty] = difficulties.get(difficulty, 0) + 1

        return {
            'total_documents': len(self.documents),
            'total_chunks': len(self.chunks),
            'categories': categories,
            'difficulties': difficulties,
            'avg_chunk_size': sum(chunk.metadata.get('chunk_size', 0) for chunk in self.chunks) / len(self.chunks) if self.chunks else 0
        }
    
    def export_metadata(self, output_path: str):
        """
        导出元数据到JSON文件
        
        Args:
            output_path: 输出文件路径
        """
        import json
        
        metadata_list = []
        for doc in self.documents:
            metadata_list.append({
                'source': doc.metadata.get('source'),
                'dish_name': doc.metadata.get('dish_name'),
                'category': doc.metadata.get('category'),
                'difficulty': doc.metadata.get('difficulty'),
                'content_length': len(doc.page_content)
            })
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metadata_list, f, ensure_ascii=False, indent=2)
        
        logger.info(f"元数据已导出到: {output_path}")

    def get_parent_documents(self, child_chunks: List[Document], target_dish_name: str = None) -> List[Document]:
        """
        根据子块获取对应的父文档（智能去重）

        Args:
            child_chunks: 检索到的子块列表
            target_dish_name: 目标菜品名称（可选，用于过滤不相关的父文档）

        Returns:
            对应的父文档列表（去重，按相关性排序）
        """
        # 统计每个父文档被匹配的次数（相关性指标）
        parent_relevance = {}
        parent_docs_map = {}

        # 收集所有相关的父文档ID和相关性分数
        for chunk in child_chunks:
            parent_id = chunk.metadata.get("parent_id")
            if parent_id:
                # 增加相关性计数
                parent_relevance[parent_id] = parent_relevance.get(parent_id, 0) + 1

                # 使用索引字典快速查找父文档（O(1) 而非 O(n)）
                if parent_id not in parent_docs_map:
                    doc = self._parent_docs_by_id.get(parent_id)
                    if doc:
                        parent_docs_map[parent_id] = doc

        # 构建去重后的父文档列表，可选过滤目标菜品
        if target_dish_name:
            target = target_dish_name.strip().replace(" ", "").replace("　", "")
            exact_match = []
            include_match = []

            for parent_id, doc in parent_docs_map.items():
                doc_name = doc.metadata.get('dish_name', '').strip().replace(" ", "").replace("　", "")

                # 1. 精确匹配：完全相等，优先级最高
                if doc_name == target:
                    exact_match.append((parent_id, doc))
                # 2. 包含匹配：一方完全包含另一方，且长度差异不超过30%
                #    防止"咖喱肥牛"匹配"咖喱肥牛饭"这类跨菜品误匹配
                elif target in doc_name or doc_name in target:
                    len_ratio = max(len(target), len(doc_name)) / min(len(target), len(doc_name))
                    if len_ratio <= 1.3:
                        include_match.append((parent_id, doc))
                # 3. 其余全部丢弃，绝不宽松匹配

            # 有精确匹配就只返回精确匹配，有包含匹配就只返回包含匹配
            # 各类别内部按relevance排序
            exact_match.sort(key=lambda x: parent_relevance.get(x[0], 0), reverse=True)
            include_match.sort(key=lambda x: parent_relevance.get(x[0], 0), reverse=True)

            if exact_match:
                parent_docs = [doc for _, doc in exact_match]
            elif include_match:
                parent_docs = [doc for _, doc in include_match]
            else:
                parent_docs = []

            # 空结果告警
            if not parent_docs:
                logger.warning(f"[ParentDocFilter] 目标菜品'{target_dish_name}'过滤后无匹配父文档，child_chunks来源: {[c.metadata.get('parent_id') for c in child_chunks]}")
        else:
            # 无target_dish_name时，按原逻辑返回所有父文档
            sorted_parent_ids = sorted(parent_relevance.keys(),
                                     key=lambda x: parent_relevance[x],
                                     reverse=True)
            parent_docs = [parent_docs_map[pid] for pid in sorted_parent_ids if pid in parent_docs_map]

        # 收集父文档名称和相关性信息用于日志
        parent_info = []
        for doc in parent_docs:
            dish_name = doc.metadata.get('dish_name', '未知菜品')
            parent_id = doc.metadata.get('parent_id')
            relevance_count = parent_relevance.get(parent_id, 0)
            parent_info.append(f"{dish_name}({relevance_count}块)")

        logger.info(f"从 {len(child_chunks)} 个子块中找到 {len(parent_docs)} 个去重父文档: {', '.join(parent_info)}")
        return parent_docs
    
    def save_chunks(self):
        """
        保存分块结果到缓存路径（带版本号和完整性校验）
        """
        if not self.chunks:
            logger.warning("没有分块数据可保存")
            return
        
        # 确保缓存目录存在
        Path(self.cache_path).mkdir(parents=True, exist_ok=True)
        
        # 保存分块数据（包含版本号和校验哈希）
        cache_data = {
            'version': self.CACHE_VERSION,
            'documents': self.documents,
            'chunks': self.chunks,
            'parent_child_map': self.parent_child_map
        }
        
        chunks_file = Path(self.cache_path) / "chunks.pkl"
        try:
            # 先序列化到内存，计算校验哈希
            serialized = pickle.dumps(cache_data)
            checksum = hashlib.md5(serialized).hexdigest()
            cache_data['checksum'] = checksum
            # 重新序列化（包含 checksum 字段）
            serialized = pickle.dumps(cache_data)

            with open(chunks_file, 'wb') as f:
                f.write(serialized)
            logger.info(f"分块数据已保存到: {chunks_file} (版本={self.CACHE_VERSION}, md5={checksum[:8]}...)")
        except Exception as e:
            logger.warning(f"保存分块数据失败: {e}")
    
    def load_chunks(self) -> bool:
        """
        从缓存路径加载分块结果（校验版本号和完整性）
        
        Returns:
            True 如果加载成功，False 否则
        """
        chunks_file = Path(self.cache_path) / "chunks.pkl"
        
        if not chunks_file.exists():
            logger.info(f"分块缓存文件不存在: {chunks_file}")
            return False
        
        try:
            with open(chunks_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            # 检查版本号
            cached_version = cache_data.get('version', 0)
            if cached_version != self.CACHE_VERSION:
                logger.warning(
                    f"缓存版本不匹配 (缓存={cached_version}, 当前={self.CACHE_VERSION})，"
                    "将重新构建索引"
                )
                return False

            # 校验完整性
            stored_checksum = cache_data.get('checksum')
            if stored_checksum:
                # 移除 checksum 字段后重新计算哈希
                verify_data = {k: v for k, v in cache_data.items() if k != 'checksum'}
                computed_checksum = hashlib.md5(pickle.dumps(verify_data)).hexdigest()
                if computed_checksum != stored_checksum:
                    logger.warning(
                        f"缓存完整性校验失败 (期望={stored_checksum[:8]}..., "
                        f"实际={computed_checksum[:8]}...)，将重新构建索引"
                    )
                    return False

            self.documents = cache_data.get('documents', [])
            self.chunks = cache_data.get('chunks', [])
            self.parent_child_map = cache_data.get('parent_child_map', {})
            # 重建 parent_id -> Document 快速索引
            self._parent_docs_by_id = {
                doc.metadata.get("parent_id"): doc
                for doc in self.documents
                if doc.metadata.get("parent_id")
            }
            
            logger.info(f"成功加载 {len(self.documents)} 个文档和 {len(self.chunks)} 个分块 (版本={cached_version})")
            return True
        except Exception as e:
            logger.warning(f"加载分块数据失败: {e}")
            return False
