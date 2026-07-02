"""
索引构建模块
"""

import logging
import time
import numpy as np
from typing import List, Dict
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def get_optimal_device() -> str:
    """
    检测可用设备并返回最优设备

    Returns:
        'cuda' 如果有可用的英伟达GPU
        'cpu'  otherwise
    """
    try:
        import torch
        if torch.cuda.is_available():
            device = 'cuda'
            logger.info(f"检测到英伟达GPU: {torch.cuda.get_device_name(0)}")
            return device
        else:
            logger.info("未检测到英伟达GPU，使用CPU")
            return 'cpu'
    except ImportError:
        logger.info("PyTorch未安装或不支持CUDA，使用CPU")
        return 'cpu'


class IndexConstructionModule:
    """索引构建模块 - 负责向量化和索引构建"""

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5", index_save_path: str = "./vector_index",
                 enable_quantization: bool = False):
        """
        初始化索引构建模块

        Args:
            model_name: 嵌入模型名称
            index_save_path: 索引保存路径
            enable_quantization: 是否启用向量量化（默认启用）
        """
        self.model_name = model_name
        self.index_save_path = index_save_path
        self.embeddings = None
        self.vectorstore = None
        self.device = get_optimal_device()  # 自动检测最优设备
        self.enable_quantization = enable_quantization  # 量化标志
        self.index_stats = {}  # 索引统计信息
        self.setup_embeddings()

    def setup_embeddings(self):
        """初始化嵌入模型（自动使用最优设备，优先使用本地缓存）"""
        import os
        
        logger.info(f"正在初始化嵌入模型: {self.model_name} (设备: {self.device})")

        # 检查本地缓存路径
        local_model_path = self._find_local_model(self.model_name)
        
        if local_model_path:
            logger.info(f"找到本地模型: {local_model_path}")
            model_name_or_path = local_model_path
        else:
            model_name_or_path = self.model_name

        # 设置缓存目录
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")

        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name_or_path,
            cache_folder=cache_dir,
            model_kwargs={
                'device': self.device,
                'trust_remote_code': False
            },
            encode_kwargs={'normalize_embeddings': True}
        )

        logger.info(f"嵌入模型初始化完成，使用设备: {self.device}")
    
    def _find_local_model(self, model_name: str) -> str:
        """
        在本地缓存中查找模型
        
        Args:
            model_name: 模型名称（如 BAAI/bge-small-zh-v1.5）
        
        Returns:
            本地模型路径，如果未找到返回 None
        """
        import os
        
        # 转换模型名称为缓存目录格式
        cache_name = model_name.replace("/", "--")
        cache_path = os.path.join(
            os.path.expanduser("~"), 
            ".cache", 
            "huggingface", 
            "hub",
            f"models--{cache_name}",
            "snapshots"
        )
        
        if os.path.exists(cache_path):
            # 获取快照目录（只有一个）
            snapshots = [f for f in os.listdir(cache_path) if os.path.isdir(os.path.join(cache_path, f))]
            if snapshots:
                return os.path.join(cache_path, snapshots[0])
        
        return None
    
    def build_vector_index(self, chunks: List[Document]) -> FAISS:
        """
        构建向量索引（支持向量量化）

        Args:
            chunks: 文档块列表

        Returns:
            FAISS向量存储对象
        """
        logger.info("正在构建FAISS向量索引...")

        if not chunks:
            raise ValueError("文档块列表不能为空")

        # 记录开始时间
        start_time = time.time()

        # 构建FAISS向量存储
        self.vectorstore = FAISS.from_documents(
            documents=chunks,
            embedding=self.embeddings
        )

        build_time = time.time() - start_time

        # 获取向量维度
        vector_dim = self._get_vector_dimension()
        raw_memory_mb = len(chunks) * vector_dim * 4 / (1024 * 1024)  # float32

        # 记录统计信息
        self.index_stats = {
            "chunk_count": len(chunks),
            "vector_dimension": vector_dim,
            "device": self.device,
            "build_time": build_time,
            "raw_memory_mb": raw_memory_mb,
            "enable_quantization": self.enable_quantization
        }

        logger.info(f"向量索引构建完成，包含 {len(chunks)} 个向量，耗时 {build_time:.2f}s")
        logger.info(f"原始向量内存占用: {raw_memory_mb:.2f} MB (维度: {vector_dim})")

        # 应用量化
        if self.enable_quantization:
            quant_start = time.time()
            self._apply_scalar_quantization()
            quant_time = time.time() - quant_start

            quantized_memory_mb = len(chunks) * vector_dim / (1024 * 1024)  # int8
            compression_ratio = raw_memory_mb / quantized_memory_mb if quantized_memory_mb > 0 else 0

            self.index_stats["quantized_memory_mb"] = quantized_memory_mb
            self.index_stats["compression_ratio"] = compression_ratio
            self.index_stats["quantization_time"] = quant_time

            logger.info(f"量化后内存占用: {quantized_memory_mb:.2f} MB (压缩比: {compression_ratio:.1f}x, 耗时: {quant_time:.2f}s)")

        return self.vectorstore

    def _get_vector_dimension(self) -> int:
        """获取向量维度"""
        try:
            if self.vectorstore and hasattr(self.vectorstore, 'index') and self.vectorstore.index:
                return self.vectorstore.index.d
        except Exception:
            logger.debug("无法从 vectorstore 获取维度信息，使用默认值 512", exc_info=True)
            pass
        # 默认值（BGE small 是 512 维）
        return 512

    def _apply_scalar_quantization(self):
        """
        应用标量量化

        使用 FAISS ScalarQuantizer 将 float32 转换为 int8
        内存占用减少约 4 倍，检索速度提升
        """
        if not self.vectorstore or not hasattr(self.vectorstore, 'index'):
            logger.warning("无法获取向量索引，跳过量化")
            return

        try:
            import faiss
            import numpy as np

            index = self.vectorstore.index

            # 如果索引已经量化，跳过
            if isinstance(index, faiss.IndexFlat):
                # 检查是否是 Flat 索引（最基础的索引类型）
                pass  # 继续处理
            elif isinstance(index, faiss.IndexScalarQuantizer):
                logger.info("索引已经量化，跳过")
                return
            elif not isinstance(index, faiss.Index):
                logger.warning(f"索引类型 {type(index)} 不支持量化，跳过")
                return

            # 获取原始向量信息
            dim = index.d
            ntotal = index.ntotal

            if ntotal == 0:
                logger.warning("索引为空，跳过量化")
                return

            logger.info(f"正在应用标量量化 (float32 -> int8)，维度: {dim}，向量数: {ntotal}")

            # 方法1：尝试直接获取所有向量（对于 Flat 索引有效）
            try:
                # 对于 IndexFlat，可以直接获取原始向量
                if isinstance(index, faiss.IndexFlatL2) or isinstance(index, faiss.IndexFlatIP):
                    # 直接从 index 获取原始向量
                    vectors = np.zeros((ntotal, dim), dtype='float32')
                    index.reconstruct(0, vectors[0])  # 预热
                    
                    # 使用批量获取
                    for i in range(ntotal):
                        vectors[i] = index.reconstruct(i)
                else:
                    # 其他索引类型，使用老方法但更高效
                    vectors = self._get_all_vectors_efficiently(index, ntotal, dim)
            except Exception as e:
                logger.warning(f"批量获取向量失败: {e}，使用原始索引")
                return

            # 创建并训练量化器
            logger.info(f"正在量化 {ntotal} 个向量...")
            quantizer = faiss.ScalarQuantizer(dim, faiss.ScalarQuantizer.QT_8bit)
            quantizer.train(vectors)

            # 创建量化索引并添加向量
            quantized_index = faiss.IndexScalarQuantizer(dim, faiss.ScalarQuantizer.QT_8bit)
            quantized_index.add(vectors)

            # 替换原始索引
            self.vectorstore.index = quantized_index

            quantized_memory_mb = ntotal * dim / (1024 * 1024)
            logger.info(f"标量量化应用成功，量化后内存: {quantized_memory_mb:.2f} MB")

        except ImportError:
            logger.warning("faiss 原生模块不可用，跳过量化（仅安装 faiss-cpu）")
        except Exception as e:
            logger.warning(f"量化失败: {e}，使用原始索引")

    def _get_all_vectors_efficiently(self, index, ntotal: int, dim: int) -> np.ndarray:
        """
        高效获取所有向量

        Args:
            index: FAISS 索引
            ntotal: 向量数量
            dim: 向量维度

        Returns:
            numpy 数组，形状为 (ntotal, dim)
        """
        import numpy as np

        vectors = np.zeros((ntotal, dim), dtype='float32')

        # 分批获取，每批100个向量
        batch_size = 100
        for start in range(0, ntotal, batch_size):
            end = min(start + batch_size, ntotal)
            for i in range(start, end):
                try:
                    vectors[i] = index.reconstruct(i)
                except Exception:
                    # 如果单个获取失败，尝试用零填充
                    logger.warning(f"向量重建构: reconstruct({i}) 失败，用零向量填充")
                    vectors[i] = np.zeros(dim, dtype='float32')

            if start % 500 == 0:
                logger.info(f"获取向量进度: {end}/{ntotal}")

        return vectors
    
    def add_documents(self, new_chunks: List[Document]):
        """
        向现有索引添加新文档
        
        Args:
            new_chunks: 新的文档块列表
        """
        if not self.vectorstore:
            raise ValueError("请先构建向量索引")
        
        logger.info(f"正在添加 {len(new_chunks)} 个新文档到索引...")
        self.vectorstore.add_documents(new_chunks)
        logger.info("新文档添加完成")

    def save_index(self):
        """
        保存向量索引到配置的路径
        """
        if not self.vectorstore:
            raise ValueError("请先构建向量索引")

        # 确保保存目录存在
        Path(self.index_save_path).mkdir(parents=True, exist_ok=True)

        self.vectorstore.save_local(self.index_save_path)
        logger.info(f"向量索引已保存到: {self.index_save_path}")
    
    def load_index(self):
        """
        从配置的路径加载向量索引

        Returns:
            加载的向量存储对象，如果加载失败返回None
        """
        if not self.embeddings:
            self.setup_embeddings()

        if not Path(self.index_save_path).exists():
            logger.info(f"索引路径不存在: {self.index_save_path}")
            return None

        try:
            self.vectorstore = FAISS.load_local(
                self.index_save_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            logger.info(f"向量索引已从 {self.index_save_path} 加载")
            return self.vectorstore
        except Exception as e:
            logger.warning(f"加载向量索引失败: {e}")
            return None
    
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """
        相似度搜索

        Args:
            query: 查询文本
            k: 返回结果数量

        Returns:
            相似文档列表
        """
        if not self.vectorstore:
            raise ValueError("请先构建或加载向量索引")

        return self.vectorstore.similarity_search(query, k=k)

    def get_index_stats(self) -> Dict:
        """
        获取索引统计信息

        Returns:
            包含索引统计信息的字典
        """
        stats = {
            "has_vectorstore": self.vectorstore is not None,
            "enable_quantization": self.enable_quantization,
            "model_name": self.model_name,
            "device": self.device,
        }

        # 从 index_stats 获取构建时的统计
        if self.index_stats:
            stats.update(self.index_stats)

        # 实时获取索引信息
        if self.vectorstore and hasattr(self.vectorstore, 'index') and self.vectorstore.index:
            index = self.vectorstore.index
            stats["vector_count"] = index.ntotal if hasattr(index, 'ntotal') else 0
            stats["vector_dimension"] = index.d if hasattr(index, 'd') else "unknown"

            # 检查是否量化索引
            try:
                import faiss
                stats["is_quantized"] = isinstance(index, faiss.IndexScalarQuantizer)
            except ImportError:
                stats["is_quantized"] = False
        else:
            stats["vector_count"] = 0
            stats["vector_dimension"] = "unknown"
            stats["is_quantized"] = False

        return stats
