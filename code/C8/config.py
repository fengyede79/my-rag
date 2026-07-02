"""
RAG系统配置文件
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ENV_FILE_PATH = Path(__file__).with_name(".env")
DEFAULT_DATA_PATH = PROJECT_ROOT / "my-rag" / "data" / "C8" / "cook"
DEFAULT_INDEX_PATH = Path(__file__).resolve().parent / "vector_index"


def _parse_bool(value: Any, default: bool) -> bool:
    """将环境变量中的布尔值转换为 Python bool。"""
    if value is None:
        return default
    if isinstance(value, bool):
        return value

    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False

    raise ValueError(f"无法解析布尔值: {value}")


def _parse_int(value: Any, default: int) -> int:
    """将环境变量中的整数值转换为 Python int。"""
    if value is None or value == "":
        return default
    return int(value)


def _parse_float(value: Any, default: float) -> float:
    """将环境变量中的浮点值转换为 Python float。"""
    if value is None or value == "":
        return default
    return float(value)


def _resolve_path(value: Any, fallback: Path) -> str:
    """将路径统一解析为绝对路径字符串。"""
    raw_path = Path(value) if value else fallback
    return str(raw_path.expanduser().resolve())


@dataclass
class RAGConfig:
    """RAG系统配置类。"""

    # 路径配置
    data_path: str = str(DEFAULT_DATA_PATH)
    index_save_path: str = str(DEFAULT_INDEX_PATH)
    rebuild_index_if_missing: bool = True

    # 模型配置
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    llm_model: str = "qwen-turbo"

    # 检索配置
    top_k: int = 3

    # 生成配置
    temperature: float = 0.1
    max_tokens: int = 2048

    def __post_init__(self):
        """初始化后的标准化处理。"""
        self.data_path = _resolve_path(self.data_path, DEFAULT_DATA_PATH)
        self.index_save_path = _resolve_path(self.index_save_path, DEFAULT_INDEX_PATH)
        self.embedding_model = str(self.embedding_model).strip()
        self.llm_model = str(self.llm_model).strip()
        self.validate()

    def validate(self):
        """校验配置合法性。"""
        if not self.embedding_model:
            raise ValueError("embedding_model 不能为空")
        if not self.llm_model:
            raise ValueError("llm_model 不能为空")
        if self.top_k <= 0:
            raise ValueError("top_k 必须大于 0")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens 必须大于 0")
        if not 0 <= self.temperature <= 2:
            raise ValueError("temperature 必须在 0 到 2 之间")

    def index_exists(self) -> bool:
        """判断索引目录是否存在。"""
        return Path(self.index_save_path).exists()

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return {
            "data_path": self.data_path,
            "index_save_path": self.index_save_path,
            "rebuild_index_if_missing": self.rebuild_index_if_missing,
            "embedding_model": self.embedding_model,
            "llm_model": self.llm_model,
            "top_k": self.top_k,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "RAGConfig":
        """从字典创建配置对象。"""
        return cls(**config_dict)

    @classmethod
    def from_env(
        cls,
        environ: Optional[Mapping[str, str]] = None,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> "RAGConfig":
        """
        从环境变量与覆盖参数中创建配置对象。

        优先级：
        1. overrides
        2. 环境变量
        3. 类默认值
        """
        env = dict(environ or os.environ)
        override_values = overrides or {}

        defaults = cls()

        config_data = {
            "data_path": override_values.get(
                "data_path",
                env.get("RAG_DATA_PATH", defaults.data_path),
            ),
            "index_save_path": override_values.get(
                "index_save_path",
                env.get("RAG_INDEX_PATH", defaults.index_save_path),
            ),
            "rebuild_index_if_missing": override_values.get(
                "rebuild_index_if_missing",
                _parse_bool(
                    env.get("RAG_REBUILD_INDEX_IF_MISSING"),
                    defaults.rebuild_index_if_missing,
                ),
            ),
            "embedding_model": override_values.get(
                "embedding_model",
                env.get("RAG_EMBEDDING_MODEL", defaults.embedding_model),
            ),
            "llm_model": override_values.get(
                "llm_model",
                env.get("RAG_LLM_MODEL", defaults.llm_model),
            ),
            "top_k": override_values.get(
                "top_k",
                _parse_int(env.get("RAG_TOP_K"), defaults.top_k),
            ),
            "temperature": override_values.get(
                "temperature",
                _parse_float(env.get("RAG_TEMPERATURE"), defaults.temperature),
            ),
            "max_tokens": override_values.get(
                "max_tokens",
                _parse_int(env.get("RAG_MAX_TOKENS"), defaults.max_tokens),
            ),
        }
        return cls.from_dict(config_data)


def load_config(overrides: Optional[Dict[str, Any]] = None) -> RAGConfig:
    """统一加载配置：先读 .env，再解析环境变量和覆盖参数。"""
    load_dotenv(ENV_FILE_PATH)
    return RAGConfig.from_env(overrides=overrides)
