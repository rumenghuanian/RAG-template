"""通用 RAG 配置 - 从环境变量读取"""
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict


def _env_bool(key: str, default: bool) -> bool:
    v = os.getenv(key)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


@dataclass
class RAGConfig:
    # 路径
    data_path: str = field(default_factory=lambda: os.getenv("DATA_PATH", "./data"))
    index_save_path: str = field(
        default_factory=lambda: os.getenv("INDEX_SAVE_PATH", "./vector_index")
    )
    file_glob: str = field(default_factory=lambda: os.getenv("FILE_GLOB", "*.md"))

    # Embedding
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
    )
    embedding_device: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_DEVICE", "cpu")
    )

    # LLM（通用 OpenAI 兼容）
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "gpt-4o-mini"))
    llm_base_url: str = field(
        default_factory=lambda: os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
    )
    llm_api_key: str = field(default_factory=lambda: os.getenv("LLM_API_KEY", ""))

    # 检索
    top_k: int = field(default_factory=lambda: int(os.getenv("TOP_K", "3")))
    enable_rewrite: bool = field(
        default_factory=lambda: _env_bool("ENABLE_REWRITE", True)
    )
    # 相关性闸门（余弦相似度）。**它只能过滤「完全跑题」，不能判断相关性**：
    # 41 条评测实测有答案问题最低 0.694、语料外问题最高 0.781，区间重叠，
    # 不存在能分开两者的阈值。取 0.45 是为了拦掉「今天天气」这类明显跑题的，
    # 同时不误伤「第 5 周讲了什么」这类结构性提问（实测 0.491）。
    min_relevance: float = field(
        default_factory=lambda: float(os.getenv("MIN_RELEVANCE", "0.45"))
    )

    # 重排序（cross-encoder）。开启后才加载模型；权重缺失/依赖缺失会自动退化。
    # 实测：gold 从未漏召，14% 的损失全在「排不进前 5」，所以重排的收益点在这里。
    rerank_enabled: bool = field(
        default_factory=lambda: _env_bool("RERANK_ENABLED", True)
    )
    rerank_model: str = field(
        default_factory=lambda: os.getenv("RERANK_MODEL", "BAAI/bge-reranker-base")
    )
    # 重排池上限。**默认 0 = 不截断，重排 RRF 给出的整个候选池**（实测 88~95 条）。
    # 不截断是有理由的：重排的价值就在于修 RRF 尾部的排序错误，先截断再重排等于
    # 让它够不着要修的地方（实测截到 50 会让 2/38 篇 gold 连候选都进不去）。
    # 想省时间可以设成 20~50，代价是候选变少。成本几乎正比于池深：
    # GPU 15ms/条（整池约 1.2s），CPU 266ms/条（整池约 24s，基本不可用）。
    rerank_candidates: int = field(
        default_factory=lambda: int(os.getenv("RERANK_CANDIDATES", "0"))
    )
    # 重排序跑在哪个设备。留空则跟随 EMBEDDING_DEVICE。
    rerank_device: str = field(
        default_factory=lambda: os.getenv("RERANK_DEVICE")
        or os.getenv("EMBEDDING_DEVICE", "cpu")
    )

    # 生成
    temperature: float = field(
        default_factory=lambda: float(os.getenv("TEMPERATURE", "0.1"))
    )
    max_tokens: int = field(
        default_factory=lambda: int(os.getenv("MAX_TOKENS", "2048"))
    )
    # 送进 prompt 的上下文预算（约等于 token 数）。要小于模型窗口，
    # 也要大于 TOP_K × 单篇平均长度，否则会白丢检索结果。
    context_max_tokens: int = field(
        default_factory=lambda: int(os.getenv("CONTEXT_MAX_TOKENS", "6000"))
    )

    def validate(self):
        if not self.llm_api_key:
            raise ValueError("请在 .env 中设置 LLM_API_KEY")
        if not self.llm_base_url:
            raise ValueError("请在 .env 中设置 LLM_BASE_URL")
        if not os.path.isdir(self.data_path):
            raise FileNotFoundError(f"数据目录不存在: {self.data_path}")
        if not any(Path(self.data_path).rglob(self.file_glob)):
            raise FileNotFoundError(
                f"数据目录里没有匹配 {self.file_glob} 的文件: {self.data_path}"
            )

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RAGConfig":
        return cls(**d)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)