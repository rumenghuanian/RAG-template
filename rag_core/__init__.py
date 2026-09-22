"""通用 RAG 骨架。

这里只导出轻量模块。pipeline 依赖 torch/faiss/huggingface，改为延迟导入，
这样 `import rag_core.tokenizer` 之类的用法不必先把重型依赖装齐。
"""

from .config import RAGConfig
from .reranker import CrossEncoderReranker
from .skill import RAGSkill

__all__ = [
    "RAGConfig",
    "BasicRAGPipeline",
    "RAGSkill",
    "CrossEncoderReranker",
]
__version__ = "1.1.0"


def __getattr__(name: str):
    if name == "BasicRAGPipeline":
        from .pipeline import BasicRAGPipeline

        return BasicRAGPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
