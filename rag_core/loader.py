"""通用文档加载器"""
import hashlib
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class DocumentLoader:
    """
    从目录加载文档，为每个文件分配确定性的 parent_id。
    metadata_extractor 由 Skill 注入，用于提取领域字段（分类、难度等）。
    """

    def __init__(
        self,
        data_path: str,
        file_glob: str = "*.md",
        metadata_extractor: Optional[Callable[[Document], Dict]] = None,
        required_metadata: tuple = ("article_id", "title"),
    ):
        self.data_path = Path(data_path).resolve()
        self.file_glob = file_glob
        self.metadata_extractor = metadata_extractor
        self.required_metadata = tuple(required_metadata or ())
        self.documents: List[Document] = []

    def _validate(self) -> None:
        """契约校验：规范字段必须存在、非空、且唯一。

        为什么要在**加载时**硬失败：这些字段以前缺失只会变成 None，然后一路静默传播 ——
        实测过换领域时（recipe 不产出 article_id）出现「检索 5 篇塌成 [None]」「评测 hit@5
        全线 0%」「read_article(None) 返回语料里某一篇」这种**比崩溃更危险**的错。
        没有 extractor 时不校验：那种用法没有领域契约可言。
        """
        if not self.metadata_extractor or not self.required_metadata:
            return
        seen: Dict[Any, str] = {}
        for d in self.documents:
            for key in self.required_metadata:
                val = d.metadata.get(key)
                if val is None or (isinstance(val, str) and not val.strip()):
                    raise ValueError(
                        f"语料元数据缺少规范字段 {key!r}：{d.metadata.get('source')}\n"
                        f"  skill 的 metadata_extractor 必须产出 {self.required_metadata}，"
                        f"且值为非空。缺了它，检索/引用/评测会一路静默失效（不会报错）。"
                    )
            aid = d.metadata.get(self.required_metadata[0])
            if aid in seen:
                raise ValueError(
                    f"规范 id 不唯一：{aid!r}\n"
                    f"  同时出现在：{seen[aid]} 与 {d.metadata.get('source')}\n"
                    f"  id 重复会让 read_article/引用匹配指向错误的文档，必须消除歧义。"
                )
            seen[aid] = str(d.metadata.get("source"))

    def load(self) -> List[Document]:
        logger.info(f"从 {self.data_path} 加载文档 (glob={self.file_glob})")
        docs: List[Document] = []

        for f in self.data_path.rglob(self.file_glob):
            try:
                content = f.read_text(encoding="utf-8")
            except Exception as e:
                logger.warning(f"读取 {f} 失败: {e}")
                continue

            # 基于相对路径生成确定性 parent_id
            try:
                rel = f.resolve().relative_to(self.data_path).as_posix()
            except Exception:
                rel = f.as_posix()
            parent_id = hashlib.md5(rel.encode("utf-8")).hexdigest()

            doc = Document(
                page_content=content,
                metadata={
                    "source": str(f),
                    "parent_id": parent_id,
                    "doc_type": "parent",
                },
            )

            # 注入领域元数据
            if self.metadata_extractor:
                try:
                    doc.metadata.update(self.metadata_extractor(doc) or {})
                except Exception as e:
                    logger.warning(f"metadata_extractor 处理 {f} 失败: {e}")

            docs.append(doc)

        self.documents = docs
        self._validate()
        logger.info(f"成功加载 {len(docs)} 个文档")
        return docs