"""通用文档加载器"""
import hashlib
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def make_path_metadata_extractor(data_path) -> Callable[[Document], Dict]:
    """没有提供 `metadata_extractor` 时的**约定实现**（新领域零配置的关键）。

    - `article_id` = 相对语料根的完整路径（**含后缀**，如 `week5/29.RAG是怎么工作的.md`）
    - `title`     = 文件名去后缀

    为什么用完整相对路径，而不是两边都省事的 `parts[-2:]`：
    后者会撞名，而且是真撞过 —— `meat_dish/红烧肉`（南派/简易两个文件），
    `soup/陈皮排骨汤.md` 与 `soup/陈皮排骨汤/陈皮排骨汤.md`（内容还重复）。
    完整路径**天然唯一**，于是「把语料丢进目录就能跑」不用先想清楚 id 规则。

    为什么含后缀：与既有的 notes 约定一致（golden/seeds 里写的 id 就长这样），
    换成去后缀会让所有现存标注对不上。领域想要别的形式，自己提供 extractor 覆盖即可。
    """
    root = Path(data_path).resolve()

    def extract(doc: Document) -> Dict:
        src = Path(str(doc.metadata.get("source", "")))
        try:
            rel = src.resolve().relative_to(root).as_posix()
        except Exception:
            # 语料根之外的文件（自定义 loader 才会遇到）：退化成文件名，
            # 重名由 _validate() 在加载期报错，不静默。
            rel = src.name
        return {"article_id": rel, "title": src.stem}

    return extract


class DocumentLoader:
    """
    从目录加载文档，为每个文件分配确定性的 parent_id。
    metadata_extractor 由 Skill 注入，用于提取领域字段（分类、难度等）；
    **不注入时用 `make_path_metadata_extractor` 的路径约定**，规范字段照样有。
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
        # 默认值而不是"没给就算了"：以前 extractor=None 时 `_validate()` 直接跳过，
        # 于是没有 article_id 也一路跑完 —— 检索去重塌成 [None]、评测 hit@5 全 0%，
        # 一行错误都不报。默认实现把这条路堵死。
        self.metadata_extractor = metadata_extractor or make_path_metadata_extractor(
            self.data_path
        )
        self.required_metadata = tuple(required_metadata or ())
        self.documents: List[Document] = []

    def _validate(self) -> None:
        """契约校验：规范字段必须存在、非空、且唯一。

        为什么要在**加载时**硬失败：这些字段以前缺失只会变成 None，然后一路静默传播 ——
        实测过换领域时（recipe 不产出 article_id）出现「检索 5 篇塌成 [None]」「评测 hit@5
        全线 0%」「read_article(None) 返回语料里某一篇」这种**比崩溃更危险**的错。
        只有显式声明 `required_metadata=()` 时才跳过校验。
        """
        if not self.required_metadata:
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