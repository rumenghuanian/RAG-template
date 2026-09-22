"""通用 Markdown 分块器（结构感知 + 父子块）"""
import hashlib
import logging
from typing import Dict, List, Tuple

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter

logger = logging.getLogger(__name__)


class DocumentSplitter:
    def __init__(self, headers: List[Tuple[str, str]], strip_headers: bool = False):
        self.headers = headers
        self.strip_headers = strip_headers
        self.parent_child_map: Dict[str, str] = {}

    def signature(self) -> str:
        """分块配置指纹。分块逻辑变了索引必须重建，所以要进数据指纹。"""
        return f"{self.headers}|strip_headers={self.strip_headers}"

    def split(self, documents: List[Document]) -> List[Document]:
        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers,
            strip_headers=self.strip_headers,
        )
        all_chunks: List[Document] = []

        for doc in documents:
            try:
                md_chunks = splitter.split_text(doc.page_content)
            except Exception as e:
                logger.warning(f"分割失败 {doc.metadata.get('source')}: {e}")
                all_chunks.append(doc)
                continue

            if len(md_chunks) <= 1:
                logger.warning(
                    f"{doc.metadata.get('source')} 未按标题分割，可能缺少标题结构"
                )

            parent_id = doc.metadata["parent_id"]
            for i, chunk in enumerate(md_chunks):
                # chunk_id 必须**确定性**：缓存索引里存的是上次构建时的 id，
                # 如果每次用 uuid4 新生成，RRF 按 chunk_id 去重时两路就永远合并不了
                # （实测 shared_chunks 恒为 0，max_rrf 恒等于 1/(k+1)）。
                child_id = hashlib.md5(f"{parent_id}:{i}".encode("utf-8")).hexdigest()
                chunk.metadata.update(doc.metadata)
                chunk.metadata.update(
                    {
                        "chunk_id": child_id,
                        "parent_id": parent_id,
                        "doc_type": "child",
                        "chunk_index": i,
                        "chunk_size": len(chunk.page_content),
                    }
                )
                self.parent_child_map[child_id] = parent_id
                all_chunks.append(chunk)

        logger.info(f"分块完成，共 {len(all_chunks)} 个 chunk")
        return all_chunks