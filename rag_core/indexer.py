"""通用向量索引构建模块"""
import logging
from pathlib import Path
from typing import List, Optional
import hashlib
import json

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)


class IndexConstructionModule:
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-zh-v1.5",
        index_save_path: str = "./vector_index",
        device: str = "cpu",
        normalize_embeddings: bool = True,
    ):
        self.model_name = model_name
        self.index_save_path = index_save_path
        self.device = device
        self.normalize_embeddings = normalize_embeddings
        self.embeddings = None
        self.vectorstore: Optional[FAISS] = None
        self.setup_embeddings()

    def setup_embeddings(self):
        logger.info(f"初始化嵌入模型: {self.model_name}")
        try:
            # 模型已在本地时不要联网。huggingface_hub 启动会去 HEAD 查版本，
            # 而它建 httpx 客户端时会解析 NO_PROXY —— 如果里面有不规范的
            # IPv6 写法（如 [::1]），会直接抛 httpx.InvalidURL 把启动打断。
            # 这里成功 = 走本地缓存，完全不碰网络。
            self.embeddings = self._build_embeddings(
                {"device": self.device, "local_files_only": True}
            )
            logger.info("嵌入模型从本地缓存加载（未联网）")
        except Exception as e:  # noqa: BLE001 - 本地没有就是没有，退回联网下载
            logger.info(f"本地缓存不可用（{type(e).__name__}），改为联网加载模型")
            self.embeddings = self._build_embeddings({"device": self.device})

    def _build_embeddings(self, model_kwargs: dict):
        return HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs=model_kwargs,
            encode_kwargs={"normalize_embeddings": self.normalize_embeddings},
        )

    def build_vector_index(self, chunks: List[Document]) -> FAISS:
        if not chunks:
            raise ValueError("chunks 不能为空")
        logger.info("构建 FAISS 索引...")
        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        logger.info(f"索引构建完成: {len(chunks)} 个向量")
        return self.vectorstore

    def add_documents(self, new_chunks: List[Document]):
        if not self.vectorstore:
            raise ValueError("请先构建索引")
        self.vectorstore.add_documents(new_chunks)

    def save_index(self):
        if not self.vectorstore:
            raise ValueError("请先构建索引")
        Path(self.index_save_path).mkdir(parents=True, exist_ok=True)
        self.vectorstore.save_local(self.index_save_path)
        logger.info(f"索引已保存: {self.index_save_path}")

    def load_index(self) -> Optional[FAISS]:
        if not Path(self.index_save_path).exists():
            return None
        try:
            self.vectorstore = FAISS.load_local(
                self.index_save_path,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
            logger.info(f"索引已加载: {self.index_save_path}")
            return self.vectorstore
        except Exception as e:
            logger.warning(f"加载索引失败: {e}")
            return None


    def _fingerprint_file(self) -> Path:
        return Path(self.index_save_path) / ".fingerprint"

    def compute_fingerprint(self, chunks, splitter_signature: str = "") -> str:
        """指纹 = embedding 模型 + 归一化 + 分块配置 + 每个 chunk 的身份、内容、来源与元数据。

        分块配置必须进指纹：改了 splitter_headers 而指纹不变，会加载旧索引配上
        新 chunk 列表，检索结果与生成上下文静默错位，且不报任何错。
        chunk_id 也要进：索引里存着它，改了 chunk 身份方案就必须重建，
        否则新切出来的块和索引里的块对不上号。

        **领域元数据同样要进**：索引的 docstore 里存着建库时的 metadata。
        实测踩过：改了 skill 的 metadata_extractor（补出规范字段 article_id），
        指纹没变 → 向量检索仍然返回**旧元数据**（没有 article_id），
        而 BM25 那一路用的是刚加载的新文档（有 article_id）——
        两条路拿到的元数据不一致，评测直接崩在「检索结果缺少 article_id」。
        """
        h = hashlib.md5()
        h.update(self.model_name.encode("utf-8"))
        h.update(str(self.normalize_embeddings).encode("utf-8"))
        h.update(splitter_signature.encode("utf-8"))
        for c in chunks:
            meta = getattr(c, "metadata", {}) or {}
            h.update(str(meta.get("chunk_id", "")).encode("utf-8"))
            h.update(c.page_content.encode("utf-8"))
            h.update(str(meta.get("source", "")).encode("utf-8"))
            # 领域元数据：排序后序列化，保证确定性
            h.update(
                json.dumps(
                    {k: str(v) for k, v in sorted(meta.items())},
                    ensure_ascii=False,
                    sort_keys=True,
                ).encode("utf-8")
            )
        return h.hexdigest()

    def is_index_fresh(self, current_fingerprint: str) -> bool:
        fp_file = self._fingerprint_file()
        if not fp_file.exists():
            logger.warning("未找到数据指纹 %s，将重建索引", fp_file)
            return False
        try:
            saved = json.loads(fp_file.read_text())["fingerprint"]
        except Exception as e:
            logger.warning(f"数据指纹读取失败({e})，将重建索引")
            return False
        if saved != current_fingerprint:
            logger.info("数据或分块配置有变化，将重建索引")
            return False
        return True

    def save_fingerprint(self, fingerprint: str):
        self._fingerprint_file().write_text(
            json.dumps({"fingerprint": fingerprint})
        )