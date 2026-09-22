"""通用 RAG Pipeline - 组装 loader/splitter/indexer/retriever/generator"""
import logging
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

from langchain_core.documents import Document

from .config import RAGConfig
from .generator import GenerationIntegrationModule
from .indexer import IndexConstructionModule
from .loader import DocumentLoader
from .reranker import CrossEncoderReranker
from .retriever import RetrievalOptimizationModule
from .skill import RAGSkill
from .splitter import DocumentSplitter

logger = logging.getLogger(__name__)


class BasicRAGPipeline:
    def __init__(self, config: RAGConfig, skill: RAGSkill):
        self.config = config
        self.skill = skill

        self.documents: List[Document] = []
        self.chunks: List[Document] = []
        self.indexer: Optional[IndexConstructionModule] = None
        self.retriever: Optional[RetrievalOptimizationModule] = None
        self.generator: Optional[GenerationIntegrationModule] = None

        # 父文档索引（parent_id -> Document）
        self._parent_index: Dict[str, Document] = {}

    # ---------- 构建 ----------
    def build(self):
        logger.info(f"开始构建 RAG Pipeline (skill={self.skill.name})")

        # 1. Loader
        loader = DocumentLoader(
            data_path=self.config.data_path,
            file_glob=self.config.file_glob,
            metadata_extractor=self.skill.metadata_extractor,
        )
        self.documents = loader.load()
        self._parent_index = {d.metadata["parent_id"]: d for d in self.documents}

        # 2. Splitter
        splitter = DocumentSplitter(
            headers=self.skill.splitter_headers,
            strip_headers=self.skill.strip_headers,
        )
        self.chunks = splitter.split(self.documents)

        # 3. Indexer（优先加载已有索引）
        self.indexer = IndexConstructionModule(
            model_name=self.config.embedding_model,
            index_save_path=self.config.index_save_path,
            device=self.config.embedding_device,
        )
        current_fp = self.indexer.compute_fingerprint(
            self.chunks, splitter.signature()
        )
        vs = None
        if self.indexer.is_index_fresh(current_fp):
            vs = self.indexer.load_index()

        if vs is None:
            logger.info("数据有变化或索引不存在，重建索引...")
            vs = self.indexer.build_vector_index(self.chunks)
            self.indexer.save_index()
            self.indexer.save_fingerprint(current_fp)

        # 4. Retriever（重排序器只构造对象，权重在第一次真正重排时才加载）
        reranker = None
        if self.config.rerank_enabled:
            reranker = CrossEncoderReranker(
                model_name=self.config.rerank_model,
                device=self.config.rerank_device,
            )
        self.retriever = RetrievalOptimizationModule(
            vs,
            self.chunks,
            reranker=reranker,
            rerank_candidates=self.config.rerank_candidates,
        )

        # 5. Generator
        self.generator = GenerationIntegrationModule(
            model_name=self.config.llm_model,
            base_url=self.config.llm_base_url,
            api_key=self.config.llm_api_key,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            prompt_registry=self.skill.prompt_registry,
            context_max_tokens=self.config.context_max_tokens,
        )

        # 6. 给 query_strategy 注入 llm（如果它需要）
        strategy = self.skill.query_strategy
        if strategy and hasattr(strategy, "set_llm"):
            strategy.set_llm(self.generator.llm)

        logger.info("RAG Pipeline 构建完成")
        return self

    # ---------- 查询 ----------
    def _prepare_context(self, question: str) -> Tuple[str, List[Document]]:
        """路由 → （可选）重写 → 检索 → 父文档回填 → 选 prompt mode。

        返回 (mode, parents)；`parents` 为空表示检索什么也没找到。
        抽出来是为了让「只要答案」和「还要出处正文」两条路走同一份准备逻辑，
        否则两边的检索口径迟早会漂移。
        """
        strategy = self.skill.query_strategy

        route = strategy.route(question) if strategy else "general"
        logger.info(f"查询路由: {route}")

        # 列表类不重写
        if strategy and self.config.enable_rewrite and route != "list":
            rewritten = strategy.rewrite(question)
        else:
            rewritten = question

        filters = strategy.extract_filters(question) if strategy else {}
        chunks = self._search(rewritten, self._pick_top_k(route), filters)
        parents = self._get_parent_documents(chunks) if chunks else []
        return self._pick_prompt_mode(route), parents

    def query_with_sources(self, question: str) -> Tuple[str, List[Document]]:
        """同 `query()`，但把**真正喂进 prompt 的父文档**一并返回。

        为什么需要：判「答案有没有编造」必须拿**模型实际看到的上下文**当参照物。
        `query(with_context=True)` 只给了 article_id，没有正文 —— 拿 gold 当参照会
        把「引用了上下文里其他文章」误判成编造（评测里踩过）。
        服务层也可以用它把出处正文返回给前端。非流式。

        ⚠️ 返回的 `page_content` 是**截断后实际喂进去的正文**，不是原始全文。
        否则判分器看到的比模型多，模型如实说「文档被截断」会被判成编造（踩过）。
        `fit_context` 是纯函数，与 `generate()` 内部那次裁剪结果一致。
        """
        if not self.retriever or not self.generator:
            raise RuntimeError("请先调用 build()")
        mode, parents = self._prepare_context(question)
        if not parents:
            return "抱歉，没有找到相关信息。", []
        answer = self.generator.generate(mode, question, parents)
        fed = [
            Document(page_content=body, metadata=dict(doc.metadata))
            for doc, body in self.generator.fit_context(parents)
        ]
        return answer, fed

    def query(
        self, question: str, stream: bool = False, with_context: bool = False
    ) -> Union[str, Iterator[str], Tuple[str, List[str]]]:
        """单轮问答。with_context=True 时返回 (答案, 实际喂给模型的 article_id 列表)。

        需要**出处正文**而不是 id 时用 `query_with_sources()`。
        流式模式下 with_context 无效（此时拿不到完整结果）。
        """
        if not self.retriever or not self.generator:
            raise RuntimeError("请先调用 build()")

        if stream:
            mode, parents = self._prepare_context(question)
            if not parents:
                return iter(["抱歉，没有找到相关信息。"])
            return self.generator.generate_stream(mode, question, parents)

        answer, parents = self.query_with_sources(question)
        if with_context:
            return answer, [d.metadata.get("article_id") for d in parents]
        return answer

    # ---------- 检索策略 ----------
    def retrieve(
        self,
        question: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """只检索不生成：最多返回 top_k **篇不同文章**命中的块，每篇只留排名最靠前的块。

        Agent 用这个拿「证据 + 出处」；query() 用父文档回填拿完整上下文。
        """
        want = top_k or self.config.top_k
        # 多取候选：chunk 常集中在同一篇文章里，按篇去重后才会缩水。
        # 开重排时排序更准、命中更集中在最好的那篇上，缩水更明显，所以多要一些。
        fetch = want * (10 if getattr(self.retriever, "reranker", None) is not None else 3)
        chunks = self._search(question, fetch, filters or {})
        seen: set = set()
        hits: List[Document] = []
        for c in chunks:
            pid = c.metadata.get("parent_id")
            if pid in seen:
                continue
            seen.add(pid)
            hits.append(c)
            if len(hits) >= want:
                break
        return hits

    def _pick_top_k(self, route: str) -> int:
        """list 意图（「推荐几个素菜」）要的是一批结果，不是 3 条。"""
        if route == "list":
            return max(self.config.top_k, 10)
        return self.config.top_k

    def _search(
        self, query: str, top_k: int, filters: Dict[str, Any]
    ) -> List[Document]:
        """检索；过滤条件组合过窄时逐级放宽，而不是直接返回「没找到」。

        「推荐简单的素菜」= category + difficulty 两个条件，组合过窄时
        原实现会静默返回空。
        """
        attempts: List[Dict[str, Any]] = []
        for f in (
            filters,
            {k: v for k, v in filters.items() if k != "difficulty"},
            {},
        ):
            if f not in attempts:
                attempts.append(f)

        for i, f in enumerate(attempts):
            chunks = self.retriever.hybrid_search(query, top_k=top_k, filters=f or None)
            if chunks:
                if i:
                    logger.warning(f"过滤条件 {filters} 无结果，已放宽为 {f or '无过滤'}")
                return chunks
        return []

    # ---------- 辅助 ----------
    def _get_parent_documents(self, chunks: List[Document]) -> List[Document]:
        """按被命中次数排序，去重回填父文档"""
        relevance: Dict[str, int] = {}
        for c in chunks:
            pid = c.metadata.get("parent_id")
            if pid:
                relevance[pid] = relevance.get(pid, 0) + 1

        sorted_pids = sorted(relevance, key=lambda x: relevance[x], reverse=True)
        parents = [
            self._parent_index[pid]
            for pid in sorted_pids
            if pid in self._parent_index
        ]
        logger.info(f"{len(chunks)} 个子块 → {len(parents)} 个父文档")
        return parents

    def _pick_prompt_mode(self, route: str) -> str:
        """
        路由 -> prompt mode 映射。
        Skill 可通过 prompt_registry 里注册的 key 覆盖默认行为。
        """
        if route in self.skill.prompt_registry:
            return route
        # 默认映射
        default_map = {"list": "basic", "detail": "basic", "general": "basic"}
        return default_map.get(route, "basic")