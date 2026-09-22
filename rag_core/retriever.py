"""通用混合检索模块（向量 + BM25 + RRF 重排 + 按 query 动态选策略）

为什么要动态：评测实测（`eval/`）三种检索在不同题型上强弱分明——

    paraphrase（换种说法）  向量 41.7%  BM25 33.3%  RRF 58.3%   → 融合最好
    rare_token（罕见术语）  向量 75.0%  BM25 100%   RRF 87.5%   → 融合反而拖后腿
    easy（照标题改写）      大家都 100%                          → 随便

所以固定权重不是最优：该让 query 自己决定偏向哪一路。
"""
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from .reranker import CrossEncoderReranker
from .tokenizer import tokenize_for_bm25

logger = logging.getLogger(__name__)

# BM25 没有原生 metadata 过滤，带过滤时只能先多取再筛，所以候选池放大一截
_BM25_OVERFETCH = 4
# FAISS 是「先取 fetch_k 个再按 filter 筛」，只取 candidate_k 会把过滤后的结果筛空
_VECTOR_FETCH_FACTOR = 5

# 只出现在不超过这么多篇里的词算「罕见」。下界必须是 1：
# df=0 表示语料里根本没有这个词，那是「答不了」的信号，不是「该走字面检索」的信号
_RARE_DF_MIN = 1
_RARE_DF = 2
# 罕见词还得长得像标识符（AgentExecutor / ABAC），否则中文短语的稀有二字组会误触发
_IDENT_MIN_ALPHA = 3
# (向量权重, BM25 权重)
_LEXICAL_WEIGHTS = (0.6, 1.4)
_EQUAL_WEIGHTS = (1.0, 1.0)


def _is_identifier(token: str) -> bool:
    return sum(ch.isascii() and ch.isalpha() for ch in token) >= _IDENT_MIN_ALPHA


class RetrievalOptimizationModule:
    def __init__(
        self,
        vectorstore: FAISS,
        chunks: List[Document],
        candidate_k: int = 20,
        reranker: Optional[CrossEncoderReranker] = None,
        rerank_candidates: int = 50,
    ):
        self.vectorstore = vectorstore
        self.chunks = chunks
        self.candidate_k = candidate_k
        self.reranker = reranker
        self.rerank_candidates = rerank_candidates
        self._setup_retrievers()
        self.token_df = self._build_token_df()

    def _setup_retrievers(self):
        # 必须传 preprocess_func：默认的 text.split() 对中文等于不分词
        self.bm25_retriever = BM25Retriever.from_documents(
            self.chunks,
            k=self.candidate_k * _BM25_OVERFETCH,
            preprocess_func=tokenize_for_bm25,
        )
        logger.info(
            "检索器初始化完成（向量候选 %d，BM25 候选 %d）",
            self.candidate_k,
            self.candidate_k * _BM25_OVERFETCH,
        )

    def _build_token_df(self) -> Dict[str, int]:
        """统计每个 token 出现在多少篇父文档里，供 plan() 判断 query 是否含罕见标识符。

        统一转小写：语料里写 `AgentExecutor`、用户可能敲 `agentexecutor`，
        不归一化就会查不到。
        """
        per_parent: Dict[str, set] = {}
        for c in self.chunks:
            pid = c.metadata.get("parent_id")
            bucket = per_parent.setdefault(pid, set())
            bucket.update(t.lower() for t in tokenize_for_bm25(c.page_content))
        df: Dict[str, int] = {}
        for tokens in per_parent.values():
            for t in tokens:
                df[t] = df.get(t, 0) + 1
        logger.info("token df 表建好了：%d 个词 / %d 篇", len(df), len(per_parent))
        return df

    # ---------- 策略 ----------
    def plan(self, query: str) -> Dict[str, Any]:
        """按 query 特征选检索策略。**不改输出条数**，只改融合权重。

        - `lexical`：query 含只在极少数文章出现的标识符（`AgentExecutor`、`ABAC`），
          字面匹配非常精确而向量容易飘，所以加权偏向 BM25
        - `semantic`：其余情况两路等权

        实测（`eval/`）：`rare_token` 组固定等权 RRF 是 87.5%，切到 lexical 后 100%；
        另外三组不受影响。净增益 +2.9%（35 条）。

        试过对枚举式提问（哪些/分别）加深向量候选池，6 条触发但 multi 覆盖率
        一点没动（46.7% → 46.7%），**已删除**——测不出效果的东西不留。
        """
        tokens = tokenize_for_bm25(query)
        rare_ids = [
            t
            for t in tokens
            if _RARE_DF_MIN <= self.token_df.get(t.lower(), 0) <= _RARE_DF
            and _is_identifier(t)
        ]
        if rare_ids:
            return {
                "mode": "lexical",
                "weights": _LEXICAL_WEIGHTS,
                "signals": rare_ids[:3],
            }
        return {"mode": "semantic", "weights": _EQUAL_WEIGHTS, "signals": []}

    # ---------- 检索 ----------
    def hybrid_search(
        self,
        query: str,
        top_k: int = 3,
        filters: Optional[Dict[str, Any]] = None,
        auto: bool = True,
        rerank: Optional[bool] = None,
    ) -> List[Document]:
        """混合检索。auto=True 时按 plan() 动态选权重；filters 下推到向量侧。

        rerank=None 时按「有没有配重排序器」决定；显式传 False 可以关掉，
        评测里靠这个把「重排前 / 重排后」两臂分开比。
        """
        p = self.plan(query) if auto else {
            "mode": "fixed",
            "weights": _EQUAL_WEIGHTS,
            "signals": [],
        }
        vector_docs = self._vector_search(query, filters)
        bm25_docs = self._bm25_search(query, filters)
        fused = self._rrf_rerank(vector_docs, bm25_docs, weights=p["weights"])
        logger.info(
            "检索策略 %s%s → %d 条",
            p["mode"],
            f"（命中罕见词 {p['signals']}）" if p["signals"] else "",
            len(fused[:top_k]),
        )

        if (self.reranker is not None if rerank is None else rerank) and self.reranker is not None:
            # 池深**与调用方的 top_k 完全无关**，只由 rerank_candidates 决定（0 = 不截断）。
            # 这很重要：retrieve(top_k=5) 会去要一批候选，而评测里 retrieve(top_k=10**6)
            # 会去要整池；若池深跟着 top_k 变，两次调用的排序口径就不可比 —— 实测这让
            # 「agent 单次检索」相对「单轮取 5 篇」白丢 2.7pp，纯属参数不一致造成的假象。
            pool = fused[: self.rerank_candidates] if self.rerank_candidates > 0 else fused
            ranked = self.reranker.rerank(query, pool)
            logger.info(
                "重排 %d 条 → %d 条（模型 %s）",
                len(pool),
                len(ranked[:top_k]),
                self.reranker.model_name,
            )
            return ranked[:top_k]
        return fused[:top_k]

    def _vector_search(
        self,
        query: str,
        filters: Optional[Dict[str, Any]],
    ) -> List[Document]:
        """带分数取回，并把余弦相似度写进 metadata。

        RRF 只编码排名不编码相似度（实测所有命中的 rrf_score 都在 0.0159~0.0164），
        所以「这条到底有多相关」只能靠向量分数判断。
        """
        k = self.candidate_k
        if not filters:
            scored = self.vectorstore.similarity_search_with_score(query, k=k)
        else:
            scored = self.vectorstore.similarity_search_with_score(
                query,
                k=k,
                fetch_k=k * _VECTOR_FETCH_FACTOR,
                filter=self._filter_predicate(filters),
            )
        return [self._with_cosine(doc, dist) for doc, dist in scored]

    @staticmethod
    def _with_cosine(doc: Document, distance: float) -> Document:
        # embedding 做了归一化、FAISS 用欧氏距离时：d^2 = 2 - 2cos
        doc.metadata["vector_cos"] = 1.0 - float(distance) ** 2 / 2
        return doc

    def _bm25_search(
        self, query: str, filters: Optional[Dict[str, Any]]
    ) -> List[Document]:
        docs = self.bm25_retriever.invoke(query)
        if not filters:
            return docs
        return [d for d in docs if self._match_filters(d, filters)]

    # ---------- 工具 ----------
    @staticmethod
    def _filter_predicate(filters: Dict[str, Any]) -> Callable[[Document], bool]:
        return lambda doc: RetrievalOptimizationModule._match_filters(doc, filters)

    @staticmethod
    def _match_filters(doc: Document, filters: Dict[str, Any]) -> bool:
        for key, value in filters.items():
            if key not in doc.metadata:
                return False
            if isinstance(value, list):
                if doc.metadata[key] not in value:
                    return False
            else:
                if doc.metadata[key] != value:
                    return False
        return True

    @staticmethod
    def _rrf_rerank(
        vector_docs: List[Document],
        bm25_docs: List[Document],
        k: int = 60,
        weights: Tuple[float, float] = _EQUAL_WEIGHTS,
    ) -> List[Document]:
        """
        使用 chunk_id 做去重 key（比 page_content md5 更安全，
        避免内容相同的 chunk 被误合并）。
        weights = (向量权重, BM25 权重)，默认等权。
        """
        w_vector, w_bm25 = weights
        scores: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}

        def _key(doc: Document, fallback_idx: int) -> str:
            return doc.metadata.get("chunk_id") or f"__noid_{fallback_idx}"

        for rank, doc in enumerate(vector_docs):
            key = _key(doc, rank)
            doc_map[key] = doc
            scores[key] = scores.get(key, 0.0) + w_vector / (k + rank + 1)

        for rank, doc in enumerate(bm25_docs):
            key = _key(doc, rank)
            # 同一个 chunk 两路都命中时不能无条件用 BM25 的对象覆盖：
            # vector_cos 是写在向量那一路对象上的（见 _with_cosine），覆盖会把它丢掉，
            # 而下游 search_notes 的相关性闸门正是靠它判断「语料里到底有没有这件事」。
            # 实测覆盖后闸门把余弦读成 0.00，对几乎所有正常问题都返回空。
            # 向量那一轮先插入，所以 setdefault 就是「带余弦的对象优先保留」。
            doc_map.setdefault(key, doc)
            scores[key] = scores.get(key, 0.0) + w_bm25 / (k + rank + 1)

        reranked = []
        for key, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            doc = doc_map[key]
            doc.metadata["rrf_score"] = score
            reranked.append(doc)

        logger.info(
            "RRF 重排: 向量 %d + BM25 %d → %d（权重 %.1f/%.1f）",
            len(vector_docs),
            len(bm25_docs),
            len(reranked),
            w_vector,
            w_bm25,
        )
        return reranked
