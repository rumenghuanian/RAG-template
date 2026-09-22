"""交叉编码器重排序（cross-encoder reranker）。

为什么需要它
------------
向量检索是**双编码器**：query 和 chunk 分别编码成向量，再比余弦。
两者从未「一起读过」，所以它只能判断「主题像不像」，分不出「这段到底有没有回答问题」。
实测（`eval/run_eval.py`）的后果是：43 条标注里 **gold 从来没漏召过**，
但 paraphrase 那 12 条只有 58.3% 进得了前 5 —— 答案检索到了，只是排不上去。

交叉编码器把 `(query, chunk)` **拼成一条输入**送进模型，直接输出相关性分数，
判别力远高于余弦。代价是每对都要过一次模型，没法预计算，所以只能用来重排
一个小候选池（几十到一两百条），不能用来检索全库。

设计要点
--------
- **懒加载**：构造对象不碰模型；第一次真正重排时才加载权重。
- **失败退化**：没装 `sentence-transformers`、权重没下载、显存/内存不够时，
  记一条 WARNING 后**原样返回**，绝不把整个检索链路带崩。
- **分数落在 [0,1]**：bge 系列单标签 CrossEncoder 默认过 sigmoid，
  所以它可以直接当阈值用（见 README 关于「重排分数能否当闸门」的实测）。
"""
import logging
from typing import List, Optional, Sequence

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-base"


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = DEFAULT_RERANK_MODEL,
        device: str = "cpu",
        max_length: int = 512,
        batch_size: int = 16,
    ):
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
        self._model = None
        self._failed = False

    # ---------- 加载 ----------
    def _load(self):
        if self._model is not None or self._failed:
            return self._model
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:  # 没装依赖不该让检索失败
            logger.warning("未安装 sentence-transformers，跳过重排序：%s", e)
            self._failed = True
            return None
        try:
            self._model = CrossEncoder(
                self.model_name, device=self.device, max_length=self.max_length
            )
        except Exception as e:  # noqa: BLE001 - 权重缺失 / OOM 一律退化为不重排
            logger.warning(
                "重排序模型 %s 加载失败，本次退回不重排：%s", self.model_name, e
            )
            self._failed = True
            return None
        logger.info("重排序模型就绪：%s（device=%s）", self.model_name, self.device)
        return self._model

    @property
    def available(self) -> bool:
        """能不能真重排。会触发一次加载，别放进热循环。"""
        return self._load() is not None

    # ---------- 打分 ----------
    def score(self, query: str, texts: Sequence[str]) -> List[float]:
        """返回每条文本的相关性分数；不可用时返回空列表（调用方据此退化）。"""
        if not texts:
            return []
        model = self._load()
        if model is None:
            return []
        pairs = [(query, t) for t in texts]
        try:
            raw = model.predict(
                pairs, batch_size=self.batch_size, show_progress_bar=False
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("重排序打分失败，退回不重排：%s", e)
            return []
        return [float(x) for x in raw]

    def rerank(
        self, query: str, docs: List[Document], top_n: Optional[int] = None
    ) -> List[Document]:
        """按相关性重排文档，并把分数写进 `metadata["rerank_score"]`。

        不可用时原样返回（截到 top_n），调用方无需分支判断。
        """
        if not docs:
            return []
        scores = self.score(query, [d.page_content for d in docs])
        if not scores:
            return docs[:top_n] if top_n else docs
        for doc, s in zip(docs, scores):
            doc.metadata["rerank_score"] = s
        order = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)
        ranked = [docs[i] for i in order]
        return ranked[:top_n] if top_n else ranked
