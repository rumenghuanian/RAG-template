"""通用生成模块 - prompt 由 Skill 注入"""
import logging
from typing import Iterator, List, Optional

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

from .tokenizer import estimate_tokens

logger = logging.getLogger(__name__)

# 文档标题里优先展示的元数据键
_LABEL_KEYS = ("dish_name", "title", "name")
_TAG_KEYS = ("category", "difficulty")


class GenerationIntegrationModule:
    def __init__(
        self,
        model_name: str,
        base_url: str,
        api_key: str,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        prompt_registry: Optional[dict] = None,
        context_max_tokens: int = 6000,
    ):
        self.llm = ChatOpenAI(
            model=model_name,
            base_url=base_url,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self.prompts = prompt_registry or {}
        self.context_max_tokens = context_max_tokens
        logger.info(f"LLM 初始化完成: {model_name} @ {base_url}")

    # ---------- 基础生成 ----------
    def generate(self, mode: str, query: str, context_docs: List[Document]) -> str:
        return "".join(self.generate_stream(mode, query, context_docs))

    def generate_stream(
        self, mode: str, query: str, context_docs: List[Document]
    ) -> Iterator[str]:
        if mode not in self.prompts:
            raise KeyError(f"未注册 prompt mode: {mode}，可用: {list(self.prompts)}")

        context = self.build_context(context_docs)
        prompt: ChatPromptTemplate = self.prompts[mode]

        chain = (
            {"question": RunnablePassthrough(), "context": lambda _: context}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        for chunk in chain.stream(query):
            yield chunk

    # ---------- 上下文构建 ----------
    @staticmethod
    def _doc_header(doc: Document, idx: int) -> str:
        header = f"【文档 {idx}】"
        for k in _LABEL_KEYS:
            if k in doc.metadata:
                header += f" {doc.metadata[k]}"
                break
        for k in _TAG_KEYS:
            if k in doc.metadata:
                header += f" | {k}: {doc.metadata[k]}"
        return header

    def fit_context(self, docs: List[Document]) -> List[tuple]:
        """按 token 预算裁剪文档，返回 `[(doc, 实际喂进去的正文)]`。

        **为什么要把「实际喂进去的正文」单独拿出来**：超预算的文档会被截断，并追加
        一行「…（内容过长，已截断）」。如果评测时拿**未截断的父文档全文**当参照去判
        忠实度，模型如实说「文档被截断」反而会被判成编造 —— 实测踩到过：
        裁判的参照物是 21,228 字全文，而模型只看到截断后的 2,733 token，
        于是唯一的「不忠实」判定完全是口径造成的假象。
        **判分器看到的东西，必须和模型看到的东西是同一份。**

        纯函数（只依赖 docs 与 context_max_tokens），所以重复调用结果一致。
        """
        fitted: List[tuple] = []
        used = 0
        for i, doc in enumerate(docs, 1):
            remaining = self.context_max_tokens - used
            if remaining <= 0:
                logger.warning(
                    "上下文预算 %d token 已用尽，丢弃剩余 %d 篇文档",
                    self.context_max_tokens,
                    len(docs) - i + 1,
                )
                break

            body = doc.page_content
            if estimate_tokens(body) > remaining:
                # 中文 1 字 ≈ 1 token，按 token 余量截字符够用
                # ponytail: 要精确到 token 就换真 tokenizer，当前诉求只是别丢文档
                logger.warning(
                    "文档 %s 超出剩余预算，截断到约 %d token",
                    doc.metadata.get("source", "?"),
                    remaining,
                )
                body = body[:remaining] + "\n…（内容过长，已截断）"

            fitted.append((doc, body))
            used += estimate_tokens(f"{self._doc_header(doc, len(fitted))}\n{body}\n")
        return fitted

    def build_context(self, docs: List[Document]) -> str:
        """按 token 预算拼上下文。

        超预算的文档**截断**而不是丢弃——原来遇到一篇大文档就 break，
        会让 top_k 命中 3 篇却只喂进去 1 篇。
        """
        if not docs:
            return "暂无相关信息。"

        parts = [
            f"{self._doc_header(doc, i)}\n{body}\n"
            for i, (doc, body) in enumerate(self.fit_context(docs), 1)
        ]
        if not parts:  # 预算为 0 的极端配置
            return "暂无相关信息。"

        divider = "\n" + "=" * 50 + "\n"
        return divider + divider.join(parts)