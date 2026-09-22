"""食谱查询策略：路由 / 重写 / 过滤条件提取"""
import logging
from typing import Any, Dict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from .metadata import CATEGORY_LABELS, DIFFICULTY_LABELS
from .prompts import build_router_prompt, build_rewrite_prompt

logger = logging.getLogger(__name__)


class RecipeQueryStrategy:
    """
    需要注入 llm 才能工作。Pipeline 构建后手动 set_llm()。
    若未注入 llm，则降级为 no-op（route=general, rewrite=原样, filters={}）。
    """

    def __init__(self):
        self.llm = None
        self._router_chain = None
        self._rewrite_chain = None

    def set_llm(self, llm):
        self.llm = llm
        router_prompt = build_router_prompt()
        rewrite_prompt = build_rewrite_prompt()
        self._router_chain = (
            {"query": RunnablePassthrough()} | router_prompt | llm | StrOutputParser()
        )
        self._rewrite_chain = (
            {"query": RunnablePassthrough()} | rewrite_prompt | llm | StrOutputParser()
        )

    # ---------- 路由 ----------
    def route(self, query: str) -> str:
        if not self._router_chain:
            return "general"
        try:
            result = self._router_chain.invoke(query).strip().lower()
            return result if result in ("list", "detail", "general") else "general"
        except Exception as e:
            logger.warning(f"路由失败: {e}")
            return "general"

    # ---------- 重写 ----------
    def rewrite(self, query: str) -> str:
        if not self._rewrite_chain:
            return query
        try:
            result = self._rewrite_chain.invoke(query).strip()
            if result and result != query:
                logger.info(f"查询重写: '{query}' → '{result}'")
            return result or query
        except Exception as e:
            logger.warning(f"重写失败: {e}")
            return query

    # ---------- 过滤条件 ----------
    def extract_filters(self, query: str) -> Dict[str, Any]:
        filters = {}
        for cat in CATEGORY_LABELS:
            if cat in query:
                filters["category"] = cat
                break
        for diff in sorted(DIFFICULTY_LABELS, key=len, reverse=True):
            if diff in query:
                filters["difficulty"] = diff
                break
        return filters