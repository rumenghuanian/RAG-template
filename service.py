"""常驻 HTTP 服务：把 RAG / Agent 暴露成 API。

为什么需要它
------------
CLI 每启动一次要付 **21 秒冷启动**（实测：import 4.8s + 建 pipeline 7.1s + 载重排模型 9.5s），
之后每个问题才 1.1 秒。做成常驻服务后这笔钱只付一次。

    # 启动（模型只加载这一次）
    .\\.venv\\Scripts\\python.exe -m uvicorn service:app --host 127.0.0.1 --port 8000

    # 健康检查：能直接看到模型/设备/语料规模，用来确认「冷启动只付了一次」
    curl http://127.0.0.1:8000/health

    # 单轮 RAG
    curl -X POST http://127.0.0.1:8000/query -H "Content-Type: application/json" ^
         -d "{\\"question\\": \\"重排序是怎么做的\\"}"

    # Agent（多步检索）
    curl -X POST http://127.0.0.1:8000/agent -H "Content-Type: application/json" ^
         -d "{\\"question\\": \\"生产环境上线一个 Agent 平台要考虑哪些方面\\"}"

设计要点
--------
- **app 工厂**：`create_app(pipeline=...)` 可注入，所以服务层能脱离真实模型做单元测试。
- **处理器写成同步函数**：FastAPI 会把同步处理器丢进线程池，不会阻塞事件循环。
- **并发闸**：4GB 显存上重排模型约 1.1GB，并发太多会 OOM，所以用一个信号量兜住
  （`SERVICE_CONCURRENCY`，默认 2）。**这个值没有实测调优过**，只是防 OOM 的保守值。
- **不泄露密钥**：异常只回类型与消息，绝不回配置内容。
"""
import logging
import os
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

import bootstrap  # noqa: E402

bootstrap.prepare()  # 必须在构造任何 HTTP 客户端之前

from fastapi import FastAPI, HTTPException  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

logger = logging.getLogger(__name__)

STATE: Dict[str, Any] = {}
# 并发闸：4GB 显存上重排约 1.1GB，并发请求多了会 OOM。这个值没有实测调优，只是保守护栏。
_SEM = threading.Semaphore(int(os.getenv("SERVICE_CONCURRENCY", "2")))


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="用户问题")
    top_k: Optional[int] = Field(None, ge=1, le=50, description="返回几篇（默认用配置）")


class AgentRequest(BaseModel):
    question: str = Field(..., min_length=1)
    max_steps: int = Field(6, ge=1, le=12)


class Source(BaseModel):
    article_id: Optional[str] = None
    title: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    sources: List[Source] = []
    seconds: float
    queued_seconds: float = 0.0


class AgentResponse(BaseModel):
    answer: str
    citations: List[Source] = []
    needs_human: bool
    stop_reason: str
    steps: int
    seconds: float
    queued_seconds: float = 0.0


def _build_default_pipeline():
    """真正加载模型。只在服务启动时调用一次。

    目录切换和 .env 读取都放在这里，**不放模块级** —— 否则测试一 import 这个模块
    就会改掉整个进程的 CWD 和环境变量，属于隐蔽的全局副作用。
    """
    from dotenv import load_dotenv

    os.chdir(BASE_DIR)  # DATA_PATH 等相对路径是相对项目根的
    env_file = BASE_DIR / ".env.notes"
    load_dotenv(env_file if env_file.exists() else BASE_DIR / ".env")

    from rag_core import BasicRAGPipeline, RAGConfig
    from rag_core.skills import SKILLS

    config = RAGConfig()
    config.validate()
    skill = os.getenv("EVAL_SKILL", "notes")
    t0 = time.time()
    pipe = BasicRAGPipeline(config, SKILLS[skill]()).build()
    logger.info("pipeline 就绪，用时 %.1fs", time.time() - t0)
    return pipe, config


def create_app(pipeline=None, agent=None) -> FastAPI:
    """app 工厂。pipeline / agent 可注入 —— 测试用假的，生产走默认构建。"""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if pipeline is not None:
            STATE["pipeline"] = pipeline
            STATE["agent"] = agent
        else:
            pipe, config = _build_default_pipeline()
            from agent import OpenAICompatLLM, RAGAgent, build_tools

            tools = build_tools(pipe, min_relevance=config.min_relevance)
            llm = OpenAICompatLLM(
                model=config.llm_model,
                base_url=config.llm_base_url,
                api_key=config.llm_api_key,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )
            STATE["pipeline"] = pipe
            STATE["agent"] = RAGAgent(llm, tools)
            STATE["llm"] = llm
            # **预热重排模型**：重排器是懒加载的，不预热的话那 ~10 秒会被推迟到
            # 第一个用户的请求里（实测首问 18.7s vs 稳态 11.6s）。服务场景下
            # 这笔钱该在启动时付清，这样 /health 报 ok 就真的是「随时可用」。
            if getattr(getattr(pipe, "retriever", None), "reranker", None) is not None:
                t0 = time.time()
                pipe.retriever.reranker.available
                logger.info("重排模型预热完成，用时 %.1fs", time.time() - t0)
        STATE["started_at"] = time.time()
        STATE["cold_start_seconds"] = round(
            time.time() - STATE.get("_boot_at", time.time()), 2
        )
        yield
        STATE.clear()

    app = FastAPI(title="basic_rag service", version="1.1.0", lifespan=lifespan)
    STATE["_boot_at"] = time.time()

    def _pipeline():
        pipe = STATE.get("pipeline")
        if pipe is None:
            raise HTTPException(status_code=503, detail="模型尚未就绪")
        return pipe

    @app.get("/health")
    def health() -> Dict[str, Any]:
        pipe = STATE.get("pipeline")
        rr = getattr(getattr(pipe, "retriever", None), "reranker", None)
        config = getattr(pipe, "config", None)
        return {
            "status": "ok" if pipe is not None else "loading",
            "cold_start_seconds": STATE.get("cold_start_seconds"),
            "uptime_seconds": round(time.time() - STATE["started_at"], 1)
            if STATE.get("started_at")
            else 0.0,
            "documents": len(pipe.documents) if pipe is not None else 0,
            "reranker": rr.model_name if rr else None,
            "rerank_device": getattr(config, "rerank_device", None),
            "embedding_model": getattr(config, "embedding_model", None),
        }

    @app.post("/query", response_model=QueryResponse)
    def query(req: QueryRequest) -> QueryResponse:
        pipe = _pipeline()
        t_wait = time.time()
        with _SEM:
            queued = time.time() - t_wait
            t0 = time.time()
            try:
                if req.top_k is None:
                    answer, context_ids = pipe.query(req.question, with_context=True)
                else:
                    answer, context_ids = _query_with_top_k(pipe, req, req.top_k)
            except Exception as e:  # noqa: BLE001 - 不回显配置，避免泄露密钥
                logger.exception("query 失败")
                raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")
        return QueryResponse(
            answer=answer,
            sources=[Source(article_id=a) for a in context_ids],
            seconds=round(time.time() - t0, 2),
            queued_seconds=round(queued, 3),
        )

    @app.post("/agent", response_model=AgentResponse)
    def agent_run(req: AgentRequest) -> AgentResponse:
        _pipeline()
        ag = STATE.get("agent")
        if ag is None:
            raise HTTPException(status_code=503, detail="Agent 尚未就绪")
        ag.max_steps = req.max_steps
        t_wait = time.time()
        with _SEM:
            queued = time.time() - t_wait
            t0 = time.time()
            try:
                result = ag.run(req.question)
            except Exception as e:  # noqa: BLE001
                logger.exception("agent 失败")
                raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")
        return AgentResponse(
            answer=result.answer,
            citations=[Source(article_id=c.get("article_id"), title=c.get("title")) for c in result.citations],
            needs_human=result.needs_human,
            stop_reason=result.stop_reason,
            steps=len(result.steps),
            seconds=round(time.time() - t0, 2),
            queued_seconds=round(queued, 3),
        )

    return app


def _query_with_top_k(pipe, req: QueryRequest, top_k: int):
    """指定 top_k 时复用 pipeline 的检索与生成。

    走的是 pipeline 的内部步骤而不是 `query()`：每请求改 `config.top_k` 会污染共享状态，
    而 `query()` 的 top_k 来自 `_pick_top_k(route)`，没有按次覆盖的入口。
    """
    chunks = pipe._search(req.question, top_k, {})
    if not chunks:
        return "抱歉，没有找到相关信息。", []
    parents = pipe._get_parent_documents(chunks)
    mode = pipe._pick_prompt_mode("general")
    answer = pipe.generator.generate(mode, req.question, parents)
    return answer, [d.metadata.get("article_id") for d in parents]


app = create_app()
