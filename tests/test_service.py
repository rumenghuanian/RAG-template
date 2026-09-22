"""服务层测试：注入假 pipeline / 假 agent，**不加载真实模型**。

这也是 app 工厂存在的理由 —— 服务层要能脱离 torch 与网络做单元测试。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402

from agent.loop import AgentResult  # noqa: E402
from service import create_app  # noqa: E402


class _FakeConfig:
    rerank_device = "cuda"
    embedding_model = "BAAI/bge-small-zh-v1.5"


class _FakeRetriever:
    reranker = None


class _FakeDoc:
    def __init__(self, aid):
        self.metadata = {"article_id": aid}


class _FakePipeline:
    """只实现服务层真正用到的接口。"""

    def __init__(self, answer="这是答案（week5/33.重排序.md）", ctx=("week5/33.重排序.md",)):
        self.config = _FakeConfig()
        self.retriever = _FakeRetriever()
        self.documents = [_FakeDoc("week5/33.重排序.md"), _FakeDoc("week5/31.文本分块.md")]
        self._answer, self._ctx = answer, list(ctx)
        self.calls = []

    def query(self, question, stream=False, with_context=False):
        self.calls.append((question, stream, with_context))
        return (self._answer, self._ctx) if with_context else self._answer


class _FakeAgent:
    def __init__(self, result=None):
        self.max_steps = 6
        self.result = result or AgentResult(
            answer="答案（week5/33.重排序.md）",
            citations=[{"article_id": "week5/33.重排序.md", "title": "重排序"}],
            steps=[{"step": 1, "tool": "search_notes", "ok": True}],
            stop_reason="answered",
            needs_human=False,
        )

    def run(self, question, history=None):
        return self.result


def _client(**kw):
    return TestClient(create_app(**kw))


def test_health_reports_model_state():
    with _client(pipeline=_FakePipeline(), agent=_FakeAgent()) as c:
        r = c.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["documents"] == 2
        assert body["rerank_device"] == "cuda"
        assert body["uptime_seconds"] >= 0


def test_query_returns_answer_and_sources():
    pipe = _FakePipeline()
    with _client(pipeline=pipe, agent=_FakeAgent()) as c:
        r = c.post("/query", json={"question": "重排序是怎么做的"})
        assert r.status_code == 200
        body = r.json()
        assert "答案" in body["answer"]
        assert body["sources"][0]["article_id"] == "week5/33.重排序.md"
        assert body["seconds"] >= 0
    # 必须用 with_context=True 拿依据，否则服务层报不出出处
    assert pipe.calls == [("重排序是怎么做的", False, True)]


def test_query_rejects_empty_question():
    with _client(pipeline=_FakePipeline(), agent=_FakeAgent()) as c:
        assert c.post("/query", json={"question": ""}).status_code == 422


def test_agent_endpoint_surfaces_needs_human_and_steps():
    with _client(pipeline=_FakePipeline(), agent=_FakeAgent()) as c:
        r = c.post("/agent", json={"question": "红烧肉怎么做"})
        assert r.status_code == 200
        body = r.json()
        assert body["needs_human"] is False
        assert body["steps"] == 1
        assert body["stop_reason"] == "answered"
        assert body["citations"][0]["article_id"] == "week5/33.重排序.md"


def test_agent_needs_human_is_passed_through():
    hitl = AgentResult(
        answer="笔记里没有讲到。", citations=[], steps=[{"step": 1, "tool": "search_notes", "ok": True}],
        stop_reason="max_steps", needs_human=True,
    )
    with _client(pipeline=_FakePipeline(), agent=_FakeAgent(hitl)) as c:
        body = c.post("/agent", json={"question": "红烧肉怎么做"}).json()
        assert body["needs_human"] is True


def test_unready_pipeline_returns_503():
    """模型没加载好时必须明确报 503，而不是抛 500 或挂住。"""
    with TestClient(create_app(pipeline=_FakePipeline(), agent=_FakeAgent())) as c:
        from service import STATE

        pipe = STATE.pop("pipeline")
        try:
            assert c.post("/query", json={"question": "x"}).status_code == 503
        finally:
            STATE["pipeline"] = pipe
