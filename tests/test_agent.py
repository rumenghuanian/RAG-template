"""Agent 层的回归测试。

核心思路：**LLM 是注入的**，所以用假 LLM 就能把 Agent 循环（工具调用、
重试、步数上限、引用收集、转人工判定）测完整，不需要 API key 和网络。

跑法：
    python -m pytest tests/ -v
    python tests/test_agent.py
"""
import os
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.documents import Document  # noqa: E402

from agent.llm import LLMReply, ToolCall  # noqa: E402
from agent.loop import RAGAgent  # noqa: E402
from agent.memory import ConversationMemory  # noqa: E402
from agent.tools import ToolRegistry, build_tools  # noqa: E402
from rag_core.skill import RAGSkill  # noqa: E402
from rag_core.skills.notes.metadata import notes_metadata_extractor  # noqa: E402

CORPUS = Path(__file__).resolve().parent.parent.parent / "Agent-100-Days"

try:
    import pytest
except ImportError:
    pytest = None


def _skip(reason: str) -> None:
    if pytest is not None and os.environ.get("PYTEST_CURRENT_TEST"):
        pytest.skip(reason)
    print(f"  skip: {reason}")


# --------------------------------------------------------------------------
# 测试替身
# --------------------------------------------------------------------------
class FakeLLM:
    """按脚本依次返回回复；只剩一条时反复返回它（模拟模型一直想调工具）。"""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def chat(self, messages, tools=None):
        self.calls.append({"messages": [dict(m) for m in messages], "tools": tools})
        if not self.replies:
            raise AssertionError("FakeLLM 脚本已用完")
        return self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]


def _doc(article_id, title, week, content="正文内容"):
    return Document(
        page_content=content,
        metadata={
            "article_id": article_id,
            "title": title,
            "week": week,
            "article_no": 1,
            "parent_id": f"p-{article_id}",
        },
    )


DOCS = [
    _doc("week5/29.RAG是怎么工作的.md", "RAG 是怎么工作的", 5),
    _doc("week5/32.检索技术.md", "检索技术", 5),
    _doc("week9/57.为什么 Agent 一定需要反思.md", "为什么 Agent 一定需要反思", 9),
]


class FakeIndex:
    """鸭子类型替身：build_tools 只用到 .documents / .retrieve / .skill。

    `.skill` 现在是**必须的**：工具参数名与 prompt 措辞都从领域身份来，
    没有 skill 就没有"按 week 筛选"这个维度（也不该凭空出现 —— 见 test_fixes 里
    「schema 里不许出现别的领域字段」那条）。这里声明成 notes 的最小身份。
    """

    def __init__(self, hits=None, docs=None):
        self.documents = docs if docs is not None else DOCS
        self._hits = hits if hits is not None else DOCS
        self.queries = []
        self.skill = RAGSkill(
            name="notes", agent_identity={"browse_arg": "week", "browse_desc": "，可按周筛选"}
        )

    def retrieve(self, question, top_k=None):
        self.queries.append(question)
        hits = self._hits[: (top_k or len(self._hits))]
        # 真实检索会给每个命中写 vector_cos（相关性闸门用它判断语料里有没有）
        for h in hits:
            h.metadata.setdefault("vector_cos", 0.9)
        return hits


def _tool_call(name, **arguments):
    return LLMReply(tool_calls=[ToolCall(name=name, arguments=arguments, id=f"c-{name}")])


# --------------------------------------------------------------------------
# 工具层
# --------------------------------------------------------------------------
def test_relevance_gate_rejects_irrelevant_hits():
    """检索永远返回 K 条，靠余弦阈值判断「语料里没有这件事」。"""
    hits = [
        Document(
            page_content="x",
            metadata={**DOCS[0].metadata, "vector_cos": 0.42},
        )
    ]
    out = build_tools(FakeIndex(hits=hits), min_relevance=0.65).invoke(
        "search_notes", {"query": "今天天气怎么样"}
    )
    assert out["results"] == []
    assert out["max_similarity"] == 0.42
    assert "没找到足够相关" in out["note"]
    assert "0.42" in out["note"] and "0.65" in out["note"]
    # 闸门只是提示不是判决：必须给出下一步动作
    assert "list_articles" in out["note"]
    assert "不要编造" in out["note"]


def test_relevance_gate_passes_relevant_hits():
    hits = [
        Document(
            page_content="x",
            metadata={**DOCS[0].metadata, "vector_cos": 0.86},
        )
    ]
    out = build_tools(FakeIndex(hits=hits), min_relevance=0.65).invoke(
        "search_notes", {"query": "RAG"}
    )
    assert len(out["results"]) == 1
    assert out["max_similarity"] == 0.86


def test_low_relevance_leads_to_human_handoff():
    """闸门 + 转人工要串起来：语料里没有 → 零出处 → 转人工。"""
    hits = [
        Document(page_content="x", metadata={**DOCS[0].metadata, "vector_cos": 0.42})
    ]
    llm = FakeLLM(
        [
            _tool_call("search_notes", query="Rust 操作系统内核"),
            LLMReply(content="笔记里没有找到相关内容。"),
        ]
    )
    result = RAGAgent(
        llm, build_tools(FakeIndex(hits=hits), min_relevance=0.65)
    ).run("怎么用 Rust 写操作系统内核？")
    assert result.citations == []
    assert result.needs_human is True


def test_citations_only_include_articles_actually_quoted():
    """检索到 3 篇、答案只引用了 1 篇 → 出处只列 1 篇。"""
    llm = FakeLLM(
        [
            _tool_call("search_notes", query="RAG"),
            LLMReply(content="分索引、检索、生成三步（见 `week5/29.RAG是怎么工作的.md`）。"),
        ]
    )
    result = RAGAgent(llm, build_tools(FakeIndex())).run("RAG 是怎么工作的？")
    assert [c["article_id"] for c in result.citations] == ["week5/29.RAG是怎么工作的.md"]


def test_citations_match_article_id_without_extension():
    llm = FakeLLM(
        [
            _tool_call("search_notes", query="RAG"),
            LLMReply(content="参考 week5/29.RAG是怎么工作的 这一篇。"),
        ]
    )
    result = RAGAgent(llm, build_tools(FakeIndex())).run("RAG")
    assert [c["article_id"] for c in result.citations] == ["week5/29.RAG是怎么工作的.md"]


def test_citations_fall_back_when_model_cites_nothing():
    llm = FakeLLM(
        [
            _tool_call("search_notes", query="RAG"),
            LLMReply(content="分索引、检索、生成三步。"),
        ]
    )
    result = RAGAgent(llm, build_tools(FakeIndex())).run("RAG")
    assert len(result.citations) == 3  # 模型没引用 → 保留全部检索结果
    assert result.needs_human is False


def test_all_searches_gated_triggers_human_handoff_even_with_catalog_citations():
    """「Rust 写内核」那轮的真实形状：检索全被拦，模型引用了目录条目。

    list_articles 给的是目录，引用它不等于有内容依据 —— 不该因此判定「有依据」。
    """
    hits = [
        Document(page_content="x", metadata={**DOCS[0].metadata, "vector_cos": 0.62})
    ]
    llm = FakeLLM(
        [
            _tool_call("search_notes", query="Rust 写操作系统内核"),
            _tool_call("list_articles"),
            LLMReply(content="笔记里没有相关内容。现有实现见 week5/29.RAG是怎么工作的.md。"),
        ]
    )
    result = RAGAgent(
        llm, build_tools(FakeIndex(hits=hits), min_relevance=0.65)
    ).run("怎么用 Rust 写操作系统内核？")
    assert result.citations  # 目录条目确实被引用了
    assert result.needs_human is True  # 但内容层面依旧没有依据


def test_catalog_answer_without_any_search_is_not_flagged():
    """「第 5 周讲了什么」：只用 list_articles，答案是就该如此，不该判定无依据。"""
    llm = FakeLLM(
        [
            _tool_call("list_articles", week=5),
            LLMReply(content="第 5 周讲 RAG，共 7 篇，见 week5/29.RAG是怎么工作的.md。"),
        ]
    )
    result = RAGAgent(llm, build_tools(FakeIndex())).run("第 5 周讲了什么？")
    assert result.needs_human is False
    assert [c["article_id"] for c in result.citations] == ["week5/29.RAG是怎么工作的.md"]


def test_retry_after_gate_succeeds_is_not_flagged():
    """一次被拦、换关键词搜到 → 不该判定无依据。"""
    hits = [
        Document(page_content="x", metadata={**DOCS[0].metadata, "vector_cos": 0.9})
    ]
    gated = Document(page_content="x", metadata={**DOCS[0].metadata, "vector_cos": 0.3})

    class TwoShotIndex(FakeIndex):
        def retrieve(self, question, top_k=None):
            self.queries.append(question)
            hits_ = gated if len(self.queries) == 1 else hits
            for h in hits_:
                h.metadata.setdefault("vector_cos", 0.9)
            return hits_

    llm = FakeLLM(
        [
            _tool_call("search_notes", query="那个东西"),
            _tool_call("search_notes", query="重排序"),
            LLMReply(content="见 week5/29.RAG是怎么工作的.md。"),
        ]
    )
    result = RAGAgent(llm, build_tools(TwoShotIndex())).run("那个东西怎么做？")
    assert result.needs_human is False


def test_schemas_are_openai_function_shaped():
    tools = build_tools(FakeIndex())
    assert set(tools.names()) == {"search_notes", "read_article", "list_articles"}
    for schema in tools.schemas():
        assert schema["type"] == "function"
        assert schema["function"]["name"]
        assert "parameters" in schema["function"]


def test_unknown_tool_returns_error_not_raise():
    out = build_tools(FakeIndex()).invoke("delete_everything", {})
    assert "error" in out
    assert "search_notes" in out["error"]


def test_bad_arguments_return_error():
    out = build_tools(FakeIndex()).invoke("search_notes", {})  # 少了必填 query
    assert "error" in out


def test_search_notes_returns_citation_fields():
    out = build_tools(FakeIndex()).invoke("search_notes", {"query": "RAG", "top_k": 2})
    assert len(out["results"]) == 2
    first = out["results"][0]
    assert first["article_id"] and first["title"] and first["week"] == 5
    assert first["snippet"]


def test_search_notes_empty_result_gives_hint():
    out = build_tools(FakeIndex(hits=[])).invoke("search_notes", {"query": "不存在的东西"})
    assert out["results"] == []
    assert "list_articles" in out["note"]


def test_read_article_returns_full_content():
    long_text = "很长的正文" * 50
    docs = [_doc("week5/29.RAG是怎么工作的.md", "RAG 是怎么工作的", 5, long_text)]
    out = build_tools(FakeIndex(docs=docs)).invoke(
        "read_article", {"article_id": "week5/29.RAG是怎么工作的.md"}
    )
    assert out["content"] == long_text
    assert out["title"] == "RAG 是怎么工作的"


def test_read_article_missing_suggests_available():
    out = build_tools(FakeIndex()).invoke("read_article", {"article_id": "week99/x.md"})
    assert "error" in out
    assert out["available_sample"]


def test_list_articles_filters_by_week():
    tools = build_tools(FakeIndex())
    assert tools.invoke("list_articles", {})["count"] == 3
    assert tools.invoke("list_articles", {"week": 5})["count"] == 2
    assert tools.invoke("list_articles", {"week": 9})["count"] == 1
    assert tools.invoke("list_articles", {"week": 3})["count"] == 0


# --------------------------------------------------------------------------
# Agent 循环
# --------------------------------------------------------------------------
def test_single_search_then_answer():
    llm = FakeLLM(
        [
            _tool_call("search_notes", query="RAG 是怎么工作的"),
            LLMReply(content="RAG 分索引、检索、生成三步。"),
        ]
    )
    index = FakeIndex()
    result = RAGAgent(llm, build_tools(index)).run("RAG 是怎么工作的？")

    assert result.answer == "RAG 分索引、检索、生成三步。"
    assert result.stop_reason == "answered"
    assert result.needs_human is False
    assert [s["tool"] for s in result.steps] == ["search_notes"]
    assert {c["article_id"] for c in result.citations} == {
        "week5/29.RAG是怎么工作的.md",
        "week5/32.检索技术.md",
        "week9/57.为什么 Agent 一定需要反思.md",
    }
    # 第二轮请求必须带着工具结果回去
    roles = [m["role"] for m in llm.calls[1]["messages"]]
    assert "tool" in roles


def test_agent_retries_with_a_new_query():
    """Agentic RAG 的关键：第一次没搜到，换关键词再搜。"""
    llm = FakeLLM(
        [
            _tool_call("search_notes", query="那个东西"),
            _tool_call("search_notes", query="反思 机制"),
            LLMReply(content="反思有三种触发机制。"),
        ]
    )
    index = FakeIndex()
    result = RAGAgent(llm, build_tools(index)).run("那个东西是怎么触发的？")

    assert index.queries == ["那个东西", "反思 机制"]
    assert len(result.steps) == 2
    assert result.stop_reason == "answered"


def test_citations_are_deduped_across_steps():
    llm = FakeLLM(
        [
            _tool_call("search_notes", query="RAG"),
            _tool_call("search_notes", query="检索"),
            LLMReply(content="答案"),
        ]
    )
    result = RAGAgent(llm, build_tools(FakeIndex())).run("RAG")
    assert len(result.citations) == 3  # 两次检索都返回同样 3 篇，去重后仍是 3


def test_agent_records_the_context_it_actually_read():
    """**为什么需要**：判「答案有没有编造」必须拿**模型实际看到的原文**当参照。

    `citations` 只有 article_id/title，没有正文。缺 `contexts` 这个字段时，
    评测只能拿 gold 当参照，于是「引用了上下文里其他文章」被误判成编造 ——
    这条测试把它钉住。
    """
    llm = FakeLLM(
        [
            _tool_call("search_notes", query="RAG"),
            _tool_call("read_article", article_id="week5/29.RAG是怎么工作的.md"),
            LLMReply(content="答案（week5/29.RAG是怎么工作的.md）"),
        ]
    )
    result = RAGAgent(llm, build_tools(FakeIndex())).run("RAG 是怎么工作的？")

    tools_used = [c["tool"] for c in result.contexts]
    assert "search_notes" in tools_used, "检索到的片段必须记进上下文"
    assert "read_article" in tools_used, "读到的全文必须记进上下文"
    assert all(c["text"] for c in result.contexts), "每条上下文都得有正文"
    read = [c for c in result.contexts if c["tool"] == "read_article"]
    assert read and read[0]["article_id"] == "week5/29.RAG是怎么工作的.md"


def test_catalog_listing_is_not_counted_as_context():
    """目录只有标题、没有正文，**不能**算进上下文。

    把它当上下文会虚增「有依据」—— needs_human 的判定里踩过同一个坑
    （「Rust 写内核」那轮被目录条目误判成有依据）。
    """
    llm = FakeLLM([_tool_call("list_articles"), LLMReply(content="第 5 周讲了 RAG。")])
    result = RAGAgent(llm, build_tools(FakeIndex())).run("第 5 周讲了什么？")
    assert result.contexts == []


def test_context_recording_respects_the_char_cap():
    """多步循环不能把上下文无限累积（灌给裁判也没意义）。"""
    from agent.loop import _MAX_CONTEXT_CHARS

    docs = [_doc(f"week5/{i}.长文.md", f"长文{i}", 5, content="正" * 3000) for i in range(40)]
    llm = FakeLLM([_tool_call("search_notes", query="长文")] * 40 + [LLMReply(content="答案")])
    result = RAGAgent(llm, build_tools(FakeIndex(docs=docs)), max_steps=40).run("长文")
    total = sum(len(c["text"]) for c in result.contexts)
    assert total <= _MAX_CONTEXT_CHARS + 3000, f"上下文没被限住：{total}"


def test_max_steps_forces_a_final_answer():
    llm = FakeLLM(
        [
            _tool_call("search_notes", query="a"),
            _tool_call("search_notes", query="b"),
            LLMReply(content="根据已有信息，结论是……"),
        ]
    )
    result = RAGAgent(llm, build_tools(FakeIndex()), max_steps=2).run("一直搜不到的问题")

    assert result.stop_reason == "max_steps"
    assert result.answer == "根据已有信息，结论是……"
    assert len(result.steps) == 2
    # 最后那次调用不能再带工具，否则模型还会继续调
    assert llm.calls[-1]["tools"] is None


def test_needs_human_when_nothing_was_retrieved():
    llm = FakeLLM(
        [
            _tool_call("search_notes", query="毫不相关"),
            LLMReply(content="笔记里没有找到相关内容。"),
        ]
    )
    result = RAGAgent(llm, build_tools(FakeIndex(hits=[]))).run("毫不相关的问题")
    assert result.citations == []
    assert result.needs_human is True  # 调了工具但零依据 → 转人工


def test_chitchat_does_not_trigger_human_handoff():
    llm = FakeLLM([LLMReply(content="你好，我是笔记助手。")])
    result = RAGAgent(llm, build_tools(FakeIndex())).run("你好")
    assert result.steps == []
    assert result.needs_human is False
    assert result.stop_reason == "answered"


def test_empty_reply_is_flagged():
    llm = FakeLLM([LLMReply(content="   ")])
    result = RAGAgent(llm, build_tools(FakeIndex())).run("随便问问")
    assert result.stop_reason == "empty_reply"


def test_tool_error_goes_back_to_model_and_loop_continues():
    llm = FakeLLM(
        [
            _tool_call("read_article", article_id="week99/nope.md"),
            LLMReply(content="那篇文章不存在，我改用检索。"),
        ]
    )
    result = RAGAgent(llm, build_tools(FakeIndex())).run("读一下 week99")
    assert result.steps[0]["ok"] is False
    assert "没有这篇文章" in result.steps[0]["error"]
    assert result.stop_reason == "answered"
    tool_msg = [m for m in llm.calls[1]["messages"] if m["role"] == "tool"][-1]
    assert "error" in tool_msg["content"]


def test_history_is_passed_to_the_model():
    llm = FakeLLM([LLMReply(content="好的")])
    history = [
        {"role": "user", "content": "上一轮问题"},
        {"role": "assistant", "content": "上一轮回答"},
    ]
    RAGAgent(llm, build_tools(FakeIndex())).run("这一轮问题", history=history)
    contents = [m["content"] for m in llm.calls[0]["messages"]]
    assert "上一轮问题" in contents
    assert "这一轮问题" in contents


# --------------------------------------------------------------------------
# 会话记忆
# --------------------------------------------------------------------------
def test_condense_without_history_is_a_noop():
    assert ConversationMemory().condense("它呢？", FakeLLM([])) == "它呢？"


def test_condense_rewrites_followup_question():
    memory = ConversationMemory()
    memory.add_user("RAG 是怎么工作的？")
    memory.add_assistant("分三步。")
    llm = FakeLLM([LLMReply(content="RAG 的检索阶段是怎么工作的？")])
    assert memory.condense("那检索阶段呢？", llm) == "RAG 的检索阶段是怎么工作的？"


def test_condense_falls_back_when_llm_fails():
    class BrokenLLM:
        def chat(self, messages, tools=None):
            raise RuntimeError("网络挂了")

    memory = ConversationMemory()
    memory.add_user("上一句")
    assert memory.condense("它呢？", BrokenLLM()) == "它呢？"


def test_memory_trims_old_turns():
    memory = ConversationMemory(max_turns=2)
    for i in range(5):
        memory.add_user(f"问{i}")
        memory.add_assistant(f"答{i}")
    assert len(memory.messages()) == 4
    assert memory.messages()[0]["content"] == "问3"


# --------------------------------------------------------------------------
# 语料元数据
# --------------------------------------------------------------------------
def test_notes_metadata_from_path_and_h1():
    doc = Document(
        page_content="# 29.RAG 是怎么工作的\n\n正文",
        # 用正斜杠：Linux 的 Path 不把 `\` 当分隔符，写成 r"F:\..." 会让
        # 整条路径变成一个 part，week 解析成 None（CI 在 ubuntu 上就是这么挂的）
        metadata={"source": "F:/x/Agent-100-Days/week5/29.RAG是怎么工作的.md"},
    )
    meta = notes_metadata_extractor(doc)
    assert meta["week"] == 5
    assert meta["article_no"] == 29
    assert meta["title"] == "RAG 是怎么工作的"
    assert meta["article_id"] == "week5/29.RAG是怎么工作的.md"
    assert meta["is_thinking"] is False


def test_notes_metadata_marks_thinking_articles():
    doc = Document(
        page_content="# 思考 & 补充学习资料\n",
        metadata={"source": "F:/x/Agent-100-Days/week7/49.思考 & 补充学习资料.md"},
    )
    meta = notes_metadata_extractor(doc)
    assert meta["is_thinking"] is True
    assert meta["title"] == "思考 & 补充学习资料"


def test_notes_metadata_on_real_corpus():
    if not CORPUS.is_dir():
        _skip(f"语料目录不存在: {CORPUS}")
        return
    files = sorted(CORPUS.rglob("week*/*.md"))
    assert len(files) == 105, f"语料应该是 105 篇，实际 {len(files)}"

    metas = []
    for f in files:
        doc = Document(
            page_content=f.read_text(encoding="utf-8"),
            metadata={"source": str(f)},
        )
        metas.append(notes_metadata_extractor(doc))

    assert all(m["week"] and m["article_no"] for m in metas), "有文章没解析出 week/article_no"
    assert all(m["title"] for m in metas), "有文章没有标题"
    assert all(m["article_id"].startswith("week") for m in metas)
    assert len({m["article_id"] for m in metas}) == 105, "article_id 必须唯一"
    assert sorted({m["week"] for m in metas}) == list(range(1, 17))
    assert sum(1 for m in metas if m["is_thinking"]) == 16, "每周一篇「思考」（含 week16 进阶补充）"


if __name__ == "__main__":
    tests = [
        v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)
    ]
    failed = []
    for t in tests:
        try:
            t()
        except Exception:
            failed.append(t.__name__)
            print(f"FAIL {t.__name__}")
            traceback.print_exc()
        else:
            print(f"PASS {t.__name__}")
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    if failed:
        print("failed:", ", ".join(failed))
    sys.exit(1 if failed else 0)
