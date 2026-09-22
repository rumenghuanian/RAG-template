"""basic_rag 缺陷修复的回归测试。

跑法（在项目自己的 venv 里）：
    python -m pytest tests/ -v
    python tests/test_fixes.py          # 不装 pytest 也能跑

不需要 faiss / langchain-huggingface 的用例在任何环境都能跑；
需要完整依赖的用例会自动跳过。
"""
import contextlib
import os
import shutil
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))

from langchain_core.documents import Document  # noqa: E402

import scoring  # noqa: E402
from rag_core import RAGConfig, RAGSkill  # noqa: E402
from rag_core.generator import GenerationIntegrationModule  # noqa: E402
from rag_core.retriever import RetrievalOptimizationModule  # noqa: E402
from rag_core.reranker import CrossEncoderReranker  # noqa: E402
from rag_core.splitter import DocumentSplitter  # noqa: E402
from rag_core.tokenizer import estimate_tokens, tokenize_for_bm25  # noqa: E402

try:  # 完整依赖（faiss / langchain-huggingface）装了才跑相关用例
    import faiss  # noqa: F401
    import langchain_huggingface  # noqa: F401

    HAS_FULL_DEPS = True
except ImportError:
    HAS_FULL_DEPS = False

try:
    import pytest
except ImportError:
    pytest = None


def _skip(reason: str) -> None:
    # PYTEST_CURRENT_TEST 只在 pytest 运行期存在，否则 pytest.skip 会直接抛出
    if pytest is not None and os.environ.get("PYTEST_CURRENT_TEST"):
        pytest.skip(reason)
    print(f"  skip: {reason}")


def _has_jieba() -> bool:
    try:
        import jieba  # noqa: F401
    except ImportError:
        return False
    return True


@contextlib.contextmanager
def _isolated_env():
    """导入评测脚本时用：它们在**导入期**就 load_dotenv()，会把 .env.notes 的
    DATA_PATH / FILE_GLOB / RERANK_DEVICE 灌进 os.environ，污染后面按默认值断言的
    用例（实测：`FILE_GLOB=week*/*.md` 漏出去，把 _tmp_config 的用例搞挂）。
    """
    saved_env, saved_cwd = dict(os.environ), os.getcwd()
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(saved_env)
        os.chdir(saved_cwd)


# --------------------------------------------------------------------------
# 缺陷 ①：BM25 中文分词
# --------------------------------------------------------------------------
def test_chinese_sentence_is_not_a_single_token():
    """默认的 text.split() 会把整句中文变成一个 token，这里必须切成多个。"""
    assert len(tokenize_for_bm25("宫保鸡丁怎么做")) > 1


def test_query_and_document_share_tokens():
    """检索能工作的前提：查询和文档有共同 token。"""
    query_tokens = set(tokenize_for_bm25("宫保鸡丁怎么做"))
    doc_tokens = set(tokenize_for_bm25("## 操作\n先腌鸡丁，再炒宫保鸡丁，出锅前放花生。"))
    assert query_tokens & doc_tokens


def test_langchain_default_preprocess_is_broken_for_chinese():
    """钉住「为什么要自己写分词」，谁改回默认这条就红。"""
    from langchain_community.retrievers.bm25 import default_preprocessing_func

    assert len(default_preprocessing_func("宫保鸡丁怎么做")) == 1
    assert len(tokenize_for_bm25("宫保鸡丁怎么做")) > 1


def test_ascii_words_are_split_and_lowercased():
    if _has_jieba():  # jieba 不保证大小写，跳过
        return
    assert tokenize_for_bm25("Add 20g Sugar") == ["add", "20", "g", "sugar"]


def test_empty_and_punctuation_only_text():
    assert tokenize_for_bm25("") == []
    assert tokenize_for_bm25("！！！，。") == []


def test_estimate_tokens_is_proportional_to_chinese_length():
    assert estimate_tokens("") == 0
    # 「步骤。」×233 = 699 字符，其中汉字 466 个
    assert estimate_tokens("步骤。" * 233) == 466


# --------------------------------------------------------------------------
# 缺陷 ②：上下文按 token 预算截断，而不是遇到大文档就 break
# --------------------------------------------------------------------------
def _generator(context_max_tokens: int) -> GenerationIntegrationModule:
    """build_context 只用到 context_max_tokens，所以不必真的建 LLM 客户端。"""
    gen = object.__new__(GenerationIntegrationModule)
    gen.context_max_tokens = context_max_tokens
    return gen


def test_generator_ctor_takes_context_max_tokens():
    """钉住参数改名：pipeline 传的是 context_max_tokens，传错会 TypeError。"""
    import inspect

    params = inspect.signature(GenerationIntegrationModule.__init__).parameters
    assert "context_max_tokens" in params
    assert "context_max_length" not in params


def test_all_three_docs_reach_context():
    """修复前：预算 2000 字符，第 3 篇超预算直接被 break 丢掉。"""
    gen = _generator(context_max_tokens=2000)
    docs = [
        Document(page_content="步骤。" * 233, metadata={"dish_name": f"菜{i}"})
        for i in (1, 2, 3)
    ]
    ctx = gen.build_context(docs)
    for i in (1, 2, 3):
        assert f"菜{i}" in ctx, f"菜{i} 没进上下文"
    assert "已截断" not in ctx  # 2000 token 够装 3 × 466


def test_oversized_doc_is_truncated_so_later_docs_still_fit():
    """一篇超预算的长文不该拦掉后面的文档。"""
    gen = _generator(context_max_tokens=1000)
    docs = [
        Document(page_content="步骤" * 900, metadata={"dish_name": "大文档"}),
        Document(page_content="小", metadata={"dish_name": "小文档"}),
    ]
    ctx = gen.build_context(docs)
    assert "大文档" in ctx
    assert "已截断" in ctx
    assert "小文档" not in ctx  # 截断已经吃满预算


def test_empty_docs():
    assert _generator(1000).build_context([]) == "暂无相关信息。"


def test_header_carries_metadata():
    gen = _generator(1000)
    ctx = gen.build_context(
        [Document(page_content="内容", metadata={"dish_name": "宫保鸡丁", "difficulty": "困难"})]
    )
    assert "宫保鸡丁" in ctx and "difficulty: 困难" in ctx


# --------------------------------------------------------------------------
# 缺陷 ⑤：过滤下推到检索层
# --------------------------------------------------------------------------
class _FakeVectorStore:
    """只记录调用参数的假向量库，用来验证过滤参数确实传下去了。"""

    def __init__(self):
        self.calls = []

    def similarity_search_with_score(self, query, **kwargs):
        self.calls.append((query, kwargs))
        # (doc, L2 距离)；距离小 → 余弦高
        return [(Document(page_content="x", metadata={"chunk_id": "c1"}), 0.2)]


def _retriever_shell(candidate_k: int = 20) -> RetrievalOptimizationModule:
    """绕过 __init__（它要建 BM25，需要 rank_bm25），只测检索参数构造。"""
    inst = object.__new__(RetrievalOptimizationModule)
    inst.vectorstore = _FakeVectorStore()
    inst.candidate_k = candidate_k
    return inst


def test_filters_are_pushed_into_vector_search():
    rt = _retriever_shell()
    rt._vector_search("宫保鸡丁", {"category": "荤菜"})
    kwargs = rt.vectorstore.calls[0][1]
    assert callable(kwargs["filter"]), "过滤条件没有下推给 FAISS"
    assert kwargs["fetch_k"] > kwargs["k"], "fetch_k 没放大，过滤后会被筛空"


def test_no_filters_means_plain_search():
    rt = _retriever_shell()
    rt._vector_search("宫保鸡丁", None)
    assert "filter" not in rt.vectorstore.calls[0][1]


def test_filter_predicate_matches_metadata():
    pred = RetrievalOptimizationModule._filter_predicate({"category": "素菜"})
    assert pred(Document(page_content="", metadata={"category": "素菜"}))
    assert not pred(Document(page_content="", metadata={"category": "荤菜"}))
    assert not pred(Document(page_content="", metadata={}))  # 缺字段 = 不匹配


def test_filter_predicate_accepts_list_values():
    pred = RetrievalOptimizationModule._filter_predicate({"difficulty": ["简单", "中等"]})
    assert pred(Document(page_content="", metadata={"difficulty": "中等"}))
    assert not pred(Document(page_content="", metadata={"difficulty": "困难"}))


def _planner(token_df: dict) -> RetrievalOptimizationModule:
    rt = object.__new__(RetrievalOptimizationModule)
    rt.token_df = token_df
    return rt


def test_plan_routes_rare_identifiers_to_lexical():
    """含只在少数文章出现的标识符 → 加权偏向 BM25（实测这组 BM25 100% / 等权 RRF 87.5%）。"""
    query = "AgentExecutor 是什么"
    # 用分词器自己的结果造 fixture，避免写死某个分词器的切法
    rt = _planner({t.lower(): 1 for t in tokenize_for_bm25(query)})
    plan = rt.plan(query)
    assert plan["mode"] == "lexical"
    assert plan["weights"][1] > plan["weights"][0]  # BM25 权重更大
    assert plan["signals"]


def test_plan_lookup_is_case_insensitive():
    """语料里写 AgentExecutor，用户可能敲 agentexecutor。"""
    rt = _planner({"agentexecutor": 1})
    assert rt.plan("agentexecutor 是什么")["mode"] == "lexical"


def test_plan_ignores_rare_chinese_phrases():
    """中文稀有二字组只是措辞罕见，不是精确标识符，不该触发 lexical。"""
    rt = _planner({"最该": 1, "段落": 30})
    assert rt.plan("检索出来一堆段落，怎么让最该看的排到最前面？")["mode"] == "semantic"


def test_plan_treats_unknown_terms_as_not_rare():
    """df=0 是「语料里根本没这个词」，那是答不了的信号，不是该走字面检索的信号。"""
    rt = _planner({"什么": 90})
    assert rt.plan("Kubernetes 怎么部署")["mode"] == "semantic"


def test_plan_common_queries_stay_semantic():
    rt = _planner({"rag": 30, "工作": 60})
    plan = rt.plan("RAG 是怎么工作的")
    assert plan["mode"] == "semantic"
    assert plan["weights"] == (1.0, 1.0)


def test_rrf_weights_shift_the_ranking():
    """权重真的能改变名次，否则 plan() 等于没接线。"""
    only_v = Document(page_content="v", metadata={"chunk_id": "v"})
    only_b = Document(page_content="b", metadata={"chunk_id": "b"})
    equal = RetrievalOptimizationModule._rrf_rerank([only_v], [only_b])
    assert {d.metadata["chunk_id"] for d in equal[:2]} == {"v", "b"}
    lexical = RetrievalOptimizationModule._rrf_rerank(
        [only_v], [only_b], weights=(0.6, 1.4)
    )
    assert lexical[0].metadata["chunk_id"] == "b"


def test_vector_cosine_is_recovered_from_l2_distance():
    """归一化向量 + 欧氏距离：d^2 = 2 - 2cos。RRF 分数没有区分度，靠这个判断相关性。"""
    import math

    doc = Document(page_content="x", metadata={})
    distance = math.sqrt(2 - 2 * 0.86)
    out = RetrievalOptimizationModule._with_cosine(doc, distance)
    assert abs(out.metadata["vector_cos"] - 0.86) < 1e-9


def test_rrf_prefers_doc_hit_by_both_retrievers():
    shared = Document(page_content="a", metadata={"chunk_id": "shared"})
    only_vector = Document(page_content="b", metadata={"chunk_id": "v"})
    only_bm25 = Document(page_content="c", metadata={"chunk_id": "b"})
    ranked = RetrievalOptimizationModule._rrf_rerank(
        [shared, only_vector], [shared, only_bm25]
    )
    assert ranked[0].metadata["chunk_id"] == "shared"
    assert len(ranked) == 3  # chunk_id 去重后不该重复


# --------------------------------------------------------------------------
# 缺陷 ③：指纹要覆盖分块配置
# --------------------------------------------------------------------------
def test_rrf_keeps_vector_cosine_when_both_retrievers_hit():
    """同一个 chunk 两路都命中时，BM25 的对象不能把带 vector_cos 的向量对象顶掉。

    闸门（search_notes 的 min_relevance）读的就是这个字段。曾经它被无条件覆盖成
    None，下游按 0.00 处理，于是**几乎所有正常问题都被判成「语料里没有这件事」**，
    Agent 一律返回「信息不足」——而 --check 只验工具调用，验不出这个。
    """
    vector_side = Document(
        page_content="同一块", metadata={"chunk_id": "c1", "vector_cos": 0.86}
    )
    bm25_side = Document(page_content="同一块", metadata={"chunk_id": "c1"})
    ranked = RetrievalOptimizationModule._rrf_rerank([vector_side], [bm25_side])
    assert len(ranked) == 1  # 去重后仍是一块
    assert ranked[0].metadata.get("vector_cos") == 0.86


def test_rrf_keeps_bm25_only_docs():
    """BM25 单路命中的块没有余弦，但必须保留 —— 否则字面召回全被丢掉。"""
    bm25_only = Document(page_content="只有字面命中", metadata={"chunk_id": "b1"})
    ranked = RetrievalOptimizationModule._rrf_rerank([], [bm25_only])
    assert [d.metadata["chunk_id"] for d in ranked] == ["b1"]
    assert "vector_cos" not in ranked[0].metadata


# --------------------------------------------------------------------------
# 重排序：排序、退化、候选池
# --------------------------------------------------------------------------
def _docs(*ids):
    return [Document(page_content=f"内容{i}", metadata={"chunk_id": c}) for i, c in enumerate(ids)]


class _StubReranker(CrossEncoderReranker):
    """只替换打分，专测排序与元数据写入，不碰真实模型。"""

    def __init__(self, scores):
        super().__init__(model_name="stub")
        self._scores = scores

    def score(self, query, texts):
        return list(self._scores)[: len(texts)]


def test_reranker_orders_by_score_and_records_it():
    r = _StubReranker([0.1, 0.9, 0.5])
    out = r.rerank("q", _docs("a", "b", "c"))
    assert [d.metadata["chunk_id"] for d in out] == ["b", "c", "a"]
    assert out[0].metadata["rerank_score"] == 0.9


def test_reranker_truncates_to_top_n():
    r = _StubReranker([0.1, 0.9, 0.5])
    out = r.rerank("q", _docs("a", "b", "c"), top_n=2)
    assert [d.metadata["chunk_id"] for d in out] == ["b", "c"]


def test_reranker_degrades_instead_of_raising():
    """权重缺失/依赖没装时不能把检索链路带崩 —— 原样返回，截到 top_n。"""
    r = CrossEncoderReranker(model_name="stub")
    r._failed = True  # 模拟加载失败
    docs = _docs("a", "b", "c")
    out = r.rerank("q", docs, top_n=2)
    assert [d.metadata["chunk_id"] for d in out] == ["a", "b"]
    assert "rerank_score" not in out[0].metadata


def test_reranker_empty_input():
    assert _StubReranker([]).rerank("q", []) == []


class _RecordingReranker(CrossEncoderReranker):
    def __init__(self):
        super().__init__(model_name="stub")
        self.received = None

    def score(self, query, texts):
        self.received = len(texts)
        return [float(i) for i in range(len(texts))]


def _hybrid_shell(reranker, docs):
    rt = object.__new__(RetrievalOptimizationModule)
    rt.reranker = reranker
    rt.rerank_candidates = 50
    rt.candidate_k = 20
    rt.token_df = {}  # plan() → semantic
    rt._vector_search = lambda q, f: list(docs)
    rt._bm25_search = lambda q, f: []
    return rt


def test_hybrid_search_reranks_when_enabled():
    docs = _docs(*[f"c{i}" for i in range(8)])
    rt = _hybrid_shell(_RecordingReranker(), docs)
    out = rt.hybrid_search("q", top_k=3, rerank=True)
    assert len(out) == 3
    # 打分是递增的，重排后应该取到分数最高的那几条（即原列表末尾）
    assert [d.metadata["chunk_id"] for d in out] == ["c7", "c6", "c5"]


def test_hybrid_search_can_bypass_reranker():
    """评测里靠 rerank=False 把「重排前」那一臂固定住。"""
    docs = _docs(*[f"c{i}" for i in range(8)])
    rr = _RecordingReranker()
    rt = _hybrid_shell(rr, docs)
    out = rt.hybrid_search("q", top_k=3, rerank=False)
    assert [d.metadata["chunk_id"] for d in out] == ["c0", "c1", "c2"]
    assert rr.received is None, "显式关闭重排时不该加载/调用模型"


def test_rerank_pool_is_independent_of_top_k():
    """池深只能由 rerank_candidates 决定，**不能跟着调用方的 top_k 变**。

    否则 retrieve(top_k=5) 与 retrieve(top_k=10**6) 的排序口径不同，
    两臂对比测的就是池深而不是排序（实测造成 2.7pp 的假差异）。
    """
    docs = _docs(*[f"c{i}" for i in range(80)])
    rr = _RecordingReranker()
    rt = _hybrid_shell(rr, docs)
    rt.rerank_candidates = 0  # 不截断
    rt.hybrid_search("q", top_k=2, rerank=True)
    small = rr.received
    rt.hybrid_search("q", top_k=10**6, rerank=True)
    assert small == rr.received == 80, "两种 top_k 必须重排同一份候选池"


def test_rerank_pool_can_be_capped():
    docs = _docs(*[f"c{i}" for i in range(80)])
    rr = _RecordingReranker()
    rt = _hybrid_shell(rr, docs)
    rt.rerank_candidates = 20
    rt.hybrid_search("q", top_k=10**6, rerank=True)
    assert rr.received == 20, "设了上限就该截断（省时间，代价是候选变少）"


def test_hybrid_search_without_reranker_configured():
    docs = _docs("a", "b")
    rt = _hybrid_shell(None, docs)
    out = rt.hybrid_search("q", top_k=2, rerank=True)  # 要求重排但没有重排器
    assert [d.metadata["chunk_id"] for d in out] == ["a", "b"]


# --------------------------------------------------------------------------
# 生成层判分规则：弃答检测与引用检测
# --------------------------------------------------------------------------
def test_long_answer_mentioning_no_notes_is_not_abstention():
    """**回归**：长答案里出现「笔记里没有讲到 X」是正常作答，不能判成弃答。

    最初写成全文子串匹配，4 条单轮回答里 3 条被误判 —— 而「编造检测」正是看这个指标。
    """
    answer = "重排序的做法分两步：" + "先初筛再精排。" * 30 + "另外，笔记里没有讲到如何调参。"
    got, why = scoring.abstained(answer)
    assert got is False, f"被误判为弃答：{why}"


def test_short_answer_starting_with_refusal_is_abstention():
    got, why = scoring.abstained("笔记里没有讲到这个问题。")
    assert got is True and "笔记里没有" in why


def test_empty_retrieval_is_abstention():
    got, why = scoring.abstained(scoring.NO_RESULT)
    assert got is True and why == "检索为空"


def test_needs_human_signal_is_trusted():
    """Agent 的 needs_human 是系统级信号，不管答案多长都采信。"""
    got, why = scoring.abstained("这是一段很长的回答。" * 40, needs_human=True)
    assert got is True and why == "needs_human"


def test_long_explained_refusal_is_still_abstention():
    """**回归**：Agent 写的是**长而完整**的弃答，不能因为「答案很长」就判成没弃答。

    早先加了「< 250 字才算弃答」的长度门槛，结果 Agent 的标准弃答
    （「结论：笔记库里没有找到与 COBOL 相关的内容」+ 解释检索了哪些关键词）
    被判成「没弃答」，直接得出「Agent 编造率 80%」这种失真的数字。
    """
    answer = (
        "**结论：笔记库里没有找到与 COBOL 或银行核心系统相关的内容。**\n"
        + "我按「COBOL 银行核心系统」「银行/金融系统架构」等关键词检索了多轮。" * 20
    )
    got, why = scoring.abstained(answer)
    assert got is True, f"长弃答被漏判：{why}"


def test_refusal_wording_variants_are_recognised():
    """模型不会每次都用同一句话，判分不能只认一种说法。"""
    for text in [
        "笔记里没有讲到这个问题。",
        "笔记库里没有找到相关内容。",
        "笔记中未提及该主题。",
        "抱歉，这个问题超出我的知识范围。",
        "检索没有命中任何内容。",
        "信息不足，无法回答。",
        "我不能凭记忆补充。",
    ]:
        got, _ = scoring.abstained(text)
        assert got is True, f"没认出弃答说法：{text}"


def test_key_point_coverage_is_fraction_of_terms_present():
    cov, missing = scoring.key_point_coverage(
        "重排序用 Cross-Encoder 做精细打分，先初筛 Top-K。", ["Cross-Encoder", "初筛", "Top-K", "图谱构建"]
    )
    assert abs(cov - 0.75) < 1e-9
    assert missing == ["图谱构建"]


def test_key_point_coverage_is_case_insensitive():
    cov, _ = scoring.key_point_coverage("cross-encoder", ["Cross-Encoder"])
    assert cov == 1.0


def test_key_point_coverage_handles_empty_terms():
    assert scoring.key_point_coverage("随便", []) == (0.0, [])


def test_cited_any_does_not_require_gold():
    """**回归**：`partial` 类没有 gold（expect 为空），
    「有没有给出相关方向」只能看它有没有指向语料里**任意**一篇。

    早先这里误用 `cited()`（判 gold），而 gold 恒为空 → 报告里出现
    「引到相关内容 0%」的假结论，我差点据此去做一个不需要做的功能。
    """
    titles = {"week5/33.重排序.md": "重排序", "week5/31.文本分块.md": "文本分块"}
    assert scoring.cited_any("参考《重排序》与《文本分块》", titles) is True
    assert scoring.cited_any("出处：week5/31.文本分块", titles) is True
    assert scoring.cited_any("笔记里没有讲到这个", titles) is False
    assert scoring.cited_any("随便一段话", {}) is False


def test_artifact_path_is_skill_scoped():
    """**回归**：评测产物必须按 skill 分开，否则换语料跑一次就覆盖掉上一套报告。

    以及 env 文件也必须跟着 skill 走 —— 早先评测脚本写死 `.env.notes`，
    于是 `EVAL_SKILL=recipe` 只换了 prompt/元数据，**语料没换而且不报错**。
    """
    import os as _os

    import bootstrap

    # 用 os.path.join 比较，别写死 '/' —— Windows 上 join('.', 'x') 是 '.\\x'
    assert bootstrap.artifact_path("golden", "notes", ".jsonl") == _os.path.join(
        ".", "golden.jsonl"
    )
    assert bootstrap.artifact_path("golden", "recipe", ".jsonl") == _os.path.join(
        ".", "golden.recipe.jsonl"
    )
    assert bootstrap.artifact_path("seeds", "notes", ".jsonl") == _os.path.join(
        ".", "seeds.jsonl"
    )
    assert bootstrap.env_file_for("recipe") in (
        _os.path.join(".", ".env.recipe"),
        _os.path.join(".", ".env"),
    ), "recipe 该用 .env（或 .env.recipe），不能是 .env.notes"


def test_run_all_deltas_flags_only_regressions():
    """**回归**：回归门禁只能报「退化」，不能把「提升」或「缺一边」当成 FAIL。

    门禁本身也是被测对象 —— 它误报一次，你就不再信任它，等于没有。
    """
    import run_all

    base = {"a/x": 0.90, "a/y": 1.00, "b/z": 0.50}
    cur = {"a/x": 0.89, "a/y": 0.80, "b/z": 0.70, "b/new": 0.10}
    rows, bad = run_all.deltas(base, cur, tol=0.02)

    assert [k for k, *_ in bad] == ["a/y"], "只该报 a/y 的退化"
    by_key = {k: (b, c, d) for k, b, c, d in rows}
    assert by_key["a/x"][2] is not None and by_key["a/x"][2] > -0.02
    assert by_key["b/z"][2] == pytest.approx(0.20)  # 提升不拦
    assert by_key["b/new"] == (None, 0.10, None)  # 新增项不算退化


def test_run_all_deltas_respects_tolerance():
    import run_all

    rows, bad = run_all.deltas({"k": 1.0}, {"k": 0.99}, tol=0.02)
    assert bad == [], "1% 的波动在 2% 阈值内，不该报"
    rows, bad = run_all.deltas({"k": 1.0}, {"k": 0.90}, tol=0.05)
    assert [k for k, *_ in bad] == ["k"]


def test_indexer_fingerprint_includes_domain_metadata():
    """**回归**：领域元数据变了也必须重建索引。

    踩过：改了 skill 的 `metadata_extractor`（补出规范字段 `article_id`），
    但指纹只含模型+内容+来源 → 判定"索引是新的"，于是**向量检索继续返回旧元数据**
    （没有 article_id），而 BM25 那一路用的是刚加载的新文档（有 article_id）。
    两条路元数据不一致，评测崩在「检索结果缺少 article_id」。
    """
    if not HAS_FULL_DEPS:
        _skip("需要 faiss + langchain-huggingface")
        return
    from rag_core.indexer import IndexConstructionModule

    def _chunks(meta_extra):
        return [
            Document(
                page_content="同内容",
                metadata={"source": "a.md", "chunk_id": "k0", **meta_extra},
            )
        ]

    idx = IndexConstructionModule.__new__(IndexConstructionModule)
    idx.model_name = "m"
    idx.normalize_embeddings = True

    before = idx.compute_fingerprint(_chunks({"category": "荤菜"}))
    same = idx.compute_fingerprint(_chunks({"category": "荤菜"}))
    after = idx.compute_fingerprint(_chunks({"category": "素菜"}))
    assert before == same, "同样的元数据必须得到同样的指纹"
    assert before != after, "元数据变了指纹必须变，否则会加载旧索引"


def test_loader_rejects_a_skill_that_forgets_the_canonical_id():
    """**回归**：skill 不产出规范字段时，加载期就必须硬失败。

    踩过（换 recipe 领域）：`metadata_extractor` 只给 `dish_name`、没有 `article_id`，
    于是 322 篇文档的 id 全是 None，一路静默传播成
    「检索 5 篇塌成 [None]」「评测 hit@5 全线 0%」「read_article(None) 返回某一篇」。
    **比崩溃更危险**，所以要在最早的入口拦住。
    """
    import tempfile
    from pathlib import Path

    from langchain_core.documents import Document as _Doc
    from rag_core.loader import DocumentLoader

    def bad_extractor(doc):
        return {"dish_name": "宫保鸡丁"}  # 就是不给 article_id / title

    def good_extractor(doc):
        return {"article_id": "a/x.md", "title": "x"}

    with tempfile.TemporaryDirectory() as tmp:
        Path(tmp, "a.md").write_text("# x\n", encoding="utf-8")

        try:
            DocumentLoader(tmp, "*.md", bad_extractor).load()
        except ValueError as e:
            assert "article_id" in str(e)
        else:
            raise AssertionError("缺规范字段时应当报错，而不是静默变成 None")

        # 没有 extractor 时不校验（那种用法没有领域契约）
        assert len(DocumentLoader(tmp, "*.md").load()) == 1

        # 规范字段齐全就通过
        assert len(DocumentLoader(tmp, "*.md", good_extractor).load()) == 1


def test_loader_rejects_duplicate_canonical_ids():
    """id 重复会让 read_article/引用匹配指向错误的文档 —— 必须拦住。"""
    import tempfile
    from pathlib import Path

    from rag_core.loader import DocumentLoader

    def same_id_everywhere(doc):
        return {"article_id": "dup", "title": "同名"}

    with tempfile.TemporaryDirectory() as tmp:
        Path(tmp, "a.md").write_text("# a\n", encoding="utf-8")
        Path(tmp, "b.md").write_text("# b\n", encoding="utf-8")
        try:
            DocumentLoader(tmp, "*.md", same_id_everywhere).load()
        except ValueError as e:
            assert "不唯一" in str(e)
        else:
            raise AssertionError("id 重复时应当报错")


def test_recipe_metadata_provides_the_canonical_fields():
    """recipe 必须把自己领域的 id 映射到规范名（这是它以前缺的那一块）。"""
    from langchain_core.documents import Document as _Doc
    from rag_core.skills.recipe.metadata import recipe_metadata_extractor

    # 布局一：文件直接放在类别目录下
    # 路径一律用正斜杠：Linux 的 Path 不把 `\` 当分隔符，写成 r"F:\..." 会让
    # parts[-2:] 拿到整条路径 —— CI 在 ubuntu 上就是这么挂的
    flat = _Doc(
        page_content="# 宫保鸡丁\n★\n",
        metadata={"source": "F:/x/data/cook/dishes/meat_dish/宫保鸡丁.md"},
    )
    m = recipe_metadata_extractor(flat)
    assert m["article_id"] == "meat_dish/宫保鸡丁.md"   # 路径最后两段
    assert m["title"] == "宫保鸡丁"
    assert m["category"] == "荤菜"

    # 布局二：菜品子目录里 → id 仍唯一可读（语料里同一道菜两种布局都存在，
    # 用「类别+菜名」会撞车，加载期唯一性校验抓过两次）
    nested = _Doc(
        page_content="# 陈皮排骨汤\n",
        metadata={"source": "F:/x/data/cook/dishes/soup/陈皮排骨汤/陈皮排骨汤.md"},
    )
    m2 = recipe_metadata_extractor(nested)
    assert m2["title"] == "陈皮排骨汤"
    assert m2["article_id"] == "陈皮排骨汤/陈皮排骨汤.md"
    assert m2["article_id"] != m["article_id"]


def test_gate_subgroups_skips_empty_kinds():
    """**回归**：某个闸门类一条标注都没有时，不能生成空分组。

    踩过（换 recipe 领域）：诊断集只有 `absent`、没有 `mentioned`，分组里留下一个
    成员为空的组 → 下游 `sum(...)/len(vals)` 直接 ZeroDivisionError，**整个评测崩掉**。
    换任何新领域，第一份标注集都不太可能四类齐全。
    """
    with _isolated_env():
        import run_eval

        only_absent = [{"kind": "absent", "note": "far", "gate_cos": 0.3, "gate_rr": 0.1}]
        groups = run_eval.gate_subgroups(only_absent)
        assert [name for name, _ in groups] == ["absent·far"]
        assert all(members for _, members in groups), "不许有空分组"

        # 两条 absent、note 不同 → 两个组，各自有人
        two = only_absent + [
            {"kind": "absent", "note": "near", "gate_cos": 0.5, "gate_rr": 0.4}
        ]
        assert [n for n, _ in run_eval.gate_subgroups(two)] == ["absent·far", "absent·near"]

        # 空输入 → 空分组（不崩）
        assert run_eval.gate_subgroups([]) == []


def test_top_articles_refuses_docs_without_id():
    """**回归**：没有 id 的检索结果要报错，不许静默去重成 `[None]`。"""
    with _isolated_env():
        import run_eval

        docs = [Document(page_content="x", metadata={"source": "a"})]
        try:
            run_eval.top_articles(docs)
        except RuntimeError as e:
            assert "article_id" in str(e)
        else:
            raise AssertionError("缺 id 时应报错")


def test_list_articles_accepts_the_skill_declared_parameter_name():
    """**回归**：`list_articles` 的**参数名**由 skill 决定，函数必须跟着接受。

    我改的时候差点自己引入一个坑：schema 里把参数名换成 skill 的 `category`，
    而函数签名还写 `week=None` → 模型按 schema 传 `category` 会 TypeError
    （被 ToolRegistry 兜成 {"error": "参数不对"}，静默失效）。
    """
    from agent.tools import build_tools
    from langchain_core.documents import Document as _Doc

    class _Skill:
        agent_identity = {
            "corpus": "菜谱库",
            "browse_arg": "category",
            "browse_field": "category",
            "browse_desc": "（可按分类筛选）",
            "id_hint": "meat_dish/宫保鸡丁",
        }

    class _Index:
        skill = _Skill()
        documents = [
            _Doc(page_content="a", metadata={"article_id": "meat_dish/宫保鸡丁", "title": "宫保鸡丁", "category": "荤菜"}),
            _Doc(page_content="b", metadata={"article_id": "vegetable_dish/手撕包菜", "title": "手撕包菜", "category": "素菜"}),
        ]

        def retrieve(self, q, top_k=None):
            return []

    tools = build_tools(_Index())
    schema = {t["function"]["name"]: t["function"] for t in tools.schemas()}
    assert "category" in schema["list_articles"]["parameters"]["properties"]

    all_items = tools.invoke("list_articles", {})
    assert all_items["count"] == 2
    only_meat = tools.invoke("list_articles", {"category": "荤菜"})
    assert only_meat["count"] == 1
    assert only_meat["articles"][0]["article_id"] == "meat_dish/宫保鸡丁"
    # 分类不存在 → 空，而不是报错
    assert tools.invoke("list_articles", {"category": "甜品"})["count"] == 0


def test_agent_system_prompt_follows_the_skill_identity():
    """**回归**：换领域后 prompt 不能还说"课程笔记库"、示例 id 也不能还是 notes 的。

    写死的后果：食谱场景模型会被告知这是课程笔记库、并照 `week5/33…` 的格式
    编造不存在的 id。
    """
    from agent.loop import build_system_prompt
    from rag_core.skills import build_notes_skill, build_recipe_skill

    notes = build_system_prompt(build_notes_skill())
    recipe = build_system_prompt(build_recipe_skill())

    assert "课程笔记库" in notes and "week5/33.重排序.md" in notes
    assert "菜谱库" in recipe and "meat_dish/宫保鸡丁" in recipe
    assert "课程笔记库" not in recipe and "week5/33" not in recipe
    # 通用默认值：没有 skill 时不夹带任何领域
    generic = build_system_prompt(None)
    assert "课程笔记库" not in generic and "菜谱库" not in generic


def test_stale_baseline_files_are_removed_from_baseline():
    """**回归**：立基线时，上一版残留的层文件必须删掉。

    踩过两次：一次是拿了 `RERANK_ENABLED=false` 的遗留报告当基线；
    一次是生成层报告被归档后，`gen.*.json` 还留在基线目录里 —— 等新报告一生成，
    门禁就会拿**week16 之前的旧基线**去比，报出一堆假退化。
    基线目录的内容必须**恰好等于** meta 里记录的那几层。
    """
    import run_all

    existing = ["gen.notes.json", "retrieval.notes.json", "agent.notes.json", "meta.notes.json"]
    keep = ["retrieval.notes.json", "agent.notes.json", "meta.notes.json"]
    assert run_all.stale_baseline_files(existing, keep) == ["gen.notes.json"]
    assert run_all.stale_baseline_files(keep, keep) == []


def test_recorded_context_is_the_truncated_text_the_model_actually_saw():
    """**回归**：判分器看到的上下文，必须等于模型看到的上下文。

    踩过：`context_texts` 记的是**未截断的父文档全文**（21,228 字），而
    `context_max_tokens` 把真正喂进去的文本截断到 2,733 token 并追加了一行
    「…（内容过长，已截断）」。模型如实说「文档被截断」，裁判拿全文一对——
    判成「原文没有这个说法」= 编造。**唯一的"不忠实"是口径造出来的。**

    所以 `query_with_sources()` 必须返回**截断后**的正文。
    """
    gen = GenerationIntegrationModule(
        model_name="m", base_url="u", api_key="k", context_max_tokens=300
    )
    docs = [
        Document(page_content="甲" * 500, metadata={"article_id": "a.md"}),
        Document(page_content="乙" * 500, metadata={"article_id": "b.md"}),
    ]
    fitted = gen.fit_context(docs)
    assert fitted, "至少该留下一篇"
    first_doc, first_body = fitted[0]
    assert first_doc.metadata["article_id"] == "a.md"
    assert "已截断" in first_body, "超预算的正文必须带截断标记"
    # 拼进 prompt 的上下文里，出现的正是这个截断后的正文
    ctx = gen.build_context(docs)
    assert first_body in ctx
    assert "甲" * 500 not in ctx, "全文不该出现在上下文里"
    # 纯函数：重复裁剪结果一致（query_with_sources 与 generate 各调一次，必须一致）
    assert gen.fit_context(docs) == fitted
    # 预算足够时不截断、不丢标记
    gen.context_max_tokens = 100000
    assert all("已截断" not in b for _, b in gen.fit_context(docs))


def test_gate_skips_gen_layer_when_its_report_is_missing():
    """**回归**：生成层没有报告时，门禁要**跳过**，不能报 FAIL。

    踩过：把 week16 之前的旧生成报告归档之后，`run_all.py --skip-run` 直接 FAIL ——
    因为执行步骤没检查报告是否存在，就去跑 `run_gen_eval.py --rescore`，
    子脚本以退出码 1 结束（「没有可重算的报告」），门禁把它当成了指标退化。
    **缺数据要跳过并说明，不该伪装成失败。**
    """
    import run_all

    assert run_all.needs_gen_rescore(no_gen=False, report_exists=True) is True
    assert run_all.needs_gen_rescore(no_gen=False, report_exists=False) is False
    assert run_all.needs_gen_rescore(no_gen=True, report_exists=True) is False


def test_bucket_of_separates_the_three_calibers():
    """**回归**：类别口径必须只有一份，且 manual（partial）不算可答。

    `partial` 没有干净 gold（`expect` 恒为空），任何「命中」类指标对它恒为 0。
    这个 bug 在生成层修过一次（改用 `cited_any`）、Agent 对照层漏了，报告里于是
    长期挂着「partial 五个臂全是 0.0%」的假数字，而且它永远不动 = 门禁里的废行。
    指标钉在极值，先怀疑指标。
    """
    assert scoring.bucket_of("partial") == "manual"
    assert scoring.bucket_of("absent") == "gate"
    assert scoring.bucket_of("mentioned") == "gate"
    for kind in scoring.ANSWERABLE_KINDS:
        assert scoring.bucket_of(kind) == "answerable"
    assert not set(scoring.ANSWERABLE_KINDS) & set(scoring.MANUAL_KINDS)
    assert not set(scoring.ANSWERABLE_KINDS) & set(scoring.GATE_KINDS)


def test_eval_scripts_all_use_the_scoring_taxonomy():
    """四个脚本原先各抄一份类别列表 → 改一处漏三处。现在必须同源。

    顺带记一条：**评测脚本的导入是有副作用的**（见 `_isolated_env`）。
    """
    with _isolated_env():
        import build_golden
        import run_agent_eval
        import run_eval
        import run_gen_eval

        for mod in (run_eval, run_gen_eval, run_agent_eval, build_golden):
            assert set(mod.GATE_KINDS) == set(scoring.GATE_KINDS), mod.__name__
            assert set(mod.MANUAL_KINDS) == set(scoring.MANUAL_KINDS), mod.__name__
            if hasattr(mod, "ANSWERABLE"):
                assert set(mod.ANSWERABLE) == set(scoring.ANSWERABLE_KINDS), mod.__name__


def test_shuffle_control_has_no_fixed_point():
    """**回归**：裁判的对照臂必须让每条答案拿到**别题**的上下文。

    若某条拿到的是自己的上下文，那一条就退化成正常臂，「忠实度没下降」可能就是它撑住的。
    所以用轮转（保证无固定点），不用随机打乱。
    """
    with _isolated_env():
        import run_gen_eval

        refs = ["a", "b", "c", "d"]
        got = run_gen_eval._rotate(refs)
        assert len(got) == len(refs)
        assert sorted(got) == sorted(refs), "只换位置，不许改动参照物内容"
        assert all(x != y for x, y in zip(refs, got)), "不许有固定点"
        assert run_gen_eval._rotate(["只有一条"]) == ["只有一条"]  # 单条时不炸


def test_judge_reference_keeps_every_source_article():
    """**回归**：裁判的参照物不许把靠后的文章整篇丢掉。

    踩过：`ref[:9000]` 再 `gold[:3000]`，实际只喂 3000 字（参照物有 24,671 字），
    答案里来自被截掉部分的内容**全被判成编造** → 真实上下文 0/6 忠实、
    打乱对照也是 0/6，指标钉死在 0。一刀切 `[:cap]` 是同一个错，只是没那么狠。
    """
    with _isolated_env():
        import run_gen_eval as g

        blocks = [
            {"article_id": "a.md", "text": "甲" * 9000},
            {"article_id": "b.md", "text": "乙" * 9000},
            {"article_id": "c.md", "text": "丙" * 9000},
        ]
        ref = g._fit_reference(blocks, cap=9000)
        for aid in ("a.md", "b.md", "c.md"):
            assert f"[{aid}]" in ref, f"{aid} 被整篇丢掉了"
        assert len(ref) <= 9000 + 3 * 200, "超上限太多（每块至多让 200 字）"
        # 短参照物不该被砍
        short = [{"article_id": "x.md", "text": "很短"}]
        assert g._fit_reference(short, cap=9000) == "[x.md]\n很短"
        assert g._fit_reference([], cap=9000) == ""
        assert g._fit_reference([{"article_id": "e", "text": ""}], cap=9000) == ""


def test_plain_answer_is_not_abstention():
    got, _ = scoring.abstained("思维链就是强制模型输出推理过程。" * 10)
    assert got is False


def test_cited_matches_article_id_and_stem_and_title():
    aid = "week5/33.重排序.md"
    assert scoring.cited(f"出处：{aid}", aid, "重排序") == "id"
    assert scoring.cited("出处：week5/33.重排序", aid, "重排序") == "id"
    assert scoring.cited("参考《重排序》一文", aid, "重排序") == "title"
    assert scoring.cited("完全不相关", aid, "重排序") == ""
    assert scoring.cited("随便", None, "重排序") == ""


def test_llm_records_real_token_usage():
    """评测要报真实成本。端点不返回 usage 时只记调用次数，不编造 token 数。"""
    from agent.llm import OpenAICompatLLM

    inst = object.__new__(OpenAICompatLLM)
    inst.usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}

    class _Usage:
        prompt_tokens = 120
        completion_tokens = 34

    inst._record_usage(_Usage())
    inst._record_usage(None)  # 兼容端点没返回 usage
    assert inst.usage == {"calls": 2, "prompt_tokens": 120, "completion_tokens": 34}


def test_llm_reply_carries_finish_reason():
    """**回归**：内容为空时必须能说出原因。

    推理模型会把 `max_tokens` 全花在思考上然后返回空 `content`；没有
    `finish_reason` 就只能靠猜，这个坑白查过两轮（见 README 判分器踩过的坑）。
    """
    from agent.llm import OpenAICompatLLM

    inst = object.__new__(OpenAICompatLLM)
    inst.model, inst.temperature, inst.max_tokens = "m", 0.0, 10
    inst.usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}

    class _Msg:
        content = None
        tool_calls = None

    class _Choice:
        message = _Msg()
        finish_reason = "length"

    class _Resp:
        choices = [_Choice()]
        usage = None

    class _Client:
        class chat:  # noqa: N801
            class completions:  # noqa: N801
                @staticmethod
                def create(**_kw):
                    return _Resp()

    inst.client = _Client()
    reply = inst.chat([{"role": "user", "content": "x"}])
    assert reply.content is None
    assert reply.finish_reason == "length"
    assert inst.usage["calls"] == 1


def test_config_rerank_defaults():
    from rag_core import RAGConfig

    c = RAGConfig()
    assert c.rerank_model == "BAAI/bge-reranker-base"
    assert c.rerank_candidates == 0, "0 = 不截断，重排整个 RRF 候选池"
    assert isinstance(c.rerank_enabled, bool)
    # 默认跟随 embedding 设备
    assert c.rerank_device == c.embedding_device


def test_splitter_signature_changes_with_headers():
    a = DocumentSplitter(headers=[("#", "h1")])
    b = DocumentSplitter(headers=[("#", "h1"), ("##", "h2")])
    c = DocumentSplitter(headers=[("#", "h1")], strip_headers=True)
    assert a.signature() != b.signature()
    assert a.signature() != c.signature()
    assert a.signature() == DocumentSplitter(headers=[("#", "h1")]).signature()


def test_indexer_fingerprint_includes_splitter_signature():
    """指纹只加了 splitter 参数就必须变，否则改分块不会触发重建。"""
    if not HAS_FULL_DEPS:
        _skip("需要 faiss + langchain-huggingface")
        return
    from rag_core.indexer import IndexConstructionModule

    idx = IndexConstructionModule.__new__(IndexConstructionModule)
    idx.model_name = "m"
    idx.normalize_embeddings = True
    chunks = [Document(page_content="x", metadata={"source": "a.md"})]
    assert idx.compute_fingerprint(chunks, "sig-a") != idx.compute_fingerprint(chunks, "sig-b")
    assert idx.compute_fingerprint(chunks, "sig-a") != idx.compute_fingerprint(
        [Document(page_content="y", metadata={"source": "a.md"})], "sig-a"
    )
    before = idx.compute_fingerprint(chunks, "sig-a")
    idx.normalize_embeddings = False
    assert idx.compute_fingerprint(chunks, "sig-a") != before


def test_embeddings_prefer_local_cache_then_fall_back():
    """模型已缓存就别联网；本地真没有才退回联网下载。"""
    if not HAS_FULL_DEPS:
        _skip("需要 faiss + langchain-huggingface")
        return
    from rag_core.indexer import IndexConstructionModule

    idx = IndexConstructionModule.__new__(IndexConstructionModule)
    idx.model_name = "m"
    idx.device = "cpu"
    idx.normalize_embeddings = True

    # 场景 1：本地有缓存 → 只调一次，且必须带 local_files_only
    calls = []

    def ok_build(model_kwargs):
        calls.append(dict(model_kwargs))
        return "EMB_LOCAL"

    idx._build_embeddings = ok_build
    idx.setup_embeddings()
    assert calls == [{"device": "cpu", "local_files_only": True}]
    assert idx.embeddings == "EMB_LOCAL"

    # 场景 2：本地没缓存 → 先抛错，再退回不带 local_files_only 的联网加载
    calls = []

    def no_cache_build(model_kwargs):
        calls.append(dict(model_kwargs))
        if model_kwargs.get("local_files_only"):
            raise OSError("no local cache")
        return "EMB_DOWNLOAD"

    idx._build_embeddings = no_cache_build
    idx.setup_embeddings()
    assert len(calls) == 2
    assert calls[0]["local_files_only"] is True
    assert "local_files_only" not in calls[1]
    assert idx.embeddings == "EMB_DOWNLOAD"


def test_chunk_ids_are_deterministic_across_splits():
    """同一个文档切两次，chunk_id 必须一样。

    否则缓存索引里的 id 和本次新切的对不上，RRF 按 chunk_id 去重时两路永远
    合并不了 —— 实测 shared_chunks 恒为 0、max_rrf 恒等于 1/(k+1)。
    """
    text = "# 标题\n\n正文一\n\n## 小节\n\n正文二\n\n## 另一节\n\n正文三"

    def split_ids() -> list:
        doc = Document(
            page_content=text, metadata={"source": "a.md", "parent_id": "p1"}
        )
        splitter = DocumentSplitter(headers=[("#", "h1"), ("##", "h2")])
        return [c.metadata["chunk_id"] for c in splitter.split([doc])]

    first, second = split_ids(), split_ids()
    assert len(first) >= 2
    assert first == second


def test_chunk_ids_are_unique_per_document_and_position():
    text = "# A\n\nx\n\n## B\n\ny"

    def split_ids(parent_id: str) -> list:
        doc = Document(page_content=text, metadata={"source": "a.md", "parent_id": parent_id})
        return [c.metadata["chunk_id"] for c in DocumentSplitter(headers=[("#", "h1"), ("##", "h2")]).split([doc])]

    a = split_ids("p1")
    b = split_ids("p2")
    assert len(set(a)) == len(a)  # 同一文档内部不重复
    assert not set(a) & set(b)  # 不同文档不撞号


def test_splitter_produces_child_chunks_with_parent_link():
    doc = Document(
        page_content="# 宫保鸡丁\n\n## 原料\n- 鸡肉\n\n## 操作\n先腌后炒。",
        metadata={"source": "g.md", "parent_id": "p1", "doc_type": "parent"},
    )
    chunks = DocumentSplitter(headers=[("#", "h1"), ("##", "h2")]).split([doc])
    assert len(chunks) >= 2
    for c in chunks:
        assert c.metadata["parent_id"] == "p1"
        assert c.metadata["chunk_id"]
        assert c.metadata["doc_type"] == "child"


# --------------------------------------------------------------------------
# 配置与包导出
# --------------------------------------------------------------------------
_TMP_DIR = Path(__file__).resolve().parent / "_tmp_config"


def test_config_default_context_max_tokens():
    saved = os.environ.pop("CONTEXT_MAX_TOKENS", None)
    try:
        assert RAGConfig(llm_api_key="k", llm_base_url="u").context_max_tokens == 6000
    finally:
        if saved is not None:
            os.environ["CONTEXT_MAX_TOKENS"] = saved


def test_config_rejects_empty_data_dir():
    _TMP_DIR.mkdir(parents=True, exist_ok=True)
    try:
        cfg = RAGConfig(llm_api_key="k", llm_base_url="u", data_path=str(_TMP_DIR))
        try:
            cfg.validate()
        except FileNotFoundError as e:
            assert "没有匹配" in str(e)
        else:
            raise AssertionError("空数据目录应该报错")
    finally:
        shutil.rmtree(_TMP_DIR, ignore_errors=True)


def test_config_accepts_populated_data_dir():
    _TMP_DIR.mkdir(parents=True, exist_ok=True)
    try:
        (_TMP_DIR / "a.md").write_text("# 菜\n", encoding="utf-8")
        RAGConfig(
            llm_api_key="k", llm_base_url="u", data_path=str(_TMP_DIR)
        ).validate()
    finally:
        shutil.rmtree(_TMP_DIR, ignore_errors=True)


def test_pipeline_wires_splitter_signature_into_fingerprint():
    """pipeline 构造不出来（要 faiss），但接线错了指纹就等于没修。"""
    src = (Path(__file__).resolve().parent.parent / "rag_core" / "pipeline.py").read_text(
        encoding="utf-8"
    )
    assert "splitter.signature()" in src
    assert "context_max_tokens=self.config.context_max_tokens" in src
    assert "metadata_filtered_search" not in src


def test_query_with_sources_returns_the_parent_docs():
    """`query_with_sources` 必须返回**喂进 prompt 的父文档正文**，不只是 id。

    判「答案有没有编造」要拿模型实际看到的上下文当参照。只给 article_id 时
    只能拿 gold 当参照，于是「引用了上下文里其他文章」会被误判成编造（评测踩过）。
    """
    if not HAS_FULL_DEPS:
        _skip("需要 faiss + langchain-huggingface")
        return
    from rag_core.pipeline import BasicRAGPipeline

    parent = Document(
        page_content="父文档正文（重排序是初筛之后再精排）",
        metadata={"article_id": "week5/33.重排序.md", "parent_id": "p1"},
    )

    class FakeRetriever:
        def hybrid_search(self, query, top_k=None, filters=None, rerank=None):
            return [
                Document(page_content="子块", metadata={"parent_id": "p1", "chunk_id": "k0"})
            ]

    class FakeGenerator:
        def generate(self, mode, question, docs):
            assert docs and docs[0].page_content.startswith("父文档正文")
            return "答案（week5/33.重排序.md）"

        def fit_context(self, docs):
            # 真实实现的契约：返回 [(doc, **实际喂进 prompt 的正文**)]
            return [(d, d.page_content) for d in docs]

    pipe = object.__new__(BasicRAGPipeline)
    pipe.config = RAGConfig(llm_api_key="k", llm_base_url="u", top_k=5)
    pipe.retriever = FakeRetriever()
    pipe.generator = FakeGenerator()
    pipe.skill = RAGSkill()
    pipe._parent_index = {"p1": parent}

    answer, parents = pipe.query_with_sources("重排序怎么做")
    assert answer.startswith("答案")
    assert [d.metadata["article_id"] for d in parents] == ["week5/33.重排序.md"]

    # 老的 with_context 路径必须给出同样的 id（两条路共用一个准备逻辑）
    assert pipe.query("重排序怎么做", with_context=True) == (
        answer,
        ["week5/33.重排序.md"],
    )


def test_query_with_sources_empty_retrieval():
    if not HAS_FULL_DEPS:
        _skip("需要 faiss + langchain-huggingface")
        return
    from rag_core.pipeline import BasicRAGPipeline

    class _Empty:
        def hybrid_search(self, query, top_k=None, filters=None, rerank=None):
            return []

    pipe = object.__new__(BasicRAGPipeline)
    pipe.config = RAGConfig(llm_api_key="k", llm_base_url="u", top_k=5)
    pipe.retriever = _Empty()
    pipe.generator = object()
    pipe.skill = RAGSkill()
    pipe._parent_index = {}

    answer, parents = pipe.query_with_sources("不存在的东西")
    assert parents == [] and "没有找到" in answer


def test_retrieve_returns_top_k_distinct_articles():
    """top_k 是「几篇文章」，不是「几个 chunk」——去重后不能缩水。"""
    if not HAS_FULL_DEPS:
        _skip("需要 faiss + langchain-huggingface")
        return
    from rag_core.pipeline import BasicRAGPipeline

    class FakeRetriever:
        def hybrid_search(self, query, top_k=None, filters=None):
            # 不管要多少，都返回同一篇文章的 chunk
            return [
                Document(
                    page_content=f"c{i}",
                    metadata={"parent_id": "same-doc", "chunk_id": f"k{i}"},
                )
                for i in range(top_k or 1)
            ]

    pipe = object.__new__(BasicRAGPipeline)
    pipe.config = RAGConfig(llm_api_key="k", llm_base_url="u", top_k=5)
    pipe.retriever = FakeRetriever()

    hits = pipe.retrieve("q", top_k=3)
    assert len(hits) == 1  # 只有一篇文章可给
    assert hits[0].metadata["chunk_id"] == "k0"  # 留排名最靠前的块


def test_retrieve_collects_distinct_articles():
    if not HAS_FULL_DEPS:
        _skip("需要 faiss + langchain-huggingface")
        return
    from rag_core.pipeline import BasicRAGPipeline

    class FakeRetriever:
        def hybrid_search(self, query, top_k=None, filters=None):
            # 交替返回 5 篇文章的 chunk
            return [
                Document(
                    page_content=f"c{i}",
                    metadata={"parent_id": f"p{i % 5}", "chunk_id": f"k{i}"},
                )
                for i in range(top_k or 1)
            ]

    pipe = object.__new__(BasicRAGPipeline)
    pipe.config = RAGConfig(llm_api_key="k", llm_base_url="u", top_k=5)
    pipe.retriever = FakeRetriever()

    hits = pipe.retrieve("q", top_k=3)
    assert len(hits) == 3
    assert len({h.metadata["parent_id"] for h in hits}) == 3


def test_package_exports_are_importable():
    assert RAGConfig is not None
    assert RAGSkill(name="x").name == "x"
    import rag_core

    assert "DEFAULT_CONFIG" not in rag_core.__all__, "DEFAULT_CONFIG 从未定义过"


def test_bm25_uses_our_tokenizer():
    """retriever 模块必须绑到我们的分词函数上。"""
    import rag_core.retriever as r

    assert r.tokenize_for_bm25 is tokenize_for_bm25


def test_pipeline_is_lazily_imported():
    """包根不该为了 import RAGConfig 就把 faiss/torch 拖进来。"""
    src = (Path(__file__).resolve().parent.parent / "rag_core" / "__init__.py").read_text(
        encoding="utf-8"
    )
    assert "from .pipeline import" not in src.split("def __getattr__")[0]


def test_lazy_pipeline_attribute_is_wired():
    """`from rag_core import BasicRAGPipeline` 这条公开路径必须仍然可用。"""
    import rag_core

    try:
        cls = rag_core.BasicRAGPipeline
    except ImportError:
        return  # 依赖没装齐 → 说明确实走到了延迟导入分支
    except AttributeError as e:
        raise AssertionError(f"__getattr__ 没接上 BasicRAGPipeline: {e}") from e
    assert cls.__name__ == "BasicRAGPipeline"


if __name__ == "__main__":
    tests = [
        v
        for k, v in sorted(globals().items())
        if k.startswith("test_") and callable(v)
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
