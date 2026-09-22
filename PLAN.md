# 从 `basic_rag` 模板到可上线的 RAG 系统：实施 Plan

> ⚠️ **历史文档（项目早期的一次核查记录）**。它当时得出的结论是
> 「骨架是好的，但只是一个能演示的 baseline」——那个判断**在当时成立**，
> 因为那时还没有评测层。之后项目补上了检索/生成层评测、LLM 裁判（含双向校准）、
> 三层回归门禁与 skill 复用契约，现状与本文的差距很大。
> **当前状态请看 `README.md`（速览在最上面）与 `TRY_IT.md`（下一步与已知未做）。**

> 本文基于对 `basic_rag/` 全部源码（`rag_core/` 9 个模块 + `skills/recipe/` 3 个文件）、`README.md`、`data/` 下 323 个 markdown 的**实际核查**写成，不是通用 RAG 教程。
> 核查中发现的数字和文件状态都可复现。

---

## 0. 结论先行（TL;DR）

这个模板**骨架是好的，但它是一个"能演示的 baseline"，不是"能上线的系统"**。

它已经帮你做完的事（不要重做）：

- 7 个模块的清晰分层：`loader → splitter → indexer → retriever → generator`，由 `pipeline.py` 组装
- **Skill 协议**：领域差异被收敛到 4 个钩子，换领域不动核心代码
- **父子块检索**（小检索、大生成）：检索命中的是标题级子块，喂给 LLM 的是整篇父文档
- **混合检索**：FAISS 向量 + BM25 关键词 + RRF 融合
- **元数据过滤**：`category` / `difficulty` 抽取与过滤链路已打通
- **数据指纹自动重建**：数据变了自动重建索引，不用手删

你要补的三层，按性价比排序：

| 层 | 内容 | 不做会怎样 |
|---|---|---|
| **① 修缺陷** | 6 个真实问题（见 §2），其中 3 个让现有能力"看起来有、实际失效" | demo 表现还行，一问边界问题就崩 |
| **② 领域化** | 写你自己的 Skill（4 个钩子） | 永远只能答菜谱 |
| **③ 工程化** | 服务层、引用溯源、评测集、可观测 | 只能自己在本机 `python main.py` 玩，无法迭代优化 |

**最小可交付（MVP）**：§3 的 Phase 0–3，约 3–5 个工作日，产出「你自己的领域 + 可问答 + 有引用 + 有 20 条冒烟集」的内部版本。

---

## 1. 模板现状（实测）

### 1.1 数据流

```
data/**/*.md
   │  loader.py          每个文件 → 1 个 parent Document
   │                     parent_id = md5(相对路径)；metadata_extractor 注入 category/difficulty
   ▼
parent docs (323 篇)
   │  splitter.py        MarkdownHeaderTextSplitter 按 #/##/### 切
   ▼
child chunks (N 块，带 parent_id / chunk_id / chunk_index)
   │  indexer.py         FAISS + bge-small-zh-v1.5(normalize) + 指纹比对
   │                     ├─ 指纹一致 → load_local
   │                     └─ 不一致   → 重建 + save_local + 写 .fingerprint
   ▼
FAISS vectorstore ─┐
                   ├─ retriever.py  向量 top5 + BM25 top5 → RRF(k=60) ─→ top_k
chunks(BM25) ──────┘
   │  pipeline.py       按 parent_id 回填父文档（按命中次数排序）
   ▼
parents (≤top_k 篇)
   │  generator.py      build_context 拼上下文（受 context_max_length 限制）
   ▼
ChatOpenAI(OpenAI 兼容) → 流式输出
```

查询侧还有一条支线（`skills/recipe/strategy.py`）：

```
question ─→ route()          LLM 分类为 list / detail / general
         ─→ rewrite()        LLM 改写（route == "list" 时跳过）
         ─→ extract_filters() 纯关键词匹配 category / difficulty（不调 LLM）
         ─→ 有 filters 走 metadata_filtered_search，否则走 hybrid_search
```

### 1.2 实测数据（这决定了你哪些设计能用）

| 指标 | 实测值 | 含义 |
|---|---|---|
| markdown 文件 | **323** 篇（10 个分类目录 + `template`） | 数据量小，FAISS 完全够用 |
| 含 `★` 难度标记 | **323 / 323（100%）**，分布 1★:27 / 2★:83 / 3★:115 / 4★:78 / 5★:20 | `difficulty` 元数据链路**完全可用**，且都在 `_DIFFICULTY_MAP` 覆盖范围内 |
| 单篇字符数 | 最小 204 / 均值 709 / **最大 4741** | ⚠️ 见 §2-④ |
| 超过 2000 字符的文档 | **4** 篇 | 与 `context_max_length=2000` 冲突 |
| 图片文件 | 328 个，其中 **326 个是 133–134 字节的 Git-LFS 指针** | 任何图文/多模态都是坏的，需 `git lfs pull` |
| `vector_index/.fingerprint` | **不存在** | 下次启动会全量重建索引 |
| `.gitignore` / `.env.example` | **都不存在**（README 里写了） | `.env` 里有密钥，有泄露风险 |
| 本机 Python | 只有 3.11.5（`F:\Python`），`import faiss` 失败；而 `__pycache__` 是 `cpython-312` | **当前状态下项目跑不起来**，需先补 3.12 + venv + 依赖 |

> 关于 `random.md`：项目里**没有**这个文件，唯一的文档是 `README.md`（333 行，我已通读）。下面所有判断都来自 `README.md` + 源码 + 数据实测。

---

## 2. 必须修的缺陷（按严重度）

### ① BM25 在中文上等于没生效 —— P0

`rag_core/retriever.py:22` 用 `BM25Retriever.from_documents(chunks, k=5)`，没有传 `preprocess_func`。
LangChain 的默认实现是（已核对 `langchain_community/retrievers/bm25.py:11`）：

```python
def default_preprocessing_func(text: str) -> List[str]:
    return text.split()          # 按空白切
```

中文没有空格 → **整行/整段被当成一个 token**。
后果：BM25 只能命中"整行完全一致"的查询，实际上全程贡献不了有效排序。所谓"混合检索"退化成**单路向量检索**，RRF 融合也失去意义。

**修**：接入中文分词。

```python
import jieba
def zh_preprocess(text: str) -> list[str]:
    return [t for t in jieba.lcut_for_search(text) if t.strip()]

self.bm25_retriever = BM25Retriever.from_documents(
    self.chunks, k=5, preprocess_func=zh_preprocess
)
```

并把它写进这一层的单测（用「宫保鸡丁」查「花生米 干辣椒」，要求 BM25 能召回对应 chunk）。

### ② 上下文裁剪比 top_k 还小，第 3 篇必然被丢 —— P0

`rag_core/generator.py:76-79`：

```python
if cur_len + len(text) > self.context_max_length:   # 默认 2000
    break                                          # ← 直接 break，不是跳过/截断
```

而数据实测：单篇均值 709 字符，`top_k=3` 回填 3 篇父文档 → 约 2100+ 字符 > 2000。
**结果：即使检索到了 3 篇，通常只有前 2 篇进入上下文；命中一篇 4741 字符的长文时只剩 1 篇。**

**修**（三选一，建议全做）：

1. 把预算改成按 **token** 计算（`tiktoken` 或模型的 tokenizer），而不是字符数
2. `break` 改成 **per-doc 截断 + continue**：超预算的文档截断到剩余预算，而不是丢弃后面所有文档
3. `context_max_length` 默认值按你的模型窗口重设（如 8k 模型 → 6000–8000），并从环境变量读

### ③ 指纹不覆盖分块配置，索引会和代码静默不一致 —— P0

`rag_core/indexer.py:76-83` 的 `compute_fingerprint` 只哈希了 `model_name + chunk 内容 + source`。

README 第 200 行也承认了："改分块逻辑（标题层级等）→ 建议手动删 `vector_index/`"。
但这靠人记。哪天你改了 `splitter_headers` 或 `strip_headers`，指纹不变 → **加载旧索引 + 新 chunk 列表**，检索结果与生成上下文错位，且没有任何报错。

**修**：把影响索引的一切都纳入指纹。

```python
def compute_fingerprint(self, chunks, splitter_signature: str = "") -> str:
    h = hashlib.md5()
    h.update(self.model_name.encode())
    h.update(splitter_signature.encode())      # headers + strip_headers
    h.update(str(self.embedding_normalize).encode())
    for c in chunks: ...
```

同时：**指纹文件缺失时不要静默全量重建**，先打一条明确的 WARNING（当前 `.fingerprint` 就是缺失状态）。

### ④ 缺 `.gitignore` 和 `.env.example`，而 `.env` 里是真密钥 —— P0（安全）

README §“目录结构”里列了这两个文件，实际目录里**都不存在**（已用 `Test-Path` 确认）。
现在 `.env` 包含 `LLM_API_KEY` 且与代码同目录。

**修**（5 分钟）：补 `.gitignore`（至少 `.env`、`vector_index/`、`__pycache__/`、`.cache/`）和 `.env.example`（键齐全、值留空）。
**并轮换一次那个 key**——如果这个目录曾被推送过，密钥应视为已泄露。

### ⑤ 元数据过滤是"检索后过滤"，召回天花板很低 —— P1

`retriever.py:31-41`：

```python
docs = self.hybrid_search(query, top_k * 3)      # 先取 9 条
for doc in docs:
    if self._match_filters(doc, filters): ...    # 再筛，筛完可能 0 条
```

「推荐简单的素菜」→ `category=素菜 + difficulty=简单`，而 2★ 只有 83/323 篇。
top 9 里同时满足两个条件的可能不到 3 条，甚至 0 条 → 返回"抱歉，没有找到相关信息"。

**修**：把过滤下推到检索层。
- 向量侧：FAISS `similarity_search(query, k, filter=predicate)` 原生支持按 metadata 过滤
- BM25 侧：先按 metadata 过滤 `self.chunks` 再建检索器（或对候选做交集）
- 兜底：过滤后为空时，放宽为"只按 category 过滤"并显式告知用户放宽了条件

### ⑥ 路由结果不影响 `top_k`，`list` 类问题只给 3 个 —— P1

`pipeline.py:99-115` 算了 `route`，但 `top_k` 始终是 `config.top_k`（默认 3）。
「推荐几个素菜」这种 list 意图期望 8–10 条，实际只给 3 条。`prompts.py:87` 里 `list` 也只是复用 `BASIC_TEMPLATE`。

**修**：`_pick_top_k(route)` —— list → 10，detail → 3，general → 5；并给 `list` 单独写一个"只输出菜名+一句话理由"的 prompt。

### ⑦ 顺带清理 —— P2

| 项 | 问题 | 处理 |
|---|---|---|
| `data/cook/dishes/template/示例菜/示例菜.md` | 会被当真实菜谱索引（README Q6 也警告了） | 移出 `DATA_PATH`，或在 loader 里加 exclude 规则 |
| `config.py:21` `file_glob` | 是硬编码字段，README 暗示可配但环境变量读不到 | 加 `os.getenv("FILE_GLOB", "*.md")` |
| `config.validate()` | 不校验 embedding 模型、不校验数据目录为空 | 补：目录为空直接报错，避免"索引成功但 0 文档" |
| 无任何测试 | `rag_core` 9 个模块 0 测试 | Phase 2 起每个修复配 1 个 pytest |

---

## 3. 分阶段 Plan

每一阶段都有**产出物**和**验收标准**——不满足就不进下一阶段。

### Phase 0　定范围（0.5 天，不写代码）

这 6 个问题答不出来，后面全是返工。建议默认值已给出。

| 决策 | 影响 | 建议默认 |
|---|---|---|
| 领域 / 用户是谁 | Skill 的 4 个钩子全部取决于此 | 先复制 recipe 改造成你自己的领域 |
| 问题类型分布 | 决定 route 分支和 prompt 数量 | 事实查询 + 操作步骤 + 列表推荐 |
| 数据来源与更新频率 | 决定增量索引 vs 全量重建 | 手工维护 md → 全量重建够用 |
| 交付形态 | 决定要不要 Phase 5 | CLI 自用 → 可跳过服务层 |
| 质量目标 | 决定要不要 Phase 6 | hit-rate@5 ≥ 0.9，答案引用准确率 100% |
| 约束 | 私有化 / 延迟 / 成本 | CPU 可跑、P95 < 5s、单次 < 0.01 元 |

**产出物**：一页 `SCOPE.md`。

---

### Phase 1　跑通基线（0.5 天）

当前**跑不起来**，别急着改代码。

1. 装 Python 3.12（`__pycache__` 表明原环境是 3.12），建 venv
   ```powershell
   py -3.12 -m venv .venv; .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   pip install jieba pytest          # 后续要用的
   ```
2. 补 `.gitignore` / `.env.example`（§2-④）
3. 把 `data/cook/dishes/template/` 移出 `DATA_PATH`
4. 决定图片怎么办：`git lfs pull` 拉真图，或明确"本期不做图文"
5. 首次运行 → 观察 chunk 数与索引构建耗时，确认 `.fingerprint` 已生成
6. 冒烟 3 类问题各 1 条：`detail`（"宫保鸡丁怎么做"）、`list`（"推荐几个素菜"）、`general`（"什么是川菜"）

**产出物**：可运行的 baseline + `SMOKE.md`（记录 3 条实际输出）。
**验收**：三类问题都返回非空答案；第二次启动**秒级**（走缓存索引，日志显示 `索引已加载`）。
> 若第二次启动仍在重建 → 指纹逻辑有问题，先修完再往下走。

📖 参考：`all-in-rag/docs/chapter1/03_get_start_rag.md`、`chapter2/04_data_load.md`

---

### Phase 2　修 P0/P1 缺陷（1 天）

按 §2 顺序做：① BM25 中文分词 → ② 上下文预算 → ③ 指纹完整性 → ⑤ 过滤下推 → ⑥ route 感知 top_k → ⑦ 清理。

每改一项，**用同一批 10 条问题对比改动前后输出**（这就是 Phase 6 评测集的雏形）。

**产出物**：修好的 `retriever.py` / `generator.py` / `indexer.py` / `pipeline.py` + `tests/` 下 6–8 个 pytest。
**验收**：
- BM25 单测：中文关键词查询能召回正确 chunk（这条测试在修复前必须失败）
- 「推荐简单的素菜」能返回 ≥3 条结果
- 一篇 4741 字符的长文 + 2 篇短文进 top_k 时，3 篇都在上下文里（或明确记录了截断策略）
- 改一次 `splitter_headers` 启动，日志明确提示"分块配置变化，重建索引"

📖 参考：`all-in-rag/docs/chapter4/11_hybrid_search.md`、`chapter2/05_text_chunking.md`

---

### Phase 3　写你自己的领域 Skill（1–2 天）—— 这是"做你的 RAG 系统"的正题

新建 `rag_core/skills/<你的领域>/`，实现 4 个钩子（模板见 README §“换一个领域”）：

| 钩子 | 你要回答的问题 | 注意 |
|---|---|---|
| `metadata_extractor` | 文档里哪些字段会被**过滤/展示**？（如 部门 / 版本 / 生效日期 / 文档类型） | 决定过滤能力上限。字段从**正文或路径**里解析，解析不出来要显式标注不合格，而不是默认成"其他" |
| `splitter_headers` | 你的文档结构是什么？（章/条、H2/H3、FAQ 的 Q/A） | 结构好就用标题切；表格/代码多的文档要单独处理（`MarkdownHeaderTextSplitter` 会切碎表格） |
| `prompt_registry` | 每种 route 期望的回答格式？（引用格式、是否给步骤、是否允许"不知道"） | 必须显式写"仅依据给定资料回答，不足时说明"，否则模型会编 |
| `query_strategy` | 需要 route/rewrite/过滤吗？ | 领域词表小就别用 LLM route（贵且不稳定），先走规则；`extract_filters` 用中文时注意关键词碰撞（如"简单"既是难度也是口语） |

同时：

- `INDEX_SAVE_PATH` **按 skill 分目录**（`./vector_index/<skill>`），避免和 recipe 索引串用
- 领域术语表 → 用于 `rewrite` 的同义词扩展（比 LLM 改写更便宜稳定）
- 拿 20 条真实问题跑一遍，把答不好的归类：**没召回 / 召回了但答错 / 答对了但格式差**，分别对应 Phase 4 / prompt / prompt

**产出物**：`rag_core/skills/<领域>/` + 自己的 `main_<领域>.py` + 独立索引目录。
**验收**：20 条真实问题里 ≥15 条答案可用且**每条都带出处**。

📖 参考：`all-in-rag/docs/chapter4/12_query_construction.md`、`chapter4/14_query_rewriting.md`、`chapter5/16_formatted_generation.md`

> **到这里就是可以拿去用的 MVP。** 下面的 Phase 4–7 是按需叠加，不要一次全上。

---

### Phase 4　检索质量提升（2–4 天，按需）

**一次只动一个变量，每次都用 Phase 6 的评测集验证。** 按收益排序：

| 手段 | 什么时候需要 | 成本 |
|---|---|---|
| **Reranker**（`bge-reranker-base` / `gte-rerank`）对 top20 重排取 top5 | 召回有但排序差，正确答案在第 5–20 位 | 低（本机 `all-in-rag/code/C3/FlagEmbedding` 已可用） |
| **提高候选池**：向量/BM25 各取 5 → 各取 20 再融合 | 真正的正确答案根本没进 top5 | 极低 |
| **分块策略**：按表格/FAQ/问答对定制切分 | 答案被切碎、表格被拆散 | 中 |
| **多路查询**：同义改写并行检索后合并 | 用户口语与文档书面语差距大 | 中 |
| **HyDE / 假设文档检索** | 短查询、语义鸿沟大 | 中（多一次 LLM 调用） |
| **滑动窗口 / 父块扩展** | 答案跨块边界 | 低（已有父块机制，调窗口即可） |
| **元数据权重**：时间新、权威文档加权 | 文档有新旧/权威差异 | 低 |

🚫 **不要做**：微调 embedding（数据量不够，收益不抵成本）、换 Milvus/pgvector（323 篇文档，FAISS 绰绰有余）。

📖 参考：`all-in-rag/docs/chapter3/06_vector_embedding.md`、`chapter3/10_index_optimization.md`、`chapter4/15_advanced_retrieval_techniques.md`

---

### Phase 5　工程化（3–5 天，对外提供服务时必做）

模板现在只有 `main.py` 的交互式 CLI。要变成"系统"需补：

1. **服务层**：FastAPI + `POST /chat`（SSE 流式）+ `GET /health`
   - 关键：把 `BasicRAGPipeline` 做成**进程内单例**（`build()` 里加载索引和模型很贵），用 `lifespan` 初始化
2. **引用溯源**：`query()` 返回值从 `str` 改成 `{"answer": ..., "citations": [{"source": ..., "title": ..., "chunk_index": ...}]}`
   - 父文档已带 `source` / `dish_name` 类字段，直接往上带即可
   - **这是 RAG 系统最重要的信任机制，优先级高于任何检索调优**
3. **多轮对话**：会话级历史 + 查询改写（把"它需要炖多久"补全成完整问题）；注意历史不能无限增长
4. **增量索引**：`indexer.add_documents()` 已有，但缺 **删除/更新**（文件删了索引里还在，是数据错误）
5. **缓存**：embedding 缓存（同文本不重算）+ 相同问题的答案缓存
6. **可观测**：每次请求记录 route / 检索耗时 / 命中 chunk_id / token 用量 / 是否命中缓存
7. **错误与降级**：LLM 超时重试、检索为空时的明确话术、LLM 不可用时返回检索结果原文

**产出物**：可 `docker compose up` 起的服务 + 接口文档。
**验收**：并发 10 请求不串数据；索引只加载一次；每个回答都能追溯到源文件。

📖 参考：`all-in-rag/docs/chapter8/01_env_architecture.md`、`03_index_retrieval.md`、`04_generation_sys.md`

---

### Phase 6　评测与迭代（2 天，与 Phase 4/5 并行，**优先级被普遍低估**）

没有评测集，Phase 4 的每次调参都是玄学。

1. **建 golden set**：50–100 条 `(问题, 标准答案要点, 正确出处)`。来源最好是真实用户问题 + 人工标注
2. **分层指标**：
   | 层 | 指标 | 目标 |
   |---|---|---|
   | 检索 | hit-rate@5 / recall@10 / MRR | hit-rate@5 ≥ 0.9 |
   | 生成 | faithfulness（答案是否有原文支撑）/ answer relevance | 无编造 |
   | 端到端 | 空回答率 / P95 延迟 / 单次成本 | 空回答率 < 5% |
3. **回归脚本**：`pytest tests/test_eval.py`，任何 chunk/prompt/模型改动前后各跑一次，对比差异表
4. **失败归因**：每条失败样本标注属于"没召回 / 排序错 / 上下文截断 / prompt 问题 / 模型能力不足"，按类修

**产出物**：`eval/golden.jsonl` + `eval/report.py`（输出对比表）。
**验收**：改任何一个参数，能立刻说出"hit-rate 从 0.78 → 0.91"。

📖 参考：`all-in-rag/docs/chapter6/18_system_evaluation.md`

---

### Phase 7　部署与运维（1–2 天）

- **配置分离**：密钥走环境变量/密钥管理，配置文件进版本库（模板现在只有 `.env`，缺配置分层）
- **容器化**：模型缓存（`HF_HOME`）挂 volume，别每次重建镜像都重下 100MB 模型
- **离线模式**：确认缓存完整后开 `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`（README Q3 有坑记录）
- **容量**：CPU 下 embedding 建索引的实际耗时；并发下 LLM 是瓶颈不是检索
- **成本**：route/rewrite 每次问答要额外调 2 次 LLM —— 用规则能省就省
- **灰度与回滚**：索引目录带版本号，新索引验证通过再切

📖 参考：`all-in-rag/docs/chapter6/19_common_tools.md`

---

## 4. 排期建议

| 周期 | 做什么 | 结果 |
|---|---|---|
| **第 1 天** | Phase 0 + Phase 1 | 跑得起来，3 类问题能答 |
| **第 2–3 天** | Phase 2（修 6 个缺陷 + 单测） | 中文 BM25 生效、过滤不再空、上下文不丢文档 |
| **第 4–5 天** | Phase 3（自己的 Skill + 20 条问题） | **可交付 MVP** |
| **第 2 周** | Phase 6（评测集）+ Phase 4（按数据挑 1–2 个手段） | 可量化、可迭代 |
| **第 3 周** | Phase 5 + Phase 7 | 可对外服务 |

---

## 5. 明确不要做的事（YAGNI）

| 冲动 | 为什么现在不做 |
|---|---|
| 上 GraphRAG / 知识图谱 | 323 篇结构化 markdown，向量检索够用；KG 的构建和运维成本远高于收益。等出现"多跳关系推理"的真实失败样本再说 |
| 上 Agentic RAG（多轮检索 + 反思） | 先有评测集证明单轮检索的瓶颈在哪，否则只是把延迟翻几倍 |
| 上多模态（图文） | 数据里 326/328 张图是 LFS 指针，先把图拉下来再谈 |
| 微调 embedding / reranker | 没有千级标注数据，微调大概率不如直接换更强的开源模型 |
| 换向量数据库（Milvus/pgvector） | 单机 323 篇文档，FAISS 内存索引是当前最优解 |
| 同时调 5 个参数 | 无法归因。一次一个，靠 Phase 6 的评测集判定 |

**进阶方向**（等基础稳固、有真实失败样本后）：`all-in-rag/docs/chapter7/20_kg_rag.md`、`chapter7/21_agentic_rag.md`

---

## 6. 上线门槛（Definition of Done）

一个 RAG 系统什么时候算"能上线"：

- [ ] 冒烟集 20 条，可用率 ≥ 75%
- [ ] golden set 50+ 条，hit-rate@5 ≥ 0.9
- [ ] **100% 的回答带可点击出处**（文件 + 章节）
- [ ] 资料不足时明确回答"资料中没有"，而不是编造（人工抽检 20 条，0 编造）
- [ ] P95 延迟 < 5s（流式首字 < 2s）
- [ ] 数据更新后自动重建索引，且第二次启动走缓存（< 3s）
- [ ] 有回归脚本，改动能证明"没变差"
- [ ] 密钥不在版本库里，`.env.example` 齐全，`.gitignore` 覆盖 `.env` / `vector_index/` / `.cache/`

---

## 附：一页速查 —— 最优先的 6 件事

1. **补 `.gitignore` + `.env.example`，并轮换已暴露的 `LLM_API_KEY`**（安全，5 分钟）
2. **`BM25Retriever` 接 jieba 分词**（否则"混合检索"是假的）
3. **`context_max_length` 从 2000 提到模型窗口量级，`break` 改 per-doc 截断**（否则 top_k=3 实际只用 2 篇）
4. **指纹纳入 `splitter_headers` 等分块配置**（否则索引与代码静默不一致）
5. **元数据过滤下推到检索层**（否则"推荐简单的素菜"返回空）
6. **写自己的 Skill + 建 20 条冒烟问题**（这才是"做一个 RAG 系统"的正题）
