# 接下来做什么 / 怎么自己试

这份文档给的是**可以照着敲的操作**：每个方向写清楚「依据是什么、改哪个文件、敲哪条命令、预期看到什么」。
所有数字都是本项目实测出来的；**没精确计时的地方我标了「未精确计时」，不编数字**。

当前状态：三项增强（闸门标注 / 生成层评测 / 常驻服务）已完成，**116 项测试通过**。
回归门禁跑通完整 PASS（含生成层，全项 +0.0%）；裁判第 5 版通过小样本校准；
语料扩容第一批完成（**99 → 105 篇**，`eval/gate_review.md` 里 5 条「近邻」缺口已补上）。

---

## 方向表：现在最该动的是哪个

| 顺序 | 方向 | 状态 | 花钱 |
|---|---|---|---|
| **1** | 一条命令跑回归 | ✅ 完成 | 0 |
| **2** | 扩容语料 | ✅ **第一批完成**（week16 六篇）；第二批 11 条真缺口待做 | 0 |
| **3** | 事实正确性 | 3a/3b ✅ 完成；**3c 全量忠实度未测**（且换语料后必须重新生成答案） | 有 |
| 4 | Agent 臂的 `partial` 声明 | 待做（Agent 只 66.7% 明说「不完全」，单轮 100%） | 几分钱 |

> 我原本把「`partial` 给相关方向」列为第 2 项，**那是错的** —— 实测 100% 已经在给方向了，
> 那个「0%」是我的指标 bug（详见 README 该节）。方向表已按修正后的数据重排。

---

## 方向 1：回归门槛（**已实现**，直接用）

```powershell
cd basic_rag

# ① 先立基线（在「正常配置」下先跑一遍，再立，见下面的坑）
.\.venv\Scripts\python.exe eval\run_eval.py --json
.\.venv\Scripts\python.exe eval\run_agent_eval.py --json
.\.venv\Scripts\python.exe eval\run_all.py --update-baseline

# ② 此后每次改完代码，一条命令看有没有退化（约 3.5 分钟，0 次 LLM 调用）
.\.venv\Scripts\python.exe eval\run_all.py
#    PASS → 退出码 0 ；FAIL → 退出码 1（可以直接挂到 CI / pre-commit）
#    --skip-run  只比已有报告，不重跑
#    --no-gen    跳过生成层确定性指标
#    --tol 0.05  放宽阈值（默认 2 个百分点）
```

它比三层：**检索层**（hit@5 / 覆盖率）、**Agent 对照**、**生成层确定性指标**
（引对率 / 要点覆盖率 / 声明缺口，用 `--rescore` 复用已保存答案，不花钱）。

**验证它真的有效**（别只看它打印 PASS）：

```powershell
$env:RERANK_ENABLED='false'
.\.venv\Scripts\python.exe eval\run_all.py --no-gen      # 应该 FAIL，且先打印「配置不一致」
Remove-Item Env:\RERANK_ENABLED
```

预期能看到 `paraphrase/rerank 66.7% → 58.3%`、`multi/rerank 100.0% → 80.0%` 被标成「退化」。

### 立基线时的两个坑（工具已加保护，但你得知道）

1. **基线抓的是「磁盘上现有的报告」。** 如果你上一次跑的是实验配置（比如刚验证完
   `RERANK_ENABLED=false`），那份报告会被立成基线，之后怎么比都 PASS（**我实测踩过**）。
   所以顺序永远是：**正常配置下先重跑一遍评测 → 再 `--update-baseline`**。
2. **配置变了指标本来就会变。** 基线里记了生效配置（`rerank_enabled` / `top_k` /
   `min_relevance` …），对比时先比配置、再比指标，并把差异打在指标表前面 ——
   免得你把「配置差异」当成「代码退化」。

标注指纹（`golden.jsonl` 的 md5）也记在基线里：**改了 seeds/golden 之后对比没有意义**，
脚本会直接提示你重新立基线，而不是报一个吓人的假 FAIL。

---

## 方向 2：扩容语料 —— **第一批已完成** ✅（week16 六篇）

**依据**：40 条闸门标注里 **37 条**是「语料里不存在」或「存在但没回答」。
这不是闸门坏了，是 99 篇课程笔记覆盖不了「GraphRAG 怎么构建」这类问题。

### 做了什么（第一批：那 5 条「近邻」缺口）

新增 `Agent-100-Days/week16/`，编号接 105：

| 文件 | 覆盖的原闸门题 |
|---|---|
| `106.RAG 评估：用 RAGAS 量化忠实度与召回.md` | 怎么用 RAGAS 评估 RAG 的忠实度？ |
| `107.ColBERT 与后期交互检索.md` | 怎么用 ColBERT 做后期交互检索？ |
| `108.SFT 与 RLHF：两条对齐路线.md` | RLHF 和 SFT 的区别是什么？ |
| `109.幻觉检测：从采样一致性到语义熵.md` | 怎么检测大模型的幻觉？ |
| `110.语义缓存：用相似度换成本.md` | 语义缓存怎么做？ |
| `111.思考 & 补充学习资料.md` | （保持「每周一篇思考」的结构） |

**顺带修了一条假闸门**：`HyDE 和多查询改写该怎么选？` 原本标成 `mentioned`，
但 week5/34 有 HyDE vs Multi-query 的对照表 —— probe 选词「查询改写」而文中叫「问题改写」，
probe 通过、标注错了。**probe 校验的是字符串，不是知识。**

### 标注与语料的变化（可核对）

| 项 | 之前 | 现在 |
|---|---|---|
| 语料 | 99 篇 | **105 篇** |
| 可答题 | 38 | **44** |
| `absent` | 22 | 17 |
| `mentioned` | 15 | 14 |
| `partial` | 3 | 3 |

### ⚠️ 换了标注就不能和旧数字比大小

可答题集从 38 变 44，且新增 6 篇是检索的**新干扰项**，`paraphrase` 这类指标**可能下降**。
这不是退化，是评测集变了 —— `run_all.py` 的标注指纹保护正是为此存在（它会拒绝硬比并提示重新立基线）。

**生成层尤其要注意**：现有全量答案是在 week16 存在**之前**生成的，对那 6 条新题（`expect` 指向
week16）必然「未命中、未引用」。所以旧报告已归档为 `eval/last_gen_report.pre_week16.json`，
**没有**进新基线 —— 生成层要重跑一次生成（约 29.5 分钟 + 真实费用）才有有效数字。

### 下一批（待办）：**已完成概念级验证**（不是只看 probe）

`probe` 只证明「这个词在语料里出现 0 次」，不证明「这个知识不存在」——
假闸门（HyDE 那条）就是这么来的。所以这批按**概念词族**重新查了一遍（含同义词、相关词）：

| 原闸门题 | 语料实际覆盖 | 判定 |
|---|---|---|
| `pytest 的 fixture 作用域怎么选？` | fixture / conftest / setup / teardown / 作用域 **全 0 次** | **真缺口**（最干净的一条） |
| `用 BeautifulSoup 怎么处理 JS 动态渲染的页面？` | BS4 只用于静态 HTML（week4/27）；Playwright 仅在 week15/100 的 E2E 测试里出现 1 次；无头浏览器 0 | **真缺口**（且能接上已有的 Playwright） |
| `怎么给 MCP 服务加 OAuth 鉴权？` | OAuth / JWT / Bearer **全 0 次**；鉴权仅 2 次泛指 | **真缺口** |
| `Kubernetes 的 HPA 怎么配？` | HPA 0 次；「自动扩缩容」只作为 K8s 的**能力**被提 4 次，没有配置方法 | **真缺口** |
| `怎么用 OpenTelemetry 做链路追踪？` | OTel 仅 1 次（「可与 LangSmith、OpenTelemetry 等集成」）；span / 导出器 / collector 全 0。**但** trace_id 在 7 篇出现 20 次、week15/99 有 10 次「调用链」 | **真缺口（工具级）**，写时必须交叉引用 week15/99，不能假装语料没讲链路追踪 |
| `漏桶限流怎么实现？` | 漏桶仅 1 句定义（week15/102）；**令牌桶反而实现了**（`TokenBucketRateLimiter`） | **真缺口**（实现层） |
| `LangGraph 的状态机怎么用` | StateGraph / add_node / MemorySaver 全 0；checkpointer 1 次（week7/47）；「状态机」10 次但讲的是**自研 Runtime** | **真缺口（API 级）**，概念已有 |
| `Redis 的 RDB 和 AOF 持久化怎么选？` | RDB / AOF / appendonly 全 0；「持久化」40 次但讲的是**会话状态** | **真缺口** |
| `aiohttp 的连接池该怎么调优？` | aiohttp 1 次；「连接池」10 次但全是**数据库**连接池 | **真缺口** |
| `Milvus 的 IVF_PQ 索引参数怎么调？` | IVF / HNSW / PQ / 索引参数 全 0；「索引类型」泛泛提 2 次 | **真缺口**，但偏底层运维，与课程主线最远 |
| `FAISS 的 IndexIVFPQ 怎么训练？` | IndexIVF / IVF / train( / 倒排 全 0 | **真缺口**，同上 |
| `Hub-and-Spoke 架构有哪些优缺点？` | week12/78 有**优点**三条（职责分离 / 可扩展 / 统一管理），**没有缺点** | **不是 `mentioned`，应改标 `partial`**：优点答得了、缺点答不了，属人工判定类 |

**还发现一个边界问题（需要你定）**：`Terraform` / `Elasticsearch` / `Grafana` 现在被标成
`far`（拒答题），但课程有「部署与运维」「监控与可观测性」两周 —— 它们算不算域内，是**产品判断**。
若算域内，可以放进第三批；`far` 闸门集从 17 条减到 14 条仍然够用。

**不补的两类**（重要）：
- **`far` 的 17 条**（红烧肉、COBOL、围棋、经络…）**绝对不能补** —— 它们是拒答测试的题目，
  补了等于毁掉量具。闸门集必须保留一批「语料真的答不了」的题，否则永远测不出系统会不会拒答。
- 闲聊（今天天气）与定价（Pinecone Serverless）不补：前者考的就是该不该拒答，后者价格会变、写了就是假的。

### 第一批补完后实测（零成本，从已有报告里查的）

5 条新题**全部在 top-5 命中自己的新文章**（多数是 top-1），检索层闭环：

| 问题 | adaptive | rerank |
|---|---|---|
| 怎么用 RAGAS 评估 RAG 的忠实度？ | 命中（第 1） | 命中（第 1） |
| 怎么用 ColBERT 做后期交互检索？ | 命中（第 1） | 命中（第 1） |
| RLHF 和 SFT 的区别是什么？ | 命中（第 1） | 命中（第 1） |
| 怎么检测大模型的幻觉？ | 命中（第 1） | 命中（**第 2**） |
| 语义缓存怎么做？ | 命中（第 1） | 命中（第 1） |

顺带量了一个怀疑点：`111.思考 & 补充学习资料` 因为列了 week16 全部链接，会不会变成「关键词磁铁」
挤掉别人？实测 **adaptive 5/44（11%）、rerank 4/44（9%）进 top-5，0 次挤掉 gold 进前 4** ——
有噪声，但影响很小，**暂不处理**。

### 端到端冒烟（花了 6 次调用，零生成成本那次不算）

三道 week16 的问题走完「检索 → 生成 → 裁判」：**命中 3/3、引用 3/3、忠实 3/3**。
其中暴露出一个**量具缺陷并已修复**（详见 README 判分器第 ⑥ 个坑）：
参照物记的是未截断全文，而模型收到的是截断后的正文 → 模型如实说「文档被截断」被判成编造。
修完参照物从 21,228 字变成模型真正看到的 12,781 字，那条假「不忠实」消失。

### 操作要点（新加文章时）

1. `.md` 放进语料根目录的 `weekN/` 下。`FILE_GLOB=week*/*.md` 只收 **week 目录下**的文件。
2. **不用手动删索引** —— 指纹含每个 chunk 的内容+来源，变了会自动重建。
3. **必须重跑 `build_golden.py`**（它会用 probe 校验：转成可答的条目要去掉 probe，
   仍在闸门里的条目其 probe 词必须仍为 0 命中）：
   ```powershell
   .\.venv\Scripts\python.exe eval\build_golden.py
   ```
4. `tests/test_agent.py` 里有一条测试**钉住了语料形状**（篇数 / 周次范围 / 「思考」篇数），
   加完文章要同步改。
5. 重跑检索层与 Agent 层（零 LLM 成本），然后重新立基线。

---

## 方向 3：事实正确性（最难，分两步走）

**依据**：LLM 裁判试了 **5 版**，前 4 版都不可信；第 5 版修掉参照物截断后**通过小样本校准**，
但**全量忠实度仍未测量**。根因之一：

> **Agent 实际看到的上下文没有被记录** —— 工具返回的原文只进了 `messages`，
> `AgentResult` 里只有 `citations`（只有 id/标题，没有正文），`run()` 一返回那段文本就丢了。
> 于是裁判只能拿 gold 当参照，把「引用了上下文里其他文章」误判成编造。

**分两步**：3a 零成本、可验证（**已完成**）；3b 要花钱、且有失败风险（**尚未做**）。

### 前置步骤 3a：记录「模型真正读到的原文」 —— **已实现** ✅

1. `agent/loop.py`：`AgentResult` 新增 `contexts`（每条 = 一个内容块：
   `{tool, article_id, title, text}`）。只收 `read_article` 的全文和 `search_notes` 的片段，
   **刻意不收 `list_articles` 的目录**（只有标题没正文，当上下文会虚增「有依据」）。
   上限 `_MAX_CONTEXT_CHARS = 40000`，超了就不再累积。
2. `rag_core/pipeline.py`：新增 `query_with_sources()`，返回
   `(答案, 喂进 prompt 的父文档)`；`query()` 重构到同一条准备逻辑上（行为不变）。
3. `eval/run_gen_eval.py`：两臂都存 `context_texts`；`--judge-only` 优先拿它当参照，
   旧报告缺这个字段时退回文章全文并**明确警告**（那样比真实上下文宽松）。

已验证（冒烟 2 条，实测数字）：

| 臂 | 引用列表 | 真读到的正文 |
|---|---|---|
| `single` | 4 篇 | 4 块 / 24,567 字 |
| `agent` | 6 篇 | 6 块 / 7,631 字 |

**这组数字正好解释了裁判之前为什么判出「忠实度 5.9%」**：那一题的 gold 只有 1 篇，
而模型实际读了 4 篇 —— 拿 gold 当参照，另外 3 篇的内容全被当成编造。

新增 5 项测试钉住：`contexts` 记到了片段与全文、目录不算上下文、字符上限生效、
`query_with_sources` 返回父文档正文、空检索时返回空列表。

### 前置步骤 3b：验裁判有没有区分力 —— **已做完** ✅（零生成成本，只花了 27 次裁判调用）

**结果：第 5 版裁判第一次通过校准。**

| 对照 | 做法 | 实测（n=6，全 `easy`，`single` 臂） |
|---|---|---|
| 真上下文 | 参照物 = 模型实际读到的正文 | **6/6 忠实 = 100%**，0 次裁判失败 |
| 打乱上下文 | `--shuffle-context`，参照物轮转一位 | **0/6 忠实**，6 次全判不忠实 |
| 注入假细节 | `eval/judge_probe.py` | 原答案 `True` / 注入矛盾数字 `False` / 注入合理但原文没有 `False` |

**修掉的两个仪器故障**（都先表现为「0/6 忠实」这种极端值，不是系统变差）：

| # | 故障 | 症状 |
|---|---|---|
| ⑤ | 参照物被截两次（`ref[:9000]` → `gold[:3000]`），实际只喂 3,000 字 | 参照物 5,378~24,671 字，裁判只看得到 12%~56% → 答案里来自被截掉部分的内容全被判编造。真臂 **0/6**、打乱臂也 **0/6**，**没有区分力** |
| ⑥ | 参照物变大后，推理把 `max_tokens=2500` 吃光 → 空 `content` | 6 条里 2 条 `judge_fail`；`finish_reason` 之前没记，只能靠猜（已加上，现在报 `length`） |

**复现命令**（改裁判 prompt 或参照物口径后必跑）：

```powershell
.\.venv\Scripts\python.exe eval\run_gen_eval.py --limit 6 --kind easy --arms single
.\.venv\Scripts\python.exe eval\run_gen_eval.py --judge-only --arms single --report eval\last_gen_subset.json
.\.venv\Scripts\python.exe eval\run_gen_eval.py --judge-only --arms single --report eval\last_gen_subset.json --shuffle-context
.\.venv\Scripts\python.exe eval\judge_probe.py
.\.venv\Scripts\python.exe eval\run_gen_eval.py --show --report eval\last_gen_shuffle.json   # 读裁判理由
```

⚠️ `--judge-only` 默认动的是**全量报告**；做小样本实验必须加 `--report eval\last_gen_subset.json`，
否则会把 76 条全量报告重新裁判一遍（白花钱）。对照臂写 `last_gen_shuffle.json`，**不覆盖**真实报告。

### 3c：全量忠实度 —— **仍未测量**（剩下的唯一一步，要花钱）

```powershell
# ① 重新生成一次（现有全量报告是加 context_texts 之前生成的）—— 约 29.5 分钟 + 真实费用
.\.venv\Scripts\python.exe eval\run_gen_eval.py
# ② 拿真实上下文裁判全量：38 条可答 × 2 臂 = 76 次裁判调用（每次参照物约 1.5 万 token）
.\.venv\Scripts\python.exe eval\run_gen_eval.py --judge-only
```

**成本量级**（用实测数字推，不是估）：生成按上一轮的真实用量
（`single` 78 次调用 / 44.7 万 prompt token；`agent` 275 次调用 / 137.6 万 prompt token），
裁判再加约 76 次 × 1.5 万 ≈ 114 万 prompt token。**做之前先确认这笔钱值不值。**

**这一版的边界（别把 6 条的结果当结论）**：n=6、全是 `easy`、只有一个臂。
它证明的是「裁判不是橡皮图章」，**没有**证明「裁判算得准」。
最难判的是「措辞合理、来源是上下文里另一篇」的断言，那仍然只能人工复核。

---

## 方向 3 之外的收获：口径 bug 与门禁（顺手修掉）

- **`partial` 在 Agent 对照层恒为 0.0%**：`run_agent_eval.py` 只在 `expect` 非空时算命中，
  而 `partial` 的 `expect` 恒空 → 五个臂全 0.0%，而且**永远不动**（门禁里的废行）。
  这个 bug 在生成层用 `cited_any` 修过、Agent 层漏了。**根因是类别口径被抄了四份**，
  现在收进 `scoring.py: bucket_of()`，四个脚本同源（有测试钉住）。
- **评测脚本的导入有副作用**：它们在导入期 `load_dotenv()`，会把 `.env.notes` 的
  `DATA_PATH` / `FILE_GLOB` 灌进 `os.environ`，污染其它按默认值断言的测试
  （实测把 `_tmp_config` 的用例搞挂）。测试里要用 `_isolated_env()` 包住。
- **完整 PASS 首次跑通**（含生成层，`exit 0`）：**每一项差值都是 +0.0%** ——
  不只是「没退化」，是逐位复现。整条评测栈现在是确定性的。

---

## 方向 4（很小）：Agent 臂的 `partial` 只有 66.7% 明说「不完全」

**依据**：`partial` 3 条里，两臂都能指出相关方向（100%），但**明确声明「语料不完全」**
单轮 100%、Agent 只有 66.7%。它列了相关资料，却没明说「这不完全等于你要的」。

**改哪里**：`agent/loop.py` 的 `SYSTEM_PROMPT` —— 在收尾规则里加一句：
「如果检索到的内容只能部分回答，要明确说清**哪一部分**没有覆盖，再列相关的几篇」。

**验证**（实测 31.7 秒 / 3 次调用）：

```powershell
.\.venv\Scripts\python.exe eval\run_gen_eval.py --arms single,agent --kind partial
.\.venv\Scripts\python.exe eval\run_gen_eval.py --show
```

**预期**：`【partial 类】` 的「声明不完全」agent 从 66.7% 升到 100%。
**这个数由 `scoring.abstained()` 判定，确定性、不需要裁判。**
但要**读一眼三份答案**确认它不是硬套模板（`--show` 就是干这个的）。

> 子集运行**不会覆盖** `eval/last_gen_report.json`（全量报告是 `--rescore` 的依据），
> 结果写到 `eval/last_gen_subset.json`，`--show` 默认读它。

---

## 命令速查

### 环境

```powershell
cd basic_rag
```

> ⚠️ `requirements.txt` 里**没有 CUDA 版 torch**。重排要 GPU 必须单独装，见 README「重排序」一节。
> 关键坑：**别用 `--index-url`** —— 索引页指向的 `download-r2.pytorch.org` 返回 403，
> pip 会静默卡住（缓存零增长）；直接用主站 URL。

### 测试（5~7 秒，必跑）

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

### 评测（零 LLM 成本）

| 命令 | 作用 | 实测耗时 |
|---|---|---|
| `python eval\run_all.py` | **回归门槛**：跑三层评测并与基线比，退化就 FAIL | 约 3.5 分钟 |
| `python eval\run_all.py --update-baseline` | 把当前结果立成基线（**先在正常配置下重跑一遍**） | 秒级 |
| `python eval\run_eval.py --json` | 检索层：分类型 hit@5 / 覆盖率 / 闸门阈值扫描 | **2.4 分钟**（今日重跑实测 **8.3 分钟**，差 3.5 倍，原因未查明） |
| `RERANK_ENABLED=false` + 上面那条 | 同上但不重排（验证回归门槛） | **0.4 分钟** |
| `python eval\run_agent_eval.py --json` | Agent vs 单轮（脚本策略） | **1.0 分钟** |
| `python eval\token_rarity.py --max-df 3` | 列低 df 术语，用来出 `rare_token` 题 | 几秒 |
| `python eval\build_golden.py` | 校验并生成 `golden.jsonl`（含 probe 机器校验） | 几秒 |
| `python eval\build_review.py` | 生成 `eval\gate_review.md` 人工复核表 | 未精确计时 |

### 评测（会调 LLM，花钱）

| 命令 | 作用 | 实测耗时 / 用量 |
|---|---|---|
| `run_gen_eval.py --limit 4` | 小样本验证判分逻辑 | **87 秒** |
| `run_gen_eval.py --limit 6 --kind easy --arms single` | 裁判校准用的 6 条 | **98.6 秒 / 6 次调用 / 28,054+18,766 tok** |
| `run_gen_eval.py --arms single --kind partial` | 只跑 partial 那 3 条 | **31.7 秒 / 3 次调用** |
| `run_gen_eval.py`（全量） | 78 题 × 2 臂 | **29.5 分钟**（1771 秒） |
| `run_gen_eval.py --rescore` | **用已保存答案重算**，改判分不必重跑 | 几秒，**0 次调用** |
| `run_gen_eval.py --show` | 打印已存报告里每题的答案与判分依据 | 几秒 |
| `run_gen_eval.py --judge-only [--report …]` | 只补裁判（默认动全量报告，**加 `--report` 才只动子集**） | 6 条约 **2 分钟**；76 条约 10 分钟（未精确计时） |
| `run_gen_eval.py --judge-only --shuffle-context` | 裁判的**打乱对照臂** | 同上 |
| `judge_probe.py` | 裁判校准探针（把已知忠实的答案弄坏） | **3 次调用** |

全量一次的真实用量（来自 API 返回，非估算）：

```
single  78 次调用   prompt 447,408 tok   completion 150,332 tok
agent  275 次调用   prompt 1,375,564 tok completion  65,080 tok
```

> `completion` 里含**推理模型的思考 token**，所以输出成本比看起来高。
> 具体花多少钱按你账号单价算，这里不替你估。

### 跑起来

```powershell
.\.venv\Scripts\python.exe main.py                 # 单轮 RAG（食谱示例，读 .env）
.\.venv\Scripts\python.exe main_agent.py --check   # 先验模型支不支持工具调用
.\.venv\Scripts\python.exe main_agent.py           # Agent（读 .env.notes）

# 常驻服务（实测冷启动 21s，稳态 11.7s/请求）
.\.venv\Scripts\python.exe -m uvicorn service:app --host 127.0.0.1 --port 8000
curl.exe http://127.0.0.1:8000/health
```

---

## 改代码时的纪律（每条都是踩出来的）

1. **指标恒为极值时，先怀疑指标。** `partial` 的「给方向 0%」是指标 bug（拿 gold 判一个
   没有 gold 的类），我差点据此去做一个不需要做的功能。**真去读几条答案就能发现。**
2. **先扩样，再下结论。** 基于 n=3 得出过「重排分数能分开语料外」，扩到 40 条就翻了。
3. **两臂数字不相等时，先怀疑自己的口径。** `agent_stop` 比 `single5` 低 2.7pp，
   查出来是**重排池深不一致**，不是机制差异。
4. **判分器也是被测对象。** 弃答判分器错了两次（全文子串、长度门槛），每次都产出
   「看着很有说服力」的假结论。它现在有单测。
5. **指标名字宁保守勿夸大。** 「要点覆盖率」不叫「正确性」；「声明缺口」不叫「编造」。
6. **测不出作用的东西就删。** 枚举式提问加深候选池：6 条触发、覆盖率 46.7% → 46.7%，删了。
7. **性能数字要用真实数据测。** 用单字符字符串测出「50 对 0.23 秒」，真实 chunk 是 250 token，
   **差了 50 倍**。
8. **LLM 裁判要两个方向的对照才可信。** 恒点头（100%）和恒摇头（0%）都会给出漂亮的假结论。
   打乱上下文抓前者，`judge_probe.py` 注入假细节抓后者。**只做一个方向等于没验。**

---

## 模板复用体检：**已修好并验证通过**（2026-09 实测）

做法：`EVAL_SKILL=recipe` 真的跑一遍（recipe 有 322 篇菜谱语料，一直留着做复用验证）。
**没有靠推测**——问题都是跑出来或代码里查出来的。

第一轮跑出来的 **10 个问题**（当时的记录，位置在下面「原始问题清单」）：3 个阻断 + 1 个崩溃 +
1 个静默给错文章 + 5 个语义错位；后来修的过程中又暴露出第 10 个（索引指纹不含领域元数据）。

### 改了什么（6 处，全部在"通用层"）

| # | 改动 | 文件 |
|---|---|---|
| 1 | `RAGSkill.required_metadata`：规范字段（`article_id`/`title`）成为 skill 契约 | `rag_core/skill.py` |
| 2 | 加载期**契约校验**：缺字段 / id 重复 → 直接报错（以前静默变 None） | `rag_core/loader.py` |
| 3 | recipe 把领域 id 映射到规范名（路径最后两段，与 notes 同一套约定） | `skills/recipe/metadata.py` |
| 4 | 领域文案从通用层挪进 skill：`agent_identity`（角色/语料/id 示例/"没找到"的说法） | `skill.py` + `agent/loop.py` + `agent/tools.py` |
| 5 | 评测层鲁棒性：闸门分组**跳过空组**（以前除零崩溃）；`top_articles` 缺 id **报错** | `eval/run_eval.py` |
| 6 | **索引指纹纳入领域元数据**：改 `metadata_extractor` 后必须重建索引 | `rag_core/indexer.py` |

第 6 条是跑的时候才暴露的：指纹原先只含「模型+内容+来源」→ 改完 extractor 后
**向量路仍从索引 docstore 返回旧元数据**（没有 article_id），而 BM25 路用的是刚加载的新文档，
两条路元数据不一致，评测崩在「检索结果缺少 article_id」。

### 验证结果（两端都通）

| 层 | 结果 |
|---|---|
| 检索层 | `easy` **4/4 = 100%**、`rare_token` **1/1 = 100%**；k=1/10/60/100 全 100%；**退出码 0** |
| Agent 对照层 | 五臂全 **100%**；gold 名次 ≤5 达 5/5 |
| 闸门 | `佛跳墙怎么做`（语料里确实没有）→ `needs_human 1/1`、检索被闸门拦下 1/1 |
| 回归 | **126 项测试通过**（新增 6 条契约测试）；notes 侧复验见下 |

**顺带证明契约校验有用**：它抓到了**我自己两次写错的 id 方案** ——
先用「目录名」当菜名，撞上 `meat_dish/红烧肉/` 下有两份变体；
改成「类别+菜名」又撞上 `soup/陈皮排骨汤.md` 与 `soup/陈皮排骨汤/陈皮排骨汤.md`。
**语料里确实存在重复内容**（同一道菜两种布局各一份）——这是数据问题，交给人决定，没有偷偷删。

### 还需要人做的

- `eval/seeds.recipe.jsonl` 现在只有 **6 条诊断集**（每条 note 都标了「诊断用」），
  要当正式评测得扩到 30 条左右，并补 `mentioned` / `partial` 类。
- recipe 语料里的重复菜谱要不要去重，是**内容决策**。

### 原始问题清单（第一轮实测，留档）

| # | 问题 | 证据 |
|---|---|---|
| 1 | id 契约缺失：recipe 产出 `category/difficulty/dish_name`，**0/322 有 `article_id`** | `有 article_id 的文档：0/322` |
| 2 | `top_articles()` 把 5 篇塌成 `[None]` → hit@5 **全线 0.0%** | 连"宫保鸡丁怎么做"都是 0 |
| 3 | `search_notes` 返回 0 条 → Agent 路径完全失效 | `命中 None 条` |
| 4 | `read_article(None)` **不报错，返回语料里某一篇** | `by_id` 键全塌成 `None` → 静默给错文章 |
| 5 | 闸门分组缺类时 **ZeroDivisionError** 崩掉整个评测 | `run_eval.py:292` |
| 6 | `list_articles` 列 322 篇但 **title 全 None** | — |
| 7 | `list_articles(week=…)` / 排序键是 notes 专有概念 | — |
| 8 | 工具描述与 SYSTEM_PROMPT 写死 notes 文案与 id 示例 | 会指示模型编造不存在的 id |
| 9 | recipe 的 `.env` 没设 `RERANK_*` → 默认开重排 + CPU → 每查询约 24 秒 | 已改成 `RERANK_ENABLED=false` 并注明 |
| 10 | 索引指纹不含领域元数据 → 向量路返回旧元数据，与 BM25 路不一致 | 修指纹 |

---

## 已知未做（别以为做完了）

- **全量忠实度仍未测量**。裁判已校准（6/6 vs 0/6 + 探针 3/3），口径缺陷也修了（第 ⑥ 个坑），
  但全量报告还没重新生成 —— 这是 **3c**，要花钱。
- **`context_max_tokens=6000` 偏小，已在冒烟里实测到截断**：文章均值 3835 字符（中文≈1 字 1 token），
  读 2~3 篇就超预算。截断本身不报错、只写日志，容易被忽略；而它**直接影响答案质量与忠实度口径**。
  调大它是「真金白银换质量」的取舍（prompt token 变多），**没有实测过不同取值的收益**。
- **`mentioned` 类闸门仍然拦不住**（AUC 余弦 0.89 / 重排 0.78）：系统会自信地回答语料答不了的问题。
  零误伤下重排只能拦下 28.6% 的 `mentioned`。真正的解法是补语料，不是调阈值。
- **`paraphrase` 只有 66.7%** —— 换种说法的题有 1/3 检索不到。这是目前最大的可测缺口，
  且**不是 LLM 的问题**。（`single10` 能到 75%、`single25` 到 83.3%，即「多给几篇」比「让模型改写」有效。）
- **两臂引用格式不一致**（单轮 100% 用标题、Agent 100% 用 article_id），前端没法给标题加链接。
- 服务层 `SERVICE_CONCURRENCY=2` 只是防 OOM 的保守值，**没有实测调优过**。
- 增量索引：源文件删除/修改靠指纹全量重建，没有单篇增删。
- 重排依赖 GPU：CPU 上整池重排约 24 秒/次检索，实际不可用。
- **一次偶发**：用管道过滤输出时 `run_all.py --skip-run` 曾返回退出码 1，
  重跑两次都是 0，**没复现、原因未查明**。
