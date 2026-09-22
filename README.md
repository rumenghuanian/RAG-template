# basic_rag —— 一套「可测量、可复用」的 RAG 系统

> 能跑起来的 RAG 很多，但大多数答不上一个问题：**你凭什么说它好？**
>
> 这个项目的重点不是把 RAG 做出来，而是**给它装上量具** —— 并且在这个过程中，
> 把自己量具的 **7 次故障**和它们造成的假结论全部抓出来、记下来。
> 下面每一个数字都能用本仓库的脚本复现；**测不出来的东西，我不写进来。**

## 一屏速览

| 维度 | 说明 |
|---|---|
| 领域可插拔 | `notes`（105 篇课程笔记 / 1168 块）、`recipe`（322 篇菜谱）—— 同一套引擎；skill 的 6 个钩子**都有默认值**，新领域最少只写一个名字 |
| 检索 | 向量 + BM25 混合 → RRF 融合（按文档频率自适应权重）→ cross-encoder 重排 → 按文章去重；父子分块（子块检索、父文档进上下文） |
| 生成 | 单轮 RAG（`pipeline.query`）与 ReAct Agent（多步检索 + 引用收集 + 转人工判定）两条路 |
| 服务 | FastAPI（`/health` `/query` `/agent`），启动时预热重排模型 |
| 评测 | 78 条标注（可答 / `absent` / `mentioned` / `partial` 四类口径）、LLM 裁判（**双向校准**）、三层回归门禁 |
| 测试 | **130 项**，含契约校验、量具口径、回归门禁的回归测试 |

## 实测数字（全部由仓库内脚本产生，可复现）

**检索层**（44 条可答标注，`python eval/run_eval.py`）

| 类型 | n | 自适应融合 | **重排** |
|---|---|---|---|
| easy | 12 | 100.0% | 100.0% |
| paraphrase（换种说法） | 12 | 58.3% | **66.7%** |
| rare_token（罕见术语） | 15 | 93.3% | **100.0%** |
| multi（答案散在多篇） | 5 | 100.0% | 100.0% |
| **总体 hit@5** | 44 | 86.4% | **90.9%** |

**闸门**（语料答不了的问题该不该拒答）：零误伤下，重排分数能拦下 **88.2%** 的域外问题，余弦只有 52.9%；
但「术语出现、语料没讲」这类难题上**余弦反而更会分**（AUC 0.89 vs 0.78）。

**生成层**（单轮臂，`python eval/run_gen_eval.py`）

| 指标 | 数值 |
|---|---|
| 引对率（材料给了 + 引对了 + 没乱弃答） | **88.6%** |
| 裁判忠实度（参照物 = 模型实际读到的正文） | **97.7%**（43/44） |
| 要点覆盖率（打乱对照 12.5%） | **78.7%**（Δ +66.2%） |
| 闸门题「声明缺口」 | **96.8%**（30/31） |

**裁判 vs 人工**（44 条我逐条标，`python eval/score_human_review.py`）：一致率 **97.4%**（37/38），
裁判偏宽 **0** 条、偏严 **1** 条。

> ⚠️ 这个数只能证明裁判**不偏严**。我那 38 条里**一条「不忠实」都没标出来**，
> 所以「偏宽 0 条」是**空的一格**，证明不了裁判抓得住编造 —— 抓编造由**注入探针**负责
> （把已知忠实的答案人为弄坏，看它抓不抓得住）。口径的边界写在「裁判凭什么可信」一节。

**成本是真数不是估算**：78 次调用 / 443,173 prompt token / 160,077 completion token / 830 秒。

**Agent 值不值得用**（等预算对照）`single25` **95.5%** > `agent_explore` 93.2% = `single10` ——
「多查询改写」没有收益，而它每次要多花 4.1 次 LLM 调用。**想答得更多，直接调大 `top_k` 更划算。**

## 这个项目最特别的地方：评测本身也是被测对象

判分器写错一次，结论就全废一次。这里 7 次故障都差点变成「结论」，每次都是靠**对照实验**抓出来的：

| # | 量具故障 | 它给出的假结论 |
|---|---|---|
| ① | 弃答判定用全文子串匹配 | 4 条正常回答里 3 条被判成「弃答」 |
| ② | 加了「< 250 字才算弃答」的长度门槛 | 长弃答被漏判 → **「Agent 编造率 80%」** |
| ③ | 裁判拿 gold 当参照，而模型实际读了 2~5 篇 | **「忠实度 5.9%」** |
| ④ | 裁判 `max_tokens` 被推理模型的思考吃光 | **「忠实度 100% vs 0%」**（74/76 次解析失败） |
| ⑤ | 参照物被截了两次（9000 → 3000 字） | 真上下文 **0/6**、打乱对照也 **0/6**（指标钉死） |
| ⑥ | 参照物记的是**未截断全文**，模型收到的是截断版 | 模型如实说「文档被截断」→ 被判成编造 |
| ⑦ | 校准探针拿**修 ⑥ 之前**生成的报告当种子（记的上下文比模型实收的多） | **「裁判校准不过关」** |

配套的两个方向校准：**打乱上下文**（抓「恒点头」）＋ **注入假细节的探针**（抓「恒摇头」）。
只做一个方向等于没验。

> ⑦ 值得单独说一句：**修好的量具，会被旧的存档数据重新弄坏**。
> 探针的判据是「已知忠实的答案 → 应当判忠实」，而「已知」来自报告里存的裁判结论；
> 那份旧报告记的上下文比模型实收的多，裁判读到模型根本没看到的那段，自然判它矛盾。
> 现在有两道闸：默认报告指向修好之后的全量报告、自动挑一条存有 `faithful=True` 的答案当种子；
> 再按 token 预算核一遍（实收正文必然 ≤ `context_max_tokens`，超了就是记错了，直接拒跑）。

## 30 秒跑起来

**先不配任何语料，看它真的在跑**（仓库自带一份示例语料，`demo` 领域）：

```powershell
git clone https://github.com/rumenghuanian/RAG-template.git; cd RAG-template
python -m venv .venv; .\.venv\Scripts\activate
pip install -r requirements.txt

$env:EVAL_SKILL="demo"
python eval\run_eval.py       # 检索层：easy hit@5 100%、域外问题被余弦拦住（零 LLM 成本）
python eval\run_agent_eval.py # Agent 对照层：闸门题 needs_human 1/1（同样零 LLM 成本）
python -m pytest tests/ -q    # 137 项
```

示例语料是 [`examples/demo_corpus/`](examples/demo_corpus)（两篇自写的手冲咖啡文档，
**只有 2 条标注**，够证明「装好依赖 → 出数字」这条路通，不代表效果）。

**换成你自己的语料**：

```powershell
copy .env.example .env        # 改 DATA_PATH 指向你的 markdown 目录（其余变量都有默认值）
python main_agent.py          # Agent 问答（生成层要 LLM_API_KEY）
python eval\build_golden.py   # 校验你自己的标注
python eval\run_eval.py       # 跑你自己的数字
python eval\run_all.py        # 三层回归门禁（退化则退出码 1）
```

> **语料自备**：两份真实语料（课程笔记 / 菜谱）都是第三方内容，没有随仓库分发 ——
> 所以 `clone` 下来**直接跑 `run_eval.py`（默认 notes）会报「数据目录不存在」**，这不是 bug。
> 只要满足 `DATA_PATH` + `FILE_GLOB` 指向一批 `.md`，字段规范（`article_id` / `title`）
> 缺了或重了**加载期就会报错**、不会静默失效，就能直接跑。
> 什么都要自己造吗？**只有语料和标注**：`LLM_API_KEY` 只有生成层要（检索/Agent 两层零成本），
> 索引目录、prompt、元数据映射、领域身份全都有默认值。

**接一个新领域**：见 [`docs/ADDING_A_SKILL.md`](docs/ADDING_A_SKILL.md) ——
**零配置版只要一个文件**（`RAGSkill(name="my_domain")`）+ 注册一行 + 一个 `.env`
（只写 `DATA_PATH`；索引默认隔离到 `vector_index/<skill>`），`metadata.py` / `prompts.py`
**需要时才加**。

这个模板的"可插拔"不是口号，而且**可复现**：
- `rag_core/skills/demo/__init__.py` 就是**零配置的活例子**——整个文件只有一个
  `RAGSkill(name="demo")`，配 `examples/demo_corpus/` 与 `.env.demo` 就能跑出数字；
- 已在 `notes`（105 篇笔记）与 `recipe`（322 篇菜谱）上跑通；
- `python eval/verify_zero_config.py` 造一个**临时第三领域**（3 篇任意 Markdown、只写一个名字），
  验证建索引 → 检索命中 → 默认 prompt → 工具 schema 无外来参数（零 LLM 成本）。

> 一条边界：**"导入即可用" ≠ "导入即可量化"**。要出数字仍然得写
> `eval/seeds.<skill>.jsonl` —— 判断"这个问题语料答得了吗"是人的事，自动化不了。

---

# 附：详细笔记

下面是从需求到实现、再到评测口径的完整记录（含每个结论的失败版本与修正过程）。

## 目录结构

```
basic_rag/                          # 项目根
├── data/                           # 你自己的语料放这里（第三方语料不进仓库，见 .gitignore）
│   └── cook/
│       └── dishes/                 # ← 你的 markdown 数据放这里
│           ├── meat_dish/
│           ├── vegetable_dish/
│           ├── soup/
│           └── ...
├── examples/
│   └── demo_corpus/                # ← 自带的示例语料（自写，可随仓库分发）
│       └── coffee/                 #   配合 EVAL_SKILL=demo 用，clone 下来即可跑出数字
├── vector_index/                   # 索引自动生成
│   └── .fingerprint                # 数据指纹（自动生成，用于判断是否重建）
├── .cache/                         # HuggingFace 模型缓存（如果配了 HF_HOME）
├── .env                            # 环境变量（自己创建，不进 git）
├── .env.example                    # 环境变量模板（食谱语料）
├── .env.notes.example              # 环境变量模板（Agent-100-Days 笔记语料 + Agent）
├── .gitignore
├── requirements.txt
├── main.py                         # 单轮 RAG 入口（食谱示例）
├── main_agent.py                   # Agent 入口（笔记语料）
├── service.py                      # 常驻 HTTP 服务（FastAPI，模型只加载一次）
├── bootstrap.py                    # 入口共用：控制台编码 + NO_PROXY 归一化
├── tests/                          # 回归测试
│   ├── test_fixes.py
│   └── test_agent.py
├── eval/                           # 检索层评测（不调 LLM）
│   ├── seeds.jsonl                 # ← 你的标注（换语料只改这个文件）
│   ├── build_golden.py             # 子串 → 真实 article_id，唯一匹配校验
│   ├── token_rarity.py             # 统计 df，用来挑 rare_token 类型的题目
│   ├── build_review.py             # 生成闸门类标注的人工复核表（gate_review.md）
│   ├── golden.jsonl                # 生成物
│   ├── run_eval.py                 # 分类型 hit@5 / cover@5 + k 敏感性 + 闸门拦截率
│   └── run_agent_eval.py           # Agent 对照评测（脚本策略，同样零 LLM 成本）
├── agent/                          # ← Agent 层（把检索当工具）
│   ├── __init__.py
│   ├── llm.py                      # OpenAI 兼容客户端（tool calling）
│   ├── tools.py                    # search_notes / read_article / list_articles
│   ├── memory.py                   # 会话记忆 + 指代消解
│   └── loop.py                     # ReAct 循环 + 引用收集 + 转人工判定
└── rag_core/                       # ← Python 包
    ├── __init__.py
    ├── config.py                   # 配置（从环境变量读）
    ├── loader.py                   # 文档加载
    ├── splitter.py                 # Markdown 结构分块
    ├── tokenizer.py                # 中文分词 + token 估算（BM25 用）
    ├── indexer.py                  # 向量索引 + 数据指纹
    ├── retriever.py                # 混合检索（向量 + BM25 + RRF + 重排）
    ├── reranker.py                 # cross-encoder 重排序（懒加载 + 失败退化）
    ├── generator.py                # LLM 生成
    ├── pipeline.py                 # 组装
    ├── skill.py                    # Skill 协议
    └── skills/
        ├── __init__.py
        ├── notes/                  # Agent-100-Days 课程笔记技能
        │   ├── __init__.py
        │   ├── metadata.py         # 周次 / 篇号 / 标题 / 是否「思考」篇
        │   └── prompts.py
        └── recipe/                 # 食谱技能
            ├── __init__.py
            ├── metadata.py         # 分类/难度提取
            ├── prompts.py          # 烹饪 prompt
            └── strategy.py         # 路由/重写/过滤
```

> 注意：Python 包叫 `rag_core`，不是 `basic_rag`。`basic_rag` 只是项目根目录名。

---

## 快速开始

### 1. 建虚拟环境并安装依赖

**在项目根目录（`basic_rag/`，也就是放着 `requirements.txt` 的那一层）执行：**

```powershell
cd basic_rag

# 本机没有注册 Python 3.12（`py -3.12` 会报 No suitable Python runtime found），
# 用已经注册的 3.11 就行，代码兼容 3.11+：
py -3.11 -m venv .venv

# 或者用 uv 装好的那个 3.12：
# & "$env:APPDATA\uv\python\cpython-3.12-windows-x86_64-none\python.exe" -m venv .venv

.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

之后每次都用这个 venv：

```powershell
cd basic_rag
.\.venv\Scripts\Activate.ps1
python main.py          # 或 python main_agent.py
```

> **必须装在项目自己的 venv 里**。全局环境如果已经有 LangChain 1.x，`BM25Retriever`
> 的 import 路径变了，本项目会直接起不来（`requirements.txt` 已经约束到 `<1.0`）。

### 2. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env`，填入你的配置：

```bash
# ===== LLM 配置（OpenAI 兼容接口，任意服务商都行）=====
LLM_MODEL=deepseek-flash
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_API_KEY=sk-你的key

# ===== Embedding 配置 =====
EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
EMBEDDING_DEVICE=cpu

# ===== 数据路径 =====
DATA_PATH=./data/cook/dishes
INDEX_SAVE_PATH=./vector_index

# ===== 可选：HuggingFace 模型缓存位置（默认在 C 盘用户目录下）=====
HF_HOME=./.cache/huggingface

# ===== 可选：离线模式（避免每次启动都去 HEAD huggingface 检查版本）=====
# 前提：模型已完整下载到本地
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1

# ===== 检索/生成参数（可选）=====
TOP_K=3
TEMPERATURE=0.1
MAX_TOKENS=2048
ENABLE_REWRITE=true
# 送进 prompt 的上下文预算（约等于 token 数），要小于模型窗口
CONTEXT_MAX_TOKENS=6000
# 数据文件匹配模式，默认 *.md
# FILE_GLOB=*.md

# ===== 重排序（可选，默认开）=====
# 关了就是纯 RRF 排序；权重缺失/依赖缺失会自动退化成不重排并打 WARNING
RERANK_ENABLED=true
RERANK_MODEL=BAAI/bge-reranker-base
# 送进 cross-encoder 的候选池大小（太小则重排无从选择，太大则 CPU 上慢）
RERANK_CANDIDATES=50
```

**常用 LLM 服务商对照：**

| 服务商 | `LLM_BASE_URL` | `LLM_MODEL` 示例 |
|---|---|---|
| DeepSeek | `https://api.deepseek.com/v1` | `deepseek-flash` |
| Moonshot | `https://api.moonshot.cn/v1` | `moonshot-v1-8k` |
| OpenAI | `https://api.openai.com/v1` | `gpt-4o-mini` |
| 通义千问 | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `qwen-plus` |
| 本地 vLLM | `http://localhost:8000/v1` | 你部署的模型名 |

### 3. 放入数据

把你的 markdown 食谱放到 `DATA_PATH` 指向的目录下。

**每个 markdown 文件建议的格式：**

```markdown
# 宫保鸡丁

★★★★ 中等难度

## 必备原料

- 鸡胸肉 300g
- 花生米 50g
- 干辣椒 10 个
- ...

## 操作

1. 鸡胸肉切丁，加生抽、淀粉腌制 15 分钟。
2. 调酱汁：生抽 + 老抽 + 醋 + 糖 + 淀粉 + 水。
3. 热锅凉油，下花生米炸香。
4. ...

## 附加内容

- 鸡丁不要切太大，否则不易熟。
- 酱汁要提前调好。
```

**约定：**
- 一级标题 `#` = 菜品名
- `★` 数量 = 难度（1~5 颗星）
- 路径里的目录名（`meat_dish` / `soup` 等）= 分类

**支持的分类目录名：**

| 目录名 | 中文分类 |
|---|---|
| `meat_dish` | 荤菜 |
| `vegetable_dish` | 素菜 |
| `soup` | 汤品 |
| `dessert` | 甜品 |
| `breakfast` | 早餐 |
| `staple` | 主食 |
| `aquatic` | 水产 |
| `condiment` | 调料 |
| `drink` | 饮品 |
| `semi-finished` | 半成品 |

> 想加新分类？在 `rag_core/skills/recipe/metadata.py` 的 `CATEGORY_MAPPING` 里加一行即可。

### 4. 运行

```bash
python main.py
```

首次运行会：
1. 加载 markdown → 分块 → 构建 FAISS 索引（保存到 `INDEX_SAVE_PATH`）
2. 生成数据指纹（`.fingerprint` 文件）
3. 启动交互式问答

**后续运行会智能判断**：
- 数据没变（指纹一致）→ 直接加载已有索引，秒级启动
- 数据有变化（新增/修改/删除 markdown）→ 自动重建索引

---

## 数据更新流程

### 加/改菜谱后，**不用手动删索引**

系统通过**数据指纹**自动判断：

```python
# pipeline.py 里的逻辑
current_fp = self.indexer.compute_fingerprint(self.chunks)
if self.indexer.is_index_fresh(current_fp):
    vs = self.indexer.load_index()      # 数据没变 → 复用
else:
    vs = self.indexer.build_vector_index(self.chunks)  # 数据变了 → 重建
    self.indexer.save_index()
    self.indexer.save_fingerprint(current_fp)
```

**指纹基于**：每个 chunk 的内容 + 源文件路径 + embedding 模型名。
**任何一个变了，指纹就变，自动触发重建。**

### 什么时候需要手动删索引？

| 场景 | 需要手动删吗 |
|---|---|
| 加/改/删 markdown | ❌ 自动检测 |
| 换 `EMBEDDING_MODEL` | ❌ 指纹包含模型名，自动重建 |
| 改分块逻辑（标题层级等） | ❌ 自动检测（指纹已包含分块配置） |
| 换 `LLM_MODEL` | ❌ 不用（LLM 和索引无关） |
| 换 `INDEX_SAVE_PATH` | ❌ 新目录不存在，自动重建 |

---

## 支持的查询类型

系统会自动路由到三种模式：

| 类型 | 例子 | 行为 |
|---|---|---|
| `list` | 「推荐几个素菜」 | 返回菜品列表 |
| `detail` | 「宫保鸡丁怎么做」 | 返回详细分步做法 |
| `general` | 「什么是川菜」 | 一般性回答 |

支持**元数据过滤**：
- 「推荐**简单**的**素菜**」→ 自动加 `difficulty=简单` + `category=素菜` 过滤

---

## 换一个领域（写新 Skill）

只需要实现 4 个钩子：

```python
# rag_core/skills/legal/__init__.py
from ...skill import RAGSkill
from langchain_core.prompts import ChatPromptTemplate

def build_legal_skill() -> RAGSkill:
    return RAGSkill(
        name="legal",
        metadata_extractor=None,                        # 不提取额外字段
        splitter_headers=[("#", "章"), ("##", "条")],   # 按章/条分块
        prompt_registry={
            "basic": ChatPromptTemplate.from_template(
                "你是法律助手。根据以下法条回答问题。\n"
                "问题:{question}\n法条:{context}\n回答:"
            ),
        },
        query_strategy=None,                            # 用默认（无路由/无重写）
    )
```

然后在 `main.py` 里换成 `build_legal_skill()` 即可，**核心 RAG 逻辑一行不动**。

**建议**：不同 skill 用不同的索引目录，避免串用：

```bash
INDEX_SAVE_PATH=./vector_index/legal
```

---

## 常见问题

**Q1: 首次运行卡在下载模型？**

`BAAI/bge-small-zh-v1.5` 约 100MB，需要联网。配了 `HF_HOME` 后会下到你指定的目录，下次走缓存。

**Q2: 模型会下到 C 盘吗？**

默认会下到 `C:\Users\<用户名>\.cache\huggingface\hub\`。如果不想占 C 盘，在 `.env` 里配 `HF_HOME=./.cache/huggingface`，模型会下到项目目录。

**Q3: 每次启动都有一堆 `HTTP HEAD https://huggingface.co/...` 日志？**

那是 `huggingface_hub` 在做版本检查，**不下载权重**，只是确认本地缓存是最新的。想彻底关掉：

```bash
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
```

前提是本地缓存已完整（第一次跑完就有）。

**Q4: 换了模型但索引没变？**

索引和 embedding 模型绑定。指纹包含 `model_name` + 归一化开关 + 分块配置 + 每个 chunk 的内容与来源，
任意一项变了都会自动触发重建。指纹缺失或损坏时也会重建，并在日志里说明原因。

**Q5: 路径找不到？**

`DATA_PATH` 是相对「运行 `python main.py` 时的当前目录」。`main.py` 里已经加了 `os.chdir(Path(__file__).parent)`，强制切换工作目录到项目根，所以从任何地方运行都没问题。

**Q6: `template` 目录被当成菜谱了？**

已经移出 `DATA_PATH`：`data/cook/dishes/template` → `data/_excluded/template`。
你自己加数据时，示例/模板类文件放在 `DATA_PATH` 外面，否则会被索引成一个假条目。

**Q7: `.env` 里改了 key 但程序还用旧的？**

`load_dotenv()` 默认**不覆盖**系统已有的环境变量。如果你之前手动设过系统变量，`.env` 里的值不会生效。检查：

```powershell
# Windows PowerShell
echo $env:LLM_API_KEY
```

有输出说明系统变量优先，需要手动清掉或改用 `load_dotenv(override=True)`。

**Q8: 加数据后必须删 `vector_index/` 吗？**

**不需要**。系统会对比数据指纹，数据变了自动重建。这是当前版本相比早期版本的重要改进。

---

## 项目文件清单（第一次跑起来最少需要）

| 文件 | 作用 |
|---|---|
| `.env` | 环境变量（必须自己创建） |
| `main.py` | 入口 |
| `requirements.txt` | 依赖 |
| `rag_core/` | 完整包 |
| `data/cook/dishes/*.md` | 至少 1 个 markdown |
| `vector_index/` | 自动生成，不用管 |

---

## 最小化改动建议

- **换索引目录**：按 skill 名分目录，如 `./vector_index/recipe`
- **加分类**：改 `rag_core/skills/recipe/metadata.py` 的 `CATEGORY_MAPPING`
- **换 prompt**：改 `rag_core/skills/recipe/prompts.py`
- **换领域**：新建 `rag_core/skills/<你的领域>/`，实现 4 个钩子
- **换 LLM**：改 `.env` 的 `LLM_MODEL` / `LLM_BASE_URL` / `LLM_API_KEY`

---

## Agent 模式：把检索当工具

`main.py` 是单轮 RAG（检索一次 → 生成）。`main_agent.py` 是 Agent 模式——
**模型自己决定何时检索、检索几次、要不要读全文**。

### 循环

```
用户问题
   │  ConversationMemory.condense()  指代消解：「那检索阶段呢？」→ 独立问题
   ▼
RAGAgent.run()  ── ReAct 循环 ───────────────────────────────┐
   │  ① llm.chat(messages, tools)                             │
   │  ② 返回 tool_calls → ToolRegistry.invoke → 结果塞回 messages
   │  ③ 不返回 tool_calls → 结束                               │ 最多 max_steps 轮
   └─────────────────────────────────────────────────────────┘
   ▼
AgentResult{answer, citations, steps, stop_reason, needs_human}
```

### 三个工具（`agent/tools.py`）

| 工具 | 作用 | 什么时候被调用 |
|---|---|---|
| `search_notes(query, top_k)` | 混合检索，返回文章片段 + `article_id` | 事实性问题的第一步 |
| `read_article(article_id)` | 读某一篇全文 | 需要细节、步骤或完整论述 |
| `list_articles(week)` | 列出文章清单，可按周筛选 | 「第几周讲了什么」 |

### 为什么手写循环，不用 LangChain 的 AgentExecutor

1. 循环、终止条件、引用收集就是这个项目的核心，必须能逐行讲清楚
2. **LLM 是注入的**，所以用假 LLM 就能把循环测完整——不需要 API key、不需要网络（见 `tests/test_agent.py`）

### 语料：Agent-100-Days 课程笔记（**105 篇**，其中 week16 六篇是后来补的）

| 指标 | 实测 |
|---|---|
| 文章 | **105 篇**（`week*/*.md`，自动排除 `.venv` / `.ipynb_checkpoints` / 代码目录） |
| 分块 | **1168 个**（块均 357 字符） |
| 文章长度 | 197–8701 字符，均值 3835（不含新增 6 篇） |
| 元数据 | `article_id` / `title` / `week` / `article_no` / `is_thinking` |
| 周次分布 | week1 = 1 篇，week2–15 各 7 篇，**week16 = 6 篇（进阶补充）** |

> **week16 的来历**：原来 99 篇里有 5 条闸门题（RAGAS / ColBERT / RLHF·SFT / 幻觉检测 / 语义缓存）
> 是「本课程本来就该讲、但确实没讲」的**领域内缺口**，而不是「语料外问题」。
> 补上这 5 篇后它们从闸门题转成了可答题（详见 `TRY_IT.md` 方向 2）。
> **刻意没补** 那 17 条 `far` 闸门题（红烧肉、COBOL、围棋…）—— 它们是拒答测试的题目，
> 补了等于毁掉量具。

> 文章均值 3835 字符，`TOP_K=5` 走单轮 RAG 时 6000 token 的上下文只装得下约 1.5 篇。
> 所以在这个语料上，Agent 模式（片段检索 + 显式 `read_article` 读全文）不只是「更花哨」，是**真的更合适**。

### 跑起来

```powershell
cd basic_rag
Copy-Item .env.notes.example .env.notes      # 填 LLM_API_KEY
python main_agent.py --check                 # 先确认模型支持工具调用
python main_agent.py
```

`--check` 花一次 API 调用验证模型是否真的返回 `tool_calls`。
**不支持 function calling 的模型会让 Agent 静默退化成单轮问答**，所以先跑这一步。

### 相关性闸门（`MIN_RELEVANCE`）—— 一个被评测否掉的方案

动机：检索**永远**会返回 K 条，而 RRF 分数只编码排名（实测所有命中的 `rrf_score` 都落在
0.0159–0.0164），所以「语料里到底有没有这件事」判断不出来，`needs_human` 也就永远不触发。

**结果：这个方案不成立。** 但结论是**扩样之后才站得住的**，中间我还错过两次：

**错误一：标注错了。** 最初手标的 8 条「语料外」里 5 条其实可答（见下一节）。

**错误二：n=3 得出的结论是假象。** 只有 3 条 `absent` 时，重排分数看着能把「语料里没有」
分开（0.34 vs 0.20），余弦分不开。扩到 **24 条 absent + 16 条 mentioned** 后：

| 判据 | `absent·far`(17) | `absent·near`(7) | `mentioned`(16) |
|---|---|---|---|
| 向量余弦 AUC | 0.95 | 0.86 | **0.84** |
| 重排分数 AUC | 0.96 | 0.87 | **0.71** |

**两者在 `absent·far` 上打平（0.95 / 0.96），而在 `mentioned` 上余弦反而更好（0.84 vs 0.71）。**
之前那句「重排分数能分开、余弦不行」纯粹是小样本的运气。

**错误三：口径比生产乐观。** 我原先用 `_vector_search` 的 top-1 余弦衡量闸门，但生产里
`search_notes` 取的是 `retrieve()` 返回的**那 5 篇**里的最高余弦——而这 5 篇可能**全是 BM25
单路命中、根本没有余弦字段**，闸门读到的就是 0.0。改成生产口径后的阈值扫描
（误伤＝可答题里被判为「没有」的比例）：

| 判据 | 阈值 | 误伤(可答) | `absent·far` | `mentioned` |
|---|---|---|---|---|
| 向量余弦 | 0.45 | **0.0%** | 47.1% → 现在 **52.9%** | 13.3% → 现在 14.3% |
| 重排分数 | 0.25 | **0.0%** | **88.2%**（不变） | 26.7% → 现在 28.6% |

AUC（1.0 完全可分 / 0.5 瞎猜）：

| 判据 | `absent·far` | `mentioned` |
|---|---|---|
| 向量余弦 | 0.95 → 现在 **0.97** | **0.85** → 现在 **0.89** |
| 重排分数 | 0.96 → 现在 **0.97** | 0.73 → 现在 **0.78** |

> **换语料后的复测（可答题 38 → 44，闸门 17 far + 14 mentioned）**：
> 结论没变，而且更清楚了——**零误伤下重排仍能拦下 88.2% 的「完全域外」，余弦只有 52.9%**；
> 而在 `mentioned`（术语被提到但没回答）这一类上，**余弦反而比重排更会分**（AUC 0.89 vs 0.78）。
> `absent·near` 这一列已经没有了（5 条全被补成可答题），难的那一半现在由 `mentioned` 代表。

**能用的只有一件事**：重排分数在**零误伤**下能拦下 88% 的「完全域外」问题（余弦只有 47%）。
**拦不住的是**：「领域内术语但语料没讲」（20%）和「术语被提到但没回答」（27%）。
而后者才是最危险的——模型会拿工具调用示例自信地回答「今天天气」。

所以闸门现在只是「完全跑题过滤器」（默认 0.45）。**它不能作为「语料里没有」的判据**，
真正的判据是模型读到上下文后自己说「笔记里没有找到相关内容」。

> 阶段①的经验：**先扩样，再下结论。** 40 条标注比 5 条多花半小时，
> 但推翻了两个「看着很有说服力」的结论。

### 四类标注，以及人工复核的结论

| 类 | 定义 | 谁能判 | 机器怎么校验 |
|---|---|---|---|
| 可答 4 类 | 语料能回答 | 人出题 | `build_golden` 校验「gold 子串唯一匹配」 |
| `absent` | 术语**完全不出现** | 机器 | `probe.absent` 的词必须 0 篇 |
| `mentioned` | 术语出现，但**被问的那一面**不存在 | 机器 + 人 | `probe.present` ≥1 篇 且 `probe.absent` 0 篇 |
| `partial` | 语料**有相关内容但不完全回答** | **只能人判** | 要求写 `note` 记录依据 |

`partial` 是人工复核后新增的一类。当时那 40 条闸门标注的复核结果：**37 条保留，3 条判为 `partial`**。
（这 40 条后来有变动：5 条「领域内缺口」在 week16 补上后转成可答题，1 条假闸门被修正 —— 见「语料」一节。）
这 3 条的检索分数其实**很高**（余弦 0.69~0.76、重排 0.29~0.93）——
它们**恰好是闸门原理上抓不到、也不该抓的那一类**：语料确实讲了相近的东西。

**它们既不该被折进「可答」**（gold 只能从检索结果反推，会变成循环标注），
**也不该算「闸门类」**（不是「语料里没有」）。所以单独立一类，不计入 hit@5 也不计入闸门统计。

复核人的总结（原文）：

> 很多的问题跟文档本身无关，所以基本所有的回答以及后续进行机器重排后都是无关无意义的答案，
> 其实改成不知道或者让他换个方向……主要的应该是知识库信息太少了。

这个判断有数据支持：40 条里 37 条属于「语料里不存在」或「存在但没回答」。
**这不是闸门坏了，是知识库太小** —— 当时 99 篇课程笔记覆盖不了「GraphRAG 怎么构建」这类问题。
（后来按这条诊断补了 week16：其中 5 条「领域内缺口」已转成可答题，见「语料」一节。）

由此引出一个**产品方向**（尚未实现）：语料答不了时，不要只说「不知道」，
而是**承认不完全 + 给出语料里最接近的方向**（`partial` 那 3 条正是天然的目标场景）。
现在的行为是二选一（要么硬答、要么干巴巴拒答），中间那档缺失。

### 闸门为什么注定不好做：`absent` 和 `mentioned` 是两件事

第一次做闸门评测时，我把 8 条问题标成「语料外」，然后得出「余弦分不开两者」的结论。
后来逐条去语料里查，**8 条里只有 3 条是真语料外**：

| 标为「语料外」的问题 | 索引内实际命中 | 真实分类 |
|---|---|---|
| 如何用 Kubernetes 部署 Redis 集群 | `week15/103.部署与运维.md`（正有一节讲容器化部署） | **可答** |
| 怎么用 Docker Compose 部署微服务 | `week15/103`（Docker Compose 小节 + 对比表） | **可答** |
| 向量数据库该选 Milvus 还是 pgvector | `week5/32.检索技术.md`（有向量库选型对比表） | **可答** |
| LangGraph 的状态机怎么用 | LangGraph 出现 5 次，但**只讲「推荐用它管理状态」，没讲状态机** | `mentioned` |
| 今天天气怎么样 | 「天气」出现 **28 次**（工具调用示例的标配例子） | `mentioned` |
| Python 的 GIL 是什么 / 怎么用 Rust 写操作系统内核 / 红烧肉怎么做 | 0 | `absent` |

被我误判成「闸门失灵」的 `Kubernetes 部署 Redis`（余弦 0.7808），**其实是检索对了**——
那篇文章确实在讲 Kubernetes 和 Redis 的部署。**用错的标注去评测，会得出错的结论，
而且错得很有说服力。**

三类问题的判据根本不同：

- `absent`（术语完全不出现）→ 文本相似度**有可能**判，重排分数已接近可用；
- `mentioned`（术语出现，但语料没回答这个问题）→ **任何基于文本相似度的判据都必然失败**。
  `今天天气怎么样` 的重排分是 **0.9612**，因为这个判断是对的：那段文本和这个问题高度相关。
  失败的不是检索，是「今天」需要实时数据。

所以正确的架构是：**重排分数只用来过滤「术语完全不存在」，其余交给模型读到上下文后自己说
「这里没有」**。想让 `mentioned` 类自动转人工，得靠别的信号（例如「检索到的片段里有没有
包含问句中的关键实体」这种可验证的检查），不能靠相似度阈值。

### 引用只列真正用到的

检索到的 ≠ 用到的。一开始把所有检索命中的文章都塞进「出处」，实测一次回答只用了 2 篇却列了 6 篇。
现在按「答案里是否真的写了这个 `article_id`」过滤（允许省略 `.md`）；
模型一篇都没引用时才回退到全部命中，并打 WARNING。

### 设计取舍

- **转人工判定**：调了工具却一条出处都没拿到 → `needs_human=True`，CLI 明确提示转人工。答不上来就别编。
- **步数上限**：到 `max_steps` 时**不带工具**再问一次，逼模型基于已有信息给结论，而不是直接报错。
- **工具错误不抛异常**：`ToolRegistry.invoke` 永远返回 dict，错误作为 observation 交回模型自己纠正。
- **成本**：每轮多 1 次指代消解调用（无历史时不发）+ N 次工具调用。用规则能省的别用 LLM。

---

## 评测（`eval/`）

```powershell
python eval/run_all.py          # 回归门槛：跑三层并与基线比，退化就 FAIL（退出码 1）
python eval/run_all.py --update-baseline   # 立基线（先在正常配置下重跑一遍评测）
python eval/build_golden.py     # 从 eval/seeds.jsonl 解析标注，生成/校验 eval/golden.jsonl
python eval/run_eval.py --json  # 跑检索层评测，明细写 eval/last_report.json
python eval/run_agent_eval.py --json  # Agent 对照评测（同样零 LLM 成本），明细写 eval/last_agent_report.json
python eval/token_rarity.py     # 挑 rare_token 类型的题目（看哪些术语只出现在少数文章里）
python eval/build_review.py     # 生成闸门类标注的人工复核表 eval/gate_review.md
```

**怎么读 `run_all.py` 的输出**（只看三处，不用翻文件）：

1. **最后一行** `结论：PASS ✓ / FAIL ✗` —— 唯一必须判的。
2. 开头两行的 `标注指纹` / `生效配置` —— 若它提示「标注指纹变了」或「配置不一致」，
   **下面的表就别看了**，那些差异来自标注/配置，不是代码退化；确认无误后重新立基线。
3. 表里标 `← 退化` 的行 —— 只在 FAIL 时才需要看，它直接告诉你是哪个类型掉了多少。

细节在 `eval/_run_all_*.log`（每层一个）和 `eval/baseline/meta.notes.json`（基线是在什么条件下测的）。

⚠️ **跑过实验配置（例如 `RERANK_ENABLED=false`）之后别立刻 `--update-baseline`** ——
它记的是「磁盘上现有的报告」，会把退化数据立成基线（这个坑踩过一次，已加配置指纹警告）。

**换语料**（`EVAL_SKILL` + 对应的一份标注）：

```powershell
# env 文件、标注、golden、报告**都按 skill 分开**，所以换语料不会覆盖上一套
EVAL_SKILL=recipe python eval/build_golden.py   # 需要 eval/seeds.recipe.jsonl（否则报错并告诉你格式）
EVAL_SKILL=recipe python eval/run_eval.py --json
```

> 早先这句写的是「换领域只改这一个环境变量」，**那是假的**：评测脚本把 env 文件写死成
> 优先 `.env.notes`，于是 `EVAL_SKILL=recipe` 只换了 prompt 和元数据提取器，**语料根本没换、
> 而且不报错** —— 实测把 99 篇 Agent 笔记当菜谱载入，还打上 `category=其他 / difficulty=未知`
> 的假标签。现在 env 文件跟着 skill 走（`bootstrap.env_file_for`），评测产物也跟着走
> （`bootstrap.artifact_path`），并有单测钉住。

**换语料只改 `eval/seeds.jsonl`**，两个脚本不认识任何具体语料——`build_golden.py` 用
「article_id / 标题的唯一子串」解析标注，匹配到 0 篇或多篇都直接报错，坏标注进不来。
手写完整 id 很容易打错，而坏掉的评测集比没有评测集更危险。

`run_eval.py` **不调用 LLM**，所以调检索参数可以随便跑、零成本。

### 实测结果

**A. 78 条标注集（week16 之前，可答 38 条）** —— 下面这组数字是当时测的，保留作为对照：

| 类型 | n | 向量 | BM25 | RRF k=60 | 自适应 | **重排** |
|---|---|---|---|---|---|---|
| `easy`（照标题改写） | 10 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| `paraphrase`（换种说法，避开标题词） | 12 | 41.7% | 33.3% | 58.3% | 58.3% | **66.7%** |
| `rare_token`（df≤3 的罕见术语） | 11 | 72.7% | **100.0%** | 81.8% | **100.0%** | **100.0%** |
| `multi`（答案散在 2~3 篇） | 5 | 80.0% | 60.0% | 80.0% | 80.0% | **100.0%** |
| **总体** | 38 | — | — | — | 84.2% | **89.5%** |

**B. 语料扩容后（105 篇，可答 44 条）** —— 补了 week16 那 6 篇之后重跑：

| 类型 | n | 自适应 | **重排** |
|---|---|---|---|
| `easy` | 12 | 100.0% | 100.0% |
| `paraphrase` | 12 | 58.3% | **66.7%** |
| `rare_token` | 15 | 93.3% | **100.0%** |
| `multi` | 5 | 100.0% | **100.0%** |
| **总体** | 44 | 86.4% | **90.9%** |

> ⚠️ **A 和 B 不能直接比大小**：题集从 38 条变成 44 条（新增 4 条 `rare_token` + 1 条 `easy`，
> 另有 1 条 `mentioned` 转正），而且新增的 6 篇本身就是检索的新干扰项。
> 有意义的是**同一组内部**的对比：
> - `paraphrase` 12 条**逐条没动**（58.3% / 66.7%）→ 新增文章没伤到检索；
> - 新增的 4 条 `rare_token` 里有 1 条在无重排时掉出 top-5，**重排后仍是 100%**；
> - `multi` 从 80% 变成 100%（同 5 条题）——推测与文档频率变化影响了 `plan()` 的动态 RRF 权重有关，
>   **这一点没有单独验证**，只记现象。

`k` 敏感性（全部可答题目）：k=1 → 76.3%，k=10 → 81.6%，k=60/100 → 78.9%。
multi 覆盖率@5：向量 36.7% / BM25 33.3% / RRF 46.7% / 自适应 46.7% / **重排 53.3%**。两路平均重叠率 51.3%。

**八个结论：**

1. **`k` 不用调。** k=10 比 k=60 高 1 条题（3pp），在 38 条样本上属于噪声。那个 60 是 2009 年
   论文的惯例值，照用即可——在这个规模上调它是浪费时间。
2. **RRF 是一个「折中」，不是「更好」。** 等权 RRF 在 `paraphrase` 上最好（58.3% vs 向量 41.7%），
   在 `rare_token` 上却**比 BM25 单路还差**（81.8% vs 100.0%）。融合的代价就在这。
3. **BM25 的价值要靠「测出来的罕见词」才能显形。** `rare_token` 组用 df≤3 的术语
   （`AgentExecutor`、`ABAC`、`AIMessageChunk` 等，由 `eval/token_rarity.py` 统计选出），
   向量只有 72.7%，BM25 满分 100%。上一版凭感觉选的 `MCP`/`mem0` 两个检索器都能命中，
   整组白测——**选题必须测量，不能凭感觉**。
4. **多跳问题要看覆盖率，不是 hit@5。** 上一版 multi 组期望最多 6 篇却只测 top-5，
   覆盖率上限只有 83%，还据此得出「RRF 是负收益」的**错误结论**。改成期望 2~3 篇后：
   **重排 53.3% > RRF 46.7% > 向量 36.7% > BM25 33.3%**。
   结论被出题方式翻转，说明**题目没修好之前不要下结论**。
5. **按 query 动态选权重，正好补上 RRF 唯一的短板**（见下节）。`rare_token` 组从 81.8% 回到
   100.0%，**+18.2%**；其余三组一点没变。
6. **上重排序器补的是「排序」而不是「召回」**（见下节）：总体 84.2% → **89.5%**，
   `multi` 80% → 100%，`paraphrase` 58.3% → 66.7%。
7. **评测第一时间抓到一个真 bug**（见 `chunk_id` 一节）：修之前两路重叠率是 **0%**，
   RRF 完全没在工作，但 hit@5 这类指标**看不出来**——因为纯向量本来就够用。
   说明**总分要配合分层诊断**（这里是重叠率），否则坏掉的东西会被当成好的。
8. **标注错了两次，两次都差点得出错误结论**（`APPROVAL`、8 条「语料外」里有 5 条其实可答）。
   见「闸门为什么注定不好做」一节。**评测集本身也要被审计。**

### 按 query 动态选检索策略

`retriever.plan(query)` 只看一件事：query 里有没有**只出现在一两篇文章里的标识符**
（`AgentExecutor`、`ABAC`、`AIMessageChunk`）。有就加权偏向 BM25（权重 0.6/1.4），没有就等权。
**零额外 LLM 调用**，纯本地计算。

边界都踩过（都有测试钉住）：

- **`df=0` 不算罕见。** `pgvector`、`GIL` 在语料里 df=0，那是「答不了」的信号，
  不是「该走字面检索」的信号——把它们当罕见词，等于让语料外问题都走 BM25。
  （这里原先举的例子是 `Kubernetes`，后来发现它 df=3、语料里有专节讲，是标注错了，见后文。）
- **中文稀有二字组不算标识符。** 「最该」df=1，但那只是措辞罕见，不是精确术语。
- **大小写要归一。** 语料里写 `AgentExecutor`，用户可能敲 `agentexecutor`。
- **df 定义必须和 `token_rarity.py` 一致**，否则标注和分类器会打架（见下）。

做过但**删掉**了：对枚举式提问（哪些/分别）加深向量候选池。6 条触发，
multi 覆盖率 46.7% → 46.7%，**零效果**——测不出作用的东西不留。

### 重排序（cross-encoder）：补的是「排序」，不是「召回」

**双编码器 vs 交叉编码器**：向量检索把 query 和 chunk **分别**编码成向量再比余弦，
两者从未「一起读过」，所以只能判断「主题像不像」，分不出「这段到底有没有回答问题」。
交叉编码器把 `(query, chunk)` **拼成一条输入**过模型，直接输出相关性分数。
代价是每对都要过一次模型、没法预计算，所以只能重排一个小候选池，不能检索全库。

`rag_core/reranker.py`，默认 `BAAI/bge-reranker-base`，`RERANK_ENABLED=true` 时启用：

| 类型 | n | 自适应 | 重排 | 增益 |
|---|---|---|---|---|
| `easy` | 10 | 100.0% | 100.0% | +0.0% |
| `paraphrase` | 12 | 58.3% | **66.7%** | **+8.3%** |
| `rare_token` | 11 | 100.0% | 100.0% | +0.0% |
| `multi` | 5 | 80.0% | **100.0%** | **+20.0%** |
| **总体** | 38 | 84.2% | **89.5%** | **+5.3%** |

multi 覆盖率@5 46.7% → 53.3%。

**代价必须说清楚，因为它比质量增益更影响可用性。同一台机器（i5-12500H / RTX 3050 Ti 4GB），
只换设备差了 17 倍：**

| | CPU（`torch` 是 `+cpu` 构建） | GPU（`torch+cu126`，`RERANK_DEVICE=cuda`） |
|---|---|---|
| 模型体积 | 1.1 GB | 同 |
| 加载时间（一次性） | 16 秒 | 11 秒 |
| 峰值内存 | 641 MB → **2.2 GB** | — |
| 一次检索（不重排） | 0.01 秒 | 0.012 秒 |
| 单条 chunk 重排 | 266 ms | **15 ms**（快 17 倍） |
| **一次检索（重排整池 88~95 条）** | 约 24 秒 | **1.24 秒** |

cross-encoder 要把每个 `(query, chunk)` 都过一次 278M 参数的模型，chunk 中位长度 331 字，
所以它天生比检索本身慢好几个数量级。结论：

- **CPU 上整池重排约 24 秒/次检索**，`main_agent.py` 每问一句要等一分钟以上，**实际不可用**。
- **GPU 上 1.24 秒/次检索**，才谈得上交互。**装 CUDA 版 torch 是这个功能可用的前提，不是优化。**
- `RERANK_CANDIDATES` 默认 **0（不截断）**：重排的价值就在于修 RRF 尾部的排序错误，
  先截断再重排等于让它够不着要修的地方——实测截到 50 会让 **2/38 篇 gold 连候选都进不去**。
  想省时间可以设 20~50，成本几乎正比于池深。
- 按篇去重只省约 25%（实测融合池 88 条 chunk 分属 33~52 篇，**并没有聚在同一篇上**），
  `max_length` 512→256 再省约 14% —— 都救不了量级。

**装 CUDA 版 torch（有个坑）**：

```powershell
# PyPI 上的 Windows torch 是 CPU 版（124MB），CUDA 版只在 download.pytorch.org。
# 但**不要用 --index-url**：索引页给出的链接指向 download-r2.pytorch.org，实测返回 403，
# pip 会静默卡住（缓存零增长）。直接用主站 URL 即可：
.\.venv\Scripts\python.exe -m pip install "https://download.pytorch.org/whl/cu126/torch-2.14.0%2Bcu126-cp311-cp311-win_amd64.whl" --no-deps --force-reinstall
```

然后在 `.env.notes` 里设 `RERANK_DEVICE=cuda`（留空则跟随 `EMBEDDING_DEVICE`）。
国内直连慢的话给 pip 加 `--proxy http://127.0.0.1:7890`（实测 0.63 → 4.41 MB/s）。

> 两个测量教训：
> ① 我第一次报的「50 对 0.23 秒」是拿 `['a']*50` 这种单字符字符串测的，真实 chunk 有 250 token，
> **差了 50 倍**——性能数字必须用真实数据测。
> ② 换了设备后我又测出「GPU 只快 2.4 倍」，因为循环里每个配置都**重新加载了一遍模型**，
> 测到的是「加载+推理」。必须先加载、预热，再测纯推理（0.015 秒/条）。

工程上的四点：

- **懒加载**：构造 `RetrievalOptimizationModule` 不碰模型，第一次真正重排才加载权重。
- **失败退化**：没装 `sentence-transformers`、权重缺失、OOM 时记一条 WARNING 后**原样返回**，
  绝不把检索链路带崩。有测试钉住。
- **候选池要够宽**：`top_k=5` 时池子只有 5 条就无从发挥，所以池子取
  `max(top_k, RERANK_CANDIDATES=50)`。另外 `retrieve()` 按篇去重时会缩水，
  开重排后候选也要铺到池子那么大。
- **评测里显式传 `rerank=False`** 固定「重排前」那一臂，否则基线会被悄悄改掉。

**它不能当相关性闸门**（见「闸门为什么注定不好做」）：它能分开「术语完全不存在」，
但对「术语出现却没回答」那一类给出的分数是 **0.96**——而且那个分数是**对的**。

### 标注自己也会错

`APPROVAL` 本来被我标成罕见词（df=1），但分类器算出 df=7。原因：`eval/token_rarity.py`
没做小写归一，把 `APPROVAL` / `Approval` / `approval` 当成三个词，而分类器归一化了。
**同一个概念两套 df 定义，标注和分类器必然打架。** 统一小写后 `approval` 真实 df=7，
换成 `AIMessageChunk`（df=1）。

所以评测集得能被自己的工具校验，不能只靠人工检查——这也是 `build_golden.py` 坚持
「子串必须唯一匹配、匹配不到就报错」的原因。

### 由评测发现的 `chunk_id` bug

`splitter.py` 原来用 `uuid.uuid4()` 生成 `chunk_id`，**每次运行都不一样**；而缓存索引里存的是
上次构建时的 id。于是 `_rrf_rerank` 按 `chunk_id` 去重时两路永远对不上：

| | 修复前 | 修复后 |
|---|---|---|
| 两路共同命中的块 | **0** | 8 – 19 |
| `max_rrf` | 恒等于 `1/61 = 0.01639` | `0.0309 – 0.0328`（≈`2/61`） |

后果不只是「融合失效」：**第一次运行**（现场建索引）id 是匹配的、**之后每次**都是错位的，
所以同一个问题跑第一遍和跑第二遍结果不同——不可复现，任何评测都失去意义。

修法：`chunk_id = md5(f"{parent_id}:{chunk_index}")`，确定性；并把 `chunk_id` 加进数据指纹，
让已存在的旧索引自动重建。

### 由评测发现的 `vector_cos` 丢失 bug（Agent 路径曾整体失效）

`_rrf_rerank` 用 `chunk_id` 合并两路时，BM25 那一轮是**无条件覆盖**：

```python
for rank, doc in enumerate(vector_docs):
    doc_map[key] = doc          # 向量对象，metadata 里带着 vector_cos
for rank, doc in enumerate(bm25_docs):
    doc_map[key] = doc          # ← 同一个 chunk 时把上面那个整个顶掉，余弦一起没了
```

BM25 取 80 条、向量取 20 条，**几乎每个向量命中都被 BM25 覆盖**，于是融合结果里
`vector_cos` 基本全丢。而 `search_notes` 的相关性闸门正是靠它判断「语料里有没有这件事」，
读不到就按 `0.00` 处理：

| 问题 | `_vector_search` 实际余弦 | 闸门读到的 | 结果 |
|---|---|---|---|
| `RAG 是怎么工作的` | 0.8577 | **0.00** | 返回空，判为「语料里没有」 |
| `文本分块有哪些策略` | 0.8282 | **0.00** | 同上 |
| `MCP 是什么` | 0.8460 | 0.5703（侥幸） | 通过 |

**后果是 Agent 模式在真实使用中几乎对每个问题都回「信息不足」**——而 `main_agent.py --check`
只验证模型会不会返回 `tool_calls`，完全验不到这一层。发现它的路径也很典型：
给零成本对照评测写记录器时，`agent` 一臂的 hit 从 82.9% 掉到 14.3%，
顺着「为什么一次检索只碰到 1.7 篇文章」查下去才挖出来。

修法：`doc_map.setdefault(key, doc)`——向量那一路先插入，等价的语义就是
「带余弦的对象优先保留」（内容同为一个 chunk，排序不受影响）。
钉住它的测试：`test_rrf_keeps_vector_cosine_when_both_retrievers_hit`、
`test_rrf_keeps_bm25_only_docs`。

### Agent 到底值不值：零成本对照（`eval/run_agent_eval.py`）

`RAGAgent(llm, tools, ...)` 的 `llm` 是注入的，所以**用一个脚本策略就能跑完整循环，不花一分钱**；
指标全部落在检索层（这一轮碰到过哪些文章），不看答案文字，也就不需要人工评分或 LLM 裁判。

Agent 的收益只可能来自两处，必须分开量：

- **① 会重试**：第一次检索被闸门拦下 → 换个说法再搜（脚本策略 `agent_stop`：一有结果就停）
- **② 看得更多**：多轮检索的并集本身就更大（`agent_explore`：把改写变体全跑完）

② 根本不需要 Agent，调大 `top_k` 就行，所以真正的对照是**等预算**的（下表是开启重排后的最终数字）：

| 臂 | 预算 | hit | paraphrase | rare_token |
|---|---|---|---|---|
| `single5` | 5 篇 | 89.5% | 66.7% | 100.0% |
| `single10` | 10 篇 | 94.7% | 83.3% | 100.0% |
| `single25` | 25 篇 | **97.4%** | **91.7%** | 100.0% |
| `agent_stop` | 5 篇 | **89.5%** | 66.7% | 100.0% |
| `agent_explore` | 实际 8.1 篇 / **4.1 次 LLM 调用** | **94.7%** | 83.3% | 100.0% |

> **语料扩容后复测（可答 44 条）**，结论没变，差距反而更清楚：

| 臂 | 总体(44) | paraphrase(12) |
|---|---|---|
| `single5` | 90.9% | 66.7% |
| `single10` | 93.2% | 75.0% |
| `single25` | **95.5%** | **83.3%** |
| `agent_stop` | 90.9% | 66.7% |
| `agent_explore` | 93.2% | 75.0% |

`agent_explore` 仍然**恰好等于** `single10`（93.2%），而**等预算的 `single25` 更高（95.5%）** ——
「多查询改写」这条路的收益依然为 0，而它每次要多花 4.1 次 LLM 调用。
**要回答得更多，直接调大 `top_k` 比让 Agent 反复改写划算。**

结论（三条，都有数据）：

1. **「会重试」的收益是 0。** `agent_stop` 和 `single5` **逐类完全相同**（都是 89.5%）——
   闸门修好后正常问题根本不会被拦，这条路一次都没触发过。
2. **「改写」不划算。** `agent_explore` 看了 8.1 篇拿到 94.7%；`single10` 看 10 篇也是
   94.7%。**同样的覆盖率，一个花 4.1 次 LLM 调用，另一个 0 次。**
3. **瓶颈是排序，不是召回。** gold 的名次分布：≤5 有 34/38（89.5%），≤10 加 2，
   ≤25 加 1，≤99 加 1，**未召回 0 条**——答案从来不是没检索到，只是排在 5 名之后。
   所以该上重排序器（已做，见上节），而不是让 agent 多搜几轮。

> **这一节我返工过两次，两次都是测量口径的错，不是机制的结论：**
> ① 第一版 `agent_stop` 比 `single5` 低 2.7pp，看着像「agent 有害」。查下去发现是
> **重排池深不一致**：`retrieve(top_k=5)` 只重排 50 条，而评测的 `single5` 走
> `retrieve(top_k=10**6)` 重排整池 88 条。把池深与 `top_k` 解耦后，两臂精确相等。
> ② 更早一版 agent 一臂只有 14.3%，那是 `vector_cos` 丢失 bug（见上节）。
> **两臂数字不相等时，先怀疑自己的口径，再怀疑机制。**

**诚实边界**：脚本策略是我写的规则，它不判断「这 5 篇对不对」（真 LLM 会）。
所以这两臂衡量的是**这套机制在给定策略下的上限**，不等于 DeepSeek 的真实水平，
用途是先证伪——上限都不划算，就没必要花真钱。

---

## 生成层评测（`eval/run_gen_eval.py`）

前面所有评测都止步于「**检索到了正确文章**」。但用户看到的是**答案**——
答非所问、引错出处、语料里没有却硬答，检索层一条都测不出来。
这是项目最大的缺口：**检索层是工程化水平，生成层还是 demo 水平。**

```powershell
python eval/run_gen_eval.py --limit 4     # 先小样本验证判分逻辑（省钱）
python eval/run_gen_eval.py               # 全量 78 题 × 2 臂
python eval/run_gen_eval.py --arms single # 只跑单轮
python eval/run_gen_eval.py --judge       # 跑生成时顺带裁判
python eval/run_gen_eval.py --judge-only --report eval/last_gen_subset.json   # 只补裁判（用已存答案）
python eval/run_gen_eval.py --judge-only --report eval/last_gen_subset.json --shuffle-context
                                          # 裁判的对照臂：配错上下文，忠实度应当掉下来
python eval/judge_probe.py                # 裁判校准探针：把已知忠实的答案弄坏，看抓不抓得住
                                          # （自动挑已知忠实的种子；旧报告会被按 token 预算拒跑）
python eval/build_human_review.py         # 生成人工复核表 eval/human_review.html（不上传，本地标）
python eval/score_human_review.py         # 标完导出 human_labels.json 后：一致率 + 混淆矩阵
python eval/verify_zero_config.py         # 零配置接新领域：造临时第三领域验证「只写一个名字能跑」
```

### 两个臂，以及一个暴露出来的产品问题

| 臂 | 走哪条路 | prompt 要求的引用格式 |
|---|---|---|
| `single` | `pipeline.query()` 单轮 RAG | **文章标题** |
| `agent` | `RAGAgent` ReAct 循环 | **article_id** |

实测两者**各自 100% 用自己那套格式**（`single` 全标题、`agent` 全 id）。
这不是评测瑕疵，是**产品问题**：前端没法给一个标题加链接，也没法把 id 变成可读标题。
要么统一成「标题 + id」，要么在服务层把 id 映射成标题。

### 指标：哪些是确定性的，哪些要花钱

**确定性、零额外成本**（默认就跑）：

| 指标 | 含义 |
|---|---|
| `context_ok` | 正确文章有没有进到生成上下文里（**这是检索的锅，不是生成的锅**）|
| `cited` | 答案正文里有没有真的点出那篇文章（id / id 去 `.md` / 标题 任一）|
| `abstained` | 有没有弃答（结构化信号，见下）|
| `cited_right` | 三者都过 ＝「材料给了、也引对了、没乱弃答」 |

> `cited_right` **不等于「答对了」**，只等于「引对了」。内容正确性需要 `--judge`。
> 名字起得保守是故意的——指标名夸大比指标缺失更危险。

**闸门类（语料答不了的问题）**：`abstained` 是正确行为，`fabricated`（没弃答）**就是编造**。

**需要 LLM 裁判**（`--judge` / `--judge-only`）：拿**模型实际读到的上下文正文**（`context_texts`）当参照，
判「有无参照物之外的**具体**断言」。**不是拿 gold 当参照** —— 见下表的坑 ③。

### 踩过的坑：弃答判分器把 3/4 条正常回答判成了弃答

最初写成「答案里含弃答词就算弃答」，结果 4 条单轮回答里 **3 条被误判**——
模型在**正常作答**时也会说「笔记里没有讲到 X」「没有相关片段」，全文子串匹配必然误伤。
而这条指标正是用来测「有没有编造」的，**判错等于结论全废**。

改成结构化信号（`eval/scoring.py`，有单元测试钉住）：

1. 检索为空（`pipeline.query` 返回固定串）→ 一定是弃答
2. Agent 的 `needs_human` 为真 → 系统级信号，直接采信
3. 否则看**开头 200 字**（`HEAD_CHARS`）有没有命中**弃答词族**（正则，覆盖「笔记里没有/笔记库里没有/
   没有检索到/超出知识范围/不能凭记忆补充」等多种说法）

**刻意不设长度门槛。** 中途加过「< 250 字才算弃答」，结果 Agent 写的**长而完整的弃答**
（「**结论：笔记库里没有找到与 COBOL 相关的内容。**」）被漏判，直接产出
「Agent 编造率 80%」这种失真的数字。删掉门槛之后误判才消失。

判分规则因此被抽成纯函数模块并单独测试 —— 判分器也是被测对象。

### 成本要报真数，不能估

`OpenAICompatLLM` 现在累计 API **真实返回**的 usage；单轮臂用 langchain 的
`get_openai_callback`。报告里给的是调用次数与 prompt/completion token，
端点不返回 usage 时**只记调用次数，不编造 token 数**。

---

## 常驻服务（`service.py`）

CLI 每启动一次要付冷启动；做成常驻服务后这笔钱只付一次。**实测（i5-12500H + RTX 3050 Ti）**：

| | 实测 |
|---|---|
| 服务冷启动（构建 pipeline + 预热重排） | **约 21s** |
| `/query` 稳态每请求 | **约 11.7s** |
| `/agent` 每请求（3 步） | **约 7.0s** |
| 闸门题（agent，转人工） | 3.5s，`needs_human=true` |

> **别把这 12 秒归给检索**：检索本身（含重排）只要 1.2 秒，剩下约 10 秒是 LLM 生成，
> 而 `deepseek-flash` 是推理模型，思考占其中很大一块。
> 想更快只能换非推理模型，或改成流式返回。

**冷启动为什么是 21s 而不是 11.7s**：重排模型是懒加载的，不预热的话那约 10 秒会被推迟进
**第一个用户的请求**（实测首问 18.7s vs 稳态 11.6s）。服务场景下这笔钱该在启动时付清——
现在 lifespan 里会预热，`/health` 报 `ok` 就真的随时可用。

```powershell
# 启动（模型只加载这一次）
.\.venv\Scripts\python.exe -m uvicorn service:app --host 127.0.0.1 --port 8000

# 健康检查：直接看到模型/设备/语料规模/冷启动耗时
curl.exe http://127.0.0.1:8000/health

# 单轮 RAG
curl.exe -X POST http://127.0.0.1:8000/query -H "Content-Type: application/json" `
  -d '{\"question\": \"重排序是怎么做的\"}'

# Agent（多步检索）
curl.exe -X POST http://127.0.0.1:8000/agent -H "Content-Type: application/json" `
  -d '{\"question\": \"生产环境上线一个 Agent 平台要考虑哪些方面\"}'
```

| 端点 | 作用 | 返回 |
|---|---|---|
| `GET /health` | 就绪状态 | 设备、重排模型、语料篇数、冷启动耗时、uptime |
| `POST /query` | 单轮 RAG | `answer` + `sources` + `seconds` + `queued_seconds` |
| `POST /agent` | ReAct Agent | `answer` + `citations` + `needs_human` + `steps` + `stop_reason` |

三个设计决定：

- **app 工厂**：`create_app(pipeline=...)` 支持注入，所以服务层能脱离 torch 与网络做单元测试
  （`tests/test_service.py` 用假 pipeline，6 项用例毫秒级跑完）。
- **处理器写成同步函数**：FastAPI 会把同步处理器丢进线程池，不阻塞事件循环。
  重活（检索 + 生成）本来就该在线程里跑。
- **并发闸**：4GB 显存上重排模型约 1.1GB，并发请求多了会 OOM，所以用信号量兜住
  （`SERVICE_CONCURRENCY`，默认 2）。**这个值没有实测调优过**，只是防 OOM 的保守值；
  `queued_seconds` 会把它暴露出来，便于以后按真实排队情况调。

`service.py` 还顺手补了 `pipeline.query(with_context=True)`——没有「这次回答依据了哪几篇」，
就没法评引用准确性，服务层也没法把出处返回给前端。

### 实测结果（78 条 = 可答 44 + `absent` 17 + `mentioned` 14 + `partial` 3）

> ✅ 这份报告是**修掉「记下未截断全文」之后**按 105 篇语料重跑的（`eval/last_gen_report.json`，
> `single` 臂）：`context_texts` 记的是**模型实收的截断正文**，78/78 条都在 `context_max_tokens`
> 预算内。这一点现在是**可校验的**：`eval/judge_probe.py` 会按 token 预算拒绝旧报告。

| 臂 | 上下文命中 | 引对 | 纯弃答 | **引对率** |
|---|---|---|---|---|
| `single` 单轮 | 88.6% | 93.2% | 0.0% | **88.6%** |

> `agent` 臂**没有重跑**。`92.1%` 那组是 99 篇语料时的数字，已归档为
> `eval/last_gen_report.pre_week16.json`；和上面不是同一套口径，**别混着引用**。

按题型拆（`single`，括号里是裁判判忠实的条数）：

| 题型 | n | 引对率 | 裁判忠实 |
|---|---|---|---|
| `easy` | 12 | 100.0% | 12/12 |
| `paraphrase` | 12 | 66.7% | 12/12 |
| `rare_token` | 15 | 100.0% | 14/15 |
| `multi` | 5 | 80.0% | 5/5 |

`paraphrase` 掉的 4 条**全部是检索层没把 gold 捞进上下文**（`context_ok=False`）：
生成层在拿到的 8 条上一条没丢。**缺口在检索，不在生成。**

闸门题（正确行为是声明「笔记里没有」）：31 条里 **30 条声明了缺口（96.8%）**。
唯一漏掉的是 `Hub-and-Spoke 架构有哪些优缺点？` —— 它正好是标注待改的那条（人工判定应为 `partial`），
也就是**生成层比我标的更准**，不是系统漏答。

`partial` 3 条：都声明了「笔记没直接回答」、也都指到了语料里的相关文章，
但 gold 本身没被检索到，所以不计入引对率。

### 答案正确性：两个口径都要有，一个都不能单独交差

LLM 裁判试了 5 版，前 4 版都不可信（见下）；第 5 版修掉参照物截断、配齐两个方向的对照后，
全量忠实度 **97.7%**（43/44，0 次解析失败）。但公开研究里 LLM 裁判与人工的一致率也只有 75%~87%，
而**我的人工复核里一条「不忠实」都没标出来**（见下）—— 所以另留一个**完全确定、可复算**的口径当底线：
从 gold 文章里**自动**提取「在这篇频繁、在别处罕见」的术语（tf × idf 取前 8 个，剥掉代码块
避免把 `doc`/`scores` 这类变量名当概念），看答案用上了多少。

| 臂 | 要点覆盖 | 打乱对照 | 差值 |
|---|---|---|---|
| `single` | **75.7%** | 10.5% | **+65.1%** |
| `agent` | **75.0%** | 9.9% | **+65.1%** |

**打乱对照**＝把答案配到「下一题的要点」上。对照只有 10%，说明这个指标**确实在衡量
「答案是否用上了这篇文章的概念」，而不是「答案里有没有中文」**。没有这一步验证，
75% 这个数字没有意义。

> 别和裁判的「打乱上下文」混了：**这个是确定性指标的对照**（把答案配到别题的要点上，
> 零成本、可复算）；裁判那个是**LLM 裁判的对照**（把答案配到别题的上下文上，要花钱）。

> **它是覆盖率，不是正确性。** 它说明「有没有用上关键概念」，说明不了「说得对不对」。
> 名字起得保守是故意的 —— **指标名夸大比指标缺失更危险**。
> 术语由脚本自动提取而非人挑：手挑会不自觉地往「答案里一定会出现」的方向选，等于自证。

### `partial` 类：系统已经在给「相关方向」了

那 3 条「语料有相关内容但不完全回答」的问题：

| `partial`（3 条） | `single` | `agent` |
|---|---|---|
| 指出语料里最接近的方向 | **100.0%** | **100.0%** |
| 明确声明「语料不完全」 | 100.0% | 66.7% |

答案里会写「笔记里没有讲知识图谱，**与向量检索相关的内容主要出现在：《检索技术》
《RAG 是怎么工作的》《文本分块》**」—— 这正是理想行为。剩下的小缺口是 **Agent 臂声明
「不完全」只有 66.7%**：它列了相关资料，但没明说「这不完全等于你要的」。

> **这里我出过一次假结论。** 我原本报的是「给方向 0%」，还据此写了个「要做给方向体验」的方案。
> 根因是**指标错了**：`cited()` 判的是「有没有引用 **gold** 文章」，而 `partial` 类的 `expect`
> **是空的** —— 这个数**永远不可能是 0 以外**。
> 之所以能发现，是因为加了 `--show`（打印每题答案）之后**真去读了那 3 条答案**。
> 教训补一条：**指标恒为极值时，先怀疑指标，别急着当发现。**

**成本（API 真实返回）**：`single` 78 次调用 / 447k prompt / 150k completion；
`agent` 275 次调用（每题 3.5 次）/ 1,376k prompt / 65k completion。

**引用格式分布**：`single` **100% 用标题**、`agent` **100% 用 article_id** ——
两臂各自自洽却互不相通，前端拿标题加不了链接、拿 id 显示不了人话。

### 一个反直觉的成本发现：推理模型让输出成本翻倍

`deepseek-flash` 是**推理模型**：每次调用先花 600~1100 token 思考，`content` 才是答案。
实测单轮臂平均 `completion_tokens` 1,927/题、最大 7,257 —— 远超配置里的 `max_tokens=2048`，
因为 DeepSeek 的 `max_tokens` 只限**最终回答**，思考不占额度但**照常计费**。

两个后果：

1. **输出成本大约翻倍**，而思考过程用户完全看不到。
2. **不能用 token 数推断答案有没有被截断** —— 我一开始就是按 `completion ≥ 2048` 去数
   「疑似截断」，数出 32/78 条，全是假警报。

### 判分器踩过的六个坑（都记在这里，因为每个都差点变成「结论」）

生成层评测的价值取决于**判分可信**。这条路上我连错五次，前四次都被自己的对照实验抓住：

| # | 错法 | 假结论 |
|---|---|---|
| ① | 弃答判定用**全文子串匹配**：正常作答里出现「笔记里没有讲到 X」也算弃答 | 4 条单轮回答 3 条被判弃答 |
| ② | 加了「< 250 字才算弃答」的长度门槛 | Agent 的长弃答被漏判 → **「Agent 编造率 80%」** |
| ③ | LLM 裁判只拿 **gold** 那一篇当参照，而模型实际看到 2~5 篇 | 裁判把来自其他上下文文章的引用全判成编造 → **「忠实度 5.9%」** |
| ④ | 裁判 `max_tokens` 太小，被推理模型的思考吃光 → 返回空内容 | **「忠实度 100% vs 0%」**（74/76 次解析失败，纯噪声） |
| ⑤ | 参照物被**截了两次**（`ref[:9000]` 之后又 `gold[:3000]`），实际只喂 3,000 字 | 参照物有 5,378~24,671 字，裁判只看得到 12%~56% → 答案里来自被截掉部分的内容**全被判成编造**：真上下文 **0/6** 忠实，打乱对照也 **0/6** |
| ⑥ | 参照物记的是**未截断的父文档全文**，而 `context_max_tokens` 会把真正喂进去的正文截断并追加「…（内容过长，已截断）」 | 模型如实说「文档被截断」，裁判拿全文一对 → 判成编造。实测唯一的「不忠实」完全是口径造的：参照物 21,228 字 vs 模型实收 2,733 token |

现在：判分规则在 `eval/scoring.py`（纯函数 + 单元测试钉住），报告**显示裁判失败条数**，
并且 `--rescore` 能用已保存的答案重算指标 —— **改判分不必重新生成，省时省钱**。

**③ 的教训最通用：判「忠实」要拿模型实际看到的上下文作参照，不是 gold。**
拿 gold 当参照会把「引用了上下文里其他文章」误判成编造 —— 而正确答案本来就该引用多篇。

**⑤ 的教训更通用：指标钉在极端值（0% 或 100%），先怀疑指标，别先写进结论。**
两次踩坑都是「0/6 忠实」，两次都不是系统变差了，是仪器坏了。所以修好之后必须补两个方向的对照。

### 裁判凭什么可信：两个方向的对照，缺一不可

LLM 裁判本身是**被测对象**。它有两种坏法，都会给出很漂亮的假结论：恒点头（忠实度 100%）
和恒摇头（忠实度 0%）。所以校准要有两个方向：

| 对照 | 做法 | 抓哪种坏法 | 实测（n=6，全 `easy`，`single` 臂） |
|---|---|---|---|
| **打乱上下文** | `--shuffle-context`：把参照物**轮转一位**，每条答案配别题的上下文 | 恒点头 | 真上下文 **6/6 忠实**，打乱 **0/6** |
| **注入假细节** | `eval/judge_probe.py`：把已知忠实的答案人为弄坏 | 恒摇头 + 灵敏度 | 原答案 `True`；注入**矛盾**数字 `False`；注入**合理但原文没有**的细节 `False` |

> 探针的种子必须**真的是**已知忠实：脚本默认用修好之后的全量报告，自动挑一条存有
> `faithful=True` 的答案，并按 token 预算拒绝「记的不是实收正文」的旧报告（否则会误判，
> 见前面第 ⑦ 个坑）。

轮转而不是随机打乱：小样本下随机可能把某条自己的上下文还给它（固定点），那一条就退化成
正常臂；轮转**保证没有固定点**（有测试钉住 `_rotate`）。

**这两个对照证明的是「裁判不是橡皮图章」，不证明「它算得准」。** 后者只能靠人工基准。

### 全量忠实度：**97.7%（43/44）**，以及人工基准说的另一半

105 篇语料重跑后（`context_texts` 记的是模型实收正文），44 条可答题里裁判判 43 条忠实、0 次解析失败。
唯一一条「不忠实」是 `rare_token` 的 `怎么用 Docker Compose 部署微服务`：

> 原文在 Kubernetes 部分明确提到高可用、自动扩缩容，系统却称笔记没有讲到，与原文矛盾。

这是**覆盖度误述**（说了句「笔记里没有」，而笔记里有），不是编造细节。
我人工复核时把它标成了「忠实」—— 所以这是我和裁判的**唯一分歧，属于口径差异**：
按「与原文矛盾」这个口径裁判没错，按我的实用标准少说比编好。

然后是我逐条标的 44 条人工基准（`eval/human_review.html` → `eval/score_human_review.py`）：

| | 条数 |
|---|---|
| 可对比 | 38（另 6 条我标了「拿不准」，不计入分母） |
| 一致率 | **97.4%**（37/38） |
| 裁判偏宽（我说不忠实、裁判说忠实） | **0** |
| 裁判偏严（我说忠实、裁判说不忠实） | **1**（就是上面那条） |

**这个数最该注意的是它的空档**：38 条里我**一条「不忠实」都没标**，所以「偏宽 0 条」不是
「裁判不偏宽」的证据，只是没有样本。偏宽方向只能靠注入探针（把已知忠实的答案人为弄坏）。
公开研究里 LLM 裁判与人工的一致率约 75%~87%，我这个 97.4% 高于它，也符合「样本里没有难例」这个解释。

我标「拿不准」的 6 条还说明一件事：它们**全部**是关于**取舍与结构**的（「没有相关信息就别硬扯」、
「这段是概念不是原因」、「这个前提该写在前面」），**没有一条是编造**。
也就是说**「忠实性」这个口径本身没盖住我不满意的地方** —— 这是口径的边界，不是裁判的错。
其中 2 条（`single#25`、`single#27`「查不到就别输出相关内容」）指向同一个待改的提示词问题：
**检索没东西时，模型会拿相关但无用的内容凑。**


### 生成层一上来就抓到一个检索层看不见的问题

`AgentSkillBot 是怎么实现的？` 是 `rare_token` 标注，检索层的 hit@5 算它**命中**
（`week4/27` 里确实有 `AgentSkillBot` 这个词）。但生成层的答案说得很清楚：

> 笔记里没有讲到名为 AgentSkillBot 的整体实现。笔记中唯一出现「AgentSkillBot」的地方，
> 是在《从工具到技能》的 `fetch_content.py` 示例里，作为请求头出现。

**词在语料里，但语料没回答这个问题。** 检索层的 hit@5 原理上测不出这件事——
它只知道「正确的文章被捞出来了」。同类被生成层抓出来的还有：
`ConversationBufferWindowMemory 怎么用？`（只讲了概念没讲用法）、
`如何用 Kubernetes 部署 Redis 集群`（只提到 Kubernetes/Redis，没讲部署步骤）、
`向量数据库该选 Milvus 还是 pgvector`（语料根本没有 pgvector）。

也就是说，**我上一轮把 8 条闸门标注里 5 条改成「可答」，其中至少 3 条改错了**——
它们属于 `mentioned`（术语出现但语料没回答）。这批标注已交回人工复核。

> 目录切换与 `.env` 读取放在 `_build_default_pipeline()` 里，**不在模块级**：
> 否则测试一 `import service` 就会改掉整个进程的 CWD，属于隐蔽的全局副作用。

---

## 已修复的缺陷（v1.0.1）

| # | 问题 | 修复 |
|---|---|---|
| 1 | BM25 用 LangChain 默认分词 `text.split()`，中文整句变**一个** token，「混合检索」实际退化成单路向量检索 | 新增 `rag_core/tokenizer.py`：装了 jieba 用 jieba，没装退化为字符二元组；`BM25Retriever` 传入 `preprocess_func` |
| 2 | `context_max_length=2000`（字符）遇到超预算文档直接 `break`，`TOP_K=3` 常常只喂进去 2 篇 | 改为 token 预算 `CONTEXT_MAX_TOKENS`（默认 6000），超预算的文档**截断**而不是丢弃 |
| 3 | 数据指纹不含分块配置，改了 `splitter_headers` 会拿旧索引配新 chunk，静默错位 | 指纹纳入分块配置 + 归一化开关；指纹缺失/损坏时明确告警 |
| 4 | 元数据过滤是「检索后再筛」（先取 `top_k*3` 再过滤），条件一窄就返回空 | 下推到 FAISS 原生 `filter=`（并放大 `fetch_k`）；BM25 侧多取再筛；过滤无结果时逐级放宽并告警 |
| 5 | `route` 不影响 `top_k`，「推荐几个素菜」只给 3 条；`list` 复用通用 prompt | `list` 意图 `top_k` 提到 10，新增 `LIST_TEMPLATE`（只输出菜名列表、不许编造） |
| 6 | `.gitignore` / `.env.example` 缺失，而 `.env` 里有真密钥 | 补齐两个文件，`.env` 已进 `.gitignore` |
| 7 | `data/cook/dishes/template/` 被当真实菜谱索引 | 移到 `data/_excluded/template/` |
| 8 | `config.py` 的 `file_glob` 读不到环境变量；数据目录为空不报错 | 支持 `FILE_GLOB`；目录里没有匹配文件时直接报错 |
| 9 | `rag_core/__init__.py` 的 `__all__` 里有从未定义的 `DEFAULT_CONFIG`；包根 import 会拖进 torch/faiss | 移出 `__all__`；`BasicRAGPipeline` 改延迟导入 |
| 10 | `requirements.txt` 写 `langchain>=0.3` 无上界，会装到 1.x，而 1.x 已把 `BM25Retriever` 移走 | 约束到 `<1.0` |
| 11 | `splitter.py` 用 `uuid.uuid4()` 生成 `chunk_id`，每次运行都不同，RRF 两路去重永远对不上（重叠率 0%），且不可复现 | 改为 `md5(parent_id:index)`；`chunk_id` 纳入数据指纹，旧索引自动重建 |
| 12 | `_rrf_rerank` 里 BM25 侧**无条件覆盖**同一个 chunk，把向量侧对象上的 `vector_cos` 一起丢掉 —— 下游相关性闸门读成 `0.00`，**Agent 对几乎所有问题都回「信息不足」**，而 `--check` 只验工具调用、验不出来 | 改为 `doc_map.setdefault(key, doc)`（向量先插入 → 带余弦的对象优先保留）；两条回归测试钉住 |

## 跑测试

```bash
python -m pytest tests/ -v      # 需要 pytest
python tests/test_fixes.py      # 不装 pytest 也能跑
```

装了完整依赖时全部用例都会跑（130 项）；只装了部分依赖时，需要 faiss 的用例会自动 skip。

### 排查：启动时报 `httpx.InvalidURL: Invalid port: ':1]'`

环境里 `NO_PROXY` 含 `[::1]`。`bootstrap.prepare()` 已经会自动忽略它，所以最新代码能自愈；
想从根上修就改系统/启动器里的变量：

```powershell
cmd /c set | findstr /i proxy      # PowerShell 的 Env: 遇到重名键会报错，用 cmd 看
# 期望看到：NO_PROXY=localhost,127.0.0.1,::1
```

## 已知限制

- `estimate_tokens` 是字符/词计数估算，不是真 tokenizer（够算上下文预算，不用于计费）。
- BM25 侧没有原生 metadata 过滤，靠「多取再筛」（`candidate_k × 4`）。数据量上到十万级再考虑按过滤条件建缓存。
- 过滤条件被放宽时只写日志，回答里不会提示用户「条件已放宽」。
- 源文件删除/修改没有增量删除能力，靠指纹全量重建兜底。
- `NO_PROXY` 里如果有 `[::1]` 这种带方括号的 IPv6 写法，httpx 在构造 Client 时会抛
  `InvalidURL: Invalid port`，**openai SDK 和 huggingface_hub 都会中招**，一个环境变量就能让整个启动失败。
  `bootstrap.prepare()` 会自动忽略这一项（`::1` 仍保留，代理行为不变）。想从根上修就把 `NO_PROXY` 写成
  `localhost,127.0.0.1,::1`。
- 嵌入模型已缓存时**不会联网**（见 `indexer.setup_embeddings`）；只有本地缓存不完整才会回退到下载。
  所以 `HF_HUB_OFFLINE=1` 不是必需的，而且实测它**修不了**上面那个 `NO_PROXY` 问题。
- **重排序模型不参与「语料里有没有」的判断**，只影响排序。它对「术语出现但语料没回答」那类
  问题会给出高分（实测 0.96），而那个高分是**正确的**——见「闸门为什么注定不好做」。
- **重排在 CPU 上不可用**（整池约 24 秒/次检索）。实测平台是 i5-12500H + RTX 3050 Ti 4GB，
  装 CUDA 版 torch 后降到 1.24 秒——**差 17 倍**。没有可用 GPU 就别开重排
  （`RERANK_ENABLED=false`），纯 RRF 是 0.01 秒。
- 重排让峰值内存从 641 MB 涨到约 **2.2 GB**（模型 1.1 GB + torch 激活）。16GB 内存够用。
- `mentioned` 类（术语出现、答案不在语料）目前**没有自动转人工机制**。相似度阈值原理上做不了。
- 闸门标注现有 **31 条**（`absent` 17 + `mentioned` 14）：「零误伤下重排拦下 88.2%」基于 **n=17**，
  `mentioned` 类的 AUC 基于 **n=14**；而且这个间隙会随重排池深变化（整池时 +0.142，截到 50 时缩到 +0.069）。
  **别拿现在的分数去定线上阈值。**
- 重排按 chunk 打分，没有利用「同一篇文章的多个 chunk」之间的信息；按篇去重实测只省 25%
  （融合池里 chunk 并没有聚在同一篇上），所以没做。
- **`AGENT-100-DAYS` 语料是中文课程笔记**，所以重排模型选了中文友好的 `bge-reranker-base`。
  换语料时记得一起换 `RERANK_MODEL`。

---

## 版本

- basic_rag **v1.5.0**
  - **人工基准**（`eval/build_human_review.py` → `eval/human_review.html` → `score_human_review.py`）：
    44 条逐条标，与裁判一致率 **97.4%**（37/38；6 条「拿不准」不计入分母），偏严 1 条、偏宽 0 条。
    结论写进 README 时**必须带上空档**：样本里一条「不忠实」都没有，偏宽方向无样本
  - **校准探针的种子加两道闸**（第 ⑦ 个坑）：默认报告换成修好之后的全量报告；不指定 `--index`
    时自动挑一条存有 `faithful=True` 的答案，指定了也要求它确实忠实；再按 token 预算拒绝
    「记的不是实收正文」的旧报告（实测旧报告 3/6 超预算、新报告 0/78）。退出码分开报：
    1 = 抓不住注入 / 2 = 种子有问题（**先别改裁判**）/ 3 = 裁判没给出结论
  - 生成层按 105 篇语料重跑：引对率 **88.6%**（44 条可答）、闸门声明缺口 30/31、
    全量忠实度 **97.7%**（43/44，0 次解析失败）—— 此前 README 里「全量忠实度仍未测量」
    的旧结论**全部作废**，`agent` 臂仍待重跑
  - 零 LLM 的评测不再要求 API key（`bootstrap.allow_llm_free_run`），
    「没配 key」与「配置写错」不再混为一谈
- basic_rag **v1.4.0**
  - **语料扩容**：99 → **105 篇**（新增 `week16/` 六篇：RAG 评估 RAGAS / ColBERT 后期交互 /
    SFT 与 RLHF / 幻觉检测 / 语义缓存 / 思考与补充资料）
  - 标注随之调整：可答 38 → **44**，`absent` 22 → 17，`mentioned` 15 → 14
  - **修掉一条假闸门**：`HyDE 和多查询改写该怎么选？` 其实 week5/34 有对照表，
    原 probe 选了「查询改写」而文中叫「问题改写」——**probe 校验字符串，不校验知识**
  - 检索层复测：总体 hit@5 自适应 **86.4%** / 重排 **90.9%**（44 条；与旧的 38 条不可直接比）；
    零误伤下重排拦下 **88.2%** 域外问题，`mentioned` 类仍是余弦更会分（AUC 0.89 vs 0.78）
  - 修掉门禁一个真 bug：生成层报告缺失时 `--rescore` 会让整个门禁假 FAIL，现在**跳过并说明**
  - **已知未做**：`agent` 臂的生成层未按新语料重跑（数字仍是 99 篇语料时的）；生成层基线待补
- basic_rag **v1.3.0**
  - **回归门禁**（`eval/run_all.py`）：三层指标（检索 / Agent 对照 / 生成层）对比基线，
    有退化就退出码 1；带**口径保护**（标注指纹 + 生效配置变了先提示，不硬比）
  - **裁判第一次被证明有区分力**（第 5 版）：参照物从 3,000 字放开到 24,000 字并按块比例分配；
    `--shuffle-context` 打乱对照 + `eval/judge_probe.py` 注入假细节探针，两个方向都验过
  - **口径只有一份**：标注类别收进 `scoring.py`（`bucket_of`），删掉 Agent 对照层里
    「`partial` 五个臂恒 0.0%」那行假指标（它永远不动，在门禁里是废行）
  - `LLMReply.finish_reason`：空内容时能说出原因（推理把 token 吃光 = `length`），不再靠猜
  - **已知未做**：全量忠实度**仍未测量**（现有全量报告没有 `context_texts`，要重跑生成）；
    —— v1.5.0 已测出 **97.7%（43/44）**，这条只描述当时状态；
    Agent 臂 `partial`「明说不完全」66.7%，低于单轮的 100%
- basic_rag **v1.2.0**
  - **生成层评测**（`eval/run_gen_eval.py` + `eval/scoring.py`）：引对率、弃答、成本真数；
    `--rescore` / `--judge-only` 让改判分不必重新生成
  - `AgentResult.contexts` / `query_with_sources()`：把「模型实际读到了什么」记下来，
    作为判忠实的参照（此前只有 `citations`，裁判只能拿 gold 当参照 → 假结论「忠实度 5.9%」）
  - **常驻服务**（`service.py`，FastAPI）：冷启动只付一次；启动时预热重排模型
  - **标注四分类**：可答 / `absent` / `mentioned` / `partial`（人工判定，带依据）
  - 闸门实测扩到 40 条 → 推翻了此前两个基于 n=3 的结论
- basic_rag **v1.1.0**
  - 新增 cross-encoder 重排序（`rag_core/reranker.py`）：hit@5 84.2% → **89.5%**，
    multi 覆盖率 46.7% → 53.3%
  - 修 `_rrf_rerank` 丢 `vector_cos` 的 bug（Agent 路径曾整体失效）
  - 评测标注改为三类：`absent` / `mentioned`（术语出现但答不了）/ 可答 —— 原 8 条「语料外」里 5 条其实可答
  - 重排池深与 `top_k` 解耦（`RERANK_CANDIDATES=0` 不截断），消除评测口径不一致
- 默认 skill: recipe
- 包名: `rag_core`
- 支持数据指纹自动重建
