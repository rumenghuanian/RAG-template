# 给这套 RAG 加一个新领域（Skill）

**目标**：把一批新的 `.md` 语料接进同一套引擎，不改 `rag_core/` 一行代码。
按本文走完，检索层评测与 Agent 对照评测都能跑出数字（**这两层不调用 LLM，零成本**）。

> 本文里的每一条"别踩"都是从真实踩坑记录里提炼的（见 `README.md` 的「评测本身也是被测对象」一节）。

---

## 0. 一个领域 skill 到底要提供什么

`RAGSkill` 有 6 个钩子，**只有前两个是必须的**：

| 钩子 | 必须? | 作用 |
|---|---|---|
| `metadata_extractor` | ✅ | `Document -> dict`。**必须产出规范字段 `article_id` / `title`** |
| `prompt_registry` | ✅ | `mode -> ChatPromptTemplate`，**至少要有一个 `basic`** |
| `splitter_headers` | 建议 | Markdown 分块用哪些标题层级 |
| `agent_identity` | 建议 | Agent 的角色/语料名/id 示例（决定 SYSTEM_PROMPT 与工具描述） |
| `query_strategy` | 可选 | 路由 / 改写 / 过滤条件提取；不提供就退化成"直接检索" |
| `required_metadata` | 有默认值 | 规范字段清单，默认 `("article_id", "title")` |

---

## 1. 最小可用 skill（复制就能改）

```
rag_core/skills/my_domain/
├── __init__.py      # 组装 RAGSkill
├── metadata.py      # 元数据提取（规范字段在这里产出）
└── prompts.py       # 生成 prompt
```

**`metadata.py`**

```python
from pathlib import Path
from langchain_core.documents import Document


def my_metadata_extractor(doc: Document) -> dict:
    src = Path(doc.metadata.get("source", ""))
    title = src.stem                       # 人能读的名字
    return {
        # 规范字段：通用层（检索去重 / 引用匹配 / 评测 / 读单篇）只认这两个名字。
        # 必须**非空**且**全局唯一**。
        "article_id": "/".join(src.parts[-2:]),
        "title": title,
        # 其余字段随便加，会进元数据、可用于过滤
        "category": src.parent.name,
    }
```

**`prompts.py`**

```python
from langchain_core.prompts import ChatPromptTemplate

TEMPLATE = """你是一个知识库助手。只根据下面的资料回答问题。

用户问题: {question}

资料:
{context}

要求：
- 只使用上面资料里出现的内容
- 回答里标明出处（标题）
- 资料里没讲到的，直接说「资料里没有找到」，不要猜

回答:"""


def build_my_prompts() -> dict:
    # 至少要有 "basic"：路由名不在 registry 里时会回落到它。
    # 变量名必须是 {question} 和 {context}。
    return {"basic": ChatPromptTemplate.from_template(TEMPLATE)}
```

**`__init__.py`**

```python
from ...skill import RAGSkill
from .metadata import my_metadata_extractor
from .prompts import build_my_prompts


def build_my_skill() -> RAGSkill:
    return RAGSkill(
        name="my_domain",
        metadata_extractor=my_metadata_extractor,
        splitter_headers=[("#", "h1"), ("##", "h2"), ("###", "h3")],
        prompt_registry=build_my_prompts(),
        query_strategy=None,          # 需要路由/改写时再给（见第 3 节）
        agent_identity={
            "role": "资料助手",
            "corpus": "内部资料库（约 200 篇）",
            "id_hint": "handbook/报销制度.md",
            "browse_arg": "category",
            "browse_field": "category",
            "browse_desc": "（可按分类筛选）",
            "empty_phrase": "资料库里没有找到相关内容",
        },
    )
```

## 2. 注册 + 配置

```python
# rag_core/skills/__init__.py
from .my_domain import build_my_skill

SKILLS = {
    "notes": build_notes_skill,
    "recipe": build_recipe_skill,
    "my_domain": build_my_skill,        # ← 加这一行
}
```

```ini
# .env.my_domain  —— 每个 skill 一份，env / 索引 / 标注 / 基线全部按 skill 隔离
DATA_PATH=./data/my_domain
FILE_GLOB=**/*.md
INDEX_SAVE_PATH=./vector_index/my_domain
EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
EMBEDDING_DEVICE=cpu
LLM_MODEL=deepseek-flash
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_API_KEY=
TOP_K=5
# CPU 上整池重排约 24 秒/次检索，实际不可用；要重排请在 GPU 机器上打开并设 RERANK_DEVICE=cuda
RERANK_ENABLED=false
```

## 3. 可选：路由 / 改写 / 过滤

只有 `query_strategy` 需要一点结构，协议是三个方法：

```python
class MyStrategy:
    def __init__(self):
        self.llm = None

    def set_llm(self, llm):        # pipeline.build() 会自动调用（若存在该方法）
        self.llm = llm

    def route(self, query: str) -> str:      # 返回值若命中 prompt_registry 的 key 就直接当 mode 用
        return "general"

    def rewrite(self, query: str) -> str:    # 只有 ENABLE_REWRITE=true 时才会被调用
        return query

    def extract_filters(self, query: str) -> dict:   # 传给向量库做 metadata 过滤
        return {}
```

**没有 llm 时必须能降级**（返回 `general` / 原样 / `{}`）—— 检索层评测会在没有 API key 的情况下跑它。

## 4. 五步跑起来

```powershell
# ① 校验标注并生成 golden（会做唯一匹配校验 + 闸门 probe 校验）
$env:EVAL_SKILL="my_domain"; python eval\build_golden.py

# ② 检索层评测（零 LLM 成本）：分类型 hit@5 / cover@5 / k 敏感性 / 闸门阈值扫描
python eval\run_eval.py --json

# ③ Agent 对照评测（零 LLM 成本）：等预算单轮 vs 脚本 Agent
python eval\run_agent_eval.py --json

# ④ 立基线（之后改代码可以用一条命令查退化）
python eval\run_all.py --update-baseline
python eval\run_all.py --skip-run          # 应输出「结论：PASS ✓」
```

标注写在 `eval/seeds.my_domain.jsonl`，一行一条：

```jsonl
{"kind": "easy", "question": "报销流程是什么？", "expect": ["报销制度"], "note": ""}
{"kind": "rare_token", "question": "ABAC 怎么配？", "expect": ["权限模型"], "note": ""}
{"kind": "absent", "question": "怎么用 COBOL 写银行核心系统？", "expect": [], "probe": {"absent": ["COBOL"]}, "note": "far"}
```

- `expect` 写 **article_id 或标题的唯一子串**，脚本会去语料里解析；命中 0 篇或多篇都会直接报错。
- `absent` / `mentioned` 是**闸门类**：正确行为是「声明没有」，不是答对。要写 `probe`，
  脚本会校验 `absent` 的词在语料里真的 0 命中。
- `partial` 是**人工判定类**（语料有相关内容但不完全回答），要求写 `note` 记录依据。

---

## 5. 别踩这些（都是真实踩过的）

| # | 坑 | 后果 |
|---|---|---|
| 1 | `metadata_extractor` 不产出 `article_id` | 检索去重塌成一条、引用匹配全失效、`read_article(None)` **返回错文档**。现在**加载期就报错**拦住 |
| 2 | `article_id` 重复 | 同上。加载期会报出是哪两个文件撞了 |
| 3 | 改了 `metadata_extractor` 却没重建索引 | 向量路仍返回**旧元数据**（索引 docstore 里存的），而 BM25 路用新文档 → 两条路不一致。指纹已含领域元数据，会自动重建 |
| 4 | 领域文案写死在通用层 | 模型被告知"这是课程笔记库"、照 `week5/33…` 编造不存在的 id。要用 `agent_identity` |
| 5 | `list_articles` 参数名与 skill 不一致 | schema 让模型传 `category`、函数只认 `week` → TypeError。用 `browse_arg` / `browse_field` |
| 6 | 闸门标注只写了 `absent`、没有 `mentioned` | 以前会 `ZeroDivisionError` 崩掉整个评测（空分组），现在已跳过空组 |
| 7 | 标注类别口径各脚本各抄一份 | 同一个 bug 只在一处被修好 → 报告里出现恒为 0 的假指标。口径只有一份，在 `eval/scoring.py` |
| 8 | 换语料/改标注后直接和旧基线比大小 | 题集变了，指标本来就会变。`run_all.py` 的标注指纹保护会拒绝硬比并提示重新立基线 |
| 9 | CPU 上开重排 | 整池重排约 24 秒/次检索。`RERANK_ENABLED=false` |
| 10 | 拿 gold 当参照判「忠实度」 | 模型实际读 2~5 篇，引用别篇会被判成编造。参照物要用 `query_with_sources()` 返回的**实际喂进 prompt 的正文** |

---

## 6. 验收清单

- [ ] `python -c "import rag_core.skills as s; s.SKILLS['my_domain']()"` 不报错
- [ ] 加载语料不报契约错（说明 `article_id` / `title` 齐全且唯一）
- [ ] `build_golden.py` 输出「闸门类 probe 全部通过校验」
- [ ] `run_eval.py` 里 **trivial 问题的 hit@5 应该是 100%**；如果全是 0%，先怀疑 id 契约而不是检索质量
- [ ] `run_agent_eval.py` 能列出五臂对比；闸门题应该触发 `needs_human`
- [ ] `run_all.py --skip-run` 输出 PASS
- [ ] `pytest tests/ -q` 全绿（契约测试不依赖你的语料）

---

## 7. 现状与边界（别把没验过的说成验过的）

- **已验证**：`notes`（105 篇课程笔记，44 条完整评测标注）与 `recipe`（322 篇菜谱）两个领域，
  同一套引擎跑通；recipe 侧用的是 6 条**诊断集**（只够验证链路，不足以评价效果）。
- **未验证**：第三个领域；多语言语料；`query_strategy` 在两个领域以上的通用性。
- **已知取舍**：语料不进仓库（第三方内容），所以 clone 下来需要自备数据；
  索引是全量重建（没有单篇增删）；重排在 CPU 上不可用。
