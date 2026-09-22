"""自带示例领域（demo）—— 也是「零配置接新领域」的活例子。

这个文件就是接一个新领域需要的**全部代码**：一个名字。

- 元数据：默认按路径约定（`article_id` = `coffee/手冲参数.md`，`title` = 文件名）
- 生成 prompt：核心层自带的 `basic`（`rag_core/prompts.py`）
- 分块：默认 `#` / `##` / `###`
- 领域身份：中性默认，`id_hint` 取自真实语料的第一篇

配套：`examples/demo_corpus/`（两篇自写示例文档）、`.env.demo`、`eval/seeds.demo.jsonl`。
跑法：`EVAL_SKILL=demo python eval/run_eval.py`（零 LLM 成本）。
"""
from ...skill import RAGSkill


def build_demo_skill() -> RAGSkill:
    return RAGSkill(name="demo")


__all__ = ["build_demo_skill"]
