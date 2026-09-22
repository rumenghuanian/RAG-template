"""核心层的**默认生成 prompt**。

为什么核心层要自带一个：`GenerationIntegrationModule.generate_stream` 在 mode 没注册时
直接抛 `KeyError`（这个行为是对的，静默套错模板更糟）。但它意味着**每个新领域都必须写一份
prompts.py**，哪怕只需要最普通的「只根据资料回答 + 标出处」—— 「把语料丢进来就能跑」
因此不成立。这里给一个能用的默认值，领域想改再覆盖（同名键优先用领域自己的）。

模板变量固定是 `{question}` / `{context}`，与 `generator.generate_stream` 里喂进去的
输入键一一对应；改这里就会改所有没写 prompts.py 的领域。
"""
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate

DEFAULT_BASIC_TEMPLATE = """你是一个知识库助手。请只根据下面的资料回答问题。

用户问题: {question}

资料:
{context}

要求：
- 只使用上面资料里出现的内容，不要补充资料之外的知识
- 回答里标明出处（文章标题）
- 资料里确实没讲到的，直接说「资料里没有讲到」，不要猜

回答:"""


def default_prompt_registry() -> dict:
    """至少保证 `basic` 存在 —— `_pick_prompt_mode` 的兜底也是它。"""
    return {"basic": ChatPromptTemplate.from_template(DEFAULT_BASIC_TEMPLATE)}


def merge_prompt_registry(skill_registry: Optional[dict]) -> dict:
    """默认模板打底，**领域注册的键优先**（同名覆盖，其余保留）。

    纯函数：不读环境、不碰全局，方便单测钉住"领域模板一定盖得住默认值"。
    """
    merged = default_prompt_registry()
    merged.update(skill_registry or {})
    return merged
