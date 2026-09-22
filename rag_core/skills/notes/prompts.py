"""Agent-100-Days 语料的生成 prompt（非 Agent 路径用，用于和 Agent 做对照）。"""
from langchain_core.prompts import ChatPromptTemplate

BASIC_TEMPLATE = """你是一个 Agent 开发学习助手。请只根据下面的课程笔记回答问题。

用户问题: {question}

课程笔记:
{context}

要求：
- 只使用上面笔记里出现的内容，不要补充笔记之外的知识
- 回答里标明出处（文章标题）
- 笔记里确实没讲到的，直接说「笔记里没有讲到」，不要猜

回答:"""

DETAIL_TEMPLATE = """你是一个 Agent 开发学习助手。请根据课程笔记，给出有结构的详细讲解。

用户问题: {question}

课程笔记:
{context}

要求：
- 只使用上面笔记里出现的内容
- 用「结论 → 要点 → 例子/步骤」的结构组织
- 每一条要点后面标出出处（文章标题）
- 笔记里没讲到的部分明确说「笔记里没有讲到」

回答:"""


def build_notes_prompts() -> dict:
    return {
        "basic": ChatPromptTemplate.from_template(BASIC_TEMPLATE),
        "detail": ChatPromptTemplate.from_template(DETAIL_TEMPLATE),
    }
