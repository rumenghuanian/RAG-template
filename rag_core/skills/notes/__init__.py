from .metadata import notes_metadata_extractor
from .prompts import build_notes_prompts

from ...skill import RAGSkill


def build_notes_skill() -> RAGSkill:
    """构造 Agent-100-Days 课程笔记 Skill。"""
    return RAGSkill(
        name="notes",
        metadata_extractor=notes_metadata_extractor,
        splitter_headers=[
            ("#", "文章"),
            ("##", "小节"),
            ("###", "子节"),
        ],
        strip_headers=False,
        prompt_registry=build_notes_prompts(),
        # Agent 路径由模型自己决定检索什么，不需要 LLM 路由/改写
        query_strategy=None,
        # 领域身份：拼 SYSTEM_PROMPT 与工具描述用（换领域必须换掉，否则模型会被告知
        # "这是课程笔记库"、示例 id 还是 notes 格式）
        agent_identity={
            "role": "Agent 开发学习助手",
            "corpus": "课程笔记库（Agent-100-Days，16 周共 105 篇）",
            "id_hint": "week5/33.重排序.md",
            "browse_arg": "week",
            "browse_desc": "，可按周筛选",
            "empty_phrase": "笔记里没有找到相关内容",
        },
    )


__all__ = ["build_notes_skill", "notes_metadata_extractor"]
