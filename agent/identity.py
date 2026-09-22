"""领域身份（agent_identity）的唯一出口。

为什么单独一个模块：SYSTEM_PROMPT 在 `agent/loop.py`，工具 schema 在 `agent/tools.py`，
两处都要用同一份身份。以前各写各的默认值，于是默认值里带着 notes 的味道
（`browse_arg="week"`、示例 id `week5/33.重排序.md`）—— 新领域忘了声明身份，模型就会
被告知"这是课程笔记库"，并照着一个不存在的 id 格式**编造 id**（实测踩过）。
"""
from typing import Optional

# 全中性默认值：不含任何领域的字段名或 id 示例
DEFAULT_IDENTITY = {
    "role": "知识库助手",
    "corpus": "知识库",
    "id_name": "article_id",
    "id_hint": "",       # 由 resolve_identity() 用**语料里真实的** id 填上
    "browse_arg": "",    # 空 = 该领域没有筛选维度，工具里就不出现这个参数
    "browse_desc": "",
    "empty_phrase": "知识库里没有找到相关内容",
}

# 语料为空（还没建索引/检索不到）时的兜底措辞。
# 故意写成一句话而不是假 id —— 之前写 `week5/33.重排序.md` 这种示例，
# 换个语料就是在教模型编造。
ID_HINT_FALLBACK = "先用 list_articles 看一篇的真实 id"


def resolve_identity(skill=None, first_article_id: Optional[str] = None) -> dict:
    """身份解析：中性默认值 → `skill.agent_identity` 覆盖 → 用真实 id 补 `id_hint`。

    **`id_hint` 必须来自真实语料**：它是给模型看的示例，拍脑袋写的示例 = 教它编造。
    调用方拿得到索引时把第一篇的 `article_id` 传进来即可。

    返回的 dict **保留全部键**（空串也保留）—— `SYSTEM_PROMPT_TEMPLATE.format(**identity)`
    要求占位符齐全；"该领域有没有筛选维度"由调用方判断空串，而不是靠键是否存在。
    """
    identity = dict(DEFAULT_IDENTITY)
    identity.update(getattr(skill, "agent_identity", None) or {})
    if not str(identity.get("id_hint") or "").strip():
        identity["id_hint"] = first_article_id or ID_HINT_FALLBACK
    return identity
