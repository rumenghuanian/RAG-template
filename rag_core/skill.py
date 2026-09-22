"""RAG Skill 协议 - 领域技能需要实现的钩子"""
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from langchain_core.documents import Document


@dataclass
class RAGSkill:
    """
    一个 RAG Skill 需要提供的所有领域相关钩子。
    全部有默认值，因此最小 skill 只需要 name 即可。
    """
    name: str = "default"

    # ① 元数据提取：Document -> dict，用于分类/难度等过滤字段
    metadata_extractor: Optional[Callable[[Document], Dict[str, Any]]] = None

    # ② 分块策略：Markdown 标题层级
    splitter_headers: List[tuple] = field(
        default_factory=lambda: [("#", "h1"), ("##", "h2"), ("###", "h3")]
    )
    strip_headers: bool = False

    # ③ Prompt 注册表：mode -> ChatPromptTemplate
    prompt_registry: Dict[str, Any] = field(default_factory=dict)

    # ④ 查询策略：rewrite / route / extract_filters
    query_strategy: Optional[Any] = None   # 需实现 QueryStrategy 协议

    # ⑤ **通用层依赖的规范字段**：每个 skill 都必须产出，且值非空、全局唯一。
    #    为什么用固定名字而不是"skill 自己声明 id 字段"：引擎/Agent/评测里有 20 多处
    #    直接读 `article_id`，让它们各自去问 skill"你的 id 字段叫什么"，改动面更大、
    #    更容易漏。所以约定：**skill 负责把自己领域的 id 映射到规范名**。
    #    漏了会在加载时直接报错，而不是像以前那样静默变成 None
    #    （实测：recipe 没产出 article_id → 检索 5 篇塌成 [None]、read_article(None) 返回错文章）。
    required_metadata: tuple = ("article_id", "title")

    # ⑥ Agent 的领域身份：用来拼 SYSTEM_PROMPT 与工具描述，
    #    否则换领域后模型仍被告知"这是课程笔记库"、示例 id 也还是 notes 的格式。
    #    可用的键见 agent/loop.py 的 DEFAULT_IDENTITY。
    agent_identity: Dict[str, str] = field(default_factory=dict)