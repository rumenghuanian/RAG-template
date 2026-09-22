"""食谱 prompt 注册表"""
from langchain_core.prompts import ChatPromptTemplate


BASIC_TEMPLATE = """你是一位专业的烹饪助手。请根据以下食谱信息回答用户的问题。

用户问题: {question}

相关食谱信息:
{context}

请提供详细、实用的回答。如果信息不足，请诚实说明。

回答:"""


STEP_BY_STEP_TEMPLATE = """你是一位专业的烹饪导师。请根据食谱信息，为用户提供详细的分步骤指导。

用户问题: {question}

相关食谱信息:
{context}

请灵活组织回答，建议包含以下部分（可根据实际内容调整）：

## 🥘 菜品介绍
[简要介绍菜品特点和难度]

## 🛒 所需食材
[列出主要食材和用量]

## 👨‍🍳 制作步骤
[详细的分步骤说明，每步包含具体操作和大概所需时间]

## 💡 制作技巧
[仅在有实用技巧时包含。如果原文的"附加内容"与烹饪无关或为空，可以省略此部分]

注意：
- 根据实际内容灵活调整结构
- 不要强行填充无关内容
- 重点突出实用性和可操作性

回答:"""


ROUTER_TEMPLATE = """根据用户的问题，将其分类为以下三种类型之一：

1. 'list' - 用户想要获取菜品列表或推荐，只需要菜名
   例如：推荐几个素菜、有什么川菜、给我3个简单的菜

2. 'detail' - 用户想要具体的制作方法或详细信息
   例如：宫保鸡丁怎么做、制作步骤、需要什么食材

3. 'general' - 其他一般性问题
   例如：什么是川菜、制作技巧、营养价值

请只返回分类结果：list、detail 或 general

用户问题: {query}

分类结果:"""


REWRITE_TEMPLATE = """你是一个智能查询分析助手。请分析用户的查询，判断是否需要重写以提高食谱搜索效果。

原始查询: {query}

分析规则：
1. 具体明确的查询（直接返回原查询）：包含具体菜品名、明确制作询问、具体烹饪技巧
2. 模糊不清的查询（需要重写）：过于宽泛、缺乏具体信息、口语化表达

重写原则：保持原意、增加烹饪术语、优先推荐简单易做的、保持简洁

示例：
- "做菜" → "简单易做的家常菜谱"
- "有饮品推荐吗" → "简单饮品制作方法"
- "宫保鸡丁怎么做" → "宫保鸡丁怎么做"（保持原查询）

请输出最终查询（如果不需要重写就返回原查询）:"""


LIST_TEMPLATE = """你是一位专业的烹饪助手。用户想要一份菜品推荐列表。

用户问题: {question}

候选菜品信息:
{context}

要求：
- 每行一条，格式：- 菜名（分类 / 难度）
- 最多 10 条，按相关度从高到低
- 只推荐上面候选里出现过的菜，不要补充候选之外的菜
- 候选不足或都不合适时直说，不要编造

回答:"""


def build_recipe_prompts() -> dict:
    return {
        "basic": ChatPromptTemplate.from_template(BASIC_TEMPLATE),
        "detail": ChatPromptTemplate.from_template(STEP_BY_STEP_TEMPLATE),
        "step_by_step": ChatPromptTemplate.from_template(STEP_BY_STEP_TEMPLATE),
        "list": ChatPromptTemplate.from_template(LIST_TEMPLATE),
    }


def build_router_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_template(ROUTER_TEMPLATE)


def build_rewrite_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_template(REWRITE_TEMPLATE)