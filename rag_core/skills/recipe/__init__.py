from .metadata import recipe_metadata_extractor, CATEGORY_LABELS, DIFFICULTY_LABELS
from .prompts import build_recipe_prompts
from .strategy import RecipeQueryStrategy

from ...skill import RAGSkill


def build_recipe_skill() -> RAGSkill:
    """构造食谱 Skill"""
    return RAGSkill(
        name="recipe",
        metadata_extractor=recipe_metadata_extractor,
        splitter_headers=[
            ("#", "主标题"),
            ("##", "二级标题"),
            ("###", "三级标题"),
        ],
        strip_headers=False,
        prompt_registry=build_recipe_prompts(),
        query_strategy=RecipeQueryStrategy(),
        agent_identity={
            "role": "菜谱助手",
            "corpus": "中文菜谱库（322 道菜，按荤素/汤品/主食等分类）",
            "id_hint": "meat_dish/宫保鸡丁",
            "browse_arg": "category",
            "browse_field": "category",
            "browse_desc": "（可按 荤菜/素菜/汤品/主食 等分类筛选）",
            "empty_phrase": "菜谱库里没有找到相关做法",
        },
    )


__all__ = [
    "build_recipe_skill",
    "recipe_metadata_extractor",
    "CATEGORY_LABELS",
    "DIFFICULTY_LABELS",
]