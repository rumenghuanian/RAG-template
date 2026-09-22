from .demo import build_demo_skill
from .notes import build_notes_skill
from .recipe import build_recipe_skill

# 领域注册表：换领域时加一行，配置里的 EVAL_SKILL 就能选到它
# （demo 是自带示例语料，让人 clone 下来就能跑出数字；它只写了一个 name，全走默认值）
SKILLS = {
    "notes": build_notes_skill,
    "recipe": build_recipe_skill,
    "demo": build_demo_skill,
}

__all__ = ["SKILLS", "build_notes_skill", "build_recipe_skill", "build_demo_skill"]
