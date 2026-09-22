from .notes import build_notes_skill
from .recipe import build_recipe_skill

# 领域注册表：换领域时加一行，配置里的 EVAL_SKILL 就能选到它
SKILLS = {
    "notes": build_notes_skill,
    "recipe": build_recipe_skill,
}

__all__ = ["SKILLS", "build_notes_skill", "build_recipe_skill"]
