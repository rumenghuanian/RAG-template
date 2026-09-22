"""食谱领域元数据提取"""
import re
from pathlib import Path

from langchain_core.documents import Document

CATEGORY_MAPPING = {
    "meat_dish": "荤菜",
    "vegetable_dish": "素菜",
    "soup": "汤品",
    "dessert": "甜品",
    "breakfast": "早餐",
    "staple": "主食",
    "aquatic": "水产",
    "condiment": "调料",
    "drink": "饮品",
    "semi-finished": "半成品",
}
CATEGORY_LABELS = list(set(CATEGORY_MAPPING.values()))
DIFFICULTY_LABELS = ["非常简单", "简单", "中等", "困难", "非常困难"]

_DIFFICULTY_MAP = {5: "非常困难", 4: "困难", 3: "中等", 2: "简单", 1: "非常简单"}


def recipe_metadata_extractor(doc: Document) -> dict:
    src = Path(doc.metadata.get("source", ""))
    path_parts = src.parts

    # 分类
    category = "其他"
    category_key = ""
    for key, value in CATEGORY_MAPPING.items():
        if key in path_parts:
            category = value
            category_key = key
            break

    # 难度（精确匹配连续星号）
    match = re.search(r"★+", doc.page_content)
    difficulty = _DIFFICULTY_MAP.get(len(match.group()), "未知") if match else "未知"

    # 菜名取文件名；**id 取路径最后两段**（与 notes 的 `week5/29.标题.md` 同一套约定）。
    #
    # 为什么不用「类别 + 菜名」这种更整齐的 id：语料里同一道菜会以两种布局同时存在
    # （实测 `soup/陈皮排骨汤.md` 与 `soup/陈皮排骨汤/陈皮排骨汤.md` 各一份），
    # 而且菜品子目录下会有多个变体（`红烧肉/南派红烧肉.md` + `简易红烧肉.md`）。
    # 加载期唯一性校验把这些都抓了出来；改用路径才能既唯一又可读，
    # 并且**不去动用户的语料**（重复内容是数据问题，交给人决定，不偷偷删）。
    title = src.stem

    return {
        "category": category,
        "difficulty": difficulty,
        # 规范字段：通用层（检索/引用/评测/读单篇）只认这两个名字。
        # 见 RAGSkill.required_metadata —— 缺了或重了都会在加载时直接报错。
        "article_id": "/".join(src.parts[-2:]),
        "title": title,
        "dish_name": title,
    }