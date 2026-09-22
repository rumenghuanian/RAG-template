"""Agent-100-Days 语料的元数据提取。

语料结构：`Agent-100-Days/week{N}/{NN}.{标题}.md`
"""
import re
from pathlib import Path

from langchain_core.documents import Document

# week5/29.RAG是怎么工作的.md
_PATH_RE = re.compile(r"/(week\d+)/(\d+)\.[^/]*\.md$")
_H1_RE = re.compile(r"^#\s+(.+)$", re.MULTILINE)
_NO_PREFIX_RE = re.compile(r"^\d+[.\s]\s*")


def notes_metadata_extractor(doc: Document) -> dict:
    src = Path(doc.metadata.get("source", ""))
    posix = src.as_posix()

    week = article_no = None
    m = _PATH_RE.search(posix)
    if m:
        week = int(m.group(1)[len("week"):])
        article_no = int(m.group(2))

    h1 = _H1_RE.search(doc.page_content)
    title = _NO_PREFIX_RE.sub("", h1.group(1)).strip() if h1 else ""
    if not title:
        title = src.stem

    return {
        # 稳定且人能读的 id，检索结果里给模型看、read_article 拿它回查
        "article_id": "/".join(src.parts[-2:]),
        "title": title,
        "week": week,
        "article_no": article_no,
        "is_thinking": "补充学习资料" in title,
    }
