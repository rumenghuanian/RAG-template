"""文本 → token 的零依赖实现。

为什么需要它：BM25 的默认分词是 `text.split()`，中文没有空格，
一句「宫保鸡丁怎么做」会被当成**一个** token，关键词检索实际失效，
所谓「混合检索」退化成单路向量检索。

装了 jieba 就用 jieba，没装就退化为字符二元组（中文检索的经典 baseline）。
"""
import logging
import re

# 中文（含扩展 A 区）连续串 | 拉丁词 | 数字
_PIECE_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]+|[A-Za-z][A-Za-z0-9_]*|\d+(?:\.\d+)?")

_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")
_WORD_RE = re.compile(r"[A-Za-z0-9_]+")

_jieba_log_quieted = False


def _jieba():
    """惰性导入 jieba，并压掉它自带的 DEBUG 日志。

    jieba 在 import 时把自己的 logger 置为 DEBUG 并挂上 handler，
    所以在别处提前 setLevel 没用 —— 必须等它 import 完再调它自己的 API。
    """
    global _jieba_log_quieted
    try:
        import jieba
    except ImportError:
        return None
    if not _jieba_log_quieted:
        jieba.setLogLevel(logging.WARNING)
        _jieba_log_quieted = True
    return jieba


def _meaningful(token: str) -> bool:
    """丢掉纯标点 token —— 中文分词器会把「！？」也切出来，对检索没有信号。"""
    return any(ch.isalnum() for ch in token)


def tokenize_for_bm25(text: str) -> list[str]:
    """BM25 分词：中文按二元组，英文/数字按词。"""
    jieba = _jieba()
    if jieba is not None:
        return [t for t in jieba.lcut_for_search(text) if _meaningful(t)]

    tokens: list[str] = []
    for piece in _PIECE_RE.findall(text):
        if piece[0].isascii():
            tokens.append(piece.lower())
        elif len(piece) == 1:
            tokens.append(piece)
        else:
            tokens.extend(piece[i : i + 2] for i in range(len(piece) - 1))
    return tokens


def estimate_tokens(text: str) -> int:
    """粗略 token 估算：中文 1 字 ≈ 1 token，英文/数字 1 词 ≈ 1 token。

    ponytail: 只为算上下文预算，不值得引 tiktoken/transformers；
    要精确预算时换真 tokenizer（英文会略微低估）。
    """
    return len(_CJK_RE.findall(text)) + len(_WORD_RE.findall(text))


__all__ = ["tokenize_for_bm25", "estimate_tokens"]
