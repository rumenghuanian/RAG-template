"""会话记忆：让「它呢？」这类指代问题能被补全后再去检索。"""
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

CONDENSE_PROMPT = """把用户的新问题改写成不依赖上文、能独立用于检索的问题。
只输出改写后的问题，不要解释，不要加引号。新问题本身已完整就原样输出。

对话上文：
{history}

新问题：{question}
改写后的问题："""


class ConversationMemory:
    def __init__(self, max_turns: int = 6):
        self.max_turns = max_turns
        self.turns: List[Dict[str, str]] = []

    def add_user(self, text: str) -> None:
        self.turns.append({"role": "user", "content": text})
        self._trim()

    def add_assistant(self, text: str) -> None:
        self.turns.append({"role": "assistant", "content": text})
        self._trim()

    def messages(self) -> List[Dict[str, str]]:
        return list(self.turns)

    def clear(self) -> None:
        self.turns.clear()

    def condense(self, question: str, llm=None) -> str:
        """把指代问题补全成独立问题。没有上文或没有 llm 时原样返回。"""
        if llm is None or not self.turns:
            return question
        history = "\n".join(
            f"{t['role']}: {t['content'][:200]}" for t in self.turns[-4:]
        )
        messages = [
            {
                "role": "user",
                "content": CONDENSE_PROMPT.format(history=history, question=question),
            }
        ]
        try:
            reply = llm.chat(messages)
        except Exception as e:  # noqa: BLE001 - 改写失败不该让整轮问答挂掉
            logger.warning("指代消解失败，回退用原问题：%s", e)
            return question
        rewritten = (reply.content or "").strip()
        return rewritten or question

    def _trim(self) -> None:
        keep = self.max_turns * 2
        if len(self.turns) > keep:
            self.turns = self.turns[-keep:]
