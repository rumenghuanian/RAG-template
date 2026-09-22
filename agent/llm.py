"""LLM 客户端：只依赖 openai SDK，接口刻意做小，方便测试时整体替换。"""
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ToolCall:
    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    id: str = ""


@dataclass
class LLMReply:
    content: Optional[str] = None
    tool_calls: List[ToolCall] = field(default_factory=list)
    # 为什么需要它：内容为空时**必须能说出原因**。推理模型把 max_tokens 全用在
    # 思考上就会返回空 content，`finish_reason="length"` 是唯一线索。
    # 之前空内容只能靠猜，白查了两轮（见 README 判分器踩过的坑）。
    finish_reason: str = ""

    @property
    def wants_tool(self) -> bool:
        return bool(self.tool_calls)


class OpenAICompatLLM:
    """任何 OpenAI 兼容端点都能用（DeepSeek / Moonshot / 通义 / vLLM ...）。

    注意：Agent 路径依赖 **function calling**，模型不支持工具调用的话
    tool_calls 会一直是空的，agent 只会退化成单轮问答。
    """

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ):
        from openai import OpenAI

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        # 累计用量：评测要报**真实**成本，不能靠估
        self.usage: Dict[str, int] = {
            "calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }
        logger.info("LLM 客户端就绪: %s @ %s", model, base_url)

    def chat(
        self, messages: List[Dict[str, Any]], tools: Optional[List[dict]] = None
    ) -> LLMReply:
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if tools:
            kwargs["tools"] = tools

        resp = self.client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message
        self._record_usage(getattr(resp, "usage", None))

        calls: List[ToolCall] = []
        for c in getattr(msg, "tool_calls", None) or []:
            calls.append(
                ToolCall(
                    name=c.function.name,
                    arguments=_loads(c.function.arguments),
                    id=c.id or "",
                )
            )
        return LLMReply(
            content=msg.content,
            tool_calls=calls,
            finish_reason=getattr(resp.choices[0], "finish_reason", "") or "",
        )

    def _record_usage(self, usage) -> None:
        """有些 OpenAI 兼容端点不返回 usage，那就只记调用次数，不编造 token 数。"""
        self.usage["calls"] += 1
        if usage is None:
            return
        self.usage["prompt_tokens"] += getattr(usage, "prompt_tokens", 0) or 0
        self.usage["completion_tokens"] += getattr(usage, "completion_tokens", 0) or 0


def _loads(raw: Optional[str]) -> Dict[str, Any]:
    """模型偶尔返回坏 JSON，别让它把整个循环带崩。"""
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("工具参数不是合法 JSON: %r", raw)
        return {}
    return data if isinstance(data, dict) else {"value": data}
