"""Agent 笔记助手入口：模型自己决定何时检索、检索几次、要不要读全文。

用法：
    Copy-Item .env.notes.example .env.notes      # 然后填 LLM_API_KEY
    python main_agent.py --check                 # 先验证模型支不支持工具调用
    python main_agent.py
"""
import argparse
import logging
import os
import sys
from pathlib import Path

os.chdir(Path(__file__).parent)
sys.path.append(str(Path(__file__).parent))

import bootstrap  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

# 优先用 .env.notes（笔记语料），没有就退回 .env
_ENV_FILE = Path(".env.notes")
load_dotenv(_ENV_FILE if _ENV_FILE.exists() else ".env")

from agent import ConversationMemory, OpenAICompatLLM, RAGAgent, build_tools  # noqa: E402
from rag_core import BasicRAGPipeline, RAGConfig  # noqa: E402
from rag_core.skills.notes import build_notes_skill  # noqa: E402

# 必须在构造任何 HTTP 客户端之前跑：控制台编码 + NO_PROXY 归一化
bootstrap.prepare()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def check_tool_calling(llm, tools) -> bool:
    """花一次调用确认模型真的会返回 tool_calls —— 不支持的话 Agent 会静默退化成单轮问答。"""
    reply = llm.chat(
        [{"role": "user", "content": "用 search_notes 工具查一下「RAG 是怎么工作的」"}],
        tools=tools.schemas(),
    )
    if reply.tool_calls:
        call = reply.tool_calls[0]
        print(f"[OK] 模型支持工具调用：{call.name}({call.arguments})")
        return True
    print("[X] 模型没有返回 tool_calls，当前模型大概率不支持 function calling。")
    print(f"   它返回的内容：{(reply.content or '')[:200]!r}")
    print("   请换 deepseek-chat / qwen-plus / gpt-4o-mini 等支持工具调用的模型。")
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Agent 笔记助手")
    parser.add_argument("--check", action="store_true", help="只验证模型是否支持工具调用")
    parser.add_argument("--max-steps", type=int, default=6, help="单轮最多几次工具调用")
    args = parser.parse_args()

    config = RAGConfig()
    config.validate()
    logger.info("语料: %s (glob=%s)", config.data_path, config.file_glob)

    pipeline = BasicRAGPipeline(config, build_notes_skill()).build()
    tools = build_tools(pipeline, min_relevance=config.min_relevance)
    llm = OpenAICompatLLM(
        model=config.llm_model,
        base_url=config.llm_base_url,
        api_key=config.llm_api_key,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )

    if args.check:
        sys.exit(0 if check_tool_calling(llm, tools) else 1)

    agent = RAGAgent(llm, tools, max_steps=args.max_steps)
    memory = ConversationMemory()

    print("=" * 66)
    print(f"Agent 笔记助手 ｜ {config.llm_model} ｜ 语料 {len(pipeline.documents)} 篇")
    print("=" * 66)

    while True:
        try:
            question = input("\n你: ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if question.lower() in ("q", "quit", "exit", ""):
            break

        standalone = memory.condense(question, llm)
        if standalone != question:
            print(f"[补全为] {standalone}")

        try:
            result = agent.run(standalone, history=memory.messages())
        except Exception:
            logger.exception("Agent 执行失败")
            continue

        print(f"\nAgent: {result.answer}")
        if result.citations:
            print("\n出处：")
            print(result.sources_text())
        if result.needs_human:
            print("\n[!] 本轮没有检索到任何依据，建议转人工确认，不要直接采信。")
        print(f"\n[工具调用 {len(result.steps)} 次 ｜ 结束原因 {result.stop_reason}]")

        memory.add_user(question)
        memory.add_assistant(result.answer)


if __name__ == "__main__":
    main()
