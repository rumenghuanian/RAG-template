"""通用 RAG 入口示例（食谱 Skill）"""
import logging
import sys
from pathlib import Path
import os
os.chdir(Path(__file__).parent)

from dotenv import load_dotenv

# 确保可以 import basic_rag
sys.path.append(str(Path(__file__).parent))

import bootstrap  # noqa: E402

# 先加载 .env，再 import config（因为 config 在实例化时读环境变量）
load_dotenv()

from rag_core  import BasicRAGPipeline, RAGConfig
from rag_core.skills import build_recipe_skill

# 必须在构造任何 HTTP 客户端之前跑：控制台编码 + NO_PROXY 归一化
bootstrap.prepare()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    # 1. 从环境变量构造配置
    config = RAGConfig()
    config.validate()
    logger.info(f"LLM: {config.llm_model} @ {config.llm_base_url}")
    logger.info(f"数据目录: {config.data_path}")

    # 2. 构造 skill
    skill = build_recipe_skill()

    # 3. 构建 pipeline（不再需要传 api_key）
    pipeline = BasicRAGPipeline(config, skill).build()

    print("=" * 60)
    print(f"RAG Skill 示例 - {skill.name}")
    print("=" * 60)

    while True:
        try:
            q = input("\n您的问题 (输入 q 退出): ").strip()
            if q.lower() in ("q", "quit", "exit", ""):
                break

            stream = input("流式输出? (y/n, 默认 y): ").strip().lower() != "n"
            print("\n回答:")
            if stream:
                for chunk in pipeline.query(q, stream=True):
                    print(chunk, end="", flush=True)
                print()
            else:
                print(pipeline.query(q, stream=False))
        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.exception("处理出错")
            print(f"错误: {e}")


if __name__ == "__main__":
    main()