"""入口脚本共用的启动准备。

Windows 上有两个坑会让程序还没跑起来就崩：

1. 控制台/管道用本地编码（中文系统是 GBK），模型输出里的 emoji 会让 `print`
   抛 UnicodeEncodeError。
2. `NO_PROXY` 里如果有 `[::1]` 这种带方括号的写法，httpx 在构造 Client 时抛
   `InvalidURL: Invalid port`。huggingface_hub 和 openai SDK 都会中招，
   于是整个启动被一个环境变量打断。

还有一个**不崩但更危险**的坑（env 文件选择），见 `env_file_for`。

只用标准库，不依赖任何项目模块。
"""
import logging
import os
import sys

logger = logging.getLogger(__name__)


def env_file_for(skill: str, root: str = ".") -> str:
    """按 skill 选 env 文件：`.env.<skill>` 存在就用它，否则退回 `.env`。

    **为什么需要这个**：评测脚本原先写死 `load_dotenv(".env.notes" if exists else ".env")`，
    于是 `EVAL_SKILL=recipe` 只换掉了 prompt 和元数据提取器，**语料根本没换、而且不报错** ——
    实测它把 99 篇 Agent 笔记当菜谱载进来，还打上 `category=其他 / difficulty=未知` 的假标签，
    整轮评测静默变成垃圾。README 里「换语料只改 EVAL_SKILL」这个承诺当时是假的。

    现在约定：`notes` → `.env.notes`，`recipe` → `.env`（或 `.env.recipe`）。
    """
    candidate = os.path.join(root, f".env.{skill}")
    if os.path.exists(candidate):
        return candidate
    return os.path.join(root, ".env")


def artifact_path(name: str, skill: str, ext: str, root: str = ".") -> str:
    """按 skill 给评测产物定路径，**避免换个 skill 就把上一套报告覆盖掉**。

    规则：`notes` 用旧名字（`<name><ext>`，保持已有文件不变），其它 skill 加后缀
    （`<name>.<skill><ext>`）。于是跑 recipe 只会写 `last_report.recipe.json`，
    碰不到 notes 的 `last_report.json`。

    为什么需要：语料和索引本来就是隔离的（两份不同的 `.env.<skill>`、两个索引目录），
    但报告文件原先没带 skill —— 换语料跑一次评测，上一份基线就没了。
    """
    if skill == "notes":
        return os.path.join(root, f"{name}{ext}")
    return os.path.join(root, f"{name}.{skill}{ext}")


def describe_corpus(skill: str, config) -> str:
    """一行说清「这次到底跑的哪份语料」，让 skill 与语料不匹配时**看得见**。"""
    return (
        f"skill={skill}  语料={config.data_path}  glob={config.file_glob}  "
        f"索引={config.index_save_path}"
    )


def allow_llm_free_run(config) -> None:
    """让**不调用 LLM** 的评测（检索层 / Agent 对照层）在没配 API key 时也能跑。

    `RAGConfig.validate()` 会硬性要求 `LLM_API_KEY`，于是「没配 key」与「配置写错了」
    被混为一谈 —— 而那两层评测本来一次请求都不发（只跑检索与判分）。
    这里给一个占位 key 让管线能构建；生成器不会被用到，真要调用会在网络层报错。
    """
    if not getattr(config, "llm_api_key", ""):
        print("（未配置 LLM_API_KEY：本次评测不调用 LLM，用占位 key 构建管线）")
        config.llm_api_key = "not-needed-for-this-eval"
    if not getattr(config, "llm_base_url", ""):
        config.llm_base_url = "http://localhost/v1"


def prepare() -> None:
    """在构造任何 HTTP 客户端之前调用一次。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass

    raw = os.environ.get("NO_PROXY") or os.environ.get("no_proxy") or ""
    if "[::1]" not in raw:
        return
    # 只删掉 httpx 解析不了的那一项；::1 还在，语义不变，代理其余配置也都不动
    kept = [h for h in raw.split(",") if h.strip() and h.strip() != "[::1]"]
    os.environ["NO_PROXY"] = os.environ["no_proxy"] = ",".join(kept)
    logger.warning("已忽略 NO_PROXY 中的 [::1]（httpx 无法解析，见 README 已知限制）")
