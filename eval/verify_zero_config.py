"""验证「零配置接一个新领域」这条承诺：**只写 `RAGSkill(name=...)`，其余全默认**。

为什么要单独一个脚本：`docs/ADDING_A_SKILL.md` 里写着"放语料 + 注册一行就能跑"，
而这种承诺最容易变成"我在两个已有领域上反推觉得没问题"。这个脚本造一个**临时第三领域**
（3 篇任意 Markdown，没有 metadata.py、没有 prompts.py、没有 agent_identity），
把检索层与 Agent 层真的跑一遍，让那句话可复现。

零 LLM 成本：只建索引 + 检索 + 拼 prompt，不调用生成。
需要 embedding 模型（本地已缓存时不会联网）与 faiss。

跑法：
    python eval/verify_zero_config.py
退出码：0 = 零配置路径可用；1 = 某一步断了（会打印断在哪）。
"""
import os
import shutil
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "eval"))

import bootstrap  # noqa: E402

bootstrap.prepare()
os.chdir(ROOT)

from agent import build_tools  # noqa: E402
from rag_core import RAGConfig  # noqa: E402
from rag_core.pipeline import BasicRAGPipeline  # noqa: E402
from rag_core.skill import RAGSkill  # noqa: E402

CORPUS = {
    "handbook/报销制度.md": (
        "# 报销制度\n\n## 提交时限\n差旅报销需在返回后 30 日内提交 OA，逾期需部门负责人签字。\n"
        "\n## 上限\n每人每年差旅报销上限 5000 元。\n"
    ),
    "handbook/请假流程.md": (
        "# 请假流程\n\n## 年假\n年假需提前 3 个工作日在系统提交，直属主管审批。\n"
    ),
    "handbook/设备申领.md": (
        "# 设备申领\n\n## 笔记本\n新入职员工可申领一台笔记本，需 IT 部门登记资产编号。\n"
    ),
}


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="zeroconfig_")
    failures = []
    try:
        for rel, text in CORPUS.items():
            path = os.path.join(tmp, rel)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)

        cfg = RAGConfig()
        cfg.data_path = tmp
        cfg.file_glob = "**/*.md"
        cfg.index_save_path = os.path.join(tmp, "_index")
        cfg.rerank_enabled = False           # CPU 上重排不可用
        bootstrap.allow_llm_free_run(cfg)    # 本脚本不调用 LLM

        # ★ 全部默认值：没有 metadata_extractor / prompts / agent_identity
        skill = RAGSkill(name="zero_config_check")
        pipe = BasicRAGPipeline(cfg, skill).build()
        print(f"① pipeline 构建成功：{len(pipe.documents)} 篇 / {len(pipe.chunks)} 块")

        ids = sorted(d.metadata["article_id"] for d in pipe.documents)
        print(f"② 默认 extractor 产出 id：{ids}")
        if set(ids) != {
            "handbook/请假流程.md",
            "handbook/报销制度.md",
            "handbook/设备申领.md",
        }:
            failures.append("默认 id 不是「相对语料根的完整路径」")

        hits = pipe.retrieve("差旅报销多久内要提交？", top_k=3)
        got = [d.metadata["article_id"] for d in hits]
        print(f"③ 检索命中：{got[:2]}")
        if "handbook/报销制度.md" not in got:
            failures.append("检索没命中正确的文档")

        mode = pipe._pick_prompt_mode("general")
        print(f"④ 没写 prompts.py 时的 prompt mode：{mode}")
        if mode != "basic":
            failures.append("默认 prompt mode 不是 basic")
        rendered = str(pipe.prompts[mode].format(question="Q", context="C"))
        if "Q" not in rendered or "C" not in rendered:
            failures.append("默认 prompt 渲染不出 question/context")

        tools = build_tools(pipe)
        schemas = {t["function"]["name"]: t["function"] for t in tools.schemas()}
        list_params = schemas["list_articles"]["parameters"]["properties"]
        print(f"⑤ list_articles 参数：{list(list_params)}（未声明筛选维度 → 应为空）")
        if list_params:
            failures.append("没声明筛选维度却出现了筛选参数")

        from agent.loop import RAGAgent

        class _NoLLM:
            def chat(self, messages):  # pragma: no cover - 不该被调用
                raise AssertionError("零配置验证不应调用 LLM")

        prompt = RAGAgent(_NoLLM(), tools).system_prompt
        print(f"⑥ SYSTEM_PROMPT 里的示例 id：{tools.identity['id_hint']}")
        if tools.identity["id_hint"] not in CORPUS and "handbook" not in tools.identity["id_hint"]:
            failures.append("id_hint 不是语料里的真实 id")
        for junk in ("week", "笔记", "菜谱"):
            if junk in prompt:
                failures.append(f"SYSTEM_PROMPT 里残留了别的领域措辞：{junk}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    if failures:
        print("\n结论：零配置路径有断点 ✗")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\n结论：只写 RAGSkill(name=...) 就能建索引、检索、拼 prompt ✓")
    print("（想量化效果仍要写 eval/seeds.<skill>.jsonl —— 标注是人的判断，无法自动生成）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
