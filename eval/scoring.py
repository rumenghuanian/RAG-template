"""生成层评测的**判分规则**，抽成纯函数模块，便于单独测试与离线重算。

这个模块被改过三次，每次都是踩了坑之后：

1. 最初用「答案里含弃答词就算弃答」全文匹配 → 4 条正常回答里 3 条被误判成弃答
   （模型正常作答时也会说「笔记里没有讲到 X」）。
2. 加了长度门槛（< 250 字才算弃答）→ 反过来漏判：Agent 写的是**长而完整的弃答**
   （「**结论：笔记库里没有找到与 COBOL 相关的内容。**」），被判成「没弃答＝编造」，
   于是「Agent 编造率 80%」这种明显失真的数字就出来了。
3. 现在：**看开头窗口 + 正则词族**，不用长度门槛，也不再依赖某一种固定说法
   （「笔记里没有」/「笔记库里没有」/「没有检索到」/「超出知识范围」都要能认出来）。

判分规则属于被测对象的一部分——它错一次，结论就全废一次。
"""
import re
from typing import Optional, Tuple

# ---------- 标注类别：**唯一权威定义** ----------
# 这三个列表原先在 run_eval / run_gen_eval / run_agent_eval / build_golden 里各抄了一份，
# 于是同一个口径 bug 只在一处被修好、另一处留着：`partial` 没有 gold（expect 恒为空），
# 「命中」类指标对它永远是 0。生成层改成了 cited_any()，Agent 对照层漏了，
# 报告里就长期挂着一行**五个臂全是 0.0% 的假数字**（钉在极值 = 先怀疑指标）。
# 现在只有这里定义，谁要判断类别都调 bucket_of()。
ANSWERABLE_KINDS = ["easy", "paraphrase", "rare_token", "multi"]
GATE_KINDS = ["absent", "mentioned"]   # 语料答不了：absent 术语不存在；mentioned 术语在但答不了
MANUAL_KINDS = ["partial"]              # 人工判读：没有干净 gold，判据是「有没有明说不完全」


def bucket_of(kind: str) -> str:
    """一个标注属于哪一类口径：answerable / gate / manual。

    manual 类**不进**任何命中率分母（它没有 expect，硬算只会得到恒 0 的假指标）。
    """
    if kind in GATE_KINDS:
        return "gate"
    if kind in MANUAL_KINDS:
        return "manual"
    return "answerable"


# 检索为空时 pipeline.query 的固定返回（系统级信号，不是猜的）
NO_RESULT = "抱歉，没有找到相关信息。"
# 只看开头这么多字。弃答是**表态**，出现在最前面；出现在中段的通常是顺带一提。
HEAD_CHARS = 200

# 弃答词族。写成正则而不是固定串：模型不会每次都用同一句话，
# 「笔记里没有」和「笔记库里没有」必须都认出来。
REFUSAL_PATTERNS = [
    # 笔记/知识库 + 没有/未 + （副词）+ 找到/讲/涉及…：模型很爱说「没有专门、系统地讲」
    r"(笔记|笔记库|课程笔记|课程库|知识库|资料库|语料)[^。！？\n]{0,16}?(没有|没|未|不)"
    r"[^。！？\n]{0,12}?(找到|讲到|讲|提到|提及|涉及|包含|收录|介绍|说明|解释|回答|覆盖)",
    # 没有/未 + 找到/检索到/命中/讲…
    r"(没有|没|未|找不到|未能)[^。！？\n]{0,10}?"
    r"(找到|检索到|命中|搜到|讲到|讲|提到|提及|覆盖|回答|介绍|说明)",
    r"抱歉[，,][^。\n]{0,24}?(没有|无法|超出|不能)",
    r"(超出|不在)[^。！？\n]{0,14}?范围",
    r"信息不足",
    r"无法(回答|确定|基于|给出)",
    # 「我不凭记忆编」「不能凭印象补充」这类明确表态
    r"不(替|会|能)[^。！？\n]{0,6}?(你|凭|靠)[^。！？\n]{0,6}?(补充|编|记忆|印象|回答)",
    r"不(能|会)凭(记忆|印象)",
]
_REFUSAL_RE = re.compile("|".join(REFUSAL_PATTERNS))


def abstained(answer: str, needs_human: bool = False) -> Tuple[bool, str]:
    """是否弃答。返回 (结论, 依据)。

    规则：
      ① 检索为空（pipeline 返回固定串）→ 一定是弃答
      ② Agent 的 `needs_human` 为真 → 系统级信号，直接采信
      ③ 否则看**开头 `HEAD_CHARS` 字**有没有命中弃答词族

    ③ 不做长度限制：长弃答（带解释、带「我检索了哪几组关键词」）同样是弃答。
    """
    text = (answer or "").strip()
    if text == NO_RESULT:
        return True, "检索为空"
    if needs_human:
        return True, "needs_human"
    m = _REFUSAL_RE.search(text[:HEAD_CHARS])
    if m:
        return True, f"开头命中「{m.group(0)[:24]}」"
    return False, ""


def key_point_coverage(answer: str, terms) -> Tuple[float, list]:
    """答案覆盖了多少「关键要点术语」。返回 (覆盖率, 缺失的词表)。

    **这是覆盖率，不是正确性。** 它只能回答「答案有没有真的用上 gold 文章里的关键概念」，
    回答不了「说得对不对」。名字起得保守是故意的 —— 指标名夸大比指标缺失更危险。

    为什么不用 LLM 裁判：试了 4 版，每版都发现新的口径问题（见 README）。
    术语覆盖率虽然粗，但**完全确定、可复算、可审计**，而且能用「打乱对照」验证区分力。

    术语表由 `run_gen_eval._derive_probes()` 自动从 gold 文章提取
    （tf × idf 取前 N 个），不由人挑。
    """
    terms = [t for t in terms if t]
    if not terms:
        return 0.0, []
    text = (answer or "").lower()
    missing = [t for t in terms if t.lower() not in text]
    return (len(terms) - len(missing)) / len(terms), missing


def cited_any(answer: str, titles_by_id) -> bool:
    """答案有没有点出**语料里任意一篇**（不要求是 gold）。

    给 `partial` 类用：那类问题没有干净的 gold（`expect` 为空），
    所以「有没有给出相关方向」只能看它有没有指向语料里的某几篇。

    ⚠️ 早先我给 partial 也用了 `cited()`（判 gold），而 gold 恒为空 →
    报告里那个「引到相关内容 0%」是**指标造成的假结论**，不是系统的行为。
    真相是：答案早就在列「相关的内容出现在这几篇」了。
    """
    for aid, title in (titles_by_id or {}).items():
        if cited(answer, aid, title):
            return True
    return False


def cited(answer: str, article_id: Optional[str], title: str = "") -> str:
    """答案里有没有点出这篇文章。返回命中的形式：`id` / `title` / 空串。

    两个臂的引用格式**不一致**（单轮 prompt 要标题、Agent 要 article_id），
    所以两种都算，同时统计各自用了哪种——这本身就是个该修的产品问题。
    """
    if not article_id or not answer:
        return ""
    stem = article_id[:-3] if article_id.endswith(".md") else article_id
    if article_id in answer or stem in answer:
        return "id"
    if title and title in answer:
        return "title"
    return ""
