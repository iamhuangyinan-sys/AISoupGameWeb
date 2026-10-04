"""四态判定与回答模板（§4.6）。

判定层设计已按实测数据调整（见下方说明），与文档 §4.2 不同。

原来的 §4.2 映射表：
    holds ≥ HIGH 且 fails ≥ HIGH  → 「是，也不是」

问题：单个问句的「真」和「假」是互补的，两条几乎不可能同时 ≥ 0.8，
所以「是，也不是」在逻辑上不可达。实测反例（汤底：酒保拔枪但枪里没子弹）：
    「枪是真枪吗？」  holds=0.72 fails=0.40  →  期望「是，也不是」，实际弃权
Jev 明明感知到了「部分成立」，是那张表读不出来。

现方案：把四态作为**一个语义问题**直接交给 Jev（一次 Choice，四选一），
本模块只负责「拿不准时弃权」这一件纯代码的事 —— 这也正是 D8 的要求。
"""

from __future__ import annotations

import config
from soup.models import Verdict, VerdictDistribution

VERDICT_LABEL = {
    Verdict.YES: "是",
    Verdict.NO: "不是",
    # 用户约定：不写逗号，就是四个字
    Verdict.BOTH: "是也不是",
    # 用户约定：irrelevant 统一写成「不重要」，不写「无关」
    Verdict.IRRELEVANT: "不重要",
}

# 弃权话术。§4.3 的原始长句是「这个我不太确定，你能换个说法再问一次吗？」，
# 用户要求回答只留一个词，所以压成「不确定」——界面上一眼扫过去就行。
# 语义没变：绝不能说「不重要」/「无关」，那是一个错误的断言。
ABSTAIN_REPLY = "不确定"


def decide(
    dist: VerdictDistribution,
    both_margin: float | None = None,
    irrelevant_min: float | None = None,
    abstain: float | None = None,
) -> Verdict:
    """三路概率分布（是 / 不是 / 不重要）→ 四态之一或弃权。

    「是，也不是」由**代码**从「是」与「不是」的置信度差派生 —— 这不靠模型自报，
    而是 Jev 校准概率的一个可测推论（D7）：

        max(三路) < ABSTAIN          → 弃权（模型自己也没主意）
        P(不重要) ≥ IRRELEVANT_MIN   → 不重要
        |P(是) − P(不是)| ≤ MARGIN   → 两边置信度相当 ⇒ 是，也不是
        否则                          → 概率高的那一侧

    为什么模型不再直接回答「是也不是」：四选一里 both 会变成挡箭牌，
    模型把概率摊给它，yes/no 就都软了。实测（29 题标注集）四种设计里，
    让模型直接选 both 的版本 either 永不触发、either 吞掉明确答案。
    """
    margin = config.JEV_BOTH_MARGIN if both_margin is None else both_margin
    irr_min = config.JEV_IRRELEVANT_MIN if irrelevant_min is None else irrelevant_min
    floor = config.JEV_ABSTAIN if abstain is None else abstain

    p_yes = dist.prob(Verdict.YES)
    p_no = dist.prob(Verdict.NO)
    p_irr = dist.prob(Verdict.IRRELEVANT)

    if max(p_yes, p_no, p_irr) < floor:
        return Verdict.ABSTAIN
    if p_irr >= irr_min:
        return Verdict.IRRELEVANT
    if abs(p_yes - p_no) <= margin:
        return Verdict.BOTH
    return Verdict.YES if p_yes > p_no else Verdict.NO


def low_confidence(dist: VerdictDistribution, confident: float | None = None) -> bool:
    """采用了答案、但把握不足 —— 这些样本是调阈值最有价值的素材（§4.3）。"""
    confident = config.JEV_CONFIDENT if confident is None else confident
    return max(dist.probabilities.values()) < confident


def is_abstain(verdict: Verdict) -> bool:
    return verdict is Verdict.ABSTAIN


def render_reply(verdict: Verdict, reason: str = "") -> str:
    """D9：回答只用模板，禁止自由生成。

    用户约定：**只回一个词，不带标点、不带任何额外字** ——
        是 / 不是 / 是也不是 / 不重要 / 不确定
    原模板的「不重要。这个和本案没有关系。」和 §4.3 的弃权长句都已压成一个词，
    理由：界面上要能一眼扫过去，不打断玩家提问的节奏。
    """
    if verdict is Verdict.ABSTAIN:
        return ABSTAIN_REPLY
    label = VERDICT_LABEL.get(verdict, ABSTAIN_REPLY)
    return f"{label}（{reason}）" if reason else label


# --- 选择题（斜杠语法，见 soup/nlu/choice.py） ----------------------------------
# 玩家写「他是自杀/被杀」，本地切出候选，**一次** Choice 让 Jev 在候选里挑一个。
#
# 为什么不是本地拆成两个是非题分别问：
# 那样会把「选项里的前提」丢掉（「他是在电梯里自杀还是被推下去的」——『电梯里』
# 可能根本不成立），而整句进 state 时 Jev 能自己发现这件事。所以候选只是**收窄了
# 答案空间**，语义判断仍然完整地交给模型。

# 「以上都不是」的 key 与话术。**不能省** —— Choice 是闭合的，只给两个候选等于
# 把模型逼进二选一：它明明想答「都不是」，也只能挑一个，那就是系统性地说谎。
CHOICE_NONE = "none"
CHOICE_NONE_REPLY = "都不是"


def option_key(index: int) -> str:
    """第 index 个候选（1-based）在 Choice 里的 key。"""
    return f"c{index}"


def choice_criteria(options: list[str]) -> dict[str, str]:
    """候选文本 → Choice 的 criteria。key 是 c1..cN，最后补一个 none。"""
    criteria = {
        option_key(i): f"汤底支持「{opt}」这个说法。" for i, opt in enumerate(options, 1)
    }
    joined = "、".join(f"「{opt}」" for opt in options)
    criteria[CHOICE_NONE] = f"汤底既不支持{joined}中的任何一个。"
    return criteria


def _option_from_key(key: str, options: list[str]) -> str | None:
    """`c2` → options[1]；格式不对或越界返回 None。"""
    if not (isinstance(key, str) and key.startswith("c")):
        return None
    try:
        index = int(key[1:]) - 1
    except ValueError:
        return None
    return options[index] if 0 <= index < len(options) else None


def decide_choice(
    dist: VerdictDistribution,
    margin: float | None = None,
    floor: float | None = None,
) -> str | None:
    """N 选一 → 命中的 key（`c1`… / `none`）；拿不准返回 None（弃权）。

    ⚠️ 不能沿用 decide()：那套阈值是给「是 / 不是 / 不重要」三路标定的，
    选项数一变，概率的分布形状和 Δ 的含义都不一样。
    ⚠️ 这两个阈值目前**没有校准数据**，所以取保守值 —— 弃权只说一句「不确定」，
    猜错了却是言之凿凿的错答案，两者代价不对等。
    """
    floor_v = config.JEV_CHOICE_MIN if floor is None else floor
    margin_v = config.JEV_CHOICE_MARGIN if margin is None else margin

    probs = {k: float(v) for k, v in dist.probabilities.items()}
    if not probs:
        return None
    ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    top_key, top_p = ranked[0]
    runner_p = ranked[1][1] if len(ranked) > 1 else 0.0

    if top_p < floor_v or (top_p - runner_p) < margin_v:
        return None
    return top_key


def choice_reply(choice: dict) -> str:
    """选择题的回答文本：命中的候选 / 都不是 / 不确定。"""
    picked = choice.get("picked")
    if not picked:
        return ABSTAIN_REPLY
    if picked == CHOICE_NONE:
        return CHOICE_NONE_REPLY
    text = _option_from_key(str(picked), choice.get("options") or [])
    return text if text is not None else ABSTAIN_REPLY


def render_turn_reply(turn) -> str:
    """一次问答的回答文本 —— 选择题与是非题的唯一出口，别再各写一份。"""
    if turn.choice is not None:
        return choice_reply(turn.choice)
    return render_reply(turn.verdict)
