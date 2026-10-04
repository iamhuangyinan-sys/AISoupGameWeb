"""游戏状态机。

主链路：

    玩家提问
      → 本地规则层 意图识别（0 成本）
      → Jev 判定层 一次 Choice 三态判定
      → 代码 拿不准时弃权（阈值）+ 从置信度差派生「是也不是」
      → （可选）DeepSeek 二次判定：弃权或 low_confidence 时复核
      → 模板回答

本地规则层**不做**复合句拆解（曾实现过，实测证伪后移除，见 soup/nlu/__init__.py）。
整句交给 Jev 做语义判断，由它识别「哪部分是已知前提、真正在问的是什么」。
"""

from __future__ import annotations

from soup import cascade
from soup.backends import make_backend, make_second_backend
from soup.backends.base import JudgeBackend
from soup.creds import Credentials
from soup.models import Soup, Turn, Verdict
from soup.nlu import Intent, detect_intent, parse_options
from soup.rules import (
    CHOICE_NONE,
    choice_criteria,
    decide,
    decide_choice,
    low_confidence,
    render_turn_reply,
)

_PENDING = {
    # ⚠️ 这两条现在是**死代码** —— app.py 里的 /hint 和 /solve 已经各自实现了，
    #    不会走到 engine.ask() 这条路。留着是为了说明「引擎本身不认识这两种意图」。
    Intent.HINT: "提示生成还没接进引擎 —— 由 app.py 的 /hint 单独处理",
    Intent.SOLVE: "通关判定还没接进引擎 —— 由 app.py 的 /solve 单独处理",
    Intent.CHAT: "闲聊不做判定",
}


class _Disabled:
    """哨兵：**显式**关闭二次判定。

    区别：`second_backend=None` 是「按配置自动决定」，而评测需要的是
    「只要主判定的原始结果」—— 不这样区分，评测会在引擎内部多打一次
    DeepSeek，既重复计费又让离线重算失去意义。
    """

    def __repr__(self) -> str:
        return "NO_SECOND_OPINION"


NO_SECOND_OPINION = _Disabled()


class GameEngine:
    """一局游戏。判定链路完整，提示与通关判定是后续里程碑。"""

    def __init__(
        self,
        soup: Soup,
        backend: JudgeBackend | None = None,
        second_backend: JudgeBackend | None = None,
        creds: Credentials | None = None,
    ) -> None:
        self.soup = soup
        # creds 来自本次请求（见 soup/creds.py）。为 None 等于「没有 key」——
        # 后端会带着空 key 构造，真正调用时报一句「去设置里填一个」，
        # **不会**去别处找一把（服务端没有可回落的东西）。
        # 评测脚本要自己构造 creds 传进来，见 eval/run_eval.py 的 creds_from_env()。
        self.creds = creds
        self.backend = backend or make_backend(creds)
        # None = 按配置自动决定；NO_SECOND_OPINION = 明确不要
        if second_backend is NO_SECOND_OPINION:
            self.second_backend = None
        elif second_backend is not None:
            self.second_backend = second_backend
        else:
            self.second_backend = make_second_backend(creds)
        self.history: list[Turn] = []

    def ask(self, question: str, replace_at: int | None = None) -> Turn:
        """问一次。返回完整 Turn，回答文本用 reply() 渲染。

        `replace_at` 给「编辑已问的问题」用：传第几问的回合号，就把那一问的结果
        顶替掉，而不是往历史里再追加一条。历史干净，后面的提示生成才不会被
        已经被改掉的那个问题带偏。
        """
        question = question.strip()
        intent = detect_intent(question)
        if intent is not Intent.YES_NO:
            raise NotImplementedError(_PENDING[intent])

        options = parse_options(question)
        if options is not None:
            return self._ask_choice(question, options, replace_at)

        result = self.backend.judge(question, self.soup, self.history)
        primary = decide(result.distribution)

        flags: list[str] = []
        if primary is Verdict.ABSTAIN:
            flags.append("abstained")
        elif low_confidence(result.distribution):
            # 采用了答案但把握不足 —— 灰区样本，调阈值的最好素材
            flags.append("low_confidence")

        verdict, opinion = cascade.review(
            question, self.soup, primary, result.distribution, self.history, self.second_backend
        )
        if opinion is not None:
            flags.append(f"second:{opinion.outcome}")
            if opinion.trigger:
                flags.append(f"second_trigger:{opinion.trigger}")

        usage = dict(result.usage)
        latency = result.latency_ms
        if opinion is not None and opinion.usage:
            usage["second_opinion"] = opinion.usage
            latency += opinion.latency_ms

        turn = Turn(
            turn=len(self.history) + 1,
            question=question,
            distribution=result.distribution,
            verdict=verdict,
            abstained=verdict is Verdict.ABSTAIN,
            latency_ms=latency,
            usage=usage,
            flags=flags,
            primary_verdict=primary,
            second=opinion.as_dict() if opinion is not None else None,
        )
        self._place(turn, replace_at)
        return turn

    def _ask_choice(
        self, question: str, options: list[str], replace_at: int | None
    ) -> Turn:
        """选择题（斜杠语法，见 soup/nlu/choice.py）。

        一次 Choice，选项就是玩家给的候选 + 一个「以上都不是」。
        语义判断仍然完整交给模型：原句照样进 state，所以选项里夹带错误前提时
        （「他是在电梯里自杀还是被推下去的」，而『电梯里』压根不成立）模型有机会选 none。

        ⚠️ 选择题**不走二次判定**：cascade 的复核 prompt 是围绕四态写的
        （「大概率是是也不是」「排查致命错误」），套到 N 选一上没有意义。
        真要复核，得另写一份 prompt + 重标一遍阈值 —— 等有数据证明有必要再说。
        眼下用 decide_choice() 的保守阈值兜：拿不准就弃权，而不是猜。
        """
        criteria = choice_criteria(options)
        result = self.backend.judge(question, self.soup, self.history, criteria=criteria)
        picked = decide_choice(result.distribution)

        # verdict 记录「被选中的命题成立与否」：命中候选 = 成立，none = 全不成立，
        # 弃权 = 不知道。具体命中哪个在 turn.choice 里（见 models.Turn）。
        if picked is None:
            verdict = Verdict.ABSTAIN
        else:
            verdict = Verdict.NO if picked == CHOICE_NONE else Verdict.YES

        flags = ["choice", f"choice_n:{len(options)}"]
        if picked is None:
            flags.append("abstained")

        turn = Turn(
            turn=len(self.history) + 1,
            question=question,
            distribution=result.distribution,
            verdict=verdict,
            abstained=picked is None,
            latency_ms=result.latency_ms,
            usage=result.usage,
            flags=flags,
            choice={
                "options": options,
                "picked": picked,
                "probabilities": result.distribution.as_dict(),
            },
        )
        self._place(turn, replace_at)
        return turn

    def _place(self, turn: Turn, replace_at: int | None) -> None:
        """把这一问放回历史：`replace_at` 顶替原来那条，否则追加。"""
        if replace_at is not None and 1 <= replace_at <= len(self.history):
            turn.turn = self.history[replace_at - 1].turn
            self.history[replace_at - 1] = turn
        else:
            self.history.append(turn)

    def reply(self, turn: Turn) -> str:
        """回答只用模板，禁止自由生成 —— 免得模型自己编一句话糊弄过去。"""
        return render_turn_reply(turn)

    def close(self) -> None:
        for backend in (self.backend, self.second_backend):
            close = getattr(backend, "close", None)
            if callable(close):
                close()
