"""数据模型（§6 数据格式）。

Soup 只有四个必需字段，没有 claims / 要素表（D6）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import config
from pydantic import BaseModel, Field


class Verdict(str, Enum):
    """四态 + 弃权（§4.2 / D8）。"""

    YES = "yes"
    NO = "no"
    BOTH = "both"
    IRRELEVANT = "irrelevant"
    ABSTAIN = "abstain"  # D8：「拿不准」必须是合法输出


# 交给 Jev 判断的三个语义类别。
# 「是也不是」不在其中 —— 它由代码从 是/不是 的置信度差派生（见 rules.decide）。
CHOICE_OPTIONS = (Verdict.YES, Verdict.NO, Verdict.IRRELEVANT)


class Soup(BaseModel):
    """一条汤。校验只保证结构合法，不做任何生成或审核（D10 / D14）。"""

    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    surface: str = Field(min_length=1)
    truth: str = Field(min_length=1)
    # 标签。在题库界面里加/删/改，存在 soup.json 里跟着汤走。
    tags: list[str] = Field(default_factory=list)

    @property
    def truth_too_long(self) -> bool:
        """汤底超长只提醒、不拒绝入库（§4.7）。"""
        return len(self.truth) > config.TRUTH_MAX_CHARS


@dataclass(frozen=True)
class VerdictDistribution:
    """一次四态判定的完整概率分布。

    ⚠️ 这取代了 §6.2 回合记录里的 `scores: {holds, fails, relevant}`。
    现在只有一个问题（四选一），所以拿到的是一条分布而不是三个维度的概率。
    """

    probabilities: dict[str, float]
    confidence: float | None = None

    def top(self) -> tuple[Verdict, float]:
        """概率最高的那一项及其概率。"""
        key, prob = max(self.probabilities.items(), key=lambda kv: kv[1])
        return Verdict(key), prob

    def prob(self, verdict: Verdict) -> float:
        return float(self.probabilities.get(verdict.value, 0.0))

    def as_dict(self) -> dict[str, float]:
        return {k: round(float(v), 4) for k, v in self.probabilities.items()}


@dataclass
class JudgeResult:
    """判定后端统一返回（§8.1 的 JudgeBackend 契约）。"""

    distribution: VerdictDistribution
    usage: dict[str, Any] = field(default_factory=dict)
    latency_ms: int = 0
    note: str = ""
    debug: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class Turn:
    """一局里的一次问答（§6.2）。"""

    turn: int
    question: str
    distribution: VerdictDistribution
    verdict: Verdict
    abstained: bool
    latency_ms: int = 0
    usage: dict[str, Any] = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)
    # 二次判定：primary_verdict 是复核前的原始判定，second 是复核留痕（见 soup/cascade.py）。
    # 两者都留着，才能在评测里分清「主判定错了」还是「复核改错了」。
    primary_verdict: Verdict | None = None
    second: dict[str, Any] | None = None
    # 选择题（斜杠语法）的结果：{options: [...], picked: "c1"|"none"|None, probabilities: {...}}。
    # 是非题为 None。
    # ⚠️ verdict 字段对选择题仍然有效，只是**看的是被选项本身**：挑中某个候选 = 该命题成立
    #    （yes），挑中 none = 所有候选都不成立（no），弃权 = abstain。所以下游的
    #    render_reply / 上色 / 计分都不用为选择题改形状 —— 具体选了哪个在 choice 里。
    choice: dict[str, Any] | None = None
