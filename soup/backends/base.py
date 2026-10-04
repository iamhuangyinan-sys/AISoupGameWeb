"""判定后端接口。

**整个项目只保留这一层抽象**，而且是有理由的：换成 DeepSeek 判定几乎不改下游 ——
`rules.decide()` 完全不知道后端是谁。后端自己负责把它的原始输出（Jev 的校准概率、
DeepSeek 的自报置信度）整理成同一形状的分布，差异就被关在这一层里了。
"""

from __future__ import annotations

from typing import Protocol

from soup.models import JudgeResult, Soup, Turn


class JudgeBackend(Protocol):
    name: str

    def judge(
        self,
        question: str,
        soup: Soup,
        history: list[Turn] | None = None,
        criteria: dict[str, str] | None = None,
    ) -> JudgeResult:
        """返回一次判定的概率分布。

        `criteria` 为 None 时走常规三态（是 / 不是 / 不重要）。
        传了 dict 就按它来 —— 选择题用它把答案空间收窄成「c1 / c2 / … / none」
        （见 soup/rules.py 的 choice_criteria），dict 的 key 就是 distribution 的 key。
        """
        ...
