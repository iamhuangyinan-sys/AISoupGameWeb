"""判定后端接口（§8.1）。

rules.decide() 完全不知道后端是谁 —— 这是切换与对比成本为零的原因（D17）。
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
        """返回一次判定的概率分布（§8.1）。

        `criteria` 为 None 时走常规三态（是 / 不是 / 不重要）。
        传了 dict 就按它来 —— 选择题用它把答案空间收窄成「c1 / c2 / … / none」
        （见 soup/rules.py 的 choice_criteria），dict 的 key 就是 distribution 的 key。
        """
        ...
