"""选择题语法（斜杠）解析 —— 0 成本。

为什么不自动识别「他是自杀还是被杀」这种自然问法：
「是选择题」其实是个**形式**问题，不是语义问题，但它没有可靠的词法标记 ——
「还是」也是副词（「你还是来了」），「或者」也能出现在是非句里。判错的代价是
给出一个言之凿凿的错答案，比让玩家多打一个字符贵得多。

所以约定一个**显式**语法：候选之间用 `/` 隔开。

    他是自杀/被杀        → 候选 ["自杀", "被杀"]
    凶手是邻居/他自己/都不是

好处是三重的：
  - 识别 100% 可靠，不需要模型，也不违反 §7.5 的成本纪律
  - 候选文本由玩家给定 —— 模型**不需要生成**选项（D9 禁止自由生成）
  - 切分是形式操作，语义仍然整句交给 Jev，不重蹈 nlu 里被证伪的「本地拆复合句」

⚠️ 与已移除的 decompose 的区别（见 soup/nlu/__init__.py）：那次是本地**代替模型判语义**
（把「A 而且 B」拆成两个独立是非题，丢掉了「A 是已知前提」这个只有模型看得见的信息）；
这里只做文本切分，原句照样进 state，Jev 依然能做 §4.1 的整句语义判断。
"""

from __future__ import annotations

import re

import config

# 半角与全角斜杠都认
_SPLIT_RE = re.compile(r"\s*[/／]\s*")

# 候选两头常见的语气/标点，切完顺手剥掉
_TRIM = " \t　,，.。;；:：!！?？、"


class ChoiceSyntaxError(ValueError):
    """斜杠语法写坏了（比如选项太多）—— 需要告诉玩家怎么改，不能默默当是非题。"""


def parse_options(question: str) -> list[str] | None:
    """「A/B」→ ["A", "B"]；不含斜杠或切不出两个候选时返回 None（当普通是非题）。

    ⚠️ 返回 None 的两种情况要分清：
      - 没有斜杠             → 其实是是非题，走常规路径
      - 有斜杠但切不出两项   → 同样走常规路径。Jev 对「A/B」这种写法会判「不重要」，
                               比本地硬猜一个答案安全。
    """
    if "/" not in question and "／" not in question:
        return None

    parts = [p.strip(_TRIM) for p in _SPLIT_RE.split(question.strip())]
    options = [p for p in parts if p]
    if len(options) < 2:
        return None

    if len(options) > config.MAX_CHOICE_OPTIONS:
        raise ChoiceSyntaxError(
            f"选择题最多 {config.MAX_CHOICE_OPTIONS} 个选项，这条有 {len(options)} 个。"
            "拆成两次问吧。"
        )
    return options
