"""本地正则意图识别（0 成本）。

成本纪律：意图识别能用正则就不用模型 —— 它在每一轮都会跑，
一旦走模型，DeepSeek 的输出计费会吃掉全部预算。
"""

from __future__ import annotations

import re
from enum import Enum


class Intent(str, Enum):
    YES_NO = "yes_no"  # 是非提问
    HINT = "hint"  # 求提示
    SOLVE = "solve"  # 提交答案
    CHAT = "chat"  # 闲聊


_HINT_PATTERNS = (
    r"提示",
    r"给点?线索",
    r"不知道了",
    r"想不出来",
    r"卡住了",
    r"帮帮我",
    r"猜不出",
)

_SOLVE_PATTERNS = (
    r"我知道了",
    r"我明白了",
    r"我猜是",
    r"答案是不是",
    r"真相是",
    r"我来解答",
)

# 只有**明确的寒暄/元对话**才算闲聊。
# 刻意不把「不像是非问句」当成闲聊：跑题问题应该交给 Jev 判「不重要」，
# 在本地拦掉等于白丢一轮 —— 它本来能告诉你「这个方向不对」。
_CHAT_PATTERNS = (
    r"^(你好|您好|哈喽|hi|hello|在吗|喂)",
    r"你是谁",
    r"怎么玩",
    r"好玩吗",
    r"谢谢",
    r"^(哈哈|嘿嘿|嗯+|哦+)",
)


def detect_intent(question: str) -> Intent:
    """先看提示/提交答案这类**动作指令**，再看是不是明确的闲聊。

    顺序很关键：「我猜是他在打嗝吧？」既是疑问句又像提交答案，
    实际是玩家在提交复述，应该走通关判定。

    兜底是 YES_NO 而不是 CHAT：拿不准就交给 Jev 判，判不动它会弃权，
    但至少有答案；本地拦掉则连机会都没有。
    """
    q = question.strip()
    if not q:
        return Intent.CHAT
    if any(re.search(p, q) for p in _SOLVE_PATTERNS):
        return Intent.SOLVE
    if any(re.search(p, q) for p in _HINT_PATTERNS):
        return Intent.HINT
    if any(re.search(p, q, re.IGNORECASE) for p in _CHAT_PATTERNS):
        return Intent.CHAT
    return Intent.YES_NO
