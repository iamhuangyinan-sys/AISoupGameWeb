"""语言处理层（§4.4，可插拔、按需触发）。

当前实现了两件 0 成本的事：
  - 本地正则意图识别（intent.py）
  - 选择题的斜杠语法解析（choice.py）

复合问题拆解（decompose）曾实现过一版，经实测**已移除**：
把「A 而且 B」切成两个子问句再合并，是把语义判断降级成机械拼接，
会给出误导性答案（例如「他打嗝，而且是因为口渴才要水吗？」被算成「是，也不是」，
而正确回答是「不是」—— 打嗝是已知前提，真正在问的只有口渴）。
改由 Jev 直接对整句做语义判断。

选择题的斜杠语法是另一回事，它只做**文本切分**、不碰语义，所以留下来了 ——
区别见 choice.py 顶部说明。
"""

from soup.nlu.choice import ChoiceSyntaxError, parse_options
from soup.nlu.intent import Intent, detect_intent

__all__ = ["ChoiceSyntaxError", "Intent", "detect_intent", "parse_options"]
