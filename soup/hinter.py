"""进度提示：根据玩家已问过的问题，给一句**方向性**提示。

和 solver 一样是单次 DeepSeek 调用，一次 ≈ ¥0.0004。

⚠️ prompt 里死守「不剧透」是这里的命门：提示是**引导**，不是答案。
   一旦说出凶手 / 手法 / 动机，这一局就没得猜了。所以要求的输出是
   「还该往哪个方向问」，而不是「真相是什么」。
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field

import config

from soup.backends.deepseek_direct import _read_usage, shared_client
from soup.creds import Credentials
from soup.models import Soup, Turn
from soup.rules import render_turn_reply


class HintError(RuntimeError):
    pass


_SYSTEM = (
    "你是海龟汤游戏的主持人。玩家已经问过一些问题了，你来判断他走到哪一步，"
    "然后给一句**方向性**提示。\n"
    "硬要求：\n"
    "  - **绝对不要说出真相**，也不要说出任何关键情节（谁做的、怎么做的、为什么）。\n"
    "  - 提示只指出「还没探索的方向」，或者「他现在卡在哪」。\n"
    "  - 一句话，30 字以内。像主持人随口提点，不要像解题报告。\n"
    "  - **不要用句号**（整条提示就一句话，句号只会显得啰嗦）。\n"
    "  - 不要复述玩家已经问过的东西。\n"
    "  - 他已经很接近了，就鼓励一句，别揭穿。\n"
    "只输出 JSON，不要输出别的：\n"
    '{"hint": "一句话提示（30 字内）", "progress": 0.0-1.0}'
)


@dataclass
class HintResult:
    hint: str
    progress: float = 0.0
    latency_ms: int = 0
    usage: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


def _label(turn: Turn) -> str:
    """给模型看的「这一问答了什么」。

    ⚠️ 必须走 render_turn_reply()而不是查 VERDICT_LABEL —— 选择题的回答是候选文本
    （「被杀」/「都不是」），照四态查表只会得到「不确定」，等于把这一问抹掉。
    """
    return render_turn_reply(turn)


def generate(soup: Soup, turns: list[Turn], creds: Credentials | None = None) -> HintResult:
    """给一句提示。`turns` 是这一局到目前为止的问与答。"""
    asked = "\n".join(f"{t.turn}. {t.question} → {_label(t)}" for t in turns)

    started = time.perf_counter()
    resp = shared_client(creds.deepseek_api_key if creds else None).chat.completions.create(
        model=config.DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {
                "role": "user",
                "content": (
                    f"汤面（玩家能看到）：{soup.surface}\n"
                    f"汤底（你知情，但**不能透露**）：{soup.truth}\n\n"
                    f"玩家已经问过的 {len(turns)} 个问题：\n{asked}"
                ),
            },
        ],
        response_format={"type": "json_object"},
        max_tokens=config.DEEPSEEK_MAX_TOKENS,
        extra_body={"thinking": {"type": config.DEEPSEEK_THINKING}},
    )
    latency_ms = int((time.perf_counter() - started) * 1000)

    raw = resp.choices[0].message.content or "{}"
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HintError(f"返回的不是合法 JSON：{raw[:200]}") from exc

    try:
        progress = float(data.get("progress", 0.0))
    except (TypeError, ValueError):
        progress = 0.0

    # 句尾标点一律剪掉 —— prompt 里已经说了不要句号，但模型只有七成听话，
    # 而这东西每轮都会显示在玩家眼前，不能靠模型自觉（用户明确说看着碍眼）。
    hint = str(data.get("hint", "")).strip().rstrip("。．.！!；;，,")

    return HintResult(
        hint=hint[:200],
        progress=min(max(progress, 0.0), 1.0),
        latency_ms=latency_ms,
        usage=_read_usage(resp),
    )
