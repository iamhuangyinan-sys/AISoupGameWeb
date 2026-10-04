"""通关判定（§4.5b）：玩家写下自己的猜测，判断是否基本还原了汤底。

为什么交给模型而不是关键词 / 相似度匹配：海龟汤的汤底是一条**因果链**，玩家几乎
不可能用同样的措辞复述 ——「孩子死了她接受不了，所以把自己关起来」和汤底的
「她疯了，整日把自己关在婴儿房内」说的是同一件事。词面匹配会把这类正确答案全判错。

成本：一次判定 ≈ ¥0.0005，远低于一局问答（¥0.011），所以玩家可以反复提交。
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field

import config

# 复用后端 B 的用量折算（§5.2 价格表 + 高低峰 + 缓存命中），计费逻辑只写一遍。
# 下划线是那个模块的私有约定，同包内引用是刻意的。
from soup.backends.deepseek_direct import _read_usage, shared_client
from soup.creds import Credentials
from soup.models import Soup


class SolveError(RuntimeError):
    pass


_SYSTEM = (
    "你是海龟汤游戏的主持人。玩家给出了他对真相的完整猜测，你来判断他是否**基本还原**了汤底。\n"
    "判定要点：\n"
    "  - 看**因果链**，不看措辞。玩家用自己的话把「谁做了什么、为什么、结果怎样」说对就算过。\n"
    "  - 关键情节缺失、或因果说反了，不算过。\n"
    "  - 玩家多说了无关紧要的细节不影响通过；用词和汤底不一样也不扣分。\n"
    "  - 宁可偏松：玩家愿意写下完整猜测，说明方向大概率是对的，别因为细节措辞卡人。\n"
    "只输出 JSON，不要输出别的：\n"
    '{"score": 0.0-1.0, "passed": true|false, '
    '"missing": "没通过时填：还差哪些关键情节（30 字内）；通过时填空字符串", '
    '"comment": "一句点评（20 字内）"}'
)


@dataclass
class SolveResult:
    passed: bool
    score: float
    missing: str = ""
    comment: str = ""
    latency_ms: int = 0
    usage: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


def _clamp(value: object) -> float:
    try:
        score = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
    return min(max(score, 0.0), 1.0)


def evaluate(guess: str, soup: Soup, creds: Credentials | None = None) -> SolveResult:
    started = time.perf_counter()
    resp = shared_client(creds.deepseek_api_key if creds else None).chat.completions.create(
        model=config.DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {
                "role": "user",
                "content": (
                    f"汤面：{soup.surface}\n\n"
                    f"汤底（标准答案）：{soup.truth}\n\n"
                    f"玩家的猜测：{guess}"
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
        # 通关判定失败是可以让玩家重试的，不像对局那样必须降级，所以直接报错。
        raise SolveError(f"返回的不是合法 JSON：{raw[:200]}") from exc

    score = _clamp(data.get("score"))
    # 模型自报的 passed 和 score 偶尔会打架（说通过却只给 0.3），拿阈值兜一道
    passed = bool(data.get("passed")) and score >= config.SOLVE_PASS_SCORE

    return SolveResult(
        passed=passed,
        score=score,
        missing=str(data.get("missing", ""))[:200],
        comment=str(data.get("comment", ""))[:200],
        latency_ms=latency_ms,
        usage=_read_usage(resp),
    )
