"""DeepSeek 直判（可替换的「后端 B」）。

两种用途，同一份实现：
  1. **整体替换判定后端** —— 把 JUDGE_BACKEND 改成 deepseek 就切换
  2. **二次判定** —— Jev 低置信/弃权时叫它复核（见 soup/cascade.py）

对它的要求：需 JSON 约束 + 自报置信度 —— 因为 DeepSeek 没有 Jev 那样的
校准概率，只能让它自己报，可靠性低一档（别把它自报的置信度当概率用）。

成本纪律：
  - 在线一律关思考模式：reasoning token 按输出计费，成本翻几倍
  - 固定前缀（system）放最前面、且跨汤稳定，吃缓存命中价（1/50）
"""

from __future__ import annotations

import json
import re
import time

import config
from openai import OpenAI
from soup.backends.base import JudgeBackend
from soup.models import (
    JudgeResult,
    Soup,
    Turn,
    Verdict,
    VerdictDistribution,
)

# 跨汤稳定 —— 放最前面吃缓存命中价
_HEAD = (
    "你是海龟汤游戏的主持人。玩家会问一个是非问题，你只能根据给定的汤底来判断，"
    "不要引入汤底以外的知识。\n"
)

# 两套选项集：
#   three —— 默认。与 Jev 的选项集保持一致（见 models.CHOICE_OPTIONS），用于「整体替换后端」。
#   four  —— 多一个 both，只在**复核**时用。加它的理由：Jev 侧的 both 是代码从 Δ 派生的，
#            DeepSeek 若只会答三态，就永远无法**确认**一个 both，只能推翻它 ——
#            实测确实如此（布偶 的 7 道 both 题，DeepSeek 全部高置信地给了 是/不是）。
#            both 给得很宽会变成挡箭牌，所以 criteria 里写了一条硬条件（必须指出两半）。
_CRITERIA = {
    "three": {
        "yes": "汤底支持这个说法。",
        "no": "汤底否定这个说法。",
        "irrelevant": "汤底不涉及这个提问所问的事情，或者这件事对汤底不重要。",
    },
    "four": {
        "yes": "汤底里**有原话**直接支持整句话。",
        "no": "汤底里**有原话**直接否定整句话。",
        "both": (
            "提问只是**部分成立**：有一部分与汤底相符、另一部分不符；"
            "或者方向大致对、但用词或范围不精确（主持人会说「可以这么说」）。"
        ),
        "irrelevant": "汤底根本不涉及这件事。",
    },
}

# 四态复核专属的开场 —— 讲清楚「你为什么会被叫来」。
#
# 依据：能转到复核的题，都是 Jev 自己拿不准的（max < JEV_CONFIDENT）。
# 布偶 实测：7 道 both 里只有 3 道进了复核集，而 DeepSeek 四态**把这 3 道全认出来了**；
# 漏掉的 4 道是因为 Jev 高置信地给了 是/不是，压根没转交。
# 所以「both 当默认、重点排查致命错误」这个先验是有数据支撑的，不是拍脑袋。
_FOUR_INSTRUCTIONS = (
    "注意你的处境：这个提问已经被另一个判定器看过，它拿不准才转交给你，"
    "所以它**大概率带点「是也不是」的味道** —— both 是你的默认倾向。\n"
    "但你的首要任务是**排查致命错误**：「该答不是却答成是」和「该答是却答成不是」。\n"
    "判断顺序：\n"
    "  1. 先查致命错误。只有汤底里有**原话**直接支持或直接否定整句话时，才选 yes / no，"
    "并在 basis 里把那句原话引出来。\n"
    "  2. 引不出原话，就选 both —— 那说明提问方向大致成立、但不够精确。\n"
    "  3. 汤底压根没提这件事，才选 irrelevant。"
)

_INSTRUCTIONS = {"three": "", "four": _FOUR_INSTRUCTIONS}

_SCHEMA = {
    "three": '{"verdict": "yes|no|irrelevant", "confidence": 0.0-1.0, "reason": "一句话"}',
    "four": (
        '{"verdict": "yes|no|both|irrelevant",'
        ' "basis": "选 yes/no 时填：汤底里直接支持或否定它的原话；其余情况填空字符串",'
        ' "confidence": 0.0-1.0, "reason": "一句话"}'
    ),
}

# 每个选项集允许的结论。four 里多出的 both 只可能在复核语境下出现。
_OPTIONS = {
    "three": (Verdict.YES, Verdict.NO, Verdict.IRRELEVANT),
    "four": (Verdict.YES, Verdict.NO, Verdict.BOTH, Verdict.IRRELEVANT),
}


def build_system(options: str = "three") -> str:
    """拼 system prompt。两个变体各自跨汤稳定，所以都能吃缓存命中价。"""
    if options not in _CRITERIA:
        raise ValueError(f"未知的选项集：{options!r}（可用：{sorted(_CRITERIA)}）")
    parts = [_HEAD.strip()]
    if _INSTRUCTIONS[options]:
        parts.append(_INSTRUCTIONS[options])
    parts.append(f"把提问归入以下 {len(_CRITERIA[options])} 类之一：")
    parts.append("\n".join(f"  {k}\n      {v}" for k, v in _CRITERIA[options].items()))
    parts.append(f"只输出 JSON，不要输出别的：\n{_SCHEMA[options]}")
    return "\n".join(parts)


def build_criteria_system(criteria: dict[str, str]) -> str:
    """按现算的 criteria 拼 prompt —— 选择题（斜杠语法）走这条。

    ⚠️ 这份 prompt 每题都不一样，吃不到缓存命中价。可以接受：选择题是少数，
    而且缓存省下的钱远小于「让玩家多问两轮」的成本。
    """
    keys = list(criteria)
    schema = "|".join(keys)
    return "\n".join(
        [
            _HEAD.strip(),
            "这次玩家给了一个**选择题**：候选说法已经列在下面。你只需要判断汤底支持哪一个。",
            "注意：选项里可能夹带着错误的前提（例如「在电梯里被人推下去的」，但『电梯里』本身"
            "就不成立）。这种情况不要勉强挑一个 —— 选 none。",
            "汤底支持不止一个候选时，选与汤底关系最直接的那一个；拿不准就选 none，不要猜。",
            f"把提问归入以下 {len(keys)} 类之一：",
            "\n".join(f"  {k}\n      {v}" for k, v in criteria.items()),
            f'只输出 JSON，不要输出别的：\n{{"verdict": "{schema}", "confidence": 0.0-1.0, "reason": "一句话"}}',
        ]
    )


class DeepSeekError(RuntimeError):
    pass


_shared_clients: dict[tuple[str, str], "OpenAI"] = {}


def shared_client(api_key: str):
    """复用的 DeepSeek client。

    solver（通关判定）、hinter（进度提示）、二次判定都是单次调用，各自 new 一个
    等于每次都白付一次 TLS 握手。

    ⚠️ **按 key 缓存**。开源后每个用户带自己的 key 来，全局单例会让第二个用户的
    请求复用第一个用户的 client —— 那是拿别人的额度在算。

    ⚠️ api_key 是**必填**的。以前可以不传然后回落 config.DEEPSEEK_API_KEY，
    那条路已经删了：现在 key 只有一个来源（浏览器请求头），
    没填就该明明白白报错，而不是去别处找一把。
    """
    key = (api_key or "").strip()
    if not key:
        raise DeepSeekError("缺少 DeepSeek API key。在网页右上角「设置」里填一个。")
    cache_key = (config.DEEPSEEK_BASE_URL, key)
    client = _shared_clients.get(cache_key)
    if client is None:
        client = OpenAI(api_key=key, base_url=config.DEEPSEEK_BASE_URL)
        _shared_clients[cache_key] = client
    return client


# 输出被截断时的兜底：verdict / confidence 总在 JSON 最前面，即使后面断了也能捞回来。
# 实测（布偶 #27）：没设 max_tokens 时偶发在 reason 中途断掉，前面的字段其实已经完整。
# 宁可要一个 reason 为空的结论，也不该把整道题变成异常 —— 实时对局不能因为截断崩掉。
_VERDICT_RE = re.compile(r'"verdict"\s*:\s*"([a-z_]+)"')
_CONF_RE = re.compile(r'"confidence"\s*:\s*([0-9.]+)')


def _salvage(raw: str) -> dict | None:
    match = _VERDICT_RE.search(raw)
    if match is None:
        return None
    data: dict[str, object] = {
        "verdict": match.group(1),
        "basis": "",
        "confidence": 0.0,
        "reason": "",
    }
    conf = _CONF_RE.search(raw)
    if conf is not None:
        data["confidence"] = conf.group(1)
    return data


class DeepSeekDirectBackend(JudgeBackend):
    name = "deepseek_direct"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        options: str = "three",
    ) -> None:
        # 没有回落：api_key=None 就是没 key，构造时不炸，让调用时报错（见 judge）
        self.api_key = api_key or ""
        self.model = model or config.DEEPSEEK_MODEL
        self.options = options
        self._system = build_system(options)
        # 日志里要能分清「三态后端」和「四态复核」—— 两者的结论含义不同
        self.name = "deepseek_direct" if options == "three" else f"deepseek_direct[{options}]"
        self._client = (
            OpenAI(api_key=self.api_key, base_url=config.DEEPSEEK_BASE_URL)
            if self.api_key
            else None
        )

    def close(self) -> None:
        close = getattr(self._client, "close", None)
        if callable(close):
            close()

    def judge(
        self,
        question: str,
        soup: Soup,
        history: list[Turn] | None = None,
        criteria: dict[str, str] | None = None,
    ) -> JudgeResult:
        if self._client is None:
            raise DeepSeekError("缺少 DeepSeek API key。在网页右上角「设置」里填一个。")

        # 常规三态用构造时算好的 prompt（跨汤稳定 → 吃缓存价）；
        # 选择题的 criteria 每题不同，只能现拼。
        system = build_criteria_system(criteria) if criteria else self._system
        allowed = tuple(criteria) if criteria else tuple(v.value for v in _OPTIONS[self.options])

        started = time.perf_counter()
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                # 同一局内稳定，放在提问之前，利于缓存
                {"role": "user", "content": f"汤面：{soup.surface}\n汤底：{soup.truth}"},
                {"role": "user", "content": f"提问：{question}"},
            ],
            response_format={"type": "json_object"},
            max_tokens=config.DEEPSEEK_MAX_TOKENS,
            # 在线一律关思考模式（reasoning token 按输出计费）
            extra_body={"thinking": {"type": config.DEEPSEEK_THINKING}},
        )
        latency_ms = int((time.perf_counter() - started) * 1000)

        raw = resp.choices[0].message.content or "{}"
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = _salvage(raw)
            if data is None:
                raise DeepSeekError(f"返回的不是合法 JSON：{raw[:200]}") from None

        verdict = str(data.get("verdict", "")).strip().lower()
        if verdict not in allowed:
            raise DeepSeekError(
                f"返回了选项集之外的 verdict：{verdict!r}（原始：{raw[:200]}）"
            )
        try:
            confidence = float(data.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = min(max(confidence, 0.0), 1.0)

        usage_dict = _read_usage(resp)

        return JudgeResult(
            distribution=_to_distribution(verdict, confidence, allowed),
            usage=usage_dict,
            latency_ms=latency_ms,
            note=str(data.get("reason", ""))[:200],
            debug={"model": self.model, "endpoint": config.DEEPSEEK_BASE_URL},
            raw=data,
        )


def _to_distribution(
    verdict: str, confidence: float, allowed: tuple[str, ...]
) -> VerdictDistribution:
    """自报置信度 → 概率分布（按当前选项集的类别数摊开）。

    ⚠️ 这是**有损映射**：DeepSeek 只给一个置信度，不给完整分布。
    把 (1 - confidence) 平均分给其余类别，是为了让下游的 rules.decide() 能统一处理。
    别把它当校准概率看 —— DeepSeek 的自报置信度可靠性低一档，这是它和 Jev 的
    根本差别（Jev 是校准过的概率，DeepSeek 只是自己填一个数）。

    `allowed` 是选项集的 key（三态是 yes/no/irrelevant，选择题是 c1/c2/…/none）。
    """
    others = [key for key in allowed if key != verdict]
    share = (1.0 - confidence) / len(others) if others else 0.0
    probabilities = {key: share for key in others}
    probabilities[verdict] = confidence
    return VerdictDistribution(probabilities=probabilities, confidence=confidence)


def _read_usage(resp) -> dict:
    """从响应里取用量，并按价格表折算美元（区分高低峰与缓存命中）。"""
    usage = resp.usage
    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
    completion_tokens = getattr(usage, "completion_tokens", 0) or 0

    # DeepSeek 会把命中缓存的前缀单独报出来，命中价只有未命中的 1/50
    details = getattr(usage, "prompt_tokens_details", None)
    cached = getattr(details, "cached_tokens", None) if details else None
    if cached is None:
        cached = getattr(usage, "prompt_cache_hit_tokens", None)
    cached = min(int(cached or 0), prompt_tokens)
    fresh = prompt_tokens - cached

    prices = config.DEEPSEEK_PRICES.get(config.DEEPSEEK_MODEL)
    if prices:
        hit_in, miss_in, out = prices[0] if config.is_deepseek_peak() else prices[1]
        cost = (cached * hit_in + fresh * miss_in + completion_tokens * out) / 1_000_000
    else:
        cost = 0.0

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cached_tokens": cached,
        "cost": cost,
    }
