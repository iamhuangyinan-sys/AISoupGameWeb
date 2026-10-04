"""Jev 直读汤底（默认后端）。

一次请求、一个问题：把「这个提问相对汤底属于哪一类」直接交给 Jev 判断。
Choice 会返回全部四路的概率分布，所以不存在信息丢失。

⚠️ **与最初的设计不同**（有实测数据支撑，见 soup/rules.py 顶部说明）：
    原设计是三个 Noul（holds / fails / relevant）再由代码组合出四态。
    实测该设计让「是，也不是」在逻辑上不可达 —— 单个命题的真与假互补，
    `holds ≥ 0.8 且 fails ≥ 0.8` 几乎不可能同时成立。
    两个反例（汤底：酒保拔枪但枪里没子弹）：
        「枪是真枪吗？」          → 0.72 / 0.40 → 弃权（期望「是，也不是」）
        「他打嗝，而且是因为口渴才要水吗？」 → 0.29 / 0.60 → 弃权（期望「不是」）

另外：模型只回答**三态**（是 / 不是 / 不重要），不再提供「是也不是」选项 ——
    「是也不是」由代码从 P(是) 与 P(不是) 的置信度差派生，见 soup/rules.py 的 decide()。
    让模型自己选 both 会被它当挡箭牌：概率一摊，是/不是 两边都含糊。

同一份代码同时适配两个 transport：
  - OpenRouter：https://openrouter.ai/api/v1/systemone
  - TypeSafe  ：https://api.typesafe.ai/v1/systemone
路径与请求体完全一致，只有 endpoint / key / model 不同 —— 见 config.py。
"""

from __future__ import annotations

import time

import config
import httpx
from soup.backends.base import JudgeBackend
from soup.creds import Credentials
from soup.models import CHOICE_OPTIONS, JudgeResult, Soup, Turn, VerdictDistribution

# 成本纪律：提问文本放进 state 一次，不在 instructions 里重复
_INSTRUCTIONS = (
    "在 `truth` 的语境下，`player_question` 这个说法属于哪一类？\n"
    "注意：如果提问里含有汤底已经确认的内容，而玩家显然只是把它当已知前提、"
    "真正想问的是另一部分，那就只判断真正在问的那一部分。"
)

# 只有三态。**故意不给「是也不是」这个选项** —— 给了它就会变成挡箭牌，
# 模型把概率摊进去，是/不是 两边都变得含糊。
# 「是也不是」由 rules.decide() 从 |P(是) − P(不是)| 派生 —— 不让模型自己选。
_CRITERIA = {
    "yes": "汤底支持这个说法。",
    "no": "汤底否定这个说法。",
    "irrelevant": "汤底不涉及这个提问所问的事情，或者这件事对汤底不重要。",
}

_VERDICT_QUESTION = {
    "verdict": {
        "type": "choice",
        "instructions": _INSTRUCTIONS,
        "criteria": _CRITERIA,
    }
}

# 选择题（斜杠语法）专用的补充说明。key 是 c1/c2/…/none，不是四态，所以 criteria 由
# rules.choice_criteria() 现算；instructions 必须换一份，否则模型会拿四态的语义去套。
_CHOICE_INSTRUCTIONS = (
    "`player_question` 里已经列出了几个候选说法，你只需要判断汤底支持**哪一个**。\n"
    "注意：\n"
    "  - 选项里可能夹带着错误的前提（例如「在电梯里被人推下去的」，但『电梯里』本身就不成立）。"
    "这种情况不要勉强挑一个 —— 选 none。\n"
    "  - 汤底支持**不止一个**候选时，选与汤底关系最直接的那一个。\n"
    "  - 拿不准就选 none，不要猜。"
)

_RETRY_STATUS = {429, 529}  # 429 超限流 / 529 服务过载 → 指数退避
_MAX_ATTEMPTS = 3


class JevError(RuntimeError):
    pass


class JevDirectBackend(JudgeBackend):
    name = "jev_direct"

    def __init__(
        self,
        endpoint: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        proxy: str | None = None,
    ) -> None:
        self.endpoint = endpoint or config.JEV_ENDPOINT
        # ⚠️ 没有回落。`api_key=None` 就是「没有 key」，不去 config 里找。
        #    以前写的是 `api_key or config.JEV_API_KEY`，那让「服务端配了兜底 key」
        #    这条路处处都能溜进来 —— 现在整套删了，这里也就没必要区分 None 和 ""。
        self.api_key = api_key or ""
        self.model = model or config.JEV_MODEL
        # 不要依赖系统代理，显式传 proxy（留空 = 跟随系统代理，不是直连）
        proxy = proxy if proxy is not None else config.JEV_PROXY
        self._client = httpx.Client(timeout=config.JEV_TIMEOUT_S, proxy=proxy or None)

    @classmethod
    def from_creds(cls, creds: Credentials | None) -> "JevDirectBackend":
        """按本次请求的凭据构造。

        端点、模型都由 transport 推出来（见 config.resolve_jev），
        所以界面上只需要选一个 transport 名字 + 填 key。
        """
        if creds is None:
            return cls()
        return cls(
            endpoint=creds.jev_endpoint,
            api_key=creds.jev_api_key,
            model=creds.jev_model,
        )

    def close(self) -> None:
        self._client.close()

    def judge(
        self,
        question: str,
        soup: Soup,
        history: list[Turn] | None = None,
        criteria: dict[str, str] | None = None,
    ) -> JudgeResult:
        if not self.api_key:
            raise JevError(
                f"缺少 Jev API key（transport={config.JEV_TRANSPORT}）。"
                "在网页右上角「设置」里填一个，或写进 .env。"
            )

        if criteria:
            spec = {
                "type": "choice",
                "instructions": _CHOICE_INSTRUCTIONS,
                "criteria": criteria,
            }
            keys = list(criteria)
        else:
            spec = _VERDICT_QUESTION["verdict"]
            keys = [v.value for v in CHOICE_OPTIONS]

        payload = {
            "model": self.model,
            "state": {
                "surface": soup.surface,
                "truth": soup.truth,
                "player_question": question,
            },
            "questions": {"verdict": spec},
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        started = time.perf_counter()
        body = self._post_with_backoff(payload, headers)
        latency_ms = int((time.perf_counter() - started) * 1000)

        answer = (body.get("answers") or {}).get("verdict")
        if not isinstance(answer, dict):
            raise JevError(f"响应缺少 answers.verdict：{body}")

        # 只保留当前选项集的类别，且补全缺失项 —— 模型偶发返回多余/缺失键时不让下游崩
        raw_probs = answer.get("probabilities") or {}
        probabilities = {k: float(raw_probs.get(k, 0.0)) for k in keys}
        if sum(probabilities.values()) <= 0:
            raise JevError(f"概率全为 0：{answer}")

        return JudgeResult(
            distribution=VerdictDistribution(
                probabilities=probabilities,
                confidence=answer.get("confidence"),
            ),
            usage=body.get("usage") or {},
            latency_ms=latency_ms,
            debug={"model": body.get("model"), "endpoint": self.endpoint},
            raw={
                "choice": answer.get("choice"),
                "probabilities": raw_probs,
                "confidence": answer.get("confidence"),
            },
        )

    def _post_with_backoff(self, payload: dict, headers: dict) -> dict:
        last_error = ""
        for attempt in range(_MAX_ATTEMPTS):
            try:
                resp = self._client.post(self.endpoint, json=payload, headers=headers)
            except httpx.HTTPError as exc:
                last_error = f"网络错误：{exc}"
            else:
                if resp.status_code == 200:
                    return resp.json()
                if resp.status_code in _RETRY_STATUS:
                    last_error = f"HTTP {resp.status_code}"
                else:
                    raise JevError(f"HTTP {resp.status_code}：{resp.text[:500]}")
            if attempt < _MAX_ATTEMPTS - 1:
                time.sleep(0.5 * (2**attempt))
        raise JevError(f"重试 {_MAX_ATTEMPTS} 次仍失败：{last_error}")
