"""二次判定：Jev 拿不准时叫 DeepSeek 复核。

这是「后端 B」的第二种用法 —— 同一个 DeepSeekDirectBackend
既能当整体替换的判定后端，也能只做复核。

触发条件（任一）：
  - 主判定是**弃权**（模型自己也没主意）
  - **low_confidence**：采用了答案，但最大概率 < JEV_CONFIDENT

合并规则**全是代码**，不让模型自己商量：

    主判定弃权 + 二次置信 ≥ SECOND_CONFIDENT  → 采用二次意见
    主判定弃权 + 二次置信不足                 → 仍弃权
    两边一致                                  → 采用，标 agree
    两边不一致 + policy=abstain（默认）       → 弃权（保守）
    两边不一致 + policy=deepseek              → 采用二次意见

⚠️ **已知风险，必须用数据说话**：二次判定会把「弃权 / 低置信」变成「有答案」，
   而弃权本来不计入错误 —— 给了答案就可能给错。所以开关二次判定前后，
   必须用同一套标注集对比**致命错误率**（该答不是却答是）。
   跑法见 eval/run_eval.py 的 `--second-opinion`。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import config
from soup.backends.base import JudgeBackend
from soup.models import Soup, Turn, Verdict, VerdictDistribution
from soup.rules import low_confidence


@dataclass
class SecondOpinion:
    """一次复核的完整留痕（进回合日志，供调阈值）。"""

    trigger: str = ""  # abstained | low_confidence | error
    verdict: str = ""  # 二次判定的结论（四态：yes/no/both/irrelevant）
    confidence: float = 0.0
    note: str = ""
    outcome: str = ""  # agree | override | abstain_on_disagree | filled | rejected | error
    backend: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    latency_ms: int = 0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def trigger_reason(
    primary: Verdict,
    dist: VerdictDistribution,
    confident: float | None = None,
    enabled: bool | None = None,
) -> str:
    """返回触发复核的原因；空字符串表示不需要复核。

    `enabled` / `confident` 可覆盖，好让评测用缓存数据免费重算不同阈值的触发情况。
    """
    enabled = config.SECOND_OPINION if enabled is None else enabled
    if not enabled:
        return ""
    if primary is Verdict.ABSTAIN:
        return "abstained"
    if low_confidence(dist, confident):
        return "low_confidence"
    return ""


def combine(
    primary: Verdict,
    second: Verdict,
    second_confidence: float,
    policy: str | None = None,
    min_confidence: float | None = None,
    soften: bool | None = None,
    both_override: float | None = None,
) -> tuple[Verdict, str]:
    """把两份意见合成一个终判。纯代码，无模型参与。

    二次判定可以是**四态**（多一个 both）—— 这正是复核的价值所在：
    Jev 侧的 both 是代码从 |Δ| 派生的，DeepSeek 若只会答三态，就永远无法**确认**
    一个 both，只能推翻它。加 both 之后三条路径都能走通：

        二次确认 both            → agree，保持 both
        二次改口成明确答案        → 只有它够自信才采纳（这是唯一能推翻 both 的路径）
        二次想把明确答案软化成 both → **默认拒绝**（见 config.SECOND_SOFTEN 的实测对照）

    ⚠️ 两条实测提醒（两锅汤 49 题）：

    1. **合并策略修不掉「主判高置信判错」的致命错误**。布偶 两道「该是却判不是」
       的题（#6 孩子是否死于意外，Jev 给 0.77；#27 冲进房间为何吓傻，Jev 给 0.83）
       都高于复核门槛，复核根本没看到它们。想动它们只能放宽 JEV_CONFIDENT，
       但那会把 DeepSeek 的 both 误报一起放进来，反而拖低准确率。

    2. **二次判定是「拿准确率换致命错误率」**。开启后合计准确率从 67.3% 变 69.4%
       （几乎持平），但致命错误 1 → 0。这个交换是划算的 —— 多答对一道题远不如
       少答错一道题重要，因为「该答不是却答是」会把玩家带偏。

    这个权衡见 eval/ 的离线扫描脚本，不要凭感觉调。
    """
    policy = (policy or config.SECOND_OPINION_POLICY).lower()
    floor = config.SECOND_CONFIDENT if min_confidence is None else min_confidence
    # 两个旋钮可覆盖，好让评测用缓存数据离线扫（不花钱）
    allow_soften = config.SECOND_SOFTEN if soften is None else soften
    override_at = config.SECOND_BOTH_OVERRIDE if both_override is None else both_override

    if primary is Verdict.ABSTAIN:
        if second_confidence >= floor:
            return second, "filled"
        return Verdict.ABSTAIN, "rejected"

    if second is primary:
        return primary, "agree"

    if second is Verdict.BOTH:
        # 主判定明确、二次想软化 —— 默认**不**跟着软化（有实测对照，见 config.SECOND_SOFTEN）
        if allow_soften:
            return Verdict.BOTH, "soften"
        return primary, "keep"

    if primary is Verdict.BOTH:
        # 二次给出明确答案 —— 唯一能推翻 both 的路径
        if second_confidence >= override_at:
            return second, "override"
        return primary, "keep_both"

    if policy == "deepseek":
        return second, "override"
    # 默认：两边不一致时保守弃权，而不是赌一边
    return Verdict.ABSTAIN, "abstain_on_disagree"


def review(
    question: str,
    soup: Soup,
    primary: Verdict,
    dist: VerdictDistribution,
    history: list[Turn],
    backend: JudgeBackend | None,
) -> tuple[Verdict, SecondOpinion | None]:
    """按需复核。任何失败都退回主判定 —— 复核不能成为新的故障点。"""
    reason = trigger_reason(primary, dist)
    if not reason:
        return primary, None

    if backend is None:
        return primary, SecondOpinion(trigger=reason, outcome="error", note="未配置二次判定后端")

    try:
        result = backend.judge(question, soup, history)
        second, _ = result.distribution.top()
        if second is Verdict.ABSTAIN:
            raise ValueError("二次判定返回了弃权 —— 四态选项集里不该有这一项")
        verdict, outcome = combine(primary, second, result.distribution.confidence or 0.0)
        return verdict, SecondOpinion(
            trigger=reason,
            verdict=second.value,
            confidence=result.distribution.confidence or 0.0,
            note=result.note,
            outcome=outcome,
            backend=getattr(backend, "name", "?"),
            usage=result.usage,
            latency_ms=result.latency_ms,
        )
    except Exception as exc:  # noqa: BLE001 — 复核失败必须降级，不能打断对局
        return primary, SecondOpinion(
            trigger=reason,
            outcome="error",
            note=f"{type(exc).__name__}: {exc}"[:200],
            backend=getattr(backend, "name", "?"),
        )
