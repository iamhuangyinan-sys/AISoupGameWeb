"""评测：用人工标注集算 §9.2 的各项指标。

这是 §12 待验证清单的执行工具。你只需要准备两样东西：

  1. 一条汤（放进 data/*.json，格式见 §6.1，四个必需字段）
  2. 一份标注（eval/labels.csv）

然后跑：

    python eval/run_eval.py

## 标注文件格式

CSV，列固定为 `soup_id,question,expected_verdict,note`：

    soup_id,question,expected_verdict,note
    00000002,是不是《植物大战僵尸》里的向日葵？,yes,汤底明写
    00000002,她玩的是单机游戏吗？,no,她不是玩家

- `expected_verdict` 取值：yes / no / both / irrelevant
  - 另有 `abstain`：你预期**系统应该弃权**的题（题目本身没法判），用来验弃权路径
  - 留空或写 `-`：跳过这一行
- `note` 随便写，会原样打印出来，方便回看当时的判断依据
- `#` 开头的行会被当成注释忽略
- 文件用 UTF-8 保存（带不带 BOM 都能读）

## 为什么一次调用就能扫阈值

每次判定的**完整四路概率分布**都会被缓存下来（logs/eval_*.json）。
所以 `--sweep` 可以在同一批数据上重算不同阈值，**不需要重新调用 API、不额外花钱**。

## 常用参数

    --soup <id>        只跑这一条汤（默认跑标注文件里出现的全部）
    --labels <path>    标注文件，默认 eval/labels.csv
    --margin <0-1>     |P(是)-P(不是)| ≤ 此值 ⇒ 判「是也不是」，默认取 config.JEV_BOTH_MARGIN
    --irrelevant-min   P(不重要) ≥ 此值 ⇒ 判「不重要」，默认取 config.JEV_IRRELEVANT_MIN
    --abstain <0-1>    最大概率 < 此值 ⇒ 弃权，默认取 config.JEV_ABSTAIN
    --sweep            打印阈值扫描表（§12 第 6 项）
    --reuse <file>     复用上次缓存的结果，完全不调用 API（免费重跑）
    --limit <n>        只跑前 n 题，用来先小规模试水
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import os  # noqa: E402

import config  # noqa: E402
from soup import cascade, store  # noqa: E402
from soup.backends.deepseek_direct import DeepSeekDirectBackend  # noqa: E402
from soup.creds import Credentials  # noqa: E402
from soup.engine import NO_SECOND_OPINION, GameEngine  # noqa: E402
from soup.models import Soup, Verdict, VerdictDistribution  # noqa: E402
from soup.rules import decide, low_confidence  # noqa: E402


def creds_from_env() -> Credentials:
    """评测脚本的 key 从**环境变量**取，不走网页那套（浏览器 localStorage）。

    为什么这里可以读环境变量、而网页不行：网页是给「别人用你的部署」的，
    那种场景下服务端绝不能有 key 可花（见 soup/creds.py）。评测是你自己在
    本机跑的分析工具，没有第三个人，所以从环境变量取一把自己的 key 是合适的。

    用法（任选一种）：

        # Windows PowerShell
        $env:JEV_API_KEY="sk-or-v1-..."; $env:DEEPSEEK_API_KEY="sk-..."
        python eval/run_eval.py

        # bash
        JEV_API_KEY=sk-or-v1-... DEEPSEEK_API_KEY=... python eval/run_eval.py

    也可以写进 .env —— config.py 里的 load_dotenv 会把它装进环境变量，
    所以这里照样读得到。（但网页那边是**不会**用它的。）
    """
    jev = (
        os.getenv("JEV_API_KEY")
        or os.getenv("OPENROUTER_API_KEY")
        or os.getenv("TYPESAFE_API_KEY")
        or ""
    ).strip()
    transport = (os.getenv("JEV_TRANSPORT") or config.DEFAULT_TRANSPORT).lower()
    if transport not in config.JEV_TRANSPORTS:
        transport = config.DEFAULT_TRANSPORT
    return Credentials(
        transport=transport,
        jev_key=jev,
        deepseek_key=(os.getenv("DEEPSEEK_API_KEY") or "").strip(),
    )

VALID_EXPECTED = {"yes", "no", "both", "irrelevant", "abstain"}
SKIP_TOKENS = {"", "-", "?"}

# 允许直接写中文，省得你记英文代码
ALIASES = {
    "yes": "yes", "是": "yes", "对": "yes",
    "no": "no", "不是": "no", "否": "no",
    "both": "both", "是也不是": "both", "是, 也不是": "both",
    "irrelevant": "irrelevant", "无关": "irrelevant", "不重要": "irrelevant",
    "abstain": "abstain", "弃权": "abstain",
}

# 展示用：不写「无关」，统一用「不重要」（用户约定）
VERDICT_ZH = {
    "yes": "是",
    "no": "不是",
    "both": "是也不是",
    "irrelevant": "不重要",
    "abstain": "弃权",
    "error": "调用失败",
}

# §9.2 的目标值，用于在报告里标出达标与否
TARGETS = {
    "accuracy": 0.90,
    "fatal": 0.02,
    "reverse_fatal": 0.05,
    "abstain_low": 0.05,
    "abstain_high": 0.15,
    "p50_ms": 500,
    "p95_ms": 1500,
}


# --- 数据装载 -----------------------------------------------------------------


@dataclass
class Label:
    soup_id: str
    question: str
    expected: str
    note: str
    line_no: int


@dataclass(frozen=True)
class SecondConfig:
    """二次判定的一组参数。冻结成值对象，方便从缓存里批量对比不同策略。"""

    confident: float = 0.70  # 触发：主判定最大概率 < 此值 ⇒ 复核
    policy: str = "abstain"  # 两边不一致时：abstain（保守）| deepseek（让它拍板）
    min_confidence: float = 0.70  # 主判定弃权时，二次意见要达到这个把握才采用
    soften: bool = False  # 二次说 both 时，是否把主判定的明确答案也软化成 both
    both_override: float = 0.95  # 主判定是 both 时，二次要多久自信才能推翻它（>1 = 绝不）

    def label(self) -> str:
        return (
            f"confident>{self.confident} policy={self.policy} min_conf>{self.min_confidence} "
            f"soften={self.soften} both>{self.both_override}"
        )


@dataclass
class Row:
    """一题的实测结果。probabilities 是缓存的核心 —— 有了它就能免费重算阈值。

    second_* 是 DeepSeek 在某一行上的原始意见。**缓存下来是刻意的**：
    合并规则是纯代码（soup/cascade.py），所以拿到原始意见后，换 policy / 换阈值
    全都能离线重算，一分钱不花。这也是为什么评测对**每一行**都取二次意见，
    而不是只取触发了的那些 —— 否则改阈值就会改变触发集合，缓存立刻失效。
    """

    soup_id: str
    question: str
    expected: str
    note: str
    probabilities: dict[str, float] = field(default_factory=dict)
    latency_ms: int = 0
    cost: float = 0.0
    error: str = ""
    second_verdict: str = ""
    second_confidence: float = 0.0
    second_cost: float = 0.0
    second_latency_ms: int = 0
    second_error: str = ""

    def triggers_second(
        self, margin: float, irr_min: float, abstain: float, second: SecondConfig
    ) -> bool:
        """这一题在当前阈值下会不会触发复核。"""
        if self.error or not self.second_verdict:
            return False
        dist = VerdictDistribution(probabilities=self.probabilities)
        primary = decide(dist, margin, irr_min, abstain)
        return bool(
            cascade.trigger_reason(primary, dist, confident=second.confident, enabled=True)
        )

    def actual(
        self,
        margin: float,
        irr_min: float,
        abstain: float,
        second: SecondConfig | None = None,
    ) -> str:
        """复用 rules.decide() + cascade.combine()，判定逻辑不在两处各写一遍。"""
        if self.error:
            return "error"
        dist = VerdictDistribution(probabilities=self.probabilities)
        primary = decide(dist, margin, irr_min, abstain)

        if second is None or not self.triggers_second(margin, irr_min, abstain, second):
            return primary.value

        verdict, _ = cascade.combine(
            primary,
            Verdict(self.second_verdict),
            self.second_confidence,
            policy=second.policy,
            min_confidence=second.min_confidence,
            soften=second.soften,
            both_override=second.both_override,
        )
        return verdict.value

    def is_low_confidence(self, confident: float) -> bool:
        dist = VerdictDistribution(probabilities=self.probabilities)
        return low_confidence(dist, confident)


def load_soups() -> dict[str, Soup]:
    """委托给 soup.store（编码与格式校验都在那儿统一处理）。"""
    return store.load_soups()


def load_labels(path: Path) -> list[Label]:
    """读标注 CSV。容错：BOM、注释行、空行、列顺序不同都能处理。"""
    if not path.exists():
        raise SystemExit(f"找不到标注文件：{path}\n先照着文件里的示例建一份。")

    text = path.read_text(encoding="utf-8-sig")
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    reader = csv.DictReader(lines)

    required = {"soup_id", "question", "expected_verdict"}
    missing = required - {(f or "").strip() for f in (reader.fieldnames or [])}
    if missing:
        raise SystemExit(
            f"标注文件缺少列：{sorted(missing)}\n"
            f"当前列：{reader.fieldnames}\n"
            "需要 soup_id, question, expected_verdict, note"
        )

    labels: list[Label] = []
    for i, row in enumerate(reader, start=2):
        question = (row.get("question") or "").strip()
        expected_raw = (row.get("expected_verdict") or "").strip().lower()
        if expected_raw in SKIP_TOKENS or not question:
            continue
        expected = ALIASES.get(expected_raw)
        if expected is None:
            raise SystemExit(
                f"第 {i} 行 expected_verdict 无法识别：{expected_raw!r}\n"
                f"可写：{sorted(VALID_EXPECTED)}\n"
                f"也可写中文：{sorted(k for k in ALIASES if k not in VALID_EXPECTED)}\n"
                "留空或写 - 表示跳过该行"
            )
        labels.append(
            Label(
                soup_id=(row.get("soup_id") or "").strip(),
                question=question,
                expected=expected,
                note=(row.get("note") or "").strip(),
                line_no=i,
            )
        )
    return labels


# --- 执行 ---------------------------------------------------------------------


def run_labels(
    labels: list[Label],
    soups: dict[str, Soup],
    fetch_second: bool = False,
    second_options: str = "four",
) -> list[Row]:
    """跑标注。

    fetch_second=True 时，对**每一题**额外取一次 DeepSeek 意见（注意有成本）。
    对每一题都取（而不是只取触发复核的题）才能让后续的阈值/policy 对比
    完全离线 —— 否则换个阈值触发集合就变，缓存作废。
    """
    rows: list[Row] = []
    second_backend = None
    creds = creds_from_env()
    if not creds.has_jev:
        raise SystemExit(
            "评测需要一把 Jev 的 key。它不在 .env 里（这个项目的网页端只用浏览器里填的），\n"
            "所以要在**环境变量**里给：\n\n"
            '    $env:JEV_API_KEY="sk-or-v1-..."   # PowerShell\n'
            "    export JEV_API_KEY=sk-or-v1-...   # bash\n\n"
            "（也可以写进 .env —— load_dotenv 会把它装进环境变量，这里读得到；"
            "但网页那边不会用它。）"
        )
    if fetch_second:
        if not creds.has_deepseek:
            raise SystemExit(
                "--second-opinion 需要 DeepSeek 的 key，同样从环境变量给：\n\n"
                '    $env:DEEPSEEK_API_KEY="sk-..."   # PowerShell\n'
                "    export DEEPSEEK_API_KEY=sk-...   # bash"
            )
        second_backend = DeepSeekDirectBackend(
            api_key=creds.deepseek_api_key, options=second_options
        )
        print(f"二次判定   = {config.DEEPSEEK_MODEL} @ {config.DEEPSEEK_BASE_URL}")
        print("             （每题都取一次意见，好让后面的阈值对比完全不花钱）")

    by_soup: dict[str, list[Label]] = defaultdict(list)
    for lb in labels:
        by_soup[lb.soup_id].append(lb)

    try:
        return _run_soups(by_soup, soups, second_backend, creds)
    finally:
        if second_backend is not None:
            second_backend.close()


def _run_soups(
    by_soup: dict[str, list[Label]],
    soups: dict[str, Soup],
    second_backend,
    creds: Credentials,
) -> list[Row]:
    rows: list[Row] = []
    for soup_id, group in by_soup.items():
        soup = soups.get(soup_id)
        if soup is None:
            raise SystemExit(
                f"标注里引用了不存在的汤 id：{soup_id!r}\n"
                f"data/ 里现有：{sorted(soups) or '（空）'}"
            )
        print(f"汤 [{soup_id}] {soup.title} —— {len(group)} 题，开始调用…")
        # NO_SECOND_OPINION：引擎里只跑 Jev。二次意见由评测自己取、自己合并，
        # 这样 primary 分布是干净的，合并规则也能离线重算。
        engine = GameEngine(soup, second_backend=NO_SECOND_OPINION, creds=creds)
        try:
            for lb in group:
                row = Row(
                    soup_id=soup_id,
                    question=lb.question,
                    expected=lb.expected,
                    note=lb.note,
                )
                try:
                    turn = engine.ask(lb.question)
                    row.probabilities = turn.distribution.as_dict()
                    row.latency_ms = turn.latency_ms
                    row.cost = float(turn.usage.get("cost") or 0.0)
                except Exception as exc:  # 单题失败不该毁掉整批
                    row.error = f"{type(exc).__name__}: {exc}"

                if second_backend is not None:
                    try:
                        opinion = second_backend.judge(lb.question, soup, [])
                        verdict, _ = opinion.distribution.top()
                        row.second_verdict = verdict.value
                        row.second_confidence = float(opinion.distribution.confidence or 0.0)
                        row.second_cost = float(opinion.usage.get("cost") or 0.0)
                        row.second_latency_ms = opinion.latency_ms
                    except Exception as exc:  # 复核失败不该毁掉整批
                        row.second_error = f"{type(exc).__name__}: {exc}"

                rows.append(row)
                mark = "!" if (row.error or row.second_error) else "."
                print(f"  {mark} {lb.question}", end="\r", flush=True)
        finally:
            engine.close()
        print(" " * 80, end="\r")
    return rows


def save_cache(rows: list[Row]) -> Path:
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = config.LOG_DIR / f"eval_{stamp}.json"
    payload = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "transport": config.JEV_TRANSPORT,
        "model": config.JEV_MODEL,
        "rows": [
            {
                "soup_id": r.soup_id,
                "question": r.question,
                "expected": r.expected,
                "note": r.note,
                "probabilities": r.probabilities,
                "latency_ms": r.latency_ms,
                "cost": r.cost,
                "error": r.error,
                "second_verdict": r.second_verdict,
                "second_confidence": r.second_confidence,
                "second_cost": r.second_cost,
                "second_latency_ms": r.second_latency_ms,
                "second_error": r.second_error,
            }
            for r in rows
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_cache(path: Path) -> list[Row]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [Row(**r) for r in payload["rows"]]


# --- 指标 ---------------------------------------------------------------------


def evaluate(
    rows: list[Row],
    margin: float,
    irr_min: float,
    abstain: float,
    second: SecondConfig | None = None,
) -> dict:
    total = len(rows)
    ok_rows = [r for r in rows if not r.error]
    actuals = {r.question: r.actual(margin, irr_min, abstain, second) for r in ok_rows}

    correct = sum(1 for r in ok_rows if actuals[r.question] == r.expected)
    decided = [r for r in ok_rows if actuals[r.question] != "abstain"]
    correct_decided = sum(1 for r in decided if actuals[r.question] == r.expected)

    # §9.2 致命错误：该答 no 却答 yes。单独统计，不平摊。
    should_no = [r for r in ok_rows if r.expected == "no"]
    fatal = [r for r in should_no if actuals[r.question] == "yes"]
    should_yes = [r for r in ok_rows if r.expected == "yes"]
    reverse_fatal = [r for r in should_yes if actuals[r.question] == "no"]

    abstained = [r for r in ok_rows if actuals[r.question] == "abstain"]
    # 过度弃权：本来有明确答案、却被弃权掉的（漏判）
    over_abstain = [r for r in abstained if r.expected != "abstain"]

    latencies = sorted(r.latency_ms for r in ok_rows if r.latency_ms)

    both_rows = [r for r in ok_rows if r.expected == "both"]
    both_hit = sum(1 for r in both_rows if actuals[r.question] == "both")

    def pct(n: int, d: int) -> float:
        return (n / d) if d else 0.0

    matrix: dict[str, dict[str, int]] = {
        e: {a: 0 for a in ("yes", "no", "both", "irrelevant", "abstain", "error")}
        for e in ("yes", "no", "both", "irrelevant", "abstain")
    }
    for r in rows:
        a = "error" if r.error else actuals[r.question]
        matrix.get(r.expected, {}).setdefault(a, 0)
        matrix[r.expected][a] = matrix[r.expected].get(a, 0) + 1

    return {
        "total": total,
        "errors": total - len(ok_rows),
        "scored": len(ok_rows),
        "should_no": len(should_no),
        "should_yes": len(should_yes),
        "correct": correct,
        "correct_decided": correct_decided,
        "decided": len(decided),
        "accuracy": pct(correct, len(ok_rows)),
        "accuracy_decided": pct(correct_decided, len(decided)),
        "fatal": fatal,
        "fatal_rate": pct(len(fatal), len(ok_rows)),
        "fatal_rate_no": pct(len(fatal), len(should_no)),
        "reverse_fatal": reverse_fatal,
        "reverse_fatal_rate": pct(len(reverse_fatal), len(should_yes)),
        "abstained": abstained,
        "abstain_rate": pct(len(abstained), len(ok_rows)),
        "over_abstain": over_abstain,
        "both_hit": both_hit,
        "both_n": len(both_rows),
        "p50": statistics.median(latencies) if latencies else 0,
        "p95": (latencies[int(len(latencies) * 0.95) - 1] if latencies else 0),
        "cost_total": sum(r.cost for r in ok_rows),
        "actuals": actuals,
        "matrix": matrix,
        "second_calls": sum(1 for r in ok_rows if r.second_verdict),
        "second_cost_total": sum(r.second_cost for r in ok_rows),
        "second_errors": [r for r in ok_rows if r.second_error],
    }


def _flag(ok: bool) -> str:
    return "达标" if ok else "未达标"


def print_report(
    m: dict,
    rows: list[Row],
    margin: float,
    irr_min: float,
    abstain: float,
    second: SecondConfig | None = None,
) -> None:
    print()
    print("=" * 96)
    title = f"逐题对照   （margin={margin} 不重要阈值={irr_min} 弃权阈值={abstain}"
    title += f" 二次判定={second.label()}）" if second else " 二次判定=关）"
    print(title)
    print("=" * 96)
    print(f"{'期望':<8}{'实际':<8}{'概率分布':<40}{'Δ':<7}{'提问'}")
    print("-" * 96)
    for r in rows:
        if r.error:
            print(f"{VERDICT_ZH[r.expected]:<8}{'调用失败':<8}{r.error[:36]:<40}{'':<7}{r.question}")
            continue
        a = m["actuals"][r.question]
        mark = " " if a == r.expected else "*"
        probs = " ".join(
            f"{VERDICT_ZH[k]}={r.probabilities.get(k, 0):.2f}"
            for k in ("yes", "no", "irrelevant")
        )
        delta = abs(r.probabilities.get("yes", 0) - r.probabilities.get("no", 0))
        print(f"{VERDICT_ZH[r.expected]:<8}{VERDICT_ZH[a]:<8}{probs:<40}{delta:<7.2f}{mark} {r.question}")
        if r.note:
            print(f"{'':<16}note: {r.note}")
    print("（* 表示与期望不符；Δ = |P(是) − P(不是)|，越小越接近「是也不是」）")

    print()
    print("=" * 96)
    print("混淆矩阵（行 = 你的标注，列 = 系统实际回答）")
    print("=" * 96)
    cols = ("yes", "no", "both", "irrelevant", "abstain", "error")
    print(f"{'':<10}" + "".join(f"{VERDICT_ZH[c]:>10}" for c in cols))
    for e in ("yes", "no", "both", "irrelevant", "abstain"):
        counts = m["matrix"].get(e, {})
        if not sum(counts.values()):
            continue
        print(f"{VERDICT_ZH[e]:<10}" + "".join(f"{counts.get(c, 0):>10}" for c in cols))

    print()
    print("=" * 96)
    print("指标（§9.2）")
    print("=" * 96)
    print(f"  题目总数                {m['total']}（可评分 {m['scored']}）")
    if m["errors"]:
        print(f"  调用失败                {m['errors']}  ⚠ 这些题未计入指标")
    print(
        f"  判定准确率              {m['correct']}/{m['scored']} = {m['accuracy']:.1%}"
        f"                              目标 > 90%   "
        f"{_flag(m['accuracy'] > TARGETS['accuracy'])}"
    )
    print(f"  判定准确率（不含弃权）  {m['correct_decided']}/{m['decided']} = {m['accuracy_decided']:.1%}")
    print()
    print(
        f"  ★ 致命错误率（该 no 却 yes）  {len(m['fatal'])}/{m['should_no']}（标注为 no 的题）"
        f" = {m['fatal_rate_no']:.1%}"
    )
    print(
        f"                                占全部题 {m['fatal_rate']:.1%}"
        f"                    目标 < 2%   {_flag(m['fatal_rate'] < TARGETS['fatal'])}"
    )
    for r in m["fatal"]:
        print(f"        ^ {r.question}")
    print(
        f"    反向致命错误率（该 yes 却 no）{len(m['reverse_fatal'])}/{m['should_yes']}"
        f" = {m['reverse_fatal_rate']:.1%}"
        f"                 目标 < 5%   {_flag(m['reverse_fatal_rate'] < TARGETS['reverse_fatal'])}"
    )
    print()
    print(
        f"  弃权率                  {len(m['abstained'])}/{m['scored']} = {m['abstain_rate']:.1%}"
        f"                         目标 5%–15%  "
        f"{_flag(TARGETS['abstain_low'] <= m['abstain_rate'] <= TARGETS['abstain_high'])}"
    )
    if m["over_abstain"]:
        print(f"    其中「本来有答案却被弃权」{len(m['over_abstain'])} 题（漏判）:")
        for r in m["over_abstain"][:10]:
            print(f"        ^ {r.question}  （期望 {r.expected}）")
    print()
    print(
        f"  延迟 P50 / P95          {m['p50']:.0f}ms / {m['p95']:.0f}ms"
        f"                     目标 <500ms / <1.5s   "
        f"{_flag(m['p50'] < TARGETS['p50_ms'])}"
    )
    print(f"  本次总成本              ${m['cost_total']:.6f}（约 ¥{m['cost_total'] * config.USD_TO_CNY:.4f}）")
    if m["total"]:
        per_q = m["cost_total"] / m["total"]
        print(
            f"  折合单题 / 单局(50问)   ${per_q:.6f} / ${per_q * 50:.6f}"
            f"（约 ¥{per_q * 50 * config.USD_TO_CNY:.4f}）"
        )
    print()


def print_sweep(rows: list[Row]) -> None:
    """§12 第 6 项：扫阈值找最优点。用缓存分布重算，不花钱。

    现在有两个可调项：margin（决定何时判「是也不是」）和 abstain（弃权线）。
    irrelevant_min 固定 —— 这批标注里只有 1 道不重要题，扫它没有意义。
    """
    ok = [r for r in rows if not r.error]
    if not ok:
        return
    irr_min = config.JEV_IRRELEVANT_MIN
    print("=" * 96)
    print(f"阈值扫描（用同一批缓存分布重算，不额外调用 API；不重要阈值固定 {irr_min}）")
    print("=" * 96)
    print(f"{'margin':>8}{'abstain':>9}{'准确率':>9}{'致命':>7}{'反向致命':>10}{'弃权':>7}{'漏判':>7}{'both命中':>10}")
    print("-" * 96)
    best = None
    for margin in (0.05, 0.10, 0.15, 0.20, 0.30):
        for abstain in (0.35, 0.40, 0.45, 0.50, 0.55):
            m = evaluate(ok, margin, irr_min, abstain)
            meets = m["fatal_rate"] < TARGETS["fatal"]
            in_band = TARGETS["abstain_low"] <= m["abstain_rate"] <= TARGETS["abstain_high"]
            tag = " ★" if meets else ""
            tag += " ◆" if in_band else ""
            print(
                f"{margin:>8.2f}{abstain:>9.2f}{m['accuracy']:>8.1%}{m['fatal_rate']:>7.1%}"
                f"{m['reverse_fatal_rate']:>10.1%}{m['abstain_rate']:>7.1%}"
                f"{len(m['over_abstain']):>7}{m['both_hit']:>4}/{m['both_n']:<5}{tag}"
            )
            # 排序：先保致命错误率达标，再要求弃权率落在 §9.2 目标带内，
            # 然后才比准确率。少了中间那项会挑出「零弃权」的档位 —— 那等于在硬猜。
            key = (
                meets,
                in_band,
                -m["fatal_rate"],
                m["accuracy"],
                m["both_hit"],
                -len(m["over_abstain"]),
            )
            if best is None or key > best[0]:
                best = (key, margin, abstain, m)
    print("（准确率* = 不含弃权；★ = 致命错误率达标；◆ = 弃权率落在 5%–15%）")
    if best:
        _, mg, ab, m = best
        print()
        print(
            f"推荐：margin={mg} abstain={ab} → 准确率 {m['accuracy']:.1%}，"
            f"致命错误率 {m['fatal_rate']:.1%}，反向致命 {m['reverse_fatal_rate']:.1%}，"
            f"弃权率 {m['abstain_rate']:.1%}，both 命中 {m['both_hit']}/{m['both_n']}"
        )
        if m["abstain_rate"] < TARGETS["abstain_low"]:
            print("  ⚠ 所有档位的弃权率都低于 5% —— 这批题要么太easy、要么覆盖不足。")
    print()


# --- 入口 ---------------------------------------------------------------------


def print_second_compare(
    rows: list[Row],
    margin: float,
    irr_min: float,
    abstain: float,
) -> None:
    """二次判定开 / 关对比。全部在缓存上离线重算，不再花一分钱。

    重点看 §8.2 决策门关心的一项：**致命错误率**。二次判定把「弃权」变成
    「有答案」——弃权本来不算错，一旦给了答案就可能答错，所以这个对比是
    判断该不该开二次判定的唯一依据。
    """
    if not any(r.second_verdict for r in rows):
        print("\n（缓存里没有二次判定数据；用 --second-opinion 跑一次，之后就能免费对比）")
        return

    configs: list[tuple[str, SecondConfig | None]] = [("关（只用 Jev）", None)]
    for confident in (0.70, 0.85):
        for policy in ("abstain", "deepseek"):
            configs.append(
                (
                    f"开 confident<{confident} {policy}",
                    SecondConfig(confident=confident, policy=policy),
                )
            )

    ok_rows = [r for r in rows if not r.error]
    avg_second_cost = (
        statistics.fmean([r.second_cost for r in ok_rows if r.second_verdict])
        if any(r.second_verdict for r in ok_rows)
        else 0.0
    )

    print()
    print("=" * 96)
    print("二次判定开 / 关对比（全部离线重算，不调用 API）")
    print("=" * 96)
    print(
        f"{'配置':<24}{'准确率':>8}{'致命':>7}{'反向致命':>10}"
        f"{'弃权':>7}{'复核数':>7}{'成本':>11}{'变化':>8}"
    )
    print("-" * 96)

    base = evaluate(rows, margin, irr_min, abstain, None)
    for label, cfg in configs:
        m = base if cfg is None else evaluate(rows, margin, irr_min, abstain, cfg)
        if cfg is None:
            triggers, cost, delta = 0, base["cost_total"], ""
        else:
            triggers = sum(1 for r in ok_rows if r.triggers_second(margin, irr_min, abstain, cfg))
            cost = m["cost_total"] + triggers * avg_second_cost
            delta = f"{(m['accuracy'] - base['accuracy']) * 100:+.1f}pp"
        print(
            f"{label:<24}{m['accuracy'] * 100:>7.1f}%{m['fatal_rate'] * 100:>6.1f}%"
            f"{m['reverse_fatal_rate'] * 100:>9.1f}%"
            f"{len(m['abstained']):>7}{triggers:>7}"
            f"{'¥' + format(cost * config.USD_TO_CNY, '.4f'):>11}{delta:>8}"
        )

    # ⚠️ 这两列必须一起看：只盯「致命」（该 no 却 yes）会漏掉更常见的「反向致命」
    # （该 yes 却 no）。实测 布偶 上开复核后致命掉到 0.0%，反向致命却仍有 14.3%。
    print(f"  （致命 = 该 no 却 yes，目标 <2%；反向致命 = 该 yes 却 no，目标 <5%。两列都要看。）")

    # 最有用的部分：到底改了哪几题、改对还是改错
    default = SecondConfig()
    changed = [
        (r, r.actual(margin, irr_min, abstain), r.actual(margin, irr_min, abstain, default))
        for r in ok_rows
        if r.triggers_second(margin, irr_min, abstain, default)
        and r.actual(margin, irr_min, abstain) != r.actual(margin, irr_min, abstain, default)
    ]
    print("-" * 96)
    if not changed:
        print(f"默认配置（confident<{default.confident} / {default.policy}）下，终判没有任何改变。")
        print("→ 说明触发复核的都是 Jev 已经有明确答案的题，开了也白开。")
    else:
        print(f"默认配置（confident<{default.confident} / {default.policy}）实际改变了 {len(changed)} 题：")
        for r, before, after in changed:
            good = "改对" if after == r.expected else ("改错" if before == r.expected else "仍错")
            print(
                f"  [{good}] {VERDICT_ZH.get(before, before)} → {VERDICT_ZH.get(after, after)}"
                f"（期望 {VERDICT_ZH.get(r.expected, r.expected)}）"
                f"  DeepSeek={VERDICT_ZH.get(r.second_verdict, r.second_verdict)}"
                f"@{r.second_confidence:.2f}"
            )
            print(f"          问：{r.question}")
            if r.note:
                print(f"          注：{r.note}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="用人工标注集评测海龟汤判定层（§9.2 指标）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--labels", default=str(ROOT / "eval" / "labels.csv"))
    parser.add_argument("--soup", default=None, help="只跑这条汤的标注")
    parser.add_argument("--margin", type=float, default=None, help="|P(是)-P(不是)| ≤ 此值 ⇒ 是也不是")
    parser.add_argument("--irrelevant-min", type=float, default=None, help="P(不重要) ≥ 此值 ⇒ 不重要")
    parser.add_argument("--abstain", type=float, default=None, help="最大概率 < 此值 ⇒ 弃权")
    parser.add_argument("--sweep", action="store_true", help="打印阈值扫描表")
    parser.add_argument("--reuse", default=None, help="复用缓存结果，不调用 API")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 n 题")
    parser.add_argument(
        "--second-opinion",
        action="store_true",
        help="对每道题额外取一次 DeepSeek 意见并缓存（有成本，但之后的所有对比都免费）",
    )
    parser.add_argument(
        "--second-options",
        default="four",
        choices=("three", "four"),
        help="复核后端的选项集：four（默认，多一个 both，能确认主判定的 both）| three",
    )
    parser.add_argument(
        "--second-soften",
        action=argparse.BooleanOptionalAction,
        default=config.SECOND_SOFTEN,
        help="二次说 both 时，是否把主判定的明确答案也软化成 both",
    )
    parser.add_argument(
        "--second-both-override",
        type=float,
        default=config.SECOND_BOTH_OVERRIDE,
        help="主判定是 both 时，二次要多久自信才能推翻它（>1 = 绝不推翻）",
    )
    parser.add_argument(
        "--compare-second",
        action="store_true",
        help="打印二次判定开 / 关对比（需要缓存里有二次判定数据，离线重算不花钱）",
    )
    parser.add_argument(
        "--second-confident",
        type=float,
        default=config.JEV_CONFIDENT,
        help="主判定最大概率低于此值就复核（默认同 JEV_CONFIDENT）",
    )
    parser.add_argument(
        "--second-policy",
        default=config.SECOND_OPINION_POLICY,
        choices=("abstain", "deepseek"),
        help="两边不一致时：abstain 保守弃权 | deepseek 让它拍板",
    )
    parser.add_argument(
        "--second-min-confidence",
        type=float,
        default=config.SECOND_CONFIDENT,
        help="主判定弃权时，二次意见要达到这个把握才采用",
    )
    args = parser.parse_args()

    margin = config.JEV_BOTH_MARGIN if args.margin is None else args.margin
    irr_min = config.JEV_IRRELEVANT_MIN if args.irrelevant_min is None else args.irrelevant_min
    abstain = config.JEV_ABSTAIN if args.abstain is None else args.abstain

    second = SecondConfig(
        confident=args.second_confident,
        policy=args.second_policy,
        min_confidence=args.second_min_confidence,
        soften=args.second_soften,
        both_override=args.second_both_override,
    )

    if args.reuse:
        rows = load_cache(Path(args.reuse))
        print(f"复用缓存：{args.reuse}（{len(rows)} 题，未调用 API）")
        cache_path = Path(args.reuse)
    else:
        soups = load_soups()
        labels = load_labels(Path(args.labels))
        if args.soup:
            labels = [lb for lb in labels if lb.soup_id == args.soup]
        if args.limit:
            labels = labels[: args.limit]
        if not labels:
            raise SystemExit("没有可跑的标注题。检查 eval/labels.csv 的 expected_verdict 列是否为空。")

        print(f"transport = {config.JEV_TRANSPORT}   model = {config.JEV_MODEL}")
        print(f"汤库      = {sorted(soups) or '（空）'}")
        print(f"标注      = {args.labels}（{len(labels)} 题）")
        print("-" * 96)
        rows = run_labels(
            labels, soups, fetch_second=args.second_opinion, second_options=args.second_options
        )
        cache_path = save_cache(rows)
        print(f"结果已缓存到：{cache_path}")

    has_second = any(r.second_verdict for r in rows)
    applied = second if has_second else None
    m = evaluate(rows, margin, irr_min, abstain, applied)
    print_report(m, rows, margin, irr_min, abstain, applied)
    if m["second_errors"]:
        print(f"\n⚠️  二次判定有 {len(m['second_errors'])} 题失败（已退回主判定）：")
        for r in m["second_errors"][:3]:
            print(f"   {r.question} → {r.second_error[:80]}")
    if args.sweep:
        print_sweep(rows)
    if args.compare_second:
        print_second_compare(rows, margin, irr_min, abstain)
    if m["second_calls"] and m["cost_total"]:
        total = m["cost_total"] + sum(
            r.second_cost for r in rows if r.second_verdict
        )
        print(
            f"\n复用一次 Jev 调用 ≈ ¥{statistics.fmean([r.cost for r in rows if r.cost]) * config.USD_TO_CNY:.6f}，"
            f"一次 DeepSeek 复核 ≈ ¥{statistics.fmean([r.second_cost for r in rows if r.second_verdict]) * config.USD_TO_CNY:.6f}"
        )
        print(f"本批总成本 ≈ ${total:.6f}（约 ¥{total * config.USD_TO_CNY:.4f}）")
    print(f"（想换阈值重算而不重新调用：--reuse \"{cache_path}\" --sweep --compare-second）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
