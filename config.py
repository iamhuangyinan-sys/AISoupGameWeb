"""集中配置：读 .env、价格表、阈值。

**能在这里调的都在这里，每个取值的理由就写在它旁边。** 想改判定行为先看这个文件，
不要去改 soup/ 里的逻辑 —— 阈值和开关一律从这个模块取，散在各处就没人找得齐了。

⚠️ 这里**没有、也不该有任何 API key**（见 soup/creds.py）。
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
# override=True：让 .env 成为唯一权威配置源。
# 否则机器上遗留的同名环境变量会**静默压过** .env（load_dotenv 默认不覆盖），
# 排查时看到的现象是「.env 里明明是空的，程序却读到了 key」。
load_dotenv(BASE_DIR / ".env", override=True)


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _env_float(name: str, default: float) -> float:
    raw = _env(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool = True) -> bool:
    raw = _env(name)
    if not raw:
        return default
    return raw.lower() not in {"0", "false", "no", "off"}


# --- Jev 的两种 transport（唯一的抽象） -----------------------------------------
# openrouter：不用 TypeSafe 账号，OpenRouter 注册充值即可用。实测直连可达。
#             ⚠️ 但 DeepSeek 是另一家的 key（见下面 DEEPSEEK_* ），不通用。
# typesafe：官方直连。但必须先有 credits 才允许签发 key（无自动试用额度）。
#
# ⚠️ **这把表里没有 key，也不该有。** 所有 key 都由浏览器在请求头里带进来
#    （见 soup/creds.py），服务端一个都不存。这样开源之后不会出现
#    「clone 下来跑起来就有人花你的钱」这种事 —— 压根没有可花的东西。
JEV_TRANSPORT = _env("JEV_TRANSPORT", "openrouter").lower()

OPENROUTER_BASE_URL = _env("OPENROUTER_BASE_URL", "https://openrouter.ai/api")

# transport -> 该 transport 的一切。**加新 transport 只改这张表**，
# 前后端都不需要跟着改（端点 /api/settings 会把它发给界面）。
JEV_TRANSPORTS = {
    "openrouter": {
        "label": "OpenRouter",
        "endpoint": f"{OPENROUTER_BASE_URL}/v1/systemone",
        "model": "typesafe/jev-1.13",  # 固定版本号，不用 ~typesafe/jev-latest
        "key_hint": "sk-or-v1-…",
        "keys_url": "https://openrouter.ai/settings/keys",
        # ⚠️ 别再写「一个 key 同时能调 Jev 和 DeepSeek」—— 不成立。
        #    DeepSeek 走的是 config.DEEPSEEK_BASE_URL（api.deepseek.com）+ 单独的
        #    DeepSeek key，OpenRouter 的 key 在那边不认（实测：只填 Jev key 时
        #    /hint 会回「缺少 DeepSeek API key」）。写成「一个 key 搞定」会把人坑到 ——
        #    填完 key 以为没事了，一按「提示」才发现少了半套配置。
        "note": "注册充值即可用，不需要 TypeSafe 账号。注意 DeepSeek 是另一家的 key，要单独申请。",
    },
    "typesafe": {
        "label": "Typesafe 官方",
        "endpoint": "https://api.typesafe.ai/v1/systemone",
        "model": "jev-1.13.0",
        "key_hint": "你的 Typesafe API key",
        "keys_url": "https://platform.typesafe.ai/",
        "note": "官方直连。注意：需要先充值才允许签发 key，没有免费试用额度。",
    },
}
DEFAULT_TRANSPORT = JEV_TRANSPORT if JEV_TRANSPORT in JEV_TRANSPORTS else "openrouter"


def resolve_jev(transport: str | None) -> tuple[str, str]:
    """transport → (endpoint, model)。

    `transport` 认不出来就退回默认那个。**这里没有 key** —— key 是每次请求带进来的，
    见 soup/creds.py。以前这个函数还负责「没传 key 就用 .env 的」，
    那个回落就是「换了账号却还在花旧账号的钱」这类怪事的来源，已经整个删掉了。
    """
    name = (transport or DEFAULT_TRANSPORT).lower()
    conf = JEV_TRANSPORTS.get(name) or JEV_TRANSPORTS[DEFAULT_TRANSPORT]
    return conf["endpoint"], conf["model"]


_JEV_CONF = JEV_TRANSPORTS[DEFAULT_TRANSPORT]
JEV_ENDPOINT = _JEV_CONF["endpoint"]
JEV_MODEL = _env("JEV_MODEL", _JEV_CONF["model"])
# ⚠️ 留空**不等于**直连。留空时 httpx 会去读**系统代理**（Windows「设置 → 代理」
#    那个，存在注册表里，跟进程的 *_PROXY 环境变量无关）—— trust_env 默认就是 True，
#    而且**显式传 proxy=None 也拦不住**（实测过）。想绕开系统代理得设 NO_PROXY。
#
#    为什么保持这个行为：需要代理才能访问 api.typesafe.ai / openrouter.ai 的网络里，
#    系统代理开着就直接能用，不用再往 .env 里抄一遍地址。
#    这个变量只在你想**强制指定**一个代理（盖过系统设置）时才需要填。
#
#    排查提示：如果 Jev 调用一直卡到超时，先去 Windows 的代理设置看看 —— 系统代理
#    配成一个已经关掉的本地端口时，表现就是「全部超时」，而日志里什么都看不出来。
JEV_PROXY = _env("JEV_PROXY") or _env("TYPESAFE_PROXY")
JEV_TIMEOUT_S = _env_float("JEV_TIMEOUT_S", 30.0)

DEEPSEEK_MODEL = _env("DEEPSEEK_MODEL", "deepseek-flash")
DEEPSEEK_BASE_URL = _env("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_THINKING = _env("DEEPSEEK_THINKING", "disabled")
# 必须显式设上限。实测踩过（布偶 #27）：不设时偶发输出在 reason 中途断掉，
# JSON 解析失败会把整道题变成异常。1024 对「verdict + basis 引原话 + 一句 reason」绰绰有余。
DEEPSEEK_MAX_TOKENS = int(_env_float("DEEPSEEK_MAX_TOKENS", 1024))

# --- 凭据来自哪里（见 soup/creds.py）------------------------------------------
# **只有一处：浏览器请求头。** 服务端不存任何 key，也没有 .env 回落。
#
# 为什么把「.env 兜底 key」这套整个删掉（原来有个 ALLOW_SERVER_KEY 开关）：
#   1. 开源之后那把兜底 key 就是最危险的东西 —— 谁访问到你的服务就花你的钱，
#      而「开关忘了关」和「.env 忘了清」两种失误都不可见
#   2. 留着它，就得在 resolve_jev / 后端构造 / DeepSeek 三处都记得不回落，
#      漏一处等于没关（实测踩过：关掉开关但 .env 里还有 key，照样能用）
#   3. 现在没有可泄漏的服务器 key，README 里那一整节「分享前务必…」也不需要了
#
# 代价：clone 下来第一次打开必须去「设置」里填一次 key。这个代价值得。

# --- 花钱接口的限流（见 soup/ratelimit.py）-------------------------------------
# 上限是**每分钟多少次**「判定 / 提示 / 通关判定」调用，0 = 关闭。
#
# 为什么默认开：这三个接口每调一次都真金白银，公网部署时一个脚本就能刷成上千元/天。
# 为什么给 60 这么宽：人打字最猛也就一分钟十几问，60 对单人玩毫无感觉，
# 但足以把「脚本刷」从上千元压到几十元 —— 它防的是自动化，不是防人。
# 身份按「带的 key」算，没带 key 才按 IP，所以不会让同一个 NAT 后面的人互相伤害。
RATE_LIMIT_PER_MIN = int(_env_float("RATE_LIMIT_PER_MIN", 60))

# --- 通关判定 -----------------------------------------------------------------
# 玩家自述真相 → DeepSeek 打分，≥ 此值算通关。
# 偏松是刻意的：玩家的措辞必然和汤底不同，判太严会让人反复重写却一直被拒，
# 而愿意写完整猜测的人方向大概率是对的（判定细节见 soup/solver.py）。
SOLVE_PASS_SCORE = _env_float("SOLVE_PASS_SCORE", 0.75)
# 单次猜测的字数上限（前端 textarea 也按这个限）
SOLVE_MAX_CHARS = int(_env_float("SOLVE_MAX_CHARS", 1000))

# 判定后端可替换：jev | deepseek（对比数据支持时切换）
JUDGE_BACKEND = _env("JUDGE_BACKEND", "jev")

# --- 判定阈值 ------------------------------------------------------------------
# 判定层是三态：是 / 不是 / 不重要。**模型不再回答「是也不是」** ——
# 四选一里 both 会变成挡箭牌，模型把概率摊给它，yes/no 就都软了。
# 「是也不是」由代码从两边置信度派生：|P(是) − P(不是)| ≤ margin ⇒ 两边相当 ⇒ 是也不是。
# 让模型自己选 both 会被它当挡箭牌 —— 概率一摊，yes/no 就都软了。
#
#   max(三路) < ABSTAIN          → 弃权（模型自己也没主意）
#   P(不重要) ≥ IRRELEVANT_MIN   → 不重要
#   |P(是) − P(不是)| ≤ MARGIN   → 是，也不是   ← 代码派生
#   否则                          → 是 / 不是 里概率高的那个
#
# 取值必须用标注集扫描确定（见 eval/run_eval.py 的 --sweep）。
#
# 两锅汤的实测（同一个 margin 在两边表现完全不同 —— 这就是必须两锅一起看的理由）：
#
#   margin   布偶 准确率  致命     微笑 准确率  致命
#   0.10        51.7%       3.4%       85.0%       5.0%
#   0.15        55.2%       3.4%       85.0%       5.0%
#   0.20        55.2%       0.0%       85.0%       5.0%   ← 取这个
#   0.25        55.2%       0.0%       85.0%       5.0%
#   0.30        51.7%       0.0%       80.0%       5.0%
#
# 取 0.20：它是 布偶 的拐点（准确率 +3.5pp，致命错误 3.4% → 0%），而在 微笑 上
# 与 0.10 完全等价 —— 那批题里没有一道的 Δ 落在 0.10~0.20 之间。两锅汤同时不吃亏。
#
# ⚠️ 早先的教训：0.30 曾经能消掉 布偶 唯一一道致命错误，但让 微笑 掉 10 个百分点 ——
# 那是拿**一个样本拟合一个参数**。0.20 没重蹈覆辙，因为它是被 布偶 的拐点选中的，
# 不是被某一道题选中的。
#
# ⚠️ 「Δ 接近 ⇒ 是也不是」和「low_confidence」在这个区间里是同一件事，不需要额外分支：
# 布偶 上 Δ ≤ 0.10 / 0.20 / 0.30 的题分别是 1 / 5 / 8 道，**没有一道**不算 low_confidence。
# 也就是说「Δ 小」蕴含「模型自己也没把握」，把 margin 调大就等于把这类题交给「是也不是」。
# 三态（是 / 不是 / 不重要）→ 四态或弃权。取值由标注集扫描确定。
#
# 两锅汤 49 题（布偶 29 + 微笑 20）实测，配合二次判定（soften=0）：
#     margin   只用 Jev       +复核      致命   反向致命
#     0.10     65.3%         65.3%       2→0    13.0%
#     0.15     67.3%         69.4%       1→0    13.0%
#     0.20     67.3%         69.4%       1→0     8.7%   ← 选它
#     0.25     65.3%         67.3%       0      8.7%
#     0.30     61.2%         65.3%       0      8.7%
# 0.20 同时拿下最高准确率和最低反向致命；再往上边际收益消失、准确率反而掉。
# ⚠️ 注意默认值只是兜底 —— **.env 里的值优先**（config 用 load_dotenv(override=True)）。
JEV_BOTH_MARGIN = _env_float("JEV_BOTH_MARGIN", 0.20)
JEV_IRRELEVANT_MIN = _env_float("JEV_IRRELEVANT_MIN", 0.50)
JEV_ABSTAIN = _env_float("JEV_ABSTAIN", 0.45)
# 采用了答案但这个概率偏低 → 标 low_confidence，落日志供调阈值
JEV_CONFIDENT = _env_float("JEV_CONFIDENT", 0.70)

# --- 二次判定（后端 B 的第二种用法） --------------------------------------------
# Jev 弃权或 low_confidence 时，叫 DeepSeek 复核（见 soup/cascade.py）。
#
# 这同时把「可替换的后端 B」这条一直悬着的设想落地了。触发条件用 low_confidence，
# 而不是给 Jev 另设一个更严的阈值 —— 理由：Jev 的校准概率只在「干净二选一」时可靠
# （实测 0.98/0.04），一旦它开始动用汤底以外的知识（微笑 #7 的致命错误、
# 「植物大战僵尸是不是策略游戏」），它的概率分布就会变得不稳定。复核正好覆盖这一类。
SECOND_OPINION = _env_bool("SECOND_OPINION", True)
# abstain（默认）：两边不一致就弃权，不赌。
# deepseek：让 DeepSeek 拍板 —— 风险高，只有数据证明它更准时才用。
SECOND_OPINION_POLICY = _env("SECOND_OPINION_POLICY", "abstain").lower()
# 主判定弃权时，二次意见要达到这个把握才采用
SECOND_CONFIDENT = _env_float("SECOND_CONFIDENT", 0.70)
# 主判定给出明确答案、二次却说是「是也不是」时，是否跟着软化。
#
# ⚠️ 默认 **关**，这是两锅汤对照出来的（49 题，脚本在 eval/ 的离线扫描里）：
#     soften=ON   → 合计准确率 65.3%，其中 微笑 从 85.0% 崩到 75.0%
#     soften=OFF  → 合计准确率 69.4%，微笑 保持 85.0%，致命错误两边都是 0
#
# 原因是四态下的 DeepSeek 极爱说 both（布偶 15/29 道、微笑 9/20 道），
# 而 微笑 压根没有 both 标准答案 —— 那 9 道全是误报。
# 也就是说「转到复核的题大概率带 both 味道」这个先验**不成立**：
# 照它做等于拿一个已经对的明确答案，去换一个并不更准的 both。
#
# 关掉 softening 后，二次判定只在「改口成明确答案」时才起作用 —— 那才是它该干的事。
# 附带好处：复核门槛定多少都不影响结果了（0.70/0.80/0.85/0.90 实测完全一样），
# 所以可以放心用回 0.70，复核数从 29/49 降到 13/49，成本最低。
SECOND_SOFTEN = _env_bool("SECOND_SOFTEN", False)
# 主判定是 both 时，二次要推翻它、给出明确答案得多自信（>1 表示绝不推翻）。
# 实测这是唯一能把 both 换成明确答案的路径，翻对了 2 道（布偶 #5、#10）。
SECOND_BOTH_OVERRIDE = _env_float("SECOND_BOTH_OVERRIDE", 0.95)

# --- 选择题（斜杠语法） ---------------------------------------------------------
# 玩家在候选之间写 `/`，一次问一个选择题：「他是自杀/被杀」。
# 解析见 soup/nlu/choice.py，判定见 soup/rules.py 的 decide_choice()。
#
# 实现方式：本地切出候选文本，**一次** Jev Choice 让它在候选里挑一个。
# 没走「先让 DeepSeek 判是不是选择题」那条路 —— 那是三重亏损：
#   成本 +¥0.0004/问（比 Jev 本身还贵）、延迟 +1 次往返（P50 已经 700ms 超标）、
#   而斜杠语法让识别变成 0 成本且 100% 可靠的形式问题。
MAX_CHOICE_OPTIONS = int(_env("MAX_CHOICE_OPTIONS", "4"))
# N 选一的判据。⚠️ 这两个值**没有标注集校准过**（是非题那套阈值不能直接沿用 ——
# N 路分布下 Δ 的含义和四态不一样），先取保守值，宁可回「不确定」也别给错答案。
JEV_CHOICE_MIN = _env_float("JEV_CHOICE_MIN", 0.50)      # 最高项低于此 → 弃权
JEV_CHOICE_MARGIN = _env_float("JEV_CHOICE_MARGIN", 0.15)  # 与次高项的差距低于此 → 弃权

# --- 按需 NLU 开关 -------------------------------------------------------------
# 拆解（NLU_DECOMPOSE）已实测证伪并移除；指代消解尚未实现。
# 语言处理层目前有：本地正则意图识别 + 选择题斜杠语法解析，都是 0 成本。

# --- 题库导入限制 -------------------------------------------------------------

TRUTH_MAX_CHARS = 600  # 上限；超过仍可入库，但提醒作者压缩
TITLE_MAX_CHARS = 60
# 标签最多几个、单个最长几个字 —— 只在录入时截断，不报错
MAX_TAGS = 12
TAG_MAX_CHARS = 12

# --- 编号（id）----------------------------------------------------------------
# 汤的 id 一律是 8 位数字：00000001 … 99999999。新加的取**最小的空位**，
# 所以删掉中间某条之后新汤会补上那个空号，编号不会一路往上涨。
# 纯文本导入格式里不写编号（那是机器字段），由 store.next_free_id() 分配。
ID_WIDTH = 8
MAX_ID_NUMBER = 99_999_999

# --- 路径 ---------------------------------------------------------------------

DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR / "logs"
WEB_DIR = BASE_DIR / "web"
# 唯一的题库文件。界面里新增/导入/修改/删除，动的都是它。
SOUP_FILE = DATA_DIR / "soup.json"
# 格式参考（不会被加载）。纯文本那份是**导入格式**，见 soup/library.py
SOUP_EXAMPLE_TXT = DATA_DIR / "soup.example.txt"
TURN_LOG = LOG_DIR / "turns.jsonl"
GRAY_LOG = LOG_DIR / "gray.jsonl"

# --- 价格表（USD / 1M token） --------------------------------------------------

USD_TO_CNY = 7.2

JEV_PRICE_IN_PER_M = 0.042
JEV_PRICE_OUT_PER_M = 0.0  # 输出免费

DEEPSEEK_PRICES = {
    # model: (peak, offpeak); 每项为 (cache_hit_in, miss_in, out)
    "deepseek-flash": ((0.006, 0.30, 1.20), (0.003, 0.15, 0.60)),
    "deepseek-v4-pro": ((0.044, 1.32, 3.96), (0.022, 0.66, 1.98)),
}

# 高峰时段：UTC 01:00–04:00 与 06:00–10:00，周一至周五
DEEPSEEK_PEAK_UTC_RANGES = ((1, 4), (6, 10))


def is_deepseek_peak(now: datetime | None = None) -> bool:
    """判断当前是否处于 DeepSeek 高峰计费时段。

    高峰：UTC 01:00–04:00 与 06:00–10:00，周一至周五
    → 北京时间 9:00–12:00、14:00–18:00 工作日。
    其余时段（含晚上、周末、中国节假日）半价。
    """
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    utc = now.astimezone(timezone.utc)
    if utc.weekday() >= 5:  # 周六 / 周日
        return False
    return any(start <= utc.hour < end for start, end in DEEPSEEK_PEAK_UTC_RANGES)
