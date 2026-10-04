"""FastAPI 入口（§8.1 / §3.1）。

链路：浏览器 → 本地规则层意图识别 → Jev 三态判定 → 代码派生四态 → 模板回答。

⚠️ **正常对局通路不返回汤底**（ask / 开局都只给汤面和回答）—— 发到浏览器就等于剧透。
唯一例外是 `POST /api/games/{id}/reveal`：玩家主动认输时才把汤底送过去，
且本局随即作废（后续 `ask` 返回 409），想重玩只能点「重置」。

对局状态现在放在内存里（`_games`）。单机单人够用，重启即清空；
持久化属 P2 的事。
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass, field

import config
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from soup import hinter, library, solver, store
from soup.backends.deepseek_direct import shared_client
from soup.backends.jev_direct import JevDirectBackend
from soup.creds import (
    H_DEEPSEEK_KEY,
    H_JEV_KEY,
    H_TRANSPORT,
    Credentials,
    from_headers,
    settings_public,
)
from soup.engine import GameEngine
from soup.models import Soup, Turn, Verdict, VerdictDistribution
from soup.nlu import ChoiceSyntaxError
from soup.ratelimit import Limiter

app = FastAPI(title="AI 海龟汤主持人", version="0.1.0")


# --- 对局状态（内存） -----------------------------------------------------------


@dataclass
class Game:
    id: str
    soup: Soup
    engine: GameEngine
    created: float = field(default_factory=time.time)
    lock: threading.Lock = field(default_factory=threading.Lock)
    # 揭晓汤底 / 通关之后本局就结束了：汤底已经给出去，再问下去没有意义
    revealed: bool = False
    solved: bool = False
    # 开这局时用的凭据。提示/通关判定要拿它去调 DeepSeek，而那两个端点
    # 单独请求时可能不带 header（前端只在提问时带），所以记在这里。
    # ⚠️ 只放内存，进程重启即丢 —— 绝不落盘、绝不进日志。
    creds: Credentials = field(default_factory=Credentials)


_games: dict[str, Game] = {}
_games_lock = threading.Lock()


def _new_game(soup: Soup, creds: Credentials) -> Game:
    game = Game(id=uuid.uuid4().hex[:12], soup=soup, engine=GameEngine(soup, creds=creds), creds=creds)
    with _games_lock:
        _games[game.id] = game
    return game


# --- 限流（见 soup/ratelimit.py、config.RATE_LIMIT_PER_MIN） --------------------
_rate_limiter = Limiter(config.RATE_LIMIT_PER_MIN)


def _identity(request: Request, creds: Credentials) -> str:
    """限流按谁算。

    ⚠️ **优先看 key，而不是 IP**。同一个 NAT 后面坐着两个人是常态，
    按 IP 算的话 A 刷爆会把 B 一起关在门外，而他们本来各花各的钱。
    只有没带 key（也就是所有人共用服务器那把 key）时，才退回按 IP 分组。
    """
    if creds.jev_key:
        return f"key:{creds.jev_key}"
    host = request.client.host if request.client else "?"
    return f"ip:{host}"


def _spend(request: Request, creds: Credentials) -> None:
    """扣一次花钱调用的额度，超了就 429。"""
    if not _rate_limiter.enabled:
        return
    allowed, retry_after = _rate_limiter.check(_identity(request, creds))
    if allowed:
        return
    raise HTTPException(
        status_code=429,
        detail=(
            f"问得太快了（上限 {config.RATE_LIMIT_PER_MIN} 次/分钟）。"
            f"歇 {max(1, int(retry_after + 1))} 秒再问。"
        ),
        headers={"Retry-After": str(max(1, int(retry_after + 1)))},
    )


def _prune_forever(interval: float = 600.0) -> None:
    """定期丢掉闲置的令牌桶。

    不做的话，公网上每个陌生 IP 都会留下一个桶，字典会一直长 ——
    限流本身是为了防刷，自己变成内存放大器就本末倒置了。
    """
    while True:
        time.sleep(interval)
        _rate_limiter.prune()


if _rate_limiter.enabled:
    threading.Thread(target=_prune_forever, name="rate-limit-prune", daemon=True).start()


def _require_key(creds: Credentials) -> None:
    """没有 key 就当场拒绝。

    ⚠️ 这里**没有开关**，也不需要开关。key 只有一个来源（浏览器请求头，见
    soup/creds.py），服务端一个都不存 —— 所以「别人不填 key 也能用、花我的钱」
    这件事在结构上就不可能发生，不需要靠「记得把 ALLOW_SERVER_KEY 设成 0」
    这种人为约定来保证（以前就是那样，而「开关忘了关」和「.env 忘了清」
    两种失误都很安静）。
    """
    if not creds.has_jev:
        raise HTTPException(
            status_code=401,
            detail="还没填 API key。请点右上角「设置」，填你自己的 key。",
        )


def _get_game(game_id: str) -> Game:
    with _games_lock:
        game = _games.get(game_id)
    if game is None:
        raise HTTPException(status_code=404, detail="对局不存在（服务重启会清空）")
    return game


# --- 请求/响应模型 -------------------------------------------------------------


class NewGame(BaseModel):
    soup_id: str | None = None


class AskBody(BaseModel):
    question: str = Field(min_length=1, max_length=200)
    # 「编辑已问的问题」重发时带上：顶替第几问（1-based 的回合号）
    replace_at: int | None = None


class SolveBody(BaseModel):
    """玩家自己写下的真相猜测。"""

    guess: str = Field(min_length=1, max_length=config.SOLVE_MAX_CHARS)


class ChoiceResult(BaseModel):
    """选择题的结果（对应 soup/models.py 的 Turn.choice）。

    前端只在**求提示**时回传历史，带上它，提示才能看到「被杀」而不是笼统的「是」。
    """

    options: list[str] = Field(default_factory=list, max_length=8)
    picked: str | None = None
    probabilities: dict[str, float] = Field(default_factory=dict)


class HistoryItem(BaseModel):
    """前端本地存下来的一条问答。

    ⚠️ 存在的理由：问答历史只存在浏览器里（单人自用，不做服务端存储），
    刷新页面后对局是**新建**的，后端 `engine.history` 是空的 ——
    这时候求提示，模型会以为玩家一个问题都没问过。
    """

    question: str = Field(min_length=1, max_length=200)
    verdict: Verdict = Verdict.ABSTAIN
    choice: ChoiceResult | None = None


class HintBody(BaseModel):
    # 只在「本局后端历史为空」时当作兜底，不覆盖后端自己记的账
    history: list[HistoryItem] = Field(default_factory=list, max_length=200)


class SoupDraft(BaseModel):
    """题库界面里新增/编辑一条汤。"""

    title: str = Field(min_length=1, max_length=config.TITLE_MAX_CHARS)
    surface: str = Field(min_length=1, max_length=4000)
    truth: str = Field(min_length=1, max_length=4000)
    tags: list[str] = Field(default_factory=list, max_length=config.MAX_TAGS)


class ImportBody(BaseModel):
    """粘贴/上传的纯文本题库（格式见 soup/library.py）。"""

    text: str = Field(min_length=1, max_length=2_000_000)


def _clean_tags(tags: list[str]) -> list[str]:
    """去空、去重、限长。界面那边也做一遍，这边是最后一道 —— 导入文本可能带怪东西。"""
    out: list[str] = []
    for raw in tags:
        tag = str(raw).strip()[: config.TAG_MAX_CHARS]
        if tag and tag not in out and len(out) < config.MAX_TAGS:
            out.append(tag)
    return out


def _turns_from_history(items: list[HistoryItem]) -> list[Turn]:
    """把前端传来的历史还原成 Turn。

    只要 question / verdict / choice / turn 四个字段 —— hinter 就是这么用的，
    概率、延迟这些复原不出来也不影响给提示。
    """
    return [
        Turn(
            turn=i,
            question=item.question,
            distribution=VerdictDistribution({}),
            verdict=item.verdict,
            abstained=item.verdict is Verdict.ABSTAIN,
            choice=item.choice.model_dump() if item.choice else None,
        )
        for i, item in enumerate(items, start=1)
    ]


def soup_public(soup: Soup) -> dict:
    """给前端的汤信息 —— 只含汤面，不含汤底。

    ⚠️ 汤底在这里是**结构性**缺席的：题库界面靠这个接口拿列表，所以界面里
    根本拿不到答案，也就无从剧透。要看答案只有「揭晓汤底」那一条路。
    """
    return {
        "id": soup.id,
        "title": soup.title,
        "surface": soup.surface,
        "tags": soup.tags,
    }


def turn_public(turn: Turn, reply: str) -> dict:
    out = {
        "no": turn.turn,
        "question": turn.question,
        "verdict": turn.verdict.value,
        "reply": reply,
        "probabilities": turn.distribution.as_dict(),
        "latency_ms": turn.latency_ms,
        "flags": turn.flags,
    }
    if turn.choice is not None:
        # 选择题：候选与命中的 key 都要给前端 —— 开发模式要显示概率分布，
        # 而「选了哪个」光看回答文本（候选原文）区分不出是 c1 还是 c2。
        out["choice"] = turn.choice
    if turn.second:
        # ⚠️ 只给结论，**不给 reason** —— 复核的 reason 是带着汤底生成的，会剧透。
        #    实测例子：「游戏类型与角色身份和战友死因的汤底核心无关」，
        #    玩家多问几轮就能从这些提示里拼出汤底。
        out["primary_verdict"] = turn.primary_verdict.value if turn.primary_verdict else None
        out["second"] = {
            "verdict": turn.second.get("verdict"),
            "confidence": turn.second.get("confidence"),
            "outcome": turn.second.get("outcome"),
        }
    return out


# --- 回合日志（§6.2） -----------------------------------------------------------
# ⚠️ 与 §6.2 的差异：原文的 `scores: {holds, fails, relevant}` 已改成三路概率分布
#    （判定层改成三态 + 代码派生 both，见 soup/rules.py）。


def _log_turn(game: Game, turn: Turn, reply: str) -> None:
    record = {
        "ts": time.time(),
        "game_id": game.id,
        "soup_id": game.soup.id,
        "turn": turn.turn,
        "question": turn.question,
        "verdict": turn.verdict.value,
        "reply": reply,
        "probabilities": turn.distribution.as_dict(),
        "abstained": turn.abstained,
        "flags": turn.flags,
        # 选择题：候选 + 命中的 key + 每个候选的概率（对应关系靠 options 的次序）
        "choice": turn.choice,
        # 复核留痕：分清「主判定错了」还是「复核改错了」（reason 只进日志，不进前端）
        "primary_verdict": turn.primary_verdict.value if turn.primary_verdict else None,
        "second": turn.second,
        "usage": turn.usage,
        "latency_ms": turn.latency_ms,
    }
    try:
        config.LOG_DIR.mkdir(parents=True, exist_ok=True)
        with config.TURN_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        # §4.3：灰区样本是调阈值最好的素材，单独落一份
        if turn.flags:
            with config.GRAY_LOG.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        pass  # 日志写不进去不该让对局挂掉


# --- 接口 ---------------------------------------------------------------------


@app.get("/api/health")
def health() -> JSONResponse:
    """环境自检。**不报任何 key**，只报「key 从哪来」这类结构信息。

    ⚠️ 原来有 `jev_configured` / `deepseek_configured` / `allow_server_key` 三个字段，
    现在整个删掉了 —— 服务器上不再有 key，这三个值已经没有意义。
    留着它们反而会让人以为「服务器配了 key」是可能的。
    """
    return JSONResponse(
        {
            "status": "ok",
            "judge_backend": config.JUDGE_BACKEND,
            # key 的唯一来源。前端可以据此确认自己的请求头会被采用。
            "keys_from": "request-headers",
            "jev_transport": config.DEFAULT_TRANSPORT,
            "jev_endpoint": config.JEV_ENDPOINT,
            "jev_model": config.JEV_MODEL,
            "deepseek_model": config.DEEPSEEK_MODEL,
            "deepseek_peak_now": config.is_deepseek_peak(),
            "rate_limit_per_min": config.RATE_LIMIT_PER_MIN,
            "thresholds": {
                "both_margin": config.JEV_BOTH_MARGIN,
                "irrelevant_min": config.JEV_IRRELEVANT_MIN,
                "abstain": config.JEV_ABSTAIN,
            },
        }
    )


@app.get("/api/settings")
def get_settings() -> JSONResponse:
    """界面「设置」对话框要的信息。

    **只发 transport 的标签、模型名、申请地址，永远不发任何 key** ——
    连「服务器上有没有 key」都只是布尔值。见 soup/creds.py 的 settings_public()。
    """
    return JSONResponse(settings_public())


@app.post("/api/settings/check")
def check_settings(request: Request) -> JSONResponse:
    """拿请求头里的 key 实际打一次最小调用，验证填得对不对。

    为什么值得有这个端点：key 填错（复制漏了字符、选错 transport、账号没额度）
    产生的报错和「模型判定失败」长得像，玩家很难分清是哪一层的问题。
    这里用一次成本极低的真实调用把它区分开。

    ⚠️ **只测请求头里带进来的 key。** 这是唯一存在的东西 —— 服务端不存 key，
    所以既不会「用服务器的额度帮你测」，也不会给出误导性的「连接正常」。

    ⚠️ **两个 key 都要测。** Jev 和 DeepSeek 是两个平台的两把 key（不通用），
    只测 Jev 会给出一句「连接正常」，而玩家换个 DeepSeek key 填错了照样在
    「提示 / 揭示真相」时才炸 —— 那时报错和判定失败混在一起，最难查。
    实测踩过：换了 DeepSeek 账号、点「测试连接」显示正常，其实是旧的那把在生效。
    """
    transport = (request.headers.get(H_TRANSPORT) or config.DEFAULT_TRANSPORT).lower()
    if transport not in config.JEV_TRANSPORTS:
        transport = config.DEFAULT_TRANSPORT
    key = (request.headers.get(H_JEV_KEY) or "").strip()
    if not key:
        return JSONResponse(
            {
                "ok": False,
                "message": "还没填 key。这里测的是你填进去的那个，不会用服务器上配的。",
            },
            status_code=400,
        )

    creds = Credentials(transport=transport, jev_key=key)
    try:
        backend = JevDirectBackend.from_creds(creds)
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"ok": False, "message": f"构造判定后端失败：{exc}"}, status_code=400)

    ds_key = (request.headers.get(H_DEEPSEEK_KEY) or "").strip()

    # 两把 key **各自独立测**，最后再合并结论。
    #
    # 为什么不在一把失败时直接 return：它们分属两个平台，坏掉的原因常常是
    # 「换了个账号只更新了其中一个」。一次只报一个的话，玩家得点两次、修两轮才发现。
    # 分开测的代价只是多一次免费调用（列模型不花钱）。
    jev_error: str | None = None
    try:
        # 极简探针：内容本身无关紧要，只为验证「这个 key 能不能调到这个模型」
        result = backend.judge(
            "这是一句测试。", Soup(id="probe", title="探针", surface="探针", truth="探针")
        )
        jev_ms = result.latency_ms
    except Exception as exc:  # noqa: BLE001 — 上游异常按人话返回
        jev_error = f"{type(exc).__name__}: {exc}"
        jev_ms = None
    finally:
        backend.close()

    ds_ok: bool | None = None
    ds_error: str | None = None
    if ds_key:
        try:
            deepseek_probe(ds_key)
            ds_ok = True
        except Exception as exc:  # noqa: BLE001
            ds_ok = False
            ds_error = f"{type(exc).__name__}: {exc}"

    # 消息拼成人话：哪把好、哪把坏、没填的那把算「跳过」而不是失败
    parts: list[str] = []
    if jev_error is None:
        parts.append(f"Jev 正常（{creds.transport} · {creds.jev_model}）")
    else:
        parts.append(f"Jev 不行 —— {jev_error}")
    if ds_ok is True:
        parts.append("DeepSeek 正常")
    elif ds_ok is False:
        parts.append(f"DeepSeek 不行 —— {ds_error}")
    else:
        parts.append("DeepSeek 没填，跳过（提示 / 通关判定会用不了）")

    # 「没填 DeepSeek」不算失败：Jev 能用就还能玩是非题，只是少三个功能。
    ok = jev_error is None and ds_ok is not False
    payload: dict = {
        "ok": ok,
        "message": "；".join(parts),
        "jev_ok": jev_error is None,
        "deepseek_ok": ds_ok,
    }
    if jev_ms is not None:
        payload["latency_ms"] = jev_ms
    return JSONResponse(payload)


def deepseek_probe(api_key: str) -> None:
    """最小成本的 DeepSeek 连通性检查：拉一次模型列表。

    刻意**不用 chat 补全**：列模型是免费的，而且同样能区分
    「key 无效 / 账号没额度 / 网络不通」这三种情况。
    换 key 却点不出问题、直到玩到一半才炸的那类坑，就是靠这一步堵住的。
    """
    client = shared_client(api_key)
    client.models.list()


@app.get("/api/soups")
def list_soups() -> JSONResponse:
    """题库列表。**不含汤底** —— 题库界面靠这个接口，所以界面里根本没有答案可漏。"""
    soups = _load_bank()
    return JSONResponse({"soups": [soup_public(s) for s in sorted(soups.values(), key=store._sort_key)]})


def _load_bank() -> dict[str, Soup]:
    """带护栏地加载题库。

    store 在遇到写坏的题库文件时会 SystemExit（对人跑脚本来说这是对的 ——
    命令行里就该立刻停下并说清楚哪儿错了）。但在请求里抛出来会变成 500 +
    一大段 traceback，界面上什么都看不到。所以转成 409，把 loader 那句人话送出去。
    """
    try:
        return store.load_soups()
    except SystemExit as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None


@app.post("/api/soups")
def create_soup(body: SoupDraft) -> JSONResponse:
    """题库界面里手写一条新汤。"""
    title = body.title.strip()
    if not title:
        raise HTTPException(status_code=422, detail="标题不能为空")

    soups = _load_bank()
    soup = Soup(
        id=store.next_free_id(soups),
        title=title,
        surface=body.surface.strip(),
        truth=body.truth.strip(),
        tags=_clean_tags(body.tags),
    )
    soups[soup.id] = soup
    store.save_soups(list(soups.values()))
    return JSONResponse({"soup": soup_public(soup)})


# ⚠️ /import 必须声明在 /{soup_id} 之前 —— FastAPI 按声明顺序匹配，
#    反过来的话 "import" 会被当成一条汤的 id。
@app.post("/api/soups/import")
def import_soups(body: ImportBody) -> JSONResponse:
    """导入纯文本题库。坏的那几条挑出来报告，好的照常入库。

    **不做去重、不做覆盖** —— 导入就是都加进来。标题重名也没关系，那是两条不同的汤。
    编号一律新分配（纯文本格式里本来就不写编号），所以永远不会有 id 冲突。
    """
    result = library.parse(body.text)
    if not result.soups:
        return JSONResponse(
            {"imported": [], "problems": [p.as_dict() for p in result.problems]},
            status_code=422,
        )

    soups = _load_bank()
    imported: list[str] = []
    for parsed in result.soups:
        soup = parsed.to_soup(store.next_free_id(soups))
        soups[soup.id] = soup
        imported.append(soup.id)

    store.save_soups(list(soups.values()))
    return JSONResponse(
        {
            "imported": imported,
            "titles": [soups[i].title for i in imported],
            "problems": [p.as_dict() for p in result.problems],
        }
    )


@app.get("/api/soups/{soup_id}")
def get_soup(soup_id: str) -> JSONResponse:
    """取一条汤的**完整**内容（含汤底）。

    为什么需要它：题库列表刻意不含汤底（防剧透），但编辑一条汤得先看到原来写的汤底。
    它只在题库界面里用；游戏里想看答案走「揭晓汤底」，那条路会把本局封掉。
    """
    soups = _load_bank()
    soup = soups.get(soup_id)
    if soup is None:
        raise HTTPException(status_code=404, detail=f"没有这条汤：{soup_id}")
    return JSONResponse({"soup": soup.model_dump()})


@app.put("/api/soups/{soup_id}")
def update_soup(soup_id: str, body: SoupDraft) -> JSONResponse:
    """改一条汤（含标签）。题库里的每一条都能改 —— 不分内置还是导入的。"""
    soups = _load_bank()
    soup = soups.get(soup_id)
    if soup is None:
        raise HTTPException(status_code=404, detail=f"没有这条汤：{soup_id}")

    soups[soup_id] = soup.model_copy(
        update={
            "title": body.title.strip() or soup.title,
            "surface": body.surface.strip(),
            "truth": body.truth.strip(),
            "tags": _clean_tags(body.tags),
        }
    )
    store.save_soups(list(soups.values()))
    return JSONResponse({"soup": soup_public(soups[soup_id])})


@app.delete("/api/soups/{soup_id}")
def delete_soup(soup_id: str) -> JSONResponse:
    """删一条汤。题库里的每一条都能删。"""
    soups = _load_bank()
    if soup_id not in soups:
        raise HTTPException(status_code=404, detail=f"没有这条汤：{soup_id}")
    del soups[soup_id]
    store.save_soups(list(soups.values()))
    return JSONResponse({"deleted": soup_id})


@app.post("/api/games")
def create_game(body: NewGame, request: Request) -> JSONResponse:
    creds = from_headers(request.headers)
    _require_key(creds)
    soups = _load_bank()
    if not soups:
        raise HTTPException(
            status_code=409,
            detail="题库是空的。点右上角「题库」进去新增一条，或把纯文本题库导进来"
            f"（格式见 {config.SOUP_EXAMPLE_TXT.name}）。",
        )
    if body.soup_id:
        soup = soups.get(body.soup_id)
        if soup is None:
            raise HTTPException(status_code=404, detail=f"没有这条汤：{body.soup_id}")
    else:
        soup = next(iter(soups.values()))

    game = _new_game(soup, creds)
    return JSONResponse({"game_id": game.id, "soup": soup_public(game.soup), "turns": []})


@app.get("/api/games/{game_id}")
def get_game(game_id: str) -> JSONResponse:
    game = _get_game(game_id)
    return JSONResponse(
        {
            "game_id": game.id,
            "soup": soup_public(game.soup),
            "turns": [turn_public(t, game.engine.reply(t)) for t in game.engine.history],
        }
    )


@app.post("/api/games/{game_id}/ask")
def ask(game_id: str, request: Request, body: AskBody) -> JSONResponse:
    game = _get_game(game_id)
    # 开局时已经拦过一次，这里再拦一次是为了把不变量写在「花钱的调用」旁边：
    # 以后要是有人加了「不带 key 也能建局」的入口，不至于静默开始刷服务器额度。
    _require_key(game.creds)
    _spend(request, game.creds)
    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="问题不能为空")

    # 同一局串行 —— engine.history 不是线程安全的
    with game.lock:
        if game.revealed or game.solved:
            raise HTTPException(status_code=409, detail="本局已经结束。点「重置」重来。")
        try:
            turn = game.engine.ask(question, replace_at=body.replace_at)
            reply = game.engine.reply(turn)
        except ChoiceSyntaxError as exc:
            # 斜杠语法写坏了（比如选项太多）—— 是玩家能自己改的，当成「没问成」告诉他就行，
            # 不要降级成判定失败（502），那看着像后端坏了。
            return JSONResponse({"kind": "pending", "reply": str(exc), "question": question})
        except NotImplementedError as exc:
            # 求提示 / 提交答案 / 闲聊 —— P3 的内容。明确告知，不假装成功。
            return JSONResponse({"kind": "pending", "reply": str(exc), "question": question})
        except Exception as exc:  # 判定后端故障：不吞掉，如实返回
            return JSONResponse(
                {
                    "kind": "error",
                    "reply": f"判定失败：{_judge_error_text(exc)}",
                    "question": question,
                },
                status_code=502,
            )

    _log_turn(game, turn, reply)
    return JSONResponse({"kind": "verdict", **turn_public(turn, reply)})


@app.post("/api/games/{game_id}/hint")
def hint(game_id: str, request: Request, body: HintBody | None = None) -> JSONResponse:
    """根据已经问过的问题给一句方向性提示（§4.5）。

    ⚠️ 提示是**引导**，不能剧透 —— 约束写在 soup/hinter.py 的 prompt 里，
    那边特意要求「只说该往哪个方向问，不说真相是什么」。
    """
    game = _get_game(game_id)
    _require_key(game.creds)
    _spend(request, game.creds)
    with game.lock:
        if game.revealed or game.solved:
            raise HTTPException(status_code=409, detail="本局已经结束。点「重置」重来。")
        history = list(game.engine.history)
        if not history and body and body.history:
            # 刷新页面后接着玩：后端这局是新建的，用前端存下来的历史补上
            history = _turns_from_history(body.history)
        if not history:
            raise HTTPException(status_code=409, detail="先问几个问题吧，不然没什么可提示的。")
        try:
            result = hinter.generate(game.soup, history, game.creds)
        except Exception as exc:  # noqa: BLE001 — 失败如实返回，不吞掉
            return JSONResponse(
                {"kind": "error", "reply": f"提示生成失败：{_judge_error_text(exc)}"},
                status_code=502,
            )
        asked = len(history)

    return JSONResponse(
        {
            "kind": "hint",
            "hint": result.hint,
            "progress": round(result.progress, 2),
            "asked": asked,
            "latency_ms": result.latency_ms,
            "cost": result.usage.get("cost", 0.0),
        }
    )


def _judge_error_text(exc: Exception) -> str:
    """判定失败时给玩家看的话。

    ⚠️ 开源之后**最常见**的一次失败就是「key 填错了 / 填成另一家的 key / 没充值」，
    而上游回的是 `JevError: HTTP 401：{"error":{"message":"User not found.",...}}` ——
    既不像给人看的，也没说下一步该干什么。这里补一句指向「设置」的提示，
    原始文本照旧留在后面（出怪问题时还能查）。
    """
    raw = f"{type(exc).__name__}: {exc}"
    lowered = raw.lower()
    if any(k in lowered for k in ("401", "403", "unauthorized", "invalid_api_key", "user not found", "no available credits")):
        return (
            f"{raw}（上游拒绝了这把 key —— 去右上角「设置」检查一下："
            "是不是填成了另一家的 key，或者账户还没充值）"
        )
    return raw


@app.post("/api/games/{game_id}/solve")
def solve(game_id: str, request: Request, body: SolveBody) -> JSONResponse:
    """玩家提交自己写的真相，判定是否通关（§4.5b）。

    ⚠️ **只有通过了才把汤底发回去**（这时本局随之结束）。没通过只回「还差什么」，
    不给汤底 —— 否则剧透之后就没得猜了。
    """
    game = _get_game(game_id)
    _require_key(game.creds)
    _spend(request, game.creds)
    with game.lock:
        if game.revealed or game.solved:
            raise HTTPException(status_code=409, detail="本局已经结束。点「重置」重来。")
        try:
            result = solver.evaluate(body.guess, game.soup, game.creds)
        except Exception as exc:  # noqa: BLE001 — 判定失败如实返回，不吞掉
            return JSONResponse(
                {"kind": "error", "reply": f"判定失败：{_judge_error_text(exc)}"},
                status_code=502,
            )
        if result.passed:
            game.solved = True

    payload = {
        "kind": "solve",
        "passed": result.passed,
        "score": round(result.score, 2),
        "missing": result.missing,
        "comment": result.comment,
        "latency_ms": result.latency_ms,
        "cost": result.usage.get("cost", 0.0),
    }
    if result.passed:
        payload["truth"] = game.soup.truth
    return JSONResponse(payload)


@app.post("/api/games/{game_id}/reveal")
def reveal(game_id: str) -> JSONResponse:
    """揭晓汤底 —— 玩家主动放弃时才调。

    这是**唯一**会把汤底发给前端的接口。汤底一到浏览器就等于剧透，
    所以本局随即作废：后续 ask 会返回 409，想再玩只能点「重置」。
    """
    game = _get_game(game_id)
    with game.lock:
        game.revealed = True
    return JSONResponse({"truth": game.soup.truth, "revealed": True})


def _page(name: str, media_type: str | None = None) -> FileResponse:
    """返回一个页面/脚本，并让浏览器**每次都回头问一遍**，别自己拿缓存。

    ⚠️ 不加这个的话，浏览器会按自己的启发式规则直接用缓存里的旧文件 ——
    开发时改完代码、界面上却还是老行为，很容易误判成「代码写错了」
    （踩过一次：改完「局末只留一个按钮」，旧标签页里仍然并排显示两个「玩过的」）。

    no-cache 的含义是「用之前先问一下服务端」，不是「不缓存」。
    实测 Starlette 1.7 的 FileResponse 不处理 If-None-Match（照样回 200），
    所以每次会重传一遍 —— 本地单人用，这点开销无所谓，**不会拿到旧文件**才是重点。
    """
    return FileResponse(
        config.WEB_DIR / name,
        media_type=media_type,
        headers={"Cache-Control": "no-cache"},
    )


@app.get("/")
def index() -> FileResponse:
    return _page("index.html")


@app.get("/library")
def library_page() -> FileResponse:
    """题库界面。独立成页 —— 800 多行的播放页再塞管理功能就太挤了。"""
    return _page("library.html")


@app.get("/store.js")
def store_js() -> FileResponse:
    """两个页面共用的本地进度读写（见 web/store.js）。

    ⚠️ 必须给 media_type，不然浏览器可能按 text/plain 拒绝执行。
    """
    return _page("store.js", "application/javascript; charset=utf-8")
