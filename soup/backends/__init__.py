"""判定后端注册表（这里刻意只留一层抽象，见 soup/backends/base.py 的说明）。"""

from __future__ import annotations

import config
from soup.backends.base import JudgeBackend
from soup.creds import Credentials
from soup.backends.deepseek_direct import DeepSeekDirectBackend
from soup.backends.jev_direct import JevDirectBackend


def make_backend(creds: Credentials | None = None, name: str | None = None) -> JudgeBackend:
    """按配置构造判定后端；换后端不改下游。

    `creds` 来自本次请求（见 soup/creds.py）—— **key 只能从这里来**，
    没有 .env 回落。为 None 等于「没有 key」，会在真正调用时报一句
    「去设置里填一个」，而不是偷偷用一把别的。
    """
    name = (name or config.JUDGE_BACKEND).lower()
    if name == "jev":
        return JevDirectBackend.from_creds(creds)
    if name == "deepseek":
        return DeepSeekDirectBackend(api_key=creds.deepseek_api_key if creds else "")
    raise ValueError(f"未知的判定后端：{name}（可用：jev | deepseek）")


def make_second_backend(creds: Credentials | None = None) -> JudgeBackend | None:
    """构造二次判定后端；未开启或没配 key 时返回 None。

    用**四态**选项集：复核的价值一大半在于能**确认**一个 both ——
    Jev 侧的 both 是代码从 |P(是)−P(不是)| 派生的，DeepSeek 若只会答三态，
    就只能推翻它、永远没法印证它（见 soup/cascade.py 的 combine）。
    """
    key = creds.deepseek_api_key if creds else ""
    if not config.SECOND_OPINION or not key:
        return None
    return DeepSeekDirectBackend(api_key=key, options="four")


__all__ = [
    "Credentials",
    "DeepSeekDirectBackend",
    "JudgeBackend",
    "JevDirectBackend",
    "make_backend",
    "make_second_backend",
]
