"""每次请求带进来的 API 凭据（§5.1）。

**这个项目里唯一的 key 来源就是浏览器。** 服务端不存 key、不读 .env 里的 key、
也没有任何兜底：

    key 存在浏览器 localStorage
      → 每次请求用请求头带给后端
        → 后端只在这一局的存活期内放在内存里
          → 上游调用时放进 Authorization 头

好处不只是「开源友好」：
  - 服务端**不持久化**任何凭据，部署到公网也不存在「服务器上存着谁的 key」
  - 每个用户花自己的额度，不存在「别人用我的 key 刷爆账单」这件事
    —— 因为服务器上压根没有可刷的 key
  - 不用再操心「开关忘了关」「.env 忘了清」这类静默失误（以前真有这两种）

⚠️ 安全边界（README 里也要写）：
  - key 存在**浏览器 localStorage**，所以只该在自己信任的机器上用这个界面
  - 请求会带着你的 key 经过**服务端**再转发给上游 —— 所以自建/自用没问题，
    但**不要把别人的服务当代理填自己的 key**，那边的运维能看到它
  - 真要部署到公网，**必须套 HTTPS**，否则 key 在网络上明文传输
  - 服务端日志、错误响应里**绝不能**回显 key —— 见 redact()。

    这条约定**验证过**（不是「看代码觉得没问题」）：用一个独一无二的哨兵字符串
    当 key 走完建局/提问/提示/通关判定，再把项目目录、`logs/` 下的回合日志、
    服务进程的 stdout/stderr、以及返回给浏览器的每个响应体全扫一遍，一个字符都没有。
    要复现的话照上面那句话做一遍即可 —— 关键是**别只 grep 源码**，
    真正容易漏的是日志和异常路径。
"""

from __future__ import annotations

from dataclasses import dataclass

import config

# 请求头名字。**统一小写**（HTTP 头不区分大小写，Starlette 取出来也是小写）。
H_TRANSPORT = "x-jev-transport"
H_JEV_KEY = "x-jev-key"
H_DEEPSEEK_KEY = "x-deepseek-key"


def redact(secret: str | None) -> str:
    """日志/报错里用的脱敏形式。只留前 6 位够认出「是哪个 key」，其余打掉。"""
    if not secret:
        return "(未设置)"
    if len(secret) <= 10:
        return "***"
    return f"{secret[:6]}…{len(secret)}位"


@dataclass(frozen=True)
class Credentials:
    """一次请求要用的全部凭据。

    frozen 是有意的：这东西会被放进 Game 对象、跨线程读，改来改去容易出鬼。

    ⚠️ 这里**没有回落**。`jev_key` 为空就是为空 —— 上层会明确报「去设置里填」，
    而不是偷偷去别处找一把。以前有过「为空就用 .env」，那是
    「换了账号却还在花旧账号的钱」这类怪事的来源，已整个删除。
    """

    transport: str = config.DEFAULT_TRANSPORT
    jev_key: str = ""
    deepseek_key: str = ""

    @property
    def jev_endpoint(self) -> str:
        return config.resolve_jev(self.transport)[0]

    @property
    def jev_model(self) -> str:
        # 「默认 transport」才吃 JEV_MODEL 覆盖；选了别的就按那套走
        if self.transport == config.DEFAULT_TRANSPORT:
            return config.JEV_MODEL
        return config.resolve_jev(self.transport)[1]

    @property
    def jev_api_key(self) -> str:
        return self.jev_key

    @property
    def deepseek_api_key(self) -> str:
        return self.deepseek_key

    @property
    def has_jev(self) -> bool:
        return bool(self.jev_key)

    @property
    def has_deepseek(self) -> bool:
        return bool(self.deepseek_key)

    def safe_repr(self) -> str:
        """给日志用 —— 只出现 transport 和 key 长度，永远不含 key 本身。"""
        return (
            f"Credentials(transport={self.transport}, "
            f"jev={redact(self.jev_key)}, deepseek={redact(self.deepseek_key)})"
        )


def from_headers(headers) -> Credentials:
    """从请求头构造。**没有 key 就是没有 key**，不去别处找。

    `headers` 传 Starlette 的 `request.headers`（大小写不敏感）即可。
    """
    transport = (headers.get(H_TRANSPORT) or config.DEFAULT_TRANSPORT).lower()
    if transport not in config.JEV_TRANSPORTS:
        transport = config.DEFAULT_TRANSPORT
    return Credentials(
        transport=transport,
        jev_key=(headers.get(H_JEV_KEY) or "").strip(),
        deepseek_key=(headers.get(H_DEEPSEEK_KEY) or "").strip(),
    )


def settings_public() -> dict:
    """给界面的「设置」对话框用。**只发标签和地址，永远不发 key 本身。**

    ⚠️ 这里刻意**不报**「服务器上有没有 key」—— 因为服务器上确实没有，
    报了反而会让界面有机会写出「可以留空」这种误导（以前就这样错过：
    界面说可以留空、后端却因为开关关掉而 401）。
    """
    return {
        "transports": [
            {
                "name": name,
                "label": conf["label"],
                "model": conf["model"],
                # ⚠️ key_hint 是**提示格式**（比如 "sk-or-v1-…"），不是真 key。
                #    任何真实凭据都不该出现在这个响应里。
                "key_hint": conf["key_hint"],
                "keys_url": conf["keys_url"],
                "note": conf["note"],
            }
            for name, conf in config.JEV_TRANSPORTS.items()
        ],
        "default_transport": config.DEFAULT_TRANSPORT,
        # 明确告诉界面：key 只能由你提供。界面靠这个把文案写成「必须填」。
        "keys_are_user_supplied": True,
    }
