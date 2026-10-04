"""花钱接口的限流。

为什么需要：`/ask`、`/hint`、`/solve` 每次调用都真金白银。本地自用没关系，
但部署到公网之后，一个几十行的脚本就能把这些接口刷成你的账单 ——
实测一局 50 问约 ¥0.011，串行受 700ms 延迟限制约 ¥20/天，
并发几十路就是上千元级别。

设计取舍：
  - **不用滑动窗口列表**。问一句记一个时间戳，得为每个身份留一份历史，
    还得定期清理，不然是内存泄漏。改成**令牌桶**：一个计数 + 上次补充时间，
    两个浮点数，O(1) 内存，也不用清理。
  - **身份按「谁的 key」算，没带 key 才按 IP**。这样 A 刷爆自己的额度不会
    把 B 一起关在门外 —— 按 IP 一把抓的话，同一个 NAT 后面的两个人会互相伤害。
  - **默认开，但给得很宽**（见 config.RATE_LIMIT_PER_MIN）。人打字最猛也就
    每分钟十几问；60 足够单人玩得毫无感觉，又能把「脚本刷」从上千元压到几十元。
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass


@dataclass
class _Bucket:
    """一个身份的令牌桶。"""

    tokens: float
    updated: float


class Limiter:
    """按身份限流。线程安全（FastAPI 的同步端点跑在线程池里）。"""

    def __init__(self, per_minute: int = 0, burst: int | None = None) -> None:
        # per_minute <= 0 = 关闭限流
        self.per_minute = max(0, per_minute)
        # 允许一口气连发多少。默认 = 一分钟的量：玩家想快速追问几句时不该被绊住，
        # 而脚本要的就是「一分钟几千次」，这个上限对它来说依然是硬墙。
        self.burst = float(burst if burst is not None else self.per_minute)
        self._buckets: dict[str, _Bucket] = {}
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self.per_minute > 0

    def check(self, identity: str) -> tuple[bool, float]:
        """扣一个令牌。返回 (是否放行, 还要等几秒)。

        放行时第二个值无意义；拒绝时是「大概还要等这么久」，用于 Retry-After。
        """
        if not self.enabled:
            return True, 0.0

        now = time.monotonic()
        rate = self.per_minute / 60.0
        with self._lock:
            bucket = self._buckets.get(identity)
            if bucket is None:
                bucket = _Bucket(tokens=self.burst, updated=now)
                self._buckets[identity] = bucket

            # 按流逝时间补充，上限 burst
            bucket.tokens = min(self.burst, bucket.tokens + (now - bucket.updated) * rate)
            bucket.updated = now

            if bucket.tokens >= 1.0:
                bucket.tokens -= 1.0
                return True, 0.0

            missing = 1.0 - bucket.tokens
            return False, missing / rate if rate > 0 else 60.0

    def prune(self, idle_seconds: float = 3600.0) -> int:
        """丢掉长时间没动的桶 —— 公网上每个陌生 IP 都会留下一个桶。

        不做这一步的话，字典会随着扫过来的 IP 数无限长下去，
        而限流本身是为了防刷，自己却成了内存放大器就本末倒置了。
        """
        cutoff = time.monotonic() - idle_seconds
        with self._lock:
            stale = [k for k, v in self._buckets.items() if v.updated < cutoff]
            for key in stale:
                del self._buckets[key]
            return len(stale)

    def tracked(self) -> int:
        with self._lock:
            return len(self._buckets)
