"""题库读写（§6.1）。

**只有一个题库文件：`data/soup.json`。** 界面里的新增 / 导入 / 修改 / 删除，
动的都是它 —— 内置的那 30 条也在这个文件里，所以它们同样能改能删。

（早先分过 mine.json / tags.json / 手写 .txt 三处，结果是「哪些能改」得靠来源判断，
到处是分支。现在没有这个区分了。手写 .txt 那种格式还在，但它的角色变成了
**导入格式** —— 通过题库界面导入进来落到 soup.json，见 soup/library.py。）

编码统一在这里处理 —— 记事本、PowerShell 写中文 JSON 时很常加 BOM，
这不该让使用者去排查。
"""

from __future__ import annotations

import json
from pathlib import Path

import config
from soup.models import Soup


def load_soup_file(path: Path) -> list[Soup]:
    """读 soup.json（支持只有一个对象或一组）。

    单独抽出来是为了让测试能指到临时文件；正常运行走 load_soups()。
    """
    text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        raise SystemExit(f"{path} 是空文件。放一条汤进去，或写成 [] 表示暂时没有汤。")
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SystemExit(
            f"{path} 不是合法 JSON：{exc}\n"
            "常见原因：多余的逗号、用了英文引号但没转义、括号没闭合。\n"
            "不想跟 JSON 打交道的话，把汤写成纯文本再从题库界面导入"
            f"（格式见 {config.SOUP_EXAMPLE_TXT.name}）。"
        ) from exc

    items = raw if isinstance(raw, list) else [raw]
    soups: list[Soup] = []
    for item in items:
        try:
            soups.append(Soup(**item))
        except Exception as exc:
            raise SystemExit(
                f"{path} 里的汤不符合格式：{exc}\n"
                "必需字段：id / title / surface / truth（tags 可选）。"
            ) from exc
    return soups


def load_soups() -> dict[str, Soup]:
    """整个题库，按 id 建索引。"""
    return {s.id: s for s in load_soup_file(config.SOUP_FILE)}


def _sort_key(soup: Soup) -> tuple[int, int, str]:
    """编号是数字的按数字排（否则 10 会排在 9 前面），不是数字的垫后面。"""
    if soup.id.isdigit():
        return (0, int(soup.id), "")
    return (1, 0, soup.id)


def save_soups(soups: list[Soup]) -> None:
    """写回 soup.json。按编号排序，文件顺序和界面顺序就一致。"""
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    payload = [s.model_dump() for s in sorted(soups, key=_sort_key)]
    config.SOUP_FILE.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def next_free_id(soups: dict[str, Soup]) -> str:
    """下一个没被占用的编号（**最小的空位**），补成 8 位。

    用最小空位而不是「最大值 +1」：删掉中间某条之后，新加的汤会补上那个空号，
    编号不会一路往上涨。
    """
    used = {int(k) for k in soups if k.isdigit()}
    n = 1
    while n in used:
        n += 1
    if n > config.MAX_ID_NUMBER:
        raise SystemExit(f"编号用完了（最多到 {config.MAX_ID_NUMBER}）")
    return f"{n:0{config.ID_WIDTH}d}"
