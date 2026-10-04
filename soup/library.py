"""题库的**导入格式** —— 纯文本，给玩家用。

JSON 那套（data/soup.json）是程序自己管的存储格式，不作为面向玩家的格式：
引号、逗号、括号错一个就整个文件报废，汤底换行还要手写 `\\n`。纯文本没这些问题。

    # 电梯
    我走进电梯准备去上学，随着电梯的上升，我知道，我再也无法去学校了。
    ---
    星期一早上，在妈妈的催促之下我心不在焉的走进电梯……

规则只有两条：

    1. `# 标题` 开始一条汤（标题就是一整行，别的什么都不用管）
    2. 一条汤里**第一个** `---`（三个以上减号，单独一行）之前是汤面，之后是汤底

再遇到 `# 标题` 就是下一条汤。

为什么是这个格式：不用转义任何东西（中文标点、引号、破折号、空行都随便写），
记事本直接写，多段落天然支持，看一眼就知道怎么改，而且和圈子分享海龟汤的排版一致。

关于标签：**导入不带标签** —— 导进来的汤默认没有标签，之后在题库界面里加就行。
省掉这一项，格式就从「三条规则」变成「两条」，少一条规则就少一处会写错的地方。

关于编号：id 一律是 8 位数字（`00000001` 起），但**格式里不写编号** ——
让玩家手写机器字段正好违反了「格式要简单」这条。编号由程序分配，见 store.next_free_id。

⚠️ 已知限制：汤面或汤底里不要有**单独一行以 `#` 开头**，那会被当成新的一条汤；
    同理不要有单独一行的三个以上减号。中文汤里基本碰不到。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from soup.models import Soup

# `# ` 开头 = 新的一条汤。`#` 后面没空格也认（「#电梯」这种写法很常见）
_TITLE_RE = re.compile(r"^#\s*(?P<title>.*)$")
# 分隔线：三个以上减号、单独一行。写 `----` 也认（人常这么写）
_RULE_RE = re.compile(r"^-{3,}$")


@dataclass
class ParsedSoup:
    """解析出来的一条汤 —— **还没有编号，也没有标签**。

    编号是全局资源（要躲开已经存在的条目），得由调用方拿当前占用情况去分配，
    见 store.next_free_id()。让 parse() 凭空造一个编号，就会出现两条汤抢同一个号。
    """

    title: str
    surface: str
    truth: str

    def to_soup(self, soup_id: str) -> Soup:
            """分配好编号之后变成一条正式的汤。**导入没有标签**，见模块顶部说明。"""
            return Soup(
                id=soup_id,
                title=self.title,
                surface=self.surface,
                truth=self.truth,
        )


@dataclass
class Problem:
    """一条汤没解析成功的原因。带上第几条和标题，错误信息才指得清地方。"""

    index: int  # 第几条（1-based）；0 表示整个文件的问题
    title: str
    message: str

    def as_dict(self) -> dict:
        return {"index": self.index, "title": self.title, "message": self.message}


@dataclass
class ParseResult:
    """⚠️ 刻意不做「一条坏就全不许导入」—— 29 条好的 + 1 条坏的，应该进 29 条并
    告诉你第 30 条哪儿不对，而不是让你对着一整个文件排查。"""

    soups: list[ParsedSoup] = field(default_factory=list)
    problems: list[Problem] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems

    def as_dict(self) -> dict:
        return {
            "count": len(self.soups),
            "titles": [s.title for s in self.soups],
            "problems": [p.as_dict() for p in self.problems],
        }


def parse(text: str) -> ParseResult:
    """纯文本 → 汤列表（还没有编号）。坏的那条单独报告，不拖累好的。"""
    result = ParseResult()

    title = ""
    surface: list[str] = []
    truth: list[str] = []
    in_truth = False
    started = False  # 是否已经遇到过 `# 标题`
    index = 0  # 当前是第几条（1-based）
    leading: list[int] = []  # 首个 `#` 之前的非空行号

    def flush() -> None:
        """收尾上一条汤 —— 有毛病就记进 problems，没毛病就进 soups。"""
        if not started:
            return
        surface_text = "\n".join(s.strip() for s in surface).strip()
        truth_text = "\n".join(s.strip() for s in truth).strip()

        if not title:
            result.problems.append(Problem(index, "(无标题)", "`#` 后面没有写标题"))
        elif not surface_text:
            result.problems.append(Problem(index, title, "汤面是空的（`# 标题` 的下一行开始写汤面）"))
        elif not in_truth:
            result.problems.append(
                Problem(index, title, "少了 `---` 分隔线 —— 汤面和汤底之间要单独写一行 `---`")
            )
        elif not truth_text:
            result.problems.append(Problem(index, title, "`---` 之后是空的，汤底没写"))
        else:
            result.soups.append(ParsedSoup(title=title, surface=surface_text, truth=truth_text))

    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.rstrip()
        match = _TITLE_RE.match(line)
        if match:
            flush()
            index += 1
            started = True
            title = match.group("title").strip()
            surface, truth, in_truth = [], [], False
            continue
        if not started:
            if line.strip():
                leading.append(lineno)
            continue
        if not in_truth and _RULE_RE.match(line.strip()):
            in_truth = True
            continue
        (truth if in_truth else surface).append(line)

    flush()

    if leading:
        result.problems.insert(
            0,
            Problem(
                0,
                "(文件开头)",
                f"第 {leading[0]} 行开始有 {len(leading)} 行不属于任何一条汤 —— "
                "每一条汤都要以 `# 标题` 开头",
            ),
        )
    return result
