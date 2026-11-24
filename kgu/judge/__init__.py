from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from kgu.graph import BiTemporalGraph
from kgu.localize import Localization
from kgu.types import Op


@dataclass
class JudgeContext:
    text: str
    explicit_ops: list[Op]
    loc: Localization
    graph: BiTemporalGraph
    at: int
    relations: list[str]
    # Khoanh vùng đệ quy (Thuật toán alg:localize): `explicit_ops` là TẬP THAY ĐỔI ĐANG XÉT O_{k-1} của lần gọi `call`
    # (lần 1 = thay đổi trực tiếp; lần k >= 2 = thay đổi gián tiếp vừa ghi ở lần k-1). `direct_ops` luôn là thay đổi
    # trực tiếp gốc của bản tin (None = chính là explicit_ops, tức lần gọi 1).
    direct_ops: list[Op] | None = None
    call: int = 1


class Judge(Protocol):
    def judge(self, ctx: JudgeContext) -> list[Op]: ...
