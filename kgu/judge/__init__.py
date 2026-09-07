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


class Judge(Protocol):
    def judge(self, ctx: JudgeContext) -> list[Op]: ...
