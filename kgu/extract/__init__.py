from __future__ import annotations

from typing import Protocol

from kgu.data import Example
from kgu.graph import BiTemporalGraph
from kgu.types import Op


class Extractor(Protocol):
    def extract(self, ex: Example, graph: BiTemporalGraph, at: int) -> list[Op]: ...
