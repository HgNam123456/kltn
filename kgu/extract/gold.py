from __future__ import annotations

from kgu.data.nba import Example
from kgu.graph import BiTemporalGraph
from kgu.ops import diff
from kgu.types import Op


class GoldAllExtractor:
    """Cận trên: toàn bộ diff trước/sau (không cần khoanh vùng)."""

    def extract(self, ex: Example, graph: BiTemporalGraph) -> list[Op]:
        return diff(ex.before, ex.after, source="gold_all")


class GoldExplicitExtractor:
    """Xấp xỉ ops 'được nói thẳng': cả 2 đầu cạnh đều là entity được nhắc trong text."""

    def extract(self, ex: Example, graph: BiTemporalGraph) -> list[Op]:
        mentioned = set(ex.mentioned)
        return [op for op in diff(ex.before, ex.after, source="gold_explicit")
                if op.h in mentioned and op.t in mentioned]
