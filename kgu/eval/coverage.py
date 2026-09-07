from __future__ import annotations

from kgu.data.nba import Example
from kgu.localize import Localization
from kgu.ops import diff
from kgu.types import Op, OpKind


def coverage(ex: Example, explicit_ops: list[Op], loc: Localization) -> tuple[int, int]:
    """(số op ngầm nằm trong vùng ứng viên, tổng số op ngầm). Đo recall của khoanh vùng."""
    implied = [op for op in diff(ex.before, ex.after) if op not in set(explicit_ops)]
    covered = 0
    for op in implied:
        if op.kind is OpKind.INVALIDATE and op.triple in loc.cut_triples:
            covered += 1
        elif op.kind is OpKind.ADD and frozenset((op.h, op.t)) in loc.add_pairs:
            covered += 1
    return covered, len(implied)
