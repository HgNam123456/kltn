from __future__ import annotations

from kgu.data import Example
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


def coverage_by_call(ex: Example, explicit_ops: list[Op], locs: list[Localization]) -> list[tuple[Op, int]]:
    """Mỗi thay đổi gián tiếp vàng (diff trước/sau trừ thay đổi trực tiếp) kèm LẦN GỌI đầu tiên mà nó nằm trong vùng
    ứng viên: locs[k-1] là vùng ứng viên của lần gọi k (khoanh vùng đệ quy); 0 = không lần gọi nào phủ.
    INVALIDATE được phủ khi triple là ứng viên của vế cắt; ADD khi cặp {h, t} là ứng viên của vế sinh.
    Thứ tự giữ đúng thứ tự của diff (ổn định)."""
    explicit = set(explicit_ops)
    cuts = [loc.cut_triples for loc in locs]
    adds = [loc.add_pairs for loc in locs]
    out: list[tuple[Op, int]] = []
    for op in diff(ex.before, ex.after):
        if op in explicit:
            continue
        if op.kind is OpKind.INVALIDATE:
            hit = next((k for k, s in enumerate(cuts, 1) if op.triple in s), 0)
        else:
            pair = frozenset((op.h, op.t))
            hit = next((k for k, s in enumerate(adds, 1) if pair in s), 0)
        out.append((op, hit))
    return out
