from kgu.eval.coverage import coverage
from kgu.data.nba import Example
from kgu.localize import localize
from kgu.ops import apply_ops
from kgu.types import Op, OpKind
from tests.conftest import PLAYS

EXPLICIT = [
    Op(OpKind.INVALIDATE, "Messi", PLAYS, "Barca"),
    Op(OpKind.ADD, "Messi", PLAYS, "Inter"),
]


def test_coverage_full_on_fixture(graph, before, after):
    ex = Example(0, "trade", "2021", "", ["Messi", "Barca", "Inter"], before, after)
    apply_ops(graph, EXPLICIT, at=1)
    loc = localize(graph, EXPLICIT, at=1)
    covered, total = coverage(ex, EXPLICIT, loc)
    assert total == 4          # 2 teammate cắt + 2 teammate thêm
    assert covered == 4


def test_coverage_zero_without_localization(before, after):
    from kgu.localize import Localization
    ex = Example(0, "trade", "2021", "", [], before, after)
    assert coverage(ex, EXPLICIT, Localization()) == (0, 4)


def test_coverage_by_call_reports_first_covering_call(graph, before, after):
    from kgu.eval.coverage import coverage_by_call
    from kgu.localize import Localization
    ex = Example(0, "trade", "2021", "", ["Messi", "Barca", "Inter"], before, after)
    apply_ops(graph, EXPLICIT, at=1)
    loc = localize(graph, EXPLICIT, at=1)
    out = coverage_by_call(ex, EXPLICIT, [loc])
    assert len(out) == 4 and {k for _, k in out} == {1}
    assert all(op not in EXPLICIT for op, _ in out)
    # vùng ứng viên chỉ xuất hiện ở lần gọi 2 → cov = 2; không lần gọi nào phủ → 0
    assert {k for _, k in coverage_by_call(ex, EXPLICIT, [Localization(), loc])} == {2}
    assert {k for _, k in coverage_by_call(ex, EXPLICIT, [Localization()])} == {0}
    assert {k for _, k in coverage_by_call(ex, EXPLICIT, [])} == {0}
    # tách hai vế: lần 1 chỉ có vế cắt, lần 2 chỉ có vế sinh
    split = coverage_by_call(ex, EXPLICIT, [Localization(cut=loc.cut), Localization(add=loc.add), loc])
    assert {(op.kind, k) for op, k in split} == {(OpKind.INVALIDATE, 1), (OpKind.ADD, 2)}
    # thứ tự và tổng khớp hàm coverage cũ
    assert (sum(k > 0 for _, k in out), len(out)) == coverage(ex, EXPLICIT, loc)
