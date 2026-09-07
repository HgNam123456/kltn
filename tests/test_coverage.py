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
