"""Đồ thị nền dùng chung: checkpoint/rollback hoàn tác add, invalidate, remove."""
from kgu.graph import BiTemporalGraph


def test_rollback_restores_everything():
    g = BiTemporalGraph.from_triples({("a", "r", "b"), ("b", "r", "c"), ("c", "r", "d")}, valid_from=0)
    snap0, nb0 = g.active(0), {e: g.neighbors(e, 0) for e in "abcd"}
    cp = g.checkpoint()
    assert g.remove(("c", "r", "d")) and not g.remove(("c", "r", "d"))
    assert g.add(("a", "r", "c"), 0) and g.add(("x", "r", "y"), 1) and g.invalidate(("a", "r", "b"), 1)
    assert g.active(0) == {("a", "r", "b"), ("b", "r", "c"), ("a", "r", "c")}
    assert g.active(1) == {("b", "r", "c"), ("a", "r", "c"), ("x", "r", "y")}
    g.rollback(cp)
    assert g.active(0) == snap0 and g.active(1) == snap0
    assert {e: g.neighbors(e, 0) for e in "abcd"} == nb0
    assert g.neighbors("x", 1) == set() and g.history(("x", "r", "y")) == []
    assert g.history(("a", "r", "b"))[0].valid_to is None


def test_nested_checkpoints():
    g = BiTemporalGraph.from_triples({("a", "r", "b")}, valid_from=0)
    c1 = g.checkpoint(); g.add(("a", "r", "c"), 0)
    c2 = g.checkpoint(); g.invalidate(("a", "r", "b"), 1)
    g.rollback(c2); assert g.active(1) == {("a", "r", "b"), ("a", "r", "c")}
    g.rollback(c1); assert g.active(0) == {("a", "r", "b")}
