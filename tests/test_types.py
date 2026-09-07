from kgu.types import Op, OpKind


def test_op_roundtrip_dict():
    op = Op(OpKind.ADD, "Messi", "plays_for", "Inter", source="news#1", quote="Messi joins Inter", conf=0.9)
    d = op.to_dict()
    assert d == {
        "kind": "ADD", "h": "Messi", "r": "plays_for", "t": "Inter",
        "source": "news#1", "quote": "Messi joins Inter", "conf": 0.9,
    }
    assert Op.from_dict(d) == op


def test_op_triple_property():
    op = Op(OpKind.INVALIDATE, "Messi", "plays_for", "Barca")
    assert op.triple == ("Messi", "plays_for", "Barca")


def test_op_is_hashable_and_ignores_provenance_in_equality():
    a = Op(OpKind.ADD, "a", "r", "b", quote="x")
    b = Op(OpKind.ADD, "a", "r", "b", quote="y")
    assert a == b
    assert len({a, b}) == 1
