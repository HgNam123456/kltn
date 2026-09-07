from pathlib import Path

from kgu.graph import BiTemporalGraph
from kgu.ops import apply_ops, diff, read_jsonl, write_jsonl
from kgu.types import Op, OpKind

T_MB = ("Messi", "plays_for", "Barca")
T_MI = ("Messi", "plays_for", "Inter")
T_PB = ("Pedri", "plays_for", "Barca")


def test_diff_produces_invalidate_then_add():
    ops = diff({T_MB, T_PB}, {T_MI, T_PB})
    assert ops == [Op(OpKind.INVALIDATE, *T_MB), Op(OpKind.ADD, *T_MI)]
    assert all(o.source == "gold" for o in ops)


def test_apply_ops_updates_graph_and_reports_effective_ops():
    g = BiTemporalGraph.from_triples({T_MB, T_PB})
    ops = [
        Op(OpKind.INVALIDATE, *T_MB),
        Op(OpKind.ADD, *T_MI),
        Op(OpKind.ADD, *T_PB),                    # đã có → không tác dụng
        Op(OpKind.INVALIDATE, "X", "plays_for", "Y"),  # không tồn tại → không tác dụng
    ]
    effective = apply_ops(g, ops, at=1)
    assert effective == ops[:2]
    assert g.active(1) == {T_MI, T_PB}
    assert g.active(0) == {T_MB, T_PB}


def test_diff_then_apply_reconstructs_after_exactly():
    before, after = {T_MB, T_PB}, {T_MI, T_PB}
    g = BiTemporalGraph.from_triples(before)
    apply_ops(g, diff(before, after), at=1)
    assert g.active(1) == after


def test_jsonl_roundtrip(tmp_path: Path):
    ops = [Op(OpKind.ADD, *T_MI, source="news#7", quote="Messi joins Inter", conf=0.8)]
    p = tmp_path / "ops.jsonl"
    write_jsonl(p, ops)
    back = read_jsonl(p)
    assert back == ops and back[0].quote == "Messi joins Inter" and back[0].conf == 0.8
