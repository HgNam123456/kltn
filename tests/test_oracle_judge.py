from kgu.judge import JudgeContext
from kgu.judge.oracle import OracleJudge
from kgu.localize import localize
from kgu.ops import apply_ops
from kgu.types import Op, OpKind
from tests.conftest import PLAYS, TEAM, FRIEND

EXPLICIT = [
    Op(OpKind.INVALIDATE, "Messi", PLAYS, "Barca"),
    Op(OpKind.ADD, "Messi", PLAYS, "Inter"),
]


def test_oracle_reconstructs_gold_after(graph, after):
    apply_ops(graph, EXPLICIT, at=1)
    loc = localize(graph, EXPLICIT, at=1)
    ctx = JudgeContext(text="", explicit_ops=EXPLICIT, loc=loc, graph=graph, at=1, relations=[PLAYS, TEAM])
    ops = OracleJudge(after).judge(ctx)
    assert set(ops) == {
        Op(OpKind.INVALIDATE, "Pedri", TEAM, "Messi"),
        Op(OpKind.INVALIDATE, "Messi", TEAM, "Pedri"),
        Op(OpKind.ADD, "Lautaro", TEAM, "Messi"),
        Op(OpKind.ADD, "Messi", TEAM, "Lautaro"),
    }
    assert Op(OpKind.INVALIDATE, "Pedri", FRIEND, "Messi") not in ops   # giữ
    apply_ops(graph, ops, at=1)
    assert graph.active(1) == after
    assert all(o.source == "oracle" for o in ops)
