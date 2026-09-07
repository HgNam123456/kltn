from kgu.judge import JudgeContext
from kgu.judge.llm import LLMJudge
from kgu.llm import FakeLLMClient
from kgu.localize import Localization, localize
from kgu.ops import apply_ops
from kgu.types import Op, OpKind
from tests.conftest import FRIEND, PLAYS, TEAM

EXPLICIT = [
    Op(OpKind.INVALIDATE, "Messi", PLAYS, "Barca"),
    Op(OpKind.ADD, "Messi", PLAYS, "Inter"),
]


def _ctx(graph):
    apply_ops(graph, EXPLICIT, at=1)
    loc = localize(graph, EXPLICIT, at=1)
    return JudgeContext("Messi leaves Barca and joins Inter.", EXPLICIT, loc, graph, 1, [PLAYS, TEAM, FRIEND])


def test_judge_maps_verdicts_to_ops(graph):
    ctx = _ctx(graph)
    cut_idx = {c.triple: i for i, c in enumerate(ctx.loc.cut)}
    add_idx = {c.neighbor: i for i, c in enumerate(ctx.loc.add)}
    client = FakeLLMClient([{
        "cut": [
            {"idx": cut_idx[("Pedri", TEAM, "Messi")], "verdict": "INVALIDATE"},
            {"idx": cut_idx[("Messi", TEAM, "Pedri")], "verdict": "INVALIDATE"},
            {"idx": cut_idx[("Pedri", PLAYS, "Barca")], "verdict": "KEEP"},
            {"idx": cut_idx[("Pedri", FRIEND, "Messi")], "verdict": "KEEP"},
            {"idx": 99, "verdict": "INVALIDATE"},                      # ngoài phạm vi → bỏ
        ],
        "add": [
            {"idx": add_idx["Lautaro"], "add": True, "r": TEAM, "direction": "neighbor_first"},
            {"idx": add_idx["Inzaghi"], "add": False},
        ],
    }])
    ops = LLMJudge(client).judge(ctx)
    assert set(ops) == {
        Op(OpKind.INVALIDATE, "Pedri", TEAM, "Messi"),
        Op(OpKind.INVALIDATE, "Messi", TEAM, "Pedri"),
        Op(OpKind.ADD, "Lautaro", TEAM, "Messi"),
    }
    assert all(o.source == "llm_judge" for o in ops)
    assert client.n_calls == 1


def test_judge_skips_llm_when_no_candidates(graph):
    client = FakeLLMClient([])
    ctx = JudgeContext("", [], Localization(), graph, 1, [PLAYS])
    assert LLMJudge(client).judge(ctx) == [] and client.n_calls == 0


def test_prompt_lists_candidates_with_indices(graph):
    ctx = _ctx(graph)
    client = FakeLLMClient([{"cut": [], "add": []}])
    LLMJudge(client).judge(ctx)
    _, user = client.calls[0]
    assert "[C0]" in user and "[A0]" in user and "Lautaro" in user and "hub=Pedri" in user
