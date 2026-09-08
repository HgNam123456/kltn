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


def test_judge_skips_llm_when_no_candidates(graph):
    client = FakeLLMClient([])
    ctx = JudgeContext("", [], Localization(), graph, 1, [PLAYS])
    assert LLMJudge(client).judge(ctx) == [] and client.n_calls == 0


def test_judge_maps_verdicts_to_ops(graph):
    ctx = _ctx(graph)
    cut_idx = {c.triple: i for i, c in enumerate(ctx.loc.cut)}
    add_idx = {c.neighbor: i for i, c in enumerate(ctx.loc.add)}
    client = FakeLLMClient([{
        "invalidate": [cut_idx[("Pedri", TEAM, "Messi")], cut_idx[("Messi", TEAM, "Pedri")], 99],  # 99 ngoài phạm vi → bỏ
        "add": [
            {"idx": add_idx["Lautaro"], "r": TEAM, "direction": "neighbor_first"},
            {"idx": add_idx["Lautaro"], "r": TEAM, "direction": "subject_first"},   # quan hệ đối xứng → 2 chiều
        ],
    }])
    ops = LLMJudge(client).judge(ctx)
    assert set(ops) == {
        Op(OpKind.INVALIDATE, "Pedri", TEAM, "Messi"),
        Op(OpKind.INVALIDATE, "Messi", TEAM, "Pedri"),
        Op(OpKind.ADD, "Lautaro", TEAM, "Messi"),
        Op(OpKind.ADD, "Messi", TEAM, "Lautaro"),
    }
    assert all(o.source == "llm_judge" for o in ops)
    assert client.n_calls == 1


def test_prompt_groups_cut_candidates_by_hub_without_repeating_anchors(graph):
    ctx = _ctx(graph)
    client = FakeLLMClient([{"invalidate": [], "add": []}])
    LLMJudge(client).judge(ctx)
    system, user = client.calls[0]
    assert "[C0]" in user and "[A0]" in user and "Lautaro" in user
    assert "Pedri:" in user                       # gom theo hub
    assert "hub=" not in user and "anchors=" not in user
    assert user.count("ENTITY THAM GIA") == 1
    assert "VÍ DỤ" in system


def test_judge_schema_restricts_relation_to_enum():
    import json
    from kgu.judge.llm import make_judge_schema
    s = json.dumps(make_judge_schema([PLAYS, TEAM]).model_json_schema())
    assert f'"enum": ["{PLAYS}", "{TEAM}"]' in s
