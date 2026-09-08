from kgu.judge import JudgeContext
from kgu.judge.llm import LLMJudge, group_add, group_cut
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


def _cut_key(p):
    return (p.participant, p.r, p.direction)


def test_group_cut_by_participant_relation_direction(graph):
    ctx = _ctx(graph)
    cut = group_cut(ctx.loc)
    keys = {_cut_key(p) for p in cut}
    # hub Pedri nối Messi (teammate 2 chiều, close_friend) và Barca (plays_for)
    assert ("Messi", TEAM, "subject_first") in keys       # (Messi, teammate, X)
    assert ("Messi", TEAM, "neighbor_first") in keys      # (X, teammate, Messi)
    assert ("Barca", PLAYS, "neighbor_first") in keys     # (X, plays_for, Barca)
    assert ("Messi", FRIEND, "neighbor_first") in keys    # (X, close_friend, Messi)
    assert sum(len(p.members) for p in cut) == len(ctx.loc.cut)   # không mất, không trùng
    p = next(p for p in cut if _cut_key(p) == ("Messi", TEAM, "subject_first"))
    assert p.hubs == ["Pedri"] and p.co_anchors == {"Barca"}


def test_group_add_by_subject_and_evidence(graph):
    ctx = _ctx(graph)
    add = group_add(ctx.loc)
    assert sum(len(p.members) for p in add) == len(ctx.loc.add)
    p = next(p for p in add if p.via_r == PLAYS)
    assert p.subject == "Messi" and p.other == "Inter" and p.via_direction == "neighbor_first"
    assert "Lautaro" in p.neighbors


def test_judge_skips_llm_when_no_candidates(graph):
    client = FakeLLMClient([])
    ctx = JudgeContext("", [], Localization(), graph, 1, [PLAYS])
    assert LLMJudge(client).judge(ctx) == [] and client.n_calls == 0


def test_judge_maps_pattern_verdicts_to_all_member_edges(graph):
    ctx = _ctx(graph)
    cut, add = group_cut(ctx.loc), group_add(ctx.loc)
    cut_idx = {_cut_key(p): i for i, p in enumerate(cut)}
    add_idx = {(p.subject, p.via_r): i for i, p in enumerate(add)}
    client = FakeLLMClient([
        {"invalidate": [cut_idx[("Messi", TEAM, "subject_first")], cut_idx[("Messi", TEAM, "neighbor_first")], 99]},  # pha CẮT
        {"add": [                                                                                                # pha SINH
            {"idx": add_idx[("Messi", PLAYS)], "r": TEAM, "direction": "neighbor_first"},
            {"idx": add_idx[("Messi", PLAYS)], "r": TEAM, "direction": "subject_first"},   # đối xứng → 2 chiều
            {"idx": 99, "r": TEAM, "direction": "subject_first"},                          # ngoài phạm vi → bỏ
        ]},
    ])
    ops = LLMJudge(client).judge(ctx)
    assert set(ops) == {
        Op(OpKind.INVALIDATE, "Pedri", TEAM, "Messi"),
        Op(OpKind.INVALIDATE, "Messi", TEAM, "Pedri"),
        Op(OpKind.ADD, "Lautaro", TEAM, "Messi"),
        Op(OpKind.ADD, "Messi", TEAM, "Lautaro"),
    }
    assert all(o.source == "llm_judge" for o in ops)
    assert client.n_calls == 2                                          # 2 pha = 2 lệnh gọi


def test_pattern_verdict_fans_out_to_every_hub():
    """Nhiều hub cùng mẫu → 1 phán quyết áp cho tất cả (điểm cốt lõi của cách gom)."""
    from kgu.graph import BiTemporalGraph
    before = {("Messi", PLAYS, "Barca"), ("Messi", TEAM, "Pedri"), ("Messi", TEAM, "Gavi"),
              ("Pedri", PLAYS, "Barca"), ("Gavi", PLAYS, "Barca"), ("Lautaro", PLAYS, "Inter")}
    g = BiTemporalGraph.from_triples(before, valid_from=0)
    apply_ops(g, EXPLICIT, at=1)
    loc = localize(g, EXPLICIT, at=1)
    ctx = JudgeContext("", EXPLICIT, loc, g, 1, [PLAYS, TEAM])
    cut = group_cut(loc)
    i = next(i for i, p in enumerate(cut) if _cut_key(p) == ("Messi", TEAM, "subject_first"))
    assert len(cut[i].members) == 2
    client = FakeLLMClient([{"invalidate": [i]}, {"add": []}])
    ops = LLMJudge(client).judge(ctx)
    assert set(ops) == {Op(OpKind.INVALIDATE, "Messi", TEAM, "Pedri"), Op(OpKind.INVALIDATE, "Messi", TEAM, "Gavi")}


def test_two_phase_prompts_list_patterns_not_edges(graph):
    ctx = _ctx(graph)
    client = FakeLLMClient([{"invalidate": []}, {"add": []}])
    LLMJudge(client).judge(ctx)
    (sys_cut, cut_user), (sys_add, add_user) = client.calls
    # pha CẮT: chỉ có mẫu cắt
    assert "[C0]" in cut_user and "[A0]" not in cut_user
    assert f"(Messi, {TEAM}, X)" in cut_user and "1 X: Pedri" in cut_user   # mẫu + số lượng + ví dụ
    assert "X cũng nối với Barca" in cut_user
    assert f"(Pedri, {TEAM}, Messi)" not in cut_user                        # không liệt kê từng cạnh
    assert "PHA CẮT" in sys_cut and "PHA SINH" not in sys_cut
    # pha SINH: chỉ có mẫu thêm
    assert "[A0]" in add_user and "[C0]" not in add_user
    assert f"bằng chứng: (X, {PLAYS}, Inter)" in add_user
    assert "PHA SINH" in sys_add and "PHA CẮT" not in sys_add
    for u in (cut_user, add_user):
        assert u.count("ENTITY THAM GIA") == 1 and "BẢN TIN" in u


def test_phase_skipped_when_it_has_no_patterns(graph):
    """Không có mẫu thêm → chỉ gọi pha CẮT."""
    from kgu.localize import Localization
    ctx = _ctx(graph)
    loc = Localization(cut=list(ctx.loc.cut), add=[])
    ctx2 = JudgeContext(ctx.text, ctx.explicit_ops, loc, ctx.graph, ctx.at, ctx.relations)
    client = FakeLLMClient([{"invalidate": []}])
    LLMJudge(client).judge(ctx2)
    assert client.n_calls == 1 and "PHA CẮT" in client.calls[0][0]


def test_prompt_truncates_examples():
    from kgu.judge.llm import _examples
    assert _examples(["a", "b", "c"], 2) == "a, b, ..."
    assert _examples(["a"], 2) == "a"


def test_add_schema_restricts_relation_to_enum():
    import json
    from kgu.judge.llm import make_add_schema, make_cut_schema
    s = json.dumps(make_add_schema([PLAYS, TEAM]).model_json_schema())
    assert f'"enum": ["{PLAYS}", "{TEAM}"]' in s
    assert list(make_cut_schema().model_fields) == ["invalidate"]
