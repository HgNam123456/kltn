import json

from kgu.graph import BiTemporalGraph
from kgu.judge import JudgeContext
from kgu.judge.naive import SYSTEM, NaiveOneHopJudge, make_naive_schema
from kgu.llm import FakeLLMClient
from kgu.localize import Localization
from kgu.ops import apply_ops
from kgu.types import Op, OpKind
from tests.conftest import COACH, FRIEND, PLAYS, SPONSOR, TEAM

EXPLICIT = [
    Op(OpKind.INVALIDATE, "Messi", PLAYS, "Barca"),
    Op(OpKind.ADD, "Messi", PLAYS, "Inter"),
]
RELATIONS = [PLAYS, TEAM, COACH, FRIEND, SPONSOR]


def _ctx(graph, explicit=EXPLICIT, relations=RELATIONS):
    """Bộ phán 1 bước không dùng khoanh vùng → Localization rỗng."""
    apply_ops(graph, explicit, at=1)
    return JudgeContext("Messi leaves Barca and joins Inter.", explicit, Localization(), graph, 1, relations)


def _listed(user: str) -> list[str]:
    return [line.strip() for line in user.splitlines() if line.startswith("  [")]


class _Boom:
    n_calls = 0

    def complete_json(self, system, user, schema):
        self.n_calls += 1
        raise RuntimeError("server down")


def test_prompt_numbers_one_hop_triples_and_leaves_out_explicit_ones(graph):
    ctx = _ctx(graph)
    client = FakeLLMClient([{"reason": "", "invalidate": [], "add": []}])
    j = NaiveOneHopJudge(client)
    assert j.judge(ctx) == [] and client.n_calls == 1 and j.n_calls == 1
    system, user = client.calls[0]
    assert system == SYSTEM
    # mọi cạnh đang hiệu lực chạm Messi / Barca / Inter, theo thứ tự thực thể tham gia (Barca, Inter, Messi) rồi theo triple
    assert _listed(user) == [
        f"[0] (Pedri, {PLAYS}, Barca)",
        f"[1] (Xavi, {COACH}, Barca)",
        f"[2] (Inzaghi, {COACH}, Inter)",
        f"[3] (Lautaro, {PLAYS}, Inter)",
        f"[4] (Adidas, {SPONSOR}, Messi)",
        f"[5] (Messi, {TEAM}, Pedri)",
        f"[6] (Pedri, {FRIEND}, Messi)",
        f"[7] (Pedri, {TEAM}, Messi)",
    ]
    # triple của thay đổi trực tiếp chỉ nằm ở phần "DIRECT CHANGES", không nằm trong danh sách đánh số
    assert f"  INVALIDATE (Messi, {PLAYS}, Barca)" in user and f"  ADD (Messi, {PLAYS}, Inter)" in user
    assert not any(f"(Messi, {PLAYS}, " in line for line in _listed(user))
    assert "INVOLVED ENTITIES: Barca, Inter, Messi" in user
    assert "ALLOWED RELATIONS: " + ", ".join(RELATIONS) in user
    assert "Messi leaves Barca and joins Inter." in user
    assert (j.n_items, j.n_listed, j.n_truncated, j.n_gated) == (1, 8, 0, 0)


def test_valid_invalidate_and_add_become_ops(graph):
    ctx = _ctx(graph)
    client = FakeLLMClient([{"reason": "teammates change with the club", "invalidate": [5, 7, 5],
                             "add": [{"h": "Messi", "r": TEAM, "t": "Lautaro"},
                                     {"h": "lautaro ", "r": TEAM, "t": "Messi"}]}])      # sai hoa/thường, thừa khoảng trắng → tên gốc
    j = NaiveOneHopJudge(client)
    ops = j.judge(ctx)
    assert ops == [
        Op(OpKind.INVALIDATE, "Messi", TEAM, "Pedri"),
        Op(OpKind.INVALIDATE, "Pedri", TEAM, "Messi"),
        Op(OpKind.ADD, "Messi", TEAM, "Lautaro"),
        Op(OpKind.ADD, "Lautaro", TEAM, "Messi"),
    ]
    assert all(o.source == "naive_1hop" for o in ops)
    assert j.n_dropped == 0 and j.n_errors == 0                          # chỉ số lặp (5 hai lần) được gộp, không tính là loại
    assert len(apply_ops(ctx.graph, ops, at=1)) == 4                     # mọi op đều áp được lên đồ thị


def test_invalid_output_is_dropped_and_counted(graph):
    ctx = _ctx(graph)
    # đường json_object không ép enum → quan hệ lạ có thể lọt tới bộ phán: dùng schema không enum để mô phỏng
    client = FakeLLMClient([{"invalidate": [-1, 8, 99, 0],
                             "add": [{"h": "Messi", "r": TEAM, "t": "Ronaldo"},        # thực thể không có trong đầu vào
                                     {"h": "Messi", "r": TEAM, "t": "Messi"},          # tự nối
                                     {"h": "Messi", "r": "rival_of", "t": "Lautaro"},  # quan hệ ngoài danh sách
                                     {"h": "Messi", "r": PLAYS, "t": "Inter"},         # đã hiệu lực (thay đổi trực tiếp)
                                     {"h": "Pedri", "r": FRIEND, "t": "Messi"},        # đã hiệu lực (cạnh đã lưu)
                                     {"h": "Messi", "r": TEAM, "t": "Lautaro"}]}])     # hợp lệ
    j = NaiveOneHopJudge(client)
    j._schema = lambda relations: make_naive_schema([])
    ops = j.judge(ctx)
    assert ops == [Op(OpKind.INVALIDATE, "Pedri", PLAYS, "Barca"), Op(OpKind.ADD, "Messi", TEAM, "Lautaro")]
    assert j.n_dropped == 8
    assert dict(j.drop_reasons) == {"bad_index": 3, "unknown_entity": 1, "self_loop": 1,
                                    "unknown_relation": 1, "already_active": 2}


def test_schema_restricts_relation_to_enum_and_puts_reason_first():
    schema = make_naive_schema([PLAYS, TEAM])
    assert list(schema.model_fields) == ["reason", "invalidate", "add"]
    assert f'"enum": ["{PLAYS}", "{TEAM}"]' in json.dumps(schema.model_json_schema())
    assert schema.model_validate({}).add == []                           # mặc định: không đổi gì


def _hub_graph(n_big: int = 500, n_small: int = 3):
    before = {("Messi", PLAYS, "Barca")}
    before |= {(f"fan{i:04d}", "follows", "Barca") for i in range(n_big)}
    before |= {(f"fan{i:04d}", "follows", "Inter") for i in range(n_big)}
    before |= {("Messi", "owns", f"car{i}") for i in range(n_small)}
    return BiTemporalGraph.from_triples(before, valid_from=0)


def test_sampling_cap_is_respected_balanced_and_deterministic():
    def run():
        ctx = _ctx(_hub_graph(), relations=[PLAYS, "follows", "owns"])
        client = FakeLLMClient([{}])
        j = NaiveOneHopJudge(client, max_triples=40)
        j.judge(ctx)
        return j, client.calls[0][1], ctx

    j, user, ctx = run()
    lines = _listed(user)
    assert len(lines) == 40 and (j.n_listed, j.n_truncated) == (40, 1)
    assert [line.split("]")[0] for line in lines] == [f"[{i}" for i in range(40)]
    # Messi chỉ có 3 cạnh → lấy cả 3; phần thừa chia lại: Barca 18, Inter 19
    assert sum("Barca)" in line for line in lines) == 18
    assert sum("Inter)" in line for line in lines) == 19
    assert sum("(Messi, owns," in line for line in lines) == 3
    assert "(40 of 1003 shown, evenly sampled)" in user
    assert "(fan0000, follows, Barca)" in user and "(fan0472, follows, Barca)" in user   # cách đều trên danh sách đã sắp
    assert len(set(lines)) == 40
    assert run()[1] == user                                              # tái lập được
    listed, total = j.context_triples(ctx)
    assert total == 1003 and all(ctx.graph.is_active(tr, 1) for tr in listed)


def test_no_sampling_at_or_below_cap():
    ctx = _ctx(_hub_graph(n_big=10), relations=[PLAYS, "follows", "owns"])
    client = FakeLLMClient([{}])
    j = NaiveOneHopJudge(client, max_triples=23)
    j.judge(ctx)
    assert len(_listed(client.calls[0][1])) == 23 and j.n_truncated == 0
    assert "(23)" in client.calls[0][1]


def test_long_relation_list_is_not_repeated_in_user_prompt(graph):
    rels = RELATIONS + [f"relation number {i:03d}" for i in range(200)]
    ctx = _ctx(graph, relations=rels)
    client = FakeLLMClient([{}])
    NaiveOneHopJudge(client).judge(ctx)
    user = client.calls[0][1]
    assert "relation number 007" not in user
    assert "ALLOWED RELATIONS: the 205 relation names in the enum of field r of the output schema" in user


def test_client_exception_returns_no_ops_and_counts_error(graph):
    ctx = _ctx(graph)
    j = NaiveOneHopJudge(_Boom())
    assert j.judge(ctx) == []
    assert j.n_errors == 1 and j.n_calls == 1 and j.n_dropped == 0


def test_no_explicit_ops_means_no_call(graph):
    client = FakeLLMClient([])
    ctx = JudgeContext("nothing happened", [], Localization(), graph, 1, RELATIONS)
    assert NaiveOneHopJudge(client).judge(ctx) == [] and client.n_calls == 0


def test_run_example_with_naive_judge_scores_indirect_changes(before, after):
    """Chạy qua pipeline thật: bộ phán 1 bước trả đúng 4 thay đổi gián tiếp → KG dự đoán trùng KG-sau."""
    from kgu.data import Example
    from kgu.extract.gold import GoldExplicitExtractor
    from kgu.pipeline import run_example

    ex = Example(idx=0, event="transfer", season="", text="Messi leaves Barca and joins Inter.",
                 mentioned=["Messi", "Barca", "Inter"], before=before, after=after)
    client = FakeLLMClient([{"invalidate": [5, 7], "add": [{"h": "Messi", "r": TEAM, "t": "Lautaro"},
                                                           {"h": "Lautaro", "r": TEAM, "t": "Messi"}]}])
    res = run_example(ex, GoldExplicitExtractor(), NaiveOneHopJudge(client), RELATIONS)
    assert res.pred_after == after and res.n_llm_calls == 1 and len(res.judged_ops) == 4
