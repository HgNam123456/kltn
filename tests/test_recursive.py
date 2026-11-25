"""Khoanh vùng đệ quy: tham số K (max_calls) của run_example — Thuật toán alg:localize, Chương 3."""
from kgu.data import Example
from kgu.extract.gold import GoldExplicitExtractor
from kgu.judge.oracle import OracleJudge
from kgu.localize import localize
from kgu.pipeline import run_example
from kgu.types import Op, OpKind
from tests.conftest import FRIEND, PLAYS, TEAM

MEMBER, PART, WORKS, HQ, BASED, LOC, RES, COLL = (
    "member_of", "part_of", "works_for", "headquartered_in", "based_in", "located_in", "resident_of", "colleague_of")
RELS = [MEMBER, PART, WORKS, HQ, BASED, LOC, RES, COLL]

# Chuỗi 3 tầng: e đổi phòng D1 → D2 (văn bản chỉ nói điều này). Tầng 1: works_for theo công ty của phòng;
# tầng 2: based_in theo trụ sở công ty; tầng 3: resident_of theo nước của thành phố.
STRUCT = {
    ("D1", PART, "C1"), ("D2", PART, "C2"),
    ("C1", HQ, "A"), ("C2", HQ, "B"),
    ("A", LOC, "VN"), ("B", LOC, "JP"),
}
CHAIN_BEFORE = STRUCT | {("e", MEMBER, "D1"), ("e", WORKS, "C1"), ("e", BASED, "A"), ("e", RES, "VN")}
CHAIN_AFTER = STRUCT | {("e", MEMBER, "D2"), ("e", WORKS, "C2"), ("e", BASED, "B"), ("e", RES, "JP")}

# Bản đầy đủ hơn: x là đồng nghiệp cũ ở D1, y là đồng nghiệp mới ở D2.
RICH_STRUCT = STRUCT | {
    ("x", MEMBER, "D1"), ("x", WORKS, "C1"), ("x", BASED, "A"), ("x", RES, "VN"),
    ("y", MEMBER, "D2"), ("y", WORKS, "C2"), ("y", BASED, "B"), ("y", RES, "JP"),
}
RICH_BEFORE = (CHAIN_BEFORE - STRUCT) | RICH_STRUCT | {("e", COLL, "x"), ("x", COLL, "e")}
RICH_AFTER = (CHAIN_AFTER - STRUCT) | RICH_STRUCT | {("e", COLL, "y"), ("y", COLL, "e")}


def _ex(before, after, mentioned=("e", "D1", "D2")):
    return Example(0, "move", "", "e moves from D1 to D2.", list(mentioned), set(before), set(after))


def _run(before, after, k, judge=None, **kw):
    ex = _ex(before, after, **kw)
    res = run_example(ex, GoldExplicitExtractor(), judge or OracleJudge(ex.after), RELS, max_calls=k)
    return ex, res


class _Recorder:
    """Bọc một bộ phán: ghi lại ngữ cảnh từng lần gọi và đếm n_calls (mỗi lần judge = số ứng viên lệnh gọi giả)."""

    def __init__(self, inner):
        self.inner = inner
        self.ctxs = []
        self.raw = []                      # vùng ứng viên CHƯA lọc của cùng lần gọi
        self.n_calls = 0

    def judge(self, ctx):
        self.ctxs.append(ctx)
        self.raw.append(localize(ctx.graph, ctx.explicit_ops, ctx.at))
        self.n_calls += len(ctx.loc.cut) + len(ctx.loc.add)
        return self.inner.judge(ctx)


def _acc(ex, res):
    add, dele = ex.after - ex.before, ex.before - ex.after
    return (sum(tr in res.pred_after for tr in add), sum(tr not in res.pred_after for tr in dele))


# ---- (a) K = 1 giữ nguyên hành vi cũ --------------------------------------------------------------------------

def test_default_is_one_call_with_old_record_fields(before, after):
    ex = Example(3, "trade", "2021", "messi joins inter", ["Messi", "Barca", "Inter"], before, after)
    rec_judge = _Recorder(OracleJudge(after))
    res = run_example(ex, GoldExplicitExtractor(), rec_judge, [PLAYS, TEAM])
    assert res.pred_after == after and res.max_calls == 1
    assert len(rec_judge.ctxs) == 1
    ctx = rec_judge.ctxs[0]
    assert ctx.call == 1 and ctx.direct_ops == res.explicit_ops and ctx.explicit_ops == res.explicit_ops
    assert ctx.loc.cut == rec_judge.raw[0].cut and ctx.loc.add == rec_judge.raw[0].add     # lần 1 không lọc gì
    rec = res.to_record(ex)
    assert rec["coverage"] == [4, 4] and rec["n_judged"] == 4 and rec["n_explicit"] == 2
    assert rec["n_cut_cand"] == len(rec_judge.raw[0].cut) and rec["n_add_cand"] == len(rec_judge.raw[0].add)
    assert rec["max_calls"] == 1 and [r["k"] for r in rec["rounds"]] == [1]
    assert all(o["round"] == 1 for o in rec["judged_ops"])
    assert {o["cov"] for o in rec["implied"]} == {1} and len(rec["implied"]) == 4
    old_keys = {"idx", "event", "n_explicit", "n_cut_cand", "n_add_cand", "n_judged", "n_llm_calls", "n_dropped",
                "coverage", "counts", "explicit_ops", "judged_ops"}
    assert old_keys <= set(rec)


def test_no_judge_runs_a_single_call(before, after):
    ex = Example(3, "trade", "2021", "", ["Messi", "Barca", "Inter"], before, after)
    res = run_example(ex, GoldExplicitExtractor(), None, [PLAYS, TEAM], max_calls=3)
    assert len(res.rounds) == 1 and res.judged_ops == [] and res.rounds[0]["n_ops"] == 0
    assert res.to_record(ex)["coverage"] == [4, 4]


# ---- (b) chuỗi 3 tầng: mỗi lần gọi thêm một tầng ---------------------------------------------------------------

def test_each_call_reaches_one_more_level():
    ex1, r1 = _run(CHAIN_BEFORE, CHAIN_AFTER, 1)
    assert _acc(ex1, r1) == (2, 2)                       # member_of (trực tiếp) + works_for (tầng 1)
    assert ("e", BASED, "A") in r1.pred_after and ("e", RES, "VN") in r1.pred_after
    ex2, r2 = _run(CHAIN_BEFORE, CHAIN_AFTER, 2)
    assert _acc(ex2, r2) == (3, 3)                       # thêm based_in (tầng 2)
    assert ("e", RES, "VN") in r2.pred_after and ("e", RES, "JP") not in r2.pred_after
    ex3, r3 = _run(CHAIN_BEFORE, CHAIN_AFTER, 3)
    assert r3.pred_after == ex3.after                    # K = 3 khớp hoàn toàn KG-sau

    rec = r3.to_record(ex3)
    cov = {(o["kind"], o["r"]): o["cov"] for o in rec["implied"]}
    assert cov == {("INVALIDATE", WORKS): 1, ("ADD", WORKS): 1, ("INVALIDATE", BASED): 2, ("ADD", BASED): 2,
                   ("INVALIDATE", RES): 3, ("ADD", RES): 3}
    assert rec["coverage"] == [6, 6] and rec["max_calls"] == 3
    assert [r["k"] for r in rec["rounds"]] == [1, 2, 3] and [r["n_ops"] for r in rec["rounds"]] == [2, 2, 2]
    by_round = {(o["kind"], o["r"]): o["round"] for o in rec["judged_ops"]}
    assert by_round == cov                               # op được kết luận đúng ở lần gọi phủ nó
    assert rec["n_cut_cand"] == sum(r["n_cut_cand"] for r in rec["rounds"])
    assert rec["n_add_cand"] == sum(r["n_add_cand"] for r in rec["rounds"])

    cov1 = {(o["kind"], o["r"]): o["cov"] for o in r1.to_record(ex1)["implied"]}
    assert cov1[("ADD", WORKS)] == 1 and cov1[("ADD", BASED)] == 0 and cov1[("INVALIDATE", RES)] == 0
    assert r1.to_record(ex1)["coverage"] == [2, 6]


def test_rich_world_matches_gold_after_with_enough_calls():
    ex1, r1 = _run(RICH_BEFORE, RICH_AFTER, 1)
    assert r1.pred_after != ex1.after
    ex3, r3 = _run(RICH_BEFORE, RICH_AFTER, 3)
    assert r3.pred_after == ex3.after
    assert all(o["cov"] >= 1 for o in r3.to_record(ex3)["implied"])


# ---- (c) không xét lại ứng viên cũ, không đụng cạnh vừa ghi ---------------------------------------------------

def test_later_calls_skip_seen_candidates_and_edges_written_for_this_text():
    ex = _ex(RICH_BEFORE, RICH_AFTER)
    judge = _Recorder(OracleJudge(ex.after))
    res = run_example(ex, GoldExplicitExtractor(), judge, RELS, max_calls=3)
    assert len(judge.ctxs) >= 2
    seen_cut, seen_pairs, written = set(), set(), {o.triple for o in res.explicit_ops}
    refiltered = 0
    for ctx, raw, loc in zip(judge.ctxs, judge.raw, res.locs):
        assert ctx.loc is loc and ctx.direct_ops == res.explicit_ops
        assert not (loc.cut_triples & seen_cut)                     # không xét lại ứng viên cắt
        assert not (loc.add_pairs & seen_pairs)                     # không xét lại cặp ứng viên sinh
        assert not (loc.cut_triples & written)                      # không cắt cạnh đã ghi khi xử lý bản tin này
        refiltered += len(raw.cut_triples - loc.cut_triples) + len(raw.add_pairs - loc.add_pairs)
        seen_cut |= loc.cut_triples
        seen_pairs |= loc.add_pairs
        written |= {o.triple for o, k in zip(res.judged_ops, res.judged_rounds) if k == ctx.call}
    assert refiltered > 0                                           # bộ lọc thật sự loại được ứng viên
    raw2 = judge.raw[1].cut_triples
    assert ("x", MEMBER, "D1") in raw2 and ("x", MEMBER, "D1") not in res.locs[1].cut_triples   # đã xét ở lần 1
    assert ("e", MEMBER, "D2") in raw2 and ("e", MEMBER, "D2") not in res.locs[1].cut_triples   # cạnh vừa thêm
    assert ("D2", PART, "C2") in res.locs[1].cut_triples                                        # ứng viên mới thật
    assert res.judged_rounds == sorted(res.judged_rounds)
    assert [ctx.call for ctx in judge.ctxs] == list(range(1, len(judge.ctxs) + 1))
    assert judge.ctxs[1].explicit_ops == [o for o, k in zip(res.judged_ops, res.judged_rounds) if k == 1]
    # vùng gộp không trùng
    assert len(res.loc.cut) == len(res.loc.cut_triples) == sum(len(loc.cut) for loc in res.locs)
    assert len(res.loc.add) == sum(len(loc.add) for loc in res.locs)


# ---- (d) dừng sớm ---------------------------------------------------------------------------------------------

def test_stops_when_a_call_concludes_nothing(before, after):
    """Ví dụ Messi: lần gọi 2 chỉ có ứng viên được giữ nguyên → thuật toán dừng sau lần gọi 2 dù K = 5."""
    ex = Example(0, "trade", "", "", ["Messi", "Barca", "Inter"], before, after)
    judge = _Recorder(OracleJudge(after))
    res = run_example(ex, GoldExplicitExtractor(), judge, [PLAYS, TEAM, FRIEND], max_calls=5)
    assert res.pred_after == after and res.max_calls == 5
    assert [r["k"] for r in res.rounds] == [1, 2] and res.rounds[1]["n_ops"] == 0
    assert ("Lautaro", PLAYS, "Inter") in res.locs[1].cut_triples          # ứng viên lần 2, được giữ nguyên
    assert ("Messi", PLAYS, "Inter") not in res.locs[1].cut_triples        # cạnh trực tiếp vừa thêm: không xét


class _OneShot:
    """Lần gọi 1 trả một ADD không mở ra ứng viên mới nào; đếm số lần được gọi."""

    def __init__(self, op):
        self.op, self.n = op, 0

    def judge(self, ctx):
        self.n += 1
        return [self.op] if ctx.call == 1 else []


def test_stops_without_calling_judge_when_no_new_candidates(before, after):
    ex = Example(0, "trade", "", "", ["Messi", "Barca", "Inter"], before, after)
    judge = _OneShot(Op(OpKind.ADD, "Messi", "coached_by", "Inzaghi"))     # hàng xóm của Inzaghi đều đã nối với Messi
    res = run_example(ex, GoldExplicitExtractor(), judge, [PLAYS, TEAM], max_calls=4)
    assert judge.n == 1 and [r["k"] for r in res.rounds] == [1] and len(res.judged_ops) == 1


def test_noop_ops_do_not_trigger_another_call(before, after):
    """Op không hiệu lực (ADD cạnh đang có) không tính là thay đổi mới."""
    ex = Example(0, "trade", "", "", ["Messi", "Barca", "Inter"], before, after)
    judge = _OneShot(Op(OpKind.ADD, "Lautaro", PLAYS, "Inter"))
    res = run_example(ex, GoldExplicitExtractor(), judge, [PLAYS, TEAM], max_calls=3)
    assert judge.n == 1 and res.judged_ops == [] and len(res.rounds) == 1


# ---- (e) số lệnh gọi theo lần gọi ----------------------------------------------------------------------------

def test_llm_calls_per_call_sum_to_total():
    ex = _ex(RICH_BEFORE, RICH_AFTER)
    judge = _Recorder(OracleJudge(ex.after))
    res = run_example(ex, GoldExplicitExtractor(), judge, RELS, max_calls=3)
    per_call = [r["n_llm_calls"] for r in res.rounds]
    assert len(per_call) >= 2 and all(c > 0 for c in per_call[:2])
    assert sum(per_call) == res.n_llm_calls == judge.n_calls
    assert per_call == [r["n_cut_cand"] + r["n_add_cand"] for r in res.rounds]

    again = run_example(ex, GoldExplicitExtractor(), judge, RELS, max_calls=3)       # judge dùng chung: chỉ đếm phần của ví dụ
    assert again.n_llm_calls == res.n_llm_calls and [r["n_llm_calls"] for r in again.rounds] == per_call


def test_seed_ops_replay_call_one_and_continue_from_call_two():
    """seed_ops: lần gọi 1 lấy từ lượt chạy trước (không hỏi bộ phán), các lần gọi sau chạy tiếp và cho cùng kết quả."""
    ex, full = _run(RICH_BEFORE, RICH_AFTER, 3)
    first = [op for op, k in zip(full.judged_ops, full.judged_rounds) if k == 1]
    assert first and any(k > 1 for k in full.judged_rounds)             # ví dụ này cần cả lần gọi sau
    rec = _Recorder(OracleJudge(ex.after))
    seeded = run_example(ex, GoldExplicitExtractor(), rec, RELS, max_calls=3, seed_ops=list(reversed(first)))
    assert [c.call for c in rec.ctxs] == [k for k in (2, 3) if k <= len(full.rounds)]   # không hỏi bộ phán ở lần gọi 1
    assert seeded.rounds[0]["seeded"] is True and seeded.rounds[0]["n_llm_calls"] == 0
    assert set(seeded.judged_ops) == set(full.judged_ops) and seeded.pred_after == full.pred_after
    assert sorted(zip(map(str, seeded.judged_ops), seeded.judged_rounds)) == sorted(zip(map(str, full.judged_ops), full.judged_rounds))
