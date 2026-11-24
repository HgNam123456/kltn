"""Bộ sinh KGU-Synth: tái lập được, không tạo quan hệ mới, nhãn trực tiếp / gián tiếp / độ sâu nhất quán với pipeline."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import gen_synth  # noqa: E402
import score_synth  # noqa: E402
from kgu.data.jsonl import load_jsonl  # noqa: E402
from kgu.extract.gold import GoldExplicitExtractor  # noqa: E402
from kgu.judge.oracle import OracleJudge  # noqa: E402
from kgu.pipeline import run_example  # noqa: E402

SCALE = 0.04                                               # mỗi loại sự kiện ít nhất một mẫu


@pytest.fixture(scope="module")
def data():
    return gen_synth.generate(seed=7, scale=SCALE)


def test_generation_is_reproducible(data):
    again = gen_synth.generate(seed=7, scale=SCALE)
    assert again["base"] == data["base"] and again["rows"] == data["rows"] and again["briefs"] == data["briefs"]
    assert gen_synth.generate(seed=8, scale=SCALE)["base"] != data["base"]


def test_every_event_type_is_present(data):
    assert {r["event"] for r in data["rows"]} == {f"{f}/{v}" for f, v, _ in gen_synth.PLAN}


def test_no_new_relation_and_few_new_entities(data):
    base_rel = {r for _, r, _ in data["base"]}
    base_ent = {e for h, _, t in data["base"] for e in (h, t)}
    for row in data["rows"]:
        changed = row["after_add"] + row["after_del"]
        assert {r for _, r, _ in changed} <= base_rel
        new = {e for h, _, t in row["after_add"] for e in (h, t)} - base_ent
        assert new == set(row["extra"]["new_entities"]) and len(new) <= 1
    for rel in gen_synth.DERIVED:                              # quan hệ gián tiếp nào cũng đã có nhiều tiền lệ trong KG nền
        assert sum(1 for _, r, _ in data["base"] if r == rel) >= 100, rel


def test_direct_changes_are_exactly_the_edges_between_mentioned_entities(data):
    for row in data["rows"]:
        mentioned = set(row["mentioned"])
        both = {("INVALIDATE", *tr) for tr in map(tuple, row["after_del"]) if tr[0] in mentioned and tr[2] in mentioned}
        both |= {("ADD", *tr) for tr in map(tuple, row["after_add"]) if tr[0] in mentioned and tr[2] in mentioned}
        assert both == {tuple(op) for op in row["extra"]["direct"]}
        n_changed = len(row["after_add"]) + len(row["after_del"])
        assert len(row["extra"]["direct"]) + len(row["extra"]["indirect"]) == n_changed


def test_depth_labels(data):
    by_event = {r["event"]: r for r in data["rows"]}
    assert by_event["no_change/football"]["extra"]["indirect"] == [] and not by_event["no_change/football"]["after_add"]
    assert by_event["f_transfer/same_league"]["extra"]["max_depth"] == 1
    assert by_event["f_transfer/other_fed"]["extra"]["max_depth"] == 2
    assert by_event["c_move/same_company"]["extra"]["max_depth"] == 1
    assert by_event["c_move/other_city"]["extra"]["max_depth"] == 2
    assert by_event["c_move/other_country"]["extra"]["max_depth"] == 3
    depth_of = {op[2]: op[4] for op in by_event["c_move/other_country"]["extra"]["indirect"]}
    assert depth_of == {"colleague_of": 1, "reports_to": 1, "works_for": 1, "based_in": 2, "resident_of": 3} or \
        depth_of == {"colleague_of": 1, "works_for": 1, "based_in": 2, "resident_of": 3}          # phòng không có trưởng phòng
    for row in data["rows"]:
        assert all(1 <= op[4] <= 3 and op[2] in gen_synth.DERIVED for op in row["extra"]["indirect"])


def test_names_do_not_nest(data):
    assert gen_synth.nested_names(data["names"]) == []
    assert gen_synth.nested_names(["Velaria", "Velaria FC"]) == [("Velaria", "Velaria FC")]


def test_check_text_catches_missing_and_leaked_entities(data):
    index = {tuple(n.split()): n for n in data["names"]}
    row = next(r for r in data["rows"] if r["event"] == "f_transfer/same_league")
    player, old, new = row["mentioned"]
    good = f"{player} has completed a permanent move from {old} to {new}, the buying club's spokesperson said."
    assert gen_synth.check_text(good, row["mentioned"], index) == []
    assert gen_synth.check_text(f"{player}'s switch from {old} to {new} is done, both clubs confirmed today.",
                                row["mentioned"], index) == []                                   # đuôi sở hữu vẫn nhận ra tên
    missing = gen_synth.check_text(f"{player} has left {old} for a new club after several seasons there.", row["mentioned"], index)
    assert missing == [f"thiếu tên: {new}"]
    other = next(n for n in data["names"] if n not in row["mentioned"])
    leaked = gen_synth.check_text(good + f" {other} was not involved.", row["mentioned"], index)
    assert leaked == [f"nhắc thực thể ngoài đề bài: {other}"]
    assert gen_synth.check_text("Too short.", [], index) == ["2 từ (cần 10–130)"]


def _write(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "synth.jsonl"
    head = {"base": [list(tr) for tr in data["base"]], "cumulative": False, "relations": data["relations"]}
    with path.open("w", encoding="utf8") as f:
        f.write(json.dumps(head) + "\n")
        for row in data["rows"]:
            f.write(json.dumps({**row, "text": "placeholder"}) + "\n")
    return path


def test_dataset_loads_and_recursion_reaches_deeper_changes(tmp_path, data):
    """Qua loader KG cho trước; với bước xét duyệt nhìn nhãn vàng, K lớn hơn không bao giờ tìm lại được ít hơn."""
    path = _write(tmp_path, data)
    examples, relations, extras = load_jsonl(path)
    assert relations == data["relations"] and len(examples) == len(data["rows"])
    gold = score_synth.load_gold(path)
    scores = {}
    for k in (1, 2, 3):
        records = []
        for ex in examples:
            res = run_example(ex, GoldExplicitExtractor(), OracleJudge(ex.after), relations, max_calls=k)
            assert res.pred_after <= ex.before | ex.after          # nhãn vàng không bao giờ thêm cạnh sai
            assert {(o.kind.value, *o.triple) for o in res.explicit_ops} == \
                {tuple(op) for op in extras[ex.idx]["direct"]}
            records.append(res.to_record(ex))
        scores[k] = score_synth.score_run(gold, records, by="family")
    correct = {k: sum(row["correct"] for row in s["by_depth"]) for k, s in scores.items()}
    assert correct[1] <= correct[2] <= correct[3]
    assert correct[2] > correct[1]                               # có thay đổi chỉ tìm được ở lần gọi thứ hai
    for s in scores.values():
        assert all(side["wrong"] == 0 for side in s["by_side"].values())
    deep = [row for row in scores[1]["by_depth"] if row["depth"] == 3 and row["side"] == "INVALIDATE"][0]
    deeper = [row for row in scores[3]["by_depth"] if row["depth"] == 3 and row["side"] == "INVALIDATE"][0]
    assert deeper["correct"] > deep["correct"]
    assert scores[1]["groups"]["no_change"]["gold"] == 0 and scores[1]["groups"]["no_change"]["exact"] >= 1


def test_net_ops_drops_contradictory_verdicts():
    judged = [{"kind": "ADD", "h": "a", "r": "r", "t": "b", "round": 1},
              {"kind": "INVALIDATE", "h": "c", "r": "r", "t": "d", "round": 2},
              {"kind": "INVALIDATE", "h": "e", "r": "r", "t": "f", "round": 1},
              {"kind": "ADD", "h": "e", "r": "r", "t": "f", "round": 2}]
    assert score_synth.net_ops(judged) == {("ADD", "a", "r", "b"): 1, ("INVALIDATE", "c", "r", "d"): 2}
