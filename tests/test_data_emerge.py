import json
from pathlib import Path

from kgu.data.emerge import ASSESSOR, load_kg_subset, load_snapshot, to_example


def _triple(h, r, t, ops, labels, emerging_head=False, verified=True):
    return {
        "triple": [h, r, t], "triple_labels": labels, "tkgu_operations": ops,
        "emerging_head": emerging_head, "emerging_tail": False,
        "llm_assessment": [
            {"llm_name": ASSESSOR, "llm_prompt_type": p, "llm_assessment": verified}
            for p in ("triple_assertion", "triple_deprecation")
        ] + [{"llm_name": "Meta-Llama-3.1-8B", "llm_prompt_type": "triple_assertion", "llm_assessment": True}],
    }


def _write_fixture(d: Path) -> None:
    snap = d / "evaluation_set" / "snapshot_2019-01-01"
    snap.mkdir(parents=True)
    inst = {
        "hash_id": "abc", "anchor_title": "Messi", "passage": "Messi left Barca for Inter.",
        "mentions": [
            {"mention_text": "Messi", "qid": "Q1", "start_char": 0, "end_char": 5},
            {"mention_text": "Barca", "qid": "Q2", "start_char": 11, "end_char": 16},
            {"mention_text": "Messi", "qid": "Q1", "start_char": 0, "end_char": 5},
            {"mention_text": "nowhere", "qid": None, "start_char": 0, "end_char": 1},
        ],
        "tkgu_triples": [
            _triple("Q1", "P54", "Q2", ["d-triples"], ["Messi", "team", "Barca"]),
            _triple("Q1", "P54", "Q3", ["d-triples", "e-triples"], ["Messi", "team", "Inter"]),
            _triple("Q9", "P31", "Q5", ["ee-kg-triples"], ["Deal", "instance of", "transfer"], True),
            _triple("Q9", "P17", "Q6", ["ee-kg-triples"], ["Deal", "country", "Italy"], True, verified=False),
        ],
    }
    (snap / "delta_2019-01-08.jsonl").write_text(json.dumps(inst) + "\n", encoding="utf8")
    (snap / "delta_2019-01-15.jsonl").write_text(json.dumps(inst) + "\n", encoding="utf8")


def test_load_snapshot_groups_gold_by_op(tmp_path):
    _write_fixture(tmp_path)
    exs = load_snapshot(tmp_path, "2019-01-01")
    assert [e.delta for e in exs] == ["2019-01-08", "2019-01-15"]
    ex = exs[0]
    assert ex.mentioned == ["Q1", "Q2"]
    assert ex.gold["d-triples"] == {("Q1", "P54", "Q2"), ("Q1", "P54", "Q3")}
    assert ex.gold["e-triples"] == {("Q1", "P54", "Q3")}
    assert ex.gold["x-triples"] == set()
    assert ex.gold["ee-kg-triples"] == {("Q9", "P31", "Q5")}          # chỉ assessor 405B tính, 8B thì không
    assert ex.unverified["ee-kg-triples"] == {("Q9", "P17", "Q6")}
    assert ex.emerging == {"Q9"}
    assert ex.labels["P54"] == "team" and ex.labels["Q3"] == "Inter"


def test_load_snapshot_limit(tmp_path):
    _write_fixture(tmp_path)
    assert len(load_snapshot(tmp_path, "2019-01-01", limit=1)) == 1


def test_kg_subset_and_to_example(tmp_path):
    _write_fixture(tmp_path)
    kg_path = tmp_path / "kg.tsv"
    kg_path.write_text(
        "Q1\tP54\tQ2\tMessi\tteam\tBarca\n"
        "Q7\tP54\tQ2\tPedri\tteam\tBarca\n"
        "Q8\tP54\tQ3\tLautaro\tteam\tInter\n",
        encoding="utf8",
    )
    kg, labels = load_kg_subset(kg_path)
    assert len(kg) == 3 and labels["Q7"] == "Pedri"
    ex = to_example(load_snapshot(tmp_path, "2019-01-01")[0], 0, kg)
    assert ex.before == {("Q1", "P54", "Q2"), ("Q7", "P54", "Q2")}   # Lautaro–Inter ngoài 1-hop
    assert ("Q1", "P54", "Q2") not in ex.after
    assert {("Q1", "P54", "Q3"), ("Q9", "P31", "Q5"), ("Q7", "P54", "Q2")} <= ex.after
