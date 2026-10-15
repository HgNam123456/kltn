import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _mod(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_load_predictions_tuples_and_keeps_empty_rows(tmp_path):
    p = tmp_path / "p.jsonl"
    p.write_text(json.dumps({"row": 0, "quintuples": [["a", "r", "b", [], []]]}) + "\n"
                 + json.dumps({"row": 3, "quintuples": []}) + "\n", encoding="utf8")
    assert _mod("score_covid_nyt").load_predictions(p) == {0: [("a", "r", "b", [], [])], 3: []}


def test_score_counts_empty_rows_and_reports_skip_variant():
    m = _mod("score_covid_nyt")
    rows = [{"row": 0, "gold": [("a", "r", "b", [], [])]}, {"row": 1, "gold": [("c", "r", "d", [], [])]}]
    emb = {"a r b": np.array([1.0, 0.0]), "c r d": np.array([0.0, 1.0])}
    out = m.score(rows, {0: [("a", "r", "b", [], [])]}, emb, [0.7, 0.9])
    assert out[0.7]["R"] == 0.5 and out[0.7]["R_skip"] == 1.0 and out[0.7]["n_rows_skip"] == 1


def test_pick_closest_to_paper_and_keeps_atom_above_direct():
    m = _mod("score_covid_nyt")
    res = {"e1": {"0.7": {"gpt41-direct": {"F1_skip": 0.50}, "gpt41-atom": {"F1_skip": 0.64}},
                  "0.8": {"gpt41-direct": {"F1_skip": 0.60}, "gpt41-atom": {"F1_skip": 0.55}}},
           "e2": {"0.7": {"gpt41-direct": {"F1_skip": 0.40}, "gpt41-atom": {"F1_skip": 0.70}}}}
    assert (m.pick(res)["embedder"], m.pick(res)["threshold"]) == ("e1", "0.7")


def test_webnlg_line_format_matches_edc_canon_kg():
    m = _mod("run_webnlg")
    assert m.format_line([("A", "leader", "B")]) == "[['A', 'leader', 'B']]"
    assert m.format_line([]) == "[]"


def test_webnlg_schema_reader_keeps_descriptions_with_commas(tmp_path):
    m = _mod("run_webnlg")
    (tmp_path / "s.csv").write_text('leader,The leader of x\ncountry,"Country, of x"\n\n', encoding="utf8")
    assert m.read_schema(tmp_path / "s.csv") == {"leader": "The leader of x", "country": "Country, of x"}


def test_webnlg_run_keeps_line_order_and_writes_empty_list_on_error(tmp_path):
    m = _mod("run_webnlg")

    def extract(text):
        if text == "bad":
            raise ValueError("unparseable")
        return [(text, "r", "x")]
    assert m.run_all(["a", "bad", "c"], extract, workers=3) == ["[['a', 'r', 'x']]", "[]", "[['c', 'r', 'x']]"]


def test_covid_runner_goes_in_date_order_and_flushes_each_row(tmp_path):
    m = _mod("run_covid_nyt")
    rows = [{"row": 0, "date": "2020-02-01", "text": "b"}, {"row": 1, "date": "2020-01-01", "text": "a"}]
    seen = []

    def step(text, date):
        seen.append(date)
        if text == "b":
            raise RuntimeError("boom")
        return [("h", "r", "t", [date], [])]
    out = tmp_path / "o.jsonl"
    m.run_rows(rows, step, out)
    assert seen == ["2020-01-01", "2020-02-01"]
    got = [json.loads(line) for line in out.read_text(encoding="utf8").splitlines()]
    assert [(g["row"], g["quintuples"]) for g in got] == [(1, [["h", "r", "t", ["2020-01-01"], []]]), (0, [])]


def test_atom_relationship_to_quintuple_normalizes_dates_and_drops_broken():
    from types import SimpleNamespace as NS
    m = _mod("atom_covid")
    rel = NS(startNode=NS(name="Virus"), name="occurs_in", endNode=NS(name="China"), t_start="2020-01-24", t_end=[None, ""])
    assert m.to_quintuple(rel) == ("Virus", "occurs_in", "China", ["2020-01-24"], [])
    assert m.to_quintuple(NS(startNode=None, name="r", endNode=NS(name="x"), t_start=[], t_end=[])) is None


def test_atom_flatten_skips_failed_containers():
    from types import SimpleNamespace as NS
    m = _mod("atom_covid")
    rel = NS(startNode=NS(name="a"), name="r", endNode=NS(name="b"), t_start=[], t_end=[])
    outs = [NS(relationships=[rel, None]), None, ValueError("bad json"), NS(relationships=None)]
    assert m.flatten(outs) == [("a", "r", "b", [], [])]


def test_kggen_relations_become_untimed_quintuples_sorted():
    m = _mod("kggen_covid")
    assert m.to_quintuples({("b", "r", "c"), ("a", "r", "b")}) == [("a", "r", "b", [], []), ("b", "r", "c", [], [])]


def test_atom_schema_hint_carries_field_descriptions():
    from pydantic import BaseModel, Field
    m = _mod("atom_covid")

    class AtomicFact(BaseModel):
        atomic_fact: list[str] = Field(description="Decompose the paragraph into atomic facts.")
    assert "Decompose the paragraph into atomic facts." in m.schema_hint(AtomicFact)


def test_covid_runner_adds_row_stats_and_calls_after_each_group(tmp_path):
    m = _mod("run_covid_nyt")
    rows = [{"row": 0, "date": "2020-01-02", "text": "b"}, {"row": 1, "date": "2020-01-01", "text": "a"}]
    done = []
    out = tmp_path / "o.jsonl"
    m.run_rows(rows, lambda text, date: [], out, extra=lambda: {"calls": 2, "sentences": 3}, after=done.append)
    got = [json.loads(line) for line in out.read_text(encoding="utf8").splitlines()]
    assert [(g["row"], g["calls"], g["sentences"]) for g in got] == [(1, 2, 3), (0, 2, 3)] and done == [1, 2]


def test_audit_sample_is_stratified_deterministic_and_skips_errors():
    m = _mod("audit_merges")
    recs = [{"id": i, "same": i % 3 == 0} for i in range(300)] + [{"id": 300, "error": "x"}]
    a = m.sample(recs, n=10, seed=0)
    assert a == m.sample(recs, n=10, seed=0) and len(a) == 10 and sum(r["same"] for r in a) == 5
    assert [r["id"] for r in a] == sorted(r["id"] for r in a)


def test_audit_sample_fills_from_the_other_side_when_one_is_short():
    m = _mod("audit_merges")
    recs = [{"id": 0, "same": True}] + [{"id": i, "same": False} for i in range(1, 20)]
    a = m.sample(recs, n=10)
    assert len(a) == 10 and sum(r["same"] for r in a) == 1


def test_audit_summary_counts_only_labelled_rows():
    m = _mod("audit_merges")
    rows = [{"same": "True", "correct": "1"}, {"same": "True", "correct": "0"},
            {"same": "False", "correct": "1"}, {"same": "False", "correct": " "}]
    assert m.summarize(rows) == {"yes": {"n": 2, "correct": 1, "accuracy": 0.5},
                                 "no": {"n": 1, "correct": 1, "accuracy": 1.0}}
