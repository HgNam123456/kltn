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
