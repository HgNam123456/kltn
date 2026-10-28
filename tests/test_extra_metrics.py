import importlib.util
from pathlib import Path

import numpy as np


def _script(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1] / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


nyt = _script("score_nyt11")
red = _script("kg_redundancy")
judge = _script("judge_support")
outputs = _script("kg_outputs")


def test_score_nyt11_exact_and_nocase():
    gold = [[("Tim Pawlenty", "/people/person/place_lived", "Minnesota")], [("Guinea", "/location/country/capital", "Conakry")]]
    pred = [[("tim pawlenty", "/people/person/place_lived", "Minnesota"), ("Minnesota", "/location/location/contains", "X")],
            [("Guinea", "/location/country/capital", "Conakry")]]
    exact, nocase = nyt.score(gold, pred, False), nyt.score(gold, pred, True)
    assert (exact["tp"], exact["pred"], exact["gold"]) == (1, 3, 2)
    assert (nocase["tp"], nocase["F1"]) == (2, 0.8)
    assert nocase["by_relation"]["/location/country/capital"]["F1"] == 1.0


def test_redundancy_counts_near_entity_pairs_and_exact_triple_dups():
    emb = {"obama": np.array([1.0, 0.0]), "barack obama": np.array([0.96, 0.28]), "chicago": np.array([0.0, 1.0]),
           "obama born in chicago": np.array([1.0, 0.0]), "barack obama born in chicago": np.array([0.99, 0.14])}
    docs = [{"id": 0, "text": "", "triples": [["Obama", "born_in", "Chicago"], ["obama", "born in", "chicago"],
                                              ["Barack Obama", "born in", "Chicago"]]}]
    m = red.measure(docs, emb, "doc", 0.8, 0.9)
    assert m["entities"] == 3 and m["entity_near_pairs"] == 1
    assert m["triple_exact_dups"] == 1 and m["triple_near_pairs"] == 1


def test_judge_summary_ignores_unjudged_triples():
    recs = [{"doc": 0, "triple": ["a", "r", "b"], "supported": True},
            {"doc": 0, "triple": ["a", "r", "c"], "supported": False},
            {"doc": 1, "triple": ["d", "r", "e"], "supported": None, "error": "x"}]
    s = judge.summarize(recs)
    assert (s["judged"], s["supported"], s["support_rate"], s["errors"]) == (2, 1, 0.5, 1)
    assert judge.prompt("T", [["a", "r", "b"]], 1).endswith("1. a | r | b")


def test_lines_loader_pairs_texts_with_prediction_lines(tmp_path):
    (tmp_path / "in.txt").write_text("first\nsecond\n", encoding="utf8")
    (tmp_path / "p.txt").write_text("[['a', 'r', 'b'], ['a', 'r', 'b']]\nnot a list\n", encoding="utf8")
    name, docs = outputs.load(f"x=lines:{tmp_path / 'p.txt'}", outputs.Sources(lines_input=tmp_path / "in.txt"))
    assert name == "x" and docs[0]["triples"] == [["a", "r", "b"]] and docs[1]["triples"] == []


knet = _script("score_knet")


def test_knet_spans_respect_word_boundaries_and_pick_closest_pair():
    text = "The house where he lived; he was born in Paris."
    assert knet.find_spans(text, "he") == [(16, 18), (26, 28)]
    facts, unmapped = knet.to_facts([["He", "PLACE_OF_BIRTH", "Paris"], ["Bob", "PLACE_OF_BIRTH", "Paris"]],
                                    {"passageText": text, "passageStart": 100}, {"PLACE_OF_BIRTH": "12"})
    assert unmapped == 1 and facts[0]["subjectStart"] == 126 and facts[0]["objectStart"] == 141
