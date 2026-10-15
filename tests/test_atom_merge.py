import numpy as np

from kgu.eval.atom_merge import gold_mentions, kg_names, resolution

E = {"A": [1, 0, 0], "A2": [1, 0, 0], "B": [0, 1, 0], "a": [1, 0, 0], "b": [0, 1, 0], "c": [0, 0, 1],
     "r": [1, 0, 0], "s": [0, 1, 0]}


def embed(texts):
    return np.array([E[t] for t in texts], dtype=float)


def test_entity_resolution_matches_hand_computation():
    r = resolution(["A", "A2", "B"], ["A", "A", "B", "b", "c"], embed, 0.8, "entity")
    # KG: 1 cặp gần trùng (A, A2); cần gộp = 5 lần xuất hiện − 3 tên KG = 2 → P = 1 − 1/2;
    # gold: 5 − 3 tên khác nhau (a, b, c) − 0 cặp gần trùng = 2 → R = 1 − 1/2
    assert (r["precision"], r["recall"], r["f1"], r["similar_pairs"], r["gold_merged"]) == (0.5, 0.5, 0.5, 1, 2)


def test_precision_when_nothing_to_merge_follows_atom_per_kind():
    assert resolution(["r", "s"], ["r", "s"], embed, kind="relation")["precision"] == 1.0
    assert resolution(["A", "B"], ["A", "B"], embed, kind="entity")["precision"] == 0.0
    assert resolution(["A", "B"], ["A", "B"], embed, kind="entity")["recall"] == 1.0


def test_names_and_mentions():
    facts = [["A", "r", "B", [], []], ["A", "s", "C", [], []]]
    assert kg_names(facts, "entity") == ["A", "B", "C"] and kg_names(facts, "relation") == ["r", "s"]
    gold = [("A", "r", "B", [], []), ("A", "r", "C", [], [])]
    assert gold_mentions(gold, "entity") == ["A", "A", "B", "C"] and gold_mentions(gold, "relation") == ["r", "r"]
