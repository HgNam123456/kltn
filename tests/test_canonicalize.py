import json

import numpy as np

from kgu.extract.canonicalize import Canonicalizer, apply_mapping, candidate_groups, kmeans
from kgu.extract.text_kg import TemporalKG
from kgu.llm import FakeLLMClient


def unit(*v):
    a = np.array(v, dtype=float)
    return a / np.linalg.norm(a)


NAMES = ["1,694 Cases", "1,700 Cases", "Italy", "WHO", "World Health Org"]
X = np.array([unit(0, 1, 0.10), unit(0, 1, 0.12), unit(0, 0, 1), unit(1, 0.05, 0), unit(1, 0.1, 0)])
COUNTS = {"1,694 Cases": 1, "1,700 Cases": 1, "Italy": 3, "WHO": 2, "World Health Org": 1}
ONE = np.zeros(5, dtype=int)


def test_kmeans_is_deterministic_and_separates_blobs():
    Y = np.array([unit(1, 0), unit(0.99, 0.1), unit(0, 1), unit(0.1, 0.99)])
    a, b = kmeans(Y, 2, seed=0), kmeans(Y, 2, seed=0)
    assert (a == b).all() and a[0] == a[1] != a[2] == a[3]


def test_kmeans_caps_k_at_number_of_points():
    assert sorted(kmeans(np.array([unit(1, 0), unit(0, 1)]), 5).tolist()) == [0, 1]


def test_candidate_groups_link_close_names_with_the_same_numbers_only():
    groups = candidate_groups(NAMES, X, ONE, COUNTS, tau=0.75)
    assert [g["members"] for g in groups[0]] == [["WHO", "World Health Org"]]   # 2 "Cases" khác số → không cặp
    assert groups[0][0]["anchor"] == "WHO" and groups[0][0]["cos"]["World Health Org"] > 0.99   # hòa tâm → nhiều fact


def test_candidate_groups_skip_blocked_pairs_and_cap_group_size():
    assert candidate_groups(NAMES, X, ONE, COUNTS, 0.75, blocked={frozenset(("WHO", "World Health Org"))}) == {}
    names = [f"Name {c}" for c in "ABCDEFGHIJ"]
    Y = np.array([unit(1, 0.01 * i) for i in range(10)])
    g = candidate_groups(names, Y, np.zeros(10, dtype=int), dict.fromkeys(names, 1), 0.75)[0][0]
    assert len(g["members"]) == 8 and g["members"][0] == g["anchor"]


def test_candidate_groups_anchor_is_nearest_to_group_centre():
    names = ["Rome", "Roma", "Rome City"]
    Y = np.array([unit(1, 0), unit(1, 0.2), unit(1, -0.2)])
    g = candidate_groups(names, Y, np.zeros(3, dtype=int), {"Rome": 1, "Roma": 5, "Rome City": 5}, 0.75)[0][0]
    assert g["anchor"] == "Rome"


AXES = ["WHO", "Countries", "Pandemic", "Italy", "5 Cases", "warns", "declares", "reports", "is located in",
        "Rome", "Roma", "Rome Airport", "Colosseum", "has", "serves"]


def fake_embed(scale=1.0):
    """Mỗi tên một trục; WHO/World Health Organization và Rome/Roma/Rome Airport sát nhau. scale ≠ 1 → vector chưa
    chuẩn hóa."""
    def vec(t):
        v = np.zeros(len(AXES) + 1)
        if t == "World Health Organization":
            v[0], v[-1] = 1, 0.1
        elif t in ("Roma", "Rome Airport"):
            v[AXES.index("Rome")], v[-1] = 1, 0.1 if t == "Roma" else -0.1
        else:
            v[AXES.index(t)] = 1
        return v * scale
    seen: list[str] = []

    def embed(texts):
        seen.extend(texts)
        return np.array([vec(t) for t in texts])
    embed.seen = seen
    return embed


def who_kg():
    kg = TemporalKG()
    kg.facts = [["WHO", "warns", "Countries", [], []], ["WHO", "declares", "Pandemic", [], []],
                ["World Health Organization", "warns", "Italy", [], []], ["Italy", "reports", "5 Cases", [], []],
                ["Italy", "is_located_in", "Pandemic", [], []]]
    return kg


def test_round_merges_yes_candidates_keeps_aliases_and_logs(tmp_path):
    kg, embed, log = who_kg(), fake_embed(scale=3.0), tmp_path / "m.jsonl"
    llm = FakeLLMClient([{"groups": [{"group": 0, "decisions": [
        {"name": "WHO", "reason": "anchor itself", "same": True},
        {"name": "World Health Organization", "reason": "same agency", "same": True}],
        "canonical": "World Health Organization"}]}])
    canon = Canonicalizer(embed, llm, log_path=log)
    ent, rel = canon.run_round(kg)
    assert (ent["calls"], ent["merged"], rel["calls"]) == (1, 1, 0)
    assert [f[0] for f in kg.facts[:3]] == ["World Health Organization"] * 3
    assert kg.aliases == {"World Health Organization": {"WHO"}}
    assert "is located in" in embed.seen                          # relation nhúng với dấu cách thay "_"
    user = llm.calls[0][1]
    assert "ANCHOR: WHO" in user and "CANDIDATE: World Health Organization" in user and "WHO | warns | Countries" in user
    recs = [json.loads(line) for line in log.read_text(encoding="utf8").splitlines()]
    assert len(recs) == 1                                          # quyết định cho chính tên mốc bị bỏ qua
    assert recs[0]["name"] == "World Health Organization" and recs[0]["same"] is True
    assert recs[0]["canonical"] == "World Health Organization" and "WHO | warns | Countries" in recs[0]["anchor_context"]
    assert canon.run_round(kg)[0]["calls"] == 0                   # không có tên mới → không hỏi lại


def test_no_decisions_are_remembered_and_invalid_canonical_falls_back_to_anchor():
    kg = TemporalKG()
    kg.facts = [["Rome", "has", "Colosseum", [], []], ["Roma", "has", "Colosseum", [], []],
                ["Rome Airport", "serves", "Italy", [], []], ["WHO", "warns", "Italy", [], []],
                ["World Health Organization", "warns", "Countries", [], []]]
    llm = FakeLLMClient([{"groups": [{"group": 0, "decisions": [
        {"name": "Roma", "reason": "Italian name of Rome", "same": True},
        {"name": "Rome Airport", "reason": "an airport, not the city", "same": False},
        {"name": "WHO", "reason": "belongs to the other group", "same": True}],
        "canonical": "Rome Airport"}]}])                       # nhóm 1 (WHO) không được trả lời → không gộp
    canon = Canonicalizer(fake_embed(), llm)
    canon.run(kg, "entity")
    assert kg.facts == [["Rome", "has", "Colosseum", [], []], ["Rome Airport", "serves", "Italy", [], []],
                        ["WHO", "warns", "Italy", [], []], ["World Health Organization", "warns", "Countries", [], []]]
    assert ("entity", frozenset(("Rome", "Rome Airport"))) in canon.rejected
    assert kg.aliases == {"Rome": {"Roma"}}


def test_failed_cluster_call_leaves_kg_unchanged(tmp_path):
    kg, log = who_kg(), tmp_path / "m.jsonl"
    before = [list(f) for f in kg.facts]
    stats = Canonicalizer(fake_embed(), FakeLLMClient([]), log_path=log).run(kg, "entity")
    assert (stats["calls"], stats["errors"], stats["merged"]) == (1, 1, 0) and kg.facts == before
    assert "error" in json.loads(log.read_text(encoding="utf8").splitlines()[0])


def test_round_on_empty_kg_does_nothing():
    embed, llm = fake_embed(), FakeLLMClient([])
    stats = Canonicalizer(embed, llm).run_round(TemporalKG())
    assert [s["names"] for s in stats] == [0, 0] and embed.seen == [] and llm.n_calls == 0


def test_apply_mapping_dedups_drops_self_loops_and_moves_aliases():
    kg = TemporalKG()
    kg.facts = [["A", "r", "B", [], []], ["A2", "r", "B", [], []], ["A2", "r", "A", [], []],
                ["A", "r", "B", ["2020-01-01"], []]]
    kg.aliases = {"A2": {"A3"}}
    apply_mapping(kg, "entity", {"A2": "A"})
    assert kg.facts == [["A", "r", "B", [], []], ["A", "r", "B", ["2020-01-01"], []]]
    assert kg.aliases == {"A": {"A2", "A3"}}
    apply_mapping(kg, "relation", {"r": "rel"})
    assert [f[1] for f in kg.facts] == ["rel", "rel"] and kg.aliases == {"A": {"A2", "A3"}}


def test_failed_cluster_is_retried_next_round(tmp_path):
    kg, llm = who_kg(), FakeLLMClient([])
    canon = Canonicalizer(fake_embed(), llm, log_path=tmp_path / "m.jsonl")
    ent, _ = canon.run_round(kg)
    assert (ent["calls"], ent["errors"]) == (1, 1) and kg.facts == who_kg().facts
    llm.responses.append({"groups": [{"group": 0, "decisions": [
        {"name": "WHO", "reason": "anchor itself", "same": True},
        {"name": "World Health Organization", "reason": "same agency", "same": True}],
        "canonical": "World Health Organization"}]})
    ent, _ = canon.run_round(kg)                                   # cụm lỗi được hỏi lại
    assert (ent["calls"], ent["errors"], ent["merged"]) == (1, 0, 1)
    assert kg.aliases == {"World Health Organization": {"WHO"}} and kg.facts[0][0] == "World Health Organization"
