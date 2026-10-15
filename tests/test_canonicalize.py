import numpy as np

from kgu.extract.canonicalize import candidate_groups, kmeans


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
