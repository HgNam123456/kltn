import importlib.util
import itertools
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("edc_eval_fast", ROOT / "scripts" / "edc_eval_fast.py")
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)


def first_best(M):
    """Đúng cách script gốc chọn: hoán vị đầu tiên theo thứ tự itertools có tổng lớn nhất."""
    best, best_s = None, None
    for perm in itertools.permutations(range(len(M)), len(M)):
        s = 0
        for i, j in enumerate(perm):
            s += M[i][j]
        if best is None or s > best_s:
            best, best_s = list(perm), s
    return best


def test_hungarian_path_picks_the_same_permutation_as_full_enumeration_with_ties():
    rng = random.Random(0)
    for n in (8, 9):
        for _ in range(4):                             # giá trị rời rạc → nhiều cặp hòa điểm, tổng cộng chính xác
            M = [[rng.choice([0, 0, 0, 0.25, 0.5, 1]) for _ in range(n)] for _ in range(n)]
            assert m.best_perm(M) == first_best(M)


def test_padded_rows_of_zeros_keep_lexicographic_order():
    M = [[1, 0, 0, 0, 0, 0, 0, 0, 0]] + [[0] * 9 for _ in range(8)]
    assert m.best_perm(M) == list(range(9))


def test_small_matrices_use_full_enumeration():
    M = [[0.5, 1], [1, 0.5]]
    assert m.best_perm(M) == [1, 0] and m.best_perm([[0.3]]) == [0]


def test_assign_value_is_the_optimum():
    M = [[4, 1, 3], [2, 0, 5], [3, 2, 2]]
    assert m._assign_value(M, [0, 1, 2], [0, 1, 2]) == max(sum(M[i][p[i]] for i in range(3))
                                                            for p in itertools.permutations(range(3)))
