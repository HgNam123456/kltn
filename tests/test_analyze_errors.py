import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("analyze_errors", Path("scripts/analyze_errors.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def _rec(event, n_add, n_del, cov):
    return {"event": event, "coverage": cov, "n_cut_cand": 3, "n_add_cand": 2,
            "counts": {"tp": 5, "fp": 1, "fn": 1, "tn": 0, "add_hit": 1, "n_add": n_add, "del_hit": 1, "n_del": n_del}}


def test_group_by_event():
    g = mod.group_records([_rec("trade", 2, 2, [4, 4]), _rec("trade", 1, 1, [1, 2]), _rec("waive", 0, 5, [0, 0])], "event")
    assert set(g) == {"trade", "waive"}
    assert g["trade"].tp == 10 and g["waive"].n_del == 5


def test_group_by_n_gold_buckets():
    g = mod.group_records([_rec("x", 2, 2, [0, 0]), _rec("x", 20, 5, [0, 0]), _rec("x", 30, 30, [0, 0])], "n_gold")
    assert set(g) == {"0-10", "11-30", "31+"}


def test_group_by_coverage():
    g = mod.group_records([_rec("x", 2, 2, [4, 4]), _rec("x", 2, 2, [1, 4])], "coverage")
    assert set(g) == {"full", "partial"}


def test_group_records_skips_error_records():
    g = mod.group_records([_rec("trade", 2, 2, [4, 4]), {"idx": 3, "event": "trade", "error": "RuntimeError: boom"}], "event")
    assert g["trade"].n_add == 2
