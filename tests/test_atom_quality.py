import numpy as np

from kgu.eval.atom_quality import dates, match_row, summarize


def test_dates_accepts_list_string_and_day_first_formats():
    assert dates(["2020-01-28"]) == dates("2020-01-28") == dates("28-01-2020")
    assert dates([]) == dates(None) == dates("") == dates("None") == frozenset()


def test_match_row_is_greedy_one_to_one_above_threshold():
    pred = [("a", "r", "b", ["2020-01-01"], []), ("a", "r", "b2", [], [])]
    gold = [("a", "r", "b", ["2020-01-01"], []), ("c", "r", "d", [], [])]
    sim = np.array([[0.95, 0.80], [0.90, 0.10]])
    out = match_row(pred, gold, sim, threshold=0.7)
    # (0,0)=0.95 thắng trước; (1,0) bị chặn vì gold 0 đã dùng; (0,1) bị chặn vì pred 0 đã dùng
    assert out == {"match": 1, "hall": 1, "om": 1, "match_t": 1, "om_t": 0, "hall_t": 0, "n_pred": 2, "n_gold": 2}


def test_temporal_categories_follow_atom_priority():
    gold = [("a", "r", "b", ["2020-01-01"], [])]
    sim = np.array([[0.99]])
    missing = match_row([("a", "r", "b", [], [])], gold, sim, 0.7)
    extra = match_row([("a", "r", "b", ["2020-01-01"], ["2020-02-01"])], gold, sim, 0.7)
    assert (missing["om_t"], missing["match_t"]) == (1, 0)
    assert (extra["hall_t"], extra["match_t"]) == (1, 0)


def test_empty_prediction_counts_as_full_omission_not_skipped():
    gold = [("a", "r", "b", [], [])]
    out = match_row([], gold, np.zeros((0, 1)), 0.7)
    assert out["match"] == 0 and out["om"] == 1
    s = summarize([out, {"match": 1, "hall": 0, "om": 0, "match_t": 1, "om_t": 0, "hall_t": 0, "n_pred": 1, "n_gold": 1}])
    assert s["R"] == 0.5 and s["n_rows"] == 2


def test_dates_are_day_first_and_year_only_is_deterministic():
    import datetime as dt
    assert dates("05-03-2020") == {dt.date(2020, 3, 5)}                 # ATOM ghi DD-MM-YYYY
    assert dates("2020") == {dt.date(2020, 1, 1)} and dates("01-2020") == {dt.date(2020, 1, 1)}
