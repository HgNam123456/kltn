from kgu.eval.metrics import Counts, score, summarize

A, B, C, D, E = [("x", "r", str(i)) for i in range(5)]


def test_perfect_prediction():
    before, gold = {A, B, C}, {B, C, D}
    c = score(before, gold, pred_after=gold)
    assert (c.tp, c.fp, c.fn) == (3, 0, 0)
    assert (c.add_hit, c.n_add, c.del_hit, c.n_del) == (1, 1, 1, 1)
    s = summarize(c)
    assert s["f1"] == 1.0 and s["add_acc"] == 1.0 and s["del_acc"] == 1.0


def test_noop_prediction_keeps_before():
    before, gold = {A, B, C}, {B, C, D}
    c = score(before, gold, pred_after=before)
    # universe = {A,B,C,D}; pred = {A,B,C}; gold = {B,C,D}
    assert (c.tp, c.fp, c.fn, c.tn) == (2, 1, 1, 0)
    assert (c.add_hit, c.del_hit) == (0, 0)
    s = summarize(c)
    assert s["precision"] == 2 / 3 and s["recall"] == 2 / 3
    assert s["add_acc"] == 0.0 and s["del_acc"] == 0.0


def test_extra_prediction_outside_universe_counts_as_fp():
    before, gold = {A}, {A}
    c = score(before, gold, pred_after={A, E})
    assert (c.tp, c.fp, c.fn) == (1, 1, 0)


def test_counts_add_and_zero_division():
    c = Counts() + Counts(tp=1)
    assert c.tp == 1
    s = summarize(Counts())
    assert s == {"precision": 0.0, "recall": 0.0, "f1": 0.0, "add_acc": 0.0, "del_acc": 0.0}
