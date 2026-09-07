from kgu.data.nba import Example
from kgu.extract.gold import GoldAllExtractor, GoldExplicitExtractor
from kgu.judge.oracle import OracleJudge
from kgu.pipeline import run_example
from kgu.eval.metrics import score, summarize
from tests.conftest import PLAYS, TEAM


def _ex(before, after):
    return Example(3, "trade", "2021", "messi joins inter", ["Messi", "Barca", "Inter"], before, after)


def test_gold_all_without_judge_is_perfect(before, after):
    res = run_example(_ex(before, after), GoldAllExtractor(), None, [PLAYS, TEAM])
    assert res.pred_after == after
    assert summarize(score(before, after, res.pred_after))["f1"] == 1.0


def test_gold_explicit_without_judge_misses_implied_edges(before, after):
    res = run_example(_ex(before, after), GoldExplicitExtractor(), None, [PLAYS, TEAM])
    s = summarize(score(before, after, res.pred_after))
    assert s["add_acc"] == 1 / 3 and s["del_acc"] == 1 / 3


def test_gold_explicit_with_oracle_is_perfect(before, after):
    res = run_example(_ex(before, after), GoldExplicitExtractor(), OracleJudge(after), [PLAYS, TEAM])
    assert res.pred_after == after
    rec = res.to_record(_ex(before, after))
    assert rec["idx"] == 3 and rec["n_explicit"] == 2 and rec["coverage"] == [4, 4]
    assert rec["counts"]["tp"] == len(after)
