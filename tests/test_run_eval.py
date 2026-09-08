import importlib.util
from pathlib import Path

from kgu.data.nba import Example
from kgu.extract.gold import GoldAllExtractor
from tests.conftest import PLAYS, TEAM

spec = importlib.util.spec_from_file_location("run_eval", Path("scripts/run_eval.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class _Boom:
    n_calls = 0

    def extract(self, ex, graph):
        raise RuntimeError("boom")


def _ex(before, after):
    return Example(7, "trade", "2021", "messi joins inter", ["Messi", "Barca", "Inter"], before, after)


def test_run_one_returns_record_on_success(before, after):
    rec = mod.run_one(_ex(before, after), GoldAllExtractor(), None, [PLAYS, TEAM])
    assert rec["idx"] == 7 and "counts" in rec and "error" not in rec


def test_run_one_returns_error_record_instead_of_raising(before, after):
    rec = mod.run_one(_ex(before, after), _Boom(), None, [PLAYS, TEAM])
    assert rec == {"idx": 7, "event": "trade", "error": "RuntimeError: boom"}
