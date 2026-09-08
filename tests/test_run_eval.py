import importlib.util
from pathlib import Path

from kgu.data import Example
from kgu.extract.gold import GoldAllExtractor
from tests.conftest import PLAYS, TEAM

spec = importlib.util.spec_from_file_location(
    "run_eval", Path(__file__).resolve().parents[1] / "scripts" / "run_eval.py"
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class _Boom:
    n_calls = 0

    def extract(self, ex, graph, at):
        raise RuntimeError("boom")


def _ex(before, after):
    return Example(7, "trade", "2021", "messi joins inter", ["Messi", "Barca", "Inter"], before, after)


def test_run_one_returns_record_on_success(before, after):
    rec = mod.run_one(_ex(before, after), GoldAllExtractor(), None, [PLAYS, TEAM])
    assert rec["idx"] == 7 and "counts" in rec and "error" not in rec


def test_run_one_returns_error_record_instead_of_raising(before, after):
    rec = mod.run_one(_ex(before, after), _Boom(), None, [PLAYS, TEAM])
    assert rec == {"idx": 7, "event": "trade", "error": "RuntimeError: boom"}


def _scored_record(n_cut, n_add, n_judged, cov=(1, 1), n_calls=1):
    counts = {"tp": 1, "fp": 0, "fn": 0, "tn": 0, "add_hit": 1, "n_add": 1, "del_hit": 1, "n_del": 1}
    return {
        "idx": 0, "event": "trade", "counts": counts, "coverage": list(cov),
        "n_cut_cand": n_cut, "n_add_cand": n_add, "n_judged": n_judged, "n_llm_calls": n_calls,
    }


def test_summarize_run_divides_averages_by_n_scored_not_n():
    records = [
        _scored_record(n_cut=2, n_add=1, n_judged=1),
        {"idx": 1, "event": "trade", "error": "RuntimeError: boom"},
    ]
    s = mod.summarize_run(records, seconds=1.0, config="gold_explicit+llm", split="test")
    assert s["n"] == 2 and s["n_scored"] == 1 and s["errors"] == 1
    assert s["avg_cut_cand"] == 2.0 and s["avg_add_cand"] == 1.0
    assert s["judged_ratio"] == 1 / 3          # n_judged=1 / (n_cut_cand=2 + n_add_cand=1)
    assert s["llm_calls"] == 1


def test_summarize_run_handles_no_records_and_no_candidates():
    s = mod.summarize_run([], seconds=0.5, config="gold_all+none", split="test")
    assert s["n"] == 0 and s["n_scored"] == 0 and s["errors"] == 0
    assert s["avg_cut_cand"] == 0.0 and s["avg_add_cand"] == 0.0 and s["judged_ratio"] == 0.0


def test_workers_produce_same_records_in_order(tmp_path, monkeypatch):
    """--workers N chạy song song nhưng kết quả và thứ tự ghi phải y hệt chạy tuần tự."""
    import json
    import sys
    data = Path(__file__).resolve().parents[1] / "data" / "raw" / "nba"
    if not data.exists():
        import pytest
        pytest.skip("chưa tải dataset")
    outs = []
    for workers in (1, 3):
        out = tmp_path / f"w{workers}.jsonl"
        monkeypatch.setattr(sys, "argv", ["run_eval", "--extractor", "gold_explicit", "--judge", "oracle",
                                          "--limit", "6", "--workers", str(workers), "--out", str(out)])
        mod.main()
        outs.append([json.loads(l) for l in out.read_text(encoding="utf8").splitlines()])
    assert [r["idx"] for r in outs[1]] == [r["idx"] for r in outs[0]] == list(range(6))
    assert outs[0] == outs[1]
