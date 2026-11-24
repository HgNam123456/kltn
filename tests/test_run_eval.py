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


# ---- khoanh vùng đệ quy: --max-calls (K) ---------------------------------------------------------------------

def test_effective_max_calls_keeps_naive_baseline_at_one_call():
    assert mod.effective_max_calls("llm", 3) == 3 and mod.effective_max_calls("oracle", 2) == 2
    assert mod.effective_max_calls("naive", 3) == 1                 # hệ so sánh 1 bước không dùng khoanh vùng
    assert mod.effective_max_calls("llm", 0) == 1


def test_run_one_passes_max_calls(before, after):
    from kgu.extract.gold import GoldExplicitExtractor
    from kgu.judge.oracle import OracleJudge
    rec1 = mod.run_one(_ex(before, after), GoldExplicitExtractor(), OracleJudge(after), [PLAYS, TEAM])
    rec3 = mod.run_one(_ex(before, after), GoldExplicitExtractor(), OracleJudge(after), [PLAYS, TEAM], max_calls=3)
    assert rec1["max_calls"] == 1 and len(rec1["rounds"]) == 1
    assert rec3["max_calls"] == 3 and len(rec3["rounds"]) == 2      # lần 2 không kết luận gì → dừng trước lần 3
    assert rec1["counts"] == rec3["counts"]


def test_summarize_run_old_style_records_count_as_first_call():
    s = mod.summarize_run([_scored_record(n_cut=2, n_add=1, n_judged=1, cov=(1, 2), n_calls=5)],
                          seconds=1.0, config="c", split="test")
    assert s["max_calls"] == 1 and s["llm_calls_by_call"] == [5] and s["coverage_by_call"] == [0.5]
    assert s["coverage"] == 0.5 and s["llm_calls"] == 5
    empty = mod.summarize_run([], seconds=0.1, config="c", split="test")
    assert empty["max_calls"] == 1 and empty["llm_calls_by_call"] == [0] and empty["coverage_by_call"] == [0.0]


def test_summarize_run_reports_calls_and_cumulative_coverage_per_call():
    def rec(covs, rounds):
        r = _scored_record(n_cut=4, n_add=2, n_judged=3, cov=(sum(c > 0 for c in covs), len(covs)),
                           n_calls=sum(n for _, n in rounds))
        r["max_calls"] = 3
        r["rounds"] = [{"k": k, "n_cut_cand": 1, "n_add_cand": 1, "n_ops": 1, "n_llm_calls": n} for k, n in rounds]
        r["implied"] = [{"kind": "ADD", "h": "a", "r": "r", "t": str(i), "cov": c} for i, c in enumerate(covs)]
        return r
    records = [rec([1, 1, 2, 3], [(1, 4), (2, 2), (3, 1)]), rec([1, 2, 0, 0], [(1, 3), (2, 1)]),
               {"idx": 9, "event": "x", "error": "RuntimeError: boom"}]
    s = mod.summarize_run(records, seconds=1.0, config="c[K=3]", split="test", max_calls=3)
    assert s["max_calls"] == 3
    assert s["llm_calls_by_call"] == [7, 3, 1] and sum(s["llm_calls_by_call"]) == s["llm_calls"] == 11
    assert s["coverage_by_call"] == [3 / 8, 5 / 8, 6 / 8] and s["coverage"] == 6 / 8
    assert mod.summarize_run(records, 1.0, "c", "test")["max_calls"] == 3      # không truyền → lấy từ record


def test_main_max_calls_on_jsonl_chain(tmp_path, monkeypatch):
    """Chạy trọn run_eval trên một tệp JSONL nhỏ có chuỗi phụ thuộc 3 tầng: K quyết định độ phủ và tag cấu hình."""
    import json
    import sys
    from tests.test_recursive import CHAIN_AFTER, CHAIN_BEFORE
    data = tmp_path / "chain.jsonl"
    row = {"idx": 0, "event": "move", "text": "e moves from D1 to D2.", "mentioned": ["e", "D1", "D2"],
           "before": sorted(CHAIN_BEFORE), "after": sorted(CHAIN_AFTER)}
    data.write_text(json.dumps(row) + "\n", encoding="utf8")
    got = {}
    for k in (None, 2, 3):
        out = tmp_path / f"k{k}.jsonl"
        argv = ["run_eval", "--jsonl", str(data), "--extractor", "gold_explicit", "--judge", "oracle", "--out", str(out)]
        monkeypatch.setattr(sys, "argv", argv + (["--max-calls", str(k)] if k else []))
        mod.main()
        got[k] = (json.loads(out.with_suffix(".summary.json").read_text(encoding="utf8")),
                  json.loads(out.read_text(encoding="utf8").splitlines()[0]))
    s1, r1 = got[None]
    assert s1["config"] == "gold_explicit+oracle" and s1["max_calls"] == 1          # tag cũ không đổi khi K = 1
    assert s1["coverage_by_call"] == [2 / 6] and r1["max_calls"] == 1 and len(r1["rounds"]) == 1
    s2, _ = got[2]
    assert s2["config"] == "gold_explicit+oracle[K=2]" and s2["coverage_by_call"] == [2 / 6, 4 / 6]
    s3, r3 = got[3]
    assert s3["config"] == "gold_explicit+oracle[K=3]" and s3["coverage_by_call"] == [2 / 6, 4 / 6, 1.0]
    assert s3["add_acc"] == 1.0 and s3["del_acc"] == 1.0 and s3["f1"] == 1.0
    assert s1["add_acc"] == 0.5 and s2["add_acc"] == 0.75
    assert s3["llm_calls_by_call"] == [0, 0, 0] and [r["k"] for r in r3["rounds"]] == [1, 2, 3]
