"""Kiểm tra scripts/night_report.py trên cây thư mục giả có đúng định dạng JSON của các kernel, và trên thư mục rỗng."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_spec = importlib.util.spec_from_file_location("night_report", Path(__file__).parents[1] / "scripts" / "night_report.py")
nr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(nr)

EMB, THR = nr.EMBEDDER, nr.THRESHOLD


def put(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf8")


def jac(m: float) -> dict:
    return {"jaccard": {"mean": m, "std": 0.1, "groups": 100}, "jaccard_triple": {"mean": 0.5, "std": 0.1, "groups": 100}}


def build_tree(root: Path) -> None:
    for rep, f1 in (("r2", 0.40), ("r3", 0.50)):
        out = root / "covid_repeat" / rep / "out"
        put(out / "stability.json", {"systems": {"atom": {f"r1-{rep}": jac(0.4 if rep == "r2" else 0.3)}}})
        sysm = {f"atom.{rep}": {"P": 0.6, "R": 0.5, "F1": f1}, "atom.r1": {"P": 0.6, "R": 0.5, "F1": 0.30}}
        put(out / "scores_covid.json", {"results": {EMB: {THR: sysm}}})
    lane1 = root / "night" / "lane1" / "out"
    put(lane1 / "lane.json", {"stages": [{"name": "nyt11", "status": "ok", "minutes": 123.4, "commit": "abcdef0123"}]})
    put(lane1 / "nyt11" / "summary.json", {
        "dataset": "nyt11",
        "edc_script": {"mine": {"Partial": {"P": 1, "R": 1, "F1": 0.321}, "Strict": {"P": 1, "R": 1, "F1": 0.2},
                                "Exact": {"P": 1, "R": 1, "F1": 0.1}, "lines": 10, "scored": 9, "empty": 1, "avg_triples": 2.5}},
        "scores_exact": {"mine": {"exact": {"P": 0.25, "R": 0.5, "F1": 0.3333}, "nocase": {"P": 0.3, "R": 0.5, "F1": 0.375},
                                  "unreadable_lines": 0, "lines": 10}, "atom": {"exact": {"P": 0.1, "R": 0.2, "F1": 0.1333}}},
        "support": {"nyt11/mine": {"triples": 20, "judged": 19, "support_rate": 0.789, "support_rate_doc_mean": 0.8, "errors": 1}}})
    put(lane1 / "knet_atom" / "summary.json", {"scores_knet": {"atom_edc": {
        "span_overlap": {"P": 0.45, "R": 0.35, "F1": 0.3938}, "span_exact": {"P": 0.4, "R": 0.3, "F1": 0.343},
        "unmapped": 7, "predicted": 123}}})
    lane2 = root / "night" / "lane2" / "out"
    put(lane2 / "quality" / "support.json", {"mine/mine": {"triples": 100, "judged": 100, "support_rate": 0.9123,
                                                           "support_rate_doc_mean": 0.9, "errors": 0}})
    put(lane2 / "quality" / "redundancy.json", {"covid/atom": {"entities": 50, "entity_near_pairs_per_100": 7.77,
                                                                "triple_exact_dups": 4, "scope": "global"}})
    put(root / "knet_baseline" / "out" / "summary.json", {"readme_dev_span_overlap": {"P": 0.718, "R": 0.691, "F1": 0.704}})
    put(root / "mine" / "v3" / "tables.json", {"all": {"mine": 85.4}})


def test_report_numbers(tmp_path: Path) -> None:
    build_tree(tmp_path)
    out = tmp_path / "night" / "REPORT.md"
    assert nr.main(["--root", str(tmp_path), "--out", str(out)]) == 0
    text = out.read_text(encoding="utf8")
    for expected in ["40.0% ± 10.0%", "r1-r3", "55.2%", "21.2%", "40.0%", "50.0%", "40.0% ± 8.2%",     # ổn định, F1 theo lượt
                     "123.4", "abcdef0", "91.2%", "7.77", "32.1%", "33.3%", "(OIE thô)", "39.4%", "123",
                     "README, không chạy lại được", "70.4%", "85.4%", "93,3"]:
        assert expected in text, expected
    assert "lỗi:" not in text


def test_empty_root(tmp_path: Path) -> None:
    out = tmp_path / "REPORT.md"
    assert nr.main(["--root", str(tmp_path / "none"), "--out", str(out)]) == 0
    text = out.read_text(encoding="utf8")
    assert "chưa có" in text and "lỗi:" not in text
