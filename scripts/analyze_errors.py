"""Nhóm kết quả eval theo loại để tìm chỗ pipeline yếu."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from kgu.eval.metrics import Counts, summarize


def _bucket(n: int) -> str:
    return "0-10" if n <= 10 else "11-30" if n <= 30 else "31+"


def _key(rec: dict, by: str) -> str:
    if by == "event":
        return str(rec.get("event", ""))
    if by == "n_gold":
        return _bucket(rec["counts"]["n_add"] + rec["counts"]["n_del"])
    if by == "coverage":
        c = rec["coverage"]
        return "full" if c[0] == c[1] else "partial"
    raise ValueError(by)


def group_records(records: list[dict], by: str) -> dict[str, Counts]:
    groups: dict[str, Counts] = defaultdict(Counts)
    for rec in records:
        k = _key(rec, by)
        groups[k] = groups[k] + Counts(**rec["counts"])
    return dict(groups)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", type=Path)
    ap.add_argument("--by", default="event", choices=["event", "n_gold", "coverage"])
    args = ap.parse_args()
    records = [json.loads(l) for l in args.results.read_text(encoding="utf8").splitlines() if l.strip()]
    sizes: dict[str, int] = defaultdict(int)
    cands: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for rec in records:
        k = _key(rec, args.by)
        sizes[k] += 1
        cands[k][0] += rec["n_cut_cand"]
        cands[k][1] += rec["n_add_cand"]
    print(f"{'group':<14}{'n':>6}{'f1':>8}{'add':>8}{'del':>8}{'cut/ex':>8}{'add/ex':>8}")
    for k, c in sorted(group_records(records, args.by).items()):
        s = summarize(c)
        n = sizes[k]
        print(f"{k:<14}{n:>6}{s['f1']:>8.3f}{s['add_acc']:>8.3f}{s['del_acc']:>8.3f}"
              f"{cands[k][0]/n:>8.1f}{cands[k][1]/n:>8.1f}")


if __name__ == "__main__":
    main()
