"""Chấm NYT11-HRL kiểu các paper trích triple (HRL, CasRel…): một triple dự đoán đúng khi (thực thể đầu, quan hệ,
thực thể đuôi) trùng hệt một triple gold cùng câu; P/R/F1 micro trên cả tập. Hai mức: exact (chỉ gộp khoảng trắng) và
nocase (thêm bỏ phân biệt hoa/thường). Kèm F1 theo từng quan hệ. File định dạng canon_kg.txt của EDC (str(list triple)
mỗi dòng); dòng không đọc được = không dự đoán (ghi số dòng như vậy).

Dùng: python scripts/score_nyt11.py --ref ref.txt --pred mine=canon_kg.txt [--pred ...] --out scores.json
"""
from __future__ import annotations

import argparse
import ast
import json
from collections import Counter
from pathlib import Path


def read_lines(path: Path) -> tuple[list[list[tuple]], int]:
    out, bad = [], 0
    for line in path.read_text(encoding="utf8").splitlines():
        try:
            ts = ast.literal_eval(line.strip()) if line.strip() else []
            out.append([tuple(map(str, t)) for t in ts if len(t) == 3])
        except Exception:
            out.append([])
            bad += 1
    return out, bad


def norm(t: tuple, nocase: bool) -> tuple:
    f = (lambda s: " ".join(s.split()).lower()) if nocase else (lambda s: " ".join(s.split()))
    return tuple(f(x) for x in t)


def prf(tp: int, n_pred: int, n_gold: int) -> dict:
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gold if n_gold else 0.0
    return {"P": round(p, 4), "R": round(r, 4), "F1": round(2 * p * r / (p + r), 4) if p + r else 0.0,
            "tp": tp, "pred": n_pred, "gold": n_gold}


def score(gold: list[list[tuple]], pred: list[list[tuple]], nocase: bool) -> dict:
    tp = n_pred = n_gold = 0
    by_rel: dict[str, Counter] = {}
    for g, p in zip(gold, pred + [[]] * (len(gold) - len(pred))):
        gs, ps = {norm(t, nocase) for t in g}, {norm(t, nocase) for t in p}
        tp, n_pred, n_gold = tp + len(gs & ps), n_pred + len(ps), n_gold + len(gs)
        for t in gs | ps:
            c = by_rel.setdefault(t[1], Counter())
            c["tp"] += t in gs and t in ps
            c["pred"] += t in ps
            c["gold"] += t in gs
    out = prf(tp, n_pred, n_gold)
    out["by_relation"] = {r: prf(c["tp"], c["pred"], c["gold"]) for r, c in sorted(by_rel.items()) if c["gold"]}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--pred", action="append", default=[])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    gold, _ = read_lines(args.ref)
    out = {}
    for spec in args.pred:
        name, path = spec.split("=", 1)
        pred, bad = read_lines(Path(path))
        out[name] = {"exact": score(gold, pred, False), "nocase": score(gold, pred, True), "unreadable_lines": bad,
                     "lines": len(pred)}
        print(name, {k: out[name][k]["F1"] for k in ("exact", "nocase")}, flush=True)
    args.out.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf8")


if __name__ == "__main__":
    main()
