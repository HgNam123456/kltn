"""Áp chuẩn hóa Add v2 (KGShape: đảo chiều / sửa quan hệ / thêm chiều nghịch đảo theo cấu trúc KG-trước) lên một
jsonl Add đã chạy, không gọi LLM. Ghi jsonl mới với `added` = sau chuẩn hóa, `added_raw` = trước; in exact-match.

Dùng: python scripts/normalize_emerge_add.py --in results/emerge_dev350_add_v1.jsonl --out results/emerge_dev350_add_v2.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from kgu.data.emerge import load_kg_subset
from kgu.judge.emerge_add import KGShape


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data/raw/emerge"))
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    with args.inp.open(encoding="utf8") as f:
        rows = [json.loads(line) for line in f]
    shapes: dict[str, KGShape] = {}
    stat = {"gold": 0, "pred_raw": 0, "pred": 0, "tp_raw": 0, "tp": 0}
    for r in rows:
        snap = r["snapshot"]
        if snap not in shapes:
            kg, _ = load_kg_subset(args.data_dir / "kg_subsets" / f"snapshot_{snap}.tsv")
            shapes[snap] = KGShape(kg)
        raw = {tuple(t) for t in r.get("added_raw") or r["added"]}
        norm = shapes[snap].normalize(raw)
        r["added_raw"], r["added"] = sorted(raw), sorted(norm)
        gold = {tuple(t) for t in r["gold"]["e-triples"]}
        stat["gold"] += len(gold)
        stat["pred_raw"] += len(raw)
        stat["pred"] += len(norm)
        stat["tp_raw"] += len(gold & raw)
        stat["tp"] += len(gold & norm)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(rows)} instances; gold {stat['gold']}: raw tp {stat['tp_raw']} / pred {stat['pred_raw']} -> "
          f"norm tp {stat['tp']} / pred {stat['pred']}  (micro R {stat['tp_raw']/stat['gold']:.3f} -> {stat['tp']/stat['gold']:.3f})")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
