"""Chỉ số gộp tên của ATOM (kgu.eval.atom_merge) trên KG cuối của từng biến thể, 2020-COVID-NYT (spec
docs/superpowers/specs/2026-10-15-condense-canonicalize-design.md §3). Gold = mọi 5-tuple gold của các nhóm ngày trong
--data. Nạp embedder → chạy trên Kaggle GPU (kaggle/covid_v2).

Dùng: python scripts/score_covid_merge.py --data covid.jsonl --kg v0=out/v0.jsonl.kg.json [--kg ...]
      [--embedder sentence-transformers/all-mpnet-base-v2] [--device cuda] [--threshold 0.8] --out merge_scores.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from kgu.eval.atom_merge import gold_mentions, kg_names, resolution
from kgu.extract.canonicalize import sentence_embedder


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--kg", action="append", default=[], help="tên=đường_dẫn.kg.json")
    ap.add_argument("--embedder", default="sentence-transformers/all-mpnet-base-v2")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threshold", type=float, default=0.8)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rows = [json.loads(line) for line in args.data.read_text(encoding="utf8").splitlines() if line.strip()]
    gold = [q for r in rows for q in r["gold"]]
    embed = sentence_embedder(args.embedder, args.device)
    out = {}
    for spec in args.kg:
        name, path = spec.split("=", 1)
        facts = json.loads(Path(path).read_text(encoding="utf8"))
        out[name] = {kind: resolution(kg_names(facts, kind), gold_mentions(gold, kind), embed, args.threshold, kind)
                     for kind in ("entity", "relation")}
        print(name, out[name], flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf8")


if __name__ == "__main__":
    main()
