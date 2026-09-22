"""Executable-R trên EMERGE: baseline kèm sẵn có QID (KG-aware GPT-5.1, relik-cie không chấm được vì
không khai op) đặt cạnh các baseline CẤU TRÚC không gọi LLM của mình.

Ứng viên cấu trúc = cạnh KG-trước mà cả hai đầu đều được nhắc trong passage.
  lookup_exists : coi mọi ứng viên là Exists (cận trên recall Exists của bước tra KG)
  lookup_oracle : ứng viên ∩ gold — cận trên của bộ phán khi chỉ được chọn trong ứng viên

Dùng: python scripts/emerge_baselines.py [--limit-per-snapshot N]
"""
from __future__ import annotations

import argparse
import collections
import json
import statistics
from pathlib import Path

from kgu.data.emerge import OPS, load_kg_subset, load_snapshot
from kgu.eval.emerge import executable_scores, shipped_predictions

SHIPPED = ["kg-aware/gpt-5.1/oracle", "kg-aware/gpt-5.1/kg_rag_32", "kg-aware/gpt-5.1/oracle_kg_rag"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data/raw/emerge"))
    ap.add_argument("--limit-per-snapshot", type=int, default=None)
    args = ap.parse_args()

    examples, preds = [], collections.defaultdict(dict)
    n_cand = []
    for kg_path in sorted((args.data_dir / "kg_subsets").glob("snapshot_*.tsv")):
        snapshot = kg_path.stem.removeprefix("snapshot_")
        kg, _ = load_kg_subset(kg_path)
        out_edges: dict[str, list] = collections.defaultdict(list)
        for tr in kg:
            out_edges[tr[0]].append(tr)
        exs = load_snapshot(args.data_dir, snapshot, args.limit_per_snapshot)
        raw = {}
        for path in sorted((args.data_dir / "evaluation_set" / f"snapshot_{snapshot}").glob("delta_*.jsonl")):
            with path.open(encoding="utf8") as f:
                for line in f:
                    d = json.loads(line)
                    raw[d["hash_id"]] = d
        for ex in exs:
            seeds = set(ex.mentioned)
            cand = {tr for q in seeds for tr in out_edges[q] if tr[2] in seeds}
            n_cand.append(len(cand))
            preds["lookup_exists"][ex.hash_id] = {"x-triples": cand}
            preds["lookup_oracle"][ex.hash_id] = {op: cand & ex.gold[op] for op in ("x-triples", "d-triples")}
            for m in SHIPPED:
                preds[m][ex.hash_id] = shipped_predictions(raw[ex.hash_id], m)
        examples.extend(exs)

    print(f"instances {len(examples)} · ứng viên/instance: trung vị {statistics.median(n_cand)}, "
          f"trung bình {statistics.mean(n_cand):.1f}, max {max(n_cand)}")
    ops = ["x-triples", "e-triples", "d-triples"]
    print(f"{'model':34s} " + " ".join(f"{OPS[o] + ' R/P':>16s}" for o in ops))
    for m in SHIPPED + ["lookup_exists", "lookup_oracle"]:
        s = executable_scores(examples, preds[m])
        print(f"{m:34s} " + " ".join(f"{s[o]['recall']:8.3f}/{s[o]['precision']:.3f} " for o in ops))


if __name__ == "__main__":
    main()
