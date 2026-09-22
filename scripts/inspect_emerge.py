"""Kiểm chứng mốc tháng 1 trên EMERGE: ops vàng có khớp KG-trước không, và vùng 1-hop quanh
entity được nhắc phủ được bao nhiêu đáp án của từng op (đặc biệt Infer: đầu kia không có trong text).

Dùng: python scripts/inspect_emerge.py [snapshot ...]
"""
from __future__ import annotations

import argparse
import collections
import json
import statistics
from pathlib import Path

from kgu.data.emerge import OPS, load_kg_subset, load_snapshot


def inspect(data_dir: Path, snapshot: str) -> dict:
    kg, _ = load_kg_subset(data_dir / "kg_subsets" / f"snapshot_{snapshot}.tsv")
    by_node: dict[str, set] = collections.defaultdict(set)
    for tr in kg:
        by_node[tr[0]].add(tr)
        by_node[tr[2]].add(tr)
    c: collections.Counter[str] = collections.Counter()
    before_sizes, neigh_sizes = [], []
    for ex in load_snapshot(data_dir, snapshot):
        seeds = set(ex.mentioned)
        before = set().union(*(by_node[q] for q in seeds)) if seeds else set()
        neigh = {n for h, _, t in before for n in (h, t)}
        before_sizes.append(len(before))
        neigh_sizes.append(len(neigh))
        for op in OPS:
            for tr in ex.gold[op]:
                c[f"{op}.n"] += 1
                c[f"{op}.in_kg"] += tr in kg
                c[f"{op}.both_mentioned"] += tr[0] in seeds and tr[2] in seeds
                # đầu không được nhắc có nằm trong vùng 1-hop không (entity mới thì bỏ qua)
                far = [q for q in (tr[0], tr[2]) if q not in seeds and q not in ex.emerging]
                c[f"{op}.far_ends"] += len(far)
                c[f"{op}.far_ends_in_1hop"] += sum(q in neigh for q in far)
    return {
        "snapshot": snapshot, "kg_rows": len(kg),
        "before_edges_median": statistics.median(before_sizes),
        "before_edges_mean": round(statistics.mean(before_sizes), 1),
        "neighbours_median": statistics.median(neigh_sizes),
        **c,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("snapshots", nargs="*")
    ap.add_argument("--data-dir", type=Path, default=Path("data/raw/emerge"))
    args = ap.parse_args()
    snapshots = args.snapshots or sorted(
        p.stem.removeprefix("snapshot_") for p in (args.data_dir / "kg_subsets").glob("snapshot_*.tsv")
    )
    total: collections.Counter[str] = collections.Counter()
    for s in snapshots:
        r = inspect(args.data_dir, s)
        print(json.dumps(r))
        total.update({k: v for k, v in r.items() if "." in k})
    print(f"\n{'op':10s} {'n':>6s} {'in_kg':>7s} {'both_ment':>10s} {'far_ends':>9s} {'far_in_1hop':>12s}")
    for op, name in OPS.items():
        n = total[f"{op}.n"] or 1
        far = total[f"{op}.far_ends"]
        cov = f"{total[f'{op}.far_ends_in_1hop'] / far:.3f}" if far else "-"
        print(f"{name:10s} {total[f'{op}.n']:6d} {total[f'{op}.in_kg'] / n:7.3f} "
              f"{total[f'{op}.both_mentioned'] / n:10.3f} {far:9d} {cov:>12s}")


if __name__ == "__main__":
    main()
