"""Chấm kết quả của run_emerge.py bằng bộ chấm mềm viết lại từ EMERGE (C và G-BERTScore-R, thang paper),
kèm các baseline có sẵn trong dataset ở format kg-prompt (kg-aware/*) để đối chiếu với bảng chính thức.

Dùng (trên Kaggle, GPU):
  python scripts/score_emerge_soft.py --judge results/emerge_dev350_v3_kaggle.jsonl \
      --add results/emerge_dev350_add_v1_kaggle.jsonl --baselines kg-aware/gpt-5.1/oracle \
      --out results/emerge_dev350_v3_addv1.soft.json
Không có --judge/--add thì chấm baseline trên các bài --per-delta đầu của mỗi delta.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from kgu.data.emerge import load_kg_subset, load_snapshot
from kgu.eval.emerge import OPERATION_TO_OP
from kgu.eval.emerge_soft import format_table, instance_labels, load_models, soft_scores, to_text


def load_results(path: Path | None) -> dict[str, dict]:
    if path is None:
        return {}
    with path.open(encoding="utf8") as f:
        return {r["hash_id"]: r for r in map(json.loads, f)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data/raw/emerge"))
    ap.add_argument("--judge", type=Path, default=None, help="jsonl của run_emerge.py (Exists/Deprecate)")
    ap.add_argument("--add", type=Path, default=None, help="jsonl của run_emerge.py --judge add")
    ap.add_argument("--model-name", default="mine/gemma4-e4b")
    ap.add_argument("--baselines", default="", help="model kg-prompt trong dataset, cách nhau bằng dấu phẩy")
    ap.add_argument("--per-delta", type=int, default=10, help="khi không có --judge/--add: số bài đầu mỗi delta")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    judged, added = load_results(args.judge), load_results(args.add)
    keep = set(judged) | set(added)
    baselines = [b for b in args.baselines.split(",") if b]

    examples = []
    mine: dict[str, dict[str, set]] = defaultdict(lambda: defaultdict(set))
    base: dict[str, dict[str, dict[str, set]]] = {b: defaultdict(lambda: defaultdict(set)) for b in baselines}
    for kg_path in sorted((args.data_dir / "kg_subsets").glob("snapshot_*.tsv")):
        snapshot = kg_path.stem.removeprefix("snapshot_")
        _, kg_labels = load_kg_subset(kg_path)
        by_id = {ex.hash_id: ex for ex in load_snapshot(args.data_dir, snapshot)}
        for path in sorted((args.data_dir / "evaluation_set" / f"snapshot_{snapshot}").glob("delta_*.jsonl")):
            with path.open(encoding="utf8") as f:
                for i, line in enumerate(f):
                    d = json.loads(line)
                    h = d["hash_id"]
                    if (h not in keep) if keep else (i >= args.per_delta):
                        continue
                    ex = by_id[h]
                    examples.append(ex)
                    lab = instance_labels(ex, kg_labels)
                    j = judged.get(h, {})
                    for v in j.get("verdicts", []):
                        mine[h]["d-triples" if v[3] == "ended" else "x-triples"].add(to_text(v[:3], lab))
                    for t in added.get(h, {}).get("added", []):
                        mine[h]["e-triples"].add(to_text(t, lab))
                    for b in baselines:
                        for t in d["predictions"].get(b, {}).get("predicted_triples") or []:
                            op = OPERATION_TO_OP.get(str(t.get("operation")).upper())
                            labels = t.get("triple_labels")
                            if op and labels and len(labels) == 3 and all(labels):
                                base[b][h][op].add(tuple(labels))
    print(f"{len(examples)} instances", flush=True)

    st, bs = load_models(args.device)
    scores = {}
    if judged or added:
        scores[args.model_name] = soft_scores(examples, mine, st, bs)
    for b in baselines:
        scores[b] = soft_scores(examples, base[b], st, bs)
    print(format_table(scores))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(scores, ensure_ascii=False, indent=1), encoding="utf8")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
