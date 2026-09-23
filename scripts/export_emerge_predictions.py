"""Xuất prediction của mình theo format EMERGE (kiểu `kg-prompt`, như baseline KG-aware) để chấm bằng
`evaluate.sh` chính chủ. Ghi lại cây `evaluation_set/snapshot_*/delta_*.jsonl` chỉ gồm các instance đã chạy,
mỗi instance thêm `predictions[<model>]`.

Dùng: python scripts/export_emerge_predictions.py --judge results/emerge_dev350_v3_softyear.jsonl \
          --add results/emerge_dev350_add_v1.jsonl --model mine/gemma4-e4b --out data/export/dev350
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from kgu.data.emerge import load_kg_subset

OP_NAME = {"x-triples": "EXISTS", "d-triples": "DEPRECATE", "e-triples": "ADD"}


def load_results(path: Path | None) -> dict[str, dict]:
    if path is None:
        return {}
    out = {}
    with path.open(encoding="utf8") as f:
        for line in f:
            r = json.loads(line)
            out[r["hash_id"]] = r
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data/raw/emerge"))
    ap.add_argument("--judge", type=Path, required=True, help="jsonl của run_emerge.py (Exists/Deprecate)")
    ap.add_argument("--add", type=Path, default=None, help="jsonl của run_emerge.py --judge add")
    ap.add_argument("--model", default="mine/gemma4-e4b")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    judged, added = load_results(args.judge), load_results(args.add)
    keep = set(judged) | set(added)
    n_inst = n_pred = 0
    for kg_path in sorted((args.data_dir / "kg_subsets").glob("snapshot_*.tsv")):
        snapshot = kg_path.stem.removeprefix("snapshot_")
        _, labels = load_kg_subset(kg_path)
        src_dir = args.data_dir / "evaluation_set" / f"snapshot_{snapshot}"
        for path in sorted(src_dir.glob("delta_*.jsonl")):
            rows = []
            with path.open(encoding="utf8") as f:
                for line in f:
                    d = json.loads(line)
                    if d["hash_id"] not in keep:
                        continue
                    lab = dict(labels)
                    for t in d["tkgu_triples"]:
                        lab.update(zip(t["triple"], t["triple_labels"]))
                    for m in d["mentions"]:
                        if m.get("qid"):
                            lab.setdefault(m["qid"], m["mention_text"])
                    triples: dict[tuple, str] = {}
                    j = judged.get(d["hash_id"], {})
                    ended = {tuple(v[:3]) for v in j.get("verdicts", []) if v[3] == "ended"}
                    for v in j.get("verdicts", []):
                        tr = tuple(v[:3])
                        triples[tr] = "DEPRECATE" if tr in ended else "EXISTS"      # Exists = mọi ứng viên không ended
                    for tr in added.get(d["hash_id"], {}).get("added", []):
                        triples.setdefault(tuple(tr), "ADD")
                    preds = [{
                        "extracted_relation": [lab.get(h, h), lab.get(r, r), lab.get(t, t)],
                        "triple_qids": [h, r, t],
                        "triple_labels": [lab.get(h, h), lab.get(r, r), lab.get(t, t)],
                        "operation": op, "minted": [False, False, False],
                    } for (h, r, t), op in sorted(triples.items())]
                    d["predictions"][args.model] = {
                        "predicted_triples": preds, "model": args.model, "model_type": "kg-prompt",
                        "model_config_name": "kgu-lookup+judge",
                    }
                    n_inst += 1
                    n_pred += len(preds)
                    rows.append(d)
            if rows:
                dest = args.out / "evaluation_set" / f"snapshot_{snapshot}" / path.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                with dest.open("w", encoding="utf8") as f:
                    for d in rows:
                        f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"exported {n_inst} instances, {n_pred} predicted triples for {args.model} -> {args.out}")


if __name__ == "__main__":
    main()
