"""Bảng MINE-1 từ các lượt chấm đã có (không gọi LLM): điểm theo bài của từng hệ trên đủ 100 bài và trên đúng tập bài mà
mỗi KG phát hành có mặt (sau realign), cột *_accuracy của dataset (judge gốc của paper) gán về đúng bài, độ dài ngữ cảnh
truy xuất, và mức đồng ý từng fact giữa judge Gemma với evaluation gốc trong *_responses.

Dùng: python scripts/mine_tables.py --data mine.parquet --out results/kaggle/mine/v3/tables.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASED = {"kggen": ("kggen_responses", "kggen_accuracy"), "graphrag_kg": ("graphrag_responses", "graphrag_accuracy"),
            "openie_kg": ("openie_responses", "openie_accuracy")}


def _score_mine():
    spec = importlib.util.spec_from_file_location("score_mine", ROOT / "scripts" / "score_mine.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf8").splitlines() if line.strip()]


def per_essay(recs: list[dict]) -> dict[int, list[int]]:
    d: dict[int, list[int]] = defaultdict(list)
    for r in recs:
        d[int(r["id"])].append(r["evaluation"] if isinstance(r["evaluation"], int) else 0)
    return d


def mean_pct(xs) -> float | None:
    xs = list(xs)
    return round(100 * sum(xs) / len(xs), 2) if xs else None


def main() -> None:
    import pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--built", type=Path, default=ROOT / "results/kaggle/mine/v2", help="scores.mine/kggen.jsonl")
    ap.add_argument("--released", type=Path, default=ROOT / "results/kaggle/mine/v3", help="scores.released_*.jsonl")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    sm = _score_mine()
    norm = sm._norm
    recs = {"mine": load(args.built / "scores.mine.jsonl"), "kggen-gemma": load(args.built / "scores.kggen.jsonl")}
    for c in RELEASED:
        recs[f"released:{c}"] = load(args.released / f"scores.released_{c}.jsonl")
    essays = {n: per_essay(r) for n, r in recs.items()}
    ids = sorted(essays["mine"])
    acc = lambda n, sub: mean_pct(sum(essays[n][i]) / len(essays[n][i]) for i in sub)

    rows = pd.read_parquet(args.data).to_dict("records")
    out: dict = {"all": {"essays": len(ids), **{n: acc(n, ids) for n in essays}}, "subsets": {}, "context_chars": {},
                 "judge_agreement": {}}
    for n, rs in recs.items():
        lens = [len(r["context"]) for r in rs if r["context"]]
        out["context_chars"][n] = round(sum(lens) / len(lens)) if lens else 0
    for c, (resp, acc_col) in RELEASED.items():
        # KG kèm điểm judge gốc của cùng hàng -> realign gán cả hai về đúng bài
        src = [{**r, c: None if r[c] is None else {"entities": [str(e) for e in r[c]["entities"]],
                                                    "relations": len(r[c]["relations"]), "acc": r[acc_col]}} for r in rows]
        graphs, _ = sm.realign(src, c, resp)
        sub = sorted(i for i in graphs if i in essays["mine"] and graphs[i]["relations"])   # như no_graph của score_mine
        orig = [graphs[i]["acc"] for i in sub if graphs[i]["acc"] == graphs[i]["acc"] and graphs[i]["acc"] is not None]
        out["subsets"][c] = {"essays": len(sub), **{n: acc(n, sub) for n in essays},
                             "dataset_judge": mean_pct(orig), "dataset_judge_essays": len(orig)}
        # đồng ý từng fact (ngữ cảnh của mình từ kg-gen 0.4.0, của paper trong retrieved_context — gần như không trùng)
        original = {norm(a["correct_answer"]): a.get("evaluation") for r in rows for a in
                    (r[resp] if r[resp] is not None else []) if a.get("correct_answer")}
        pairs = [(int(r["evaluation"] == 1), int(original[norm(r["query"])] == 1)) for r in recs[f"released:{c}"]
                 if r["context"] and norm(r["query"]) in original]
        out["judge_agreement"][c] = {"facts": len(pairs), "gemma_yes": mean_pct(g for g, _ in pairs),
                                     "original_yes": mean_pct(o for _, o in pairs),
                                     "agree": mean_pct(int(g == o) for g, o in pairs)}
    args.out.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf8")
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
