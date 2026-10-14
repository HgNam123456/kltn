"""Chấm 5-tuple trên 2020-COVID-NYT (HF lairgiyassir/2020-COVID-NYT, 122 nhóm ngày 01–06/2020, 3.918 gold) bằng
metric ATOM (kgu.eval.atom_quality) với embedder local. Đọc data/covid_nyt/2020_covid_nyt.jsonl
(scripts/convert_covid_nyt.py), không nạp pickle. Nạp embedder → chạy trên Kaggle GPU (kaggle/covid_calib, home_bench).

Mỗi câu "h r t" chỉ nhúng một lần cho mọi hệ thống; một lượt chấm nhiều embedder × ngưỡng.
Khóa *_skip = bỏ bài hệ thống không ra gì (như script gốc của ATOM); khóa thường = tính bài đó 0 điểm.
--calibrate: chấm 6 cột GPT-4.1 có sẵn (quintuples_gpt41[_runN] = trích thẳng, quintuples_gpt41_from_factoids[_runN] =
ATOM; trung bình 3 lượt) và chọn embedder/ngưỡng có F1_skip gần Table 2 của paper ATOM nhất (0,504 / 0,638).

Dùng: python scripts/score_covid_nyt.py --data data/covid_nyt/2020_covid_nyt.jsonl [--calibrate] [--pred tên=file.jsonl]
      --embedder <hf>... --threshold <x>... --device cuda --out scores.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from kgu.eval.atom_quality import match_row, summarize

CALIB = {"gpt41-direct": ["quintuples_gpt41", "quintuples_gpt41_run2", "quintuples_gpt41_run3"],
         "gpt41-atom": ["quintuples_gpt41_from_factoids", "quintuples_gpt41_from_factoids_run2",
                        "quintuples_gpt41_from_factoids_run3"]}
PAPER_F1 = {"gpt41-direct": 0.504, "gpt41-atom": 0.638}
PREFIX = {"intfloat/e5-large-v2": "query: "}       # e5 cần tiền tố; so khớp đối xứng → "query: " cả hai phía


def load_dataset(path) -> list[dict]:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf8").splitlines() if line.strip()]
    for r in rows:
        r["gold"] = [tuple(q) for q in r["gold"]]
    return rows


def load_predictions(path) -> dict[int, list[tuple]]:
    out = {}
    for line in Path(path).read_text(encoding="utf8").splitlines():
        if line.strip():
            d = json.loads(line)
            out[d["row"]] = [tuple(q) for q in d["quintuples"]]
    return out


def text(q) -> str:
    return f"{q[0]} {q[1]} {q[2]}"


def score(rows: list[dict], preds: dict[int, list[tuple]], emb: dict[str, np.ndarray],
          thresholds: list[float]) -> dict[float, dict]:
    per = {t: [] for t in thresholds}
    for r in rows:
        pred = [q for q in preds.get(r["row"], []) if len(q) >= 3]
        if pred and r["gold"]:
            sim = np.stack([emb[text(q)] for q in pred]) @ np.stack([emb[text(q)] for q in r["gold"]]).T
        else:
            sim = np.zeros((len(pred), len(r["gold"])))
        for t in thresholds:
            per[t].append(match_row(pred, r["gold"], sim, t))
    out = {}
    for t, p in per.items():
        out[t] = summarize(p)
        out[t].update({f"{k}_skip": v for k, v in summarize([x for x in p if x["n_pred"]]).items()})
    return out


def pick(results: dict) -> dict:
    """results[embedder][ngưỡng][hệ thống] → chỉ số. Chọn cặp giữ thứ tự ATOM > trích thẳng và có tổng lệch F1_skip so
    với paper nhỏ nhất."""
    best: dict = {}
    for emb_name, by_t in results.items():
        for t, systems in by_t.items():
            f1 = {k: systems[k]["F1_skip"] for k in PAPER_F1}
            if f1["gpt41-atom"] <= f1["gpt41-direct"]:
                continue
            gap = sum(abs(f1[k] - PAPER_F1[k]) for k in PAPER_F1)
            if not best or gap < best["gap"]:
                best = {"embedder": emb_name, "threshold": t, "gap": round(gap, 4), **f1}
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--pred", action="append", default=[], help="tên=đường_dẫn.jsonl")
    ap.add_argument("--embedder", nargs="+", default=["sentence-transformers/all-mpnet-base-v2"])
    ap.add_argument("--threshold", nargs="+", type=float, default=[0.7])
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    from sentence_transformers import SentenceTransformer

    rows = load_dataset(args.data)
    systems: dict[str, list[dict[int, list[tuple]]]] = {}
    if args.calibrate:
        for name, cols in CALIB.items():
            systems[name] = [{r["row"]: [tuple(q) for q in r["calib"].get(c) or []] for r in rows} for c in cols]
    for spec in args.pred:
        name, path = spec.split("=", 1)
        systems[name] = [load_predictions(path)]
    texts = sorted({text(q) for r in rows for q in r["gold"]}
                   | {text(q) for runs in systems.values() for p in runs for qs in p.values() for q in qs if len(q) >= 3})
    print(f"{len(rows)} nhóm ngày, {len(systems)} hệ thống, {len(texts)} câu cần nhúng", flush=True)
    results: dict = {}
    for emb_name in args.embedder:
        model = SentenceTransformer(emb_name, device=args.device)
        vecs = model.encode([PREFIX.get(emb_name, "") + s for s in texts], batch_size=128, normalize_embeddings=True)
        emb = dict(zip(texts, vecs))
        results[emb_name] = {}
        for name, runs in systems.items():
            s = [score(rows, p, emb, args.threshold) for p in runs]
            for t in args.threshold:
                m = {k: round(float(np.mean([x[t][k] for x in s])), 4) for k in s[0][t]}
                results[emb_name].setdefault(str(t), {})[name] = m
                print(f"{emb_name} @{t} {name}: F1 {m['F1']} F1_skip {m['F1_skip']} F1_t {m['F1_t']} "
                      f"MATCH {m['MATCH']} HALL {m['HALL']}", flush=True)
        del model
    out = {"data": str(args.data), "n_texts": len(texts), "results": results}
    if args.calibrate:
        out["pick"] = pick(results)
        print("pick", out["pick"], flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf8")


if __name__ == "__main__":
    main()
