"""Độ ổn định giữa các lượt chạy trên 2020-COVID-NYT như paper ATOM (arXiv 2510.22590v2, Bảng 1): mỗi nhóm ngày,
Jaccard giữa tập 5-tuple của lượt 1 và lượt r; trung bình ± std theo nhóm (paper, GPT-4.1: ATOM 0,552 / 0,534, trích
thẳng 0,212 / 0,218 cho lượt 2 / 3). Paper không nói cách so 5-tuple → so chính xác sau khi chuẩn hóa chữ hoa/khoảng
trắng/gạch dưới, thời gian đọc thành tập ngày (kgu.eval.atom_quality.dates). Thêm hai biến thể: triple bỏ thời gian, và
Jaccard mềm = m / (|A| + |B| − m) với m = số cặp ghép 1-1 theo cosine "h r t" > ngưỡng (cùng cách ghép của bộ chấm).
Nhóm mà cả hai lượt đều rỗng bị bỏ; một lượt rỗng → 0.
--released: thêm ba lượt GPT-4.1 do tác giả phát hành trong dataset (cột calib), để đối chiếu với số của paper.

Dùng: python scripts/covid_stability.py --data covid.jsonl --run mine=r1:a.jsonl,r2:b.jsonl [--run ...] [--released]
      --embedder sentence-transformers/all-mpnet-base-v2 --threshold 0.7 --device cuda --out stability.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
from pathlib import Path

import numpy as np

from kgu.eval.atom_quality import dates, match_row

_spec = importlib.util.spec_from_file_location("score_covid_nyt", Path(__file__).with_name("score_covid_nyt.py"))
_score = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_score)


def norm(s) -> str:
    return re.sub(r"\s+", " ", str(s).replace("_", " ")).strip().lower()


def key(q, with_time: bool = True) -> tuple:
    k = (norm(q[0]), norm(q[1]), norm(q[2]))
    if with_time:
        k += (dates(q[3]) if len(q) > 3 else frozenset(), dates(q[4]) if len(q) > 4 else frozenset())
    return k


def jaccard(a: set, b: set) -> float | None:
    if not a and not b:
        return None
    return len(a & b) / len(a | b)


def soft_jaccard(a: list, b: list, emb: dict, threshold: float) -> float | None:
    if not a and not b:
        return None
    if not a or not b:
        return 0.0
    sim = np.stack([emb[_score.text(q)] for q in a]) @ np.stack([emb[_score.text(q)] for q in b]).T
    m = match_row(a, b, sim, threshold)["match"]
    return m / (len(a) + len(b) - m)


def stats(values: list) -> dict:
    v = [x for x in values if x is not None]
    return {"mean": round(float(np.mean(v)), 4) if v else None, "std": round(float(np.std(v)), 4) if v else None,
            "groups": len(v)}


def compare(rows: list[dict], ref: dict, other: dict, emb: dict | None, threshold: float) -> dict:
    exact, triple, soft = [], [], []
    for r in rows:
        a = [q for q in ref.get(r["row"], []) if len(q) >= 3]
        b = [q for q in other.get(r["row"], []) if len(q) >= 3]
        exact.append(jaccard({key(q) for q in a}, {key(q) for q in b}))
        triple.append(jaccard({key(q, False) for q in a}, {key(q, False) for q in b}))
        if emb is not None:
            soft.append(soft_jaccard(a, b, emb, threshold))
    out = {"jaccard": stats(exact), "jaccard_triple": stats(triple)}
    if emb is not None:
        out["jaccard_soft"] = stats(soft)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--run", action="append", default=[], help="hệ=nhãn:file.jsonl,nhãn:file.jsonl (đầu tiên = gốc)")
    ap.add_argument("--released", action="store_true")
    ap.add_argument("--embedder", default="sentence-transformers/all-mpnet-base-v2")
    ap.add_argument("--threshold", type=float, default=0.7)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rows = _score.load_dataset(args.data)
    systems: dict[str, list[tuple[str, dict]]] = {}
    for spec in args.run:
        name, runs = spec.split("=", 1)
        systems[name] = [(lab, _score.load_predictions(path)) for lab, path in (x.split(":", 1) for x in runs.split(","))]
    if args.released:
        for name, cols in _score.CALIB.items():
            systems[f"released:{name}"] = [(f"r{i}", {r["row"]: [tuple(q) for q in r["calib"].get(c) or []] for r in rows})
                                           for i, c in enumerate(cols, 1)]
    emb = None
    if args.embedder:
        from sentence_transformers import SentenceTransformer

        texts = sorted({_score.text(q) for runs in systems.values() for _, p in runs for qs in p.values()
                        for q in qs if len(q) >= 3})
        model = SentenceTransformer(args.embedder, device=args.device)
        emb = dict(zip(texts, model.encode(texts, batch_size=128, normalize_embeddings=True)))
    out = {"threshold": args.threshold, "embedder": args.embedder, "systems": {}}
    for name, runs in systems.items():
        (ref_lab, ref), rest = runs[0], runs[1:]
        out["systems"][name] = {f"{ref_lab}-{lab}": compare(rows, ref, p, emb, args.threshold) for lab, p in rest}
        print(name, json.dumps(out["systems"][name]), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf8")


if __name__ == "__main__":
    main()
