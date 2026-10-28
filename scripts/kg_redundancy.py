"""Độ gọn của KG (không cần gold), mọi hệ trên mọi benchmark:
- tên thực thể gần trùng: cặp tên khác nhau (sau chuẩn hóa chữ hoa/khoảng trắng/gạch dưới) có cosine > --entity-tau
  (0,8, như chỉ số gộp tên của ATOM đã dùng ở COVID-NYT), tính trên KG tích lũy (--scope global: COVID-NYT) hoặc trong
  từng văn bản (--scope doc: MINE-1, câu/đoạn của benchmark EDC); báo số cặp và số cặp trên 100 tên;
- triple trùng: trong từng văn bản, cặp triple khác nhau có cosine "h r t" > --triple-tau (0,9) và số triple trùng hệt
  sau chuẩn hóa; báo trên 100 triple.
Embedder all-mpnet-base-v2 (như bộ chấm COVID-NYT). Đọc output bằng scripts/kg_outputs.py.

Dùng: python scripts/kg_redundancy.py --system mine=covid:v2.jsonl --covid-data covid.jsonl --scope global \
          --device cuda --out redundancy.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
from pathlib import Path

import numpy as np


def _outputs():
    spec = importlib.util.spec_from_file_location("kg_outputs", Path(__file__).with_name("kg_outputs.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).replace("_", " ")).strip().lower()


def near_pairs(names: list[str], emb: dict, tau: float, block: int = 2048) -> int:
    """Số cặp (i < j) có cosine > tau; vector đã chuẩn hóa."""
    if len(names) < 2:
        return 0
    m = np.stack([emb[n] for n in names]).astype(np.float32)
    count = 0
    for s in range(0, len(names), block):
        sim = m[s:s + block] @ m.T
        rows = np.arange(s, min(s + block, len(names)))[:, None]
        count += int(((sim > tau) & (np.arange(len(names))[None, :] > rows)).sum())
    return count


def measure(docs: list[dict], emb: dict, scope: str, entity_tau: float, triple_tau: float) -> dict:
    ent_pairs = ent_total = 0
    groups = [docs] if scope == "global" else [[d] for d in docs]
    for g in groups:
        names = sorted({norm(x) for d in g for t in d["triples"] for x in (t[0], t[2])})
        ent_total += len(names)
        ent_pairs += near_pairs(names, emb, entity_tau)
    tri_total = tri_exact = tri_pairs = 0
    for d in docs:
        keys = [" ".join(norm(x) for x in t) for t in d["triples"]]
        tri_total += len(keys)
        uniq = sorted(set(keys))
        tri_exact += len(keys) - len(uniq)
        tri_pairs += near_pairs(uniq, emb, triple_tau)
    per100 = lambda a, b: round(100 * a / b, 2) if b else None          # noqa: E731
    return {"docs": len(docs), "entities": ent_total, "entity_near_pairs": ent_pairs,
            "entity_near_pairs_per_100": per100(ent_pairs, ent_total), "triples": tri_total,
            "triple_exact_dups": tri_exact, "triple_near_pairs": tri_pairs,
            "triple_near_pairs_per_100": per100(tri_pairs, tri_total), "scope": scope}


def main() -> None:
    from sentence_transformers import SentenceTransformer

    ap = argparse.ArgumentParser()
    ap.add_argument("--system", action="append", default=[])
    ap.add_argument("--covid-data")
    ap.add_argument("--mine-data")
    ap.add_argument("--lines-input")
    ap.add_argument("--scope", choices=["global", "doc"], required=True)
    ap.add_argument("--entity-tau", type=float, default=0.8)
    ap.add_argument("--triple-tau", type=float, default=0.9)
    ap.add_argument("--embedder", default="sentence-transformers/all-mpnet-base-v2")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    ko = _outputs()
    src = ko.Sources(args.covid_data, args.mine_data, args.lines_input)
    systems = dict(ko.load(spec, src) for spec in args.system)
    texts = sorted({norm(x) for docs in systems.values() for d in docs for t in d["triples"] for x in (t[0], t[2])}
                   | {" ".join(norm(x) for x in t) for docs in systems.values() for d in docs for t in d["triples"]})
    model = SentenceTransformer(args.embedder, device=args.device)
    emb = dict(zip(texts, model.encode(texts, batch_size=256, normalize_embeddings=True)))
    out = json.loads(args.out.read_text(encoding="utf8")) if args.out.exists() else {}
    for name, docs in systems.items():
        out[name] = measure(docs, emb, args.scope, args.entity_tau, args.triple_tau)
        print(name, out[name], flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf8")


if __name__ == "__main__":
    main()
