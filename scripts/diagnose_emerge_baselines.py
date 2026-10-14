"""Vì sao baseline OIE chạy lại bằng Gemma 4 thấp hơn bản GPT-5.1: so từng triple gold (đã xác nhận) với dự đoán, chỉ
bằng so khớp chữ, trên evaluation_set dev350 đã gộp prediction (kaggle/dataset_dev350/).

Cột in ra (tỉ lệ trên số gold): pair = có triple dự đoán nối đúng 2 thực thể của gold (khớp mềm: chứa nhau hoặc
SequenceMatcher >= 0,8, hai chiều); pair+rel = trong đó tên quan hệ trùng nhãn Wikidata của gold; exact = trùng nguyên
chuỗi; head = có triple nhắc tới head. Thêm: tỉ lệ tên quan hệ dự đoán là nhãn property Wikidata (danh sách lấy từ
schema EDC của EMERGE), và trần chung khi gộp mọi lượt chạy.

Dùng: python scripts/diagnose_emerge_baselines.py [--data kaggle/dataset_dev350/evaluation_set]
"""
from __future__ import annotations

import argparse
import json
import re
import urllib.request
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

from kgu.data.emerge import _parse

PROPS_URL = ("https://huggingface.co/datasets/klimzaporojets/emerge-benchmark/resolve/main/"
             "indices/relik_edc_relation_indexes/2025-01-01/documents.jsonl")
MODELS = {
    "EDC ICL GPT": "edc-plus-open-ai/gpt-5.1/non-canonicalized",
    "EDC ICL Gemma": "edc-plus-llamacpp/gemma-4-e4b",
    "EDC ZS GPT": "edc-plus-zshot-open_ai/GPT-5_1",
    "EDC ZS Gemma": "edc-plus-zshot-llamacpp/gemma-4-e4b",
    "KGGen GPT": "kg-gen/azure/gpt5.1",
    "KGGen Gemma": "kg-gen/llamacpp/gemma-4-e4b",
    "KARMA Gemma": "karma/llamacpp/gemma-4-e4b",
}
OPS = ["x-triples", "e-triples", "d-triples"]


def norm(s) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", str(s).replace("_", " ").lower()).split())


def ent_match(a: str, b: str) -> bool:
    a, b = norm(a), norm(b)
    if not a or not b:
        return False
    return a == b or a in b or b in a or SequenceMatcher(None, a, b).ratio() >= 0.8


def links(gold: tuple[str, str, str], pred) -> bool:
    gh, _, gt = gold
    return (ent_match(gh, pred[0]) and ent_match(gt, pred[2])) or (ent_match(gh, pred[2]) and ent_match(gt, pred[0]))


def preds_for(p: dict) -> dict[str, list]:
    """Cùng cách bộ chấm EMERGE phân op: oie_add → Exists/Add/Mint+Add, oie_deprecate → Deprecate; kg-gen → mọi op thêm."""
    out = defaultdict(list)
    if "predicted_triples_oie" in p:
        for t in p["predicted_triples_oie"].get("oie_add", []):
            for op in ("x-triples", "e-triples", "ee-triples"):
                out[op].append(t[:3])
        for t in p["predicted_triples_oie"].get("oie_deprecate", []):
            out["d-triples"].append(t[:3])
    else:
        for t in p.get("predicted_triples", []):
            for op in ("x-triples", "e-triples", "ee-triples"):
                out[op].append(t["extracted_relation"][:3])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("kaggle/dataset_dev350/evaluation_set"))
    ap.add_argument("--props", type=Path, default=Path("data/raw/emerge/wikidata_properties_2025.jsonl"))
    args = ap.parse_args()
    if not args.props.exists():
        args.props.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(PROPS_URL, args.props)
    wd_labels = {norm(json.loads(line)["text"]) for line in args.props.open(encoding="utf8")}

    rows = []
    for path in sorted(args.data.glob("snapshot_*/delta_*.jsonl")):
        snapshot = path.parent.name.removeprefix("snapshot_")
        for line in path.open(encoding="utf8"):
            r = json.loads(line)
            rows.append((_parse(r, snapshot, path.stem.removeprefix("delta_")), r["predictions"]))
    models = {m: k for m, k in MODELS.items() if any(k in p for _, p in rows)}

    stats = {m: defaultdict(Counter) for m in models}
    rel_vocab = {m: Counter() for m in models}
    for ex, preds in rows:
        for m, key in models.items():
            pm = preds_for(preds[key]) if key in preds else {}
            for t in pm.get("x-triples", []):
                rel_vocab[m]["wd" if norm(t[1]) in wd_labels else "free"] += 1
            for op in OPS:
                pr = pm.get(op, [])
                stats[m][op]["n_pred"] += len(pr)
                for g in ex.gold[op]:
                    gold = tuple(ex.labels.get(x, x) for x in g)
                    pair = [p for p in pr if links(gold, p)]
                    s = stats[m][op]
                    s["gold"] += 1
                    s["exact"] += any(norm(" ".join(map(str, p))) == norm(" ".join(gold)) for p in pr)
                    s["pair"] += bool(pair)
                    s["pair_rel"] += any(norm(p[1]) == norm(gold[1]) for p in pair)
                    s["head"] += any(ent_match(gold[0], p[0]) or ent_match(gold[0], p[2]) for p in pr)

    for op in OPS:
        print(f"== {op}\n{'model':14s} {'gold':>5s} {'pred':>6s} {'exact':>6s} {'pair':>6s} {'pair+rel':>8s} {'head':>6s}")
        for m in models:
            s = stats[m][op]
            g = max(s["gold"], 1)
            print(f"{m:14s} {s['gold']:5d} {s['n_pred']:6d} {s['exact']/g:6.1%} {s['pair']/g:6.1%} "
                  f"{s['pair_rel']/g:8.1%} {s['head']/g:6.1%}")
    print("\nTên quan hệ dự đoán (Exists/Add) là nhãn property Wikidata:")
    for m, c in rel_vocab.items():
        print(f"  {m:14s} {c['wd'] / max(sum(c.values()), 1):6.1%} của {sum(c.values())}")

    for op in ("x-triples", "e-triples"):
        hit = tot = 0
        for ex, p in rows:
            allp = [t for key in models.values() if key in p for t in preds_for(p[key]).get(op, [])]
            for g in ex.gold[op]:
                tot += 1
                hit += any(links(tuple(ex.labels.get(x, x) for x in g), q) for q in allp)
        print(f"Trần chung {op}: gộp {len(models)} lượt nối được {hit}/{tot} = {hit / max(tot, 1):.1%}")


if __name__ == "__main__":
    main()
