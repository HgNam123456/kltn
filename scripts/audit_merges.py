"""Kiểm tay quyết định gộp tên (spec docs/superpowers/specs/2026-10-15-condense-canonicalize-design.md §3): rút mẫu
từ <out>.merges.jsonl ra CSV (một nửa "yes", một nửa "no", seed cố định); người kiểm điền cột `correct` = 1 (quyết
định đúng) / 0 (sai); chạy lại với --labels để tính tỉ lệ đúng của "yes" và "no".

Dùng: python scripts/audit_merges.py --merges out/v2.jsonl.merges.jsonl --n 100 --out audit.csv
      python scripts/audit_merges.py --labels audit.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

FIELDS = ["id", "kind", "anchor", "name", "cosine", "same", "reason", "canonical", "anchor_context", "name_context",
          "correct"]


def sample(records: list[dict], n: int = 100, seed: int = 0) -> list[dict]:
    recs = [r for r in records if "same" in r]       # bỏ dòng lỗi lệnh gọi
    yes, no = [r for r in recs if r["same"]], [r for r in recs if not r["same"]]
    k_yes = min(len(yes), n // 2)
    k_no = min(len(no), n - k_yes)
    k_yes = min(len(yes), n - k_no)                  # một bên thiếu → lấy thêm từ bên kia
    rng = random.Random(seed)
    return sorted(rng.sample(yes, k_yes) + rng.sample(no, k_no), key=lambda r: r["id"])


def summarize(rows: list[dict]) -> dict:
    out = {}
    for label in ("yes", "no"):
        rs = [r for r in rows if (r["same"] == "True") == (label == "yes") and r["correct"].strip() in ("0", "1")]
        ok = sum(r["correct"].strip() == "1" for r in rs)
        out[label] = {"n": len(rs), "correct": ok, "accuracy": round(ok / len(rs), 3) if rs else None}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--merges", type=Path)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--labels", type=Path, help="CSV đã điền cột correct → in tỉ lệ đúng")
    args = ap.parse_args()
    if args.labels:
        with args.labels.open(encoding="utf8", newline="") as f:
            print(json.dumps(summarize(list(csv.DictReader(f))), ensure_ascii=False))
        return
    lines = args.merges.read_text(encoding="utf8").splitlines()
    records = [{"id": i, **json.loads(line)} for i, line in enumerate(lines) if line.strip()]
    picked = sample(records, args.n, args.seed)
    with args.out.open("w", encoding="utf8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in picked:
            w.writerow({**r, "anchor_context": " || ".join(r["anchor_context"]),
                        "name_context": " || ".join(r["name_context"]), "correct": ""})
    print(f"{len(picked)} quyết định ({sum(r['same'] for r in picked)} yes) → {args.out}")


if __name__ == "__main__":
    main()
