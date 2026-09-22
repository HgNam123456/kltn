"""Dựng KG-trước thu gọn cho EMERGE: stream KG snapshot từ HuggingFace, chỉ giữ cạnh 1-hop
quanh entity được nhắc trong passage (không lưu file gốc 400–600 MB/snapshot).

Cạnh đi ra (subject được nhắc) giữ hết. Cạnh đi vào (object được nhắc) giữ tối đa --inbound-cap
mỗi entity để entity hub (United States, human...) không kéo theo hàng triệu dòng; tổng số cạnh vào
thật vẫn được đếm trong file .stats.json.

Dùng: python scripts/build_emerge_subgraphs.py [snapshot ...]   (mặc định: mọi snapshot đã tải)
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import time
import urllib.request
from pathlib import Path

RESOLVE = "https://huggingface.co/datasets/klimzaporojets/emerge-benchmark/resolve/main/kg_snapshots/"


def mentioned_qids(snapshot_dir: Path) -> set[str]:
    out: set[str] = set()
    for path in sorted(snapshot_dir.glob("delta_*.jsonl")):
        with path.open(encoding="utf8") as f:
            for line in f:
                out.update(m["qid"] for m in json.loads(line)["mentions"] if m.get("qid"))
    return out


def build(snapshot: str, data_dir: Path, inbound_cap: int) -> dict:
    seeds = mentioned_qids(data_dir / "evaluation_set" / f"snapshot_{snapshot}")
    out_dir = data_dir / "kg_subsets"
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"snapshot_{snapshot}.tsv"
    inbound: collections.Counter[str] = collections.Counter()
    n_rows = n_kept = 0
    start = time.time()
    url = f"{RESOLVE}kg_snapshot_{snapshot}.tsv.gz"
    with urllib.request.urlopen(url) as resp, gzip.open(resp, "rt", encoding="utf8") as src, \
            dest.open("w", encoding="utf8", newline="\n") as dst:
        for line in src:
            n_rows += 1
            if n_rows % 5_000_000 == 0:
                print(f"  {snapshot}: {n_rows:,} rows, kept {n_kept:,}, {time.time() - start:.0f}s", flush=True)
            h, _, rest = line.partition("\t")
            t = rest.split("\t", 2)[1] if rest else ""
            if h in seeds:
                if t in seeds:
                    inbound[t] += 1
            elif t in seeds:
                inbound[t] += 1
                if inbound[t] > inbound_cap:
                    continue
            else:
                continue
            dst.write(line)
            n_kept += 1
    stats = {
        "snapshot": snapshot, "seeds": len(seeds), "rows_scanned": n_rows, "rows_kept": n_kept,
        "inbound_cap": inbound_cap, "seconds": round(time.time() - start),
        "capped_entities": {q: n for q, n in inbound.most_common() if n > inbound_cap},
    }
    dest.with_suffix(".stats.json").write_text(json.dumps(stats, indent=1), encoding="utf8")
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("snapshots", nargs="*")
    ap.add_argument("--data-dir", type=Path, default=Path("data/raw/emerge"))
    ap.add_argument("--inbound-cap", type=int, default=500)
    args = ap.parse_args()
    snapshots = args.snapshots or sorted(
        p.name.removeprefix("snapshot_") for p in (args.data_dir / "evaluation_set").glob("snapshot_*")
    )
    for s in snapshots:
        print(f"building {s} ...", flush=True)
        stats = build(s, args.data_dir, args.inbound_cap)
        print(f"  done: kept {stats['rows_kept']:,}/{stats['rows_scanned']:,} rows in {stats['seconds']}s")


if __name__ == "__main__":
    main()
