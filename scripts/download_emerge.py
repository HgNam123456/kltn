"""Tải test set EMERGE (arXiv 2507.03617) vào data/raw/emerge/evaluation_set/.

Chỉ tải evaluation_set (35 file JSONL, ~175 MB). KG snapshot (3,7 GB nén) và index không tải ở đây.
"""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

REPO = "klimzaporojets/emerge-benchmark"
TREE = f"https://huggingface.co/api/datasets/{REPO}/tree/main/evaluation_set?recursive=true"
RESOLVE = f"https://huggingface.co/datasets/{REPO}/resolve/main/"


def main(out_dir: Path) -> None:
    with urllib.request.urlopen(TREE) as resp:
        files = [f for f in json.load(resp) if f["type"] == "file"]
    for f in files:
        dest = out_dir / f["path"]
        if dest.exists() and dest.stat().st_size == f["size"]:
            print(f"skip {f['path']} (exists)")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {f['path']} ...")
        urllib.request.urlretrieve(RESOLVE + f["path"], dest)
        print(f"  -> {dest} ({dest.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/raw/emerge"))
