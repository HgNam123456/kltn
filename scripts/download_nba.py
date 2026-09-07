"""Tải dataset NBAtransactions (GUpdater, EMNLP 2019) vào data/raw/nba/."""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

BASE = "https://raw.githubusercontent.com/esddse/GUpdater/master/data/"
FILES = [
    "NBAtransactions_train.json", "NBAtransactions_valid.json", "NBAtransactions_test.json",
    "entity2id.txt", "relation2id.txt", "token2id.txt",
]


def main(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        dest = out_dir / name
        if dest.exists():
            print(f"skip {name} (exists)")
            continue
        print(f"downloading {name} ...")
        urllib.request.urlretrieve(BASE + name, dest)
        print(f"  -> {dest} ({dest.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/raw/nba"))
