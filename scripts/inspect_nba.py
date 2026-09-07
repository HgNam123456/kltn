"""Spike: in 3 mẫu đã giải mã + thống kê để ghi vào docs/data-notes.md."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path


def load_id_map(path: Path) -> dict[int, str]:
    out: dict[int, str] = {}
    for line in path.read_text(encoding="utf8").splitlines():
        if "\t" not in line:
            continue  # dòng đầu là số đếm
        name, idx = line.rsplit("\t", 1)
        out[int(idx)] = name
    return out


def main(data_dir: Path, split: str = "test") -> None:
    ent = load_id_map(data_dir / "entity2id.txt")
    rel = load_id_map(data_dir / "relation2id.txt")
    tok = load_id_map(data_dir / "token2id.txt")
    data = json.loads((data_dir / f"NBAtransactions_{split}.json").read_text(encoding="utf8"))
    print(f"{split}: {len(data)} examples; keys = {sorted(data[0].keys())}")

    def name(x):  # text / mentioned có thể là id hoặc chuỗi
        return ent.get(x, str(x)) if isinstance(x, int) else str(x)

    for ex in data[:3]:
        text = " ".join(tok.get(t, str(t)) if isinstance(t, int) else str(t) for t in ex["text"])
        before = {(ent[h], rel[r], ent[t]) for h, r, t in ex["subgraph_before"]}
        after = {(ent[h], rel[r], ent[t]) for h, r, t in ex["subgraph_after"]}
        print("=" * 80)
        print("event:", ex["event"], "| season:", ex["season"])
        print("text :", text)
        print("mentioned:", [name(m) for m in ex["text_mentioned_entities"]])
        print("|before| =", len(before), "|after| =", len(after))
        print("ADDED  :", sorted(after - before)[:10])
        print("DELETED:", sorted(before - after)[:10])

    n_add = n_del = 0
    rel_dir = Counter()  # (relation, head-is-team?) để xác định hướng quan hệ
    events = Counter()
    for ex in data:
        before = {tuple(t) for t in ex["subgraph_before"]}
        after = {tuple(t) for t in ex["subgraph_after"]}
        n_add += len(after - before)
        n_del += len(before - after)
        events[ex["event"]] += 1
        for h, r, t in list(before)[:50]:
            rel_dir[(rel[r], ent[h].endswith(("s", "ers")) and "_" in ent[h])] += 1
    print("=" * 80)
    print(f"avg added = {n_add/len(data):.2f}, avg deleted = {n_del/len(data):.2f}")
    print("events:", events.most_common())
    print("relation/head-looks-like-team:", rel_dir.most_common())


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/raw/nba"), sys.argv[2] if len(sys.argv) > 2 else "test")
