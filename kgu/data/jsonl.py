"""Bộ dữ liệu cập nhật KG ở dạng JSONL chung (một dòng một bài toán), dùng cho các bộ ngoài NBA
(ICEWS14-Event, YAGO-Event sinh bởi scripts/prep_cascade.py; ELKEN qua scripts/run_elken.py).

Mỗi dòng: {"idx", "event", "season", "text", "mentioned": [tên], "before": [[h, r, t]], "after": [[h, r, t]],
"extra": {...tùy bộ}}. Quan hệ = tập quan hệ xuất hiện trong before ∪ after của cả tệp.
"""
from __future__ import annotations

import json
from pathlib import Path

from kgu.data import Example
from kgu.types import Triple


def _triples(rows: list) -> set[Triple]:
    return {(str(h), str(r), str(t)) for h, r, t in rows}


def load_jsonl(path: Path, limit: int | None = None) -> tuple[list[Example], list[str], list[dict]]:
    """→ (examples, relations, extras). extras[i] là trường "extra" của dòng i (có thể rỗng)."""
    examples, extras, rels = [], [], set()
    with path.open(encoding="utf8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            before, after = _triples(d["before"]), _triples(d["after"])
            rels |= {r for _, r, _ in before | after}
            examples.append(Example(
                idx=int(d.get("idx", len(examples))), event=str(d.get("event", "")), season=str(d.get("season", "")),
                text=str(d.get("text", "")), mentioned=[str(m) for m in d.get("mentioned", [])],
                before=before, after=after,
            ))
            extras.append(d.get("extra", {}) or {})
            if limit is not None and len(examples) >= limit:
                break
    return examples, sorted(rels), extras
