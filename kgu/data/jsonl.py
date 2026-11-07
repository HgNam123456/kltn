"""Bộ dữ liệu cập nhật KG ở dạng JSONL chung (một dòng một bài toán), dùng cho các bộ ngoài NBA
(ICEWS14-Event, YAGO-Event sinh bởi scripts/prep_cascade.py; TKGQA bởi scripts/prep_tkgqa.py; ELKEN qua scripts/run_elken.py).

Hai định dạng:

1. Lát cắt (mỗi bài tự chứa KG): {"idx", "event", "season", "text", "mentioned": [tên], "before": [[h, r, t]],
   "after": [[h, r, t]], "extra": {...}}. Quan hệ = tập quan hệ xuất hiện trong before ∪ after của cả tệp.

2. KG cho trước (giao thức "train là KG nền"): dòng đầu là header {"base": [[h, r, t]], "cumulative": bool,
   "relations": [...]} — toàn bộ thực thể và triple đã biết trước; mỗi bài sau đó chỉ ghi phần chênh lệch:
   "before_add"/"before_del" (so với trạng thái trước bài: base nếu cumulative=false, KG-sau của bài trước nếu
   cumulative=true) và "after_add"/"after_del" (so với KG-trước của chính bài). Tập quan hệ ĐÓNG = quan hệ trong
   base (mỗi nhãn là một quan hệ riêng, không gộp, không sinh nhãn mới); nếu header có "relations" thì dùng đúng
   danh sách đó. Loader dựng KG-trước của từng bài từ base khi truy cập (không giữ N bản sao đồ thị).
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from kgu.data import Example
from kgu.types import Triple


def _triples(rows: list) -> set[Triple]:
    return {(str(h), str(r), str(t)) for h, r, t in rows}


class GivenKGDataset(Sequence):
    """Sequence[Example] dựng từ header base + chênh lệch từng bài. extras[i] = trường "extra" của bài i."""

    def __init__(self, base: set[Triple], rows: list[dict], cumulative: bool) -> None:
        self.base = base
        self.extras: list[dict] = []
        self._meta: list[dict] = []
        self._before_delta: list[tuple[set[Triple], set[Triple]]] = []   # (thêm, bớt) so với base
        self._after_delta: list[tuple[set[Triple], set[Triple]]] = []    # (thêm, bớt) so với KG-trước
        cur_add, cur_del = set(), set()                                  # trạng thái hiện tại so với base
        for d in rows:
            b_add, b_del = _triples(d.get("before_add", [])), _triples(d.get("before_del", []))
            if cumulative:
                cur_add, cur_del = (cur_add | b_add) - b_del, (cur_del | b_del) - b_add
            else:
                cur_add, cur_del = b_add, b_del
            a_add, a_del = _triples(d.get("after_add", [])), _triples(d.get("after_del", []))
            self._before_delta.append((set(cur_add), set(cur_del)))
            self._after_delta.append((a_add, a_del))
            if cumulative:                                               # KG-sau của bài này là trạng thái kế tiếp
                cur_add, cur_del = (cur_add | a_add) - a_del, (cur_del | a_del) - a_add
            self._meta.append(d)
            self.extras.append(d.get("extra", {}) or {})

    def __len__(self) -> int:
        return len(self._meta)

    def apply_before(self, graph, i: int, at: int = 0) -> None:
        """Đưa đồ thị nền (dựng từ self.base tại `at`) về trạng thái KG-trước của bài i; caller rollback sau khi chạy."""
        b_add, b_del = self._before_delta[i]
        for tr in b_del:
            graph.remove(tr)
        for tr in b_add:
            graph.add(tr, at)

    def __getitem__(self, i):
        if isinstance(i, slice):
            return [self[j] for j in range(*i.indices(len(self)))]
        d = self._meta[i]
        b_add, b_del = self._before_delta[i]
        a_add, a_del = self._after_delta[i]
        before = (self.base | b_add) - b_del
        after = (before | a_add) - a_del
        return Example(
            idx=int(d.get("idx", i)), event=str(d.get("event", "")), season=str(d.get("season", "")),
            text=str(d.get("text", "")), mentioned=[str(m) for m in d.get("mentioned", [])],
            before=before, after=after,
        )


def load_jsonl(path: Path, limit: int | None = None) -> tuple[Sequence[Example], list[str], list[dict]]:
    """→ (examples, relations, extras). extras[i] là trường "extra" của dòng i (có thể rỗng)."""
    with path.open(encoding="utf8") as f:
        lines = [json.loads(line) for line in f if line.strip()]
    if lines and "base" in lines[0]:                                     # định dạng KG cho trước
        head, rows = lines[0], lines[1:]
        if limit is not None:
            rows = rows[:limit]
        base = _triples(head["base"])
        rels = head.get("relations") or sorted({r for _, r, _ in base})
        ds = GivenKGDataset(base, rows, bool(head.get("cumulative", False)))
        return ds, [str(r) for r in rels], ds.extras

    examples, extras, rels = [], [], set()
    for d in lines:
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
