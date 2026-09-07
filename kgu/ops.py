from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from kgu.graph import BiTemporalGraph
from kgu.types import Op, OpKind, Triple


def diff(before: set[Triple], after: set[Triple], source: str = "gold") -> list[Op]:
    """Ops vàng: INVALIDATE cho cạnh mất, ADD cho cạnh mới. Thứ tự ổn định."""
    inv = [Op(OpKind.INVALIDATE, *tr, source=source) for tr in sorted(before - after)]
    add = [Op(OpKind.ADD, *tr, source=source) for tr in sorted(after - before)]
    return inv + add


def apply_ops(graph: BiTemporalGraph, ops: Iterable[Op], at: int) -> list[Op]:
    """Executor bi-temporal: INVALIDATE = đóng valid_to=at, ADD = mở valid_from=at."""
    effective: list[Op] = []
    for op in ops:
        if op.kind is OpKind.INVALIDATE:
            ok = graph.invalidate(op.triple, valid_to=at)
        else:
            ok = graph.add(op.triple, valid_from=at)
        if ok:
            effective.append(op)
    return effective


def write_jsonl(path: Path, ops: Iterable[Op]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf8") as f:
        for op in ops:
            f.write(json.dumps(op.to_dict(), ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[Op]:
    with path.open(encoding="utf8") as f:
        return [Op.from_dict(json.loads(line)) for line in f if line.strip()]
