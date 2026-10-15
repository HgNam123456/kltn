"""Chỉ số gộp tên (entity/relation resolution) của ATOM, cài lại theo evaluation/atom/merge/evaluate_atom_merge.py
(AuvaLab/itext2kg) với embedder truyền vào thay cho text-embedding-3-large:
- similar = số cặp tên khác nhau trong KG có cosine > ngưỡng (bản gốc 0,8);
- precision = 1 − similar / (số lần xuất hiện trong gold − số tên của KG); khi mẫu ≤ 0: entity → 0, relation → 1 nếu
  bằng nhau, ngược lại 0 (như bản gốc);
- recall = 1 − similar / (số lần xuất hiện − số tên gold khác nhau (không phân biệt hoa thường) − số cặp tên gold có
  cosine > ngưỡng); mẫu ≤ 0 → 1. Cả hai kẹp [0, 1]."""
from __future__ import annotations

import numpy as np


def similar_pairs(names: list[str], embed, threshold: float = 0.8) -> int:
    if len(names) < 2:
        return 0
    X = np.asarray(embed(list(names)), dtype=float)
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    iu = np.triu_indices(len(names), 1)
    return int(((X @ X.T)[iu] > threshold).sum())


def kg_names(facts: list[list], kind: str) -> list[str]:
    if kind == "entity":
        return sorted({f[0] for f in facts} | {f[2] for f in facts})
    return sorted({f[1] for f in facts})


def gold_mentions(quintuples: list, kind: str) -> list[str]:
    if kind == "entity":
        return [q[0] for q in quintuples] + [q[2] for q in quintuples]
    return [q[1] for q in quintuples]


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


def resolution(names: list[str], mentions: list[str], embed, threshold: float = 0.8, kind: str = "entity") -> dict:
    names = sorted(set(names))
    sim = similar_pairs(names, embed, threshold)
    expected = len(mentions) - len(names)
    if expected > 0:
        p = _clamp(1 - sim / expected)
    else:
        p = (1.0 if expected == 0 else 0.0) if kind == "relation" else 0.0
    uniq = sorted({m.lower() for m in mentions if m})
    merged = len(mentions) - len(uniq) - similar_pairs(uniq, embed, threshold) if len(uniq) >= 2 else 0
    r = 1.0 if merged <= 0 else _clamp(1 - sim / merged)
    f1 = 2 * p * r / (p + r) if p > 0 and r > 0 else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4), "kg_names": len(names),
            "similar_pairs": sim, "gold_mentions": len(mentions), "gold_unique": len(uniq), "gold_merged": merged}
