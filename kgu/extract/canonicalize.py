"""Gộp tên entity/relation của KG tích lũy (spec docs/superpowers/specs/2026-10-15-condense-canonicalize-design.md
§2.3): nhúng tên → k-means (k = số tên / cluster_size) → trong mỗi cụm nối các cặp cosine ≥ τ có cùng tập chữ số thành
nhóm, tên mốc gần tâm nhóm nhất → một lệnh gọi LLM mỗi cụm có tên mới: yes/no + lý do cho từng ứng viên, chọn tên chung
→ đổi tên, bỏ fact trùng, tên cũ thành bí danh. k-means viết bằng numpy (không thêm scikit-learn)."""
from __future__ import annotations

import re

import numpy as np


def kmeans(X: np.ndarray, k: int, seed: int = 0, iters: int = 50) -> np.ndarray:
    """k-means++ với seed cố định (chạy lại ra cùng cụm). Khoảng cách tính qua tích vô hướng để không tạo mảng n×k×d."""
    X = np.asarray(X, dtype=float)
    n = len(X)
    k = max(1, min(k, n))
    rng = np.random.default_rng(seed)
    sq = (X ** 2).sum(1)
    first = X[rng.integers(n)]
    centers = [first]
    d = sq - 2 * X @ first + first @ first
    while len(centers) < k:
        d = np.maximum(d, 0)
        if d.sum() <= 0:                               # mọi điểm trùng tâm đã chọn
            break
        c = X[rng.choice(n, p=d / d.sum())]
        centers.append(c)
        d = np.minimum(d, sq - 2 * X @ c + c @ c)
    C = np.array(centers)
    labels = None
    for _ in range(iters):
        new = (sq[:, None] - 2 * X @ C.T + (C ** 2).sum(1)[None]).argmin(1)
        if labels is not None and (new == labels).all():
            break
        labels = new
        for j in range(len(C)):
            if (labels == j).any():
                C[j] = X[labels == j].mean(0)
    return labels


def _numbers(name: str) -> tuple[str, ...]:
    return tuple(re.findall(r"\d+", name.replace(",", "")))


def candidate_groups(names: list[str], X: np.ndarray, labels: np.ndarray, counts: dict[str, int], tau: float,
                     blocked=frozenset(), max_size: int = 8) -> dict[int, list[dict]]:
    """Trong mỗi cụm: nối cặp cosine ≥ tau, cùng tập chữ số ("1,694 Cases" ≠ "1,700 Cases"), không bị chặn (đã "no");
    mỗi thành phần liên thông ≥ 2 tên là một nhóm; mốc = gần tâm nhóm nhất (hòa → nhiều fact hơn); nhóm quá max_size →
    giữ mốc + các tên cosine cao nhất với mốc."""
    out: dict[int, list[dict]] = {}
    for c in sorted(set(np.asarray(labels).tolist())):
        idx = [i for i in range(len(names)) if labels[i] == c]
        parent = {i: i for i in idx}

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        S = X[idx] @ X[idx].T
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                i, j = idx[a], idx[b]
                if (S[a, b] >= tau and _numbers(names[i]) == _numbers(names[j])
                        and frozenset((names[i], names[j])) not in blocked):
                    parent[find(i)] = find(j)
        comps: dict[int, list[int]] = {}
        for i in idx:
            comps.setdefault(find(i), []).append(i)
        groups = []
        for comp in comps.values():
            if len(comp) < 2:
                continue
            centre = X[comp].mean(0)
            anchor = max(comp, key=lambda i: (round(float(X[i] @ centre), 6), counts[names[i]]))
            others = sorted((i for i in comp if i != anchor), key=lambda i: -float(X[i] @ X[anchor]))[: max_size - 1]
            groups.append({"anchor": names[anchor], "members": [names[anchor]] + [names[i] for i in others],
                           "cos": {names[i]: round(float(X[i] @ X[anchor]), 4) for i in others}})
        if groups:
            out[int(c)] = groups
    return out
