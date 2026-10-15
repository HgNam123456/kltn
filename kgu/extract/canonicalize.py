"""Gộp tên entity/relation của KG tích lũy (spec docs/superpowers/specs/2026-10-15-condense-canonicalize-design.md
§2.3): nhúng tên → k-means (k = số tên / cluster_size) → trong mỗi cụm nối các cặp cosine ≥ τ có cùng tập chữ số thành
nhóm, tên mốc gần tâm nhóm nhất → một lệnh gọi LLM mỗi cụm có tên mới: yes/no + lý do cho từng ứng viên, chọn tên chung
→ đổi tên, bỏ fact trùng, tên cũ thành bí danh. k-means viết bằng numpy (không thêm scikit-learn)."""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, create_model

ENTITY_SYSTEM = """You clean the entity names of a knowledge graph built from news. Each GROUP has an ANCHOR name and \
CANDIDATE names that look similar, each followed by facts where it appears. For every candidate decide if it names \
exactly the same real-world entity as the anchor: write a short reason first, then "same": true or false. Related, \
broader, narrower, a part of, or a different instance (another place, person, number or date) is NOT the same. Then \
give "canonical": the clearest complete name among the anchor and the candidates you marked same."""

RELATION_SYSTEM = """You clean the relation names of a knowledge graph built from news. Each GROUP has an ANCHOR \
relation and CANDIDATE relations that look similar, each followed by facts that use it. For every candidate decide if \
it means exactly the same as the anchor in the same direction (head -> tail): write a short reason first, then \
"same": true or false. Broader, narrower, opposite or reversed relations are NOT the same. Then give "canonical": the \
clearest name among the anchor and the candidates you marked same."""


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


def _collect(facts: list[list], kind: str) -> tuple[list[str], Counter, dict[str, list[str]]]:
    """→ (tên đã sắp xếp, số fact chứa tên, tối đa 3 fact mới nhất chứa tên)."""
    counts: Counter = Counter()
    context: dict[str, list[str]] = {}
    for f in reversed(facts):
        for name in ((f[0], f[2]) if kind == "entity" else (f[1],)):
            counts[name] += 1
            ctx = context.setdefault(name, [])
            if len(ctx) < 3:
                ctx.append(f"{f[0]} | {f[1]} | {f[2]}")
    return sorted(counts), counts, context


def _merge_model(names: list[str]) -> type[BaseModel]:
    lit = Literal[tuple(names)]
    decision = create_model("Decision", name=(lit, ...), reason=(str, ...), same=(bool, ...))
    group = create_model("Group", group=(int, ...), decisions=(list[decision], ...), canonical=(lit, ...))
    return create_model("Merges", groups=(list[group], ...))


def apply_mapping(kg, kind: str, mapping: dict[str, str]) -> None:
    """Đổi tên trong mọi fact, bỏ fact trùng hệt và vòng tự thân; entity: tên cũ (và bí danh của nó) thành bí danh của
    tên chung."""
    if not mapping:
        return
    new, seen = [], set()
    for f in kg.facts:
        if kind == "entity":
            f = [mapping.get(f[0], f[0]), f[1], mapping.get(f[2], f[2]), f[3], f[4]]
        else:
            f = [f[0], mapping.get(f[1], f[1]), f[2], f[3], f[4]]
        key = (f[0], f[1], f[2], tuple(f[3]), tuple(f[4]))
        if f[0].lower() != f[2].lower() and key not in seen:
            seen.add(key)
            new.append(f)
    kg.facts[:] = new
    if kind == "entity":
        for old, can in mapping.items():
            kg.aliases.setdefault(can, set()).update({old, *kg.aliases.pop(old, set())})


class Canonicalizer:
    def __init__(self, embed, llm, tau: float = 0.75, cluster_size: int = 10, log_path=None, seed: int = 0):
        self.embed, self.llm, self.tau, self.cluster_size, self.seed = embed, llm, tau, cluster_size, seed
        self.log_path = Path(log_path) if log_path else None
        self.rejected: set[tuple[str, frozenset]] = set()          # (kind, cặp mốc–ứng viên) đã bị "no"
        self.seen: dict[str, set[str]] = {"entity": set(), "relation": set()}
        self.round = 0

    def run_round(self, kg) -> list[dict]:
        self.round += 1
        return [self.run(kg, "entity"), self.run(kg, "relation")]

    def run(self, kg, kind: str) -> dict:
        names, counts, context = _collect(kg.facts, kind)
        stats = {"kind": kind, "names": len(names), "clusters": 0, "groups": 0, "calls": 0, "errors": 0, "merged": 0}
        if len(names) >= 2:
            X = np.asarray(self.embed([n.replace("_", " ") for n in names] if kind == "relation" else names),
                           dtype=float)
            X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
            labels = kmeans(X, math.ceil(len(names) / self.cluster_size), self.seed)
            blocked = {pair for k, pair in self.rejected if k == kind}
            by_cluster = candidate_groups(names, X, labels, counts, self.tau, blocked)
            stats.update(clusters=len(set(labels.tolist())), groups=sum(map(len, by_cluster.values())))
            mapping: dict[str, str] = {}
            for c, groups in by_cluster.items():
                if all(m in self.seen[kind] for g in groups for m in g["members"]):
                    continue                                       # cụm không có tên mới → đã xét ở lượt trước
                stats["calls"] += 1
                try:
                    out = self._ask(kind, groups, context)
                except Exception as e:                             # lệnh gọi hỏng → bỏ cụm này, KG giữ nguyên
                    stats["errors"] += 1
                    self._log({"round": self.round, "kind": kind, "cluster": c, "error": repr(e)[:300]})
                    continue
                mapping.update(self._decide(kind, c, groups, context, out))
            apply_mapping(kg, kind, mapping)
            stats["merged"] = len(mapping)
            self.seen[kind] |= set(mapping.values())
        self.seen[kind] |= set(names)
        return stats

    def _ask(self, kind: str, groups: list[dict], context: dict[str, list[str]]):
        names = sorted({m for g in groups for m in g["members"]})
        lines: list[str] = []
        for i, g in enumerate(groups):
            lines.append(f"GROUP {i}")
            for j, m in enumerate(g["members"]):
                lines.append(f"{'ANCHOR' if j == 0 else 'CANDIDATE'}: {m}")
                lines += [f"    {t}" for t in context[m]]
        system = ENTITY_SYSTEM if kind == "entity" else RELATION_SYSTEM
        return self.llm.complete_json(system, "\n".join(lines), _merge_model(names))

    def _decide(self, kind: str, cluster: int, groups: list[dict], context: dict[str, list[str]], out) -> dict[str, str]:
        answers: dict[int, object] = {}
        for g in out.groups:
            answers.setdefault(g.group, g)                         # trùng số nhóm → lấy lần đầu
        mapping: dict[str, str] = {}
        for i, g in enumerate(groups):
            a = answers.get(i)
            if a is None:
                continue
            anchor, cands = g["members"][0], g["members"][1:]
            decided: dict[str, tuple[bool, str]] = {}
            for d in a.decisions:                                  # bỏ quyết định cho mốc / tên ngoài nhóm
                if d.name in cands and d.name not in decided:
                    decided[d.name] = (d.same, d.reason)
            same = [n for n in cands if decided.get(n, (False, ""))[0]]
            canonical = a.canonical if a.canonical in (anchor, *same) else anchor
            for name, (ok, reason) in decided.items():
                if not ok:
                    self.rejected.add((kind, frozenset((anchor, name))))
                self._log({"round": self.round, "kind": kind, "cluster": cluster, "group": i, "name": name,
                           "anchor": anchor, "cosine": g["cos"][name], "same": ok, "reason": reason,
                           "canonical": canonical if ok else None,
                           "anchor_context": context[anchor], "name_context": context[name]})
            if same:
                mapping.update({n: canonical for n in (anchor, *same) if n != canonical})
        return mapping

    def _log(self, rec: dict) -> None:
        if self.log_path:
            with self.log_path.open("a", encoding="utf8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def sentence_embedder(model_name: str, device: str):
    """Embedder thật (sentence-transformers, chỉ dùng trên Kaggle): list[str] → ndarray đã chuẩn hóa."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)
    return lambda texts: model.encode(list(texts), batch_size=128, normalize_embeddings=True)
