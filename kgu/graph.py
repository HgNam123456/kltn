from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from kgu.types import Triple


@dataclass
class Edge:
    h: str
    r: str
    t: str
    valid_from: int
    valid_to: int | None = None

    @property
    def triple(self) -> Triple:
        return (self.h, self.r, self.t)

    def active_at(self, at: int) -> bool:
        return self.valid_from <= at and (self.valid_to is None or self.valid_to > at)


class BiTemporalGraph:
    """Đồ thị có hướng, đa quan hệ; mỗi triple có thể có nhiều khoảng hiệu lực.

    Không bao giờ xóa: INVALIDATE chỉ đóng `valid_to` của khoảng đang mở.
    """

    def __init__(self) -> None:
        self._edges: dict[Triple, list[Edge]] = defaultdict(list)
        self._by_entity: dict[str, set[Triple]] = defaultdict(set)
        self._journal: list[tuple] = []          # nhật ký thay đổi để rollback (đồ thị nền dùng chung nhiều bài)

    @classmethod
    def from_triples(cls, triples: Iterable[Triple], valid_from: int = 0) -> "BiTemporalGraph":
        g = cls()
        for tr in triples:
            g.add(tr, valid_from)
        return g

    # ---- mutation -------------------------------------------------------
    def _open(self, triple: Triple) -> Edge | None:
        for e in self._edges.get(triple, ()):
            if e.valid_to is None:
                return e
        return None

    def add(self, triple: Triple, valid_from: int) -> bool:
        if self._open(triple) is not None:
            return False
        h, r, t = triple
        self._edges[triple].append(Edge(h, r, t, valid_from))
        self._by_entity[h].add(triple)
        self._by_entity[t].add(triple)
        self._journal.append(("add", triple))
        return True

    def invalidate(self, triple: Triple, valid_to: int) -> bool:
        e = self._open(triple)
        if e is None:
            return False
        e.valid_to = valid_to
        self._journal.append(("inv", e))
        return True

    def remove(self, triple: Triple) -> bool:
        """Xóa cứng mọi khoảng của triple — CHỈ dùng khi dựng KG-trước từ KG nền dùng chung (không phải thao tác cập nhật)."""
        es = self._edges.pop(triple, None)
        if not es:
            return False
        for e in (triple[0], triple[2]):
            self._by_entity[e].discard(triple)
        self._journal.append(("rm", triple, es))
        return True

    def checkpoint(self) -> int:
        return len(self._journal)

    def rollback(self, checkpoint: int) -> None:
        """Hoàn tác mọi add/invalidate/remove sau checkpoint (thứ tự ngược) → đồ thị về đúng trạng thái cũ."""
        while len(self._journal) > checkpoint:
            entry = self._journal.pop()
            if entry[0] == "add":
                triple = entry[1]
                es = self._edges[triple]
                es.pop()
                if not es:
                    del self._edges[triple]
                    for e in (triple[0], triple[2]):
                        self._by_entity[e].discard(triple)
            elif entry[0] == "inv":
                entry[1].valid_to = None
            else:
                triple, es = entry[1], entry[2]
                self._edges[triple] = es
                for e in (triple[0], triple[2]):
                    self._by_entity[e].add(triple)

    # ---- queries --------------------------------------------------------
    def is_active(self, triple: Triple, at: int) -> bool:
        return any(e.active_at(at) for e in self._edges.get(triple, ()))

    def active(self, at: int) -> set[Triple]:
        return {tr for tr, es in self._edges.items() if any(e.active_at(at) for e in es)}

    def entities(self, at: int) -> set[str]:
        out: set[str] = set()
        for h, _, t in self.active(at):
            out.add(h)
            out.add(t)
        return out

    def edges_of(self, entity: str, at: int) -> set[Triple]:
        return {tr for tr in self._by_entity.get(entity, ()) if self.is_active(tr, at)}

    def neighbors(self, entity: str, at: int) -> set[str]:
        out: set[str] = set()
        for h, _, t in self.edges_of(entity, at):
            out.add(t if h == entity else h)
        out.discard(entity)
        return out

    def history(self, triple: Triple) -> list[Edge]:
        return list(self._edges.get(triple, ()))

    def to_records(self) -> list[dict]:
        return [
            {"h": e.h, "r": e.r, "t": e.t, "valid_from": e.valid_from, "valid_to": e.valid_to}
            for tr in sorted(self._edges)
            for e in self._edges[tr]
        ]
