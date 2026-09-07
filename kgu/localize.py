from __future__ import annotations

from dataclasses import dataclass, field

from kgu.graph import BiTemporalGraph
from kgu.types import Op, OpKind, Triple


@dataclass(frozen=True)
class CutCandidate:
    triple: Triple
    hub: str                     # entity nối với >= 2 participant
    anchors: frozenset[str]      # các participant mà hub nối tới


@dataclass(frozen=True)
class AddCandidate:
    subject: str                 # a: đầu cạnh mới (a, r, b)
    neighbor: str                # n: hàng xóm của b chưa nối với a
    via: Triple                  # cạnh b–n làm bằng chứng bối cảnh


@dataclass
class Localization:
    cut: list[CutCandidate] = field(default_factory=list)
    add: list[AddCandidate] = field(default_factory=list)

    @property
    def cut_triples(self) -> set[Triple]:
        return {c.triple for c in self.cut}

    @property
    def add_pairs(self) -> set[frozenset[str]]:
        return {frozenset((c.subject, c.neighbor)) for c in self.add}


def _edges_between(graph: BiTemporalGraph, x: str, y: str, at: int) -> set[Triple]:
    return {tr for tr in graph.edges_of(x, at) if y in (tr[0], tr[2])}


def localize(graph: BiTemporalGraph, explicit_ops: list[Op], at: int) -> Localization:
    """Khoanh vùng cấu trúc 2 vế. Gọi SAU khi explicit_ops đã được áp tại `at`."""
    participants = {e for op in explicit_ops for e in (op.h, op.t)}
    loc = Localization()

    # ---- vế cắt --------------------------------------------------------
    seen_cut: set[Triple] = set()
    for op in explicit_ops:
        if op.kind is not OpKind.INVALIDATE:
            continue
        a, b = op.h, op.t
        # `a` và `b` đã không còn kề nhau tại `at`; hub phải kề a hoặc b
        hubs = (graph.neighbors(a, at) | graph.neighbors(b, at)) - participants
        for hub in sorted(hubs):
            anchors = graph.neighbors(hub, at) & participants
            if len(anchors) < 2:
                continue
            for p in sorted(anchors):
                for tr in sorted(_edges_between(graph, hub, p, at)):
                    if tr not in seen_cut:
                        seen_cut.add(tr)
                        loc.cut.append(CutCandidate(tr, hub, frozenset(anchors)))

    # ---- vế sinh -------------------------------------------------------
    seen_add: set[tuple[str, str]] = set()
    for op in explicit_ops:
        if op.kind is not OpKind.ADD:
            continue
        a, b = op.h, op.t
        fresh = graph.neighbors(b, at) - graph.neighbors(a, at) - {a, b}
        for n in sorted(fresh):
            if (a, n) in seen_add:
                continue
            via = sorted(_edges_between(graph, b, n, at))[0]
            seen_add.add((a, n))
            loc.add.append(AddCandidate(a, n, via))

    return loc
