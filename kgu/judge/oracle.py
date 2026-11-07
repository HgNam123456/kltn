from __future__ import annotations

from kgu.judge import JudgeContext
from kgu.types import Op, OpKind, Triple


class OracleJudge:
    """Bộ phán nhìn KG-sau (gold). Chỉ dùng để đo cận trên của khoanh vùng."""

    def __init__(self, gold_after: set[Triple]) -> None:
        self.gold_after = gold_after
        self._by_pair: dict[frozenset[str], list[Triple]] = {}       # chỉ mục cặp đầu–cuối (KG nền có thể ~1M cạnh)
        for tr in gold_after:
            self._by_pair.setdefault(frozenset((tr[0], tr[2])), []).append(tr)

    def judge(self, ctx: JudgeContext) -> list[Op]:
        ops: list[Op] = []
        for c in ctx.loc.cut:
            if c.triple not in self.gold_after:
                ops.append(Op(OpKind.INVALIDATE, *c.triple, source="oracle"))
        seen: set[Triple] = set()
        for c in ctx.loc.add:
            for tr in sorted(self._by_pair.get(frozenset((c.subject, c.neighbor)), ())):
                if tr not in seen and not ctx.graph.is_active(tr, ctx.at):
                    seen.add(tr)
                    ops.append(Op(OpKind.ADD, *tr, source="oracle"))
        return ops
