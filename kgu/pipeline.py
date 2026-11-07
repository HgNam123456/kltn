from __future__ import annotations

import os
from dataclasses import dataclass, field

from kgu.data import Example
from kgu.eval.coverage import coverage
from kgu.eval.metrics import score
from kgu.extract import Extractor
from kgu.graph import BiTemporalGraph
from kgu.judge import Judge, JudgeContext
from kgu.localize import Localization, localize
from kgu.ops import apply_ops
from kgu.types import Op, OpKind, Triple

AT_BEFORE = 0
AT_AFTER = 1


@dataclass
class PipelineResult:
    idx: int
    event: str
    explicit_ops: list[Op]
    loc: Localization
    judged_ops: list[Op]
    pred_after: set[Triple]
    n_llm_calls: int = 0
    n_dropped: int = 0
    graph: BiTemporalGraph = field(default_factory=BiTemporalGraph, repr=False)

    def to_record(self, ex: Example) -> dict:
        return {
            "idx": self.idx,
            "event": self.event,
            "n_explicit": len(self.explicit_ops),
            "n_cut_cand": len(self.loc.cut),
            "n_add_cand": len(self.loc.add),
            "n_judged": len(self.judged_ops),
            "n_llm_calls": self.n_llm_calls,
            "n_dropped": self.n_dropped,
            "coverage": list(coverage(ex, self.explicit_ops, self.loc)),
            "counts": score(ex.before, ex.after, self.pred_after).to_dict(),
            "explicit_ops": [o.to_dict() for o in self.explicit_ops],
            "judged_ops": [o.to_dict() for o in self.judged_ops],
        }


def _calls(*objs) -> int:
    return sum(getattr(o, "n_calls", 0) for o in objs if o is not None)


def _dropped(extractor) -> int:
    return getattr(extractor, "n_dropped", 0)


def run_example(ex: Example, extractor: Extractor, judge: Judge | None, relations: list[str],
                graph: BiTemporalGraph | None = None) -> PipelineResult:
    """`graph`: đồ thị ĐÃ ở trạng thái ex.before tại AT_BEFORE (KG nền dùng chung, caller rollback sau); None = dựng mới."""
    calls_before = _calls(extractor, judge)                      # extractor/judge dùng chung qua nhiều ví dụ
    dropped_before = _dropped(extractor)                          # LLMExtractor.n_dropped cũng cộng dồn qua nhiều ví dụ
    if graph is None:
        graph = BiTemporalGraph.from_triples(ex.before, valid_from=AT_BEFORE)
    explicit = extractor.extract(ex, graph, AT_BEFORE)
    explicit = apply_ops(graph, explicit, at=AT_AFTER)           # [2] áp ops TRƯỚC
    loc = localize(graph, explicit, at=AT_AFTER,                 # [3] khoanh vùng
                   symmetric_add=os.environ.get("KGU_SYMMETRIC_ADD") == "1")
    judged: list[Op] = []
    if judge is not None:                                        # [4] bộ phán
        ctx = JudgeContext(ex.text, explicit, loc, graph, AT_AFTER, relations)
        judged = apply_ops(graph, judge.judge(ctx), at=AT_AFTER)
    n_calls = _calls(extractor, judge) - calls_before            # chỉ đếm lệnh gọi của ví dụ này
    n_dropped = _dropped(extractor) - dropped_before              # chỉ đếm op bị loại của ví dụ này
    pred_after = set(ex.before)                                  # = graph.active(AT_AFTER), tính từ ops hiệu lực (O(ops))
    for op in explicit + judged:
        if op.kind is OpKind.ADD:
            pred_after.add(op.triple)
        else:
            pred_after.discard(op.triple)
    return PipelineResult(ex.idx, ex.event, explicit, loc, judged, pred_after, n_calls, n_dropped, graph)
