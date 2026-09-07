from __future__ import annotations

from dataclasses import dataclass, field

from kgu.data.nba import Example
from kgu.eval.coverage import coverage
from kgu.eval.metrics import score
from kgu.extract import Extractor
from kgu.graph import BiTemporalGraph
from kgu.judge import Judge, JudgeContext
from kgu.localize import Localization, localize
from kgu.ops import apply_ops
from kgu.types import Op, Triple

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
            "coverage": list(coverage(ex, self.explicit_ops, self.loc)),
            "counts": score(ex.before, ex.after, self.pred_after).to_dict(),
            "explicit_ops": [o.to_dict() for o in self.explicit_ops],
            "judged_ops": [o.to_dict() for o in self.judged_ops],
        }


def _calls(*objs) -> int:
    return sum(getattr(o, "n_calls", 0) for o in objs if o is not None)


def run_example(ex: Example, extractor: Extractor, judge: Judge | None, relations: list[str]) -> PipelineResult:
    calls_before = _calls(extractor, judge)                      # extractor/judge dùng chung qua nhiều ví dụ
    graph = BiTemporalGraph.from_triples(ex.before, valid_from=AT_BEFORE)
    explicit = extractor.extract(ex, graph)
    explicit = apply_ops(graph, explicit, at=AT_AFTER)           # [2] áp ops TRƯỚC
    loc = localize(graph, explicit, at=AT_AFTER)                 # [3] khoanh vùng
    judged: list[Op] = []
    if judge is not None:                                        # [4] bộ phán
        ctx = JudgeContext(ex.text, explicit, loc, graph, AT_AFTER, relations)
        judged = apply_ops(graph, judge.judge(ctx), at=AT_AFTER)
    n_calls = _calls(extractor, judge) - calls_before            # chỉ đếm lệnh gọi của ví dụ này
    return PipelineResult(ex.idx, ex.event, explicit, loc, judged, graph.active(AT_AFTER), n_calls, graph)
