from __future__ import annotations

import os
from dataclasses import dataclass, field

from kgu.data import Example
from kgu.eval.coverage import coverage, coverage_by_call
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
    loc: Localization                       # vùng ứng viên GỘP của mọi lần gọi (không trùng)
    judged_ops: list[Op]
    pred_after: set[Triple]
    n_llm_calls: int = 0
    n_dropped: int = 0
    graph: BiTemporalGraph = field(default_factory=BiTemporalGraph, repr=False)
    max_calls: int = 1                      # K: số lần gọi tối đa của khoanh vùng đệ quy
    rounds: list[dict] = field(default_factory=list)        # mỗi lần gọi đã chạy: k, số ứng viên, số op, số lệnh gọi LLM
    judged_rounds: list[int] = field(default_factory=list)  # song song judged_ops: op được kết luận ở lần gọi nào
    locs: list[Localization] = field(default_factory=list, repr=False)   # vùng ứng viên của từng lần gọi (đã lọc)

    def to_record(self, ex: Example) -> dict:
        rounds = self.judged_rounds or [1] * len(self.judged_ops)
        locs = self.locs or [self.loc]
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
            "judged_ops": [{**o.to_dict(), "round": k} for o, k in zip(self.judged_ops, rounds)],
            "max_calls": self.max_calls,
            "rounds": list(self.rounds),
            "implied": [{"kind": o.kind.value, "h": o.h, "r": o.r, "t": o.t, "cov": k}
                        for o, k in coverage_by_call(ex, self.explicit_ops, locs)],
        }


def _calls(*objs) -> int:
    return sum(getattr(o, "n_calls", 0) for o in objs if o is not None)


def _dropped(extractor) -> int:
    return getattr(extractor, "n_dropped", 0)


def run_example(ex: Example, extractor: Extractor, judge: Judge | None, relations: list[str],
                graph: BiTemporalGraph | None = None, max_calls: int = 1) -> PipelineResult:
    """`graph`: đồ thị ĐÃ ở trạng thái ex.before tại AT_BEFORE (KG nền dùng chung, caller rollback sau); None = dựng mới.

    `max_calls` = K, số lần gọi tối đa của thuật toán khoanh vùng đệ quy (Chương 3, Thuật toán alg:localize):
    lần gọi k khoanh vùng quanh O_{k-1} (O_0 = thay đổi trực tiếp đã áp), bước xét duyệt cho O_k, O_k được ghi vào KG
    rồi làm đầu vào của lần gọi k+1; dừng khi O_k rỗng hoặc k > K. K = 1 là hành vi một lần gọi như trước.
    """
    calls_before = _calls(extractor, judge)                      # extractor/judge dùng chung qua nhiều ví dụ
    dropped_before = _dropped(extractor)                          # LLMExtractor.n_dropped cũng cộng dồn qua nhiều ví dụ
    if graph is None:
        graph = BiTemporalGraph.from_triples(ex.before, valid_from=AT_BEFORE)
    explicit = extractor.extract(ex, graph, AT_BEFORE)
    explicit = apply_ops(graph, explicit, at=AT_AFTER)           # [2] áp ops TRƯỚC
    symmetric = os.environ.get("KGU_SYMMETRIC_ADD") == "1"

    merged = Localization()
    locs: list[Localization] = []
    rounds: list[dict] = []
    judged: list[Op] = []
    judged_rounds: list[int] = []
    seen_cut: set[Triple] = set()                                # triple đã là ứng viên cắt ở lần gọi trước
    seen_pairs: set[frozenset[str]] = set()                      # cặp {subject, neighbor} đã là ứng viên sinh
    written: set[Triple] = {op.triple for op in explicit}        # cạnh đã ghi (ADD/INVALIDATE) khi xử lý bản tin này
    current = explicit                                           # O_{k-1}: tập thay đổi đang xét
    for k in range(1, max(1, max_calls) + 1):
        if k > 1 and not current:                                # không còn thay đổi mới → dừng
            break
        loc = localize(graph, current, at=AT_AFTER, symmetric_add=symmetric)    # [3] khoanh vùng quanh O_{k-1}
        if k > 1:                                                # bỏ ứng viên đã xét và cạnh đã ghi khi xử lý bản tin này
            loc = Localization(
                cut=[c for c in loc.cut if c.triple not in seen_cut and c.triple not in written],
                add=[c for c in loc.add if frozenset((c.subject, c.neighbor)) not in seen_pairs],
            )
            if not loc.cut and not loc.add:                      # không còn ứng viên mới → dừng, không gọi bước xét duyệt
                break
        seen_cut |= loc.cut_triples
        seen_pairs |= loc.add_pairs
        locs.append(loc)
        merged.cut += loc.cut
        merged.add += loc.add
        new_ops: list[Op] = []
        calls_k = _calls(judge)
        if judge is not None:                                    # [4] bộ phán trên ứng viên của lần gọi k
            ctx = JudgeContext(ex.text, current, loc, graph, AT_AFTER, relations, direct_ops=explicit, call=k)
            new_ops = apply_ops(graph, judge.judge(ctx), at=AT_AFTER)           # [5] ghi O_k
        rounds.append({"k": k, "n_cut_cand": len(loc.cut), "n_add_cand": len(loc.add),
                       "n_ops": len(new_ops), "n_llm_calls": _calls(judge) - calls_k})
        judged += new_ops
        judged_rounds += [k] * len(new_ops)
        written |= {op.triple for op in new_ops}
        current = new_ops                                        # O_k là đầu vào của lần gọi k+1 (judge None → rỗng)

    n_calls = _calls(extractor, judge) - calls_before            # chỉ đếm lệnh gọi của ví dụ này
    n_dropped = _dropped(extractor) - dropped_before              # chỉ đếm op bị loại của ví dụ này
    pred_after = set(ex.before)                                  # = graph.active(AT_AFTER), tính từ ops hiệu lực (O(ops))
    for op in explicit + judged:
        if op.kind is OpKind.ADD:
            pred_after.add(op.triple)
        else:
            pred_after.discard(op.triple)
    return PipelineResult(ex.idx, ex.event, explicit, merged, judged, pred_after, n_calls, n_dropped, graph,
                          max_calls=max(1, max_calls), rounds=rounds, judged_rounds=judged_rounds, locs=locs)
