from __future__ import annotations

from dataclasses import asdict, dataclass

from kgu.types import Triple


@dataclass
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    add_hit: int = 0
    n_add: int = 0
    del_hit: int = 0
    n_del: int = 0

    def __add__(self, other: "Counts") -> "Counts":
        return Counts(**{k: getattr(self, k) + getattr(other, k) for k in asdict(self)})

    def to_dict(self) -> dict:
        return asdict(self)


def score(before: set[Triple], gold_after: set[Triple], pred_after: set[Triple]) -> Counts:
    """Theo GUpdate: F1 trên before ∪ gold_after; acc riêng cho cạnh thêm / cạnh xóa.

    Triple dự đoán nằm ngoài universe vẫn bị tính FP (không cho thêm bừa).
    """
    universe = before | gold_after | pred_after
    c = Counts()
    for tr in universe:
        label, pred = tr in gold_after, tr in pred_after
        if label and pred:
            c.tp += 1
        elif label and not pred:
            c.fn += 1
        elif not label and pred:
            c.fp += 1
        else:
            c.tn += 1
    added, deleted = gold_after - before, before - gold_after
    c.n_add, c.n_del = len(added), len(deleted)
    c.add_hit = sum(1 for tr in added if tr in pred_after)
    c.del_hit = sum(1 for tr in deleted if tr not in pred_after)
    return c


def _div(a: int, b: int) -> float:
    return a / b if b else 0.0


def summarize(c: Counts) -> dict:
    p = _div(c.tp, c.tp + c.fp)
    r = _div(c.tp, c.tp + c.fn)
    return {
        "precision": p,
        "recall": r,
        "f1": _div(2 * p * r, p + r) if (p + r) else 0.0,
        "add_acc": _div(c.add_hit, c.n_add),
        "del_acc": _div(c.del_hit, c.n_del),
    }
