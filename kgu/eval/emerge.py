from __future__ import annotations

import random
from typing import Iterable, Mapping, Sequence

from kgu.data.emerge import OPS, EmergeExample
from kgu.types import Triple

Predictions = Mapping[str, Mapping[str, set[Triple]]]    # hash_id -> mã op -> triple QID

# Nhãn `operation` trong prediction của baseline KG-aware -> mã op.
OPERATION_TO_OP = {
    "EXISTS": "x-triples", "ADD": "e-triples", "MINT_ADD": "ee-triples", "MINT+ADD": "ee-triples",
    "INFER": "ee-kg-triples", "DEPRECATE": "d-triples",
}


def _is_qid_triple(tr) -> bool:
    return (len(tr) == 3 and all(isinstance(x, str) and x for x in tr)
            and tr[0][0] == "Q" and tr[1][0] == "P" and tr[2][0] == "Q")


def shipped_predictions(raw_instance: dict, model: str) -> dict[str, set[Triple]]:
    """Prediction kèm sẵn trong dataset của một model có khai `operation` (họ kg-aware/*)."""
    out: dict[str, set[Triple]] = {op: set() for op in OPS}
    for t in raw_instance["predictions"].get(model, {}).get("predicted_triples") or []:
        op = OPERATION_TO_OP.get(str(t.get("operation")).upper())
        qids = t.get("triple_qids") or ()
        if op and _is_qid_triple(qids):
            out[op].add(tuple(qids))
    return out


def instance_scores(examples: Iterable[EmergeExample], preds: Predictions) -> dict[str, dict[str, tuple[float, float]]]:
    """(recall, precision) của từng instance có gold đã xác nhận cho op đó: mã op -> hash_id -> điểm."""
    rows: dict[str, dict[str, tuple[float, float]]] = {op: {} for op in OPS}
    for ex in examples:
        for op in OPS:
            gold = ex.gold[op]
            if not gold:
                continue
            pred = {tr for tr in preds.get(ex.hash_id, {}).get(op, ()) if _is_qid_triple(tr)}
            tp = len(gold & pred)
            rows[op][ex.hash_id] = (tp / len(gold), tp / len(pred) if pred else 0.0)
    return rows


def executable_scores(examples: Iterable[EmergeExample], preds: Predictions) -> dict[str, dict[str, float]]:
    """Executable-R của EMERGE (cie_exact_match): khớp chính xác bộ ba QID, tính theo từng instance
    rồi lấy trung bình trên các instance có gold đã xác nhận cho op đó; instance không dự đoán tính 0."""
    return {
        op: {
            "n": len(r),
            "recall": sum(x for x, _ in r.values()) / len(r) if r else 0.0,
            "precision": sum(y for _, y in r.values()) / len(r) if r else 0.0,
        }
        for op, r in instance_scores(examples, preds).items()
    }


def bootstrap_ci(values: Sequence[float], n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """Khoảng tin cậy percentile cho trung bình, lấy mẫu lại theo instance. Dùng cho điểm của một hệ,
    hoặc cho hiệu điểm từng instance của hai hệ (bootstrap ghép cặp)."""
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(rng.choices(values, k=n)) / n for _ in range(n_boot))
    return means[int(n_boot * alpha / 2)], means[int(n_boot * (1 - alpha / 2)) - 1]
