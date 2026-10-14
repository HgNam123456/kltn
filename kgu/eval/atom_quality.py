"""Metric 5-tuple của ATOM (Lairgi et al., EACL 2026), cài lại từ evaluation/atom/quintuples_quality/
calculate_quintuples_quality.py của AuvaLab/itext2kg, tách khỏi embedder:
- mỗi bài (nhóm ngày): ghép dự đoán ↔ gold 1-1 tham lam theo cosine của "h r t" giảm dần, chỉ cặp > ngưỡng;
- MATCH = khớp / gold, HALL = dự đoán thừa / dự đoán, OM = gold sót / gold;
- trong cặp khớp: MATCH_t nếu t_start và t_end đều khớp; OM_t nếu dự đoán thiếu mốc gold có; HALL_t nếu dự đoán có
  mốc gold không có; còn lại OM_t. Biến thể thời gian chia cho số gold.
Khác bản gốc (ghi trong notes): ngày đọc thành tập date (list rỗng/None/"" = không có mốc; bản gốc so chuỗi repr của
list); bài có gold mà không có dự đoán tính MATCH = 0 thay vì bị bỏ qua (lọc ở phía gọi nếu cần cách cũ).
"""
from __future__ import annotations

import datetime as dt
from functools import lru_cache

import dateparser
import numpy as np


@lru_cache(maxsize=None)
def _parse(s: str) -> dt.date | None:
    d = dateparser.parse(s, settings={"DATE_ORDER": "YMD", "PREFER_DAY_OF_MONTH": "first",
                                    "PREFER_MONTH_OF_YEAR": "first"})
    if d is None and "-" in s:                                              # "24-01-2020"
        d = dateparser.parse(s, settings={"DATE_ORDER": "DMY"})
    return d.date() if d is not None else None


def dates(v) -> frozenset[dt.date]:
    vals = v if isinstance(v, (list, tuple)) else [v]
    out = set()
    for x in vals:
        s = "" if x is None else str(x).strip()
        if s.lower() in ("", "none", "null", "[]"):
            continue
        d = _parse(s)
        if d is not None:
            out.add(d)
    return frozenset(out)


def _t(q, i):
    return dates(q[i]) if len(q) > i else frozenset()


def match_row(pred: list[tuple], gold: list[tuple], sim: np.ndarray, threshold: float) -> dict[str, int]:
    pairs = sorted(((sim[i, j], i, j) for i in range(len(pred)) for j in range(len(gold)) if sim[i, j] > threshold),
                   reverse=True)
    used_p, used_g, out = set(), set(), dict.fromkeys(("match", "match_t", "om_t", "hall_t"), 0)
    for _, i, j in pairs:
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        out["match"] += 1
        ps, pe, gs, ge = _t(pred[i], 3), _t(pred[i], 4), _t(gold[j], 3), _t(gold[j], 4)
        if ps == gs and pe == ge:
            out["match_t"] += 1
        elif (not ps and gs) or (not pe and ge):
            out["om_t"] += 1
        elif (ps and not gs) or (pe and not ge):
            out["hall_t"] += 1
        else:
            out["om_t"] += 1
    out.update(hall=len(pred) - out["match"], om=len(gold) - out["match"], n_pred=len(pred), n_gold=len(gold))
    return out


def summarize(rows: list[dict]) -> dict[str, float]:
    rows = [r for r in rows if r["n_gold"]]
    mean = lambda f: float(np.mean([f(r) for r in rows])) if rows else 0.0          # noqa: E731
    hall = mean(lambda r: r["hall"] / r["n_pred"] if r["n_pred"] else 0.0)
    p = mean(lambda r: 1 - r["hall"] / r["n_pred"] if r["n_pred"] else 0.0)
    rec = mean(lambda r: r["match"] / r["n_gold"])
    p_t = mean(lambda r: r["match_t"] / r["n_pred"] if r["n_pred"] else 0.0)        # fact đúng cả mốc thời gian
    r_t = mean(lambda r: r["match_t"] / r["n_gold"])
    f1 = lambda a, b: 2 * a * b / (a + b) if a + b else 0.0                        # noqa: E731
    return {"MATCH": rec, "HALL": hall, "OM": mean(lambda r: r["om"] / r["n_gold"]),
            "MATCH_t": r_t, "OM_t": mean(lambda r: r["om_t"] / r["n_gold"]),
            "HALL_t": mean(lambda r: r["hall_t"] / r["n_gold"]),
            "P": p, "R": rec, "F1": f1(p, rec), "P_t": p_t, "F1_t": f1(p_t, r_t), "n_rows": len(rows)}
