"""Bộ chấm mềm viết lại từ repo EMERGE (hai cột C và G-R trong bảng chính của paper), để có số cùng thang
với bảng công bố mà không phải chạy cả `evaluation.s0x_evaluate_predictions`.

Cách chính chủ làm (src/evaluation/scorers/completeness_scorer.py, scorers/misc/graph_matching.py):
- Mỗi triple (gold lẫn dự đoán) thành một chuỗi nhãn "head relation tail", chữ thường, `_` -> khoảng trắng.
- Completeness (C): nhúng bằng sentence-transformer all-mpnet-base-v2; mỗi gold lấy cosine lớn nhất với mọi
  dự đoán cùng op, tính "đã phủ" nếu > 0,9. C = tỉ lệ gold đã phủ, gộp trên toàn bộ triple gold đã xác nhận.
  Không có vế precision: dự đoán thừa không bị trừ.
- G-BERTScore-R (G-R): BERTScore-F1 (bert-base-uncased, không idf) giữa mọi cặp gold x dự đoán trong một bài,
  ghép 1-1 bằng Hungarian, R = tổng điểm cặp đã ghép / số gold; trung bình theo bài. G-P = tổng / số dự đoán.
- Bài có gold mà không có dự đoán: mọi gold = 0. Bài không có gold cho op đó: bỏ qua.
Cần GPU để nhanh (chạy trên Kaggle); CPU vẫn chạy được với vài trăm bài.
"""
from __future__ import annotations

from typing import Iterable, Mapping, Sequence

from kgu.data.emerge import OPS, EmergeExample

TextTriple = tuple[str, str, str]
TextPredictions = Mapping[str, Mapping[str, set[TextTriple]]]      # hash_id -> mã op -> triple nhãn

ST_MODEL = "all-mpnet-base-v2"
BERT_MODEL = "bert-base-uncased"
THRESHOLD = 0.9


def triple_text(triple: Sequence[str]) -> str:
    """Chuỗi so khớp của một triple nhãn, chuẩn hóa như split_to_edges + normalize_triple_string của EMERGE."""
    return " ".join(" ".join(str(x).replace("_", " ") for x in triple).lower().strip().split())


def instance_labels(ex: EmergeExample, kg_labels: Mapping[str, str]) -> dict[str, str]:
    """Nhãn để đổi triple QID của mình sang chuỗi: nhãn KG-trước, đè bằng nhãn trên triple vàng, thiếu thì
    lấy chữ trong passage (giống scripts/export_emerge_predictions.py, tức đúng thứ đã gửi cho bộ chấm chính chủ)."""
    lab = dict(kg_labels)
    lab.update(ex.labels)
    for m in ex.mentions:
        lab.setdefault(m.qid, m.text)
    return lab


def to_text(triple: Sequence[str], lab: Mapping[str, str]) -> TextTriple:
    return tuple(lab.get(x, x) for x in triple)      # type: ignore[return-value]


def gold_texts(ex: EmergeExample, op: str) -> list[str]:
    return [triple_text(to_text(t, ex.labels)) for t in sorted(ex.gold[op])]


def load_models(device: str = "cuda"):
    """(sentence-transformer cho C, BERTScorer cho G-R). Import muộn để module dùng được khi không cài torch."""
    from bert_score import BERTScorer
    from sentence_transformers import SentenceTransformer

    st = SentenceTransformer(ST_MODEL, device=device)
    bs = BERTScorer(model_type=BERT_MODEL, lang="en", idf=False, device=device, batch_size=256)
    return st, bs


def completeness_hits(gold: list[list[str]], pred: list[list[str]], st, threshold: float = THRESHOLD) -> list[list[bool]]:
    """Mỗi gold của mỗi bài: có dự đoán nào cosine > threshold không."""
    from sentence_transformers import util

    flat_g = [t for g in gold for t in g]
    flat_p = [t for p in pred for t in p]
    emb_g = st.encode(flat_g, convert_to_tensor=True, show_progress_bar=False) if flat_g else None
    emb_p = st.encode(flat_p, convert_to_tensor=True, show_progress_bar=False) if flat_p else None
    out: list[list[bool]] = []
    ig = ip = 0
    for g, p in zip(gold, pred):
        if not g or not p:
            out.append([False] * len(g))
        else:
            sim = util.cos_sim(emb_g[ig:ig + len(g)], emb_p[ip:ip + len(p)])
            out.append([float(x) > threshold for x in sim.max(dim=1).values.tolist()])
        ig += len(g)
        ip += len(p)
    return out


def gbert_scores(gold: list[list[str]], pred: list[list[str]], bs) -> list[tuple[float, float]]:
    """(G-P, G-R) từng bài: BERTScore-F1 mọi cặp, ghép Hungarian (như get_bert_score_fast của EMERGE)."""
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    pairs: dict[tuple[str, str], int] = {}
    refs, cands = [], []
    for g, p in zip(gold, pred):
        for x in g:
            for y in p:
                if (x, y) not in pairs:
                    pairs[(x, y)] = len(refs)
                    refs.append(x)
                    cands.append(y)
    f1 = bs.score(cands=cands, refs=refs, verbose=False)[2].cpu().numpy() if refs else np.zeros(0)
    out: list[tuple[float, float]] = []
    for g, p in zip(gold, pred):
        if not g or not p:
            out.append((0.0, 0.0))
            continue
        m = np.array([[f1[pairs[(x, y)]] for y in p] for x in g])
        rows, cols = linear_sum_assignment(m, maximize=True)
        s = float(m[rows, cols].sum())
        out.append((s / len(p), s / len(g)))
    return out


def soft_scores(examples: Iterable[EmergeExample], preds: TextPredictions, st, bs,
                ops: Sequence[str] = ("x-triples", "e-triples", "ee-triples", "d-triples")) -> dict[str, dict]:
    """mã op -> {n (bài có gold), n_gold, C, G-R, G-P, per_instance: hash_id -> (C bài, G-R bài)}."""
    exs = list(examples)
    out: dict[str, dict] = {}
    for op in ops:
        rows = [ex for ex in exs if ex.gold[op]]
        gold = [gold_texts(ex, op) for ex in rows]
        pred = [sorted({triple_text(t) for t in preds.get(ex.hash_id, {}).get(op, ())}) for ex in rows]
        hits = completeness_hits(gold, pred, st)
        gj = gbert_scores(gold, pred, bs)
        n_gold = sum(len(g) for g in gold)
        out[op] = {
            "name": OPS[op], "n": len(rows), "n_gold": n_gold,
            "C": sum(h for hs in hits for h in hs) / n_gold if n_gold else 0.0,
            "G-R": sum(r for _, r in gj) / len(rows) if rows else 0.0,
            "G-P": sum(p for p, _ in gj) / len(rows) if rows else 0.0,
            "per_instance": {ex.hash_id: (sum(hs) / len(hs), r) for ex, hs, (_, r) in zip(rows, hits, gj)},
        }
    return out


def format_table(scores: Mapping[str, Mapping[str, dict]]) -> str:
    """Bảng thang 100 như paper: hàng = hệ, cột = op (C / G-R)."""
    ops = [op for op in ("x-triples", "e-triples", "ee-triples", "d-triples")
           if any(op in s for s in scores.values())]
    head = f"{'model':40s}" + "".join(f"{OPS[op] + ' C / G-R':>22s}" for op in ops)
    lines = [head, "-" * len(head)]
    for model, s in scores.items():
        cells = "".join(f"{100 * s[op]['C']:>11.1f} / {100 * s[op]['G-R']:<8.1f}" if op in s else f"{'-':>22s}"
                        for op in ops)
        lines.append(f"{model:40s}{cells}")
    return "\n".join(lines)
