"""Pipeline của mình trên MINE-1 (benchmark của KGGen, HF josancamon/kg-gen-MINE-evaluation-dataset: 100 bài luận,
mỗi bài 15 fact). Mỗi bài một KG riêng, dựng bằng kgu.extract.text_kg.TemporalKG như bản COVID-NYT hiện tại: đưa từng
đoạn của bài vào lần lượt (rút câu đơn rồi trích theo câu, thấy fact đã lưu có liên quan), cuối bài gộp tên một vòng
(kgu.extract.canonicalize, k-means + một lệnh gọi LLM mỗi cụm). Bài luận không có thời gian → bỏ t_start/t_end.
Ra jsonl {"id", "graph": {"entities", "edges", "relations"}, ...} đúng khuôn Graph của kg-gen để scripts/score_mine.py
chấm như KG của KGGen. Các bài chạy song song (--workers), embedder dùng chung có khóa.

Dùng: KGU_LLM_BASE_URL=http://127.0.0.1:8080/v1 python scripts/run_mine.py --data mine.parquet --out out/mine.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

MIN_WORDS = 8          # đoạn ngắn hơn (tiêu đề, câu lẻ) ghép vào đoạn sau
OBS_DATE = "none"      # bài luận không có ngày quan sát


def paragraphs(essay: str) -> list[str]:
    """Tách bài thành đoạn theo dòng trống; bỏ ``` bao quanh; đoạn dưới MIN_WORDS từ dồn vào đoạn kế tiếp."""
    blocks = [b.strip() for b in re.split(r"\n\s*\n", essay.replace("```", "")) if b.strip()]
    out, buf = [], []
    for b in blocks:
        buf.append(b)
        if sum(len(x.split()) for x in buf) >= MIN_WORDS:
            out.append("\n".join(buf))
            buf = []
    if buf:
        out.append("\n".join(buf))
    return out


def graph_dict(facts: list[list]) -> dict:
    """facts của TemporalKG -> Graph của kg-gen; quan hệ snake_case viết lại thành chữ thường có khoảng trắng."""
    rels = sorted({(h, r.replace("_", " "), t) for h, r, t, *_ in facts if h and t and h != t})
    return {"entities": sorted({x for h, _, t in rels for x in (h, t)}), "edges": sorted({r for _, r, _ in rels}),
            "relations": [list(x) for x in rels]}


def build(essay: str, llm, embed=None, log_path=None) -> tuple[dict, dict]:
    from kgu.extract.text_kg import TemporalKG

    kg = TemporalKG(condense=True)
    paras = paragraphs(essay)
    stats = {"paragraphs": len(paras), "errors": 0}
    for p in paras:
        try:
            kg.update(p, OBS_DATE, llm)
        except Exception as e:                       # lỗi một đoạn → bỏ đoạn đó, giữ phần KG đã có
            stats["errors"] += 1
            stats["last_error"] = repr(e)[:200]
    if embed is not None and kg.facts:
        from kgu.extract.canonicalize import Canonicalizer

        try:
            stats["canon"] = Canonicalizer(embed, llm, log_path=log_path).run_round(kg)
        except Exception as e:
            stats["canon_error"] = repr(e)[:200]
    return graph_dict(kg.facts), stats


def main() -> None:
    import pandas as pd

    from kgu.llm import client_from_env

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="parquet của MINE (đường dẫn hoặc URL)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-canon", action="store_true")
    ap.add_argument("--embedder", default="sentence-transformers/all-mpnet-base-v2")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    rows = [r for r in pd.read_parquet(args.data).to_dict("records") if r["essay_content"]][: args.limit]
    llm = client_from_env()
    embed = None
    if not args.no_canon:
        from kgu.extract.canonicalize import sentence_embedder

        raw, lock = sentence_embedder(args.embedder, args.device), threading.Lock()

        def embed(names):
            with lock:
                return raw(names)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    logs = args.out.with_suffix(".merges")
    logs.mkdir(exist_ok=True)

    def one(r):
        t0 = time.time()
        graph, stats = build(r["essay_content"], llm, embed, logs / f"{r['id']}.jsonl")
        return {"id": int(r["id"]), "graph": graph, "seconds": round(time.time() - t0, 1), **stats}

    with ThreadPoolExecutor(max_workers=args.workers) as pool, args.out.open("w", encoding="utf8") as f:
        for rec in pool.map(one, rows):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
    print(f"{len(rows)} bài, {llm.n_calls} lệnh gọi LLM → {args.out}", flush=True)


if __name__ == "__main__":
    main()
