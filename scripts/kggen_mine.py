"""Baseline KGGen 0.4.0 (pip kg-gen) trên MINE-1 với Gemma 4 qua llama-server: generate() trên từng bài luận, cùng
cấu hình model như scripts/kggen_covid.py ("hosted_vllm/local", temperature 0, max_tokens 4096); --cluster bật bước
gom cụm thực thể/quan hệ của KGGen. Các bài chạy song song (--workers) như experiments/MINE/_1_evaluation.py.
Ra jsonl {"id", "graph", "seconds"[, "error"]} cho scripts/score_mine.py.

Dùng: KGU_LLM_BASE_URL=http://127.0.0.1:8080/v1 python scripts/kggen_mine.py --data mine.parquet --out out/kggen.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def to_dict(graph) -> dict:
    return {"entities": sorted(map(str, graph.entities)), "edges": sorted(map(str, graph.edges)),
            "relations": [[str(s), str(p), str(o)] for s, p, o in sorted(graph.relations)]}


def main() -> None:
    import pandas as pd
    from kg_gen import KGGen

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--cluster", action="store_true")
    args = ap.parse_args()
    rows = [r for r in pd.read_parquet(args.data).to_dict("records") if r["essay_content"]][: args.limit]
    kg = KGGen(model="hosted_vllm/local", api_base=os.environ.get("KGU_LLM_BASE_URL", "http://127.0.0.1:8080/v1"),
               api_key=os.environ.get("KGU_LLM_API_KEY", "none"), temperature=0.0, max_tokens=4096)

    def one(r):
        t0 = time.time()
        rec = {"id": int(r["id"])}
        try:
            rec["graph"] = to_dict(kg.generate(input_data=r["essay_content"], cluster=args.cluster))
        except Exception as e:                     # lỗi một bài → KG rỗng (mọi fact 0 điểm)
            rec["graph"], rec["error"] = None, repr(e)[:300]
        rec["seconds"] = round(time.time() - t0, 1)
        return rec

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool, args.out.open("w", encoding="utf8") as f:
        for rec in pool.map(one, rows):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
    print(f"kggen: xong {len(rows)} bài → {args.out}", flush=True)


if __name__ == "__main__":
    main()
