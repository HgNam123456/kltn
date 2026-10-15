"""Pipeline của mình trên 2020-COVID-NYT: KG tích lũy theo thứ tự ngày (kgu.extract.text_kg.TemporalKG). Tuần tự
(mỗi nhóm ngày thấy KG của các ngày trước), ghi + flush từng dòng để kernel chết giữa chừng vẫn còn kết quả.
Ra jsonl {"row", "quintuples", "seconds"} cho scripts/score_covid_nyt.py; KG cuối ở <out>.kg.json.
Cờ (spec docs/superpowers/specs/2026-10-15-condense-canonicalize-design.md): --condense = v1 (rút câu đơn + trích bổ
sung); thêm --canon-every N = v2 (gộp tên k-means/LLM sau mỗi N nhóm ngày và sau nhóm cuối; log quyết định ở
<out>.merges.jsonl, thống kê lượt ở <out>.canon.json). Mỗi dòng output có thêm "calls" và thống kê của TemporalKG.

Dùng: KGU_LLM_BASE_URL=http://127.0.0.1:8080/v1 python scripts/run_covid_nyt.py
      --data data/covid_nyt/2020_covid_nyt.jsonl --out out/mine.jsonl [--limit N]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

from kgu.extract.text_kg import TemporalKG

_spec = importlib.util.spec_from_file_location("score_covid_nyt", Path(__file__).with_name("score_covid_nyt.py"))
_score = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_score)


def run_rows(rows: list[dict], step, out: Path, extra=None, after=None) -> None:
    """step(text, date) -> list 5-tuple; dùng chung cho mình (TemporalKG) và baseline (atom_covid, kggen_covid).
    extra() -> dict ghép vào dòng output; after(k) gọi sau nhóm ngày thứ k (đếm từ 1, theo thứ tự ngày)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf8") as f:
        for k, r in enumerate(sorted(rows, key=lambda r: (r["date"], r["row"])), 1):
            t0 = time.time()
            try:
                qs = step(r["text"], r["date"])
            except Exception as e:                   # lỗi một nhóm ngày → rỗng (tính 0 điểm), chạy tiếp
                print(f"!!! row {r['row']}: {e!r}"[:300], flush=True)
                qs = []
            rec = {"row": r["row"], "quintuples": [list(q) for q in qs], "seconds": round(time.time() - t0, 2)}
            if extra:
                rec.update(extra())
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if after:
                after(k)


def main() -> None:
    from kgu.llm import client_from_env

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--condense", action="store_true", help="v1: rút câu đơn + trích bổ sung")
    ap.add_argument("--canon-every", type=int, default=0, help="v2: gộp tên sau mỗi N nhóm ngày; 0 = tắt")
    ap.add_argument("--tau", type=float, default=0.75)
    ap.add_argument("--embedder", default="sentence-transformers/all-mpnet-base-v2")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    rows = sorted(_score.load_dataset(args.data), key=lambda r: (r["date"], r["row"]))[: args.limit]
    kg, llm = TemporalKG(condense=args.condense), client_from_env()
    canon, rounds, last = None, [], [0]
    if args.canon_every:
        from kgu.extract.canonicalize import Canonicalizer, sentence_embedder

        log = Path(f"{args.out}.merges.jsonl")
        log.unlink(missing_ok=True)
        canon = Canonicalizer(sentence_embedder(args.embedder, args.device), llm, tau=args.tau, log_path=log)

    def extra() -> dict:
        n, last[0] = llm.n_calls - last[0], llm.n_calls
        return {"calls": n, **kg.last_stats}

    def after(k: int) -> None:
        if canon and (k % args.canon_every == 0 or k == len(rows)):
            t0 = time.time()
            stats = canon.run_round(kg)
            rounds.append({"after_group": k, "seconds": round(time.time() - t0, 1), "stats": stats})
            print(f"gộp tên sau nhóm {k}: {stats}", flush=True)
            last[0] = llm.n_calls                    # lệnh gọi gộp tên không tính vào nhóm ngày sau

    run_rows(rows, lambda text, date: kg.update(text, date, llm), args.out, extra=extra, after=after)
    Path(f"{args.out}.kg.json").write_text(json.dumps(kg.facts, ensure_ascii=False), encoding="utf8")
    Path(f"{args.out}.aliases.json").write_text(
        json.dumps({k: sorted(v) for k, v in kg.aliases.items()}, ensure_ascii=False), encoding="utf8")
    if canon:
        Path(f"{args.out}.canon.json").write_text(json.dumps(rounds, ensure_ascii=False, indent=1), encoding="utf8")
    print(f"{len(rows)} nhóm ngày, KG cuối {len(kg.facts)} fact, {llm.n_calls} lệnh gọi LLM → {args.out}")


if __name__ == "__main__":
    main()
