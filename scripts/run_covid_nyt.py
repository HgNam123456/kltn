"""Pipeline của mình trên 2020-COVID-NYT: KG tích lũy theo thứ tự ngày (kgu.extract.text_kg.TemporalKG). Tuần tự
(mỗi nhóm ngày thấy KG của các ngày trước), ghi + flush từng dòng để kernel chết giữa chừng vẫn còn kết quả.
Ra jsonl {"row", "quintuples", "seconds"} cho scripts/score_covid_nyt.py; KG cuối ở <out>.kg.json.

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


def run_rows(rows: list[dict], kg, llm, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf8") as f:
        for r in sorted(rows, key=lambda r: (r["date"], r["row"])):
            t0 = time.time()
            try:
                qs = kg.update(r["text"], r["date"], llm)
            except Exception as e:                   # lỗi một nhóm ngày → rỗng (tính 0 điểm), chạy tiếp
                print(f"!!! row {r['row']}: {e!r}"[:300], flush=True)
                qs = []
            f.write(json.dumps({"row": r["row"], "quintuples": [list(q) for q in qs],
                                "seconds": round(time.time() - t0, 2)}, ensure_ascii=False) + "\n")
            f.flush()


def main() -> None:
    from kgu.llm import client_from_env

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    rows = sorted(_score.load_dataset(args.data), key=lambda r: (r["date"], r["row"]))[: args.limit]
    kg, llm = TemporalKG(), client_from_env()
    run_rows(rows, kg, llm, args.out)
    Path(f"{args.out}.kg.json").write_text(json.dumps(kg.facts, ensure_ascii=False), encoding="utf8")
    print(f"{len(rows)} nhóm ngày, KG cuối {len(kg.facts)} fact, {llm.n_calls} lệnh gọi LLM → {args.out}")


if __name__ == "__main__":
    main()
