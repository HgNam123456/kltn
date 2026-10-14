"""Pipeline của mình (module Add, quan hệ ép theo schema — kgu.extract.text_kg.extract_schema) trên WebNLG test của
EDC (clear-nus/edc datasets/webnlg.txt, 1.165 câu). Ra file định dạng canon_kg.txt của EDC (1 dòng/câu, str của list
triple) để chấm bằng evaluate/evaluation_script.py của EDC. Dùng cùng file few-shot OIE của EDC cho WebNLG.

Dùng: KGU_LLM_BASE_URL=http://127.0.0.1:8080/v1 python scripts/run_webnlg.py --input edc/datasets/webnlg.txt
      --schema edc/schemas/webnlg_schema.csv --examples edc/few_shot_examples/webnlg/oie_few_shot_examples.txt
      --out out/mine_webnlg.txt [--limit N] [--workers 4]
"""
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from kgu.extract.text_kg import extract_schema
from kgu.llm import client_from_env


def read_schema(path: Path) -> dict[str, str]:
    with open(path, encoding="utf8", newline="") as f:
        return {row[0].strip(): (row[1].strip() if len(row) > 1 else "") for row in csv.reader(f) if row and row[0].strip()}


def format_line(triples) -> str:
    return str([list(t) for t in triples])


def run_all(texts: list[str], extract, workers: int) -> list[str]:
    def one(t: str) -> str:
        try:
            return format_line(extract(t))
        except Exception as e:                       # lỗi một câu → dòng "[]", giữ thứ tự dòng cho script chấm
            print(f"!!! {e!r}"[:300], flush=True)
            return "[]"

    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(one, texts))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--schema", type=Path, required=True)
    ap.add_argument("--examples", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    texts = args.input.read_text(encoding="utf8").splitlines()[: args.limit]
    rels = read_schema(args.schema)
    examples = args.examples.read_text(encoding="utf8") if args.examples else ""
    llm = client_from_env()
    lines = run_all(texts, lambda t: extract_schema(t, rels, llm, examples), args.workers)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf8")
    print(f"{len(lines)} câu, {sum(line != '[]' for line in lines)} câu có triple, {llm.n_calls} lệnh gọi LLM → {args.out}")


if __name__ == "__main__":
    main()
