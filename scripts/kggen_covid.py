"""Baseline KGGen 0.4.0 (pip kg-gen) trên 2020-COVID-NYT với Gemma 4 qua llama-server: generate() trên từng nhóm
ngày (không gom cụm, không chia chunk — đoạn dài nhất ~5,4 nghìn ký tự). KGGen không có thời gian → 5-tuple
(h, r, t, [], []). model "hosted_vllm/local" (LiteLLM gọi thẳng /v1/chat/completions; tiền tố openai/ ở bản mới bị
ép sang Responses API), temperature 0, max_tokens 4096 như kernel baselines EMERGE.

Dùng: KGU_LLM_BASE_URL=http://127.0.0.1:8080/v1 python scripts/kggen_covid.py
      --data data/covid_nyt/2020_covid_nyt.jsonl --out out/kggen.jsonl [--limit N]
"""
from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path


def _script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def to_quintuples(relations) -> list[tuple]:
    return [(str(s), str(p), str(o), [], []) for s, p, o in sorted(relations)]


def main() -> None:
    from kg_gen import KGGen

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    rows = sorted(_script("score_covid_nyt").load_dataset(args.data), key=lambda r: (r["date"], r["row"]))[: args.limit]
    kg = KGGen(model="hosted_vllm/local", api_base=os.environ.get("KGU_LLM_BASE_URL", "http://127.0.0.1:8080/v1"),
               api_key=os.environ.get("KGU_LLM_API_KEY", "none"), temperature=0.0, max_tokens=4096)
    _script("run_covid_nyt").run_rows(rows, lambda text, date: to_quintuples(kg.generate(input_data=text).relations),
                                      args.out)
    print(f"kggen: xong {len(rows)} nhóm ngày → {args.out}", flush=True)


if __name__ == "__main__":
    main()
