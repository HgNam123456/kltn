"""KGGen 0.4.0 trên benchmark định dạng EDC (một văn bản mỗi dòng), Gemma 4 qua llama-server, cấu hình như
scripts/kggen_covid.py (hosted_vllm/local, temperature 0, max_tokens 4096, không gom cụm). Quan hệ của KGGen là chữ tự
do → ra <out> (định dạng canon_kg.txt của EDC, chấm thô) và <out>.json (list triple mỗi dòng) để nạp vào Schema
Definition + Canonicalization của EDC (biến $EDC_OIE_FROM, scripts/edc_patch.py), như ATOM ở scripts/atom_edc.py.
Lỗi một dòng → [] (ghi stats).

Dùng: KGU_LLM_BASE_URL=http://127.0.0.1:8080/v1 python scripts/kggen_edc.py --input input.txt --out out/kggen/oie_kg.txt
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def to_triples(relations) -> list[list[str]]:
    return [[str(s), str(p), str(o)] for s, p, o in sorted(relations)]


def main() -> None:
    from kg_gen import KGGen

    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    texts = args.input.read_text(encoding="utf8").splitlines()[: args.limit]
    kg = KGGen(model="hosted_vllm/local", api_base=os.environ.get("KGU_LLM_BASE_URL", "http://127.0.0.1:8080/v1"),
               api_key=os.environ.get("KGU_LLM_API_KEY", "none"), temperature=0.0, max_tokens=4096)
    errors = []

    def one(i_text):
        i, text = i_text
        try:
            return to_triples(kg.generate(input_data=text).relations)
        except Exception as e:
            errors.append({"line": i, "error": repr(e)[:200]})
            return []

    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as pool:
        triples = list(pool.map(one, enumerate(texts)))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(str(t) for t in triples) + "\n", encoding="utf8")
    Path(f"{args.out}.json").write_text(json.dumps(triples, ensure_ascii=False), encoding="utf8")
    stats = {"lines": len(texts), "triples": sum(map(len, triples)), "empty": sum(not t for t in triples),
             "errors": len(errors), "seconds": round(time.time() - t0, 1), "error_samples": errors[:5]}
    Path(f"{args.out}.stats.json").write_text(json.dumps(stats, indent=1), encoding="utf8")
    print(json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
