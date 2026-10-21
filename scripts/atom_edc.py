"""ATOM (Lairgi et al., Findings EACL 2026; AuvaLab/itext2kg 1.1.0) trên benchmark của EDC (WebNLG / REBEL / Wiki-NRE
trong repo clear-nus/edc: một câu mỗi dòng), Gemma 4 qua llama-server. Hai bước như script đánh giá của ATOM, giống
scripts/atom_covid.py --mode atom: AtomicFact trên từng câu, rồi RelationshipsExtractor trên từng factoid với
system_query = Prompt.temporal_system_query + Prompt.EXAMPLES (prompt và ví dụ của ATOM, không dùng few-shot của bộ).
Câu của EDC không có ngày quan sát → obs_timestamp = OBS.
5-tuple → triple (head, relation, tail), bỏ t_start/t_end vì benchmark EDC chỉ chấm triple; ngày mà ATOM đặt vào
t_start/t_end (vd. năm sinh) vì thế mất — đếm ở stats. Bỏ triple trùng hệt trong một câu (hai 5-tuple chỉ khác thời gian).
Đầu ra: <out> (định dạng canon_kg.txt của EDC, chấm thô), <out>.json (list triple mỗi câu, nạp vào Schema Definition +
Canonicalization của EDC thay cho OIE của EDC), <out>.factoids.json, <out>.stats.json.

Dùng: KGU_LLM_BASE_URL=http://127.0.0.1:8080/v1 python scripts/atom_edc.py --input edc/datasets/webnlg.txt \
          --out out/webnlg/atom/oie_kg.txt [--limit N] [--chunk 32]
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import time
from pathlib import Path

OBS = "not given (the text has no observation date)"


def _script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def to_triples(quintuples: list[tuple]) -> tuple[list[list[str]], int]:
    """5-tuple → triple, bỏ trùng giữ thứ tự; trả (triple, số 5-tuple có thời gian)."""
    seen, out, timed = set(), [], 0
    for h, r, t, ts, te in quintuples:
        timed += bool(ts or te)
        if (h, r, t) not in seen:
            seen.add((h, r, t))
            out.append([h, r, t])
    return out, timed


def factoid_list(o) -> list[str]:
    return [str(f).strip() for f in (getattr(o, "atomic_fact", None) or []) if str(f).strip()]


def main() -> None:
    from itext2kg.atom.models import AtomicFact, Prompt, RelationshipsExtractor

    atom = _script("atom_covid")
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=32, help="số câu mỗi lô (mỗi bước gửi cả lô cùng lúc)")
    args = ap.parse_args()
    texts = args.input.read_text(encoding="utf8").splitlines()[: args.limit]
    parser, loop = atom.make_parser(), asyncio.new_event_loop()
    run = loop.run_until_complete
    system_query = Prompt.temporal_system_query(obs_timestamp=OBS) + Prompt.EXAMPLES.value
    factoids: list[list[str]] = []
    triples: list[list[list[str]]] = []
    stats = {"sentences": len(texts), "factoid_errors": 0, "tuple_errors": 0, "quintuples": 0, "timed": 0}
    t0 = time.time()
    for i in range(0, len(texts), args.chunk):
        chunk = texts[i:i + args.chunk]
        try:
            outs = run(parser.extract_information_as_json_for_context(AtomicFact, chunk, return_exceptions=True))
        except Exception as e:
            print(f"!!! factoids {i}: {e!r}"[:300], flush=True)
            outs = []
        facts = [factoid_list(outs[j]) if j < len(outs) else [] for j in range(len(chunk))]
        stats["factoid_errors"] += sum(1 for j in range(len(chunk)) if j >= len(outs) or isinstance(outs[j], Exception))
        flat = [f for fs in facts for f in fs]
        try:
            rel_outs = run(parser.extract_information_as_json_for_context(
                output_data_structure=RelationshipsExtractor, contexts=flat, system_query=system_query,
                return_exceptions=True)) if flat else []
        except Exception as e:
            print(f"!!! quintuples {i}: {e!r}"[:300], flush=True)
            rel_outs = []
        stats["tuple_errors"] += sum(1 for k in range(len(flat)) if k >= len(rel_outs) or isinstance(rel_outs[k], Exception))
        k = 0
        for fs in facts:
            quints = atom.flatten(rel_outs[k:k + len(fs)])
            k += len(fs)
            ts, timed = to_triples(quints)
            stats["quintuples"] += len(quints)
            stats["timed"] += timed
            factoids.append(fs)
            triples.append(ts)
        print(f"atom {min(i + args.chunk, len(texts))}/{len(texts)} ({round(time.time() - t0)}s)", flush=True)
    stats["factoids"] = sum(map(len, factoids))
    stats["triples"] = sum(map(len, triples))
    stats["seconds"] = round(time.time() - t0, 1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(str(t) for t in triples) + "\n", encoding="utf8")
    Path(f"{args.out}.json").write_text(json.dumps(triples, ensure_ascii=False), encoding="utf8")
    Path(f"{args.out}.factoids.json").write_text(json.dumps(factoids, ensure_ascii=False), encoding="utf8")
    Path(f"{args.out}.stats.json").write_text(json.dumps(stats, indent=1), encoding="utf8")
    print(json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
