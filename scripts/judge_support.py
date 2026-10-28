"""Độ chính xác theo văn bản nguồn (không cần gold): LLM judge xét từng triple một hệ trích ra có được văn bản nguồn nói
tới không. Bổ sung cho các chỉ số chỉ đo recall (Completeness của EMERGE, MINE-1) và cho benchmark có gold thiếu
(NYT11). Mỗi lệnh gọi một văn bản + tối đa --chunk triple đánh số; judge trả có/không cho từng số (JSON schema). Judge
là Gemma 4 E4B qua llama-server (cùng mô hình với các hệ → có thể thiên vị; đo độ tin bằng mẫu kiểm tay).
Triple 5-tuple của COVID-NYT bỏ thời gian (chỉ xét head/relation/tail).

Dùng: KGU_LLM_BASE_URL=... python scripts/judge_support.py --system mine=covid:out/v2.jsonl --covid-data covid.jsonl \
          [--mine-data mine.parquet] [--lines-input input.txt] --out support.json
Ra support.json (mỗi hệ: triple, đã phán, được hỗ trợ, tỉ lệ micro, tỉ lệ trung bình theo văn bản) và
support.<hệ>.jsonl (phán quyết từng triple).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pydantic import BaseModel

SYSTEM = """You check knowledge-graph triples against the source text they were extracted from.
A triple "head | relation | tail" is SUPPORTED when the text states it or it follows directly from the text: paraphrases are fine, pronouns may be resolved, and the relation may be written as a code or identifier (for example /people/person/place_lived) — judge what it means.
It is NOT supported when the text does not say it, says something different, or the triple is malformed or meaningless (the head or tail is not something the text talks about, or the relation does not hold between them in the text).
Give exactly one verdict for every numbered triple."""


class Verdict(BaseModel):
    id: int
    supported: bool


class Verdicts(BaseModel):
    verdicts: list[Verdict]


def _outputs():
    spec = importlib.util.spec_from_file_location("kg_outputs", Path(__file__).with_name("kg_outputs.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def prompt(text: str, triples: list[list[str]], start: int) -> str:
    listing = "\n".join(f"{start + i}. {h} | {r} | {t}" for i, (h, r, t) in enumerate(triples))
    return f"TEXT:\n{text}\n\nTRIPLES:\n{listing}"


def judge_doc(doc: dict, llm, chunk: int) -> list[dict]:
    out = []
    ts = doc["triples"]
    for s in range(0, len(ts), chunk):
        part = ts[s:s + chunk]
        try:
            got = {v.id: v.supported for v in llm.complete_json(SYSTEM, prompt(doc["text"], part, 1), Verdicts).verdicts}
            err = None
        except Exception as e:
            got, err = {}, repr(e)[:200]
        for i, t in enumerate(part, 1):
            out.append({"doc": doc["id"], "triple": t, "supported": got.get(i), **({"error": err} if err else {})})
    return out


def summarize(records: list[dict]) -> dict:
    judged = [r for r in records if r["supported"] is not None]
    by_doc: dict = {}
    for r in judged:
        by_doc.setdefault(r["doc"], []).append(r["supported"])
    return {"triples": len(records), "judged": len(judged), "supported": sum(r["supported"] for r in judged),
            "support_rate": round(sum(r["supported"] for r in judged) / len(judged), 4) if judged else None,
            "support_rate_doc_mean": round(sum(sum(v) / len(v) for v in by_doc.values()) / len(by_doc), 4)
            if by_doc else None, "docs": len(by_doc), "errors": sum("error" in r for r in records)}


def main() -> None:
    from kgu.llm import client_from_env

    ap = argparse.ArgumentParser()
    ap.add_argument("--system", action="append", default=[])
    ap.add_argument("--covid-data")
    ap.add_argument("--mine-data")
    ap.add_argument("--lines-input")
    ap.add_argument("--chunk", type=int, default=30)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    ko = _outputs()
    src = ko.Sources(args.covid_data, args.mine_data, args.lines_input)
    llm = client_from_env()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    summary = json.loads(args.out.read_text(encoding="utf8")) if args.out.exists() else {}
    for spec in args.system:
        name, docs = ko.load(spec, src)
        docs = [d for d in docs if d["triples"]]
        with ThreadPoolExecutor(args.workers) as pool:
            records = [r for rs in pool.map(lambda d: judge_doc(d, llm, args.chunk), docs) for r in rs]
        safe = name.replace(":", "_").replace("/", "_")
        with args.out.with_name(f"{args.out.stem}.{safe}.jsonl").open("w", encoding="utf8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        summary[name] = summarize(records)
        print(name, summary[name], flush=True)
        args.out.write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf8")


if __name__ == "__main__":
    main()
