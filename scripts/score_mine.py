"""Chấm MINE-1 như experiments/MINE/_1_evaluation.py của KGGen (stair-lab/kg-gen @ main): với mỗi fact (query),
KGGen.retrieve lấy 8 node gần nhất theo all-MiniLM-L6-v2 rồi mở 2 bước (kg-gen 0.4.0), một LLM judge (dspy
ChainOfThought, signature EvaluateResponse chép nguyên văn) trả 1 nếu ngữ cảnh chứa thông tin của fact. Điểm = trung
bình theo bài của tỉ lệ fact đúng. Khác bản gốc DUY NHẤT ở judge: Gemma 4 E4B qua llama-server (temperature 0) thay cho
openai/gpt-5 — nên chấm lại cả KG phát hành sẵn trong dataset (kggen / graphrag_kg / openie_kg) để biết judge làm lệch
bao nhiêu so với cột *_accuracy của dataset.

Dùng: KGU_LLM_BASE_URL=http://127.0.0.1:8080/v1 python scripts/score_mine.py --data mine.parquet \
          --kg mine=out/mine.jsonl --kg kggen-gemma=out/kggen.jsonl --released kggen graphrag_kg openie_kg \
          --out out/mine_scores.json
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

RETRIEVAL_MODEL = "all-MiniLM-L6-v2"


def verdict(x) -> int:
    """Đầu ra evaluation của judge -> 0/1 (chuỗi lạ, None -> 0)."""
    try:
        return 1 if int(str(x).strip()) == 1 else 0
    except (TypeError, ValueError):
        return 0


def accuracy(per_essay: dict[int, list[int]]) -> dict:
    """Trung bình theo bài (như bảng của KGGen) + micro trên mọi fact."""
    accs = [sum(v) / len(v) for v in per_essay.values() if v]
    n = sum(len(v) for v in per_essay.values())
    return {"essays": len(accs), "queries": n, "accuracy": round(sum(accs) / len(accs), 5) if accs else 0.0,
            "micro": round(sum(sum(v) for v in per_essay.values()) / n, 5) if n else 0.0}


def released_accuracy(values) -> dict:
    """Cột kggen_accuracy / graphrag_accuracy / openie_accuracy của dataset ("53.33%") -> trung bình theo bài."""
    accs = [float(v.rstrip("%")) / 100 for v in values if isinstance(v, str) and v.strip()]
    return {"essays": len(accs), "accuracy": round(sum(accs) / len(accs), 5) if accs else 0.0}


def make_judge(base_url: str, api_key: str):
    import dspy

    class EvaluateResponse(dspy.Signature):
        """Determine whether the context contains the information stated in the correct answer. Respond with 1 if yes, 0 if no."""

        context: str = dspy.InputField(desc="The context to evaluate")
        correct_answer: str = dspy.InputField(desc="The correct answer to check for")
        evaluation: int = dspy.OutputField(desc="1 if context contains the correct answer, 0 otherwise")

    lm = dspy.LM(model="hosted_vllm/local", api_base=base_url, api_key=api_key, temperature=0.0, max_tokens=2048,
                 cache=False)
    program = dspy.ChainOfThought(EvaluateResponse)

    def judge(context: str, answer: str) -> int:
        with dspy.context(lm=lm):
            return verdict(program(context=context, correct_answer=answer).evaluation)

    return judge


def score_graphs(graphs: dict[int, dict], queries: dict[int, list[str]], judge, workers: int) -> tuple[dict, list]:
    """graphs: id bài -> Graph dict của kg-gen. Bài thiếu KG (lỗi dựng) -> mọi fact 0."""
    from kg_gen import KGGen

    kg = KGGen(retrieval_model=RETRIEVAL_MODEL)
    jobs = []
    for i, qs in queries.items():
        g = graphs.get(i)
        if g and g.get("relations"):
            graph = kg.to_nx(KGGen.from_dict(g))
            emb, _ = kg.generate_embeddings(graph)
            jobs += [(i, q, kg.retrieve(q, emb, graph)[2]) for q in qs]
        else:
            jobs += [(i, q, None) for q in qs]

    def one(job):
        i, q, ctx = job
        try:
            return i, q, ctx, judge(ctx, q) if ctx else 0
        except Exception as e:                      # lỗi judge một fact -> 0, ghi lại
            return i, q, ctx, f"error: {e!r}"[:200]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(one, jobs))
    per_essay: dict[int, list[int]] = {i: [] for i in queries}
    for i, _, _, v in results:
        per_essay[i].append(v if isinstance(v, int) else 0)
    out = accuracy(per_essay)
    out["judge_errors"] = sum(not isinstance(v, int) for *_, v in results)
    out["no_graph"] = sum(1 for i in queries if not (graphs.get(i) or {}).get("relations"))
    return out, [{"id": i, "query": q, "context": c, "evaluation": v} for i, q, c, v in results]


def main() -> None:
    import pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--kg", action="append", default=[], help="tên=jsonl {id, graph} (scripts/run_mine.py, kggen_mine.py)")
    ap.add_argument("--released", nargs="*", default=[], help="cột KG phát hành trong dataset: kggen graphrag_kg openie_kg")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rows = [r for r in pd.read_parquet(args.data).to_dict("records") if r["essay_content"]]
    queries = {int(r["id"]): [str(q) for q in r["generated_queries"]] for r in rows}
    judge = make_judge(os.environ.get("KGU_LLM_BASE_URL", "http://127.0.0.1:8080/v1"),
                       os.environ.get("KGU_LLM_API_KEY", "none"))
    systems: dict[str, dict[int, dict]] = {}
    for spec in args.kg:
        name, path = spec.split("=", 1)
        systems[name] = {int(d["id"]): d["graph"] for d in map(json.loads, Path(path).read_text(encoding="utf8").splitlines())}
    for col in args.released:
        systems[f"released:{col}"] = {int(r["id"]): ({k: [list(x) if not isinstance(x, str) else x for x in v]
                                                      for k, v in r[col].items()} if r[col] is not None else None)
                                      for r in rows}

    summary = {"judge": "gemma-4-E4B (llama-server, dspy ChainOfThought, temperature 0)",
               "retrieval": f"kg-gen 0.4.0 KGGen.retrieve, {RETRIEVAL_MODEL}, k=8, depth=2",
               "essays": len(queries), "systems": {}}
    for col in ("kggen", "graphrag", "openie"):
        if f"{col}_accuracy" in rows[0]:
            summary[f"released_judge:{col}"] = released_accuracy([r[f"{col}_accuracy"] for r in rows])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for name, graphs in systems.items():
        res, recs = score_graphs(graphs, queries, judge, args.workers)
        summary["systems"][name] = res
        print(f"=== {name}: {res}", flush=True)
        safe = name.replace(":", "_").replace("/", "_")
        with args.out.with_name(f"{args.out.stem}.{safe}.jsonl").open("w", encoding="utf8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        args.out.write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf8")
    print(json.dumps(summary, indent=1, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
