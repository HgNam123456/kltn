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
import re
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
    """Cột kggen_accuracy / graphrag_accuracy / openie_accuracy của dataset (0.5333 hoặc "53.33%") -> trung bình theo bài."""
    accs = []
    for v in values:
        if isinstance(v, str) and v.strip():
            accs.append(float(v.strip().rstrip("%")) / 100)
        elif isinstance(v, (int, float)) and v == v:          # bỏ None, NaN
            accs.append(float(v))
    return {"essays": len(accs), "accuracy": round(sum(accs) / len(accs), 5) if accs else 0.0}


def _norm(s) -> str:
    return re.sub(r"\W+", " ", str(s).lower()).strip()


def realign(rows: list[dict], kg_col: str, resp_col: str) -> tuple[dict[int, dict], dict]:
    """Dataset phát hành lệch hàng: ở ~80% hàng, cột KG + cột responses là của bài ở hàng TRƯỚC (KG hàng i chấm bằng
    fact của bài i-1). Gán mỗi KG về bài có generated_queries trùng nhiều nhất với các correct_answer trong responses
    cùng hàng (>= nửa số câu); hàng không có responses thì theo tỉ lệ từ trong tên thực thể của KG có mặt trong bài.
    Trả (id bài -> Graph dict, thống kê cách gán)."""
    queries = [{_norm(q) for q in r["generated_queries"]} for r in rows]
    words = [set(_norm(r["essay_content"]).split()) for r in rows]
    best: dict[int, tuple[float, dict]] = {}
    how = {"by_answers": 0, "by_words": 0, "dropped": 0}
    for r in rows:
        g = r[kg_col]
        if g is None:
            continue
        answers = {_norm(a["correct_answer"]) for a in (r[resp_col] if r[resp_col] is not None else [])
                   if a.get("correct_answer")}
        j, score, kind = None, 0.0, None
        if answers:
            j = max(range(len(rows)), key=lambda k: len(answers & queries[k]))
            score, kind = len(answers & queries[j]), "by_answers"
            if score < max(1, len(answers) // 2):
                j = None
        if j is None:
            kw = {w for e in g["entities"] for w in _norm(e).split()}
            if kw:
                j = max(range(len(rows)), key=lambda k: len(kw & words[k]))
                score, kind = len(kw & words[j]) / len(kw), "by_words"
                if score == 0:
                    j = None
        if j is None:
            how["dropped"] += 1
            continue
        target = int(rows[j]["id"])
        if target in best:                       # hai KG về cùng một bài: giữ KG khớp tốt hơn
            how["dropped"] += 1
            if best[target][0] >= score:
                continue
        best[target] = (score, g)
        how[kind] += 1
    return {k: g for k, (_, g) in best.items()}, how


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

    all_rows = pd.read_parquet(args.data).to_dict("records")
    rows = [r for r in all_rows if r["essay_content"]]
    queries = {int(r["id"]): [str(q) for q in r["generated_queries"]] for r in rows}
    judge = make_judge(os.environ.get("KGU_LLM_BASE_URL", "http://127.0.0.1:8080/v1"),
                       os.environ.get("KGU_LLM_API_KEY", "none"))
    systems: dict[str, dict[int, dict]] = {}
    for spec in args.kg:
        name, path = spec.split("=", 1)
        systems[name] = {int(d["id"]): d["graph"] for d in map(json.loads, Path(path).read_text(encoding="utf8").splitlines())}
    realigned = {}
    for col in args.released:
        src = [{**r, col: ({k: [list(map(str, x)) if not isinstance(x, str) else x for x in v]
                            for k, v in r[col].items()} if r[col] is not None else None)} for r in all_rows]
        resp = {"kggen": "kggen_responses"}.get(col, col.replace("_kg", "") + "_responses")
        systems[f"released:{col}"], realigned[col] = realign(src, col, resp)
        print(f"realign {col}: {realigned[col]}", flush=True)

    summary = {"judge": "gemma-4-E4B (llama-server, dspy ChainOfThought, temperature 0)",
               "retrieval": f"kg-gen 0.4.0 KGGen.retrieve, {RETRIEVAL_MODEL}, k=8, depth=2",
               "essays": len(queries), "realign": realigned, "systems": {}}
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
