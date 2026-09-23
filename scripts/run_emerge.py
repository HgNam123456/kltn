"""Chạy bộ phán Exists/Deprecate trên EMERGE và chấm Executable-R (kèm bootstrap CI 95%), đặt cạnh
baseline KG-aware kèm sẵn trên ĐÚNG tập instance đó.

Dùng: python scripts/run_emerge.py --per-delta 10 --batch-size 12 --out results/emerge_dev350_v1_batch.jsonl
      python scripts/run_emerge.py --batch-size 1  ...   (mỗi ứng viên một lệnh gọi có/không)
      python scripts/run_emerge.py --judge none          (chỉ tra KG, mọi ứng viên = Exists, không gọi LLM)
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm

from kgu.data.emerge import OPS, load_kg_subset, load_snapshot
from kgu.eval.emerge import bootstrap_ci, instance_scores, shipped_predictions
from kgu.judge.emerge import candidates, index_out_edges, judge, to_ops
from kgu.judge.emerge_add import KGShape, allowed_relations, index_relations, judge_add
from kgu.llm import client_from_env

SHIPPED = ["kg-aware/gpt-5.1/oracle", "kg-aware/gpt-5.1/kg_rag_32", "kg-aware/gpt-5.1/oracle_kg_rag"]
REPORT_OPS = ["x-triples", "d-triples", "e-triples"]


def _stat(values: list[float]) -> dict:
    lo, hi = bootstrap_ci(values)
    return {"mean": sum(values) / len(values) if values else 0.0, "ci95": [lo, hi]}


def summarize(rows: dict[str, tuple[float, float]]) -> dict:
    return {"n": len(rows), "recall": _stat([r for r, _ in rows.values()]),
            "precision": _stat([p for _, p in rows.values()])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data/raw/emerge"))
    ap.add_argument("--per-delta", type=int, default=10, help="số instance đầu mỗi file delta (100 = toàn bộ)")
    ap.add_argument("--offset", type=int, default=0, help="bỏ qua N instance đầu mỗi file delta (tập chưa nhìn)")
    ap.add_argument("--judge", choices=["llm", "none", "add"], default="llm",
                    help="llm: phán Exists/Deprecate; add: chỉ bộ phán Add; none: không gọi LLM")
    ap.add_argument("--batch-size", type=int, default=12, help="số ứng viên mỗi lệnh gọi; 1 = phán từng cạnh")
    ap.add_argument("--add", action="store_true", help="chạy thêm bộ phán Add (một lệnh gọi/instance)")
    ap.add_argument("--exists", choices=["all", "llm"], default="all")
    ap.add_argument("--no-add-norm", action="store_true",
                    help="tắt chuẩn hóa Add v2 (đảo chiều / sửa quan hệ / thêm chiều nghịch đảo theo cấu trúc KG)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path, default=Path("results/emerge_dev.jsonl"))
    args = ap.parse_args()

    llm = client_from_env() if args.judge != "none" else None
    if args.judge == "add":
        args.add = True
    jobs, raw = [], {}
    for kg_path in sorted((args.data_dir / "kg_subsets").glob("snapshot_*.tsv")):
        snapshot = kg_path.stem.removeprefix("snapshot_")
        kg, labels = load_kg_subset(kg_path)
        out_edges = index_out_edges(kg)
        rels_of = index_relations(kg) if args.add else {}
        shape = KGShape(kg) if args.add and not args.no_add_norm else None
        seen: dict[str, int] = {}
        for ex in load_snapshot(args.data_dir, snapshot):
            seen[ex.delta] = seen.get(ex.delta, 0) + 1
            if args.offset < seen[ex.delta] <= args.offset + args.per_delta:
                jobs.append((ex, candidates(ex, out_edges), {**labels, **ex.labels},
                             allowed_relations(ex, rels_of) if args.add else [], shape))
        keep = {ex.hash_id for ex, *_ in jobs}
        for path in sorted((args.data_dir / "evaluation_set" / f"snapshot_{snapshot}").glob("delta_*.jsonl")):
            with path.open(encoding="utf8") as f:
                for line in f:
                    d = json.loads(line)
                    if d["hash_id"] in keep:
                        raw[d["hash_id"]] = d

    def run(job):
        ex, cand, labels, relations, shape = job
        rec = {"hash_id": ex.hash_id, "snapshot": ex.snapshot, "delta": ex.delta, "title": ex.title}
        added: set = set()
        added_raw: set = set()
        try:
            verdicts = (judge(ex, cand, labels, llm, args.batch_size) if args.judge == "llm"
                        else {tr: "holds" for tr in cand})
            if args.add and llm:
                added_raw = judge_add(ex, cand, relations, labels, llm)
                added = shape.normalize(added_raw) if shape else added_raw
        except Exception as e:          # ghi lỗi theo instance, không dừng cả lượt chạy
            verdicts, rec["error"] = {}, repr(e)
        rec["verdicts"] = [[*tr, v] for tr, v in verdicts.items()]
        rec["added"] = sorted(added)
        rec["added_raw"] = sorted(added_raw)
        rec["gold"] = {op: sorted(ex.gold[op]) for op in REPORT_OPS}
        return ex.hash_id, {**to_ops(verdicts, args.exists), "e-triples": added}, rec

    start = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(tqdm(pool.map(run, jobs), total=len(jobs)))
    seconds = time.time() - start

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf8") as f:
        for _, _, rec in results:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    examples = [ex for ex, *_ in jobs]
    rows = {"mine": instance_scores(examples, {h: p for h, p, _ in results})}
    for m in SHIPPED:
        rows[m] = instance_scores(examples, {h: shipped_predictions(raw[h], m) for h in raw})
    summary = {
        "n": len(jobs), "judge": args.judge, "batch_size": args.batch_size, "exists": args.exists,
        "errors": sum("error" in r for _, _, r in results),
        "add": args.add, "add_norm": args.add and not args.no_add_norm,
        "with_candidates": sum(bool(c) for _, c, *_ in jobs), "candidates": sum(len(c) for _, c, *_ in jobs),
        "llm_calls": llm.n_calls if llm else 0, "seconds": round(seconds),
        "seconds_per_instance": round(seconds / len(jobs), 2),
    }
    for name, r in rows.items():
        summary[name] = {OPS[op]: summarize(r[op]) for op in REPORT_OPS}
    # Bootstrap ghép cặp: hiệu recall từng instance so với GPT-5.1 oracle. CI không chứa 0 = khác biệt thật.
    summary["mine_minus_oracle_recall"] = {
        OPS[op]: _stat([rows["mine"][op][h][0] - rows[SHIPPED[0]][op][h][0] for h in rows["mine"][op]])
        for op in REPORT_OPS
    }
    args.out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf8")

    print(f"\nn={summary['n']} có ứng viên={summary['with_candidates']} ứng viên={summary['candidates']} "
          f"llm_calls={summary['llm_calls']} errors={summary['errors']} {summary['seconds']}s "
          f"({summary['seconds_per_instance']} s/instance)")

    def cell(s):
        return f"{s['mean']:.3f} [{s['ci95'][0]:.3f}, {s['ci95'][1]:.3f}]"

    for op in REPORT_OPS:
        print(f"\n{OPS[op]} (n={summary['mine'][OPS[op]]['n']})   recall [CI95]            precision [CI95]")
        for name in rows:
            s = summary[name][OPS[op]]
            print(f"  {name:32s} {cell(s['recall']):26s} {cell(s['precision'])}")
        print(f"  {'mine − oracle (recall, ghép cặp)':32s} {cell(summary['mine_minus_oracle_recall'][OPS[op]])}")


if __name__ == "__main__":
    main()
