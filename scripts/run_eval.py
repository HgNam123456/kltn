"""Chạy pipeline trên 1 split với 1 cấu hình extractor/judge; ghi JSONL + summary."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from tqdm import tqdm

from kgu.data.nba import Vocab, load_split
from kgu.eval.metrics import Counts, summarize
from kgu.extract.gold import GoldAllExtractor, GoldExplicitExtractor
from kgu.judge.oracle import OracleJudge
from kgu.pipeline import run_example


def build_extractor(name: str, relations: list[str]):
    if name == "gold_all":
        return GoldAllExtractor()
    if name == "gold_explicit":
        return GoldExplicitExtractor()
    if name == "llm":
        from kgu.extract.llm import LLMExtractor
        from kgu.llm import client_from_env
        return LLMExtractor(client_from_env(), relations)
    raise SystemExit(f"unknown extractor {name}")


def build_llm_judge(name: str):
    """Judge LLM tạo MỘT lần (dùng chung client, đếm n_calls liên tục); oracle tạo per-example."""
    if name == "llm":
        from kgu.judge.llm import LLMJudge
        from kgu.llm import client_from_env
        return LLMJudge(client_from_env())
    if name in ("none", "oracle"):
        return None
    raise SystemExit(f"unknown judge {name}")


def run_one(ex, extractor, judge, relations: list[str]) -> dict:
    """Một ví dụ → record; nếu lỗi (LLM, mạng, tràn ctx...) → record lỗi để lượt chạy không abort."""
    try:
        return run_example(ex, extractor, judge, relations).to_record(ex)
    except Exception as e:  # noqa: BLE001 — cố ý: ghi lại mọi lỗi per-example
        return {"idx": ex.idx, "event": ex.event, "error": f"{type(e).__name__}: {e}"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data/raw/nba"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--extractor", default="gold_explicit", choices=["gold_all", "gold_explicit", "llm"])
    ap.add_argument("--judge", default="none", choices=["none", "oracle", "llm"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    relations = Vocab.load(args.data_dir).relations
    examples = load_split(args.data_dir, args.split, limit=args.limit)
    extractor = build_extractor(args.extractor, relations)
    llm_judge = build_llm_judge(args.judge)

    total = Counts()
    cov_hit = cov_all = n_cut = n_add = n_calls = n_err = 0
    t0 = time.time()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf8") as f:
        for ex in tqdm(examples, desc=f"{args.extractor}+{args.judge}"):
            judge = OracleJudge(ex.after) if args.judge == "oracle" else llm_judge
            rec = run_one(ex, extractor, judge, relations)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if "error" in rec:
                n_err += 1
                continue
            total = total + Counts(**rec["counts"])
            cov_hit += rec["coverage"][0]
            cov_all += rec["coverage"][1]
            n_cut += rec["n_cut_cand"]
            n_add += rec["n_add_cand"]
            n_calls += rec["n_llm_calls"]

    n = len(examples)
    s = summarize(total)
    s.update({
        "config": f"{args.extractor}+{args.judge}", "split": args.split, "n": n,
        "errors": n_err,
        "coverage": cov_hit / cov_all if cov_all else 0.0,
        "avg_cut_cand": n_cut / n if n else 0.0, "avg_add_cand": n_add / n if n else 0.0,
        "llm_calls": n_calls, "seconds": round(time.time() - t0, 1),
    })
    args.out.with_suffix(".summary.json").write_text(json.dumps(s, indent=2), encoding="utf8")
    print(json.dumps(s, indent=2))


if __name__ == "__main__":
    main()
