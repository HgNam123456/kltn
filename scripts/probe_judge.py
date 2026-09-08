"""Chấm bộ phán LLM ở MỨC MẪU: mỗi mẫu cắt/thêm có đáp án vàng suy từ KG-sau. In confusion theo loại mẫu.

Dùng để thử nhanh biến thể prompt/schema mà không cần chạy full pipeline:
    python scripts/probe_judge.py --limit 20 --workers 4
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

from kgu.data.nba import Vocab, load_split
from kgu.extract.gold import GoldExplicitExtractor
from kgu.graph import BiTemporalGraph
from kgu.judge import JudgeContext
from kgu.judge.llm import LLMJudge, group_add, group_cut
from kgu.llm import client_from_env
from kgu.localize import localize
from kgu.ops import apply_ops
from kgu.pipeline import AT_AFTER, AT_BEFORE


def gold_cut(pattern, after: set) -> bool:
    """Mẫu cắt là 'cần vô hiệu' nếu đa số cạnh thành viên không còn trong KG-sau."""
    gone = sum(1 for c in pattern.members if c.triple not in after)
    return gone * 2 > len(pattern.members)


def gold_add(pattern, after: set) -> set[tuple[str, str]]:
    """Tập (r, direction) mà đa số thành viên có trong KG-sau."""
    out = set()
    rels = {t[1] for t in after}
    for r in rels:
        for direction in ("subject_first", "neighbor_first"):
            hit = 0
            for c in pattern.members:
                h, t = (c.subject, c.neighbor) if direction == "subject_first" else (c.neighbor, c.subject)
                hit += (h, r, t) in after
            if hit * 2 > len(pattern.members):
                out.add((r, direction))
    return out


def probe_one(ex, relations, judge_factory):
    g = BiTemporalGraph.from_triples(ex.before, valid_from=AT_BEFORE)
    explicit = apply_ops(g, GoldExplicitExtractor().extract(ex, g, AT_BEFORE), at=AT_AFTER)
    loc = localize(g, explicit, at=AT_AFTER)
    ctx = JudgeContext(ex.text, explicit, loc, g, AT_AFTER, relations)
    cut, add = group_cut(loc), group_add(loc)
    judge = judge_factory()
    rows = []
    for batch in judge._batches(cut):
        out = judge.client.complete_json(judge.SYSTEM_CUT, judge._prompt_cut(ctx, batch), judge._cut_schema)
        pred = set(out.invalidate)
        for i, p in enumerate(batch):
            rows.append(("cut", ex.event, f"{p.r} {p.direction}", gold_cut(p, ex.after), i in pred))
    for batch in judge._batches(add):
        out = judge.client.complete_json(judge.SYSTEM_ADD, judge._prompt_add(ctx, batch), judge._add_schema(relations))
        pred = {}
        for v in out.add:
            pred.setdefault(v.idx, set()).add((v.r, v.direction))
        for i, p in enumerate(batch):
            g_ = gold_add(p, ex.after)
            rows.append(("add", ex.event, f"via {p.via_r} {p.via_direction}", bool(g_), bool(pred.get(i)),
                         g_ == pred.get(i, set())))
    return ex.idx, rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data/raw/nba"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--batch", type=int, default=1, help="mẫu / lệnh gọi; 0 = cả pha một lệnh gọi")
    args = ap.parse_args()
    relations = Vocab.load(args.data_dir).relations
    examples = load_split(args.data_dir, args.split, limit=args.limit)

    def factory():
        j = LLMJudge(client_from_env(), batch_size=args.batch or None)
        from kgu.judge import llm as m
        j.SYSTEM_CUT, j.SYSTEM_ADD = m.SYSTEM_CUT, m.SYSTEM_ADD
        return j

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(lambda ex: probe_one(ex, relations, factory), examples))

    conf: dict[str, Counter] = {}
    exact = Counter()
    for idx, rows in sorted(results):
        for row in rows:
            kind, event, key, gold, pred = row[:5]
            conf.setdefault(f"{kind:<4} {key}", Counter())[(gold, pred)] += 1
            if kind == "add":
                exact[(gold, row[5])] += 1
    print(f"{'mẫu':<36} {'TP':>4} {'FN':>4} {'FP':>4} {'TN':>4}")
    for key, c in sorted(conf.items()):
        print(f"{key:<36} {c[(True, True)]:>4} {c[(True, False)]:>4} {c[(False, True)]:>4} {c[(False, False)]:>4}")
    print("add: đúng cả (r, direction) khi gold có:", exact[(True, True)], "/", exact[(True, True)] + exact[(True, False)])


if __name__ == "__main__":
    main()
