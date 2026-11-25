"""Chấm một lượt chạy run_eval trên ChainUpdate theo ĐỘ SÂU của thay đổi gián tiếp.

    python scripts/score_chainupdate.py --data data/chainupdate/chainupdate.jsonl RUN.jsonl [RUN2.jsonl ...]
                                        [--by family] [--first N] [--json OUT]

--first N chỉ chấm các mẫu có idx < N, để so nhiều lượt chạy trên cùng một tập mẫu khi có lượt bị dừng giữa chừng
(run_eval ghi record theo thứ tự idx); dòng cuối bị cắt dở của lượt bị dừng được bỏ qua.

Với mỗi lượt chạy, in:
  - theo độ sâu (1, 2, 3) và theo vế (thêm, cắt): số thay đổi gián tiếp vàng, số nằm trong vùng ứng viên (độ phủ),
    số được bước xét duyệt kết luận đúng (recall);
  - số thao tác kết luận sai và độ chính xác của từng vế, tách theo lần gọi của thuật toán khoanh vùng;
  - số mẫu có KG thu được trùng hoàn toàn với KG-sau vàng, và số lệnh gọi LLM theo lần gọi.
Độ sâu lấy từ nhãn của bộ dữ liệu (extra.indirect); lần gọi phủ lấy từ trường "implied" của record.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

SIDES = ("ADD", "INVALIDATE")
SIDE_VI = {"ADD": "thêm", "INVALIDATE": "cắt"}


def load_gold(path: Path) -> dict[int, dict]:
    """idx -> {"event", "domain", "family", "indirect": {(kind, h, r, t): độ sâu}}."""
    out: dict[int, dict] = {}
    with path.open(encoding="utf8") as f:
        for n, line in enumerate(f):
            if n == 0 or not line.strip():                       # dòng đầu là KG nền
                continue
            d = json.loads(line)
            ex = d["extra"]
            out[int(d["idx"])] = {"event": d["event"], "domain": ex["domain"], "family": ex["family"],
                                  "indirect": {tuple(op[:4]): int(op[4]) for op in ex["indirect"]}}
    return out


def net_ops(judged: list[dict]) -> dict[tuple, int]:
    """Thao tác gián tiếp có hiệu lực cuối cùng -> lần gọi đã kết luận. Một triple bị kết luận hai lần trái nhau thì bỏ."""
    seen: dict[tuple, list[tuple[str, int]]] = defaultdict(list)
    for o in judged:
        seen[(o["h"], o["r"], o["t"])].append((o["kind"], int(o.get("round", 1))))
    return {(kinds[0][0], *tr): kinds[0][1] for tr, kinds in seen.items() if len(kinds) == 1}


def load_records(path: Path, first: int | None = None) -> list[dict]:
    """Record của một lượt chạy; bỏ dòng cuối bị cắt dở (lượt bị dừng) và, nếu có `first`, các mẫu có idx >= first."""
    records = []
    with path.open(encoding="utf8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if first is None or rec["idx"] < first:
                records.append(rec)
    return records


def merge_plus(records: list[dict], extra: list[dict]) -> list[dict]:
    """Ghép một lượt chạy bổ sung vào một lượt chạy đã có, theo idx: thao tác gián tiếp và số lệnh gọi LLM của lượt bổ
    sung được cộng vào. Dùng cho lượt chỉ xét các nhóm mà bước lọc theo cấu trúc đã giữ nguyên (--judge-gate-sides
    cut_rejected, K = 1): mỗi nhóm là một lệnh gọi riêng nên ghép với lượt có lọc cho đúng kết quả của một lượt không lọc
    ở vế cắt. Chỉ giữ các mẫu có ở cả hai lượt; mẫu đã ghép được đánh dấu để tính "trùng hoàn toàn" lại từ nhãn vàng."""
    by_idx = {r["idx"]: r for r in extra if "error" not in r}
    out = []
    for rec in records:
        ex = by_idx.get(rec["idx"])
        if ex is None or "error" in rec:
            continue
        rounds = [dict(rd) for rd in rec.get("rounds", [])]
        for rd in ex.get("rounds", []):
            same = next((r for r in rounds if r["k"] == rd["k"]), None)
            if same is None:
                rounds.append(dict(rd))
            else:
                same["n_llm_calls"] += rd["n_llm_calls"]
        out.append({**rec, "judged_ops": rec["judged_ops"] + ex["judged_ops"], "rounds": rounds, "_merged": True})
    return out


def upto_round(records: list[dict], n: int) -> list[dict]:
    """Kết quả của K = n lấy từ một lượt chạy K lớn hơn: chỉ giữ thao tác, độ phủ và lệnh gọi của các lần gọi 1..n.
    Đúng vì lần gọi k chỉ phụ thuộc vào các lần gọi trước nó."""
    out = []
    for rec in records:
        if "error" in rec:
            out.append(rec)
            continue
        out.append({**rec,
                    "judged_ops": [o for o in rec["judged_ops"] if int(o.get("round", 1)) <= n],
                    "implied": [o for o in rec.get("implied", []) if 0 < int(o["cov"]) <= n],
                    "rounds": [rd for rd in rec.get("rounds", []) if rd["k"] <= n], "_merged": True})
    return out


def _cell() -> dict:
    return {"gold": 0, "covered": 0, "correct": 0}


def score_run(gold: dict[int, dict], records: list[dict], by: str | None = None) -> dict:
    depth: dict[tuple[str, int], dict] = defaultdict(_cell)                 # (vế, độ sâu)
    cov_call: dict[tuple[str, int, int], int] = defaultdict(int)            # (vế, độ sâu, lần gọi phủ)
    wrong: dict[tuple[str, int], int] = defaultdict(int)                    # (vế, lần gọi)
    right_call: dict[tuple[str, int], int] = defaultdict(int)
    groups: dict[str, dict] = defaultdict(lambda: {"n": 0, "exact": 0, "gold": 0, "covered": 0, "correct": 0, "wrong": 0})
    n = exact = errors = 0
    calls: dict[int, int] = defaultdict(int)
    for rec in records:
        if "error" in rec:
            errors += 1
            continue
        g = gold[rec["idx"]]
        n += 1
        pred = net_ops(rec["judged_ops"])
        if rec.get("_merged"):          # lượt đã ghép: KG thu được trùng KG-sau vàng khi tập thao tác gián tiếp trùng nhãn
            is_exact = set(pred) == set(g["indirect"])
        else:
            is_exact = rec["counts"]["fp"] == 0 and rec["counts"]["fn"] == 0
        exact += is_exact
        for rd in rec.get("rounds", []):
            calls[rd["k"]] += rd["n_llm_calls"]
        cov = {(o["kind"], o["h"], o["r"], o["t"]): int(o["cov"]) for o in rec.get("implied", [])}
        grp = groups[g[by]] if by else None
        if grp is not None:
            grp["n"] += 1
            grp["exact"] += is_exact
        for op, d in g["indirect"].items():
            cell = depth[(op[0], d)]
            cell["gold"] += 1
            k = cov.get(op, 0)
            cell["covered"] += k > 0
            if k:
                cov_call[(op[0], d, k)] += 1
            hit = op in pred
            cell["correct"] += hit
            if hit:
                right_call[(op[0], pred[op])] += 1
            if grp is not None:
                grp["gold"] += 1
                grp["covered"] += k > 0
                grp["correct"] += hit
        for op, k in pred.items():
            if op not in g["indirect"]:
                wrong[(op[0], k)] += 1
                if grp is not None:
                    grp["wrong"] += 1

    def ratio(a: int, b: int) -> float | None:
        return round(a / b, 4) if b else None

    table = []
    for side in SIDES:
        for d in sorted({d for s, d in depth if s == side}):
            c = depth[(side, d)]
            table.append({"side": side, "depth": d, **c, "coverage": ratio(c["covered"], c["gold"]),
                          "recall": ratio(c["correct"], c["gold"]),
                          "covered_at_call": {k: v for (s, dd, k), v in sorted(cov_call.items()) if s == side and dd == d}})
    sides = {}
    for side in SIDES:
        tot = {key: sum(depth[(s, d)][key] for s, d in depth if s == side) for key in ("gold", "covered", "correct")}
        n_wrong = sum(v for (s, _), v in wrong.items() if s == side)
        sides[side] = {**tot, "wrong": n_wrong, "coverage": ratio(tot["covered"], tot["gold"]),
                       "recall": ratio(tot["correct"], tot["gold"]),
                       "precision": ratio(tot["correct"], tot["correct"] + n_wrong),
                       "correct_by_call": {k: v for (s, k), v in sorted(right_call.items()) if s == side},
                       "wrong_by_call": {k: v for (s, k), v in sorted(wrong.items()) if s == side}}
    return {"n": n, "errors": errors, "exact": exact, "exact_rate": ratio(exact, n), "by_depth": table, "by_side": sides,
            "llm_calls": sum(calls.values()), "llm_calls_by_call": dict(sorted(calls.items())),
            "groups": {k: {**v, "coverage": ratio(v["covered"], v["gold"]), "recall": ratio(v["correct"], v["gold"]),
                           "precision": ratio(v["correct"], v["correct"] + v["wrong"])}
                       for k, v in sorted(groups.items())}}


def _pct(x: float | None) -> str:
    return "   -  " if x is None else f"{x * 100:5.1f}%"


def render(name: str, s: dict) -> str:
    lines = [f"== {name}: {s['n']} mẫu, {s['exact']} mẫu trùng hoàn toàn KG-sau vàng ({_pct(s['exact_rate']).strip()}), "
             f"{s['llm_calls']} lệnh gọi LLM {s['llm_calls_by_call'] or ''}",
             f"{'vế':<5}{'độ sâu':>7}{'vàng':>8}{'phủ':>8}{'độ phủ':>9}{'đúng':>8}{'recall':>9}   phủ lần đầu ở lần gọi"]
    for row in s["by_depth"]:
        lines.append(f"{SIDE_VI[row['side']]:<5}{row['depth']:>7}{row['gold']:>8}{row['covered']:>8}{_pct(row['coverage']):>9}"
                     f"{row['correct']:>8}{_pct(row['recall']):>9}   {row['covered_at_call']}")
    for side, t in s["by_side"].items():
        lines.append(f"{SIDE_VI[side]:<5}{'mọi':>7}{t['gold']:>8}{t['covered']:>8}{_pct(t['coverage']):>9}{t['correct']:>8}"
                     f"{_pct(t['recall']):>9}   sai {t['wrong']} (theo lần gọi {t['wrong_by_call']}), độ chính xác {_pct(t['precision']).strip()}")
    if s["groups"]:
        lines.append(f"{'nhóm':<16}{'mẫu':>5}{'trùng':>7}{'vàng':>8}{'độ phủ':>9}{'recall':>9}{'sai':>7}{'ch.xác':>9}")
        for k, v in s["groups"].items():
            lines.append(f"{k:<16}{v['n']:>5}{v['exact']:>7}{v['gold']:>8}{_pct(v['coverage']):>9}{_pct(v['recall']):>9}"
                         f"{v['wrong']:>7}{_pct(v['precision']):>9}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path, help="tệp JSONL do scripts/run_eval.py ghi")
    ap.add_argument("--data", type=Path, default=Path("data/chainupdate/chainupdate.jsonl"))
    ap.add_argument("--by", choices=["family", "domain", "event"], default=None, help="thêm bảng theo nhóm sự kiện")
    ap.add_argument("--first", type=int, default=None, help="chỉ chấm các mẫu có idx < N")
    ap.add_argument("--json", type=Path, default=None, help="ghi kết quả của mọi lượt chạy vào một tệp JSON")
    ap.add_argument("--write-seed", type=Path, default=None,
                    help="ghi thao tác lần gọi 1 của lượt chạy đầu tiên (sau khi ghép --plus) ra tệp, dùng cho run_eval --seed-call1")
    ap.add_argument("--max-round", type=int, default=None,
                    help="chỉ tính các lần gọi 1..N của mỗi lượt chạy (kết quả của K = N lấy từ lượt K lớn hơn)")
    ap.add_argument("--plus", type=Path, default=None,
                    help="lượt chạy bổ sung (vd lượt cut_rejected) được ghép vào từng lượt chạy theo idx trước khi chấm")
    args = ap.parse_args()

    gold = load_gold(args.data)
    extra = load_records(args.plus, args.first) if args.plus else None
    out = {}
    for run in args.runs:
        records = load_records(run, args.first)
        name = run.stem + (f" [idx<{args.first}]" if args.first is not None else "")
        if extra is not None:
            records = merge_plus(records, extra)
            name += f" + {args.plus.stem}"
        if args.write_seed is not None and run == args.runs[0]:
            with args.write_seed.open("w", encoding="utf8", newline="\n") as f:
                for rec in records:
                    if "error" not in rec:
                        ops = [o for o in rec["judged_ops"] if int(o.get("round", 1)) == 1]
                        f.write(json.dumps({"idx": rec["idx"], "judged_ops": ops}, ensure_ascii=False) + "\n")
        if args.max_round is not None:
            records = upto_round(records, args.max_round)
            name += f" [lần gọi <= {args.max_round}]"
        out[name] = score_run(gold, records, by=args.by)
        print(render(name, out[name]), end="\n\n")
    if args.json:
        args.json.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf8")


if __name__ == "__main__":
    main()
