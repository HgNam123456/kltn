"""Chạy pipeline trên 1 split với 1 cấu hình extractor/judge; ghi JSONL + summary."""
from __future__ import annotations

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
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


def build_llm_judge(name: str, batch: int | None = 1, gate: float | None = None, naive_max_triples: int = 80,
                    gate_sides: str = "both"):
    """Judge LLM tạo MỘT lần (dùng chung client, đếm n_calls liên tục); oracle tạo per-example."""
    if name == "llm":
        from kgu.judge.llm import LLMJudge
        from kgu.llm import client_from_env
        return LLMJudge(client_from_env(), batch_size=batch, gate=gate, gate_sides=gate_sides)
    if name == "naive":                                # hệ so sánh: LLM tự quyết trên triple 1 bước, không khoanh vùng
        from kgu.judge.naive import NaiveOneHopJudge
        from kgu.llm import client_from_env
        return NaiveOneHopJudge(client_from_env(), max_triples=naive_max_triples)
    if name in ("none", "oracle"):
        return None
    raise SystemExit(f"unknown judge {name}")


def effective_max_calls(judge_name: str, max_calls: int) -> int:
    """K thực dùng: hệ so sánh 1 bước (--judge naive) không dùng khoanh vùng nên luôn chỉ một lần gọi."""
    return 1 if judge_name == "naive" else max(1, max_calls)


def run_one(ex, extractor, judge, relations: list[str], graph=None, max_calls: int = 1) -> dict:
    """Một ví dụ → record; nếu lỗi (LLM, mạng, tràn ctx...) → record lỗi để lượt chạy không abort."""
    try:
        return run_example(ex, extractor, judge, relations, graph=graph, max_calls=max_calls).to_record(ex)
    except Exception as e:  # noqa: BLE001 — cố ý: ghi lại mọi lỗi per-example
        return {"idx": ex.idx, "event": ex.event, "error": f"{type(e).__name__}: {e}"}


def summarize_run(records: list[dict], seconds: float, config: str, split: str,
                  max_calls: int | None = None) -> dict:
    """Gộp danh sách record (kể cả record lỗi) thành summary.

    `n` đếm MỌI record (kể cả lỗi); `n_scored` = n - errors chỉ đếm record có "counts". Các trung bình
    (avg_cut_cand, avg_add_cand) và coverage chia cho n_scored, không phải n — record lỗi không có
    n_cut_cand/n_add_cand nên chia cho n sẽ làm trung bình bị pha loãng sai khi có lỗi.

    Khoanh vùng đệ quy: `max_calls` = K (None → lấy từ record, mặc định 1); `llm_calls_by_call[k-1]` = số lệnh gọi LLM
    của bước xét duyệt ở lần gọi k; `coverage_by_call[k-1]` = độ phủ CỘNG DỒN sau lần gọi k (phần thay đổi gián tiếp
    vàng đã nằm trong vùng ứng viên của một lần gọi <= k). Record kiểu cũ (không có "rounds"/"implied") được tính vào
    lần gọi 1.
    """
    k_max = max(1, max_calls if max_calls is not None else max((r.get("max_calls", 1) for r in records), default=1))
    calls_by_call = [0] * k_max
    cov_by_call = [0] * k_max                           # số thay đổi gián tiếp được phủ LẦN ĐẦU ở lần gọi k
    total = Counts()
    cov_hit = cov_all = n_cut = n_add = n_judged = n_calls = n_err = 0
    for rec in records:
        if "error" in rec:
            n_err += 1
            continue
        total = total + Counts(**rec["counts"])
        cov_hit += rec["coverage"][0]
        cov_all += rec["coverage"][1]
        n_cut += rec["n_cut_cand"]
        n_add += rec["n_add_cand"]
        n_judged += rec["n_judged"]
        n_calls += rec["n_llm_calls"]
        if "rounds" in rec:
            for rd in rec["rounds"]:
                if 1 <= rd["k"] <= k_max:
                    calls_by_call[rd["k"] - 1] += rd["n_llm_calls"]
        else:
            calls_by_call[0] += rec["n_llm_calls"]
        if "implied" in rec:
            for op in rec["implied"]:
                if 1 <= op["cov"] <= k_max:
                    cov_by_call[op["cov"] - 1] += 1
        else:
            cov_by_call[0] += rec["coverage"][0]

    n = len(records)
    n_scored = n - n_err
    n_cands = n_cut + n_add
    s = summarize(total)
    s.update({
        "config": config, "split": split, "n": n, "n_scored": n_scored,
        "errors": n_err,
        "coverage": cov_hit / cov_all if cov_all else 0.0,
        "avg_cut_cand": n_cut / n_scored if n_scored else 0.0,
        "avg_add_cand": n_add / n_scored if n_scored else 0.0,
        "judged_ratio": n_judged / n_cands if n_cands else 0.0,
        "llm_calls": n_calls, "seconds": round(seconds, 1),
        "max_calls": k_max,
        "llm_calls_by_call": calls_by_call,
        "coverage_by_call": [sum(cov_by_call[:k + 1]) / cov_all if cov_all else 0.0 for k in range(k_max)],
    })
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data/raw/nba"))
    ap.add_argument("--jsonl", type=Path, default=None,
                    help="bộ dữ liệu JSONL chung (kgu.data.jsonl: ICEWS14-Event, YAGO-Event...) thay cho NBA")
    ap.add_argument("--split", default="test")
    ap.add_argument("--extractor", default="gold_explicit", choices=["gold_all", "gold_explicit", "llm"])
    ap.add_argument("--judge", default="none", choices=["none", "oracle", "llm", "naive"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--offset", type=int, default=0,
                    help="bỏ qua N mẫu đầu (sau --limit), để chia một bộ có các mẫu độc lập cho nhiều kernel chạy song song")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=1,
                    help="số tin gửi song song (khớp --parallel của llama-server để dùng continuous batching)")
    ap.add_argument("--judge-batch", type=int, default=1, help="mẫu / lệnh gọi LLM judge; 0 = cả pha một lệnh gọi")
    ap.add_argument("--judge-gate", type=float, default=None,
                    help="phương án C: ngưỡng đồng xuất hiện để cổng cấu trúc lọc mẫu trước LLM (vd 0.5); bỏ = LLM thuần")
    ap.add_argument("--judge-gate-sides", choices=["both", "cut", "add", "cut_rejected"], default="both",
                    help="vế có bước lọc theo cấu trúc: both (mặc định); add = chỉ lọc vế sinh, mọi nhóm của vế cắt tới LLM; "
                         "cut = chỉ lọc vế cắt; cut_rejected = CHỈ gửi LLM các nhóm của vế cắt mà bước lọc giữ nguyên "
                         "(không xét gì khác), để ghép với một lượt có lọc đã chạy")
    ap.add_argument("--max-calls", type=int, default=1,
                    help="K của Thuật toán khoanh vùng đệ quy (Chương 3): số lần gọi tối đa; lần gọi k khoanh vùng quanh "
                         "các thay đổi vừa ghi ở lần k-1. Mặc định 1 = một lần gọi (các số đã báo cáo); --judge naive luôn 1")
    ap.add_argument("--naive-max-triples", type=int, default=80,
                    help="--judge naive: số triple 1 bước tối đa đưa vào prompt (vượt → lấy mẫu đều theo thực thể tham gia)")
    args = ap.parse_args()
    max_calls = effective_max_calls(args.judge, args.max_calls)

    if args.jsonl is not None:
        from kgu.data.jsonl import load_jsonl
        examples, relations, _ = load_jsonl(args.jsonl, limit=args.limit)
        args.split = args.jsonl.stem
    else:
        relations = Vocab.load(args.data_dir).relations
        examples = load_split(args.data_dir, args.split, limit=args.limit)

    examples = examples[args.offset:]

    # Mỗi thread một extractor/judge riêng (client riêng): n_calls/n_dropped cộng dồn theo thread, nên
    # hiệu trước/sau trong run_example vẫn đúng cho từng ví dụ khi chạy song song.
    local = threading.local()
    judges: list = []                                  # mọi judge LLM đã tạo (mỗi thread một) → gộp bộ đếm vào summary

    def components():
        if not hasattr(local, "extractor"):
            local.extractor = build_extractor(args.extractor, relations)
            local.judge = build_llm_judge(args.judge, batch=args.judge_batch or None, gate=args.judge_gate,
                                          naive_max_triples=args.naive_max_triples, gate_sides=args.judge_gate_sides)
            if local.judge is not None:
                judges.append(local.judge)
        return local.extractor, local.judge

    # KG cho trước (GivenKGDataset): dựng đồ thị nền MỘT lần mỗi thread, mỗi bài áp chênh lệch rồi rollback —
    # tránh dựng lại đồ thị ~1M cạnh cho từng bài. Kết quả đồng nhất với dựng mới từ ex.before.
    from kgu.data.jsonl import GivenKGDataset
    given = examples if isinstance(examples, GivenKGDataset) else None

    def base_graph():
        if given is not None and not hasattr(local, "graph"):
            from kgu.graph import BiTemporalGraph
            local.graph = BiTemporalGraph.from_triples(given.base, valid_from=0)
        return getattr(local, "graph", None)

    def work(i: int) -> dict:
        ex = examples[i]
        extractor, llm_judge = components()
        judge = OracleJudge(ex.after) if args.judge == "oracle" else llm_judge
        g = base_graph()
        if g is None:
            return run_one(ex, extractor, judge, relations, max_calls=max_calls)
        cp = g.checkpoint()
        try:
            given.apply_before(g, i)
            return run_one(ex, extractor, judge, relations, graph=g, max_calls=max_calls)
        finally:
            g.rollback(cp)

    records: list[dict] = []
    t0 = time.time()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf8") as f, ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for rec in tqdm(pool.map(work, range(len(examples))), total=len(examples), desc=f"{args.extractor}+{args.judge}"):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            records.append(rec)

    tag = f"{args.extractor}+{args.judge}" + (f"[batch={args.judge_batch},gate={args.judge_gate}]" if args.judge == "llm" else "")
    if args.judge == "llm" and args.judge_gate_sides != "both":   # tag cũ giữ nguyên khi lọc cả hai vế
        tag += f"[gate_sides={args.judge_gate_sides}]"
    if args.judge == "naive":
        tag = f"{args.extractor}+naive1hop[max={args.naive_max_triples}]"
    if max_calls != 1:                                          # tag cũ giữ nguyên khi K = 1
        tag += f"[K={max_calls}]"
    s = summarize_run(records, time.time() - t0, tag, args.split, max_calls=max_calls)
    if judges:
        s["n_gated"] = sum(j.n_gated for j in judges)           # mẫu bị cổng cấu trúc chặn, không hỏi LLM
        s["n_off_menu"] = sum(j.n_off_menu for j in judges)     # mục add LLM trả về ngoài "lựa chọn hợp lệ"
    if judges and args.judge == "naive":                        # bộ đếm riêng của hệ so sánh 1 bước
        n_items = sum(j.n_items for j in judges)
        s["naive_dropped"] = sum(j.n_dropped for j in judges)   # mục LLM trả về bị hậu kiểm loại
        s["naive_drop_reasons"] = dict(sum((j.drop_reasons for j in judges), start=type(judges[0].drop_reasons)()))
        s["naive_errors"] = sum(j.n_errors for j in judges)     # lệnh gọi lỗi → bản tin không có thay đổi gián tiếp
        s["naive_avg_listed"] = sum(j.n_listed for j in judges) / n_items if n_items else 0.0
        s["naive_truncated"] = sum(j.n_truncated for j in judges)   # bản tin có nhiều triple 1 bước hơn mức trần
    args.out.with_suffix(".summary.json").write_text(json.dumps(s, indent=2), encoding="utf8")
    print(json.dumps(s, indent=2))


if __name__ == "__main__":
    main()
