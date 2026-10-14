"""Đổi 2020-COVID-NYT (benchmark của ATOM, Lairgi et al., EACL 2026) từ pickle phát hành sang JSONL — chạy MỘT LẦN.

Nguồn: HF lairgiyassir/2020-COVID-NYT (giấy phép MIT), trùng byte với
datasets/atom/nyt_news/2020_nyt_COVID_last_version_ready.pkl của AuvaLab/itext2kg. Pickle có thể chạy code tùy ý khi
nạp, nên chỉ nạp ở đây (file tải từ đúng nguồn trên); mọi script khác và kernel Kaggle chỉ đọc JSONL đã commit.

Mỗi dòng = 1 nhóm ngày: {"row", "date" (YYYY-MM-DD), "text" (các lead paragraph của ngày đó), "gold" [[h,r,t,ts,te]],
"calib" {cột quintuples_gpt41*: [[...]]}} — 6 cột GPT-4.1 có sẵn dùng để hiệu chỉnh bộ chấm (scripts/score_covid_nyt.py).

Dùng: python -I scripts/convert_covid_nyt.py --pkl <2020-COVID-NYT.pkl> --out data/covid_nyt/2020_covid_nyt.jsonl
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

CALIB_COLS = ["quintuples_gpt41", "quintuples_gpt41_run2", "quintuples_gpt41_run3", "quintuples_gpt41_from_factoids",
              "quintuples_gpt41_from_factoids_run2", "quintuples_gpt41_from_factoids_run3"]


def plain(v):
    """tuple/list/numpy/Timestamp → JSON thuần (giữ nguyên chuỗi ngày như dataset ghi)."""
    if v is None:
        return None
    if isinstance(v, (list, tuple)) or type(v).__name__ == "ndarray":
        return [plain(x) for x in v]
    if isinstance(v, (dt.date, dt.datetime)) or type(v).__name__ == "Timestamp":
        return v.strftime("%Y-%m-%d")
    if isinstance(v, float) and v != v:                                      # NaN
        return None
    if type(v).__module__ == "numpy":
        return v.item()
    return v


def quints(v) -> list:
    return [plain(q) for q in v] if isinstance(v, (list, tuple)) or type(v).__name__ == "ndarray" else []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    import pandas as pd
    df = pd.read_pickle(args.pkl)
    print("cột:", list(df.columns), "| số dòng:", len(df))
    text_col = next(c for c in ("lead_paragraph_observation_date", "lead_paragraph") if c in df.columns)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_gold = 0
    with args.out.open("w", encoding="utf8") as f:
        for i, (_, r) in enumerate(df.iterrows()):
            text = r[text_col]
            rec = {"row": i, "date": plain(r["date"]),
                   "text": "\n".join(map(str, text)) if isinstance(text, (list, tuple)) else str(text),
                   "gold": quints(r["quintuples_g_truth"]),
                   "calib": {c: quints(r[c]) for c in CALIB_COLS if c in df.columns}}
            n_gold += len(rec["gold"])
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"ghi {len(df)} nhóm ngày, {n_gold} gold → {args.out}")


if __name__ == "__main__":
    main()
