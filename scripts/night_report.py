"""Gộp đầu ra các kernel chạy qua đêm (covid_repeat r2/r3, làn night lane1/lane2, knet_baseline, MINE-1 v3) thành một
REPORT.md tiếng Việt: trạng thái kernel, độ ổn định COVID-NYT, độ chính xác theo văn bản nguồn, độ gọn, NYT11-HRL,
KnowledgeNet fold 4 và bảng P/R/F1 gộp. Chỉ đọc JSON, không tính nặng; thiếu file thì ghi "chưa có", mỗi mục tự bắt
lỗi; luôn thoát với mã 0. Đầu ra Kaggle nằm dưới thư mục con out/ (kaggle kernels output giữ đường dẫn tương đối).

Dùng: python scripts/night_report.py --root results/kaggle --out results/kaggle/night/REPORT.md
"""
from __future__ import annotations

import argparse
import json
import statistics
import traceback
from pathlib import Path
from typing import Callable

LANES = {"lane1": ["nyt11", "knet_atom"], "lane2": ["quality", "knet_main"]}
EMBEDDER, THRESHOLD = "sentence-transformers/all-mpnet-base-v2", "0.7"
PAPER_STABILITY = {"ATOM (GPT-4.1)": (0.552, 0.534), "trích thẳng (GPT-4.1)": (0.212, 0.218)}
README_DEV = {"P": 0.718, "R": 0.691, "F1": 0.704}
RAW_OIE = {"atom", "kggen"}
MINE_ACC = {"mine/mine": "mine", "mine/kggen_gemma": "kggen-gemma", "mine/released_kggen": "released:kggen",
            "mine/released_graphrag": "released:graphrag_kg", "mine/released_openie": "released:openie_kg"}
NA = "—"


def load(*candidates: Path) -> dict | None:
    for p in candidates:
        try:
            if p.is_file():
                return json.loads(p.read_text(encoding="utf8"))
        except Exception:
            continue
    return None


def find(base: Path, name: str) -> dict | None:
    return load(base / "out" / name, base / name)


def pct(x, nd: int = 1) -> str:
    return f"{100 * x:.{nd}f}%" if isinstance(x, (int, float)) else NA


def num(x, nd: int = 2) -> str:
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return NA if x is None else str(x)
    return str(x) if isinstance(x, int) else f"{x:.{nd}f}"


def table(header: list[str], rows: list[list]) -> str:
    if not rows:
        return "chưa có\n"
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines) + "\n"


def get(d, *keys):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def prf(d: dict | None) -> list[str]:
    return [pct(get(d, k)) for k in ("P", "R", "F1")]


class Data:
    """Đọc một lần mọi nguồn; các mục báo cáo chỉ tra cứu trong đây."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.repeat = {r: root / "covid_repeat" / r for r in ("r2", "r3")}
        self.lane_json: dict[str, dict | None] = {}
        self.stage_dir: dict[str, Path] = {}
        for lane, stages in LANES.items():
            base = root / "night" / lane
            self.lane_json[lane] = find(base, "lane.json")
            for st in stages:
                self.stage_dir[st] = base / "out" / st
        self.stage: dict[str, dict] = {st: self._stage(d) for st, d in self.stage_dir.items()}
        self.baseline = find(root / "knet_baseline", "summary.json")
        self.mine_tables = load(root / "mine" / "v3" / "tables.json")

    @staticmethod
    def _stage(d: Path) -> dict:
        """summary.json của runner (ưu tiên) hoặc từng file JSON rời trong thư mục bộ dữ liệu / thư mục việc."""
        summ = load(d / "summary.json") or {}
        out = dict(summ)
        sub = [p for p in (d / "nyt11", d / "knet") if p.is_dir()]
        for key in ("scores_exact", "scores_knet", "support", "redundancy"):
            if key not in out:
                for base in [d, *sub]:
                    fname = "support.json" if key == "support" else f"{key}.json"
                    v = load(base / fname)
                    if v is not None:
                        out[key] = v
                        break
        return out

    def all_of(self, key: str) -> dict:
        """Gộp khóa dict (support, redundancy) của mọi việc: tên hệ (dạng bộ/hệ) → số đo."""
        merged: dict = {}
        for st in ("quality", "nyt11", "knet_atom", "knet_main"):
            v = self.stage[st].get(key)
            if isinstance(v, dict):
                merged.update(v)
        return merged


def sec_status(d: Data) -> str:
    rows = []
    for lane, stages in LANES.items():
        lj = d.lane_json[lane]
        by = {s.get("name"): s for s in (lj or {}).get("stages", [])}
        for st in stages:
            s = by.get(st)
            if s:
                rows.append([f"{lane}/{st}", s.get("status", NA), num(s.get("minutes"), 1), str(s.get("commit", NA))[:7]])
            else:
                rows.append([f"{lane}/{st}", "chưa có lane.json" if lj is None else "không có trong lane.json", NA, NA])
    for r, base in d.repeat.items():
        has = [n for n in ("scores_covid.json", "stability.json", "summary.json") if find(base, n) is not None]
        s = find(base, "summary.json") or {}
        rows.append([f"covid_repeat/{r}", ("có " + ", ".join(has)) if has else "chưa có", num(s.get("total_minutes"), 1),
                     str(s.get("kltn_commit", NA))[:7]])
    b = d.baseline
    rows.append(["knet_baseline", NA if b is None else ("chạy lại được" if b.get("rerun") else "không chạy lại được")
                 + (f" (ok={b.get('ok')})" if "ok" in b else ""), num((b or {}).get("total_minutes"), 1), NA])
    return table(["kernel / việc", "trạng thái", "phút", "commit"], rows)


def sec_stability(d: Data) -> str:
    stab: dict[str, dict] = {}
    for base in d.repeat.values():
        s = find(base, "stability.json")
        for name, pairs in (s or {}).get("systems", {}).items():
            stab.setdefault(name, {}).update(pairs)
    rows = []
    for name, pairs in stab.items():
        for pair, v in sorted(pairs.items()):
            cell = lambda k: (f"{pct(get(v, k, 'mean'))} ± {pct(get(v, k, 'std'))}" if get(v, k, "mean") is not None else NA)  # noqa: E731
            rows.append([name, pair, cell("jaccard"), cell("jaccard_triple"), cell("jaccard_soft"),
                         num(get(v, "jaccard", "groups"))])
    for name, (a, b) in PAPER_STABILITY.items():
        rows.append([f"{name} — số paper", "r1-r2", pct(a), NA, NA, NA])
        rows.append([f"{name} — số paper", "r1-r3", pct(b), NA, NA, NA])
    out = "Jaccard theo nhóm ngày (trung bình ± std), lượt 1 so với lượt r. Hàng \"số paper\" là Bảng 1 của ATOM " \
          "(arXiv 2510.22590v2); hàng released:* là ba lượt GPT-4.1 của tác giả chấm bằng cùng mã.\n\n"
    out += table(["hệ", "cặp lượt", "Jaccard 5-tuple", "Jaccard triple", "Jaccard mềm", "nhóm"], rows)
    # F1 từng lượt
    runs: dict[str, dict[str, dict]] = {}
    for rep in ("r2", "r3"):
        sc = find(d.repeat[rep], "scores_covid.json")
        res = get(sc, "results")
        if not res:
            continue
        by_t = res.get(EMBEDDER) or next(iter(res.values()))
        systems = by_t.get(THRESHOLD) or next(iter(by_t.values()))
        for key, m in systems.items():
            name, _, run = key.rpartition(".")
            runs.setdefault(name, {}).setdefault(run, m)
    rows = []
    for name, by_run in runs.items():
        f1s = [by_run[r].get("F1") for r in ("r1", "r2", "r3") if r in by_run]
        cells = [pct(by_run[r].get("F1")) if r in by_run else NA for r in ("r1", "r2", "r3")]
        ms = lambda k: [by_run[r].get(k) for r in by_run if by_run[r].get(k) is not None]  # noqa: E731
        sd = statistics.pstdev(f1s) if len(f1s) > 1 else None
        rows.append([name, *cells, f"{pct(statistics.mean(f1s))} ± {pct(sd)}" if f1s else NA,
                     pct(statistics.mean(ms("P"))) if ms("P") else NA, pct(statistics.mean(ms("R"))) if ms("R") else NA])
    out += "\nCOVID-NYT F1 theo lượt (embedder all-mpnet-base-v2, ngưỡng 0,7; std là độ lệch chuẩn tổng thể của các lượt có):\n\n"
    out += table(["hệ", "F1 r1", "F1 r2", "F1 r3", "F1 trung bình ± std", "P TB", "R TB"], rows)
    return out


def split_name(name: str) -> tuple[str, str]:
    ds, _, sysname = name.partition("/")
    return (ds, sysname) if sysname else ("?", name)


def sec_support(d: Data) -> str:
    rows = [[*split_name(n), pct(get(v, "support_rate")), pct(get(v, "support_rate_doc_mean")), num(get(v, "triples")),
             num(get(v, "judged")), num(get(v, "errors"))] for n, v in sorted(d.all_of("support").items())]
    return table(["bộ", "hệ", "support_rate", "support_rate theo văn bản (TB)", "triples", "judged", "errors"], rows)


def sec_redundancy(d: Data) -> str:
    red = d.all_of("redundancy")
    cols = ["entities", "entity_near_pairs", "entity_near_pairs_per_100", "triples", "triple_exact_dups", "triple_near_pairs",
            "triple_near_pairs_per_100"]
    rows = [[*split_name(n), *[num(get(v, c)) for c in cols], get(v, "scope") or NA] for n, v in sorted(red.items())]
    return table(["bộ", "hệ", *cols, "phạm vi"], rows)


def sec_nyt11(d: Data) -> str:
    st = d.stage["nyt11"]
    ex, edc = st.get("scores_exact") or {}, st.get("edc_script") or {}
    rows = []
    for name in list(dict.fromkeys([*ex, *edc])):
        label = f"{name} (OIE thô)" if name in RAW_OIE else name
        e = ex.get(name, {})
        rows.append([label, *prf(e.get("exact")), *prf(e.get("nocase")),
                     *[pct(get(edc, name, k, "F1")) for k in ("Partial", "Strict", "Exact")],
                     num(get(edc, name, "lines") if name in edc else e.get("lines")), num(get(edc, name, "scored")),
                     num(get(edc, name, "empty")), num(get(edc, name, "avg_triples")), num(e.get("unreadable_lines"))])
    return table(["hệ", "exact P", "R", "F1", "nocase P", "R", "F1", "EDC Partial F1", "Strict F1", "Exact F1", "dòng",
                  "chấm được", "rỗng", "triple TB", "dòng hỏng"], rows)


def sec_knet(d: Data) -> str:
    scores: dict = {}
    for st in ("knet_atom", "knet_main"):
        scores.update(d.stage[st].get("scores_knet") or {})
    rows = []
    for name, v in scores.items():
        label = f"{name} (OIE thô)" if name in RAW_OIE else name
        rows.append([label, *prf(v.get("span_overlap")), *prf(v.get("span_exact")), num(v.get("predicted")),
                     num(v.get("unmapped"))])
    b = d.baseline or {}
    rerun = b.get("rerun")
    if rerun:
        rows.append(["Baseline 5 (paper, chạy lại)", *prf(rerun.get("span_overlap")), *prf(rerun.get("span_exact")), NA, NA])
    else:
        rows.append(["Baseline 5 (README, không chạy lại được)", *prf(b.get("readme_dev_span_overlap") or README_DEV),
                     NA, NA, NA, NA, NA])
    out = table(["hệ", "overlap P", "R", "F1", "exact P", "R", "F1", "predicted", "unmapped"], rows)
    out += ("\nGhi chú: ánh xạ văn bản → vị trí ký tự của mình có trần 93,3 F1 span-overlap (dùng chính gold làm dự đoán). "
            "Tập \"dev\" của tác giả chính là fold 4 (cùng tập với các hàng trên); hàng README là số họ công bố, không phải mình chạy lại.\n")
    return out


def sec_merged(d: Data) -> str:
    rows = []
    # COVID-NYT: trung bình các lượt
    runs: dict[str, list[dict]] = {}
    for rep in ("r2", "r3"):
        res = get(find(d.repeat[rep], "scores_covid.json"), "results")
        if res:
            by_t = res.get(EMBEDDER) or next(iter(res.values()))
            systems = by_t.get(THRESHOLD) or next(iter(by_t.values()))
            for key, m in systems.items():
                name, _, run = key.rpartition(".")
                if run == "r1" and any(r.get("_run") == "r1" for r in runs.get(name, [])):
                    continue
                runs.setdefault(name, []).append({**m, "_run": run})
    for name, lst in runs.items():
        mean = {k: statistics.mean(r[k] for r in lst if r.get(k) is not None) for k in ("P", "R", "F1") if lst}
        rows.append(["COVID-NYT", f"{name} (TB {len(lst)} lượt)", *prf(mean)])
    # MINE-1
    support = d.all_of("support")
    acc = (d.mine_tables or {}).get("all", {})
    names = [n for n in support if n.startswith("mine/")]
    for n in sorted(names):
        key = MINE_ACC.get(n)
        a = acc.get(key) if key else None
        rows.append(["MINE-1", n.split("/", 1)[1], pct(get(support[n], "support_rate")),
                     f"{a:.1f}%" if isinstance(a, (int, float)) else NA, NA])
    # NYT11, KnowledgeNet
    for name, v in (d.stage["nyt11"].get("scores_exact") or {}).items():
        rows.append(["NYT11-HRL", f"{name} (OIE thô)" if name in RAW_OIE else name, *prf(v.get("exact"))])
    for st in ("knet_atom", "knet_main"):
        for name, v in (d.stage[st].get("scores_knet") or {}).items():
            rows.append(["KnowledgeNet (span-overlap)", f"{name} (OIE thô)" if name in RAW_OIE else name,
                         *prf(v.get("span_overlap"))])
    b = d.baseline or {}
    ov = get(b, "rerun", "span_overlap") or b.get("readme_dev_span_overlap") or README_DEV
    rows.append(["KnowledgeNet (span-overlap)", "Baseline 5 (" + ("chạy lại" if get(b, "rerun", "span_overlap") else "README") + ")",
                 *prf(ov)])
    out = table(["bộ", "hệ", "P", "R", "F1"], rows)
    out += ("\nGhi chú: ở MINE-1 không có P/R/F1 theo gold; cột P là tỷ lệ triple được văn bản nguồn hỗ trợ (judge Gemma), "
            "cột R là MINE accuracy (results/kaggle/mine/v3/tables.json, 100 bài), F1 không xác định (—).\n")
    return out


SECTIONS: list[tuple[str, Callable[[Data], str]]] = [
    ("0. Trạng thái kernel", sec_status),
    ("1. B — Độ ổn định COVID-NYT", sec_stability),
    ("2. A — Độ chính xác theo văn bản nguồn (judge Gemma)", sec_support),
    ("3. C — Độ gọn", sec_redundancy),
    ("4. D — NYT11-HRL", sec_nyt11),
    ("5. F — KnowledgeNet fold 4", sec_knet),
    ("6. Bảng P/R/F1 gộp", sec_merged),
]


def build(root: Path) -> str:
    try:
        d = Data(root)
    except Exception:
        return "# Báo cáo chạy đêm\n\nlỗi đọc dữ liệu:\n\n```\n" + traceback.format_exc() + "```\n"
    parts = [f"# Báo cáo chạy đêm\n\nNguồn: `{root}`. Tỷ lệ in dưới dạng phần trăm, một chữ số thập phân.\n"]
    for title, fn in SECTIONS:
        try:
            body = fn(d)
        except Exception as e:
            body = f"lỗi: {e!r}\n"
        parts.append(f"## {title}\n\n{body}")
    return "\n".join(parts)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("results/kaggle"))
    ap.add_argument("--out", type=Path, default=Path("results/kaggle/night/REPORT.md"))
    args = ap.parse_args(argv)
    text = build(args.root)
    try:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf8")
    except Exception as e:
        print(f"không ghi được {args.out}: {e!r}")
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", "replace").decode())
    return 0


if __name__ == "__main__":
    main()
