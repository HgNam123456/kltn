"""TKGQA (Ong et al. 2023, Data 8(3):61; OSF doi 10.17605/OSF.IO/XQWA4, CC BY 4.0): 5.721 bản tin M&A, mỗi tin kèm
fact tường minh (bidder, giai đoạn thương vụ, target, ngày) và TKG toàn cục trước/sau. KG khởi tạo từ Wikidata
(owner of, subsidiary, business division, board member, owned by); khi thương vụ MUA thành công, luật của tác giả
chuyển tài sản của target sang bidder, đóng chủ sở hữu/ban điều hành cũ của target và gán chủ sở hữu/ban điều hành của
bidder cho target — tức thay đổi gián tiếp bản tin không nhắc (bài báo gọi là "implicit changes").

Script này chuyển tệp pickle tkg_{test,val,train}_data_1110.pickle (mỗi bản ghi mang cả đồ thị toàn cục) thành
định dạng kgu.data.jsonl: KG-trước/sau = fact ĐANG MỞ (quad 4 phần) trong 1-hop của bidder và target. Pickle chỉ chứa
dict/list/str (đã quét opcode), vẫn mở bằng Unpickler từ chối mọi lớp.

Dùng: python scripts/prep_tkgqa.py --pickle <tkg_test_data_1110.pickle> --out data/cascade/tkgqa_test.jsonl
      [--only-implicit]  (chỉ giữ bài có >= 1 thay đổi gián tiếp)
"""
from __future__ import annotations

import argparse
import ast
import json
import pickle
import statistics
from pathlib import Path


class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):  # noqa: D401
        raise pickle.UnpicklingError(f"từ chối lớp {module}.{name}")

    def persistent_load(self, pid):
        raise pickle.UnpicklingError("từ chối persistent id")


def convert(pickle_path: Path, only_implicit: bool = False, idx0: int = 0) -> tuple[list[dict], dict]:
    with pickle_path.open("rb") as f:
        recs = SafeUnpickler(f).load()
    out, sizes = [], []
    n_imp = n_add = n_del = 0
    for r in recs:
        explicit = tuple(ast.literal_eval(r["tuples"]) if isinstance(r["tuples"], str) else r["tuples"])
        head, tail = explicit[0], explicit[2]
        ents = {head, tail}

        def open_triples(quads) -> set[tuple[str, str, str]]:
            return {(q[0], q[1], q[2]) for q in quads if len(q) == 4 and (q[0] in ents or q[2] in ents)}

        before, after = open_triples(r["before_tkg"]), open_triples(r["after_tkg"])
        implicit_add = (after - before) - {explicit[:3]}
        implicit_del = {t for t in before - after if {t[0], t[2]} != ents}
        if only_implicit and not (implicit_add or implicit_del):
            continue
        n_imp += bool(implicit_add or implicit_del)
        n_add += len(implicit_add)
        n_del += len(implicit_del)
        sizes.append(len(before))
        out.append({
            "idx": idx0 + len(out), "event": r["relations"], "season": str(r["date"]), "text": r["sentence_text"],
            "mentioned": [head, tail], "before": sorted(before), "after": sorted(after),
            "extra": {"article_id": r["article_id"], "trigger": list(explicit[:3]), "split": pickle_path.stem,
                      "n_implicit_add": len(implicit_add), "n_implicit_del": len(implicit_del)},
        })
    stats = {"source": pickle_path.name, "records": len(recs), "kept": len(out), "with_implicit": n_imp,
             "implicit_add": n_add, "implicit_del": n_del,
             "before_edges_mean": round(statistics.mean(sizes), 1) if sizes else 0,
             "before_edges_median": statistics.median(sizes) if sizes else 0, "before_edges_max": max(sizes, default=0)}
    return out, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pickle", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--only-implicit", action="store_true")
    args = ap.parse_args()
    rows, stats = [], []
    for p in args.pickle:
        r, s = convert(p, args.only_implicit, idx0=len(rows))
        rows += r
        stats.append(s)
        print(json.dumps(s, ensure_ascii=False), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    args.out.with_suffix(".stats.json").write_text(json.dumps(stats, indent=1, ensure_ascii=False), encoding="utf8")


if __name__ == "__main__":
    main()
