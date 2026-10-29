"""Chấm KnowledgeNet bằng chính evaluator.py của nhóm tác giả: đổi triple chữ (canon_kg.txt, một đoạn mỗi dòng theo
map.json của scripts/prep_knet.py) về fact có vị trí ký tự trong văn bản gốc — tìm mọi lần xuất hiện chữ của thực thể
trong đoạn (không phân biệt hoa/thường, khoảng trắng tùy ý) và lấy cặp (đầu, đuôi) gần nhau nhất (đại từ như "He" lặp
nhiều lần trong một đoạn); thực thể không tìm thấy trong đoạn → bỏ fact đó (đếm "unmapped") — rồi chạy evaluator.py -f <fold> với span_overlap (chỉ số Text F1 của bảng xếp hạng) và span_exact.
URI để trống: không chấm được Link F1 (cần liên kết Wikidata).

Dùng: python scripts/score_knet.py --kn train.json --evaluator evaluator.py --map map.json --pred mine=canon_kg.txt \
          --workdir out/knet/eval --out out/knet/scores_knet.json
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path


def find_spans(text: str, s: str) -> list[tuple[int, int]]:
    toks = s.split()
    if not toks:
        return []
    pat = r"(?<!\w)" + r"\s+".join(map(re.escape, toks)) + r"(?!\w)"      # ranh giới từ: "he" không khớp trong "the"
    return [(m.start(), m.end()) for m in re.finditer(pat, text, re.I)]


def closest_pair(subj: list, obj: list) -> tuple | None:
    pairs = [(abs(a[0] - b[0]), a, b) for a in subj for b in obj if a[1] <= b[0] or b[1] <= a[0]]
    return min(pairs)[1:] if pairs else None


def to_facts(triples: list, passage: dict, name_to_id: dict[str, str],
             annotated: set[str] | None = None) -> tuple[list[dict], int]:
    """annotated: thuộc tính được gán nhãn đầy đủ trên đoạn này. Fact thuộc tính khác bị bỏ (không đếm unmapped):
    evaluator.py từ chối cả file khi gặp chúng, và cách chấm của tác giả (baseline chỉ phân loại thuộc tính được gán
    nhãn) vốn không tính chúng."""
    facts, unmapped = [], 0
    for t in triples:
        if len(t) != 3 or str(t[1]) not in name_to_id:
            unmapped += 1
            continue
        if annotated is not None and name_to_id[str(t[1])] not in annotated:
            continue
        pair = closest_pair(find_spans(passage["passageText"], str(t[0])), find_spans(passage["passageText"], str(t[2])))
        if not pair:
            unmapped += 1
            continue
        s, o = pair
        base = passage["passageStart"]
        fact = {"propertyId": name_to_id[str(t[1])], "subjectStart": base + s[0], "subjectEnd": base + s[1],
                "objectStart": base + o[0], "objectEnd": base + o[1], "subjectUri": "", "objectUri": "",
                "subjectText": str(t[0]), "objectText": str(t[2])}
        if fact not in facts:
            facts.append(fact)
    return facts, unmapped


def read_preds(path: Path) -> list[list]:
    out = []
    for line in path.read_text(encoding="utf8").splitlines():
        try:
            out.append(ast.literal_eval(line.strip()) if line.strip() else [])
        except Exception:
            out.append([])
    return out


def parse_global(text: str) -> dict:
    """Dòng #global ở mục Macro Evaluation (= đếm gộp mọi thuộc tính) và ma trận TP/FP/FN."""
    out = {}
    macro = text.split("Macro Evaluation", 1)[-1]
    m = re.search(r"#global\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", macro)
    if m:
        out.update(P=float(m.group(1)), R=float(m.group(2)), F1=float(m.group(3)))
    m = re.search(r"#global\s+(\d+)\s+(\d+)\s+(\d+)\s*\n", text)
    if m:
        out.update(TP=int(m.group(1)), FP=int(m.group(2)), FN=int(m.group(3)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kn", type=Path, required=True)
    ap.add_argument("--evaluator", type=Path, required=True)
    ap.add_argument("--map", type=Path, required=True)
    ap.add_argument("--pred", action="append", default=[])
    ap.add_argument("--workdir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    mp = json.loads(args.map.read_text(encoding="utf8"))
    name_to_id = {v: k for k, v in mp["properties"].items()}
    docs = [json.loads(line) for line in args.kn.read_text(encoding="utf8").splitlines() if line.strip()]
    docs = [d for d in docs if d["fold"] == mp["fold"]]
    annotated = {p["passageId"]: {a["propertyId"] for a in p["exhaustivelyAnnotatedProperties"]}
                 for d in docs for p in d["passages"]}
    args.workdir.mkdir(parents=True, exist_ok=True)
    out = {}
    for spec in args.pred:
        name, path = spec.split("=", 1)
        preds = read_preds(Path(path))
        by_passage, unmapped = {}, 0
        for i, p in enumerate(mp["passages"]):
            facts, u = to_facts(preds[i] if i < len(preds) else [], p, name_to_id, annotated.get(p["passageId"]))
            by_passage[p["passageId"]] = facts
            unmapped += u
        pred_file = args.workdir / f"pred_{name.replace('/', '_')}.json"
        with pred_file.open("w", encoding="utf8") as f:
            for d in docs:
                d2 = {**d, "passages": [{**p, "facts": by_passage.get(p["passageId"], [])} for p in d["passages"]]}
                f.write(json.dumps(d2, ensure_ascii=False) + "\n")
        res = {"unmapped": unmapped, "predicted": sum(map(len, by_passage.values()))}
        for method in ("span_overlap", "span_exact"):
            r = subprocess.run([sys.executable, str(args.evaluator), str(args.kn), str(pred_file), "-e", method, "-c",
                                "-f", str(mp["fold"])], capture_output=True, text=True, encoding="utf-8",
                               env={**os.environ, "PYTHONUTF8": "1"})      # evaluator mở file bằng bảng mã mặc định
            (args.workdir / f"{pred_file.stem}.{method}.txt").write_text(r.stdout + r.stderr, encoding="utf8")
            res[method] = parse_global(r.stdout)
        out[name] = res
        print(name, res, flush=True)
    args.out.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf8")


if __name__ == "__main__":
    main()
