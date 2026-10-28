"""KnowledgeNet (Mesquita et al., EMNLP 2019; github.com/diffbot/knowledge-net, CC BY-SA 4.0) → thư mục định dạng
benchmark của EDC cho kaggle/extra_bench. Test (fold 5) bị giấu → dùng fold 4 như README của nhóm tác giả (họ gọi là
"dev"; baseline 5 của họ: span_overlap F1 0,704 trên fold này). Mỗi dòng input là một đoạn (passage, thường 1 câu) có
thuộc tính được gán nhãn đầy đủ; quan hệ = tên thuộc tính (DATE_OF_BIRTH, ...), mô tả lấy từ dữ liệu. Few-shot: với mỗi
thuộc tính, đoạn ngắn nhất ở fold 1–3 có fact của thuộc tính đó (≤ 3 fact), triple viết bằng đúng chữ trong đoạn
(subjectText/objectText, kể cả đại từ) vì bộ chấm so vị trí ký tự.
Ra: input.txt, ref.txt (triple chữ, chỉ để chấm phụ bằng script EDC), schema.csv, oie/sd_few_shot_examples.txt,
map.json (vị trí từng đoạn trong văn bản gốc, để scripts/score_knet.py đổi triple về vị trí ký tự).

Dùng: python scripts/prep_knet.py --kn knowledge-net/dataset/train.json --fold 4 --out data/knet
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

MAX_FACTS = 3


def one_line(s: str) -> str:
    return " ".join(s.split())


def facts_as_triples(passage: dict, names: dict[str, str]) -> list[list[str]]:
    out = []
    for f in passage["facts"]:
        t = [one_line(f["subjectText"]), names[f["propertyId"]], one_line(f["objectText"])]
        if t not in out:
            out.append(t)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kn", type=Path, required=True)
    ap.add_argument("--fold", type=int, default=4)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    docs = [json.loads(line) for line in args.kn.read_text(encoding="utf8").splitlines() if line.strip()]
    names, descs = {}, {}
    for d in docs:
        for p in d["passages"]:
            for a in p["exhaustivelyAnnotatedProperties"]:
                names[a["propertyId"]] = a["propertyName"]
                descs[a["propertyName"]] = a["propertyDescription"]
    test = [(d, p) for d in docs if d["fold"] == args.fold for p in d["passages"] if p["exhaustivelyAnnotatedProperties"]]
    pool = [(d, p) for d in docs if d["fold"] not in (args.fold, 5) for p in d["passages"]
            if p["facts"] and len(p["facts"]) <= MAX_FACTS]
    examples = []
    for pid in sorted(names, key=lambda x: names[x]):
        cands = [x for x in pool if any(f["propertyId"] == pid for f in x[1]["facts"]) and x not in examples]
        if cands:
            examples.append(min(cands, key=lambda x: len(x[1]["passageText"])))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "input.txt").write_text("\n".join(one_line(p["passageText"]) for _, p in test) + "\n", encoding="utf8")
    (args.out / "ref.txt").write_text("\n".join(str(facts_as_triples(p, names)) for _, p in test) + "\n", encoding="utf8")
    with (args.out / "schema.csv").open("w", encoding="utf8", newline="") as f:
        csv.writer(f).writerows(sorted(descs.items()))
    oie, sd = [], []
    for i, (_, p) in enumerate(examples, 1):
        ts = facts_as_triples(p, names)
        rels = list(dict.fromkeys(t[1] for t in ts))
        oie.append(f"Example {i}:\nText: {one_line(p['passageText'])}\nTriplets: {ts}")
        sd.append(f"Example {i}:\nText: {one_line(p['passageText'])}\nTriples: {ts}\nRelations: {rels}\nAnswer:\n"
                  + "\n".join(f"{r}: {descs[r]}" for r in rels))
    (args.out / "oie_few_shot_examples.txt").write_text("\n\n".join(oie) + "\n", encoding="utf8")
    (args.out / "sd_few_shot_examples.txt").write_text("\n\n".join(sd) + "\n", encoding="utf8")
    (args.out / "map.json").write_text(json.dumps(
        {"fold": args.fold, "properties": names,
         "passages": [{"documentId": d["documentId"], "passageId": p["passageId"], "passageStart": p["passageStart"],
                       "passageText": p["passageText"]} for d, p in test]}, ensure_ascii=False), encoding="utf8")
    print(f"fold {args.fold}: {len(test)} đoạn, {sum(len(p['facts']) for _, p in test)} fact, {len(names)} thuộc tính, "
          f"{len(examples)} ví dụ few-shot → {args.out}")


if __name__ == "__main__":
    main()
