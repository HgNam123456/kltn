"""NYT11-HRL (Takanobu et al., AAAI 2019; bản tiền xử lý trong repo HRL-RE, tải từ Google Drive của họ) → thư mục theo
định dạng benchmark của EDC để mọi hệ chạy như WebNLG/REBEL/Wiki-NRE:
  input.txt (1 câu/dòng), ref.txt (str(list triple) mỗi dòng, quan hệ là đường dẫn Freebase gốc),
  schema.csv (quan hệ, mô tả — mô tả tự viết theo chiều của gold), oie_few_shot_examples.txt + sd_few_shot_examples.txt
  (ví dụ lấy từ train theo quy tắc cố định), gold.jsonl.
Test 369 câu / 370 triple, 12 quan hệ; train gán nhãn xa (distant supervision), test gán tay. Dữ liệu NYT thuộc LDC →
không đưa lên repo; thư mục ra được tải lên dataset Kaggle riêng tư.

Dùng: python scripts/prep_nyt11.py --raw <thư mục có test.json, train.json> --out <thư mục ra>
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

SCHEMA = {
    "/location/location/contains": "The subject location contains the object location (a smaller place, district, "
                                   "building or institution located inside it).",
    "/location/administrative_division/country": "The subject administrative division (state, province, region or "
                                                 "city) is located in the object country.",
    "/location/country/administrative_divisions": "The subject country has the object (a state, province, region or "
                                                  "city) as one of its administrative divisions.",
    "/people/person/place_lived": "The subject person lives or lived in the object location.",
    "/business/person/company": "The subject person works for, leads or is a member of the object company or "
                                "organization.",
    "/people/person/nationality": "The subject person is a citizen or national of the object country.",
    "/location/neighborhood/neighborhood_of": "The subject neighborhood is a neighborhood of the object city.",
    "/location/country/capital": "The object city is the capital of the subject country.",
    "/people/person/place_of_birth": "The subject person was born in the object location.",
    "/people/deceased_person/place_of_death": "The subject person died in the object location.",
    "/business/company/founders": "The object person founded the subject company or organization.",
    "/people/person/children": "The object person is a child of the subject person.",
}
# một ví dụ few-shot cho mỗi quan hệ này (câu train ngắn nhất ≤ MAX_CHARS trong FIRST_N dòng đầu, có quan hệ đó)
FEW_SHOT_RELATIONS = ["/location/location/contains", "/people/person/nationality", "/people/person/place_lived",
                      "/business/person/company", "/location/administrative_division/country",
                      "/business/company/founders"]
FIRST_N, MAX_CHARS = 5000, 180


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf8").splitlines() if line.strip()]


def clean(text: str) -> str:
    return " ".join(text.split())


def triples(rec: dict) -> list[list[str]]:
    out = []
    for r in rec["relations"]:
        t = [r["em1"], r["rtext"], r["em2"]]
        if t not in out:
            out.append(t)
    return out


def pick_examples(train: list[dict]) -> list[dict]:
    picked = []
    for rel in FEW_SHOT_RELATIONS:
        cands = [x for x in train[:FIRST_N] if any(r["rtext"] == rel for r in x["relations"])
                 and len(clean(x["sentext"])) <= MAX_CHARS and x not in picked]
        if cands:
            picked.append(min(cands, key=lambda x: len(clean(x["sentext"]))))
    return picked


def oie_examples(examples: list[dict]) -> str:
    return "\n\n".join(f"Example {i}:\nText: {clean(x['sentext'])}\nTriplets: {triples(x)}"
                       for i, x in enumerate(examples, 1)) + "\n"


def sd_examples(examples: list[dict]) -> str:
    blocks = []
    for i, x in enumerate(examples, 1):
        ts = triples(x)
        rels = list(dict.fromkeys(t[1] for t in ts))
        answer = "\n".join(f"{r}: {SCHEMA[r]}" for r in rels)
        blocks.append(f"Example {i}:\nText: {clean(x['sentext'])}\nTriples: {ts}\nRelations: {rels}\nAnswer:\n{answer}")
    return "\n\n".join(blocks) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    test, train = read_jsonl(args.raw / "test.json"), read_jsonl(args.raw / "train.json")
    assert {r["rtext"] for x in test for r in x["relations"]} <= set(SCHEMA), "quan hệ test ngoài schema"
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "input.txt").write_text("\n".join(clean(x["sentext"]) for x in test) + "\n", encoding="utf8")
    (args.out / "ref.txt").write_text("\n".join(str(triples(x)) for x in test) + "\n", encoding="utf8")
    with (args.out / "schema.csv").open("w", encoding="utf8", newline="") as f:
        csv.writer(f).writerows(SCHEMA.items())
    examples = pick_examples(train)
    (args.out / "oie_few_shot_examples.txt").write_text(oie_examples(examples), encoding="utf8")
    (args.out / "sd_few_shot_examples.txt").write_text(sd_examples(examples), encoding="utf8")
    with (args.out / "gold.jsonl").open("w", encoding="utf8") as f:
        for i, x in enumerate(test):
            f.write(json.dumps({"id": i, "text": clean(x["sentext"]), "triples": triples(x)}, ensure_ascii=False) + "\n")
    print(f"{len(test)} câu test, {sum(len(triples(x)) for x in test)} triple, {len(examples)} ví dụ few-shot → {args.out}")


if __name__ == "__main__":
    main()
