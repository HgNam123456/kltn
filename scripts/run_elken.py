"""ELKEN (Peng et al. 2024, "Event-level Knowledge Editing", github THU-KEG/Event-Level-Knowledge-Editing) chạy như
bài toán cập nhật KG từ văn bản: sự kiện (một câu) → các fact phải đổi, phần lớn KHÔNG được nói thẳng trong câu
(chết → chức CEO bỏ trống, "previous CEO" = người cũ; chuyển nhượng → đội/giải/áo số...).

Chuyển sang dạng KG: mỗi sự kiện một bài toán
  - KG-trước  = fact "local" (phải giữ nguyên; data cho sẵn)  ∪  fact cũ suy từ câu hỏi "previous-*"/"ex"
                (câu hỏi "previous CEO của S là ai?" đáp P → trước sự kiện có (S, chief executive officer, P)).
  - KG-sau    = KG-trước  −  các fact cũ đó  +  (S, r, đáp án) cho câu hỏi fact có đáp án ≠ unknown.
  - Entity được nhắc = chủ ngữ của mọi câu hỏi (benchmark gốc cũng cho LLM biết chủ ngữ qua câu hỏi).
Quan hệ = nhãn tiếng Anh của rel_id (bảng REL dưới). Tân ngữ dự đoán được so với đáp án + alias (không phân biệt
hoa/thường, bỏ dấu câu); khớp thì thay bằng tên vàng rồi chấm bằng kgu.eval.metrics.score (F1 trên vũ trụ
before ∪ after ∪ pred, add_acc, del_acc) + locality (fact local còn nguyên).
Sự kiện không có câu hỏi fact (chỉ có tendency) bị bỏ; đáp án "unknown" không sinh fact (bài gốc chấm "unknown").
test.json trên GitHub bị cụt ở cuối (JSON không đóng) → bỏ item dở dang cuối cùng.

Bộ trích = LLMExtractor của pipeline nhưng lời nhắc riêng (tiếng Anh, cho phép hệ quả suy ra từ sự kiện); không
chạy khoanh vùng/bộ phán vì KG-trước chỉ có vài cạnh "local" (không có cấu trúc để khoanh).
Dùng: python scripts/run_elken.py --elken <repo> --split test --out results/elken_test.jsonl [--limit N] [--workers 4]
"""
from __future__ import annotations

import argparse
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, create_model
from tqdm import tqdm

from kgu.eval.metrics import Counts, score, summarize
from kgu.graph import BiTemporalGraph
from kgu.types import Triple

# rel_id → nhãn quan hệ (từ mẫu câu hỏi của bộ dữ liệu; P* là thuộc tính Wikidata)
REL = {
    "P551": "residence (city)", "P108": "employer", "P39": "position held", "major": "college major",
    "P159": "headquarters location (city)", "P69": "educated at", "P937": "work location (city)",
    "P19": "place of birth (city)", "P169": "chief executive officer", "P26": "spouse", "P54": "member of sports team",
    "jersey-number": "jersey number", "P112": "founded by", "P571": "inception date", "P118": "league",
    "P22": "father", "P25": "mother", "P27": "country of citizenship", "P21": "gender", "P50": "author",
    "P123": "publisher", "P577": "publication date", "P136": "genre", "P495": "country of origin",
    "P414": "stock exchange", "ipo-time": "IPO date", "stock-code": "stock code", "price-per-share": "IPO price per share",
    "P286": "head coach", "Q13204680": "father-in-law", "Q723868": "mother-in-law", "P291": "place of publication (city)",
    "P407": "language of work", "P488": "chairperson", "P6": "head of government", "sibling": "sibling",
    "P44": "stepmother of", "P43": "stepfather of", "P634": "captain", "P35": "head of state",
    "P20": "place of death (city)", "P570": "date of death", "acquire-time": "acquisition date",
    "office-located": "office location (city)", "P1128": "number of employees", "close-time": "closing date",
    "P749": "parent organization", "P101": "field of work", "P1412": "language spoken", "P17": "country",
    "P103": "native language", "P106": "occupation", "P184": "doctoral advisor", "P1546": "motto",
    "previous-ceo": "chief executive officer", "previous-chairman": "chairperson", "previous-coach": "head coach",
    "previous-head-of-state": "head of state", "previous-head-of-govern": "head of government", "ex": "spouse",
}
PREVIOUS = {"previous-ceo", "previous-chairman", "previous-coach", "previous-head-of-state", "previous-head-of-govern", "ex"}

SYSTEM = """You update a knowledge graph after an event. Given one event sentence, the entities involved, the allowed
relations and the facts currently stored about those entities, list the fact changes the event implies:
- ADD (h, r, t): a fact that becomes true because of the event. Include consequences that are not stated literally
  but follow from the event with certainty (a person who dies no longer holds a position; a transferred player now
  belongs to the new club and its league; a merger gives the acquired company a new parent organization).
  Use the entity's own knowledge where it is needed to name the object (e.g. the league of a well-known club).
- INVALIDATE (h, r, t): a stored fact that stops being true because of the event (copy it exactly from the list).
Rules: h must be one of the listed entities; r must be one of the allowed relations; t is a short name, date or
value. Do not add a fact whose value cannot be determined ("unknown"). Do not add facts unrelated to the event.
Return {"ops": []} if nothing changes."""

FEW_SHOT = """EXAMPLE (format only):
Event: Kim Ki Nam passed away in Nashville.
Entities: Kim Ki Nam, Samsung Electronics. Stored: (Samsung Electronics, chief executive officer, Kim Ki Nam)
Correct output:
{"ops": [
  {"kind": "INVALIDATE", "h": "Samsung Electronics", "r": "chief executive officer", "t": "Kim Ki Nam"},
  {"kind": "ADD", "h": "Kim Ki Nam", "r": "place of death (city)", "t": "Nashville"}
]}
"""


def norm(s: str) -> str:
    s = s.lower().strip().replace("_", " ")
    s = re.sub(r"[\"'’`.,;:!?()\[\]]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def load_elken(path: Path) -> list[dict]:
    s = path.read_text(encoding="utf8")
    try:
        return json.loads(s)
    except json.JSONDecodeError:                       # tệp cụt ở cuối → bỏ item dở dang
        cut = max(s.rfind("\n    },\n    {"), s.rfind("\r\n    },\r\n    {"))
        assert cut > 0, "không vá được JSON"
        return json.loads(s[:cut] + "\n    }\n]\n")


def subject_name(q: dict) -> str:
    return q["subject"].get("name") or q["subject"].get("ent2name") or ""


def build_examples(items: list[dict]) -> list[dict]:
    out = []
    for i, it in enumerate(items):
        qas = it.get("fact", {}).get("qas") or []
        if not qas:
            continue
        before: set[Triple] = set()
        local: set[Triple] = set()
        adds: dict[Triple, set[str]] = {}
        for q in it["fact"].get("local_qas", []):
            tr = (subject_name(q), REL.get(q["rel_id"], q["rel_id"]), q["answer"]["name"])
            local.add(tr)
            before.add(tr)
        for q in qas:
            s, r, ans = subject_name(q), REL.get(q["rel_id"], q["rel_id"]), q["answer"]["name"]
            if ans == "unknown":
                continue
            if q["rel_id"] in PREVIOUS:
                before.add((s, r, ans))                 # fact cũ, phải bị vô hiệu
            else:
                adds[(s, r, ans)] = {norm(ans)} | {norm(a) for a in q["answer"].get("alias", [])}
        invalid = {tr for tr in before if tr not in local}
        after = (before - invalid) | set(adds)
        mentioned = sorted({subject_name(q) for q in qas} | {subject_name(q) for q in it["fact"].get("local_qas", [])})
        out.append({"idx": i, "event_type": it["event_type"], "text": it["event"].strip('"'), "mentioned": mentioned,
                    "before": before, "after": after, "local": local, "aliases": {k: sorted(v) for k, v in adds.items()},
                    "n_unknown": sum(1 for q in qas if q["answer"]["name"] == "unknown")})
    return out


def make_schema(relations: list[str]) -> type[BaseModel]:
    rel_type = Literal[tuple(relations)]  # type: ignore[valid-type]
    op = create_model("OpOut", kind=(Literal["ADD", "INVALIDATE"], ...), h=(str, ...), r=(rel_type, ...), t=(str, ...))
    return create_model("OpsOut", ops=(list[op], ...))


def prompt(ex: dict, relations: list[str]) -> str:
    g = ex["before"]
    lines = [FEW_SHOT, f"Event: {ex['text']}", "", "Entities: " + ", ".join(ex["mentioned"]),
             "Allowed relations: " + ", ".join(relations), "", "Stored facts about these entities:"]
    lines += [f"  ({h}, {r}, {t})" for h, r, t in sorted(g)] or ["  (none)"]
    lines += ["", "List the ADD / INVALIDATE operations the event implies."]
    return "\n".join(lines)


def run_one(ex: dict, client, schema, relations: list[str]) -> dict:
    t0 = time.time()
    try:
        out = client.complete_json(SYSTEM, prompt(ex, relations), schema)
        raw_ops = [(o.kind, o.h, o.r, o.t) for o in out.ops]
        err = None
    except Exception as e:  # noqa: BLE001
        raw_ops, err = [], f"{type(e).__name__}: {e}"
    mentioned = set(ex["mentioned"])
    g = BiTemporalGraph.from_triples(ex["before"], valid_from=0)
    by_ent = {norm(m): m for m in mentioned}
    canon = {(norm(tr[0]), tr[1]): (tr, al) for tr, al in ex["aliases"].items()}
    pred_after = set(ex["before"])
    ops, dropped = [], 0
    for kind, h, r, t in raw_ops:
        h = by_ent.get(norm(h), h)
        if h not in mentioned or r not in relations or not t.strip() or norm(t) == "unknown":
            dropped += 1
            continue
        if kind == "INVALIDATE":
            hit = next((tr for tr in ex["before"] if tr[0] == h and tr[1] == r and norm(tr[2]) == norm(t)), None)
            if hit is None:
                dropped += 1
                continue
            pred_after.discard(hit)
            ops.append(("INVALIDATE", *hit))
        else:
            gold = canon.get((norm(h), r))
            if gold and norm(t) in gold[1]:
                t = gold[0][2]                             # khớp alias → tên vàng
            pred_after.add((h, r, t))
            ops.append(("ADD", h, r, t))
    c = score(ex["before"], ex["after"], pred_after)
    loc_keep = sum(1 for tr in ex["local"] if tr in pred_after)
    return {"idx": ex["idx"], "event_type": ex["event_type"], "text": ex["text"], "ops": ops, "dropped": dropped,
            "error": err, "counts": c.to_dict(), "n_local": len(ex["local"]), "local_keep": loc_keep,
            "n_unknown": ex["n_unknown"], "gold_add": sorted(ex["after"] - ex["before"]),
            "gold_del": sorted(ex["before"] - ex["after"]), "seconds": round(time.time() - t0, 2)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--elken", type=Path, required=True, help="thư mục repo Event-Level-Knowledge-Editing")
    ap.add_argument("--split", default="test", choices=["test", "train"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    items = load_elken(args.elken / "data" / "original" / f"{args.split}.json")
    examples = build_examples(items)[: args.limit]
    relations = sorted(set(REL.values()))
    from kgu.llm import client_from_env
    local = threading.local()

    def client():
        if not hasattr(local, "c"):
            local.c = client_from_env()
        return local.c

    schema = make_schema(relations)
    t0 = time.time()
    records = []
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf8") as f, ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for rec in tqdm(pool.map(lambda ex: run_one(ex, client(), schema, relations), examples), total=len(examples),
                        desc=f"elken-{args.split}"):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            records.append(rec)

    total = Counts()
    n_local = keep = 0
    by_type: dict[str, Counts] = {}
    for rec in records:
        c = Counts(**rec["counts"])
        total = total + c
        by_type[rec["event_type"]] = by_type.get(rec["event_type"], Counts()) + c
        n_local += rec["n_local"]
        keep += rec["local_keep"]
    s = summarize(total)
    s.update({"split": args.split, "n_events": len(records), "n_items_with_fact": len(examples),
              "n_items_total": len(items), "errors": sum(1 for r in records if r["error"]),
              "gold_add": total.n_add, "gold_del": total.n_del, "pred_ops": sum(len(r["ops"]) for r in records),
              "dropped": sum(r["dropped"] for r in records), "locality": keep / n_local if n_local else None,
              "n_unknown_answers": sum(r["n_unknown"] for r in records),
              "by_event_type": {k: summarize(v) | {"n_add": v.n_add, "n_del": v.n_del} for k, v in sorted(by_type.items())},
              "seconds": round(time.time() - t0, 1)})
    args.out.with_suffix(".summary.json").write_text(json.dumps(s, indent=2, ensure_ascii=False), encoding="utf8")
    print(json.dumps({k: v for k, v in s.items() if k != "by_event_type"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
