"""Chạy KARMA (bản gốc 03/2025: YuxingLu613/KARMA @ 82d68ba, một file karma_pipeline.py) trên EMERGE qua server
OpenAI-compatible (llama-server), ghi prediction dạng kg-gen để bước merge của EMERGE gộp vào evaluation_set.

KARMA gốc gắn cứng miền y sinh: Reader bỏ đoạn có điểm "liên quan tới tri thức y sinh" <= 0,2, NER chỉ tìm bệnh/thuốc/
gene, ví dụ quan hệ là treats/causes. --variant biomed chạy nguyên văn; --variant general chỉ thay các cụm chỉ miền
trong prompt (GENERAL_PATCH), few-shot trong system prompt giữ nguyên. Conflict Resolution chỉ bắt 4 cặp quan hệ y sinh
đối nghĩa và không bao giờ xóa triple cũ → KARMA không sinh Deprecate; mỗi bài chạy với KG rỗng như các baseline khác.

Dùng: python scripts/karma_emerge.py --karma tools/karma/karma_pipeline.py --input <evaluation_set> --output <dir> \
          --variant general --name karma/llamacpp/gemma-4-e4b --base-url http://127.0.0.1:8080/v1 --workers 4
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ENC = "encyclopedic"
GENERAL_PATCH: list[tuple[str, str]] = [
    # Reader: điểm liên quan quyết định đoạn nào được giữ (ngưỡng 0,2)
    ("You are a biomedical text relevance scorer.", f"You are an {ENC} text relevance scorer."),
    ("new biomedical knowledge (e.g., relationships between diseases, drugs, genes)",
     f"new {ENC} knowledge (e.g., relationships between people, organizations, places, works, events)"),
    # Summarizer
    ("Summarize the following biomedical text", f"Summarize the following {ENC} text"),
    ("retaining key domain terms (genes, proteins, drugs, diseases, etc.)",
     "retaining key domain terms (people, organizations, places, works, events, dates, etc.)"),
    # Entity extraction
    ("Extract biomedical entities from the text below.", "Extract named entities from the text below."),
    ("Include potential diseases, drugs, genes, proteins, chemicals, etc.",
     "Include people, organizations, places, creative works, events, dates, etc."),
    ("Assign an entity type (Disease, Drug, Gene, Protein, Chemical, etc.)",
     "Assign an entity type (Person, Organization, Location, Work, Event, Date, etc.)"),
    ("(e.g., UMLS:C0018681 for headache)", "(e.g., Wikidata Q90 for Paris)"),
    # Relationship extraction: bộ quan hệ chuẩn của miền đích (Wikidata) thay cho bộ y sinh
    ("Use standardized relationships such as 'treats', 'causes', 'inhibits', 'activates', 'interacts_with', "
     "'associated_with', etc.",
     "Use standardized relationships such as 'member of', 'employer', 'educated at', 'located in', 'part of', "
     "'award received', etc."),
    ("How important this relationship is in the biomedical domain", f"How important this relationship is in the {ENC} domain"),
    ("of biomedical relationships.", f"of {ENC} relationships."),
    # Schema alignment, conflict resolution, evaluator
    ("Classify each entity by its biomedical type: Drug, Disease, Gene, Protein, Chemical, RNA, Pathway, Cell, or Other.",
     "Classify each entity by its type: Person, Organization, Location, Work, Event, Date, Concept, or Other."),
    ("two potentially contradicting biomedical statements", "two potentially contradicting statements"),
    ("based on biological knowledge", "based on world knowledge"),
    ('domain="biomedical"', f'domain="{ENC}"'),
]


def patch_source(src: str, patches: list[tuple[str, str]]) -> str:
    """Thay mọi chỗ khớp; cụm nào không khớp thì báo lỗi (source upstream đã đổi, bản vá không còn đúng)."""
    for old, new in patches:
        if old not in src:
            raise ValueError(f"bản vá không khớp source KARMA: {old[:60]!r}")
        src = src.replace(old, new)
    return src


def load_karma(path: Path, variant: str) -> types.ModuleType:
    """Nạp karma_pipeline.py (đã vá nếu variant=general). PyPDF2 chỉ dùng để đọc PDF → thay bằng module rỗng."""
    sys.modules.setdefault("PyPDF2", types.ModuleType("PyPDF2"))
    src = Path(path).read_text(encoding="utf8")
    if variant == "general":
        src = patch_source(src, GENERAL_PATCH)
    module = types.ModuleType("karma_pipeline")
    sys.modules["karma_pipeline"] = module          # @dataclass tra module qua sys.modules
    exec(compile(src, str(path), "exec"), module.__dict__)
    logging.getLogger().setLevel(logging.WARNING)   # karma_pipeline bật INFO ở root → mỗi lượt gọi HTTP một dòng
    return module


def to_prediction(triples: list[tuple[str, str, str]], name: str) -> dict:
    """Format kg-gen: bộ chấm EMERGE đưa mọi triple vào Exists/Add/Mint+Add/Infer, không có Deprecate."""
    return {
        "predicted_triples": [{"extracted_relation": list(t), "triple_qids": [None] * 3, "triple_labels": [None] * 3}
                              for t in triples],
        "model": name,
        "model_type": "kg-gen",
    }


def run_one(karma_cls, instance: dict, base_url: str, model: str) -> dict:
    """Một bài = một pipeline KARMA mới (KG rỗng), như các baseline khác của EMERGE."""
    t0 = time.time()
    karma = karma_cls(api_key="none", base_url=base_url, model_name=model)
    triples = karma.process_document(instance["passage"])
    return {
        "hash_id": instance["hash_id"],
        "triples": [(t.head, t.relation, t.tail) for t in triples],
        "n_relevant_segments": len(karma.intermediate.relevant_segments),
        "n_relationships": len(karma.intermediate.relationships),
        "seconds": round(time.time() - t0, 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--karma", type=Path, required=True, help="đường dẫn karma_pipeline.py (commit 82d68ba)")
    ap.add_argument("--input", type=Path, required=True, help="evaluation_set/ (snapshot_*/delta_*.jsonl)")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--variant", choices=["biomed", "general"], required=True)
    ap.add_argument("--name", required=True, help="tên model ghi vào predictions")
    ap.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--model", default="local")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    karma_cls = load_karma(args.karma, args.variant).KARMA
    files = sorted(args.input.rglob("delta_*.jsonl"))
    jobs = [(f, json.loads(line)) for f in files for line in f.read_text(encoding="utf8").splitlines() if line.strip()]
    print(f"{len(jobs)} bài, variant={args.variant}, workers={args.workers}", flush=True)

    def work(job):
        f, inst = job
        try:
            return f, run_one(karma_cls, inst, args.base_url, args.model)
        except Exception as e:                                       # một bài lỗi không làm hỏng cả lượt
            print(f"!!! {inst['hash_id']}: {e!r}", flush=True)
            return f, {"hash_id": inst["hash_id"], "triples": [], "error": repr(e)}

    out: dict[Path, list[dict]] = {}
    with ThreadPoolExecutor(args.workers) as pool:
        for i, (f, rec) in enumerate(pool.map(work, jobs), 1):
            pred = to_prediction(rec.pop("triples"), args.name)
            pred["karma_stats"] = {k: v for k, v in rec.items() if k != "hash_id"}
            out.setdefault(f, []).append({"hash_id": rec["hash_id"], "predictions": {args.name: pred}})
            if i % 25 == 0:
                print(f"{i}/{len(jobs)}", flush=True)
    for f, rows in out.items():
        dest = args.output / f.relative_to(args.input)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf8")


if __name__ == "__main__":
    main()
