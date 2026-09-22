from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from kgu.data import Example
from kgu.types import Triple

__all__ = ["OPS", "ASSESSOR", "Mention", "EmergeExample", "load_snapshot", "load_kg_subset", "to_example"]

# Mã op nội bộ của EMERGE -> tên trong paper.
OPS = {
    "x-triples": "Exists",
    "e-triples": "Add",
    "ee-triples": "Mint+Add",
    "ee-kg-triples": "Infer",
    "d-triples": "Deprecate",
}

# Bộ chấm chính chủ chỉ tính gold được LLM này xác nhận là passage có hỗ trợ (config llm_assessors).
ASSESSOR = "Meta-Llama-3.1-405B_prompt_v1"


@dataclass(frozen=True)
class Mention:
    qid: str
    text: str
    start: int
    end: int


@dataclass
class EmergeExample:
    hash_id: str
    snapshot: str
    delta: str
    title: str
    text: str
    mentions: list[Mention]
    gold: dict[str, set[Triple]]                       # mã op -> triple (QID, PID, QID), đã được xác nhận
    unverified: dict[str, set[Triple]]                 # gold bị ASSESSOR bác, bộ chấm chính chủ bỏ qua
    emerging: set[str] = field(default_factory=set)    # entity chưa có trong KG snapshot
    labels: dict[str, str] = field(default_factory=dict)

    @property
    def mentioned(self) -> list[str]:
        return list(dict.fromkeys(m.qid for m in self.mentions))


def _verified(t: dict, op: str) -> bool:
    prompt = "triple_deprecation" if op == "d-triples" else "triple_assertion"
    return any(a["llm_name"] == ASSESSOR and a["llm_prompt_type"] == prompt and a["llm_assessment"]
               for a in t.get("llm_assessment", []))


def _parse(d: dict, snapshot: str, delta: str) -> EmergeExample:
    gold: dict[str, set[Triple]] = {op: set() for op in OPS}
    unverified: dict[str, set[Triple]] = {op: set() for op in OPS}
    emerging: set[str] = set()
    labels: dict[str, str] = {}
    for t in d["tkgu_triples"]:
        triple = tuple(t["triple"])
        labels.update(zip(triple, t["triple_labels"]))
        if t.get("emerging_head"):
            emerging.add(triple[0])
        if t.get("emerging_tail"):
            emerging.add(triple[2])
        for op in t["tkgu_operations"]:
            (gold if _verified(t, op) else unverified)[op].add(triple)
    return EmergeExample(
        hash_id=d["hash_id"], snapshot=snapshot, delta=delta,
        title=d.get("anchor_title", ""), text=d["passage"],
        mentions=[Mention(m["qid"], m["mention_text"], m["start_char"], m["end_char"])
                  for m in d["mentions"] if m.get("qid")],
        gold=gold, unverified=unverified, emerging=emerging, labels=labels,
    )


def load_snapshot(data_dir: Path, snapshot: str, limit: int | None = None) -> list[EmergeExample]:
    """Đọc mọi delta của một snapshot (vd "2019-01-01") trong data_dir/evaluation_set/."""
    out: list[EmergeExample] = []
    for path in sorted((data_dir / "evaluation_set" / f"snapshot_{snapshot}").glob("delta_*.jsonl")):
        delta = path.stem.removeprefix("delta_")
        with path.open(encoding="utf8") as f:
            for line in f:
                if limit is not None and len(out) >= limit:
                    return out
                out.append(_parse(json.loads(line), snapshot, delta))
    return out


def load_kg_subset(path: Path) -> tuple[set[Triple], dict[str, str]]:
    """Đọc file do scripts/build_emerge_subgraphs.py sinh: h, r, t, nhãn h, nhãn r, nhãn t."""
    triples: set[Triple] = set()
    labels: dict[str, str] = {}
    with path.open(encoding="utf8") as f:
        for line in f:
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 3:
                continue
            triples.add((cols[0], cols[1], cols[2]))
            labels.update(zip(cols[:3], cols[3:6]))
    return triples, labels


def to_example(ex: EmergeExample, idx: int, kg: set[Triple]) -> Example:
    """Đưa về khuôn (before, after) của pipeline: before = cạnh 1-hop quanh entity được nhắc,
    after = before − Deprecate + Add/Mint+Add/Infer."""
    seeds = set(ex.mentioned)
    before = {tr for tr in kg if tr[0] in seeds or tr[2] in seeds}
    added = ex.gold["e-triples"] | ex.gold["ee-triples"] | ex.gold["ee-kg-triples"]
    return Example(
        idx=idx, event=ex.title, season=ex.snapshot, text=ex.text, mentioned=ex.mentioned,
        before=before, after=(before - ex.gold["d-triples"]) | added,
    )
