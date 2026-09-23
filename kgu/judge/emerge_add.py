from __future__ import annotations

from collections import Counter, defaultdict
from typing import Iterable

from pydantic import BaseModel

from kgu.data.emerge import EmergeExample
from kgu.llm import LLMClient
from kgu.types import Triple

SYSTEM = """You maintain a knowledge graph. You get a Wikipedia passage, a numbered list of ENTITIES the passage \
mentions that already exist in the graph, the facts ALREADY stored between them, and the RELATIONS the graph uses.
List the NEW facts the passage states between two listed entities, using only the listed relations. A fact counts \
only if the passage says it (directly or by clear paraphrase). Skip facts already stored. Skip relations not in the \
list. Prefer few, certain facts over many guesses. Give the entity numbers and the relation name exactly as listed.

Example:
ENTITIES: [0] John Roe  [1] Ohio  [2] Mary Major  [3] Republican Party
STORED: [0] position held Governor of Ohio
RELATIONS: member of political party, head of government, spouse, place of birth
Passage: "John Roe, a Republican, served as governor of Ohio until 2019 and was succeeded by Mary Major."
-> {"facts": [{"head": 0, "relation": "member of political party", "tail": 3}, \
{"head": 1, "relation": "head of government", "tail": 2}]}"""


class AddFact(BaseModel):
    head: int
    relation: str
    tail: int


class AddFacts(BaseModel):
    facts: list[AddFact]


def index_relations(kg: Iterable[Triple]) -> dict[str, set[str]]:
    """entity -> quan hệ xuất hiện trên cạnh 1-hop của nó (cả hai chiều)."""
    out: dict[str, set[str]] = defaultdict(set)
    for h, r, t in kg:
        out[h].add(r)
        out[t].add(r)
    return out


def allowed_relations(ex: EmergeExample, rels_of: dict[str, set[str]]) -> list[str]:
    """Quan hệ được phép = quan hệ trên lân cận 1-hop của entity được nhắc (phủ 83% gold Add)."""
    return sorted(set().union(*(rels_of.get(q, set()) for q in ex.mentioned)) if ex.mentioned else set())


def entity_labels(ex: EmergeExample, labels: dict[str, str]) -> dict[str, str]:
    """Nhãn KG trước, thiếu thì lấy chữ trong passage (entity không có cạnh nào trong subset không có nhãn KG)."""
    out = dict(labels)
    for m in ex.mentions:
        out.setdefault(m.qid, m.text)
    return out


def render(ex: EmergeExample, stored: list[Triple], relations: list[str], labels: dict[str, str]) -> str:
    labels = entity_labels(ex, labels)
    ents = ex.mentioned
    idx = {q: i for i, q in enumerate(ents)}
    ent_lines = "  ".join(f"[{i}] {labels.get(q, q)}" for i, q in enumerate(ents))
    stored_lines = [f"[{idx[h]}] {labels.get(r, r)} [{idx[t]}]" for h, r, t in stored if h in idx and t in idx]
    return (f"GRAPH DATE: {ex.snapshot}\nPASSAGE (Wikipedia page: {ex.title}):\n{ex.text}\n\n"
            f"ENTITIES: {ent_lines}\n"
            f"STORED: {'; '.join(stored_lines) if stored_lines else '(none)'}\n"
            f"RELATIONS: {', '.join(labels.get(r, r) for r in relations)}")


class KGShape:
    """Thống kê cấu trúc KG-trước để chuẩn hóa đầu ra Add (v2), không cần kiểu entity (P31 chỉ phủ 65%):
    - out_rels/in_rels: quan hệ mỗi entity đã phát ra / đã nhận trên cạnh 1-hop của nó;
    - inverse: cặp quan hệ nghịch đảo (has part(s) <-> part of, position held <-> position holder, ...) rút từ
      các cặp (h, r, t) & (t, r', h) cùng có trong KG (>= min_count lần và >= min_share số cạnh của r)."""

    def __init__(self, kg: Iterable[Triple], min_count: int = 30, min_share: float = 0.3):
        self.out_rels: dict[str, set[str]] = defaultdict(set)
        self.in_rels: dict[str, set[str]] = defaultdict(set)
        by_pair: dict[tuple[str, str], set[str]] = defaultdict(set)
        rel_tot: Counter[str] = Counter()
        for h, r, t in kg:
            self.out_rels[h].add(r)
            self.in_rels[t].add(r)
            by_pair[(h, t)].add(r)
            rel_tot[r] += 1
        inv_cnt: Counter[tuple[str, str]] = Counter()
        for (h, t), rs in by_pair.items():
            for r2 in by_pair.get((t, h), ()):
                for r in rs:
                    if r != r2:
                        inv_cnt[(r, r2)] += 1
        self.inverse: dict[str, str] = {}
        for (r, r2), c in inv_cnt.most_common():
            if c >= min_count and c / rel_tot[r] >= min_share and r not in self.inverse:
                self.inverse[r] = r2

    def normalize(self, triples: Iterable[Triple]) -> set[Triple]:
        """(1) đảo chiều nếu chiều ngược hợp với cạnh sẵn có của hai entity hơn (phim phát ra `cast member`,
        diễn viên không); (2) nếu quan hệ chưa từng xuất hiện ở cả hai đầu mà có đúng một quan hệ h đã phát ra
        và t đã nhận thì đổi sang quan hệ đó; (3) thêm chiều nghịch đảo vì gold Wikidata lưu cả hai chiều."""
        out: set[Triple] = set()
        for h, r, t in triples:
            fwd = (r in self.out_rels[h]) + (r in self.in_rels[t])
            rev = (r in self.out_rels[t]) + (r in self.in_rels[h])
            if rev > fwd:
                h, t, fwd, rev = t, h, rev, fwd
            if fwd == 0:
                cands = self.out_rels[h] & self.in_rels[t]
                cands_rev = self.out_rels[t] & self.in_rels[h]
                if len(cands) == 1 and not cands_rev:
                    r = next(iter(cands))
                elif len(cands_rev) == 1 and not cands:
                    r, h, t = next(iter(cands_rev)), t, h
            out.add((h, r, t))
            if r in self.inverse:
                out.add((t, self.inverse[r], h))
        return out


def judge_add(ex: EmergeExample, stored: list[Triple], relations: list[str], labels: dict[str, str],
              llm: LLMClient) -> set[Triple]:
    """Một lệnh gọi/instance. Trả về triple QID mới; bỏ nhãn quan hệ không khớp, chỉ số ngoài danh sách,
    cạnh đã có trong graph."""
    if len(ex.mentioned) < 2 or not relations:
        return set()
    by_label: dict[str, str] = {}
    for r in relations:
        by_label.setdefault(labels.get(r, r).lower(), r)
    ents = ex.mentioned
    stored_set = set(stored)
    out: set[Triple] = set()
    for f in llm.complete_json(SYSTEM, render(ex, stored, relations, labels), AddFacts).facts:
        r = by_label.get(f.relation.strip().lower())
        if r is None or not (0 <= f.head < len(ents)) or not (0 <= f.tail < len(ents)) or f.head == f.tail:
            continue
        tr = (ents[f.head], r, ents[f.tail])
        if tr not in stored_set:
            out.add(tr)
    return out
