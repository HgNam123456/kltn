"""Pipeline của mình khi không có KG Wikidata (benchmark trích xuất sân nhà của baseline):
- extract_schema: module Add với bộ quan hệ lấy từ schema dataset (WebNLG) thay cho bộ quan hệ của KG lân cận,
  ép bằng enum trong json_schema (constrained decoding của llama-server);
- TemporalKG: KG tự tích lũy qua tin tức theo thời gian. Mỗi đoạn: localize = fact đã lưu có head/tail xuất hiện
  trong đoạn (STORED) + quan hệ đã dùng (RELATIONS) → một lệnh gọi trả fact đoạn này nói (Exists + Add, kèm mốc
  thời gian) và fact đã lưu bị đoạn này nói là hết đúng (Deprecate → đóng t_end).
Phần tĩnh của prompt (quan hệ, ví dụ) đặt trước, văn bản cần trích đặt cuối để llama-server dùng lại KV cache.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Literal

from pydantic import BaseModel, create_model

SCHEMA_SYSTEM = """Extract the facts the text states as (subject, relation, object) triples, using ONLY the listed \
relations. Subjects and objects are names or values copied from the text, written like in the examples. A fact counts \
only if the text says it (directly or by clear paraphrase). Give every such fact, no duplicates."""

TEMPORAL_SYSTEM = """You maintain a temporal knowledge graph built from dated news. You get news passages with \
their observation date, STORED facts already in the graph that mention entities of these passages, and the RELATIONS \
the graph already uses.
1. Break the passages into atomic facts (head, relation, tail, valid_from, valid_until) and list ALL of them: one \
relation per fact, head and tail are short entity names (not clauses). A sentence usually gives several facts: who \
did what, where things are (X is_located_in Y), what belongs to what, quantities, statuses, causes. Reuse the exact \
entity and relation names of STORED facts and RELATIONS when they mean the same thing; otherwise create a short \
present-tense snake_case relation.
2. valid_from: the date the fact starts or happens, ONLY if the passage states or implies a time ("on Sunday", "last \
month", "in 2003", "17 years ago"): resolve it against the observation date to YYYY-MM-DD. valid_until: ONLY if the \
passage says the fact stopped or will stop at some time. Otherwise leave them "". Do not put the observation date on \
facts with no time in the passage; a one-time event gets valid_from only.
3. In "ended", give the numbers of STORED facts the passage says are no longer true.

Example:
OBSERVATION DATE: 2020-03-02
STORED: [0] Italy | has_status | Low Case Count
RELATIONS: has_status, is_located_in, causes
Passage: "Italy, which had few cases until last week, reported 1,694 cases on Sunday, most of them in Lombardy, a \
northern region. Schools in Milan will stay closed."
-> {"facts": [{"head": "Italy", "relation": "reports", "tail": "1,694 Cases", "valid_from": "2020-03-01", \
"valid_until": ""}, {"head": "1,694 Cases", "relation": "are_located_in", "tail": "Lombardy", "valid_from": "", \
"valid_until": ""}, {"head": "Lombardy", "relation": "is_located_in", "tail": "Italy", "valid_from": "", \
"valid_until": ""}, {"head": "Schools", "relation": "are_located_in", "tail": "Milan", "valid_from": "", \
"valid_until": ""}, {"head": "Schools", "relation": "has_status", "tail": "Closed", "valid_from": "", \
"valid_until": ""}], "ended": [0]}"""


def _schema_model(relations: list[str]) -> type[BaseModel]:
    fact = create_model("Fact", subject=(str, ...), relation=(Literal[tuple(relations)], ...), object=(str, ...))
    return create_model("Facts", facts=(list[fact], ...))


def extract_schema(text: str, relations: dict[str, str] | list[str], llm,
                   examples: str = "") -> list[tuple[str, str, str]]:
    """relations: tên → mô tả (mô tả có thể rỗng) hoặc danh sách tên."""
    rels = relations if isinstance(relations, dict) else dict.fromkeys(relations, "")
    by_lower = {r.lower(): r for r in rels}
    listing = "\n".join(f"- {r}: {d}" if d else f"- {r}" for r, d in rels.items())
    user = f"RELATIONS:\n{listing}\n\n" + (f"{examples.strip()}\n\n" if examples else "") + f"Text: {text}"
    out: list[tuple[str, str, str]] = []
    for f in llm.complete_json(SCHEMA_SYSTEM, user, _schema_model(list(rels))).facts:
        r = by_lower.get(str(f.relation).strip().lower())        # đường json_object không ép enum → kiểm lại
        s, o = str(f.subject).strip(), str(f.object).strip()
        if r and s and o and s.lower() != o.lower() and (s, r, o) not in out:
            out.append((s, r, o))
    return out


class TFact(BaseModel):
    head: str
    relation: str
    tail: str
    valid_from: str
    valid_until: str


class TOut(BaseModel):
    facts: list[TFact]
    ended: list[int]


def _ds(v: str) -> list[str]:
    return [v.strip()] if v and v.strip() else []


class TemporalKG:
    def __init__(self, max_stored: int = 40, max_relations: int = 60):
        self.facts: list[list] = []                    # [head, relation, tail, t_start(list), t_end(list)]
        self.max_stored, self.max_relations = max_stored, max_relations

    def _stored(self, text: str) -> list[int]:
        low = text.lower()
        seen = lambda e: re.search(rf"(?<!\w){re.escape(e.lower())}(?!\w)", low) is not None    # noqa: E731
        hits = [i for i, f in enumerate(self.facts) if seen(f[0]) or seen(f[2])]
        return hits[-self.max_stored:]                 # giữ fact mới nhất khi phải cắt

    def update(self, text: str, obs_date: str, llm) -> list[tuple[str, str, str, list[str], list[str]]]:
        stored = self._stored(text)
        rels = [r for r, _ in Counter(f[1] for f in self.facts).most_common(self.max_relations)]
        lines = "\n".join(f"[{k}] {self.facts[i][0]} | {self.facts[i][1]} | {self.facts[i][2]}"
                          for k, i in enumerate(stored))
        user = (f"OBSERVATION DATE: {obs_date}\nSTORED:\n{lines or '(none)'}\n"
                f"RELATIONS: {', '.join(rels) or '(none)'}\nPassage: {text}")
        res = llm.complete_json(TEMPORAL_SYSTEM, user, TOut)
        out = []
        for k in res.ended:
            if 0 <= k < len(stored) and not self.facts[stored[k]][4]:
                f = self.facts[stored[k]]
                f[4] = [obs_date]
                out.append((f[0], f[1], f[2], list(f[3]), [obs_date]))     # đoạn này nói fact đã hết → fact của đoạn
        for f in res.facts:
            q = (f.head.strip(), f.relation.strip(), f.tail.strip(), _ds(f.valid_from), _ds(f.valid_until))
            if q[0] and q[1] and q[2] and q[0].lower() != q[2].lower() and q not in out:
                out.append(q)
                if not any(g[:3] == list(q[:3]) for g in self.facts):
                    self.facts.append(list(q))
        return out
