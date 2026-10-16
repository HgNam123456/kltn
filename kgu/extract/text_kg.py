"""Pipeline của mình khi không có KG Wikidata (benchmark trích xuất sân nhà của baseline):
- extract_schema: module Add với bộ quan hệ lấy từ schema dataset (WebNLG) thay cho bộ quan hệ của KG lân cận,
  ép bằng enum trong json_schema (constrained decoding của llama-server);
- TemporalKG: KG tự tích lũy qua tin tức theo thời gian. Mỗi đoạn: localize = fact đã lưu có head/tail xuất hiện
  trong đoạn (STORED) + quan hệ đã dùng (RELATIONS) → một lệnh gọi trả fact đoạn này nói (Exists + Add, kèm mốc
  thời gian) và fact đã lưu bị đoạn này nói là hết đúng (Deprecate → đóng t_end). Luật quy đổi mốc thời gian tương
  đối và cách đặt tên quan hệ lấy theo quy ước công bố trong schema của ATOM (itext2kg Relationship.t_start/name) —
  quy ước gold của 2020-COVID-NYT, như WebNLG dùng schema của dataset.
Khi condense=True (spec 2026-10-15-condense-canonicalize-design.md §2.2): đoạn tin được rút thành câu đơn
(kgu.extract.condense), trích trên câu đánh số, câu chưa có fact được trích bổ sung một lệnh nữa.
Phần tĩnh của prompt (quan hệ, ví dụ) đặt trước, văn bản cần trích đặt cuối để llama-server dùng lại KV cache.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Literal

from pydantic import BaseModel, create_model

from kgu.extract.condense import condense as condense_text

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
2. valid_from: the date the fact starts or happens, ONLY if the passage states or implies a time; valid_until: ONLY \
if the passage says the fact stopped or will stop at some time. Otherwise leave them "". Do not put the observation \
date on facts with no time in the passage; a one-time event gets valid_from only. Write YYYY-MM-DD and resolve \
relative times against the observation date: today → the observation date; yesterday → one day before; this week → \
Monday of its week; last week → Monday of the week before; this month → first day of its month; last month → first \
day of the month before; this year → January 1st of its year; last year → January 1st of the year before; a bare \
year → January 1st of that year.
3. Relations are single present-tense verbs (is_located_in, reports, cancels); for something no longer true keep the \
affirmative relation and set valid_until.
4. In "ended", give the numbers of STORED facts the passage says are no longer true.

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


# metadata đầu mỗi đoạn COVID-NYT; chỉ bỏ đúng "Observation date: <ngày>" vì 2/122 đoạn dính liền văn bản, không xuống dòng
_HEADER = re.compile(r"(?im)^[ \t]*observation date:[ \t]*\d{4}-\d{2}-\d{2}[ \t]*\n?")


def _ds(v: str) -> list[str]:
    return [v.strip()] if v and v.strip() else []


SENTENCE_NOTE = """

The passages are given as numbered simple SENTENCES. For each fact also give "sentence": the number of the sentence it comes from. Extract every fact the sentences state; a sentence may state several facts or none (an opinion, a question). Head and tail are short entity names written as in the text (normal words with spaces, never snake_case). Never use a date, a weekday or the observation date as head or tail: times go only in valid_from and valid_until."""


class TFactS(TFact):
    sentence: int


class TOutS(BaseModel):
    facts: list[TFactS]
    ended: list[int]


def _numbered(sents: list[str], idx) -> str:
    return "SENTENCES:\n" + "\n".join(f"[{i}] {sents[i - 1]}" for i in idx)


class TemporalKG:
    def __init__(self, max_stored: int = 40, max_relations: int = 60, condense: bool = False):
        self.facts: list[list] = []                    # [head, relation, tail, t_start(list), t_end(list)]
        self.max_stored, self.max_relations = max_stored, max_relations
        self.use_condense = condense                   # False = v0: trích thẳng trên đoạn tin
        self.aliases: dict[str, set[str]] = {}         # tên chung → bí danh (bước gộp tên, kgu.extract.canonicalize)
        self.last_stats: dict = {}                     # thống kê của lần update gần nhất

    def _stored(self, text: str) -> list[int]:
        low = text.lower()
        seen = lambda e: any(re.search(rf"(?<!\w){re.escape(n.lower())}(?!\w)", low)        # noqa: E731
                             for n in (e, *self.aliases.get(e, ())))
        hits = [i for i, f in enumerate(self.facts) if seen(f[0]) or seen(f[2])]
        return hits[-self.max_stored:]                 # giữ fact mới nhất khi phải cắt

    def update(self, text: str, obs_date: str, llm) -> list[tuple[str, str, str, list[str], list[str]]]:
        stored = self._stored(text)
        rels = [r for r, _ in Counter(f[1] for f in self.facts).most_common(self.max_relations)]
        lines = "\n".join(f"[{k}] {self.facts[i][0]} | {self.facts[i][1]} | {self.facts[i][2]}"
                          for k, i in enumerate(stored))
        head = (f"OBSERVATION DATE: {obs_date}\nSTORED:\n{lines or '(none)'}\n"
                f"RELATIONS: {', '.join(rels) or '(none)'}\n")
        self.last_stats = {}
        sents: list[str] = []
        if self.use_condense:
            try:
                sents, cstats = condense_text(_HEADER.sub("", text), llm)
                self.last_stats.update(cstats)
            except Exception as e:                     # rút câu hỏng → trích trên đoạn gốc như v0
                self.last_stats["condense_error"] = repr(e)[:200]
        if sents:
            system = TEMPORAL_SYSTEM + SENTENCE_NOTE
            res = llm.complete_json(system, head + _numbered(sents, range(1, len(sents) + 1)), TOutS)
            facts = list(res.facts)
            todo = sorted(set(range(1, len(sents) + 1)) - {f.sentence for f in facts})
            self.last_stats["refill_sentences"] = len(todo)
            if todo:
                try:                                   # câu chưa có fact → trích riêng; "ended" của lệnh này bỏ qua
                    facts += llm.complete_json(system, head + _numbered(sents, todo), TOutS).facts
                except Exception as e:
                    self.last_stats["refill_error"] = repr(e)[:200]
        else:
            res = llm.complete_json(TEMPORAL_SYSTEM, head + f"Passage: {text}", TOut)
            facts = res.facts
        alias_of = {a: c for c, al in self.aliases.items() for a in al}   # bí danh → tên chung
        out = []
        for k in res.ended:
            if 0 <= k < len(stored) and not self.facts[stored[k]][4]:
                f = self.facts[stored[k]]
                f[4] = [obs_date]
                out.append((f[0], f[1], f[2], list(f[3]), [obs_date]))     # đoạn này nói fact đã hết → fact của đoạn
        for f in facts:
            q = (f.head.strip(), f.relation.strip(), f.tail.strip(), _ds(f.valid_from), _ds(f.valid_until))
            if q[0] and q[1] and q[2] and q[0].lower() != q[2].lower() and q not in out:
                out.append(q)
                m = [alias_of.get(q[0], q[0]), q[1], alias_of.get(q[2], q[2]), q[3], q[4]]   # lưu dưới tên chung
                if not any(g[:3] == m[:3] for g in self.facts):
                    self.facts.append(m)
        return out
