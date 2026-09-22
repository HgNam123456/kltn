from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Literal

from pydantic import BaseModel

from kgu.data.emerge import EmergeExample
from kgu.llm import LLMClient
from kgu.types import Triple

SYSTEM = """You maintain a knowledge graph. GRAPH DATE is when the graph was last updated; TODAY is when the Wikipedia passage was written, a few weeks later. You get the passage and numbered facts CURRENTLY stored in the graph.
For every fact answer two questions, in this order:
1. discussed: does the passage talk about this fact (state it, imply it, or say it changed)?
2. ended: only if discussed - does the passage say the fact STOPPED being true recently, i.e. between about one year before GRAPH DATE and TODAY? Signs: a term "from X to <that year>", "until", stepped down, resigned, abdicated, was succeeded or replaced by, was sacked, left the club, was sold to, divorced. Past tense ("served as", "was the") with an end date in that window means ended = true.
ended = false when the fact still holds, when it ended LONG before GRAPH DATE (the graph already knows), or when it is true forever (born in, country, instance of, results of a past election or match).

Example (GRAPH DATE 2019-01-01, TODAY 2019-01-15):
Passage: "John Roe (born 1950) is an American politician who served as the 30th governor of Ohio from 2011 to 2019. He was succeeded by Mary Major. He previously served as mayor of Dayton from 1994 to 2002. He is married to Ann Roe. Dayton FC sacked head coach Bob Low on 10 January 2019."
[0] John Roe | position held | Governor of Ohio -> discussed true, ended true (term ran to 2019)
[1] Governor of Ohio | officeholder | John Roe -> discussed true, ended true (same fact, other direction)
[2] Ohio | head of government | John Roe -> discussed true, ended true (he was succeeded)
[3] John Roe | position held | Mayor of Dayton -> discussed true, ended false (ended in 2002, long ago)
[4] Dayton FC | head coach | Bob Low -> discussed true, ended true (sacked this month)
[5] John Roe | spouse | Ann Roe -> discussed true, ended false
[6] John Roe | country of citizenship | United States -> discussed true, ended false
[7] Ohio | capital | Columbus -> discussed false, ended false"""


class FactVerdict(BaseModel):
    id: int
    discussed: bool
    ended: bool


class Verdicts(BaseModel):
    facts: list[FactVerdict]


class SingleVerdict(BaseModel):
    discussed: bool
    ended: bool


def _verdict(discussed: bool, ended: bool) -> str:
    return "unrelated" if not discussed else "ended" if ended else "holds"


def candidates(ex: EmergeExample, out_edges: dict[str, Iterable[Triple]]) -> list[Triple]:
    """Cạnh KG-trước mà cả hai đầu đều được nhắc trong passage. Thứ tự ổn định."""
    seeds = set(ex.mentioned)
    return sorted({tr for q in seeds for tr in out_edges.get(q, ()) if tr[2] in seeds})


def index_out_edges(kg: Iterable[Triple]) -> dict[str, list[Triple]]:
    out: dict[str, list[Triple]] = defaultdict(list)
    for tr in kg:
        out[tr[0]].append(tr)
    return out


def render(ex: EmergeExample, cand: list[Triple], labels: dict[str, str]) -> str:
    lines = [f"[{i}] {labels.get(h, h)} | {labels.get(r, r)} | {labels.get(t, t)}"
             for i, (h, r, t) in enumerate(cand)]
    return (f"GRAPH DATE: {ex.snapshot}\nTODAY: {ex.delta}\n"
            f"PASSAGE (Wikipedia page: {ex.title}):\n{ex.text}\n\n"
            f"FACTS CURRENTLY IN THE GRAPH:\n" + "\n".join(lines))


def judge(ex: EmergeExample, cand: list[Triple], labels: dict[str, str], llm: LLMClient,
          batch_size: int = 12) -> dict[Triple, str]:
    """Verdict ("ended" / "holds" / "unrelated") cho từng ứng viên. batch_size=1: mỗi cạnh một lệnh gọi
    có/không; lớn hơn: gom lô, cạnh model bỏ sót tính "unrelated"."""
    out: dict[Triple, str] = {}
    for start in range(0, len(cand), batch_size):
        batch = cand[start:start + batch_size]
        user = render(ex, batch, labels)
        if batch_size == 1:
            v = llm.complete_json(SYSTEM, user, SingleVerdict)
            out[batch[0]] = _verdict(v.discussed, v.ended)
            continue
        got = {f.id: _verdict(f.discussed, f.ended)
               for f in llm.complete_json(SYSTEM, user, Verdicts).facts if 0 <= f.id < len(batch)}
        out.update({tr: got.get(i, "unrelated") for i, tr in enumerate(batch)})
    return out


def to_ops(verdicts: dict[Triple, str], exists: Literal["llm", "all"] = "all") -> dict[str, set[Triple]]:
    """exists="all": mọi ứng viên không bị phán "ended" đều là Exists (tra KG đã đủ tin);
    exists="llm": chỉ cạnh model phán "holds"."""
    ended = {tr for tr, v in verdicts.items() if v == "ended"}
    keep = {tr for tr, v in verdicts.items() if v == "holds"} if exists == "llm" else set(verdicts) - ended
    return {"x-triples": keep, "d-triples": ended}
