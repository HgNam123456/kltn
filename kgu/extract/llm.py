from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from kgu.data.nba import Example
from kgu.graph import BiTemporalGraph
from kgu.llm import LLMClient
from kgu.types import Op, OpKind


class OpOut(BaseModel):
    kind: Literal["ADD", "INVALIDATE"]
    h: str
    r: str
    t: str
    quote: str = ""


class OpsOut(BaseModel):
    ops: list[OpOut]


SYSTEM = """Bạn là bộ trích xuất thay đổi cho knowledge graph.
Cho một bản tin và các cạnh hiện có của những entity được nhắc tới, hãy liệt kê CHỈ những thay đổi
mà văn bản KHẲNG ĐỊNH TRỰC TIẾP:
- INVALIDATE (h, r, t): một cạnh đang có trở nên không còn đúng.
- ADD (h, r, t): một cạnh mới được văn bản khẳng định.
Quy tắc:
- Chỉ dùng entity trong danh sách được cung cấp và quan hệ trong danh sách quan hệ. Không bịa entity mới.
- Không suy diễn hệ quả gián tiếp (đồng đội, HLV...). Việc đó do bước sau làm.
- Với mỗi op, chép nguyên văn đoạn text làm căn cứ vào "quote".
- Nếu không có thay đổi nào được khẳng định, trả về ops rỗng."""


class LLMExtractor:
    """LLM #1: text + ngữ cảnh entity → ops tường minh."""

    def __init__(self, client: LLMClient, relations: list[str], max_edges_per_entity: int = 40) -> None:
        self.client, self.relations, self.max_edges = client, relations, max_edges_per_entity
        self.n_dropped = 0

    @property
    def n_calls(self) -> int:
        return self.client.n_calls

    def _prompt(self, ex: Example, graph: BiTemporalGraph) -> str:
        lines = [f"BẢN TIN:\n{ex.text}", "", "QUAN HỆ CHO PHÉP: " + ", ".join(self.relations), "",
                 "ENTITY ĐƯỢC NHẮC VÀ CẠNH HIỆN CÓ:"]
        for e in ex.mentioned:
            edges = sorted(graph.edges_of(e, 0))[: self.max_edges]
            lines.append(f"- {e}:")
            lines.extend(f"    ({h}, {r}, {t})" for h, r, t in edges)
            if not edges:
                lines.append("    (chưa có cạnh nào)")
        lines += ["", "Liệt kê các op ADD / INVALIDATE được văn bản khẳng định trực tiếp."]
        return "\n".join(lines)

    def extract(self, ex: Example, graph: BiTemporalGraph) -> list[Op]:
        out = self.client.complete_json(SYSTEM, self._prompt(ex, graph), OpsOut)
        known = graph.entities(0) | set(ex.mentioned)
        ops: list[Op] = []
        for o in out.ops:
            triple = (o.h, o.r, o.t)
            active = graph.is_active(triple, 0)
            valid = (
                o.r in self.relations and o.h in known and o.t in known
                and ((o.kind == "INVALIDATE" and active) or (o.kind == "ADD" and not active))
            )
            if not valid:
                self.n_dropped += 1
                continue
            op = Op(OpKind(o.kind), o.h, o.r, o.t, source="llm", quote=o.quote)
            if op not in ops:
                ops.append(op)
        return ops
