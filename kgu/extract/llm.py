from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, create_model

from kgu.data import Example
from kgu.graph import BiTemporalGraph
from kgu.llm import LLMClient
from kgu.types import Op, OpKind


def make_ops_schema(relations: list[str]) -> type[BaseModel]:
    """Schema output với `r` là enum quan hệ → constrained decoding ép model chọn đúng ô quan hệ."""
    rel_type = Literal[tuple(relations)] if relations else str  # type: ignore[valid-type]
    op_out = create_model(
        "OpOut",
        kind=(Literal["ADD", "INVALIDATE"], ...),
        h=(str, ...), r=(rel_type, ...), t=(str, ...), quote=(str, ""),
    )
    return create_model("OpsOut", ops=(list[op_out], ...))


SYSTEM = """Bạn là bộ trích xuất thay đổi cho knowledge graph.
Cho một bản tin và các cạnh hiện có của những entity được nhắc tới, hãy liệt kê CHỈ những thay đổi
mà văn bản KHẲNG ĐỊNH TRỰC TIẾP:
- INVALIDATE (h, r, t): một cạnh ĐANG CÓ trong danh sách trở nên không còn đúng → chép nguyên cạnh đó.
- ADD (h, r, t): một cạnh mới được văn bản khẳng định.
Quy tắc:
- h và t PHẢI là tên entity y hệt trong danh sách entity. r PHẢI là một trong QUAN HỆ CHO PHÉP.
- Giữ cùng chiều (h, r, t) như các cạnh hiện có cùng loại quan hệ (vd nếu cạnh có dạng (Đội, <player>, Cầu_thủ)
  thì cạnh mới cũng là (Đội, <player>, Cầu_thủ)).
- Không bịa entity mới. Không suy diễn hệ quả gián tiếp (đồng đội, HLV...). Việc đó do bước sau làm.
- Với mỗi op, chép nguyên văn đoạn text làm căn cứ vào "quote".
- Nếu không có thay đổi nào được khẳng định, trả về ops rỗng."""

FEW_SHOT = """VÍ DỤ (miền khác, chỉ minh họa định dạng):
Bản tin: Messi rời Barca để gia nhập Inter.
Entity: Messi, Barca, Inter. Cạnh hiện có của Messi: (Messi, plays_for, Barca)
Kết quả đúng:
{"ops": [
  {"kind": "INVALIDATE", "h": "Messi", "r": "plays_for", "t": "Barca", "quote": "Messi rời Barca"},
  {"kind": "ADD", "h": "Messi", "r": "plays_for", "t": "Inter", "quote": "gia nhập Inter"}
]}
"""


class LLMExtractor:
    """LLM #1: text + ngữ cảnh entity → ops tường minh."""

    def __init__(self, client: LLMClient, relations: list[str], max_edges_per_entity: int = 40) -> None:
        self.client, self.relations, self.max_edges = client, relations, max_edges_per_entity
        self.schema = make_ops_schema(relations)
        self.n_dropped = 0

    @property
    def n_calls(self) -> int:
        return self.client.n_calls

    def _prompt(self, ex: Example, graph: BiTemporalGraph, at: int) -> str:
        lines = [FEW_SHOT, "BẢN TIN:", ex.text, "", "QUAN HỆ CHO PHÉP: " + ", ".join(self.relations),
                 "ENTITY (chỉ được dùng các tên này): " + ", ".join(ex.mentioned), "",
                 "CẠNH HIỆN CÓ CỦA TỪNG ENTITY:"]
        for e in ex.mentioned:
            edges = sorted(graph.edges_of(e, at))[: self.max_edges]
            lines.append(f"- {e}:")
            lines.extend(f"    ({h}, {r}, {t})" for h, r, t in edges)
            if not edges:
                lines.append("    (chưa có cạnh nào)")
        lines += ["", "Liệt kê các op ADD / INVALIDATE được văn bản khẳng định trực tiếp."]
        return "\n".join(lines)

    def extract(self, ex: Example, graph: BiTemporalGraph, at: int) -> list[Op]:
        out = self.client.complete_json(SYSTEM, self._prompt(ex, graph, at), self.schema)
        known = graph.entities(at) | set(ex.mentioned)
        ops: list[Op] = []
        for o in out.ops:
            triple = (o.h, o.r, o.t)
            active = graph.is_active(triple, at)
            valid = (
                o.r in self.relations and o.h in known and o.t in known      # r vẫn kiểm tra: đường json_object không ép enum
                and ((o.kind == "INVALIDATE" and active) or (o.kind == "ADD" and not active))
            )
            if not valid:
                self.n_dropped += 1
                continue
            op = Op(OpKind(o.kind), o.h, o.r, o.t, source="llm", quote=o.quote)
            if op not in ops:
                ops.append(op)
        return ops
