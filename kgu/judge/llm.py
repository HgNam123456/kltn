from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, create_model

from kgu.judge import JudgeContext
from kgu.llm import LLMClient
from kgu.localize import AddCandidate, CutCandidate, Localization
from kgu.types import Op, OpKind

Direction = Literal["subject_first", "neighbor_first"]


# ---------------------------------------------------------------------------
# Gom ứng viên theo MẪU QUAN HỆ. LLM phán mẫu; phán quyết ánh xạ xuống mọi cạnh trong mẫu.
# ---------------------------------------------------------------------------

@dataclass
class CutPattern:
    """Mẫu (p, r, X) hoặc (X, r, p): p là participant, X chạy trên các hub."""
    participant: str
    r: str
    direction: str                                   # subject_first: (p, r, X); neighbor_first: (X, r, p)
    members: list[CutCandidate] = field(default_factory=list)

    @property
    def hubs(self) -> list[str]:
        return [c.hub for c in self.members]

    @property
    def co_anchors(self) -> set[str]:
        """Các participant KHÁC mà hub X cũng nối tới (bối cảnh: X là đồng đội cũ, CLB cũ, ...)."""
        return {a for c in self.members for a in c.anchors} - {self.participant}


@dataclass
class AddPattern:
    """Mẫu a ~ X với cùng cạnh bằng chứng (X, r', b) hoặc (b, r', X)."""
    subject: str
    via_r: str
    via_direction: str                               # neighbor_first: (X, r', b); subject_first: (b, r', X)
    other: str                                       # b
    members: list[AddCandidate] = field(default_factory=list)

    @property
    def neighbors(self) -> list[str]:
        return [c.neighbor for c in self.members]


def group_cut(loc: Localization) -> list[CutPattern]:
    out: dict[tuple[str, str, str], CutPattern] = {}
    for c in loc.cut:
        h, r, t = c.triple
        if h == c.hub:
            key = (t, r, "neighbor_first")
        else:
            key = (h, r, "subject_first")
        out.setdefault(key, CutPattern(*key)).members.append(c)
    return list(out.values())


def group_add(loc: Localization) -> list[AddPattern]:
    out: dict[tuple[str, str, str, str], AddPattern] = {}
    for c in loc.add:
        vh, vr, vt = c.via
        if vh == c.neighbor:
            key = (c.subject, vr, "neighbor_first", vt)
        else:
            key = (c.subject, vr, "subject_first", vh)
        out.setdefault(key, AddPattern(*key)).members.append(c)
    return list(out.values())


# ---------------------------------------------------------------------------
# Schema output (constrained decoding)
# ---------------------------------------------------------------------------

def _add_verdict_model(relations: list[str]) -> type[BaseModel]:
    rel_type = Literal[tuple(relations)] if relations else str  # type: ignore[valid-type]
    return create_model("AddVerdict", idx=(int, ...), r=(rel_type, ...), direction=(Direction, "subject_first"))


def make_cut_schema() -> type[BaseModel]:
    """Pha CẮT: chỉ số mẫu [Ci] cần INVALIDATE (mặc định KEEP)."""
    return create_model("CutOut", invalidate=(list[int], []))


def make_add_schema(relations: list[str]) -> type[BaseModel]:
    """Pha SINH: các {idx, r, direction} cho mẫu [Ai] cần thêm (mặc định bỏ qua). `r` là enum → constrained decoding."""
    return create_model("AddOut", add=(list[_add_verdict_model(relations)], []))


def make_judge_schema(relations: list[str]) -> type[BaseModel]:
    """Schema gộp 1 pha (giữ để tương thích / so sánh)."""
    return create_model("JudgeOut", invalidate=(list[int], []), add=(list[_add_verdict_model(relations)], []))


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_COMMON = """Một bản tin đã được áp các thay đổi TƯỜNG MINH lên knowledge graph. Bạn phán các MẪU QUAN HỆ bị ảnh hưởng GIÁN TIẾP.
Mỗi mẫu gộp nhiều cạnh cùng dạng (X chạy trên một nhóm entity); phán quyết cho mẫu áp cho MỌI cạnh trong mẫu.
Chỉ dùng quan hệ trong danh sách. Không đoán sự kiện chưa xảy ra."""

SYSTEM_CUT = _COMMON + """

PHA CẮT. Trả về invalidate = danh sách số i của các mẫu [Ci] KHÔNG CÒN ĐÚNG sau thay đổi. Không liệt kê = KEEP.
- Fact TRẠNG THÁI (đang chơi cho, là đồng đội, là HLV của) chỉ bị vô hiệu khi điều kiện của nó mất đi.
- Fact SỰ KIỆN LỊCH SỬ (đã ghi bàn, đã vô địch, sinh tại) LUÔN KEEP.
- Chỉ entity THAM GIA thay đổi mới đổi trạng thái. X không tham gia → quan hệ của X với nơi X vẫn ở → KEEP.
  Quan hệ giữa X và entity đã RỜI KHỎI nơi của X (đồng đội) → cắt.

VÍ DỤ (miền khác): tin "Messi rời Barca sang Inter", đã áp INVALIDATE (Messi, plays_for, Barca), ADD (Messi, plays_for, Inter).
  [C0] (X, plays_for, Barca) — 2 X: Pedri, Gavi; X cũng nối với Messi
  [C1] (Messi, teammate, X) — 2 X: Pedri, Gavi; X cũng nối với Barca
  [C2] (X, teammate, Messi) — 2 X: Pedri, Gavi; X cũng nối với Barca
  [C3] (Pedri, close_friend, X) — 1 X: Messi
  Kết quả đúng: {"invalidate": [1, 2]}
  (C0 giữ: Pedri, Gavi không tham gia, vẫn ở Barca. C3 giữ: bạn thân không phụ thuộc CLB.
   C1, C2 cắt: đồng đội chỉ đúng khi cùng CLB, Messi đã rời Barca.)"""

SYSTEM_ADD = _COMMON + """

PHA SINH. Trả về add = danh sách {idx, r, direction} cho các mẫu [Ai] mà cấu trúc mới THỰC SỰ hàm ý. Không liệt kê = bỏ qua.
- direction: subject_first = (subject, r, X); neighbor_first = (X, r, subject). Quan hệ đối xứng (đồng đội) → 2 mục cùng idx, 2 direction.
- Chỉ thêm cho entity vừa GIA NHẬP nơi mới: quan hệ với những X đang ở nơi đó (bằng chứng nối X với nơi mới).
  Mẫu mà bằng chứng nối X với entity CŨ / nơi cũ (X là đồng đội cũ) → bỏ qua: X không đi theo.
- Không thêm cạnh ngoài danh sách mẫu.

VÍ DỤ (miền khác): tin "Messi rời Barca sang Inter", đã áp INVALIDATE (Messi, plays_for, Barca), ADD (Messi, plays_for, Inter).
  [A0] Messi ~ X — 2 X: Lautaro, Barella; bằng chứng: (X, plays_for, Inter)
  [A1] Messi ~ X — 1 X: Inzaghi; bằng chứng: (X, head_coach, Inter)
  [A2] Inter ~ X — 2 X: Pedri, Gavi; bằng chứng: (X, teammate, Messi)
  Kết quả đúng: {"add": [{"idx": 0, "r": "teammate", "direction": "subject_first"},
                         {"idx": 0, "r": "teammate", "direction": "neighbor_first"}]}
  (A0 thêm 2 chiều: Messi thành đồng đội của cầu thủ Inter. A1 bỏ: không có quan hệ cầu thủ–HLV trong danh sách.
   A2 bỏ: đồng đội cũ của Messi không sang Inter.)"""

SYSTEM = SYSTEM_CUT + "\n\n" + SYSTEM_ADD.split("\n\n", 1)[1]   # bản gộp 1 pha (tương thích)


def _examples(names: list[str], k: int = 2) -> str:
    shown = ", ".join(names[:k])
    return shown + (", ..." if len(names) > k else "")


class LLMJudge:
    """Phương án B, 2 pha: (1) CẮT — phán mẫu giữ/bỏ; (2) SINH — phán mẫu thêm. Mỗi pha 1 lệnh gọi, prompt ngắn tập trung."""

    def __init__(self, client: LLMClient, n_examples: int = 2) -> None:
        self.client = client
        self.n_examples = n_examples
        self._add_schemas: dict[tuple[str, ...], type[BaseModel]] = {}
        self._cut_schema = make_cut_schema()

    @property
    def n_calls(self) -> int:
        return self.client.n_calls

    def _add_schema(self, relations: list[str]) -> type[BaseModel]:
        key = tuple(relations)
        if key not in self._add_schemas:
            self._add_schemas[key] = make_add_schema(relations)
        return self._add_schemas[key]

    def _cut_line(self, i: int, p: CutPattern) -> str:
        shape = f"({p.participant}, {p.r}, X)" if p.direction == "subject_first" else f"(X, {p.r}, {p.participant})"
        line = f"  [C{i}] {shape} — {len(p.members)} X: {_examples(p.hubs, self.n_examples)}"
        if p.co_anchors:
            line += "; X cũng nối với " + ", ".join(sorted(p.co_anchors))
        return line

    def _add_line(self, i: int, p: AddPattern) -> str:
        via = f"(X, {p.via_r}, {p.other})" if p.via_direction == "neighbor_first" else f"({p.other}, {p.via_r}, X)"
        return (f"  [A{i}] {p.subject} ~ X — {len(p.members)} X: {_examples(p.neighbors, self.n_examples)}; "
                f"bằng chứng: {via}")

    def _header(self, ctx: JudgeContext) -> list[str]:
        participants = sorted({e for op in ctx.explicit_ops for e in (op.h, op.t)})
        lines = ["BẢN TIN:", ctx.text, "", "THAY ĐỔI TƯỜNG MINH ĐÃ ÁP:"]
        lines += [f"  {o.kind.value} ({o.h}, {o.r}, {o.t})" for o in ctx.explicit_ops]
        lines += ["ENTITY THAM GIA: " + ", ".join(participants),
                  "QUAN HỆ CHO PHÉP: " + ", ".join(ctx.relations), ""]
        return lines

    def _prompt_cut(self, ctx: JudgeContext, cut: list[CutPattern]) -> str:
        lines = self._header(ctx) + ["MẪU CẮT (cạnh hiện có; mặc định KEEP):"]
        lines += [self._cut_line(i, p) for i, p in enumerate(cut)]
        lines += ["", "Trả về invalidate = các số i của [Ci] cần vô hiệu."]
        return "\n".join(lines)

    def _prompt_add(self, ctx: JudgeContext, add: list[AddPattern]) -> str:
        lines = self._header(ctx) + ["MẪU THÊM (cặp chưa nối trong bối cảnh mới; mặc định bỏ qua):"]
        lines += [self._add_line(i, p) for i, p in enumerate(add)]
        lines += ["", "Trả về add = các {idx, r, direction} cho [Ai] cần thêm."]
        return "\n".join(lines)

    def judge(self, ctx: JudgeContext) -> list[Op]:
        ops: list[Op] = []
        cut, add = group_cut(ctx.loc), group_add(ctx.loc)
        if cut:                                                          # pha 1: CẮT
            out = self.client.complete_json(SYSTEM_CUT, self._prompt_cut(ctx, cut), self._cut_schema)
            for i in out.invalidate:
                if 0 <= i < len(cut):
                    ops += [Op(OpKind.INVALIDATE, *c.triple, source="llm_judge") for c in cut[i].members]
        if add:                                                          # pha 2: SINH
            out = self.client.complete_json(SYSTEM_ADD, self._prompt_add(ctx, add), self._add_schema(ctx.relations))
            for v in out.add:
                if not (0 <= v.idx < len(add)) or v.r not in ctx.relations:   # r vẫn kiểm tra: đường json_object không ép enum
                    continue
                for c in add[v.idx].members:
                    h, t = (c.subject, c.neighbor) if v.direction == "subject_first" else (c.neighbor, c.subject)
                    if not ctx.graph.is_active((h, v.r, t), ctx.at):
                        ops.append(Op(OpKind.ADD, h, v.r, t, source="llm_judge"))
        return list(dict.fromkeys(ops))
