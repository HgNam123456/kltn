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
    """Pha CẮT: `reasoning` ngắn (model suy luận trước), rồi chỉ số mẫu [Ci] cần INVALIDATE (mặc định KEEP)."""
    return create_model("CutOut", reasoning=(str, ""), invalidate=(list[int], []))


def make_add_schema(relations: list[str]) -> type[BaseModel]:
    """Pha SINH: `reasoning` ngắn, rồi các {idx, r, direction} cho mẫu [Ai] cần thêm. `r` là enum → constrained decoding."""
    return create_model("AddOut", reasoning=(str, ""), add=(list[_add_verdict_model(relations)], []))


def make_judge_schema(relations: list[str]) -> type[BaseModel]:
    """Schema gộp 1 pha (giữ để tương thích / so sánh)."""
    return create_model("JudgeOut", invalidate=(list[int], []), add=(list[_add_verdict_model(relations)], []))


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_COMMON = """Một bản tin đã được áp các thay đổi TƯỜNG MINH lên knowledge graph. Bạn phán các MẪU QUAN HỆ bị ảnh hưởng GIÁN TIẾP.
Mỗi mẫu gộp nhiều cạnh cùng dạng (X chạy trên một nhóm entity); phán quyết cho mẫu áp cho MỌI cạnh trong mẫu.
Mọi cạnh trong KG là TRẠNG THÁI HIỆN TẠI (đang là, đang thuộc), không phải lịch sử: cạnh không còn đúng phải bị vô hiệu.
Chỉ dùng quan hệ trong danh sách. Không đoán sự kiện chưa xảy ra."""

SYSTEM_CUT = _COMMON + """

PHA CẮT. Mỗi mẫu [Ci] được nêu bằng MỘT fact cụ thể (một X đại diện); hỏi: sau bản tin, fact đó còn đúng không?
Trả về invalidate = danh sách số i của các mẫu mà fact KHÔNG CÒN ĐÚNG. Không liệt kê = KEEP. Phán quyết áp cho cả nhóm X.
- X không tham gia thay đổi, X không đi đâu cả, X vẫn ở nguyên chỗ cũ. Mỗi dòng ghi rõ P (bên kia của fact)
  vừa tách khỏi Q; X vẫn nối với cả P lẫn Q trước khi phán.
- Fact "X r P" sai khi r chỉ có nghĩa lúc X và P cùng nhóm Q, mà P vừa tách khỏi Q (vd. đồng đội, cùng công ty).
- Fact "X r G" với G là nhóm của X KHÔNG sai chỉ vì G vừa tách khỏi một người khác (X vẫn thuộc G).
- "trong KG: N% cặp r cùng chung hàng xóm qua r0" là bằng chứng đo được: N cao (≈100%) → r đi kèm cùng nhóm
  qua r0, fact phụ thuộc cạnh vừa vô hiệu → thường sai. N thấp (≈0%) → r độc lập → thường giữ.
- Fact SỰ KIỆN LỊCH SỬ (đã ghi bàn, đã vô địch, sinh tại) LUÔN đúng.
- reasoning: 1–2 câu ngắn, trả lời từng fact rồi kết luận.

VÍ DỤ (miền khác): tin "Messi rời Barca sang Inter", đã áp INVALIDATE (Messi, plays_for, Barca), ADD (Messi, plays_for, Inter).
  [C0] "Pedri plays_for Barca" — đại diện cho nhóm 2 X: Pedri, Gavi
    - X = Pedri: không tham gia, không rời đi
    - Barca vừa tách khỏi Messi (qua plays_for)
    - trong KG: 0% cặp plays_for cùng chung hàng xóm qua plays_for
  [C1] "Messi teammate Pedri" — đại diện cho nhóm 2 X: Pedri, Gavi
    - X = Pedri: không tham gia, không rời đi
    - Messi vừa tách khỏi Barca (qua plays_for)
    - trong KG: 100% cặp teammate cùng chung hàng xóm qua plays_for
  [C2] "Pedri close_friend Messi" — đại diện cho nhóm 1 X: Messi
    - X = Messi: không tham gia, không rời đi
    - Messi vừa tách khỏi Barca (qua plays_for)
    - trong KG: 40% cặp close_friend cùng chung hàng xóm qua plays_for
  Kết quả đúng: {"reasoning": "C0 đúng: Pedri vẫn thuộc Barca, plays_for độc lập (0%). C1 sai: đồng đội luôn đi kèm cùng CLB (100%), Messi đã rời Barca. C2 đúng: bạn thân không phụ thuộc CLB (40%, không phải quy luật).",
                 "invalidate": [1]}"""

SYSTEM_ADD = _COMMON + """

PHA SINH. Trả về add = danh sách {idx, r, direction} cho các mẫu [Ai] mà cấu trúc mới THỰC SỰ hàm ý. Không liệt kê = bỏ qua.
- direction: subject_first = (subject, r, X); neighbor_first = (X, r, subject). Quan hệ đối xứng (đồng đội) → 2 mục cùng idx, 2 direction.
- Chỉ thêm cho entity vừa GIA NHẬP nơi mới: quan hệ với những X đang ở nơi đó (bằng chứng nối X với nơi mới).
  Mẫu mà bằng chứng nối X với entity CŨ / nơi cũ (X là đồng đội cũ) → bỏ qua: X không đi theo.
- Không thêm cạnh ngoài danh sách mẫu.
- "tương tự": trong KG hiện có, các entity S cùng vai với subject (cùng loại cạnh tới nơi mới) đang có quan hệ gì
  với X, kèm tỷ lệ. Subject vừa vào cùng vai với S nên thường cần đúng các cạnh đó; "không có cạnh S–X nào" → bỏ qua.
- reasoning: 1–2 câu ngắn giải thích trước khi kết luận.

VÍ DỤ (miền khác): tin "Messi rời Barca sang Inter", đã áp INVALIDATE (Messi, plays_for, Barca), ADD (Messi, plays_for, Inter).
  [A0] Messi ~ X — nhóm 2 X: Lautaro, Barella
    - bằng chứng: (X, plays_for, Inter)
    - tương tự: (S, teammate, X) 100%, (X, teammate, S) 100%
  [A1] Messi ~ X — nhóm 1 X: Inzaghi
    - bằng chứng: (X, head_coach, Inter)
    - tương tự: không có cạnh S–X nào
  [A2] Inter ~ X — nhóm 2 X: Pedri, Gavi
    - bằng chứng: (X, teammate, Messi)
    - tương tự: không có cạnh S–X nào
  Kết quả đúng: {"reasoning": "Messi vào Inter nên thành đồng đội 2 chiều với cầu thủ Inter như các cầu thủ khác; HLV không có quan hệ với cầu thủ trong KG; đồng đội cũ không sang Inter.",
                 "add": [{"idx": 0, "r": "teammate", "direction": "subject_first"},
                         {"idx": 0, "r": "teammate", "direction": "neighbor_first"}]}"""

SYSTEM = SYSTEM_CUT + "\n\n" + SYSTEM_ADD.split("\n\n", 1)[1]   # bản gộp 1 pha (tương thích)


def _examples(names: list[str], k: int = 2) -> str:
    shown = ", ".join(names[:k])
    return shown + (", ..." if len(names) > k else "")


class LLMJudge:
    """Phương án B, 2 pha: (1) CẮT — phán mẫu giữ/bỏ; (2) SINH — phán mẫu thêm.

    Mỗi pha chia mẫu thành lô `batch_size` mẫu / lệnh gọi (mặc định 1: mỗi lệnh gọi một mẫu, prompt ngắn nhất;
    chỉ số [Ci]/[Ai] là chỉ số CỤC BỘ trong lô). batch_size=None → cả pha trong một lệnh gọi.
    """

    def __init__(self, client: LLMClient, n_examples: int = 2, batch_size: int | None = 1,
                 gate: float | None = None) -> None:
        self.client = client
        self.n_examples = n_examples
        self.batch_size = batch_size
        self.gate = gate                     # phương án C (lai): ngưỡng đồng xuất hiện; None = hỏi LLM mọi mẫu
        self.n_gated = 0                     # số mẫu bị cổng cấu trúc KEEP/bỏ qua mà không hỏi LLM
        self._add_schemas: dict[tuple[str, ...], type[BaseModel]] = {}
        self._cut_schema = make_cut_schema()
        self._cooccur_cache: dict[tuple, float] = {}

    @property
    def n_calls(self) -> int:
        return self.client.n_calls

    def _add_schema(self, relations: list[str]) -> type[BaseModel]:
        key = tuple(relations)
        if key not in self._add_schemas:
            self._add_schemas[key] = make_add_schema(relations)
        return self._add_schemas[key]

    # ---- render một mẫu thành khối ngắn -------------------------------------------------------

    @staticmethod
    def _detached(p: CutPattern, ctx: JudgeContext) -> list[tuple[str, str]]:
        """(Q, r0): P vừa tách khỏi Q qua r0 — các INVALIDATE tường minh có P, với Q trong co_anchors của mẫu."""
        cuts = sorted({(o.t if o.h == p.participant else o.h, o.r) for o in ctx.explicit_ops
                       if o.kind is OpKind.INVALIDATE and p.participant in (o.h, o.t)})
        return [(q, r0) for q, r0 in cuts if q in p.co_anchors]

    def _cut_dependency(self, p: CutPattern, ctx: JudgeContext) -> float:
        """Mức phụ thuộc cấu trúc của mẫu cắt vào thay đổi: max đồng xuất hiện (r, r0) trên các cạnh vừa vô hiệu."""
        r0s = {r0 for _, r0 in self._detached(p, ctx)}
        return max((self._cooccur(p.r, r0, ctx) for r0 in r0s), default=0.0)

    def _cut_block(self, i: int, p: CutPattern, ctx: JudgeContext) -> list[str]:
        x0 = p.hubs[0]                                                  # một X đại diện → fact cụ thể
        fact = f"{p.participant} {p.r} {x0}" if p.direction == "subject_first" else f"{x0} {p.r} {p.participant}"
        lines = [f'[C{i}] "{fact}" — đại diện cho nhóm {len(p.members)} X: {_examples(p.hubs, self.n_examples)}',
                 f"  - X = {x0}: không tham gia, không rời đi"]
        cuts = self._detached(p, ctx)
        for q, r0 in cuts:
            lines.append(f"  - {p.participant} vừa tách khỏi {q} (qua {r0})")
        for r0 in sorted({r0 for _, r0 in cuts}):
            lines.append(f"  - trong KG: {self._cooccur(p.r, r0, ctx) * 100:.0f}% cặp {p.r} cùng chung hàng xóm qua {r0}")
        return lines

    def _cooccur(self, r: str, r0: str, ctx: JudgeContext) -> float:
        """Bằng chứng cấu trúc: tỷ lệ cặp (u, r, v) trong KG-trước mà u và v cùng chung một hàng xóm qua r0.
        Cao → quan hệ r đi kèm "cùng nhóm qua r0" (đồng đội ↔ cùng CLB); thấp → r độc lập với r0."""
        key = (r, r0, id(ctx.graph))
        if key in self._cooccur_cache:
            return self._cooccur_cache[key]
        g = ctx.graph
        edges = [tr for tr in g.active(0) if tr[1] == r]                 # KG-trước (at = 0)
        if not edges:
            self._cooccur_cache[key] = 0.0
            return 0.0

        def nb_via(e: str) -> set[str]:
            return {tr[0] if tr[2] == e else tr[2] for tr in g.edges_of(e, 0) if tr[1] == r0}

        memo: dict[str, set[str]] = {}
        hit = 0
        for u, _, v in edges:
            nu = memo.setdefault(u, nb_via(u)) - {v}
            nv = memo.setdefault(v, nb_via(v)) - {u}
            hit += bool(nu & nv)
        self._cooccur_cache[key] = hit / len(edges)
        return self._cooccur_cache[key]

    def _add_block(self, i: int, p: AddPattern, ctx: JudgeContext) -> list[str]:
        via = f"(X, {p.via_r}, {p.other})" if p.via_direction == "neighbor_first" else f"({p.other}, {p.via_r}, X)"
        lines = [f"[A{i}] {p.subject} ~ X — nhóm {len(p.members)} X: {_examples(p.neighbors, self.n_examples)}",
                 f"  - bằng chứng: {via}"]
        analogy = self._analogy(p, ctx)
        if analogy:
            lines.append("  - tương tự: " + ", ".join(analogy))
        return lines

    @staticmethod
    def _analogy(p: AddPattern, ctx: JudgeContext, max_siblings: int = 30) -> list[str]:
        """Bằng chứng cấu trúc từ KG hiện có: các entity S cùng vai với subject (cùng cạnh ADD tới `other`)
        đang có quan hệ gì với các X của mẫu. Không dùng luật miền: chỉ đếm cạnh đang tồn tại."""
        g, at = ctx.graph, ctx.at
        siblings: set[str] = set()
        for op in ctx.explicit_ops:
            if op.kind is not OpKind.ADD:
                continue
            if op.h == p.subject and op.t == p.other:
                siblings |= {tr[0] for tr in g.edges_of(p.other, at) if tr[1] == op.r and tr[2] == p.other}
            elif op.t == p.subject and op.h == p.other:
                siblings |= {tr[2] for tr in g.edges_of(p.other, at) if tr[1] == op.r and tr[0] == p.other}
        siblings.discard(p.subject)
        siblings = set(sorted(siblings)[:max_siblings])
        xs = p.neighbors
        pairs = [(s, x) for s in siblings for x in xs if s != x]      # S có thể trùng X: chỉ bỏ cặp (s, s)
        total = len(pairs)
        if not total:
            return []
        out = []
        for r in ctx.relations:
            for direction, mk in (("subject_first", lambda s, x: (s, r, x)), ("neighbor_first", lambda s, x: (x, r, s))):
                hit = sum(1 for s, x in pairs if g.is_active(mk(s, x), at))
                if hit * 2 >= total:
                    shape = f"(S, {r}, X)" if direction == "subject_first" else f"(X, {r}, S)"
                    out.append(f"{shape} {hit * 100 // total}%")
        return out or ["không có cạnh S–X nào"]

    # ---- prompt -------------------------------------------------------------------------------

    def _header(self, ctx: JudgeContext) -> list[str]:
        participants = sorted({e for op in ctx.explicit_ops for e in (op.h, op.t)})
        lines = ["BẢN TIN:", ctx.text, "", "THAY ĐỔI TƯỜNG MINH ĐÃ ÁP:"]
        lines += [f"  {o.kind.value} ({o.h}, {o.r}, {o.t})" for o in ctx.explicit_ops]
        lines += ["ENTITY THAM GIA: " + ", ".join(participants),
                  "QUAN HỆ CHO PHÉP: " + ", ".join(ctx.relations), ""]
        return lines

    def _prompt_cut(self, ctx: JudgeContext, cut: list[CutPattern]) -> str:
        lines = self._header(ctx) + ["MẪU CẮT (mỗi mẫu một fact đại diện; phán quyết áp cho cả nhóm X; mặc định KEEP):"]
        for i, p in enumerate(cut):
            lines += self._cut_block(i, p, ctx)
        lines += ["", "Sau bản tin, fact nào KHÔNG còn đúng? Trả về invalidate = các số i của [Ci] đó."]
        return "\n".join(lines)

    def _prompt_add(self, ctx: JudgeContext, add: list[AddPattern]) -> str:
        lines = self._header(ctx) + ["MẪU THÊM (cặp chưa nối trong bối cảnh mới; mặc định bỏ qua):"]
        for i, p in enumerate(add):
            lines += self._add_block(i, p, ctx)
        lines += ["", "Trả về add = các {idx, r, direction} cho [Ai] cần thêm."]
        return "\n".join(lines)

    def _batches(self, items: list):
        if not items:
            return []
        k = self.batch_size or len(items)
        return [items[i:i + k] for i in range(0, len(items), k)]

    # ---- phán ---------------------------------------------------------------------------------

    def _gated(self, cut: list[CutPattern], add: list[AddPattern], ctx: JudgeContext):
        """Cổng cấu trúc (phương án C): chỉ đẩy lên LLM các mẫu mà KG cho thấy CÓ THỂ bị ảnh hưởng.
        - cắt: đồng xuất hiện (r, r0) >= gate → quan hệ r đi kèm "cùng nhóm qua r0" → LLM phán; dưới → KEEP.
        - thêm: có "tương tự" (entity cùng vai đang có cạnh với X) → LLM phán; không có → bỏ qua."""
        if self.gate is None:
            return cut, add
        keep_cut = [p for p in cut if self._cut_dependency(p, ctx) >= self.gate]
        keep_add = [p for p in add if any("%" in a for a in self._analogy(p, ctx))]
        self.n_gated += (len(cut) - len(keep_cut)) + (len(add) - len(keep_add))
        return keep_cut, keep_add

    def judge(self, ctx: JudgeContext) -> list[Op]:
        ops: list[Op] = []
        cut, add = self._gated(group_cut(ctx.loc), group_add(ctx.loc), ctx)
        for batch in self._batches(cut):                                 # pha 1: CẮT
            out = self.client.complete_json(SYSTEM_CUT, self._prompt_cut(ctx, batch), self._cut_schema)
            for i in out.invalidate:
                if 0 <= i < len(batch):
                    ops += [Op(OpKind.INVALIDATE, *c.triple, source="llm_judge") for c in batch[i].members]
        for batch in self._batches(add):                                 # pha 2: SINH
            out = self.client.complete_json(SYSTEM_ADD, self._prompt_add(ctx, batch), self._add_schema(ctx.relations))
            for v in out.add:
                if not (0 <= v.idx < len(batch)) or v.r not in ctx.relations:   # r vẫn kiểm tra: đường json_object không ép enum
                    continue
                for c in batch[v.idx].members:
                    h, t = (c.subject, c.neighbor) if v.direction == "subject_first" else (c.neighbor, c.subject)
                    if not ctx.graph.is_active((h, v.r, t), ctx.at):
                        ops.append(Op(OpKind.ADD, h, v.r, t, source="llm_judge"))
        return list(dict.fromkeys(ops))
