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

PHA SINH. Mỗi mẫu [Ai] "subject ~ X": subject vừa có cạnh mới tới một entity (nơi mới); X là các entity đang nối với nơi
mới qua fact ghi ở dòng "fact KG". Hỏi: subject có cần nối với X không, bằng quan hệ nào, chiều nào?
Trả về add = danh sách {idx, r, direction}; direction: subject_first = (subject, r, X); neighbor_first = (X, r, subject).
Triple (h, r, t) đọc là "h r t": (paypal, owned by, eBay) nghĩa là PayPal thuộc sở hữu của eBay.

- MỌI dòng dưới mẫu ("fact KG", "tương tự", "lựa chọn hợp lệ") là fact ĐÃ CÓ SẴN trong KG, đã được kiểm chứng. Chúng KHÔNG
  phải điều cần tìm trong bản tin. Thay đổi gián tiếp theo định nghĩa là điều bản tin KHÔNG nói ra: bản tin chỉ kể thương
  vụ/chuyển nhượng, không bao giờ liệt kê X. KHÔNG được bỏ qua mẫu với lý do "bản tin không nhắc tới X / không nhắc tới r".
- "tương tự": các entity S đã ở cùng vai với subject từ trước (cùng loại cạnh tới nơi mới) đang có cạnh gì với chính các
  X này, kèm tỉ lệ. Đó là QUY LUẬT đo được của KG: >= 50% S có cạnh (S, r, X) nghĩa là ai vào vai đó cũng có cạnh đó; 61%
  hay 100% đều là quy luật. Subject vừa vào cùng vai với S nên cần đúng các cạnh đó.
- "lựa chọn hợp lệ": các {r, direction} ứng với quy luật ấy. QUY TẮC: trả về CHÍNH các lựa chọn hợp lệ, đúng r và đúng
  direction; không thêm quan hệ khác dù nghe hợp lý, không đổi chiều, không trả 2 mục cho cùng lựa chọn. Mẫu không có
  lựa chọn hợp lệ ("không có cạnh S–X nào") → bỏ qua.
- CHIỀU THỪA HƯỞNG. Mỗi mẫu ghi dòng "vai": subject là CHỦ NGỮ/TÂN NGỮ của ADD (A, r0, B) và số cạnh của hai bên trong KG
  TRƯỚC thay đổi. Bên GIA NHẬP là bên MỚI: ít cạnh hơn hẳn (thường 0–vài cạnh), vừa được nối vào một bên đã có nhiều
  bối cảnh; bên nhiều cạnh là bên TIẾP NHẬN. Khi chênh lệch rõ, dòng "vai" đã kết luận sẵn "subject là bên GIA NHẬP" hay
  "bên TIẾP NHẬN": làm theo, chỉ đảo lại khi bản tin nói rõ ngược lại. Khi dòng "vai" ghi "chưa rõ", dùng nghĩa r0: hành
  động A làm lên B (mua lại, thu nhận, tuyển) → TÂN NGỮ B gia nhập A (bên mua không phải bên gia nhập); trạng thái A
  thuộc về B (chơi cho, làm việc tại) → CHỦ NGỮ A gia nhập B.
  Chỉ bên GIA NHẬP thừa hưởng hàng xóm của bên kia. Mẫu mà subject là bên TIẾP NHẬN → BỎ, dù tỉ lệ "tương tự" 100%: tỉ lệ
  đó chỉ phản chiếu phép thừa hưởng theo chiều ngược lại (hai bên đã chia sẻ hàng xóm nên nhìn từ phía nào cũng 100%).
- MẶC ĐỊNH LẤY mọi lựa chọn hợp lệ của mẫu mà subject là bên gia nhập. Chỉ BỎ khi bản tin cho thấy rõ: (a) thay đổi tường
  minh chưa thành hiện thực (mới dự định / đang cân nhắc / đã hủy), hoặc (b) nơi mới trong mẫu thực ra là nơi CŨ subject
  vừa rời (X không đi theo).
- reasoning: 1–2 câu: thay đổi đã hoàn tất chưa, subject là bên gia nhập hay tiếp nhận (theo dòng vai), lựa chọn nào
  lấy / bỏ.

VÍ DỤ (miền khác): tin "Messi rời Barca sang Inter", đã áp INVALIDATE (Messi, plays_for, Barca), ADD (Messi, plays_for, Inter).
  [A0] Messi ~ X — nhóm 2 X: Lautaro, Barella
    - vai: subject là CHỦ NGỮ của ADD (Messi, plays_for, Inter); số cạnh KG-trước: Messi 3, Inter 41 → subject là bên GIA NHẬP
    - fact KG: (X, plays_for, Inter)
    - tương tự: (S, teammate, X) 100%, (X, teammate, S) 100%
    - lựa chọn hợp lệ: {"r": "teammate", "direction": "subject_first"}; {"r": "teammate", "direction": "neighbor_first"}
  [A1] Messi ~ X — nhóm 1 X: Inzaghi
    - vai: subject là CHỦ NGỮ của ADD (Messi, plays_for, Inter); số cạnh KG-trước: Messi 3, Inter 41 → subject là bên GIA NHẬP
    - fact KG: (X, head_coach, Inter)
    - tương tự: không có cạnh S–X nào
  [A2] Inter ~ X — nhóm 2 X: Pedri, Gavi
    - vai: subject là TÂN NGỮ của ADD (Messi, plays_for, Inter); số cạnh KG-trước: Inter 41, Messi 3 → subject là bên TIẾP NHẬN
    - fact KG: (X, teammate, Messi)
    - tương tự: không có cạnh S–X nào
  [A3] Messi ~ X — nhóm 1 X: Zhang
    - vai: subject là CHỦ NGỮ của ADD (Messi, plays_for, Inter); số cạnh KG-trước: Messi 3, Inter 41 → subject là bên GIA NHẬP
    - fact KG: (Inter, owned_by, X)
    - tương tự: (S, sponsored_by, X) 67%
    - lựa chọn hợp lệ: {"r": "sponsored_by", "direction": "subject_first"}
  [A4] Inter ~ X — nhóm 1 X: Adidas
    - vai: subject là TÂN NGỮ của ADD (Messi, plays_for, Inter); số cạnh KG-trước: Inter 41, Messi 3 → subject là bên TIẾP NHẬN
    - fact KG: (Messi, sponsored_by, X)
    - tương tự: (S, sponsored_by, X) 100%
    - lựa chọn hợp lệ: {"r": "sponsored_by", "direction": "subject_first"}
  Kết quả đúng: {"reasoning": "Chuyển nhượng đã xong; Messi (3 cạnh) là bên gia nhập, Inter (41 cạnh) là bên tiếp nhận. A0: cầu thủ Inter đều là đồng đội 2 chiều (100%) → lấy cả 2 dù bản tin không nhắc Lautaro/Barella. A1, A2: không có lựa chọn → bỏ. A3: 67% cầu thủ Inter được chủ CLB tài trợ là quy luật → lấy sponsored_by subject_first, không thêm owned_by hay chiều ngược. A4: subject Inter là bên tiếp nhận, X là nhà tài trợ của người mới đến → bỏ dù 100%.",
                 "add": [{"idx": 0, "r": "teammate", "direction": "subject_first"},
                         {"idx": 0, "r": "teammate", "direction": "neighbor_first"},
                         {"idx": 3, "r": "sponsored_by", "direction": "subject_first"}]}
  Nếu tin là "Messi đang cân nhắc sang Inter" (chưa xong) → add rỗng."""

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
        self.n_off_menu = 0                  # số mục add LLM trả về ngoài "lựa chọn hợp lệ" (đo độ tuân thủ prompt)
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

    def _cooccur(self, r: str, r0: str, ctx: JudgeContext, max_edges: int = 2000) -> float:
        """Bằng chứng cấu trúc: tỷ lệ cặp (u, r, v) trong KG-trước mà u và v cùng chung một hàng xóm qua r0.
        Cao → quan hệ r đi kèm "cùng nhóm qua r0" (đồng đội ↔ cùng CLB); thấp → r độc lập với r0."""
        key = (r, r0, id(ctx.graph))
        if key in self._cooccur_cache:
            return self._cooccur_cache[key]
        g = ctx.graph
        edges = LLMJudge._rel_index(g).get(r, [])                        # KG-trước (at = 0), qua chỉ mục r → [(h, t)]
        if not edges:
            self._cooccur_cache[key] = 0.0
            return 0.0
        if len(edges) > max_edges:                                       # KG nền lớn: lấy mẫu đều, tái lập được
            edges = sorted(edges)[:: max(1, len(edges) // max_edges)][:max_edges]

        def nb_via(e: str) -> set[str]:
            return {tr[0] if tr[2] == e else tr[2] for tr in g.edges_of(e, 0) if tr[1] == r0}

        memo: dict[str, set[str]] = {}
        hit = 0
        for u, v in edges:
            nu = memo.setdefault(u, nb_via(u)) - {v}
            nv = memo.setdefault(v, nb_via(v)) - {u}
            hit += bool(nu & nv)
        self._cooccur_cache[key] = hit / len(edges)
        return self._cooccur_cache[key]

    def _add_block(self, i: int, p: AddPattern, ctx: JudgeContext) -> list[str]:
        via = f"(X, {p.via_r}, {p.other})" if p.via_direction == "neighbor_first" else f"({p.other}, {p.via_r}, X)"
        lines = [f"[A{i}] {p.subject} ~ X — nhóm {len(p.members)} X: {_examples(p.neighbors, self.n_examples)}"]
        role = self._role(p, ctx)
        if role is not None:                                            # vai của subject trong cạnh ADD nối subject–other
            op, d_s, d_o, side = role
            who = {"join": "subject là bên GIA NHẬP", "host": "subject là bên TIẾP NHẬN"}.get(side, "chưa rõ, xét nghĩa r0")
            lines.append(f"  - vai: subject là {'CHỦ NGỮ' if op.h == p.subject else 'TÂN NGỮ'} của ADD ({op.h}, {op.r}, {op.t}); "
                         f"số cạnh KG-trước: {p.subject} {d_s}, {p.other} {d_o} → {who}")
        lines.append(f"  - fact KG: {via}")
        analogy = self._analogy(p, ctx)
        if analogy:
            lines.append("  - tương tự: " + ", ".join(analogy))
        menu = self._menu(p, ctx)
        if menu:                                                        # thực đơn {r, direction} LLM được phép chọn
            lines.append("  - lựa chọn hợp lệ: " + "; ".join(f'{{"r": "{r}", "direction": "{d}"}}' for r, d in menu))
        return lines

    @staticmethod
    def _role(p: AddPattern, ctx: JudgeContext) -> tuple[Op, int, int, str] | None:
        """(op ADD nối subject–other, số cạnh KG-trước của subject, của other, phía): phía = "join" (subject là bên mới,
        ít cạnh hơn hẳn: một bên 0 cạnh hoặc chênh >= 5 lần), "host" (subject là bên tiếp nhận), "unclear". None = mẫu
        không gắn với cạnh ADD tường minh nào."""
        for op in ctx.explicit_ops:
            if op.kind is OpKind.ADD and {op.h, op.t} == {p.subject, p.other}:
                d_s, d_o = len(ctx.graph.edges_of(p.subject, 0)), len(ctx.graph.edges_of(p.other, 0))
                clear = (d_s == 0) != (d_o == 0) or max(d_s, d_o) >= 5 * max(1, min(d_s, d_o))
                side = "unclear" if not clear or d_s == d_o else "join" if d_s < d_o else "host"
                return op, d_s, d_o, side
        return None

    @staticmethod
    def _menu(p: AddPattern, ctx: JudgeContext) -> list[tuple[str, str]]:
        """Các (r, direction) có bằng chứng "tương tự" (>= 50% cặp S–X): tập lựa chọn hợp lệ cho LLM ở mẫu này."""
        return [(r, d) for r, d, _ in (LLMJudge._analogy_pairs(p, ctx) or [])]

    @staticmethod
    def _analogy(p: AddPattern, ctx: JudgeContext, max_siblings: int = 30, max_xs: int = 200) -> list[str]:
        """Bằng chứng cấu trúc từ KG hiện có: các entity S cùng vai với subject (cùng cạnh ADD tới `other`)
        đang có quan hệ gì với các X của mẫu. Không dùng luật miền: chỉ đếm cạnh đang tồn tại."""
        pairs = LLMJudge._analogy_pairs(p, ctx, max_siblings, max_xs)
        if pairs is None:
            return []
        return [f"(S, {r}, X) {pct}%" if d == "subject_first" else f"(X, {r}, S) {pct}%" for r, d, pct in pairs]             or ["không có cạnh S–X nào"]

    @staticmethod
    def _analogy_pairs(p: AddPattern, ctx: JudgeContext, max_siblings: int = 30,
                       max_xs: int = 200) -> list[tuple[str, str, int]] | None:
        """[(r, direction, %)] các quan hệ mà >= 50% cặp (S, X) đang có; None = không có S nào (không có bằng chứng)."""
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
        xs = p.neighbors[:max_xs]                                     # tỉ lệ trên mẫu X đầu (KG nền lớn)
        pairs = [(s, x) for s in siblings for x in xs if s != x]      # S có thể trùng X: chỉ bỏ cặp (s, s)
        total = len(pairs)
        if not total:
            return LLMJudge._global_rule(p, ctx)                      # không có S cùng vai → quy luật toàn KG
        xset = set(xs)
        hits: dict[tuple[str, str], int] = {}                           # (r, direction) → số cặp (S, X) có cạnh
        for s_ in siblings:                                               # duyệt cạnh của S thay vì |S|·|X|·|R| phép dò
            for h, r, t in g.edges_of(s_, at):
                if h == s_ and t in xset and t != s_:
                    hits[(r, "subject_first")] = hits.get((r, "subject_first"), 0) + 1
                elif t == s_ and h in xset and h != s_:
                    hits[(r, "neighbor_first")] = hits.get((r, "neighbor_first"), 0) + 1
        out = []
        for r in ctx.relations:
            for direction in ("subject_first", "neighbor_first"):
                hit = hits.get((r, direction), 0)
                if hit * 2 >= total:
                    out.append((r, direction, hit * 100 // total))
        return out

    _REL_INDEX: dict[int, tuple[int, dict[str, list[tuple[str, str]]]]] = {}   # id(graph) → (checkpoint, r → [(h, t)])

    @staticmethod
    def _rel_index(g) -> dict[str, list[tuple[str, str]]]:
        """Chỉ mục r → [(h, t)] của KG-trước (at = 0), dựng MỘT lần cho mỗi trạng thái đồ thị (khóa = checkpoint =
        độ dài nhật ký; đồ thị nền dùng chung đổi trạng thái mỗi bài → dựng lại). Không sắp; người dùng sắp theo r cần."""
        cp = g.checkpoint() if hasattr(g, "checkpoint") else -1
        hit = LLMJudge._REL_INDEX.get(id(g))
        if hit is not None and hit[0] == cp:
            return hit[1]
        idx: dict[str, list[tuple[str, str]]] = {}
        for h, r, t in g.active(0):
            idx.setdefault(r, []).append((h, t))
        LLMJudge._REL_INDEX[id(g)] = (cp, idx)
        return idx

    @staticmethod
    def _global_rule(p: AddPattern, ctx: JudgeContext, max_pairs: int = 100, max_xs: int = 20,
                     max_scan: int = 5000) -> list[tuple[str, str, int]] | None:
        """Dự phòng khi `other` chưa có entity S nào cùng vai với subject: đo QUY LUẬT TOÀN KG cho cùng dạng cấu hình.
        Với mọi cặp (A, r0, B) đang có trong KG (r0 = quan hệ của cạnh ADD tường minh nối subject–other, cùng chiều),
        lấy các X nối với A qua đúng fact-KG của mẫu (via_r, via_direction) và đếm tỉ lệ bộ (A, B, X) có cạnh (B, r, X) /
        (X, r, B). >= 50% → lựa chọn hợp lệ. Vẫn chỉ đếm cạnh đang tồn tại trong KG cho trước, không dùng luật miền.
        None = không có cặp (A, r0, B) nào có X → không có bằng chứng."""
        g, at = ctx.graph, ctx.at
        out: list[tuple[str, str, int]] = []
        seen_any = False
        for op in ctx.explicit_ops:
            if op.kind is not OpKind.ADD or {op.h, op.t} != {p.subject, p.other}:
                continue
            subj_is_head = op.h == p.subject                          # (subject, r0, other) hay (other, r0, subject)
            key = ("global", op.r, subj_is_head, p.via_r, p.via_direction, id(g))
            cache = getattr(ctx, "_rule_cache", None)
            if cache is None:
                cache = {}
                try:
                    ctx._rule_cache = cache
                except Exception:  # noqa: BLE001 — ctx không cho gán thuộc tính: bỏ cache
                    pass
            if key in cache:
                res = cache[key]
            else:
                triples = []                                          # (A, B, X): A đóng vai other, B đóng vai subject
                n_pairs = n_scanned = 0
                for h, t in sorted(LLMJudge._rel_index(g).get(op.r, ())):   # KG-trước (sắp để tái lập), bỏ cạnh vừa thêm
                    a, b = (t, h) if subj_is_head else (h, t)
                    if b == p.subject and a == p.other:
                        continue
                    n_scanned += 1
                    if n_pairs >= max_pairs or n_scanned > max_scan:  # chỉ cặp CÓ X mới tính vào mẫu
                        break
                    xs = [tr[2] if p.via_direction == "subject_first" else tr[0] for tr in g.edges_of(a, 0)
                          if tr[1] == p.via_r and (tr[0] == a if p.via_direction == "subject_first" else tr[2] == a)]
                    xs = [x for x in xs[:max_xs] if x != b]
                    if xs:
                        n_pairs += 1
                        triples += [(a, b, x) for x in xs]
                if not triples:
                    res = None
                else:
                    want: dict[str, set[str]] = {}                        # b → {x} cần kiểm; tỉ lệ trên cặp (B, X) PHÂN BIỆT
                    for _, b, x in triples:
                        want.setdefault(b, set()).add(x)
                    n_bx = sum(len(v) for v in want.values())
                    hits: dict[tuple[str, str], int] = {}
                    for b, xs_b in want.items():
                        for h, r, t in g.edges_of(b, 0):
                            if h == b and t in xs_b:
                                hits[(r, "subject_first")] = hits.get((r, "subject_first"), 0) + 1
                            elif t == b and h in xs_b:
                                hits[(r, "neighbor_first")] = hits.get((r, "neighbor_first"), 0) + 1
                    res = []
                    for r in ctx.relations:
                        for direction in ("subject_first", "neighbor_first"):
                            hit = hits.get((r, direction), 0)
                            if hit * 2 >= n_bx:
                                res.append((r, direction, hit * 100 // n_bx))
                cache[key] = res
            if res is not None:
                seen_any = True
                out += [o for o in res if o not in out]
        return out if seen_any else None

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
        lines = self._header(ctx) + ["MẪU THÊM (cặp chưa nối; lấy các lựa chọn hợp lệ trừ khi bản tin cho thấy thay đổi chưa xong):"]
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
        - thêm: có "tương tự" (entity cùng vai đang có cạnh với X) và subject không phải bên tiếp nhận rõ ràng
          (theo số cạnh KG-trước) → LLM phán; không → bỏ qua."""
        if self.gate is None:
            return cut, add
        keep_cut = [p for p in cut if self._cut_dependency(p, ctx) >= self.gate]
        # thêm: cần bằng chứng "tương tự" VÀ subject không phải bên TIẾP NHẬN rõ ràng (bên nhiều cạnh không thừa hưởng
        # hàng xóm của bên mới; tỉ lệ 100% ở đó chỉ phản chiếu phép thừa hưởng chiều ngược — LLM 4B vẫn lấy ~1/5 số mẫu này)
        keep_add = [p for p in add if any("%" in a for a in self._analogy(p, ctx))
                    and (self._role(p, ctx) or (None, 0, 0, "unclear"))[3] != "host"]
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
                menu = self._menu(batch[v.idx], ctx)
                if menu and (v.r, v.direction) not in menu:                       # LLM vi phạm "lựa chọn hợp lệ": đếm và BỎ
                    self.n_off_menu += 1                                           # (mẫu có bằng chứng → chỉ nhận đúng bằng chứng)
                    continue
                for c in batch[v.idx].members:
                    h, t = (c.subject, c.neighbor) if v.direction == "subject_first" else (c.neighbor, c.subject)
                    if not ctx.graph.is_active((h, v.r, t), ctx.at):
                        ops.append(Op(OpKind.ADD, h, v.r, t, source="llm_judge"))
        return list(dict.fromkeys(ops))
