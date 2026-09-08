from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, create_model

from kgu.judge import JudgeContext
from kgu.llm import LLMClient
from kgu.types import Op, OpKind


def make_judge_schema(relations: list[str]) -> type[BaseModel]:
    """Output chỉ liệt kê NGOẠI LỆ: chỉ số cần INVALIDATE, và các ADD (mặc định KEEP / bỏ qua).

    `r` là enum quan hệ → constrained decoding. Output ngắn (vài chục token) thay vì một verdict/ứng viên.
    """
    rel_type = Literal[tuple(relations)] if relations else str  # type: ignore[valid-type]
    add_verdict = create_model(
        "AddVerdict",
        idx=(int, ...), r=(rel_type, ...),
        direction=(Literal["subject_first", "neighbor_first"], "subject_first"),
    )
    return create_model("JudgeOut", invalidate=(list[int], []), add=(list[add_verdict], []))


SYSTEM = """Bạn là bộ phán cập nhật knowledge graph. Một bản tin đã được áp các thay đổi TƯỜNG MINH.
Hãy phán các ứng viên bị ảnh hưởng GIÁN TIẾP. Chỉ trả về NGOẠI LỆ:
- invalidate: danh sách số i của các ứng viên cắt [Ci] KHÔNG CÒN ĐÚNG sau thay đổi. Không liệt kê = KEEP.
  Fact TRẠNG THÁI (đang chơi cho, là đồng đội, là HLV của) có thể bị vô hiệu.
  Fact SỰ KIỆN LỊCH SỬ (đã ghi bàn, đã vô địch, sinh tại) LUÔN KEEP.
- add: danh sách {idx, r, direction} cho các ứng viên thêm [Ai] mà cấu trúc mới THỰC SỰ hàm ý, theo cùng mẫu
  với cạnh bằng chứng. direction: subject_first = (subject, r, neighbor); neighbor_first = (neighbor, r, subject).
  Quan hệ đối xứng (đồng đội) → trả 2 mục cùng idx với 2 direction. Không liệt kê = bỏ qua.
- Không đoán sự kiện chưa xảy ra. Không thêm cạnh ngoài danh sách ứng viên. Chỉ dùng quan hệ trong danh sách.

VÍ DỤ (miền khác): tin "Messi rời Barca sang Inter", đã áp INVALIDATE (Messi, plays_for, Barca) và ADD (Messi, plays_for, Inter).
  Ứng viên cắt: Pedri: [C0] (Pedri, plays_for, Barca) [C1] (Pedri, teammate, Messi) [C2] (Pedri, close_friend, Messi)
  Ứng viên thêm: [A0] Messi ~ Lautaro (bằng chứng: (Lautaro, plays_for, Inter))
  Kết quả đúng: {"invalidate": [1], "add": [{"idx": 0, "r": "teammate", "direction": "subject_first"},
                                            {"idx": 0, "r": "teammate", "direction": "neighbor_first"}]}
  (C0 giữ: Pedri vẫn ở Barca. C2 giữ: bạn thân không phụ thuộc CLB. C1 cắt: đồng đội chỉ đúng khi cùng CLB.)"""


class LLMJudge:
    """Phương án A: 1 lệnh gọi phán toàn bộ danh sách ứng viên."""

    def __init__(self, client: LLMClient) -> None:
        self.client = client
        self._schemas: dict[tuple[str, ...], type[BaseModel]] = {}

    @property
    def n_calls(self) -> int:
        return self.client.n_calls

    def _schema(self, relations: list[str]) -> type[BaseModel]:
        key = tuple(relations)
        if key not in self._schemas:
            self._schemas[key] = make_judge_schema(relations)
        return self._schemas[key]

    def _prompt(self, ctx: JudgeContext) -> str:
        participants = sorted({e for op in ctx.explicit_ops for e in (op.h, op.t)})
        lines = ["BẢN TIN:", ctx.text, "", "THAY ĐỔI TƯỜNG MINH ĐÃ ÁP:"]
        lines += [f"  {o.kind.value} ({o.h}, {o.r}, {o.t})" for o in ctx.explicit_ops]
        lines += ["ENTITY THAM GIA: " + ", ".join(participants),
                  "QUAN HỆ CHO PHÉP: " + ", ".join(ctx.relations), "",
                  "ỨNG VIÊN CẮT (cạnh hiện có, gom theo entity trung gian; mặc định KEEP):"]
        by_hub: dict[str, list[str]] = {}
        for i, c in enumerate(ctx.loc.cut):
            by_hub.setdefault(c.hub, []).append(f"[C{i}] ({c.triple[0]}, {c.triple[1]}, {c.triple[2]})")
        lines += [f"  {hub}: " + " ".join(items) for hub, items in by_hub.items()]
        lines += ["", "ỨNG VIÊN THÊM (cặp chưa nối trong bối cảnh mới; mặc định bỏ qua):"]
        lines += [f"  [A{i}] {c.subject} ~ {c.neighbor}  (bằng chứng: ({c.via[0]}, {c.via[1]}, {c.via[2]}))"
                  for i, c in enumerate(ctx.loc.add)]
        lines += ["", "Trả về invalidate = các số i của [Ci] cần vô hiệu; add = các {idx, r, direction} cho [Ai] cần thêm."]
        return "\n".join(lines)

    def judge(self, ctx: JudgeContext) -> list[Op]:
        if not ctx.loc.cut and not ctx.loc.add:
            return []
        out = self.client.complete_json(SYSTEM, self._prompt(ctx), self._schema(ctx.relations))
        ops: list[Op] = []
        for i in out.invalidate:
            if 0 <= i < len(ctx.loc.cut):
                ops.append(Op(OpKind.INVALIDATE, *ctx.loc.cut[i].triple, source="llm_judge"))
        for v in out.add:
            if not (0 <= v.idx < len(ctx.loc.add)) or v.r not in ctx.relations:   # r vẫn kiểm tra: đường json_object không ép enum
                continue
            c = ctx.loc.add[v.idx]
            h, t = (c.subject, c.neighbor) if v.direction == "subject_first" else (c.neighbor, c.subject)
            if not ctx.graph.is_active((h, v.r, t), ctx.at):
                ops.append(Op(OpKind.ADD, h, v.r, t, source="llm_judge"))
        return list(dict.fromkeys(ops))
