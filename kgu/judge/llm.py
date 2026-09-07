from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from kgu.judge import JudgeContext
from kgu.llm import LLMClient
from kgu.types import Op, OpKind


class CutVerdict(BaseModel):
    idx: int
    verdict: Literal["KEEP", "INVALIDATE"]


class AddVerdict(BaseModel):
    idx: int
    add: bool
    r: str = ""
    direction: Literal["subject_first", "neighbor_first"] = "subject_first"


class JudgeOut(BaseModel):
    cut: list[CutVerdict] = []
    add: list[AddVerdict] = []


SYSTEM = """Bạn là bộ phán cập nhật knowledge graph. Một bản tin đã được áp các thay đổi TƯỜNG MINH.
Bây giờ hãy phán từng ứng viên bị ảnh hưởng GIÁN TIẾP:
- Ứng viên CẮT [Ci]: cạnh hiện có. Trả KEEP nếu vẫn đúng, INVALIDATE nếu cấu trúc mới làm nó không còn đúng.
  Fact TRẠNG THÁI (đang chơi cho, là đồng đội, là HLV của) có thể bị vô hiệu.
  Fact SỰ KIỆN LỊCH SỬ (đã ghi bàn, đã vô địch, sinh tại) LUÔN KEEP.
- Ứng viên THÊM [Ai]: cặp (subject, neighbor) trong bối cảnh mới. Trả add=true kèm quan hệ r và chiều
  (subject_first = (subject, r, neighbor); neighbor_first = (neighbor, r, subject)) CHỈ KHI cấu trúc mới
  thực sự hàm ý cạnh đó theo cùng mẫu với các cạnh bằng chứng. Ngược lại add=false.
- Không đoán sự kiện chưa xảy ra. Không thêm cạnh ngoài danh sách ứng viên. Chỉ dùng quan hệ trong danh sách."""


class LLMJudge:
    """Phương án A: 1 lệnh gọi phán toàn bộ danh sách ứng viên."""

    def __init__(self, client: LLMClient) -> None:
        self.client = client

    @property
    def n_calls(self) -> int:
        return self.client.n_calls

    def _prompt(self, ctx: JudgeContext) -> str:
        lines = [f"BẢN TIN:\n{ctx.text}", "", "THAY ĐỔI TƯỜNG MINH ĐÃ ÁP:"]
        lines += [f"  {o.kind.value} ({o.h}, {o.r}, {o.t})" for o in ctx.explicit_ops]
        lines += ["", "QUAN HỆ CHO PHÉP: " + ", ".join(ctx.relations), "", "ỨNG VIÊN CẮT:"]
        for i, c in enumerate(ctx.loc.cut):
            lines.append(f"  [C{i}] ({c.triple[0]}, {c.triple[1]}, {c.triple[2]})  hub={c.hub} anchors={sorted(c.anchors)}")
        lines += ["", "ỨNG VIÊN THÊM:"]
        for i, c in enumerate(ctx.loc.add):
            lines.append(f"  [A{i}] subject={c.subject} neighbor={c.neighbor} via=({c.via[0]}, {c.via[1]}, {c.via[2]})")
        lines += ["", "Phán từng ứng viên. cut[].idx là số i của [Ci]; add[].idx là số i của [Ai]."]
        return "\n".join(lines)

    def judge(self, ctx: JudgeContext) -> list[Op]:
        if not ctx.loc.cut and not ctx.loc.add:
            return []
        out = self.client.complete_json(SYSTEM, self._prompt(ctx), JudgeOut)
        ops: list[Op] = []
        for v in out.cut:
            if v.verdict == "INVALIDATE" and 0 <= v.idx < len(ctx.loc.cut):
                ops.append(Op(OpKind.INVALIDATE, *ctx.loc.cut[v.idx].triple, source="llm_judge"))
        for v in out.add:
            if not v.add or not (0 <= v.idx < len(ctx.loc.add)) or v.r not in ctx.relations:
                continue
            c = ctx.loc.add[v.idx]
            h, t = (c.subject, c.neighbor) if v.direction == "subject_first" else (c.neighbor, c.subject)
            if not ctx.graph.is_active((h, v.r, t), ctx.at):
                ops.append(Op(OpKind.ADD, h, v.r, t, source="llm_judge"))
        return list(dict.fromkeys(ops))
