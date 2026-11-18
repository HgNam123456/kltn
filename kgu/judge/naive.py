"""Hệ so sánh "lấy cạnh 1 bước rồi để LLM tự quyết" (kiểu DYNAMO / Graphiti) cho thay đổi gián tiếp.

LLM đọc bản tin, các thay đổi trực tiếp đã áp và các triple 1 bước quanh thực thể của những thay đổi đó, rồi tự liệt kê
triple nào phải vô hiệu và triple mới nào phải thêm. KHÔNG dùng khoanh vùng (ctx.loc), không gom mẫu, không đếm bằng
chứng cấu trúc trên KG, không có bộ lọc cấu trúc. Mỗi bản tin đúng một lệnh gọi LLM.
"""
from __future__ import annotations

from collections import Counter
from typing import Literal

from pydantic import BaseModel, create_model

from kgu.judge import JudgeContext
from kgu.llm import LLMClient
from kgu.types import Op, OpKind, Triple

SOURCE = "naive_1hop"

SYSTEM = """You maintain a knowledge graph (KG) of facts (h, r, t), read as "h r t". Every stored fact is a CURRENT state, not history: a fact that is no longer true must be invalidated.
A news text has arrived and its DIRECT changes (the facts it states) have already been applied to the KG. You are given the news text, the direct changes, and a numbered list of stored facts around the involved entities.
Find the INDIRECT changes: consequences of the direct changes that the news does not state. Return:
- invalidate: the numbers i of the stored facts [i] that are no longer true because of the news.
- add: new facts {h, r, t} that become true because of the news but are not stated in it.

Rules:
- Only changes that follow from the direct changes. A stored fact that the direct changes do not affect stays as it is. The default is to change nothing: empty lists are a valid answer.
- Do not repeat the direct changes. Do not add a fact that is already stored.
- In add, h and t are entity names copied exactly from the input (involved entities or entities in the stored facts); r is one of the allowed relations.
- Do not guess events that have not happened (plans, rumours, talks).
- reason: 1-2 short sentences, written before the two lists.

EXAMPLE (another domain). News: "Lan moves from the Sales department to the Research department."
DIRECT CHANGES ALREADY APPLIED:
  INVALIDATE (Lan, works_in, Sales)
  ADD (Lan, works_in, Research)
INVOLVED ENTITIES: Lan, Research, Sales
STORED FACTS:
  [0] (Lan, colleague_of, Minh)
  [1] (Minh, colleague_of, Lan)
  [2] (Minh, works_in, Sales)
  [3] (Lan, lives_in, Hanoi)
  [4] (Tuan, works_in, Research)
  [5] (Tuan, manages, Research)
Correct answer: {"reason": "Lan left Sales, so she and Minh, who stays in Sales, are no longer colleagues. She joined Research, where Tuan works, so they become colleagues. Where Lan lives and who else works in each department do not change.",
                 "invalidate": [0, 1],
                 "add": [{"h": "Lan", "r": "colleague_of", "t": "Tuan"}, {"h": "Tuan", "r": "colleague_of", "t": "Lan"}]}
If the news were "Lan is considering a move to Research" (not done yet), both lists would be empty."""


def make_naive_schema(relations: list[str]) -> type[BaseModel]:
    """{"reason": str, "invalidate": [int], "add": [{"h", "r", "t"}]}; `r` là enum của danh sách quan hệ → constrained decoding."""
    rel_type = Literal[tuple(relations)] if relations else str  # type: ignore[valid-type]
    fact = create_model("NaiveFact", h=(str, ...), r=(rel_type, ...), t=(str, ...))
    return create_model("NaiveOut", reason=(str, ""), invalidate=(list[int], []), add=(list[fact], []))


def _even_sample(items: list, k: int) -> list:
    """k phần tử cách đều của danh sách đã sắp (tái lập được); k >= len → cả danh sách."""
    n = len(items)
    if k >= n:
        return list(items)
    return [items[(i * n) // k] for i in range(k)]


class NaiveOneHopJudge:
    """Bộ phán so sánh: một lệnh gọi LLM mỗi bản tin trên các triple 1 bước quanh thực thể tham gia.

    `max_triples`: số triple tối đa đưa vào prompt; vượt → mỗi thực thể tham gia một phần bằng nhau (phần thừa của thực
    thể ít cạnh chia lại cho thực thể nhiều cạnh), trong mỗi phần lấy mẫu cách đều trên danh sách đã sắp.
    `max_relation_chars`: danh sách quan hệ dài hơn ngưỡng này (vd. ICEWS 230 quan hệ) không lặp lại trong prompt người
    dùng — client đã chép schema (có enum quan hệ) vào prompt hệ thống, lặp lại sẽ tốn ~1,6k token trong ô 8,192.
    """

    def __init__(self, client: LLMClient, max_triples: int = 80, max_relation_chars: int = 1500) -> None:
        self.client = client
        self.max_triples = max_triples
        self.max_relation_chars = max_relation_chars
        self.n_gated = 0                     # không có cổng cấu trúc; giữ để summary của run_eval dùng chung
        self.n_off_menu = 0                  # không có "lựa chọn hợp lệ"; giữ cùng lý do
        self.n_dropped = 0                   # mục LLM trả về bị loại ở bước hậu kiểm
        self.n_errors = 0                    # lệnh gọi lỗi (ngoại lệ từ client) → bản tin đó không có thay đổi gián tiếp
        self.drop_reasons: Counter[str] = Counter()
        self.n_items = 0                     # số bản tin đã hỏi LLM
        self.n_listed = 0                    # tổng số triple đã đưa vào prompt
        self.n_truncated = 0                 # số bản tin có nhiều triple 1 bước hơn max_triples (đã lấy mẫu)
        self._schemas: dict[tuple[str, ...], type[BaseModel]] = {}

    @property
    def n_calls(self) -> int:
        return self.client.n_calls

    def _schema(self, relations: list[str]) -> type[BaseModel]:
        key = tuple(relations)
        if key not in self._schemas:
            self._schemas[key] = make_naive_schema(relations)
        return self._schemas[key]

    # ---- ngữ cảnh 1 bước ----------------------------------------------------------------------

    @staticmethod
    def participants(ctx: JudgeContext) -> list[str]:
        return sorted({e for op in ctx.explicit_ops for e in (op.h, op.t)})

    def context_triples(self, ctx: JudgeContext) -> tuple[list[Triple], int]:
        """→ (triple đưa vào prompt, tổng số triple 1 bước). Chỉ dùng chỉ mục theo thực thể của đồ thị (edges_of),
        không quét toàn bộ cạnh. Triple nối hai thực thể tham gia chỉ tính cho thực thể đứng trước (theo tên)."""
        explicit = {op.triple for op in ctx.explicit_ops}
        seen: set[Triple] = set()
        per: list[list[Triple]] = []
        for p in self.participants(ctx):
            own = sorted(tr for tr in ctx.graph.edges_of(p, ctx.at) if tr not in explicit and tr not in seen)
            seen.update(own)
            per.append(own)
        total = len(seen)
        if total <= self.max_triples:
            return [tr for own in per for tr in own], total
        quota = [0] * len(per)                                          # chia đều, phần thừa chia lại
        left = self.max_triples
        order = sorted(range(len(per)), key=lambda i: (len(per[i]), i))  # ít cạnh trước: lấy hết rồi nhường phần thừa
        for n_done, i in enumerate(order):
            quota[i] = min(len(per[i]), left // (len(order) - n_done))
            left -= quota[i]
        return [tr for own, q in zip(per, quota) for tr in _even_sample(own, q)], total

    # ---- prompt -------------------------------------------------------------------------------

    def build_prompt(self, ctx: JudgeContext, listed: list[Triple], total: int) -> str:
        lines = ["NEWS:", ctx.text, "", "DIRECT CHANGES ALREADY APPLIED:"]
        lines += [f"  {o.kind.value} ({o.h}, {o.r}, {o.t})" for o in ctx.explicit_ops]
        lines.append("INVOLVED ENTITIES: " + ", ".join(self.participants(ctx)))
        if sum(len(r) for r in ctx.relations) <= self.max_relation_chars:
            lines.append("ALLOWED RELATIONS: " + ", ".join(ctx.relations))
        else:
            lines.append(f"ALLOWED RELATIONS: the {len(ctx.relations)} relation names in the enum of field r of the output schema")
        shown = f"{len(listed)} of {total} shown, evenly sampled" if len(listed) < total else f"{len(listed)}"
        lines += ["", f"STORED FACTS around the involved entities, after the direct changes ({shown}):"]
        lines += [f"  [{i}] ({h}, {r}, {t})" for i, (h, r, t) in enumerate(listed)] or ["  (none)"]
        lines += ["", "Which stored facts are no longer true, and which new facts become true, because of the news? "
                      "Return invalidate = the numbers i of those [i]; add = the new {h, r, t}. Empty lists if nothing changes."]
        return "\n".join(lines)

    # ---- phán ---------------------------------------------------------------------------------

    def _drop(self, reason: str) -> None:
        self.n_dropped += 1
        self.drop_reasons[reason] += 1

    def judge(self, ctx: JudgeContext) -> list[Op]:
        if not ctx.explicit_ops:                                        # không có thay đổi trực tiếp → không có gì để suy ra
            return []
        listed, total = self.context_triples(ctx)
        self.n_items += 1
        self.n_listed += len(listed)
        self.n_truncated += total > len(listed)
        try:
            out = self.client.complete_json(SYSTEM, self.build_prompt(ctx, listed, total), self._schema(ctx.relations))
        except Exception:  # noqa: BLE001 — một lệnh gọi hỏng (mạng, tràn ngữ cảnh, JSON sai sau các lần thử lại) không làm dừng cả lượt chạy
            self.n_errors += 1
            return []

        ops: list[Op] = []
        for i in dict.fromkeys(out.invalidate):                         # lặp chỉ số: gộp, không tính là loại
            if 0 <= i < len(listed):
                ops.append(Op(OpKind.INVALIDATE, *listed[i], source=SOURCE))
            else:
                self._drop("bad_index")

        known = set(self.participants(ctx)) | {e for h, _, t in listed for e in (h, t)}
        folded: dict[str, str] = {}                                     # tên viết khác hoa/thường, thừa khoảng trắng → tên gốc
        for e in sorted(known):
            folded.setdefault(e.strip().casefold(), e)
        relations = set(ctx.relations)

        def resolve(name: str) -> str | None:
            return name if name in known else folded.get(name.strip().casefold())

        for v in out.add:
            h, t = resolve(v.h), resolve(v.t)
            if h is None or t is None:
                self._drop("unknown_entity")
            elif h == t:
                self._drop("self_loop")
            elif v.r not in relations:                                  # r vẫn kiểm tra: đường json_object không ép enum
                self._drop("unknown_relation")
            elif ctx.graph.is_active((h, v.r, t), ctx.at):
                self._drop("already_active")
            else:
                ops.append(Op(OpKind.ADD, h, v.r, t, source=SOURCE))
        return list(dict.fromkeys(ops))
