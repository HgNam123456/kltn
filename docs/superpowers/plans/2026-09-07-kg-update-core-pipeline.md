# KG Update Core Pipeline (NBAtransactions, tháng 1–2) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Dựng pipeline "ops tường minh → executor bi-temporal → khoanh vùng 2 vế → bộ phán → đo so với KG-sau" chạy được end-to-end trên NBAtransactions, trước với ops vàng (tháng 1), sau với LLM #1 + bộ phán LLM (tháng 2), kèm bảng số sơ bộ so với GUpdate.

**Architecture:** Package Python `kgu` thuần, không phụ thuộc DB: đồ thị bi-temporal in-memory là nguồn sự thật lúc dev (spec nói Neo4j chỉ là bản chiếu, dựng lại được từ JSONL). Mỗi bước pipeline là một module có interface nhỏ (`Extractor`, `Judge`) để thay thế bằng gold / LLM / link-prediction mà không sửa phần còn lại. LLM được gọi qua một `LLMClient` OpenAI-compatible (vLLM trên Colab, OpenRouter, Ollama) với `FakeLLMClient` cho test.

**Tech Stack:** Python 3.11, pydantic v2, openai (client OpenAI-compatible), pytest. Không cần GPU cục bộ.

**Spec:** `docs/superpowers/specs/2026-09-05-ke-hoach-notion.md`

## Global Constraints

- Không xóa cạnh: `INVALIDATE` = đóng dấu `valid_to`, cạnh vẫn nằm trong lịch sử.
- Áp ops tường minh **trước** khi khoanh vùng / phán.
- Khoanh vùng không dùng số hop và không dùng luật theo miền: vế cắt = entity nối trực tiếp với ≥ 2 entity tham gia sự kiện; vế sinh = hàng xóm của `b` chưa nối với `a`.
- Bộ phán chỉ phán trong vùng ứng viên; không bao giờ thêm top-k mù.
- Mọi ops/triple lưu được ra JSONL kèm provenance (`source`, `quote`, `conf`).
- Metric theo GUpdate: added = after − before, deleted = before − after; F1 trên `before ∪ after`; acc cạnh thêm / acc cạnh xóa. Mốc so sánh: F1 0,9866 · add 0,8926 · del 0,8988.
- Không code riêng cho miền NBA trong `kgu/` (tên quan hệ, tên đội… chỉ được xuất hiện trong test fixture và prompt example).
- Chạy lệnh Python bằng `.venv/Scripts/python.exe` (Windows). Mọi lệnh dưới đây viết tắt là `python`.

---

## Cấu trúc file

```
pyproject.toml                 # package kgu + deps + pytest config
kgu/__init__.py
kgu/types.py                   # Triple, Op, OpKind — kiểu dùng chung
kgu/data/__init__.py
kgu/data/nba.py                # đọc NBAtransactions → Example (tên thật, không id)
kgu/graph.py                   # BiTemporalGraph
kgu/ops.py                     # diff(before, after) → gold ops; apply_ops(graph, ops, at)
kgu/localize.py                # khoanh vùng 2 vế → Localization(cut, add)
kgu/judge/__init__.py          # Judge protocol + JudgeContext
kgu/judge/oracle.py            # OracleJudge (nhìn KG-sau) — cận trên của bộ phán
kgu/judge/llm.py               # LLMJudge (phương án A)
kgu/extract/__init__.py        # Extractor protocol
kgu/extract/gold.py            # GoldAllExtractor, GoldExplicitExtractor
kgu/extract/llm.py             # LLMExtractor (LLM #1)
kgu/llm.py                     # LLMClient protocol, OpenAICompatClient, FakeLLMClient
kgu/pipeline.py                # run_example(ex, extractor, judge) → PipelineResult
kgu/eval/__init__.py
kgu/eval/metrics.py            # Counts, score(), aggregate()
kgu/eval/coverage.py           # candidate recall của khoanh vùng
scripts/download_nba.py        # tải data GUpdater vào data/raw/nba/
scripts/inspect_nba.py         # in mẫu + thống kê → docs/data-notes.md
scripts/run_eval.py            # CLI chạy 1 cấu hình trên 1 split, ghi results/*.jsonl + summary
scripts/analyze_errors.py      # nhóm lỗi theo event / số ops
tests/conftest.py              # fixture graph nhỏ (Messi/Barca/Inter/Pedri/Lautaro)
tests/test_*.py
docs/data-notes.md             # ghi nhận từ spike dữ liệu (hướng quan hệ, thống kê)
```

---

### Task 1: Scaffold package + tải dữ liệu + spike khảo sát

**Files:**
- Create: `pyproject.toml`, `kgu/__init__.py`, `kgu/types.py`, `tests/__init__.py`, `tests/test_types.py`
- Create: `scripts/download_nba.py`, `scripts/inspect_nba.py`, `docs/data-notes.md`
- Modify: `.gitignore` (thêm `data/raw/`, `results/`)

**Interfaces:**
- Produces: `kgu.types.Triple = tuple[str, str, str]`, `kgu.types.OpKind` (Enum `ADD`, `INVALIDATE`), `kgu.types.Op` (frozen dataclass: `kind, h, r, t, source="", quote="", conf=1.0`, property `triple`, method `to_dict()`, classmethod `from_dict()`).

- [ ] **Step 1: Tạo `pyproject.toml`**

```toml
[project]
name = "kgu"
version = "0.1.0"
description = "LLM-driven knowledge graph update from news"
requires-python = ">=3.11"
dependencies = [
    "pydantic>=2.7",
    "openai>=1.40",
    "tqdm>=4.66",
]

[project.optional-dependencies]
dev = ["pytest>=8.0"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
include = ["kgu*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Cài package ở chế độ editable**

Run: `python -m pip install -e ".[dev]"`
Expected: kết thúc bằng `Successfully installed kgu-0.1.0 ...`

- [ ] **Step 3: Viết test cho `kgu/types.py`**

`tests/__init__.py` để trống. `tests/test_types.py`:

```python
from kgu.types import Op, OpKind


def test_op_roundtrip_dict():
    op = Op(OpKind.ADD, "Messi", "plays_for", "Inter", source="news#1", quote="Messi joins Inter", conf=0.9)
    d = op.to_dict()
    assert d == {
        "kind": "ADD", "h": "Messi", "r": "plays_for", "t": "Inter",
        "source": "news#1", "quote": "Messi joins Inter", "conf": 0.9,
    }
    assert Op.from_dict(d) == op


def test_op_triple_property():
    op = Op(OpKind.INVALIDATE, "Messi", "plays_for", "Barca")
    assert op.triple == ("Messi", "plays_for", "Barca")


def test_op_is_hashable_and_ignores_provenance_in_equality():
    a = Op(OpKind.ADD, "a", "r", "b", quote="x")
    b = Op(OpKind.ADD, "a", "r", "b", quote="y")
    assert a == b
    assert len({a, b}) == 1
```

- [ ] **Step 4: Chạy test, xác nhận fail**

Run: `python -m pytest tests/test_types.py -v`
Expected: FAIL với `ModuleNotFoundError: No module named 'kgu.types'`

- [ ] **Step 5: Viết `kgu/__init__.py` (trống) và `kgu/types.py`**

```python
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

Triple = tuple[str, str, str]


class OpKind(str, Enum):
    ADD = "ADD"
    INVALIDATE = "INVALIDATE"


@dataclass(frozen=True)
class Op:
    """Một thao tác thay đổi trên KG. Provenance không tham gia so sánh/hash."""

    kind: OpKind
    h: str
    r: str
    t: str
    source: str = field(default="", compare=False)
    quote: str = field(default="", compare=False)
    conf: float = field(default=1.0, compare=False)

    @property
    def triple(self) -> Triple:
        return (self.h, self.r, self.t)

    def to_dict(self) -> dict:
        return {
            "kind": self.kind.value, "h": self.h, "r": self.r, "t": self.t,
            "source": self.source, "quote": self.quote, "conf": self.conf,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Op":
        return cls(
            OpKind(d["kind"]), d["h"], d["r"], d["t"],
            source=d.get("source", ""), quote=d.get("quote", ""), conf=float(d.get("conf", 1.0)),
        )
```

- [ ] **Step 6: Chạy test, xác nhận pass**

Run: `python -m pytest tests/test_types.py -v`
Expected: 3 passed

- [ ] **Step 7: Viết `scripts/download_nba.py`**

```python
"""Tải dataset NBAtransactions (GUpdater, EMNLP 2019) vào data/raw/nba/."""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

BASE = "https://raw.githubusercontent.com/esddse/GUpdater/master/data/"
FILES = [
    "NBAtransactions_train.json", "NBAtransactions_valid.json", "NBAtransactions_test.json",
    "entity2id.txt", "relation2id.txt", "token2id.txt",
]


def main(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        dest = out_dir / name
        if dest.exists():
            print(f"skip {name} (exists)")
            continue
        print(f"downloading {name} ...")
        urllib.request.urlretrieve(BASE + name, dest)
        print(f"  -> {dest} ({dest.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/raw/nba"))
```

- [ ] **Step 8: Thêm vào `.gitignore` rồi tải dữ liệu**

Thêm hai dòng vào cuối `.gitignore`:
```
data/raw/
results/
```
Run: `python scripts/download_nba.py`
Expected: 6 file trong `data/raw/nba/`, file train ~52 MB.

- [ ] **Step 9: Viết `scripts/inspect_nba.py` (spike, chưa dùng loader)**

```python
"""Spike: in 3 mẫu đã giải mã + thống kê để ghi vào docs/data-notes.md."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path


def load_id_map(path: Path) -> dict[int, str]:
    out: dict[int, str] = {}
    for line in path.read_text(encoding="utf8").splitlines():
        if "\t" not in line:
            continue  # dòng đầu là số đếm
        name, idx = line.rsplit("\t", 1)
        out[int(idx)] = name
    return out


def main(data_dir: Path, split: str = "test") -> None:
    ent = load_id_map(data_dir / "entity2id.txt")
    rel = load_id_map(data_dir / "relation2id.txt")
    tok = load_id_map(data_dir / "token2id.txt")
    data = json.loads((data_dir / f"NBAtransactions_{split}.json").read_text(encoding="utf8"))
    print(f"{split}: {len(data)} examples; keys = {sorted(data[0].keys())}")

    def name(x):  # text / mentioned có thể là id hoặc chuỗi
        return ent.get(x, str(x)) if isinstance(x, int) else str(x)

    for ex in data[:3]:
        text = " ".join(tok.get(t, str(t)) if isinstance(t, int) else str(t) for t in ex["text"])
        before = {(ent[h], rel[r], ent[t]) for h, r, t in ex["subgraph_before"]}
        after = {(ent[h], rel[r], ent[t]) for h, r, t in ex["subgraph_after"]}
        print("=" * 80)
        print("event:", ex["event"], "| season:", ex["season"])
        print("text :", text)
        print("mentioned:", [name(m) for m in ex["text_mentioned_entities"]])
        print("|before| =", len(before), "|after| =", len(after))
        print("ADDED  :", sorted(after - before)[:10])
        print("DELETED:", sorted(before - after)[:10])

    n_add = n_del = 0
    rel_dir = Counter()  # (relation, head-is-team?) để xác định hướng quan hệ
    events = Counter()
    for ex in data:
        before = {tuple(t) for t in ex["subgraph_before"]}
        after = {tuple(t) for t in ex["subgraph_after"]}
        n_add += len(after - before)
        n_del += len(before - after)
        events[ex["event"]] += 1
        for h, r, t in list(before)[:50]:
            rel_dir[(rel[r], ent[h].endswith(("s", "ers")) and "_" in ent[h])] += 1
    print("=" * 80)
    print(f"avg added = {n_add/len(data):.2f}, avg deleted = {n_del/len(data):.2f}")
    print("events:", events.most_common())
    print("relation/head-looks-like-team:", rel_dir.most_common())


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/raw/nba"), sys.argv[2] if len(sys.argv) > 2 else "test")
```

- [ ] **Step 10: Chạy spike và ghi nhận**

Run: `python scripts/inspect_nba.py data/raw/nba test > docs/data-notes-raw.txt`
Đọc output rồi viết `docs/data-notes.md` với **đúng các mục sau** (điền số thật từ output, không để trống).
Kết quả đã quan sát khi viết plan (2026-09-07, split test) để đối chiếu:
- 438 ví dụ; keys = `event, season, subgraph_after, subgraph_before, text, text_mentioned_entities`.
- `text` là list token id (giải mã ra câu tiếng Anh đọc được, có ký tự lạ `�` và `<unk>`); `text_mentioned_entities` là list entity id.
- Entity cầu thủ gắn mùa giải: `Thaddeus_Young_2017-18`; đội: `Indiana_Pacers`.
- `<player>` lưu **cả hai chiều**: có `(Indiana_Pacers, <player>, Thaddeus_Young_2017-18)` lẫn `(Anthony_Randolph_2014-15, <player>, Cleveland_Cavaliers)`. `<teammate>` lưu cả 2 chiều. Pipeline vì thế không được giả định chiều.
- avg added = 38,2, avg deleted = 39,7 trên split test (cao hơn số trung bình toàn bộ trong paper).
- events: trade 150, released 139, free_agency 107, draft 18, head_coach 10, d_league 6, retirement 5, overseas 3.

```markdown
# NBAtransactions — ghi nhận từ spike (2026-09-xx)

## Cấu trúc
- keys mỗi example: ...
- `text`: list token id | list chuỗi (chọn một)
- `text_mentioned_entities`: list entity id | list chuỗi (chọn một)

## Hướng quan hệ (quan trọng cho khoanh vùng)
- `<player>`: head = ..., tail = ...   (ví dụ thật: ...)
- `<teammate>`: có lưu cả 2 chiều (a,teammate,b) và (b,teammate,a)? có/không
- `<head_coach>`: head = ..., tail = ...
- `<general_mananger>`: head = ..., tail = ...

## Thống kê split test
- số example, avg added, avg deleted, phân bố event

## Ví dụ 1 tin đầy đủ (text + added + deleted)
...
```

- [ ] **Step 11: Commit**

```bash
git add pyproject.toml kgu tests scripts .gitignore docs/data-notes.md
git commit -m "feat: scaffold kgu package, Op types, NBA data download + spike notes"
```

---

### Task 2: Loader NBAtransactions → `Example`

**Files:**
- Create: `kgu/data/__init__.py`, `kgu/data/nba.py`, `tests/test_data_nba.py`

**Interfaces:**
- Consumes: `kgu.types.Triple`
- Produces:
  - `kgu.data.nba.Example` dataclass: `idx: int, event: str, season: str, text: str, mentioned: list[str], before: set[Triple], after: set[Triple]`
  - `kgu.data.nba.Vocab.load(data_dir: Path) -> Vocab` với `.entity`, `.relation`, `.token` (dict[int,str]) và `.relations -> list[str]`
  - `kgu.data.nba.load_split(data_dir: Path, split: str, limit: int | None = None) -> list[Example]`

- [ ] **Step 1: Viết test với fixture tí hon trong `tmp_path`**

`tests/test_data_nba.py`:

```python
import json
from pathlib import Path

from kgu.data.nba import Vocab, load_split


def _write_fixture(d: Path) -> None:
    (d / "entity2id.txt").write_text("3\nBarca\t0\nMessi\t1\nInter\t2\n", encoding="utf8")
    (d / "relation2id.txt").write_text("1\n<player>\t0\n", encoding="utf8")
    (d / "token2id.txt").write_text("3\nmessi\t0\njoins\t1\ninter\t2\n", encoding="utf8")
    ex = {
        "event": "free agency", "season": "2021-22",
        "text": [0, 1, 2],
        "text_mentioned_entities": [1, 2],
        "subgraph_before": [[0, 0, 1]],
        "subgraph_after": [[2, 0, 1]],
    }
    (d / "NBAtransactions_test.json").write_text(json.dumps([ex, ex]), encoding="utf8")


def test_vocab_skips_count_line(tmp_path):
    _write_fixture(tmp_path)
    v = Vocab.load(tmp_path)
    assert v.entity == {0: "Barca", 1: "Messi", 2: "Inter"}
    assert v.relations == ["<player>"]


def test_load_split_decodes_names(tmp_path):
    _write_fixture(tmp_path)
    exs = load_split(tmp_path, "test")
    assert len(exs) == 2
    ex = exs[0]
    assert ex.idx == 0 and ex.event == "free agency"
    assert ex.text == "messi joins inter"
    assert ex.mentioned == ["Messi", "Inter"]
    assert ex.before == {("Barca", "<player>", "Messi")}
    assert ex.after == {("Inter", "<player>", "Messi")}


def test_load_split_limit(tmp_path):
    _write_fixture(tmp_path)
    assert len(load_split(tmp_path, "test", limit=1)) == 1
```

- [ ] **Step 2: Chạy test, xác nhận fail**

Run: `python -m pytest tests/test_data_nba.py -v`
Expected: FAIL `ModuleNotFoundError: No module named 'kgu.data'`

- [ ] **Step 3: Viết `kgu/data/__init__.py` (trống) và `kgu/data/nba.py`**

```python
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from kgu.types import Triple


@dataclass
class Example:
    idx: int
    event: str
    season: str
    text: str
    mentioned: list[str]
    before: set[Triple]
    after: set[Triple]


def load_id_map(path: Path) -> dict[int, str]:
    """Đọc file `name<TAB>id`; bỏ qua dòng đầu (số đếm) và dòng không có TAB."""
    out: dict[int, str] = {}
    for line in path.read_text(encoding="utf8").splitlines():
        if "\t" not in line:
            continue
        name, idx = line.rsplit("\t", 1)
        out[int(idx)] = name
    return out


@dataclass
class Vocab:
    entity: dict[int, str]
    relation: dict[int, str]
    token: dict[int, str]

    @classmethod
    def load(cls, data_dir: Path) -> "Vocab":
        return cls(
            entity=load_id_map(data_dir / "entity2id.txt"),
            relation=load_id_map(data_dir / "relation2id.txt"),
            token=load_id_map(data_dir / "token2id.txt"),
        )

    @property
    def relations(self) -> list[str]:
        return [self.relation[k] for k in sorted(self.relation)]

    def triple(self, raw: list[int]) -> Triple:
        h, r, t = raw
        return (self.entity[h], self.relation[r], self.entity[t])

    def _name(self, x, table: dict[int, str]) -> str:
        return table[x] if isinstance(x, int) else str(x)

    def text(self, raw: list) -> str:
        return " ".join(self._name(t, self.token) for t in raw)

    def entities(self, raw: list) -> list[str]:
        return [self._name(e, self.entity) for e in raw]


def load_split(data_dir: Path, split: str, limit: int | None = None) -> list[Example]:
    vocab = Vocab.load(data_dir)
    raw = json.loads((data_dir / f"NBAtransactions_{split}.json").read_text(encoding="utf8"))
    if limit is not None:
        raw = raw[:limit]
    return [
        Example(
            idx=i,
            event=str(d.get("event", "")),
            season=str(d.get("season", "")),
            text=vocab.text(d["text"]),
            mentioned=vocab.entities(d.get("text_mentioned_entities", [])),
            before={vocab.triple(t) for t in d["subgraph_before"]},
            after={vocab.triple(t) for t in d["subgraph_after"]},
        )
        for i, d in enumerate(raw)
    ]
```

- [ ] **Step 4: Chạy test, xác nhận pass**

Run: `python -m pytest tests/test_data_nba.py -v`
Expected: 3 passed

- [ ] **Step 5: Kiểm tra trên dữ liệu thật**

Run: `python -c "from pathlib import Path; from kgu.data.nba import load_split; xs=load_split(Path('data/raw/nba'),'test'); print(len(xs)); print(xs[0].text); print(len(xs[0].before), len(xs[0].after))"`
Expected: in số example của split test (~422–438), một câu tiếng Anh đọc được, hai số cạnh > 0. Nếu `text` không đọc được (toàn số) → sửa `Vocab.text` theo ghi nhận trong `docs/data-notes.md`.

- [ ] **Step 6: Commit**

```bash
git add kgu/data tests/test_data_nba.py
git commit -m "feat: NBAtransactions loader producing named Example"
```

---

### Task 3: `BiTemporalGraph`

**Files:**
- Create: `kgu/graph.py`, `tests/test_graph.py`

**Interfaces:**
- Consumes: `kgu.types.Triple`
- Produces `kgu.graph.BiTemporalGraph`:
  - `BiTemporalGraph.from_triples(triples: Iterable[Triple], valid_from: int = 0) -> BiTemporalGraph`
  - `add(triple, valid_from: int) -> bool` (False nếu đã active)
  - `invalidate(triple, valid_to: int) -> bool` (False nếu không active)
  - `is_active(triple, at: int) -> bool`
  - `active(at: int) -> set[Triple]`
  - `entities(at: int) -> set[str]`
  - `edges_of(entity: str, at: int) -> set[Triple]`
  - `neighbors(entity: str, at: int) -> set[str]`
  - `history(triple) -> list[Edge]` với `Edge(h, r, t, valid_from, valid_to)`
  - `to_records() -> list[dict]` (JSONL-able)
- Quy ước thời gian: số nguyên; cạnh active tại `at` khi `valid_from <= at` và (`valid_to is None` hoặc `valid_to > at`). KG-trước = thời điểm 0, sau khi áp tin = thời điểm 1.

- [ ] **Step 1: Viết test**

`tests/test_graph.py`:

```python
from kgu.graph import BiTemporalGraph

T_MB = ("Messi", "plays_for", "Barca")
T_MI = ("Messi", "plays_for", "Inter")
T_PB = ("Pedri", "plays_for", "Barca")


def test_from_triples_all_active_at_0():
    g = BiTemporalGraph.from_triples([T_MB, T_PB])
    assert g.active(0) == {T_MB, T_PB}
    assert g.entities(0) == {"Messi", "Barca", "Pedri"}


def test_invalidate_keeps_history_and_hides_after():
    g = BiTemporalGraph.from_triples([T_MB])
    assert g.invalidate(T_MB, valid_to=1) is True
    assert g.is_active(T_MB, 0) is True      # lịch sử vẫn truy vấn được
    assert g.is_active(T_MB, 1) is False
    assert g.active(1) == set()
    assert g.invalidate(T_MB, valid_to=1) is False  # đã inactive


def test_add_then_readd_is_noop_when_active():
    g = BiTemporalGraph()
    assert g.add(T_MI, valid_from=1) is True
    assert g.add(T_MI, valid_from=1) is False
    assert g.active(0) == set()
    assert g.active(1) == {T_MI}


def test_invalidate_then_readd_creates_second_interval():
    g = BiTemporalGraph.from_triples([T_MB])
    g.invalidate(T_MB, 1)
    g.add(T_MB, 2)
    hist = g.history(T_MB)
    assert [(e.valid_from, e.valid_to) for e in hist] == [(0, 1), (2, None)]
    assert g.is_active(T_MB, 1) is False and g.is_active(T_MB, 2) is True


def test_neighbors_and_edges_of_respect_time():
    g = BiTemporalGraph.from_triples([T_MB, T_PB])
    g.invalidate(T_MB, 1)
    g.add(T_MI, 1)
    assert g.neighbors("Messi", 0) == {"Barca"}
    assert g.neighbors("Messi", 1) == {"Inter"}
    assert g.edges_of("Barca", 1) == {T_PB}
    assert g.neighbors("Barca", 1) == {"Pedri"}


def test_to_records():
    g = BiTemporalGraph.from_triples([T_MB])
    g.invalidate(T_MB, 1)
    assert g.to_records() == [
        {"h": "Messi", "r": "plays_for", "t": "Barca", "valid_from": 0, "valid_to": 1}
    ]
```

- [ ] **Step 2: Chạy test, xác nhận fail**

Run: `python -m pytest tests/test_graph.py -v`
Expected: FAIL `ModuleNotFoundError: No module named 'kgu.graph'`

- [ ] **Step 3: Viết `kgu/graph.py`**

```python
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from kgu.types import Triple


@dataclass
class Edge:
    h: str
    r: str
    t: str
    valid_from: int
    valid_to: int | None = None

    @property
    def triple(self) -> Triple:
        return (self.h, self.r, self.t)

    def active_at(self, at: int) -> bool:
        return self.valid_from <= at and (self.valid_to is None or self.valid_to > at)


class BiTemporalGraph:
    """Đồ thị có hướng, đa quan hệ; mỗi triple có thể có nhiều khoảng hiệu lực.

    Không bao giờ xóa: INVALIDATE chỉ đóng `valid_to` của khoảng đang mở.
    """

    def __init__(self) -> None:
        self._edges: dict[Triple, list[Edge]] = defaultdict(list)
        self._by_entity: dict[str, set[Triple]] = defaultdict(set)

    @classmethod
    def from_triples(cls, triples: Iterable[Triple], valid_from: int = 0) -> "BiTemporalGraph":
        g = cls()
        for tr in triples:
            g.add(tr, valid_from)
        return g

    # ---- mutation -------------------------------------------------------
    def _open(self, triple: Triple) -> Edge | None:
        for e in self._edges.get(triple, ()):
            if e.valid_to is None:
                return e
        return None

    def add(self, triple: Triple, valid_from: int) -> bool:
        if self._open(triple) is not None:
            return False
        h, r, t = triple
        self._edges[triple].append(Edge(h, r, t, valid_from))
        self._by_entity[h].add(triple)
        self._by_entity[t].add(triple)
        return True

    def invalidate(self, triple: Triple, valid_to: int) -> bool:
        e = self._open(triple)
        if e is None:
            return False
        e.valid_to = valid_to
        return True

    # ---- queries --------------------------------------------------------
    def is_active(self, triple: Triple, at: int) -> bool:
        return any(e.active_at(at) for e in self._edges.get(triple, ()))

    def active(self, at: int) -> set[Triple]:
        return {tr for tr, es in self._edges.items() if any(e.active_at(at) for e in es)}

    def entities(self, at: int) -> set[str]:
        out: set[str] = set()
        for h, _, t in self.active(at):
            out.add(h)
            out.add(t)
        return out

    def edges_of(self, entity: str, at: int) -> set[Triple]:
        return {tr for tr in self._by_entity.get(entity, ()) if self.is_active(tr, at)}

    def neighbors(self, entity: str, at: int) -> set[str]:
        out: set[str] = set()
        for h, _, t in self.edges_of(entity, at):
            out.add(t if h == entity else h)
        out.discard(entity)
        return out

    def history(self, triple: Triple) -> list[Edge]:
        return list(self._edges.get(triple, ()))

    def to_records(self) -> list[dict]:
        return [
            {"h": e.h, "r": e.r, "t": e.t, "valid_from": e.valid_from, "valid_to": e.valid_to}
            for tr in sorted(self._edges)
            for e in self._edges[tr]
        ]
```

- [ ] **Step 4: Chạy test, xác nhận pass**

Run: `python -m pytest tests/test_graph.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add kgu/graph.py tests/test_graph.py
git commit -m "feat: in-memory bi-temporal graph with valid_from/valid_to"
```

---

### Task 4: Ops — `diff` (ops vàng) và `apply_ops` (executor)

**Files:**
- Create: `kgu/ops.py`, `tests/test_ops.py`

**Interfaces:**
- Consumes: `Op`, `OpKind`, `BiTemporalGraph`
- Produces:
  - `kgu.ops.diff(before: set[Triple], after: set[Triple], source: str = "gold") -> list[Op]` — INVALIDATE cho `before − after`, ADD cho `after − before`, sắp xếp ổn định (INVALIDATE trước, rồi theo triple).
  - `kgu.ops.apply_ops(graph: BiTemporalGraph, ops: Iterable[Op], at: int) -> list[Op]` — trả về các op **thực sự** có tác dụng (bỏ ADD trùng / INVALIDATE cạnh không active).
  - `kgu.ops.write_jsonl(path: Path, ops: Iterable[Op])`, `kgu.ops.read_jsonl(path: Path) -> list[Op]`

- [ ] **Step 1: Viết test**

`tests/test_ops.py`:

```python
from pathlib import Path

from kgu.graph import BiTemporalGraph
from kgu.ops import apply_ops, diff, read_jsonl, write_jsonl
from kgu.types import Op, OpKind

T_MB = ("Messi", "plays_for", "Barca")
T_MI = ("Messi", "plays_for", "Inter")
T_PB = ("Pedri", "plays_for", "Barca")


def test_diff_produces_invalidate_then_add():
    ops = diff({T_MB, T_PB}, {T_MI, T_PB})
    assert ops == [Op(OpKind.INVALIDATE, *T_MB), Op(OpKind.ADD, *T_MI)]
    assert all(o.source == "gold" for o in ops)


def test_apply_ops_updates_graph_and_reports_effective_ops():
    g = BiTemporalGraph.from_triples({T_MB, T_PB})
    ops = [
        Op(OpKind.INVALIDATE, *T_MB),
        Op(OpKind.ADD, *T_MI),
        Op(OpKind.ADD, *T_PB),                    # đã có → không tác dụng
        Op(OpKind.INVALIDATE, "X", "plays_for", "Y"),  # không tồn tại → không tác dụng
    ]
    effective = apply_ops(g, ops, at=1)
    assert effective == ops[:2]
    assert g.active(1) == {T_MI, T_PB}
    assert g.active(0) == {T_MB, T_PB}


def test_diff_then_apply_reconstructs_after_exactly():
    before, after = {T_MB, T_PB}, {T_MI, T_PB}
    g = BiTemporalGraph.from_triples(before)
    apply_ops(g, diff(before, after), at=1)
    assert g.active(1) == after


def test_jsonl_roundtrip(tmp_path: Path):
    ops = [Op(OpKind.ADD, *T_MI, source="news#7", quote="Messi joins Inter", conf=0.8)]
    p = tmp_path / "ops.jsonl"
    write_jsonl(p, ops)
    back = read_jsonl(p)
    assert back == ops and back[0].quote == "Messi joins Inter" and back[0].conf == 0.8
```

- [ ] **Step 2: Chạy test, xác nhận fail**

Run: `python -m pytest tests/test_ops.py -v`
Expected: FAIL `ModuleNotFoundError: No module named 'kgu.ops'`

- [ ] **Step 3: Viết `kgu/ops.py`**

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from kgu.graph import BiTemporalGraph
from kgu.types import Op, OpKind, Triple


def diff(before: set[Triple], after: set[Triple], source: str = "gold") -> list[Op]:
    """Ops vàng: INVALIDATE cho cạnh mất, ADD cho cạnh mới. Thứ tự ổn định."""
    inv = [Op(OpKind.INVALIDATE, *tr, source=source) for tr in sorted(before - after)]
    add = [Op(OpKind.ADD, *tr, source=source) for tr in sorted(after - before)]
    return inv + add


def apply_ops(graph: BiTemporalGraph, ops: Iterable[Op], at: int) -> list[Op]:
    """Executor bi-temporal: INVALIDATE = đóng valid_to=at, ADD = mở valid_from=at."""
    effective: list[Op] = []
    for op in ops:
        if op.kind is OpKind.INVALIDATE:
            ok = graph.invalidate(op.triple, valid_to=at)
        else:
            ok = graph.add(op.triple, valid_from=at)
        if ok:
            effective.append(op)
    return effective


def write_jsonl(path: Path, ops: Iterable[Op]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf8") as f:
        for op in ops:
            f.write(json.dumps(op.to_dict(), ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[Op]:
    with path.open(encoding="utf8") as f:
        return [Op.from_dict(json.loads(line)) for line in f if line.strip()]
```

- [ ] **Step 4: Chạy test, xác nhận pass**

Run: `python -m pytest tests/test_ops.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add kgu/ops.py tests/test_ops.py
git commit -m "feat: gold diff ops and bi-temporal executor with JSONL provenance"
```

---

### Task 5: Metrics theo GUpdate

**Files:**
- Create: `kgu/eval/__init__.py`, `kgu/eval/metrics.py`, `tests/test_metrics.py`

**Interfaces:**
- Produces:
  - `kgu.eval.metrics.Counts` dataclass: `tp, fp, fn, tn, add_hit, n_add, del_hit, n_del` (int), method `__add__`, method `to_dict()`.
  - `kgu.eval.metrics.score(before, gold_after, pred_after) -> Counts` — universe = `before ∪ gold_after ∪ pred_after` (triple dự đoán ngoài `before ∪ gold_after` bị tính FP); label = triple ∈ gold_after; pred = triple ∈ pred_after. `add_hit` = số cạnh added (gold_after − before) có trong pred; `del_hit` = số cạnh deleted (before − gold_after) **không** có trong pred.
  - `kgu.eval.metrics.summarize(c: Counts) -> dict` với `precision, recall, f1, add_acc, del_acc` (micro; chia cho 0 → 0.0).

- [ ] **Step 1: Viết test**

`tests/test_metrics.py`:

```python
from kgu.eval.metrics import Counts, score, summarize

A, B, C, D, E = [("x", "r", str(i)) for i in range(5)]


def test_perfect_prediction():
    before, gold = {A, B, C}, {B, C, D}
    c = score(before, gold, pred_after=gold)
    assert (c.tp, c.fp, c.fn) == (3, 0, 0)
    assert (c.add_hit, c.n_add, c.del_hit, c.n_del) == (1, 1, 1, 1)
    s = summarize(c)
    assert s["f1"] == 1.0 and s["add_acc"] == 1.0 and s["del_acc"] == 1.0


def test_noop_prediction_keeps_before():
    before, gold = {A, B, C}, {B, C, D}
    c = score(before, gold, pred_after=before)
    # universe = {A,B,C,D}; pred = {A,B,C}; gold = {B,C,D}
    assert (c.tp, c.fp, c.fn, c.tn) == (2, 1, 1, 0)
    assert (c.add_hit, c.del_hit) == (0, 0)
    s = summarize(c)
    assert s["precision"] == 2 / 3 and s["recall"] == 2 / 3
    assert s["add_acc"] == 0.0 and s["del_acc"] == 0.0


def test_extra_prediction_outside_universe_counts_as_fp():
    before, gold = {A}, {A}
    c = score(before, gold, pred_after={A, E})
    assert (c.tp, c.fp, c.fn) == (1, 1, 0)


def test_counts_add_and_zero_division():
    c = Counts() + Counts(tp=1)
    assert c.tp == 1
    s = summarize(Counts())
    assert s == {"precision": 0.0, "recall": 0.0, "f1": 0.0, "add_acc": 0.0, "del_acc": 0.0}
```

- [ ] **Step 2: Chạy test, xác nhận fail**

Run: `python -m pytest tests/test_metrics.py -v`
Expected: FAIL `ModuleNotFoundError: No module named 'kgu.eval'`

- [ ] **Step 3: Viết `kgu/eval/__init__.py` (trống) và `kgu/eval/metrics.py`**

```python
from __future__ import annotations

from dataclasses import asdict, dataclass

from kgu.types import Triple


@dataclass
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    add_hit: int = 0
    n_add: int = 0
    del_hit: int = 0
    n_del: int = 0

    def __add__(self, other: "Counts") -> "Counts":
        return Counts(**{k: getattr(self, k) + getattr(other, k) for k in asdict(self)})

    def to_dict(self) -> dict:
        return asdict(self)


def score(before: set[Triple], gold_after: set[Triple], pred_after: set[Triple]) -> Counts:
    """Theo GUpdate: F1 trên before ∪ gold_after; acc riêng cho cạnh thêm / cạnh xóa.

    Triple dự đoán nằm ngoài universe vẫn bị tính FP (không cho thêm bừa).
    """
    universe = before | gold_after | pred_after
    c = Counts()
    for tr in universe:
        label, pred = tr in gold_after, tr in pred_after
        if label and pred:
            c.tp += 1
        elif label and not pred:
            c.fn += 1
        elif not label and pred:
            c.fp += 1
        else:
            c.tn += 1
    added, deleted = gold_after - before, before - gold_after
    c.n_add, c.n_del = len(added), len(deleted)
    c.add_hit = sum(1 for tr in added if tr in pred_after)
    c.del_hit = sum(1 for tr in deleted if tr not in pred_after)
    return c


def _div(a: int, b: int) -> float:
    return a / b if b else 0.0


def summarize(c: Counts) -> dict:
    p = _div(c.tp, c.tp + c.fp)
    r = _div(c.tp, c.tp + c.fn)
    return {
        "precision": p,
        "recall": r,
        "f1": _div(2 * p * r, p + r) if (p + r) else 0.0,
        "add_acc": _div(c.add_hit, c.n_add),
        "del_acc": _div(c.del_hit, c.n_del),
    }
```

- [ ] **Step 4: Chạy test, xác nhận pass**

Run: `python -m pytest tests/test_metrics.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add kgu/eval tests/test_metrics.py
git commit -m "feat: GUpdate-style metrics (F1, add acc, del acc)"
```

---

### Task 6: Khoanh vùng cấu trúc 2 vế (`localize`)

**Files:**
- Create: `kgu/localize.py`, `tests/conftest.py`, `tests/test_localize.py`

**Interfaces:**
- Consumes: `BiTemporalGraph`, `Op`, `OpKind`
- Produces:
  - `kgu.localize.CutCandidate(triple: Triple, hub: str, anchors: frozenset[str])` — `hub` là entity nối với ≥ 2 participant; `anchors` là các participant nó nối tới.
  - `kgu.localize.AddCandidate(subject: str, neighbor: str, via: Triple)` — `subject` = `a` (đầu cạnh mới), `neighbor` = hàng xóm của `b` chưa nối `a`, `via` = cạnh `b–neighbor` làm bằng chứng.
  - `kgu.localize.Localization(cut: list[CutCandidate], add: list[AddCandidate])` với `.cut_triples -> set[Triple]`, `.add_pairs -> set[frozenset[str]]`.
  - `kgu.localize.localize(graph: BiTemporalGraph, explicit_ops: list[Op], at: int) -> Localization` — gọi **sau** khi đã `apply_ops(explicit_ops, at)`.
- Định nghĩa (không phụ thuộc miền):
  - participants `P` = mọi entity xuất hiện trong `explicit_ops`.
  - **Vế cắt:** với mỗi INVALIDATE `(a, r, b)`: xét từng `c ∉ P` có `|neighbors(c, at) ∩ P| ≥ 2` **và** `c` kề `a` hoặc `b` tại `at`; ứng viên = mọi cạnh active tại `at` giữa `c` và một participant. (Với `a` bị cắt khỏi `b`, `c` vẫn kề `a` là điều đáng nghi: Pedri–Messi teammate.) Không dùng edge tới chính participant khác nhau ngoài `c`.
  - **Vế sinh:** với mỗi ADD `(a, r, b)`: với mỗi `n ∈ neighbors(b, at) − neighbors(a, at) − {a, b}`: ứng viên `AddCandidate(a, n, via)` với `via` là một cạnh active giữa `b` và `n`. Ứng viên trùng `(subject, neighbor)` gộp lại (giữ `via` đầu tiên).
  - Kết quả sắp xếp ổn định để test và log có thể tái lập.

- [ ] **Step 1: Viết fixture chung `tests/conftest.py`**

```python
import pytest

from kgu.graph import BiTemporalGraph

# Thế giới nhỏ: Messi rời Barca sang Inter.
PLAYS = "plays_for"
TEAM = "teammate"
COACH = "head_coach"
FRIEND = "close_friend"
SPONSOR = "sponsor"

BEFORE = {
    ("Messi", PLAYS, "Barca"),
    ("Pedri", PLAYS, "Barca"),
    ("Pedri", TEAM, "Messi"),
    ("Messi", TEAM, "Pedri"),
    ("Xavi", COACH, "Barca"),
    ("Pedri", FRIEND, "Messi"),
    ("Adidas", SPONSOR, "Messi"),        # một chân: chỉ nối Messi
    ("Lautaro", PLAYS, "Inter"),
    ("Inzaghi", COACH, "Inter"),
}
AFTER = (BEFORE - {
    ("Messi", PLAYS, "Barca"),
    ("Pedri", TEAM, "Messi"),
    ("Messi", TEAM, "Pedri"),
}) | {
    ("Messi", PLAYS, "Inter"),
    ("Lautaro", TEAM, "Messi"),
    ("Messi", TEAM, "Lautaro"),
}


@pytest.fixture
def before():
    return set(BEFORE)


@pytest.fixture
def after():
    return set(AFTER)


@pytest.fixture
def graph(before):
    return BiTemporalGraph.from_triples(before, valid_from=0)
```

- [ ] **Step 2: Viết test `tests/test_localize.py`**

```python
from kgu.localize import AddCandidate, CutCandidate, localize
from kgu.ops import apply_ops
from kgu.types import Op, OpKind
from tests.conftest import COACH, FRIEND, PLAYS, SPONSOR, TEAM

EXPLICIT = [
    Op(OpKind.INVALIDATE, "Messi", PLAYS, "Barca"),
    Op(OpKind.ADD, "Messi", PLAYS, "Inter"),
]


def _run(graph):
    apply_ops(graph, EXPLICIT, at=1)
    return localize(graph, EXPLICIT, at=1)


def test_cut_side_finds_common_neighbour_edges(graph):
    loc = _run(graph)
    # Pedri kề cả Messi và Barca → mọi cạnh Pedri–{Messi,Barca} là ứng viên
    assert loc.cut_triples == {
        ("Pedri", PLAYS, "Barca"),
        ("Pedri", TEAM, "Messi"),
        ("Messi", TEAM, "Pedri"),
        ("Pedri", FRIEND, "Messi"),
    }
    pedri = [c for c in loc.cut if c.hub == "Pedri"]
    assert all(c.anchors == frozenset({"Messi", "Barca"}) for c in pedri)


def test_cut_side_ignores_one_leg_entities(graph):
    loc = _run(graph)
    assert ("Adidas", SPONSOR, "Messi") not in loc.cut_triples   # chỉ nối 1 participant
    assert ("Xavi", COACH, "Barca") not in loc.cut_triples       # chỉ nối Barca


def test_add_side_lists_new_context_members(graph):
    loc = _run(graph)
    assert loc.add_pairs == {
        frozenset({"Messi", "Lautaro"}),
        frozenset({"Messi", "Inzaghi"}),
    }
    lautaro = next(c for c in loc.add if c.neighbor == "Lautaro")
    assert lautaro == AddCandidate("Messi", "Lautaro", ("Lautaro", PLAYS, "Inter"))


def test_add_side_skips_neighbours_already_linked(graph):
    graph.add(("Messi", FRIEND, "Lautaro"), valid_from=0)
    loc = _run(graph)
    assert frozenset({"Messi", "Lautaro"}) not in loc.add_pairs


def test_no_ops_no_candidates(graph):
    loc = localize(graph, [], at=1)
    assert loc.cut == [] and loc.add == []


def test_results_are_deterministic(graph, before):
    from kgu.graph import BiTemporalGraph
    a = _run(graph)
    b = _run(BiTemporalGraph.from_triples(before))
    assert a == b
```

- [ ] **Step 3: Chạy test, xác nhận fail**

Run: `python -m pytest tests/test_localize.py -v`
Expected: FAIL `ModuleNotFoundError: No module named 'kgu.localize'`

- [ ] **Step 4: Viết `kgu/localize.py`**

```python
from __future__ import annotations

from dataclasses import dataclass, field

from kgu.graph import BiTemporalGraph
from kgu.types import Op, OpKind, Triple


@dataclass(frozen=True)
class CutCandidate:
    triple: Triple
    hub: str                     # entity nối với >= 2 participant
    anchors: frozenset[str]      # các participant mà hub nối tới


@dataclass(frozen=True)
class AddCandidate:
    subject: str                 # a: đầu cạnh mới (a, r, b)
    neighbor: str                # n: hàng xóm của b chưa nối với a
    via: Triple                  # cạnh b–n làm bằng chứng bối cảnh


@dataclass
class Localization:
    cut: list[CutCandidate] = field(default_factory=list)
    add: list[AddCandidate] = field(default_factory=list)

    @property
    def cut_triples(self) -> set[Triple]:
        return {c.triple for c in self.cut}

    @property
    def add_pairs(self) -> set[frozenset[str]]:
        return {frozenset((c.subject, c.neighbor)) for c in self.add}


def _edges_between(graph: BiTemporalGraph, x: str, y: str, at: int) -> set[Triple]:
    return {tr for tr in graph.edges_of(x, at) if y in (tr[0], tr[2])}


def localize(graph: BiTemporalGraph, explicit_ops: list[Op], at: int) -> Localization:
    """Khoanh vùng cấu trúc 2 vế. Gọi SAU khi explicit_ops đã được áp tại `at`."""
    participants = {e for op in explicit_ops for e in (op.h, op.t)}
    loc = Localization()

    # ---- vế cắt --------------------------------------------------------
    seen_cut: set[Triple] = set()
    for op in explicit_ops:
        if op.kind is not OpKind.INVALIDATE:
            continue
        a, b = op.h, op.t
        # `a` và `b` đã không còn kề nhau tại `at`; hub phải kề a hoặc b
        hubs = (graph.neighbors(a, at) | graph.neighbors(b, at)) - participants
        for hub in sorted(hubs):
            anchors = graph.neighbors(hub, at) & participants
            if len(anchors) < 2:
                continue
            for p in sorted(anchors):
                for tr in sorted(_edges_between(graph, hub, p, at)):
                    if tr not in seen_cut:
                        seen_cut.add(tr)
                        loc.cut.append(CutCandidate(tr, hub, frozenset(anchors)))

    # ---- vế sinh -------------------------------------------------------
    seen_add: set[tuple[str, str]] = set()
    for op in explicit_ops:
        if op.kind is not OpKind.ADD:
            continue
        a, b = op.h, op.t
        fresh = graph.neighbors(b, at) - graph.neighbors(a, at) - {a, b}
        for n in sorted(fresh):
            if (a, n) in seen_add:
                continue
            via = sorted(_edges_between(graph, b, n, at))[0]
            seen_add.add((a, n))
            loc.add.append(AddCandidate(a, n, via))

    return loc
```

- [ ] **Step 5: Chạy test, xác nhận pass**

Run: `python -m pytest tests/test_localize.py -v`
Expected: 6 passed. Nếu `test_cut_side_finds_common_neighbour_edges` fail vì thiếu `("Pedri", PLAYS, "Barca")`: kiểm tra `_edges_between` có lấy cạnh theo cả 2 chiều không.

- [ ] **Step 6: Commit**

```bash
git add kgu/localize.py tests/conftest.py tests/test_localize.py
git commit -m "feat: structural localization (cut side: shared neighbours, add side: new context)"
```

---

### Task 7: Judge protocol + `OracleJudge`; Extractor protocol + gold extractors

**Files:**
- Create: `kgu/judge/__init__.py`, `kgu/judge/oracle.py`, `kgu/extract/__init__.py`, `kgu/extract/gold.py`, `tests/test_oracle_judge.py`, `tests/test_gold_extractors.py`

**Interfaces:**
- Consumes: `Localization`, `BiTemporalGraph`, `Example`, `diff`
- Produces:
  - `kgu.judge.JudgeContext` dataclass: `text: str, explicit_ops: list[Op], loc: Localization, graph: BiTemporalGraph, at: int, relations: list[str]`
  - `kgu.judge.Judge` Protocol: `judge(ctx: JudgeContext) -> list[Op]` (INVALIDATE cho cut candidate cần cắt; ADD cho add candidate cần thêm, quan hệ do judge chọn).
  - `kgu.judge.oracle.OracleJudge(gold_after: set[Triple])` — cận trên: cut → INVALIDATE nếu triple ∉ gold_after; add → mọi triple trong gold_after nối `subject`–`neighbor` (2 chiều) mà chưa active.
  - `kgu.extract.Extractor` Protocol: `extract(ex: Example, graph: BiTemporalGraph) -> list[Op]`
  - `kgu.extract.gold.GoldAllExtractor` — toàn bộ `diff(before, after)` (cận trên).
  - `kgu.extract.gold.GoldExplicitExtractor` — chỉ các op có cả `h` và `t` ∈ `ex.mentioned` (xấp xỉ "5,6% được nói thẳng trong text").

- [ ] **Step 1: Viết test `tests/test_oracle_judge.py`**

```python
from kgu.judge import JudgeContext
from kgu.judge.oracle import OracleJudge
from kgu.localize import localize
from kgu.ops import apply_ops
from kgu.types import Op, OpKind
from tests.conftest import PLAYS, TEAM, FRIEND

EXPLICIT = [
    Op(OpKind.INVALIDATE, "Messi", PLAYS, "Barca"),
    Op(OpKind.ADD, "Messi", PLAYS, "Inter"),
]


def test_oracle_reconstructs_gold_after(graph, after):
    apply_ops(graph, EXPLICIT, at=1)
    loc = localize(graph, EXPLICIT, at=1)
    ctx = JudgeContext(text="", explicit_ops=EXPLICIT, loc=loc, graph=graph, at=1, relations=[PLAYS, TEAM])
    ops = OracleJudge(after).judge(ctx)
    assert set(ops) == {
        Op(OpKind.INVALIDATE, "Pedri", TEAM, "Messi"),
        Op(OpKind.INVALIDATE, "Messi", TEAM, "Pedri"),
        Op(OpKind.ADD, "Lautaro", TEAM, "Messi"),
        Op(OpKind.ADD, "Messi", TEAM, "Lautaro"),
    }
    assert Op(OpKind.INVALIDATE, "Pedri", FRIEND, "Messi") not in ops   # giữ
    apply_ops(graph, ops, at=1)
    assert graph.active(1) == after
    assert all(o.source == "oracle" for o in ops)
```

- [ ] **Step 2: Viết test `tests/test_gold_extractors.py`**

```python
from kgu.data.nba import Example
from kgu.extract.gold import GoldAllExtractor, GoldExplicitExtractor
from kgu.graph import BiTemporalGraph
from kgu.ops import diff
from kgu.types import Op, OpKind
from tests.conftest import PLAYS


def _ex(before, after, mentioned):
    return Example(0, "trade", "2021", "messi joins inter", mentioned, before, after)


def test_gold_all_returns_full_diff(before, after):
    ex = _ex(before, after, ["Messi", "Inter"])
    assert GoldAllExtractor().extract(ex, BiTemporalGraph.from_triples(before)) == diff(before, after)


def test_gold_explicit_keeps_only_ops_between_mentioned_entities(before, after):
    ex = _ex(before, after, ["Messi", "Inter", "Barca"])
    ops = GoldExplicitExtractor().extract(ex, BiTemporalGraph.from_triples(before))
    assert set(ops) == {
        Op(OpKind.INVALIDATE, "Messi", PLAYS, "Barca"),
        Op(OpKind.ADD, "Messi", PLAYS, "Inter"),
    }
```

- [ ] **Step 3: Chạy test, xác nhận fail**

Run: `python -m pytest tests/test_oracle_judge.py tests/test_gold_extractors.py -v`
Expected: FAIL `ModuleNotFoundError`

- [ ] **Step 4: Viết `kgu/judge/__init__.py`**

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from kgu.graph import BiTemporalGraph
from kgu.localize import Localization
from kgu.types import Op


@dataclass
class JudgeContext:
    text: str
    explicit_ops: list[Op]
    loc: Localization
    graph: BiTemporalGraph
    at: int
    relations: list[str]


class Judge(Protocol):
    def judge(self, ctx: JudgeContext) -> list[Op]: ...
```

- [ ] **Step 5: Viết `kgu/judge/oracle.py`**

```python
from __future__ import annotations

from kgu.judge import JudgeContext
from kgu.types import Op, OpKind, Triple


class OracleJudge:
    """Bộ phán nhìn KG-sau (gold). Chỉ dùng để đo cận trên của khoanh vùng."""

    def __init__(self, gold_after: set[Triple]) -> None:
        self.gold_after = gold_after

    def judge(self, ctx: JudgeContext) -> list[Op]:
        ops: list[Op] = []
        for c in ctx.loc.cut:
            if c.triple not in self.gold_after:
                ops.append(Op(OpKind.INVALIDATE, *c.triple, source="oracle"))
        seen: set[Triple] = set()
        for c in ctx.loc.add:
            pair = {c.subject, c.neighbor}
            for tr in sorted(self.gold_after):
                if {tr[0], tr[2]} == pair and tr not in seen and not ctx.graph.is_active(tr, ctx.at):
                    seen.add(tr)
                    ops.append(Op(OpKind.ADD, *tr, source="oracle"))
        return ops
```

- [ ] **Step 6: Viết `kgu/extract/__init__.py` và `kgu/extract/gold.py`**

`kgu/extract/__init__.py`:
```python
from __future__ import annotations

from typing import Protocol

from kgu.data.nba import Example
from kgu.graph import BiTemporalGraph
from kgu.types import Op


class Extractor(Protocol):
    def extract(self, ex: Example, graph: BiTemporalGraph) -> list[Op]: ...
```

`kgu/extract/gold.py`:
```python
from __future__ import annotations

from kgu.data.nba import Example
from kgu.graph import BiTemporalGraph
from kgu.ops import diff
from kgu.types import Op


class GoldAllExtractor:
    """Cận trên: toàn bộ diff trước/sau (không cần khoanh vùng)."""

    def extract(self, ex: Example, graph: BiTemporalGraph) -> list[Op]:
        return diff(ex.before, ex.after, source="gold_all")


class GoldExplicitExtractor:
    """Xấp xỉ ops 'được nói thẳng': cả 2 đầu cạnh đều là entity được nhắc trong text."""

    def extract(self, ex: Example, graph: BiTemporalGraph) -> list[Op]:
        mentioned = set(ex.mentioned)
        return [op for op in diff(ex.before, ex.after, source="gold_explicit")
                if op.h in mentioned and op.t in mentioned]
```

- [ ] **Step 7: Chạy test, xác nhận pass**

Run: `python -m pytest tests/test_oracle_judge.py tests/test_gold_extractors.py -v`
Expected: 3 passed

- [ ] **Step 8: Commit**

```bash
git add kgu/judge kgu/extract tests/test_oracle_judge.py tests/test_gold_extractors.py
git commit -m "feat: Judge/Extractor protocols, oracle judge, gold extractors"
```

---

### Task 8: Pipeline + coverage + CLI đánh giá (mốc tháng 1)

**Files:**
- Create: `kgu/pipeline.py`, `kgu/eval/coverage.py`, `scripts/run_eval.py`, `tests/test_pipeline.py`, `tests/test_coverage.py`
- Modify: `README.md`

**Interfaces:**
- Produces:
  - `kgu.pipeline.PipelineResult` dataclass: `idx, event, explicit_ops: list[Op], loc: Localization, judged_ops: list[Op], pred_after: set[Triple], n_llm_calls: int = 0`, `to_record(ex: Example) -> dict` (JSON-able: idx, event, n_explicit, n_cut_cand, n_add_cand, n_judged, counts, coverage).
  - `kgu.pipeline.run_example(ex: Example, extractor: Extractor, judge: Judge | None, relations: list[str]) -> PipelineResult` — `graph = from_triples(ex.before, 0)` → `explicit = extractor.extract` → `apply_ops(at=1)` → `loc = localize` → `judged = judge.judge(ctx)` nếu có judge → `apply_ops(judged, 1)` → `pred_after = graph.active(1)`.
  - `kgu.eval.coverage.coverage(ex, explicit_ops, loc) -> tuple[int, int]` = (số op ngầm được vùng ứng viên bao phủ, tổng số op ngầm). Op ngầm = `diff(before, after)` trừ `explicit`. INVALIDATE ngầm được phủ nếu triple ∈ `loc.cut_triples`; ADD ngầm được phủ nếu `frozenset({h,t})` ∈ `loc.add_pairs`.
  - CLI `scripts/run_eval.py --data-dir data/raw/nba --split test --extractor {gold_all,gold_explicit,llm} --judge {none,oracle,llm} [--limit N] --out results/<name>.jsonl`: ghi 1 dòng/record, in bảng tổng hợp (f1, add_acc, del_acc, coverage, avg #cut, avg #add, tổng llm calls), và ghi `results/<name>.summary.json`. Tùy chọn `llm` của extractor/judge sẽ được nối trong Task 10–11; ở task này để `raise SystemExit("llm not wired yet")`.

- [ ] **Step 1: Viết test `tests/test_coverage.py`**

```python
from kgu.eval.coverage import coverage
from kgu.data.nba import Example
from kgu.localize import localize
from kgu.ops import apply_ops
from kgu.types import Op, OpKind
from tests.conftest import PLAYS

EXPLICIT = [
    Op(OpKind.INVALIDATE, "Messi", PLAYS, "Barca"),
    Op(OpKind.ADD, "Messi", PLAYS, "Inter"),
]


def test_coverage_full_on_fixture(graph, before, after):
    ex = Example(0, "trade", "2021", "", ["Messi", "Barca", "Inter"], before, after)
    apply_ops(graph, EXPLICIT, at=1)
    loc = localize(graph, EXPLICIT, at=1)
    covered, total = coverage(ex, EXPLICIT, loc)
    assert total == 4          # 2 teammate cắt + 2 teammate thêm
    assert covered == 4


def test_coverage_zero_without_localization(before, after):
    from kgu.localize import Localization
    ex = Example(0, "trade", "2021", "", [], before, after)
    assert coverage(ex, EXPLICIT, Localization()) == (0, 4)
```

- [ ] **Step 2: Viết test `tests/test_pipeline.py`**

```python
from kgu.data.nba import Example
from kgu.extract.gold import GoldAllExtractor, GoldExplicitExtractor
from kgu.judge.oracle import OracleJudge
from kgu.pipeline import run_example
from kgu.eval.metrics import score, summarize
from tests.conftest import PLAYS, TEAM


def _ex(before, after):
    return Example(3, "trade", "2021", "messi joins inter", ["Messi", "Barca", "Inter"], before, after)


def test_gold_all_without_judge_is_perfect(before, after):
    res = run_example(_ex(before, after), GoldAllExtractor(), None, [PLAYS, TEAM])
    assert res.pred_after == after
    assert summarize(score(before, after, res.pred_after))["f1"] == 1.0


def test_gold_explicit_without_judge_misses_implied_edges(before, after):
    res = run_example(_ex(before, after), GoldExplicitExtractor(), None, [PLAYS, TEAM])
    s = summarize(score(before, after, res.pred_after))
    assert s["add_acc"] == 1 / 3 and s["del_acc"] == 1 / 3


def test_gold_explicit_with_oracle_is_perfect(before, after):
    res = run_example(_ex(before, after), GoldExplicitExtractor(), OracleJudge(after), [PLAYS, TEAM])
    assert res.pred_after == after
    rec = res.to_record(_ex(before, after))
    assert rec["idx"] == 3 and rec["n_explicit"] == 2 and rec["coverage"] == [4, 4]
    assert rec["counts"]["tp"] == len(after)
```

- [ ] **Step 3: Chạy test, xác nhận fail**

Run: `python -m pytest tests/test_coverage.py tests/test_pipeline.py -v`
Expected: FAIL `ModuleNotFoundError`

- [ ] **Step 4: Viết `kgu/eval/coverage.py`**

```python
from __future__ import annotations

from kgu.data.nba import Example
from kgu.localize import Localization
from kgu.ops import diff
from kgu.types import Op, OpKind


def coverage(ex: Example, explicit_ops: list[Op], loc: Localization) -> tuple[int, int]:
    """(số op ngầm nằm trong vùng ứng viên, tổng số op ngầm). Đo recall của khoanh vùng."""
    implied = [op for op in diff(ex.before, ex.after) if op not in set(explicit_ops)]
    covered = 0
    for op in implied:
        if op.kind is OpKind.INVALIDATE and op.triple in loc.cut_triples:
            covered += 1
        elif op.kind is OpKind.ADD and frozenset((op.h, op.t)) in loc.add_pairs:
            covered += 1
    return covered, len(implied)
```

- [ ] **Step 5: Viết `kgu/pipeline.py`**

```python
from __future__ import annotations

from dataclasses import dataclass, field

from kgu.data.nba import Example
from kgu.eval.coverage import coverage
from kgu.eval.metrics import score
from kgu.extract import Extractor
from kgu.graph import BiTemporalGraph
from kgu.judge import Judge, JudgeContext
from kgu.localize import Localization, localize
from kgu.ops import apply_ops
from kgu.types import Op, Triple

AT_BEFORE = 0
AT_AFTER = 1


@dataclass
class PipelineResult:
    idx: int
    event: str
    explicit_ops: list[Op]
    loc: Localization
    judged_ops: list[Op]
    pred_after: set[Triple]
    n_llm_calls: int = 0
    graph: BiTemporalGraph = field(default_factory=BiTemporalGraph, repr=False)

    def to_record(self, ex: Example) -> dict:
        return {
            "idx": self.idx,
            "event": self.event,
            "n_explicit": len(self.explicit_ops),
            "n_cut_cand": len(self.loc.cut),
            "n_add_cand": len(self.loc.add),
            "n_judged": len(self.judged_ops),
            "n_llm_calls": self.n_llm_calls,
            "coverage": list(coverage(ex, self.explicit_ops, self.loc)),
            "counts": score(ex.before, ex.after, self.pred_after).to_dict(),
            "explicit_ops": [o.to_dict() for o in self.explicit_ops],
            "judged_ops": [o.to_dict() for o in self.judged_ops],
        }


def _calls(*objs) -> int:
    return sum(getattr(o, "n_calls", 0) for o in objs if o is not None)


def run_example(ex: Example, extractor: Extractor, judge: Judge | None, relations: list[str]) -> PipelineResult:
    calls_before = _calls(extractor, judge)                      # extractor/judge dùng chung qua nhiều ví dụ
    graph = BiTemporalGraph.from_triples(ex.before, valid_from=AT_BEFORE)
    explicit = extractor.extract(ex, graph)
    explicit = apply_ops(graph, explicit, at=AT_AFTER)           # [2] áp ops TRƯỚC
    loc = localize(graph, explicit, at=AT_AFTER)                 # [3] khoanh vùng
    judged: list[Op] = []
    if judge is not None:                                        # [4] bộ phán
        ctx = JudgeContext(ex.text, explicit, loc, graph, AT_AFTER, relations)
        judged = apply_ops(graph, judge.judge(ctx), at=AT_AFTER)
    n_calls = _calls(extractor, judge) - calls_before            # chỉ đếm lệnh gọi của ví dụ này
    return PipelineResult(ex.idx, ex.event, explicit, loc, judged, graph.active(AT_AFTER), n_calls, graph)
```

- [ ] **Step 6: Chạy test, xác nhận pass**

Run: `python -m pytest tests/test_coverage.py tests/test_pipeline.py -v`
Expected: 5 passed

- [ ] **Step 7: Viết `scripts/run_eval.py`**

```python
"""Chạy pipeline trên 1 split với 1 cấu hình extractor/judge; ghi JSONL + summary."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from tqdm import tqdm

from kgu.data.nba import Vocab, load_split
from kgu.eval.metrics import Counts, summarize
from kgu.extract.gold import GoldAllExtractor, GoldExplicitExtractor
from kgu.judge.oracle import OracleJudge
from kgu.pipeline import run_example


def build_extractor(name: str, relations: list[str]):
    if name == "gold_all":
        return GoldAllExtractor()
    if name == "gold_explicit":
        return GoldExplicitExtractor()
    if name == "llm":
        from kgu.extract.llm import LLMExtractor
        from kgu.llm import client_from_env
        return LLMExtractor(client_from_env(), relations)
    raise SystemExit(f"unknown extractor {name}")


def build_llm_judge(name: str):
    """Judge LLM tạo MỘT lần (dùng chung client, đếm n_calls liên tục); oracle tạo per-example."""
    if name == "llm":
        from kgu.judge.llm import LLMJudge
        from kgu.llm import client_from_env
        return LLMJudge(client_from_env())
    if name in ("none", "oracle"):
        return None
    raise SystemExit(f"unknown judge {name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data/raw/nba"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--extractor", default="gold_explicit", choices=["gold_all", "gold_explicit", "llm"])
    ap.add_argument("--judge", default="none", choices=["none", "oracle", "llm"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    relations = Vocab.load(args.data_dir).relations
    examples = load_split(args.data_dir, args.split, limit=args.limit)
    extractor = build_extractor(args.extractor, relations)
    llm_judge = build_llm_judge(args.judge)

    total = Counts()
    cov_hit = cov_all = n_cut = n_add = n_calls = 0
    t0 = time.time()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf8") as f:
        for ex in tqdm(examples, desc=f"{args.extractor}+{args.judge}"):
            judge = OracleJudge(ex.after) if args.judge == "oracle" else llm_judge
            res = run_example(ex, extractor, judge, relations)
            rec = res.to_record(ex)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            total = total + Counts(**rec["counts"])
            cov_hit += rec["coverage"][0]
            cov_all += rec["coverage"][1]
            n_cut += rec["n_cut_cand"]
            n_add += rec["n_add_cand"]
            n_calls += rec["n_llm_calls"]

    s = summarize(total)
    s.update({
        "config": f"{args.extractor}+{args.judge}", "split": args.split, "n": len(examples),
        "coverage": cov_hit / cov_all if cov_all else 0.0,
        "avg_cut_cand": n_cut / len(examples), "avg_add_cand": n_add / len(examples),
        "llm_calls": n_calls, "seconds": round(time.time() - t0, 1),
    })
    args.out.with_suffix(".summary.json").write_text(json.dumps(s, indent=2), encoding="utf8")
    print(json.dumps(s, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 8: Chạy 3 cấu hình mốc tháng 1 trên split test**

```bash
python scripts/run_eval.py --extractor gold_all      --judge none   --out results/nba_test_gold_all.jsonl
python scripts/run_eval.py --extractor gold_explicit --judge none   --out results/nba_test_gold_explicit_nojudge.jsonl
python scripts/run_eval.py --extractor gold_explicit --judge oracle --out results/nba_test_gold_explicit_oracle.jsonl
```
Expected (đã kiểm chứng trước bằng đúng code trong plan này trên split test, 438 ví dụ, 2026-09-07):

| Cấu hình | f1 | add_acc | del_acc | coverage | avg_cut_cand | avg_add_cand |
|---|---|---|---|---|---|---|
| gold_all+none | 1.000 | 1.000 | 1.000 | 0.0 (0/0, không có op ngầm) | 6.5 | 2.5 |
| gold_explicit+none | 0.926 | 0.056 | 0.056 | 1.0 | 70.3 | 31.0 |
| gold_explicit+oracle | 1.000 | 1.000 | 1.000 | 1.0 | 70.3 | 31.0 |

- `gold_all+none` = 1.0 chứng minh loader + executor + metric đúng.
- `gold_explicit+none` cho add/del ≈ 0,056 — trùng con số "5,6% thay đổi được nói thẳng" của paper → cách xấp xỉ ops tường minh hợp lý; F1 0,926 gần baseline IE thuần (0,9429).
- `gold_explicit+oracle` coverage = 1.0 → khoanh vùng 2 vế bao phủ toàn bộ cạnh ngầm trên NBA. Nếu lần chạy lại lệch khỏi bảng trên → có regression, tìm nguyên nhân trước khi đi tiếp.
- Ghi chú cho tháng 2: mỗi tin có ~70 ứng viên cắt + ~31 ứng viên thêm → prompt bộ phán LLM dài (~100 dòng ứng viên) nhưng vẫn vừa 1 lệnh gọi.

- [ ] **Step 9: Ghi số vào README**

Thêm vào `README.md` mục "Kết quả sơ bộ (NBAtransactions test)" với bảng 3 dòng cấu hình × cột f1 / add_acc / del_acc / coverage / avg_cut / avg_add, và câu lệnh chạy lại. Đồng thời mục "Cài đặt": tạo venv, `pip install -e ".[dev]"`, `python scripts/download_nba.py`, `python -m pytest`.

- [ ] **Step 10: Commit**

```bash
git add kgu/pipeline.py kgu/eval/coverage.py scripts/run_eval.py tests/test_pipeline.py tests/test_coverage.py README.md docs/data-notes.md
git commit -m "feat: end-to-end pipeline, localization coverage, eval CLI; month-1 gold-ops numbers"
```

---

### Task 9: `LLMClient` (OpenAI-compatible) + `FakeLLMClient`

**Files:**
- Create: `kgu/llm.py`, `tests/test_llm_client.py`

**Interfaces:**
- Produces:
  - `kgu.llm.LLMClient` Protocol: `complete_json(system: str, user: str, schema: type[T]) -> T` với `T` là `pydantic.BaseModel`.
  - `kgu.llm.FakeLLMClient(responses: list[dict])` — trả lần lượt từng dict qua `schema.model_validate`; lưu `calls: list[tuple[str, str]]`.
  - `kgu.llm.OpenAICompatClient(base_url, api_key, model, temperature=0.0, max_retries=3, timeout=120)` — gọi `chat.completions` với `response_format={"type": "json_object"}`; nhúng JSON schema vào system prompt; parse bằng `schema.model_validate_json`; khi lỗi thì gửi lại kèm thông báo lỗi; hết retry → raise `LLMOutputError`.
  - `kgu.llm.client_from_env() -> OpenAICompatClient` đọc `KGU_LLM_BASE_URL` (mặc định `http://localhost:8080/v1` — llama-server local), `KGU_LLM_API_KEY` (mặc định `"none"`), `KGU_LLM_MODEL` (mặc định `local`).
  - `kgu.llm.extract_json(text: str) -> str` — cắt phần từ `{` đầu đến `}` cuối, bỏ ```json fence nếu có.

- [ ] **Step 1: Viết test**

`tests/test_llm_client.py`:

```python
import pytest
from pydantic import BaseModel

from kgu.llm import FakeLLMClient, LLMOutputError, OpenAICompatClient, extract_json


class Out(BaseModel):
    answer: int


def test_fake_client_returns_in_order_and_records_calls():
    c = FakeLLMClient([{"answer": 1}, {"answer": 2}])
    assert c.complete_json("sys", "u1", Out).answer == 1
    assert c.complete_json("sys", "u2", Out).answer == 2
    assert c.calls == [("sys", "u1"), ("sys", "u2")]
    assert c.n_calls == 2


def test_extract_json_strips_fences_and_prose():
    fence = "`" * 3
    assert extract_json(f'Sure!\n{fence}json\n{{"answer": 3}}\n{fence}\nDone.') == '{"answer": 3}'


class _FakeCompletions:
    def __init__(self, contents):
        self.contents = list(contents)
        self.kwargs = []

    def create(self, **kwargs):
        self.kwargs.append(kwargs)
        content = self.contents.pop(0)
        msg = type("M", (), {"content": content})()
        choice = type("C", (), {"message": msg})()
        return type("R", (), {"choices": [choice]})()


def _client_with(contents):
    c = OpenAICompatClient(base_url="http://x", api_key="k", model="m", max_retries=2)
    c._completions = _FakeCompletions(contents)
    return c


def test_openai_client_retries_on_invalid_json_then_succeeds():
    c = _client_with(['{"answer": "not-int"}', '{"answer": 5}'])
    assert c.complete_json("sys", "user", Out).answer == 5
    assert c.n_calls == 2
    assert "not-int" in c._completions.kwargs[1]["messages"][-1]["content"] or "error" in c._completions.kwargs[1]["messages"][-1]["content"].lower()


def test_openai_client_raises_after_retries():
    c = _client_with(["garbage", "still garbage"])
    with pytest.raises(LLMOutputError):
        c.complete_json("sys", "user", Out)
```

- [ ] **Step 2: Chạy test, xác nhận fail**

Run: `python -m pytest tests/test_llm_client.py -v`
Expected: FAIL `ModuleNotFoundError: No module named 'kgu.llm'`

- [ ] **Step 3: Viết `kgu/llm.py`**

```python
from __future__ import annotations

import json
import os
import re
from typing import Protocol, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


class LLMOutputError(RuntimeError):
    pass


class LLMClient(Protocol):
    n_calls: int

    def complete_json(self, system: str, user: str, schema: type[T]) -> T: ...


def extract_json(text: str) -> str:
    text = re.sub(r"`{3}(?:json)?", "", text)   # bỏ fence markdown (không viết 3 backtick liền để không phá markdown)
    start, end = text.find("{"), text.rfind("}")
    return text[start:end + 1] if start != -1 and end > start else text.strip()


class FakeLLMClient:
    def __init__(self, responses: list[dict]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, str]] = []
        self.n_calls = 0

    def complete_json(self, system: str, user: str, schema: type[T]) -> T:
        self.calls.append((system, user))
        self.n_calls += 1
        return schema.model_validate(self.responses.pop(0))


class OpenAICompatClient:
    """Gọi bất kỳ endpoint OpenAI-compatible (vLLM, Ollama, OpenRouter, DashScope)."""

    def __init__(self, base_url: str, api_key: str, model: str, temperature: float = 0.0,
                 max_retries: int = 3, timeout: float = 120.0) -> None:
        self.model, self.temperature, self.max_retries = model, temperature, max_retries
        self.n_calls = 0
        self._completions = None
        self._init = dict(base_url=base_url, api_key=api_key, timeout=timeout)

    @property
    def completions(self):
        if self._completions is None:
            from openai import OpenAI
            self._completions = OpenAI(**self._init).chat.completions
        return self._completions

    def complete_json(self, system: str, user: str, schema: type[T]) -> T:
        system_full = (
            f"{system}\n\nTrả lời DUY NHẤT một JSON object hợp lệ theo schema sau, không giải thích:\n"
            + json.dumps(schema.model_json_schema(), ensure_ascii=False)
        )
        messages = [{"role": "system", "content": system_full}, {"role": "user", "content": user}]
        last_err: Exception | None = None
        for _ in range(self.max_retries):
            self.n_calls += 1
            resp = self.completions.create(
                model=self.model, messages=messages, temperature=self.temperature,
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content or ""
            try:
                return schema.model_validate_json(extract_json(raw))
            except (ValidationError, ValueError) as e:
                last_err = e
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user", "content":
                    f"Output không hợp lệ (error: {str(e)[:500]}). Xuất lại đúng schema, chỉ JSON."})
        raise LLMOutputError(f"invalid JSON after {self.max_retries} tries: {last_err}")


def client_from_env() -> OpenAICompatClient:
    """Mặc định: llama-server local (scripts/llm_server.ps1) trên http://localhost:8080/v1."""
    return OpenAICompatClient(
        base_url=os.environ.get("KGU_LLM_BASE_URL", "http://localhost:8080/v1"),
        api_key=os.environ.get("KGU_LLM_API_KEY", "none"),
        model=os.environ.get("KGU_LLM_MODEL", "local"),
    )
```

- [ ] **Step 4: Chạy test, xác nhận pass**

Run: `python -m pytest tests/test_llm_client.py -v`
Expected: 4 passed

- [ ] **Step 5: Ghi cách cấu hình LLM vào README**

Thêm mục "Cấu hình LLM" với các lựa chọn và biến môi trường. Mặc định của `client_from_env()` là **A** (đã dựng sẵn 2026-09-07: `scripts/llm_server.ps1` + `models/Qwen3.5-2B-Q8_0.gguf` + `tools/llama.cpp/` bản Vulkan chạy trên AMD Radeon 780M; kiểm tra bằng `python scripts/llm_smoke.py`):
```
# A. Local llama.cpp (Vulkan, AMD 780M) — MẶC ĐỊNH.  Khởi động:  .\scripts\llm_server.ps1
set KGU_LLM_BASE_URL=http://localhost:8080/v1   KGU_LLM_MODEL=local   KGU_LLM_API_KEY=none
#    Model khác: .\scripts\llm_server.ps1 -Model models\<file>.gguf  (GGUF Qwen2.5-7B-Instruct Q4_K_M ~4,7 GB cho số cuối)
# B. Colab + vLLM (Qwen2.5-7B-Instruct, GPU T4) qua ngrok/cloudflared:
set KGU_LLM_BASE_URL=https://<tunnel>/v1   KGU_LLM_MODEL=Qwen/Qwen2.5-7B-Instruct  KGU_LLM_API_KEY=none
# C. OpenRouter (hosted):
set KGU_LLM_BASE_URL=https://openrouter.ai/api/v1  KGU_LLM_MODEL=qwen/qwen-2.5-7b-instruct  KGU_LLM_API_KEY=<key>
```
Vì llama-server hỗ trợ `response_format={"type":"json_schema", ...}` (constrained decoding thật, thay cho outlines/xgrammar trong spec), `OpenAICompatClient.complete_json` nên gửi `json_schema` khi có, và chỉ lùi về `json_object` nếu server trả lỗi 400 — sửa đoạn `response_format` trong Task 9 Step 3 thành:
```python
            # Qwen3.5 là model có thinking: tắt qua chat_template_kwargs (llama-server đọc extra_body),
            # nếu không nội dung sẽ nằm trong reasoning_content và content rỗng.
            extra = {"chat_template_kwargs": {"enable_thinking": False}}
            try:
                resp = self.completions.create(
                    model=self.model, messages=messages, temperature=self.temperature,
                    response_format={"type": "json_schema",
                                     "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema()}},
                    extra_body=extra,
                )
            except Exception:  # server không hỗ trợ json_schema → lùi về json_object
                resp = self.completions.create(
                    model=self.model, messages=messages, temperature=self.temperature,
                    response_format={"type": "json_object"}, extra_body=extra,
                )
```
Đã đo trên máy dev (2026-09-07, Qwen3.5-2B Q8_0, Vulkan 780M): sinh ~20 tok/s, đọc prompt ~106 tok/s, JSON schema được tuân thủ 100% (constrained decoding). Ước tính split test 438 tin × 2 lệnh gọi ≈ 1–2 giờ với 2B.

- [ ] **Step 6: Commit**

```bash
git add kgu/llm.py tests/test_llm_client.py README.md
git commit -m "feat: OpenAI-compatible LLM client with JSON retry + fake client for tests"
```

---

### Task 10: LLM #1 — `LLMExtractor` (ops tường minh từ text)

**Files:**
- Create: `kgu/extract/llm.py`, `tests/test_llm_extractor.py`

**Interfaces:**
- Consumes: `LLMClient`, `Example`, `BiTemporalGraph`, `Op`
- Produces: `kgu.extract.llm.LLMExtractor(client: LLMClient, relations: list[str], max_edges_per_entity: int = 40)` với `extract(ex, graph) -> list[Op]` và thuộc tính `n_calls` (ủy quyền `client.n_calls`).
- Prompt: system giải thích nhiệm vụ (chỉ ghi điều văn bản khẳng định, không suy đoán, không tạo entity mới ngoài danh sách), user chứa: text, danh sách quan hệ, các entity được nhắc kèm cạnh hiện tại của chúng (tối đa `max_edges_per_entity` mỗi entity). Schema output:
  ```python
  class OpOut(BaseModel): kind: Literal["ADD","INVALIDATE"]; h: str; r: str; t: str; quote: str = ""
  class OpsOut(BaseModel): ops: list[OpOut]
  ```
- Lọc sau LLM (bắt buộc): `r ∈ relations`; `h`, `t` ∈ `graph.entities(0) ∪ ex.mentioned`; INVALIDATE chỉ giữ nếu triple đang active tại 0; ADD chỉ giữ nếu chưa active. Op giữ lại có `source="llm"`, `quote` từ output, `conf=1.0`. Ops bị loại được đếm vào `self.n_dropped`.

- [ ] **Step 1: Viết test**

`tests/test_llm_extractor.py`:

```python
from kgu.data.nba import Example
from kgu.extract.llm import LLMExtractor
from kgu.graph import BiTemporalGraph
from kgu.llm import FakeLLMClient
from kgu.types import Op, OpKind
from tests.conftest import PLAYS, TEAM


def _ex(before, after):
    return Example(0, "trade", "2021", "Messi leaves Barca and joins Inter.", ["Messi", "Barca", "Inter"], before, after)


def test_extractor_parses_and_filters(before, after):
    client = FakeLLMClient([{"ops": [
        {"kind": "INVALIDATE", "h": "Messi", "r": PLAYS, "t": "Barca", "quote": "leaves Barca"},
        {"kind": "ADD", "h": "Messi", "r": PLAYS, "t": "Inter", "quote": "joins Inter"},
        {"kind": "ADD", "h": "Messi", "r": "owns", "t": "Inter"},            # quan hệ lạ → bỏ
        {"kind": "ADD", "h": "Messi", "r": PLAYS, "t": "Mars FC"},          # entity lạ → bỏ
        {"kind": "INVALIDATE", "h": "Messi", "r": PLAYS, "t": "Inter"},     # không active → bỏ
        {"kind": "ADD", "h": "Pedri", "r": PLAYS, "t": "Barca"},            # đã active → bỏ
    ]}])
    ext = LLMExtractor(client, relations=[PLAYS, TEAM])
    ops = ext.extract(_ex(before, after), BiTemporalGraph.from_triples(before))
    assert ops == [
        Op(OpKind.INVALIDATE, "Messi", PLAYS, "Barca"),
        Op(OpKind.ADD, "Messi", PLAYS, "Inter"),
    ]
    assert ops[0].source == "llm" and ops[0].quote == "leaves Barca"
    assert ext.n_dropped == 4 and ext.n_calls == 1


def test_prompt_contains_text_relations_and_mentioned_edges(before, after):
    client = FakeLLMClient([{"ops": []}])
    LLMExtractor(client, relations=[PLAYS, TEAM]).extract(_ex(before, after), BiTemporalGraph.from_triples(before))
    system, user = client.calls[0]
    assert "Messi leaves Barca" in user
    assert PLAYS in user and TEAM in user
    assert "(Messi, plays_for, Barca)" in user
    assert "ADD" in system and "INVALIDATE" in system
```

- [ ] **Step 2: Chạy test, xác nhận fail**

Run: `python -m pytest tests/test_llm_extractor.py -v`
Expected: FAIL `ModuleNotFoundError: No module named 'kgu.extract.llm'`

- [ ] **Step 3: Viết `kgu/extract/llm.py`**

```python
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
```

- [ ] **Step 4: Chạy test, xác nhận pass**

Run: `python -m pytest tests/test_llm_extractor.py -v`
Expected: 2 passed

- [ ] **Step 5: Chạy thử với LLM thật trên 20 ví dụ**

Cấu hình biến môi trường theo README rồi:
Run: `python scripts/run_eval.py --extractor llm --judge none --limit 20 --out results/nba_test_llm_nojudge_20.jsonl`
Expected: chạy hết 20 ví dụ không exception; in summary. Mở JSONL, đọc `explicit_ops` của 5 record đầu và so với `gold_explicit` cùng idx trong `results/nba_test_gold_explicit_nojudge.jsonl`. Nếu LLM hay trả entity sai chính tả (vd thiếu dấu gạch dưới), thêm bước chuẩn hóa `name.replace(" ", "_")` trước khi kiểm tra `known` và ghi nhận vào `docs/data-notes.md`.

- [ ] **Step 6: Commit**

```bash
git add kgu/extract/llm.py tests/test_llm_extractor.py
git commit -m "feat: LLM extractor for explicit ops with post-validation"
```

---

### Task 11: Bộ phán A — `LLMJudge` (1 lệnh gọi cho toàn bộ danh sách)

**Files:**
- Create: `kgu/judge/llm.py`, `tests/test_llm_judge.py`

**Interfaces:**
- Consumes: `LLMClient`, `JudgeContext`, `CutCandidate`, `AddCandidate`
- Produces: `kgu.judge.llm.LLMJudge(client: LLMClient)` với `judge(ctx) -> list[Op]`, thuộc tính `n_calls`. Nếu `ctx.loc` rỗng → không gọi LLM, trả `[]`.
- Schema output:
  ```python
  class CutVerdict(BaseModel): idx: int; verdict: Literal["KEEP", "INVALIDATE"]
  class AddVerdict(BaseModel): idx: int; add: bool; r: str = ""; direction: Literal["subject_first", "neighbor_first"] = "subject_first"
  class JudgeOut(BaseModel): cut: list[CutVerdict] = []; add: list[AddVerdict] = []
  ```
- Prompt user: text; ops tường minh đã áp; danh sách cut candidate đánh số `[C0]..` dạng `(h, r, t)` kèm `hub` và `anchors`; danh sách add candidate `[A0]..` dạng `subject=…, neighbor=…, bằng chứng via=(…)`; danh sách quan hệ. System prompt nêu rõ: fact trạng thái có thể vô hiệu, fact sự kiện lịch sử thì KEEP; chỉ ADD khi cấu trúc mới thực sự hàm ý; không đoán sự kiện chưa xảy ra.
- Lọc: idx ngoài phạm vi → bỏ; ADD với `r ∉ relations` → bỏ; ADD tạo triple đã active → bỏ. Op có `source="llm_judge"`.

- [ ] **Step 1: Viết test**

`tests/test_llm_judge.py`:

```python
from kgu.judge import JudgeContext
from kgu.judge.llm import LLMJudge
from kgu.llm import FakeLLMClient
from kgu.localize import Localization, localize
from kgu.ops import apply_ops
from kgu.types import Op, OpKind
from tests.conftest import FRIEND, PLAYS, TEAM

EXPLICIT = [
    Op(OpKind.INVALIDATE, "Messi", PLAYS, "Barca"),
    Op(OpKind.ADD, "Messi", PLAYS, "Inter"),
]


def _ctx(graph):
    apply_ops(graph, EXPLICIT, at=1)
    loc = localize(graph, EXPLICIT, at=1)
    return JudgeContext("Messi leaves Barca and joins Inter.", EXPLICIT, loc, graph, 1, [PLAYS, TEAM, FRIEND])


def test_judge_maps_verdicts_to_ops(graph):
    ctx = _ctx(graph)
    cut_idx = {c.triple: i for i, c in enumerate(ctx.loc.cut)}
    add_idx = {c.neighbor: i for i, c in enumerate(ctx.loc.add)}
    client = FakeLLMClient([{
        "cut": [
            {"idx": cut_idx[("Pedri", TEAM, "Messi")], "verdict": "INVALIDATE"},
            {"idx": cut_idx[("Messi", TEAM, "Pedri")], "verdict": "INVALIDATE"},
            {"idx": cut_idx[("Pedri", PLAYS, "Barca")], "verdict": "KEEP"},
            {"idx": cut_idx[("Pedri", FRIEND, "Messi")], "verdict": "KEEP"},
            {"idx": 99, "verdict": "INVALIDATE"},                      # ngoài phạm vi → bỏ
        ],
        "add": [
            {"idx": add_idx["Lautaro"], "add": True, "r": TEAM, "direction": "neighbor_first"},
            {"idx": add_idx["Inzaghi"], "add": False},
        ],
    }])
    ops = LLMJudge(client).judge(ctx)
    assert set(ops) == {
        Op(OpKind.INVALIDATE, "Pedri", TEAM, "Messi"),
        Op(OpKind.INVALIDATE, "Messi", TEAM, "Pedri"),
        Op(OpKind.ADD, "Lautaro", TEAM, "Messi"),
    }
    assert all(o.source == "llm_judge" for o in ops)
    assert client.n_calls == 1


def test_judge_skips_llm_when_no_candidates(graph):
    client = FakeLLMClient([])
    ctx = JudgeContext("", [], Localization(), graph, 1, [PLAYS])
    assert LLMJudge(client).judge(ctx) == [] and client.n_calls == 0


def test_prompt_lists_candidates_with_indices(graph):
    ctx = _ctx(graph)
    client = FakeLLMClient([{"cut": [], "add": []}])
    LLMJudge(client).judge(ctx)
    _, user = client.calls[0]
    assert "[C0]" in user and "[A0]" in user and "Lautaro" in user and "hub=Pedri" in user
```

- [ ] **Step 2: Chạy test, xác nhận fail**

Run: `python -m pytest tests/test_llm_judge.py -v`
Expected: FAIL `ModuleNotFoundError: No module named 'kgu.judge.llm'`

- [ ] **Step 3: Viết `kgu/judge/llm.py`**

```python
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
```

- [ ] **Step 4: Chạy test, xác nhận pass**

Run: `python -m pytest tests/test_llm_judge.py -v`
Expected: 3 passed

- [ ] **Step 5: Chạy toàn bộ test**

Run: `python -m pytest -q`
Expected: tất cả pass, 0 failed.

- [ ] **Step 6: Chạy các cấu hình tháng 2 (bảng 1 sơ bộ)**

```bash
python scripts/run_eval.py --extractor gold_explicit --judge llm --limit 50 --out results/nba_test_goldexp_llmjudge_50.jsonl
python scripts/run_eval.py --extractor llm --judge llm --limit 50 --out results/nba_test_llm_llm_50.jsonl
```
Expected: cả hai chạy hết; số `llm_calls` ≈ 50 (judge) và ≈ 100 (extractor + judge). Ghi f1 / add_acc / del_acc vào README bảng "Kết quả sơ bộ" (đánh dấu là n=50). Khi endpoint ổn định, chạy lại không `--limit` để có số toàn split test.

- [ ] **Step 7: Commit**

```bash
git add kgu/judge/llm.py tests/test_llm_judge.py README.md
git commit -m "feat: single-call LLM judge over localized candidates; preliminary table 1"
```

---

### Task 12: Phân tích lỗi theo loại (kết thúc tháng 2)

**Files:**
- Create: `scripts/analyze_errors.py`, `tests/test_analyze_errors.py`

**Interfaces:**
- Produces: `scripts/analyze_errors.py <results.jsonl> [--by event|n_gold|coverage]` in bảng: nhóm → n, f1, add_acc, del_acc, avg cut cand, avg add cand. Hàm thuần `group_records(records: list[dict], by: str) -> dict[str, Counts]` để test.
  - `by=event`: khóa = `rec["event"]`.
  - `by=n_gold`: khóa = bucket của `counts.n_add + counts.n_del`: `"0-10"`, `"11-30"`, `"31+"`.
  - `by=coverage`: khóa = `"full"` nếu `coverage[0] == coverage[1]` else `"partial"`.

- [ ] **Step 1: Viết test**

`tests/test_analyze_errors.py`:

```python
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("analyze_errors", Path("scripts/analyze_errors.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def _rec(event, n_add, n_del, cov):
    return {"event": event, "coverage": cov, "n_cut_cand": 3, "n_add_cand": 2,
            "counts": {"tp": 5, "fp": 1, "fn": 1, "tn": 0, "add_hit": 1, "n_add": n_add, "del_hit": 1, "n_del": n_del}}


def test_group_by_event():
    g = mod.group_records([_rec("trade", 2, 2, [4, 4]), _rec("trade", 1, 1, [1, 2]), _rec("waive", 0, 5, [0, 0])], "event")
    assert set(g) == {"trade", "waive"}
    assert g["trade"].tp == 10 and g["waive"].n_del == 5


def test_group_by_n_gold_buckets():
    g = mod.group_records([_rec("x", 2, 2, [0, 0]), _rec("x", 20, 5, [0, 0]), _rec("x", 30, 30, [0, 0])], "n_gold")
    assert set(g) == {"0-10", "11-30", "31+"}


def test_group_by_coverage():
    g = mod.group_records([_rec("x", 2, 2, [4, 4]), _rec("x", 2, 2, [1, 4])], "coverage")
    assert set(g) == {"full", "partial"}
```

- [ ] **Step 2: Chạy test, xác nhận fail**

Run: `python -m pytest tests/test_analyze_errors.py -v`
Expected: FAIL (file không tồn tại → `FileNotFoundError` hoặc `AttributeError`)

- [ ] **Step 3: Viết `scripts/analyze_errors.py`**

```python
"""Nhóm kết quả eval theo loại để tìm chỗ pipeline yếu."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from kgu.eval.metrics import Counts, summarize


def _bucket(n: int) -> str:
    return "0-10" if n <= 10 else "11-30" if n <= 30 else "31+"


def _key(rec: dict, by: str) -> str:
    if by == "event":
        return str(rec.get("event", ""))
    if by == "n_gold":
        return _bucket(rec["counts"]["n_add"] + rec["counts"]["n_del"])
    if by == "coverage":
        c = rec["coverage"]
        return "full" if c[0] == c[1] else "partial"
    raise ValueError(by)


def group_records(records: list[dict], by: str) -> dict[str, Counts]:
    groups: dict[str, Counts] = defaultdict(Counts)
    for rec in records:
        groups[_key(rec, by)] = groups[_key(rec, by)] + Counts(**rec["counts"])
    return dict(groups)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", type=Path)
    ap.add_argument("--by", default="event", choices=["event", "n_gold", "coverage"])
    args = ap.parse_args()
    records = [json.loads(l) for l in args.results.read_text(encoding="utf8").splitlines() if l.strip()]
    sizes: dict[str, int] = defaultdict(int)
    cands: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for rec in records:
        k = _key(rec, args.by)
        sizes[k] += 1
        cands[k][0] += rec["n_cut_cand"]
        cands[k][1] += rec["n_add_cand"]
    print(f"{'group':<14}{'n':>6}{'f1':>8}{'add':>8}{'del':>8}{'cut/ex':>8}{'add/ex':>8}")
    for k, c in sorted(group_records(records, args.by).items()):
        s = summarize(c)
        n = sizes[k]
        print(f"{k:<14}{n:>6}{s['f1']:>8.3f}{s['add_acc']:>8.3f}{s['del_acc']:>8.3f}"
              f"{cands[k][0]/n:>8.1f}{cands[k][1]/n:>8.1f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Chạy test, xác nhận pass**

Run: `python -m pytest tests/test_analyze_errors.py -v`
Expected: 3 passed

- [ ] **Step 5: Chạy phân tích trên kết quả đã có và ghi nhận**

```bash
python scripts/analyze_errors.py results/nba_test_gold_explicit_oracle.jsonl --by coverage
python scripts/analyze_errors.py results/nba_test_gold_explicit_oracle.jsonl --by event
python scripts/analyze_errors.py results/nba_test_llm_llm_50.jsonl --by event
```
Viết `docs/error-analysis-month2.md`: 3 bảng output ở trên + 5 ví dụ lỗi cụ thể (idx, text, op sai/thiếu, nguyên nhân giả định: tin nhiều giao dịch / phủ định / entity mới / khoanh vùng bỏ sót / LLM sai chính tả entity).

- [ ] **Step 6: Commit**

```bash
git add scripts/analyze_errors.py tests/test_analyze_errors.py docs/error-analysis-month2.md
git commit -m "feat: error analysis by event / gold size / coverage; month-2 notes"
```

---

## Ngoài phạm vi plan này (plan riêng, theo lộ trình tháng 3–4)

1. **Bộ phán B/C** — `kgu/judge/anyburl.py` (xuất KG tại `at` ra file triple, chạy AnyBURL bằng Java 11, đọc luật + confidence, chấm ứng viên vế sinh, ngưỡng 0,9, vùng xám → `LLMJudge`), `kgu/judge/ultra.py`, bảng 2 ablation.
2. **Neo4j projection + demo Streamlit** — `kgu/store/neo4j.py` nạp `BiTemporalGraph.to_records()` với constraint unique; app dán tin → xem graph đổi.
3. **WikiFactDiff adapter** — `kgu/data/wikifactdiff.py` trả cùng kiểu `Example`; bảng 3 leave-one-domain-out.
4. **Dataset tiếng Việt** + đo chi phí nạp miền mới.
5. **Fine-tune** instruction-tune đa miền trên Colab (nhãn tự sinh từ `diff`).

## Self-review (đã chạy khi viết plan)

- Spec coverage tháng 1–2: [1] LLM ops → Task 10; [2] executor bi-temporal → Task 3–4; [3] khoanh vùng 2 vế → Task 6; [4] bộ phán A → Task 11 (B/C là plan sau, đúng lộ trình tháng 3); [5] đo so KG-sau → Task 5, 8; ops vàng + mốc tháng 1 → Task 8; bảng 1 sơ bộ + phân tích lỗi → Task 11–12; JSONL provenance → Task 4; "không xóa" → Task 3; "fact lịch sử KEEP" → system prompt Task 11.
- Type consistency: `Op(kind, h, r, t, source, quote, conf)`; `localize(graph, ops, at)`; `JudgeContext(text, explicit_ops, loc, graph, at, relations)`; `run_example(ex, extractor, judge, relations)`; `score(before, gold_after, pred_after) -> Counts`; `coverage(ex, explicit_ops, loc) -> (int, int)` — dùng thống nhất ở mọi task.
- Giả định cần xác nhận ở Task 1 spike: hướng của 4 quan hệ NBA và `<teammate>` có lưu 2 chiều không. Fixture test đã cố tình lưu teammate 2 chiều để pipeline không phụ thuộc giả định này.
