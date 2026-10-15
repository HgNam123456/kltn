# Rút câu đơn + gộp tên k-means/LLM cho TemporalKG — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Thêm vào pipeline COVID-NYT (`TemporalKG`) bước rút đoạn tin thành câu đơn rồi mới trích triple (có trích bổ sung cho câu bị sót), và bước gộp tên entity/relation định kỳ bằng k-means + một lệnh gọi LLM mỗi cụm; đo bằng ablation v0/v1/v2 trên Kaggle.

**Architecture:** `kgu/extract/condense.py` (rút câu + phát hiện mục sót) được `TemporalKG.update` gọi khi `condense=True`; `kgu/extract/canonicalize.py` (k-means numpy, nhóm ứng viên theo ngưỡng τ, `Canonicalizer` gọi LLM theo cụm và đổi tên trong KG) được `scripts/run_covid_nyt.py` gọi mỗi N nhóm ngày; `kgu/eval/atom_merge.py` cài lại chỉ số gộp tên của ATOM. Kernel `kaggle/covid_v2/` chạy v0/v1/v2 và chấm.

**Tech Stack:** Python 3.11, pydantic 2 (schema động bằng `create_model` + `Literal` để llama-server ép enum), numpy, sentence-transformers (chỉ trên Kaggle), pytest với `kgu.llm.FakeLLMClient`; Kaggle CLI, llama-server (Gemma 4 E4B).

**Spec:** `docs/superpowers/specs/2026-10-15-condense-canonicalize-design.md`

## Global Constraints

- LLM và GPU chỉ chạy trên Kaggle (kernel), không bật llama-server hay nạp embedder trên máy local; máy local chỉ chạy unit test.
- Không thêm scikit-learn; k-means viết bằng numpy (numpy đã có trong extra `bench`).
- `TemporalKG(condense=False)` (mặc định) giữ nguyên hành vi v0: các test hiện có trong `tests/test_text_kg.py` phải pass không sửa.
- Embedder gộp tên: `sentence-transformers/all-mpnet-base-v2`; bộ chấm trích xuất: all-mpnet-base-v2 @ 0,7; chỉ số gộp tên của ATOM: ngưỡng 0,8.
- Mặc định: τ = 0,75; cluster_size = 10; nhóm tối đa 8 tên; tối đa 3 fact ngữ cảnh mỗi tên; gộp tên mỗi 10 nhóm ngày và sau nhóm cuối.
- Chỉ chỉnh prompt và τ trên 20 nhóm ngày đầu (dev); số chính báo trên 102 nhóm held-out.
- Docstring/comment tiếng Việt như codebase; prompt gửi LLM bằng tiếng Anh. Phần tĩnh của prompt đặt trước, văn bản thay đổi đặt cuối (tận dụng KV cache).
- Chạy test: `.venv/Scripts/python.exe -m pytest <file> -q` từ thư mục repo (Git Bash trên Windows).
- Mọi commit kết thúc bằng dòng `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`.

## Review Focus

- LLM trả `sentence` ngoài khoảng 1..n (0, n+1, 99): fact vẫn được giữ, câu đó không được coi là đã có fact, câu thật sự thiếu vẫn được trích bổ sung — test ở Task 2.
- LLM trả quyết định cho chính tên mốc hoặc cho tên thuộc nhóm khác/cụm khác: bị bỏ qua, không gộp nhầm — test ở Task 4.
- Lượt gộp tên chạy khi KG rỗng hoặc chỉ có 1 tên: không gọi embedder, không gọi LLM, không lỗi — test ở Task 4.
- Rút câu trả JSON hỏng (output bị cắt do quá dài): quay về trích trên đoạn gốc như v0 và ghi `condense_error` — test ở Task 2.
- Embedder trả vector chưa chuẩn hóa: vẫn tính cosine đúng (chuẩn hóa trong `Canonicalizer.run`) — test ở Task 4.

---

### Task 1: Rút câu đơn và phát hiện mục sót (`kgu/extract/condense.py`)

**Files:**
- Create: `kgu/extract/condense.py`
- Test: `tests/test_condense.py`

**Interfaces:**
- Consumes: `llm.complete_json(system: str, user: str, schema: type[BaseModel]) -> BaseModel` (`kgu/llm.py`).
- Produces: `condense(text: str, llm) -> tuple[list[str], dict]` (thống kê có khóa `sentences`, `missing_before`, `missing_after`, có thể có `fill_error`); `missing_items(text: str, sentences: list[str], limit: int = 15) -> list[str]`.

- [ ] **Step 1: Viết test hỏng**

`tests/test_condense.py`:

```python
from kgu.extract.condense import condense, missing_items
from kgu.llm import FakeLLMClient

TEXT = "Italy reported 1,694 cases on Sunday, most of them in Lombardy."


def test_missing_items_compares_numbers_by_digits_and_names_case_insensitively():
    assert missing_items(TEXT, ["Italy reported 1694 cases on sunday."]) == ["Lombardy"]


def test_missing_items_skips_leading_function_words():
    assert missing_items("The World Health Organization said it. But nothing else.",
                         ["World Health Organization said something."]) == []


def test_condense_asks_once_more_for_missing_items():
    llm = FakeLLMClient([{"sentences": ["Italy reported 1,694 cases on Sunday."]},
                         {"sentences": ["Most of the 1,694 cases are in Lombardy.",
                                        "Italy reported 1,694 cases on Sunday."]}])
    sents, stats = condense(TEXT, llm)
    assert sents == ["Italy reported 1,694 cases on Sunday.", "Most of the 1,694 cases are in Lombardy."]
    assert stats == {"sentences": 2, "missing_before": 1, "missing_after": 0}
    assert llm.calls[0][1] == f"Passage: {TEXT}" and "ITEMS: Lombardy" in llm.calls[1][1]


def test_condense_makes_one_call_when_nothing_is_missing():
    llm = FakeLLMClient([{"sentences": ["Italy reported 1,694 cases on Sunday.", "Most cases are in Lombardy.", " "]}])
    sents, stats = condense(TEXT, llm)
    assert llm.n_calls == 1 and sents == ["Italy reported 1,694 cases on Sunday.", "Most cases are in Lombardy."]


def test_condense_keeps_sentences_when_fill_call_fails():
    llm = FakeLLMClient([{"sentences": ["Italy reported 1,694 cases on Sunday."]}])   # lệnh bổ sung: hết response → lỗi
    sents, stats = condense(TEXT, llm)
    assert sents == ["Italy reported 1,694 cases on Sunday."]
    assert stats["missing_after"] == 1 and "fill_error" in stats
```

- [ ] **Step 2: Chạy test, phải hỏng**

Run: `.venv/Scripts/python.exe -m pytest tests/test_condense.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'kgu.extract.condense'`

- [ ] **Step 3: Viết code**

`kgu/extract/condense.py`:

```python
"""Rút đoạn tin thành câu đơn trước khi trích (spec docs/superpowers/specs/2026-10-15-condense-canonicalize-design.md
§2.1): mỗi câu một ý, tự đứng được, giữ nguyên số/tên/cụm thời gian (quy đổi ngày vẫn do bước trích làm). Sau lệnh gọi,
số và cụm viết hoa của đoạn mà không câu nào chứa bị coi là sót → một lệnh gọi bổ sung câu cho chúng."""
from __future__ import annotations

import re

from pydantic import BaseModel

CONDENSE_SYSTEM = """Rewrite the news passage as a list of short simple sentences.
- Each sentence states exactly ONE fact or idea.
- Each sentence stands alone: replace pronouns and vague references (he, she, it, they, the agency, the city) with \
the full name they refer to.
- Keep every number, name, place and time expression exactly as written in the passage ("on Sunday", "last week", \
"1,694 cases"); do not convert dates.
- Keep ALL the information: every fact, number, place, cause, status and statement of the passage must appear in some \
sentence. Do not add anything the passage does not say.

Example:
Passage: "Anna, a teacher, joined a course at Hanoi University on Monday, which she said was too short."
-> {"sentences": ["Anna is a teacher.", "Anna joined a course at Hanoi University on Monday.", \
"Anna said the course at Hanoi University was too short."]}"""

FILL_SYSTEM = """You get a news passage, the simple sentences already written from it, and ITEMS of the passage that \
no sentence mentions yet. Write the missing short simple sentences (one fact each, standing alone, numbers, names and \
time expressions kept as written) that state what the passage says about these items. Do not repeat sentences already \
written. Return {"sentences": [...]} with only the new sentences."""

STOP = frozenset("""a an the and or but if so yet as at by for from in into of on onto to with without within about
after before during since until while when where what which who whom whose why how this that these those there here
it its he him his she her they them their we us our you your i me my mr ms mrs dr some many most more all any each
every no not one also just still only even""".split())

_NUM = re.compile(r"\d[\d,.]*\d|\d")
_CAP = re.compile(r"\b[A-Z][\w'’&-]*(?:\s+[A-Z][\w'’&-]*)*")


class Sentences(BaseModel):
    sentences: list[str]


def _digits(s: str) -> str:
    return re.sub(r"\D", "", s)


def _clean(sentences: list[str]) -> list[str]:
    out: list[str] = []
    for s in sentences:
        s = s.strip()
        if s and s not in out:
            out.append(s)
    return out


def missing_items(text: str, sentences: list[str], limit: int = 15) -> list[str]:
    """Số (so theo chữ số, "1,694" = "1694") và cụm viết hoa (bỏ từ chức năng ở đầu) của đoạn mà không câu nào chứa
    (không phân biệt hoa thường)."""
    joined = " ".join(sentences)
    low, nums = joined.lower(), {_digits(m) for m in _NUM.findall(joined)}
    out: list[str] = []
    for m in _NUM.findall(text):
        if _digits(m) not in nums and m not in out:
            out.append(m)
    for m in _CAP.findall(text):
        words = m.split()
        while words and words[0].lower() in STOP:
            words = words[1:]
        phrase = " ".join(words)
        if phrase and phrase.lower() not in low and phrase not in out:
            out.append(phrase)
    return out[:limit]


def condense(text: str, llm) -> tuple[list[str], dict]:
    """→ (câu đơn, thống kê). Lệnh gọi đầu lỗi → ném lỗi (TemporalKG quay về v0); lệnh bổ sung lỗi → giữ câu đã có."""
    sents = _clean(llm.complete_json(CONDENSE_SYSTEM, f"Passage: {text}", Sentences).sentences)
    miss = missing_items(text, sents) if sents else []
    stats = {"sentences": len(sents), "missing_before": len(miss), "missing_after": len(miss)}
    if miss:
        user = (f"Passage: {text}\n\nSENTENCES:\n" + "\n".join(f"- {s}" for s in sents)
                + "\n\nITEMS: " + "; ".join(miss))
        try:
            sents = _clean(sents + llm.complete_json(FILL_SYSTEM, user, Sentences).sentences)
            stats.update(sentences=len(sents), missing_after=len(missing_items(text, sents)))
        except Exception as e:                       # bổ sung hỏng → giữ câu đã có
            stats["fill_error"] = repr(e)[:200]
    return sents, stats
```

- [ ] **Step 4: Chạy test, phải pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_condense.py -q`
Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add kgu/extract/condense.py tests/test_condense.py
git commit -m "feat: condense news passages into simple standalone sentences, with a fill call for missing numbers/names

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: `TemporalKG` trích trên câu đánh số, trích bổ sung, khớp bí danh

**Files:**
- Modify: `kgu/extract/text_kg.py` (import; thêm `SENTENCE_NOTE`, `TFactS`, `TOutS`, `_numbered` sau `TOut`/`_ds`; sửa `TemporalKG.__init__`, `_stored`, `update` — dòng 94–125 hiện tại)
- Test: `tests/test_text_kg.py` (thêm test, không sửa test cũ)

**Interfaces:**
- Consumes: `condense(text, llm) -> tuple[list[str], dict]` (Task 1).
- Produces: `TemporalKG(max_stored=40, max_relations=60, condense=False)`; thuộc tính `facts: list[list]` (`[head, relation, tail, t_start: list[str], t_end: list[str]]`), `aliases: dict[str, set[str]]` (tên chung → bí danh), `last_stats: dict` (thống kê lần `update` gần nhất: `sentences`, `missing_before`, `missing_after`, `refill_sentences`, có thể `fill_error`/`condense_error`/`refill_error`; `{}` ở chế độ v0); `update(text, obs_date, llm) -> list[tuple]` như cũ.

- [ ] **Step 1: Viết test hỏng** — thêm vào cuối `tests/test_text_kg.py`:

```python
def _f(head, rel, tail, sentence):
    return {"head": head, "relation": rel, "tail": tail, "valid_from": "", "valid_until": "", "sentence": sentence}


def test_condense_mode_extracts_on_numbered_sentences_and_refills_uncovered_ones():
    kg = TemporalKG(condense=True)
    kg.facts = [["Italy", "has_status", "Low Case Count", [], []]]
    llm = FakeLLMClient([
        {"sentences": ["Italy reports 5 cases.", "Lombardy is in Italy.", "Milan is in Lombardy."]},
        {"facts": [_f("Italy", "reports", "5 Cases", 1), _f("Rome", "is_in", "Italy", 99)], "ended": []},
        {"facts": [_f("Lombardy", "is_located_in", "Italy", 2), _f("Milan", "is_located_in", "Lombardy", 3)],
         "ended": [0]},
    ])
    out = kg.update("Italy reports 5 cases in Lombardy, Italy, and Milan.", "2020-03-02", llm)
    assert out == [("Italy", "reports", "5 Cases", [], []), ("Rome", "is_in", "Italy", [], []),
                   ("Lombardy", "is_located_in", "Italy", [], []), ("Milan", "is_located_in", "Lombardy", [], [])]
    assert "[1] Italy reports 5 cases.\n[2] Lombardy is in Italy.\n[3] Milan is in Lombardy." in llm.calls[1][1]
    assert "[2] Lombardy is in Italy.\n[3] Milan" in llm.calls[2][1] and "[1] Italy" not in llm.calls[2][1]
    assert kg.facts[0][4] == []                       # "ended" của lệnh bổ sung bị bỏ qua
    assert kg.last_stats == {"sentences": 3, "missing_before": 0, "missing_after": 0, "refill_sentences": 2}


def test_condense_mode_falls_back_to_passage_when_condense_gives_nothing():
    kg = TemporalKG(condense=True)
    llm = FakeLLMClient([{"sentences": []},
                         {"facts": [{"head": "Virus", "relation": "occurs_in", "tail": "China",
                                     "valid_from": "", "valid_until": ""}], "ended": []}])
    assert kg.update("Virus spreads in China.", "2020-01-28", llm) == [("Virus", "occurs_in", "China", [], [])]
    assert "Passage: Virus spreads in China." in llm.calls[1][1] and "SENTENCES" not in llm.calls[1][1]


def test_condense_mode_falls_back_to_passage_when_condense_output_is_broken():
    kg = TemporalKG(condense=True)
    llm = FakeLLMClient([{"sentence": "truncated"}, {"facts": [], "ended": []}])
    assert kg.update("Virus spreads in China.", "2020-01-28", llm) == []
    assert "condense_error" in kg.last_stats and "Passage: Virus spreads in China." in llm.calls[1][1]


def test_stored_matches_passage_through_aliases():
    kg = TemporalKG()
    kg.facts = [["World Health Organization", "warns", "Countries", [], []]]
    kg.aliases = {"World Health Organization": {"WHO"}}
    llm = FakeLLMClient([{"facts": [], "ended": []}])
    kg.update("WHO said the risk is high.", "2020-03-01", llm)
    assert "[0] World Health Organization | warns | Countries" in llm.calls[0][1]
```

(Ghi chú cho test đầu: câu 1 có fact, fact `sentence: 99` ngoài khoảng vẫn giữ nhưng không phủ câu nào → câu 2, 3 được trích bổ sung. Đoạn tin chứa đủ số 5 và các tên Italy, Lombardy, Milan → không có lệnh bổ sung câu, tổng 3 lệnh gọi.)

- [ ] **Step 2: Chạy test, phải hỏng**

Run: `.venv/Scripts/python.exe -m pytest tests/test_text_kg.py -q`
Expected: 4 test mới FAIL (`TypeError: TemporalKG.__init__() got an unexpected keyword argument 'condense'` / `AttributeError` với `aliases`), 6 test cũ PASS.

- [ ] **Step 3: Viết code** trong `kgu/extract/text_kg.py`.

Thêm import (sau `from pydantic import BaseModel, create_model`):

```python
from kgu.extract.condense import condense as condense_text
```

Thêm ngay sau hàm `_ds`:

```python
SENTENCE_NOTE = """

The passages are given as numbered simple SENTENCES. For each fact also give "sentence": the number of the sentence \
it comes from. Every sentence states at least one fact; do not skip any sentence."""


class TFactS(TFact):
    sentence: int


class TOutS(BaseModel):
    facts: list[TFactS]
    ended: list[int]


def _numbered(sents: list[str], idx) -> str:
    return "SENTENCES:\n" + "\n".join(f"[{i}] {sents[i - 1]}" for i in idx)
```

Thay toàn bộ class `TemporalKG` bằng:

```python
class TemporalKG:
    def __init__(self, max_stored: int = 40, max_relations: int = 60, condense: bool = False):
        self.facts: list[list] = []                    # [head, relation, tail, t_start(list), t_end(list)]
        self.max_stored, self.max_relations = max_stored, max_relations
        self.use_condense = condense                   # False = v0: trích thẳng trên đoạn tin
        self.aliases: dict[str, set[str]] = {}         # tên chung → bí danh (bước gộp tên, kgu.extract.canonicalize)
        self.last_stats: dict = {}                     # thống kê của lần update gần nhất

    def _stored(self, text: str) -> list[int]:
        low = text.lower()
        seen = lambda e: any(re.search(rf"(?<!\w){re.escape(n.lower())}(?!\w)", low)        # noqa: E731
                             for n in (e, *self.aliases.get(e, ())))
        hits = [i for i, f in enumerate(self.facts) if seen(f[0]) or seen(f[2])]
        return hits[-self.max_stored:]                 # giữ fact mới nhất khi phải cắt

    def update(self, text: str, obs_date: str, llm) -> list[tuple[str, str, str, list[str], list[str]]]:
        stored = self._stored(text)
        rels = [r for r, _ in Counter(f[1] for f in self.facts).most_common(self.max_relations)]
        lines = "\n".join(f"[{k}] {self.facts[i][0]} | {self.facts[i][1]} | {self.facts[i][2]}"
                          for k, i in enumerate(stored))
        head = (f"OBSERVATION DATE: {obs_date}\nSTORED:\n{lines or '(none)'}\n"
                f"RELATIONS: {', '.join(rels) or '(none)'}\n")
        self.last_stats = {}
        sents: list[str] = []
        if self.use_condense:
            try:
                sents, cstats = condense_text(text, llm)
                self.last_stats.update(cstats)
            except Exception as e:                     # rút câu hỏng → trích trên đoạn gốc như v0
                self.last_stats["condense_error"] = repr(e)[:200]
        if sents:
            system = TEMPORAL_SYSTEM + SENTENCE_NOTE
            res = llm.complete_json(system, head + _numbered(sents, range(1, len(sents) + 1)), TOutS)
            facts = list(res.facts)
            todo = sorted(set(range(1, len(sents) + 1)) - {f.sentence for f in facts})
            self.last_stats["refill_sentences"] = len(todo)
            if todo:
                try:                                   # câu chưa có fact → trích riêng; "ended" của lệnh này bỏ qua
                    facts += llm.complete_json(system, head + _numbered(sents, todo), TOutS).facts
                except Exception as e:
                    self.last_stats["refill_error"] = repr(e)[:200]
        else:
            res = llm.complete_json(TEMPORAL_SYSTEM, head + f"Passage: {text}", TOut)
            facts = res.facts
        out = []
        for k in res.ended:
            if 0 <= k < len(stored) and not self.facts[stored[k]][4]:
                f = self.facts[stored[k]]
                f[4] = [obs_date]
                out.append((f[0], f[1], f[2], list(f[3]), [obs_date]))     # đoạn này nói fact đã hết → fact của đoạn
        for f in facts:
            q = (f.head.strip(), f.relation.strip(), f.tail.strip(), _ds(f.valid_from), _ds(f.valid_until))
            if q[0] and q[1] and q[2] and q[0].lower() != q[2].lower() and q not in out:
                out.append(q)
                if not any(g[:3] == list(q[:3]) for g in self.facts):
                    self.facts.append(list(q))
        return out
```

Sửa docstring đầu module, thêm sau đoạn mô tả `TemporalKG`:

```
  Khi condense=True (spec 2026-10-15-condense-canonicalize-design.md §2.2): đoạn tin được rút thành câu đơn
  (kgu.extract.condense), trích trên câu đánh số, câu chưa có fact được trích bổ sung một lệnh nữa.
```

- [ ] **Step 4: Chạy test, phải pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_text_kg.py tests/test_condense.py -q`
Expected: `15 passed`

- [ ] **Step 5: Commit**

```bash
git add kgu/extract/text_kg.py tests/test_text_kg.py
git commit -m "feat: TemporalKG condense mode — extract on numbered simple sentences, refill uncovered ones, match stored facts via aliases

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: k-means và nhóm ứng viên (`kgu/extract/canonicalize.py`, phần không gọi LLM)

**Files:**
- Create: `kgu/extract/canonicalize.py`
- Test: `tests/test_canonicalize.py`

**Interfaces:**
- Produces: `kmeans(X: np.ndarray, k: int, seed: int = 0, iters: int = 50) -> np.ndarray` (nhãn cụm, int, độ dài n); `candidate_groups(names: list[str], X: np.ndarray, labels: np.ndarray, counts: dict[str, int], tau: float, blocked: set[frozenset] = frozenset(), max_size: int = 8) -> dict[int, list[dict]]` — mỗi nhóm `{"anchor": str, "members": list[str] (mốc đứng đầu), "cos": dict[str, float] (cosine của từng ứng viên với mốc)}`; `X` đã chuẩn hóa L2.

- [ ] **Step 1: Viết test hỏng**

`tests/test_canonicalize.py`:

```python
import numpy as np

from kgu.extract.canonicalize import candidate_groups, kmeans


def unit(*v):
    a = np.array(v, dtype=float)
    return a / np.linalg.norm(a)


NAMES = ["1,694 Cases", "1,700 Cases", "Italy", "WHO", "World Health Org"]
X = np.array([unit(0, 1, 0.10), unit(0, 1, 0.12), unit(0, 0, 1), unit(1, 0.05, 0), unit(1, 0.1, 0)])
COUNTS = {"1,694 Cases": 1, "1,700 Cases": 1, "Italy": 3, "WHO": 2, "World Health Org": 1}
ONE = np.zeros(5, dtype=int)


def test_kmeans_is_deterministic_and_separates_blobs():
    Y = np.array([unit(1, 0), unit(0.99, 0.1), unit(0, 1), unit(0.1, 0.99)])
    a, b = kmeans(Y, 2, seed=0), kmeans(Y, 2, seed=0)
    assert (a == b).all() and a[0] == a[1] != a[2] == a[3]


def test_kmeans_caps_k_at_number_of_points():
    assert sorted(kmeans(np.array([unit(1, 0), unit(0, 1)]), 5).tolist()) == [0, 1]


def test_candidate_groups_link_close_names_with_the_same_numbers_only():
    groups = candidate_groups(NAMES, X, ONE, COUNTS, tau=0.75)
    assert [g["members"] for g in groups[0]] == [["WHO", "World Health Org"]]   # 2 "Cases" khác số → không cặp
    assert groups[0][0]["anchor"] == "WHO" and groups[0][0]["cos"]["World Health Org"] > 0.99   # hòa tâm → nhiều fact


def test_candidate_groups_skip_blocked_pairs_and_cap_group_size():
    assert candidate_groups(NAMES, X, ONE, COUNTS, 0.75, blocked={frozenset(("WHO", "World Health Org"))}) == {}
    names = [f"Name {c}" for c in "ABCDEFGHIJ"]
    Y = np.array([unit(1, 0.01 * i) for i in range(10)])
    g = candidate_groups(names, Y, np.zeros(10, dtype=int), dict.fromkeys(names, 1), 0.75)[0][0]
    assert len(g["members"]) == 8 and g["members"][0] == g["anchor"]


def test_candidate_groups_anchor_is_nearest_to_group_centre():
    names = ["Rome", "Roma", "Rome City"]
    Y = np.array([unit(1, 0), unit(1, 0.2), unit(1, -0.2)])
    g = candidate_groups(names, Y, np.zeros(3, dtype=int), {"Rome": 1, "Roma": 5, "Rome City": 5}, 0.75)[0][0]
    assert g["anchor"] == "Rome"
```

- [ ] **Step 2: Chạy test, phải hỏng**

Run: `.venv/Scripts/python.exe -m pytest tests/test_canonicalize.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'kgu.extract.canonicalize'`

- [ ] **Step 3: Viết code**

`kgu/extract/canonicalize.py` (phần đầu; Task 4 thêm phần LLM vào cùng file):

```python
"""Gộp tên entity/relation của KG tích lũy (spec docs/superpowers/specs/2026-10-15-condense-canonicalize-design.md
§2.3): nhúng tên → k-means (k = số tên / cluster_size) → trong mỗi cụm nối các cặp cosine ≥ τ có cùng tập chữ số thành
nhóm, tên mốc gần tâm nhóm nhất → một lệnh gọi LLM mỗi cụm có tên mới: yes/no + lý do cho từng ứng viên, chọn tên chung
→ đổi tên, bỏ fact trùng, tên cũ thành bí danh. k-means viết bằng numpy (không thêm scikit-learn)."""
from __future__ import annotations

import re

import numpy as np


def kmeans(X: np.ndarray, k: int, seed: int = 0, iters: int = 50) -> np.ndarray:
    """k-means++ với seed cố định (chạy lại ra cùng cụm). Khoảng cách tính qua tích vô hướng để không tạo mảng n×k×d."""
    X = np.asarray(X, dtype=float)
    n = len(X)
    k = max(1, min(k, n))
    rng = np.random.default_rng(seed)
    sq = (X ** 2).sum(1)
    first = X[rng.integers(n)]
    centers = [first]
    d = sq - 2 * X @ first + first @ first
    while len(centers) < k:
        d = np.maximum(d, 0)
        if d.sum() <= 0:                               # mọi điểm trùng tâm đã chọn
            break
        c = X[rng.choice(n, p=d / d.sum())]
        centers.append(c)
        d = np.minimum(d, sq - 2 * X @ c + c @ c)
    C = np.array(centers)
    labels = None
    for _ in range(iters):
        new = (sq[:, None] - 2 * X @ C.T + (C ** 2).sum(1)[None]).argmin(1)
        if labels is not None and (new == labels).all():
            break
        labels = new
        for j in range(len(C)):
            if (labels == j).any():
                C[j] = X[labels == j].mean(0)
    return labels


def _numbers(name: str) -> tuple[str, ...]:
    return tuple(re.findall(r"\d+", name.replace(",", "")))


def candidate_groups(names: list[str], X: np.ndarray, labels: np.ndarray, counts: dict[str, int], tau: float,
                     blocked=frozenset(), max_size: int = 8) -> dict[int, list[dict]]:
    """Trong mỗi cụm: nối cặp cosine ≥ tau, cùng tập chữ số ("1,694 Cases" ≠ "1,700 Cases"), không bị chặn (đã "no");
    mỗi thành phần liên thông ≥ 2 tên là một nhóm; mốc = gần tâm nhóm nhất (hòa → nhiều fact hơn); nhóm quá max_size →
    giữ mốc + các tên cosine cao nhất với mốc."""
    out: dict[int, list[dict]] = {}
    for c in sorted(set(np.asarray(labels).tolist())):
        idx = [i for i in range(len(names)) if labels[i] == c]
        parent = {i: i for i in idx}

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        S = X[idx] @ X[idx].T
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                i, j = idx[a], idx[b]
                if (S[a, b] >= tau and _numbers(names[i]) == _numbers(names[j])
                        and frozenset((names[i], names[j])) not in blocked):
                    parent[find(i)] = find(j)
        comps: dict[int, list[int]] = {}
        for i in idx:
            comps.setdefault(find(i), []).append(i)
        groups = []
        for comp in comps.values():
            if len(comp) < 2:
                continue
            centre = X[comp].mean(0)
            anchor = max(comp, key=lambda i: (round(float(X[i] @ centre), 6), counts[names[i]]))
            others = sorted((i for i in comp if i != anchor), key=lambda i: -float(X[i] @ X[anchor]))[: max_size - 1]
            groups.append({"anchor": names[anchor], "members": [names[anchor]] + [names[i] for i in others],
                           "cos": {names[i]: round(float(X[i] @ X[anchor]), 4) for i in others}})
        if groups:
            out[int(c)] = groups
    return out
```

- [ ] **Step 4: Chạy test, phải pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_canonicalize.py -q`
Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add kgu/extract/canonicalize.py tests/test_canonicalize.py
git commit -m "feat: numpy k-means and threshold-linked candidate groups for name canonicalization

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: `Canonicalizer` — một lệnh gọi LLM mỗi cụm, đổi tên, bí danh, log

**Files:**
- Modify: `kgu/extract/canonicalize.py` (thêm import, prompt, `_collect`, `_merge_model`, `apply_mapping`, `Canonicalizer`, `sentence_embedder`)
- Test: `tests/test_canonicalize.py` (thêm test)

**Interfaces:**
- Consumes: `kmeans`, `candidate_groups` (Task 3); `TemporalKG.facts`, `TemporalKG.aliases` (Task 2); `llm.complete_json`.
- Produces: `Canonicalizer(embed: Callable[[list[str]], np.ndarray], llm, tau: float = 0.75, cluster_size: int = 10, log_path: str | Path | None = None, seed: int = 0)` với `run_round(kg) -> list[dict]` (thống kê entity rồi relation, mỗi dict có `kind`, `names`, `clusters`, `groups`, `calls`, `errors`, `merged`), `run(kg, kind: str) -> dict`, thuộc tính `rejected: set[tuple[str, frozenset]]`, `seen: dict[str, set[str]]`, `round: int`; `apply_mapping(kg, kind: str, mapping: dict[str, str]) -> None`; `sentence_embedder(model_name: str, device: str) -> Callable[[list[str]], np.ndarray]`. Mỗi dòng log (JSONL) có `round, kind, cluster, group, name, anchor, cosine, same, reason, canonical, anchor_context, name_context`, hoặc `round, kind, cluster, error` khi lệnh gọi lỗi.

- [ ] **Step 1: Viết test hỏng** — thêm vào `tests/test_canonicalize.py`:

```python
import json

from kgu.extract.canonicalize import Canonicalizer, apply_mapping
from kgu.extract.text_kg import TemporalKG
from kgu.llm import FakeLLMClient

AXES = ["WHO", "Countries", "Pandemic", "Italy", "5 Cases", "warns", "declares", "reports", "is located in",
        "Rome", "Roma", "Rome Airport", "Colosseum", "has", "serves"]


def fake_embed(scale=1.0):
    """Mỗi tên một trục; WHO/World Health Organization và Rome/Roma/Rome Airport sát nhau. scale ≠ 1 → vector chưa
    chuẩn hóa."""
    def vec(t):
        v = np.zeros(len(AXES) + 1)
        if t == "World Health Organization":
            v[0], v[-1] = 1, 0.1
        elif t in ("Roma", "Rome Airport"):
            v[AXES.index("Rome")], v[-1] = 1, 0.1 if t == "Roma" else -0.1
        else:
            v[AXES.index(t)] = 1
        return v * scale
    seen: list[str] = []

    def embed(texts):
        seen.extend(texts)
        return np.array([vec(t) for t in texts])
    embed.seen = seen
    return embed


def who_kg():
    kg = TemporalKG()
    kg.facts = [["WHO", "warns", "Countries", [], []], ["WHO", "declares", "Pandemic", [], []],
                ["World Health Organization", "warns", "Italy", [], []], ["Italy", "reports", "5 Cases", [], []],
                ["Italy", "is_located_in", "Pandemic", [], []]]
    return kg


def test_round_merges_yes_candidates_keeps_aliases_and_logs(tmp_path):
    kg, embed, log = who_kg(), fake_embed(scale=3.0), tmp_path / "m.jsonl"
    llm = FakeLLMClient([{"groups": [{"group": 0, "decisions": [
        {"name": "WHO", "reason": "anchor itself", "same": True},
        {"name": "World Health Organization", "reason": "same agency", "same": True}],
        "canonical": "World Health Organization"}]}])
    canon = Canonicalizer(embed, llm, log_path=log)
    ent, rel = canon.run_round(kg)
    assert (ent["calls"], ent["merged"], rel["calls"]) == (1, 1, 0)
    assert [f[0] for f in kg.facts[:3]] == ["World Health Organization"] * 3
    assert kg.aliases == {"World Health Organization": {"WHO"}}
    assert "is located in" in embed.seen                          # relation nhúng với dấu cách thay "_"
    user = llm.calls[0][1]
    assert "ANCHOR: WHO" in user and "CANDIDATE: World Health Organization" in user and "WHO | warns | Countries" in user
    recs = [json.loads(line) for line in log.read_text(encoding="utf8").splitlines()]
    assert len(recs) == 1                                          # quyết định cho chính tên mốc bị bỏ qua
    assert recs[0]["name"] == "World Health Organization" and recs[0]["same"] is True
    assert recs[0]["canonical"] == "World Health Organization" and "WHO | warns | Countries" in recs[0]["anchor_context"]
    assert canon.run_round(kg)[0]["calls"] == 0                   # không có tên mới → không hỏi lại


def test_no_decisions_are_remembered_and_invalid_canonical_falls_back_to_anchor():
    kg = TemporalKG()
    kg.facts = [["Rome", "has", "Colosseum", [], []], ["Roma", "has", "Colosseum", [], []],
                ["Rome Airport", "serves", "Italy", [], []], ["WHO", "warns", "Italy", [], []],
                ["World Health Organization", "warns", "Countries", [], []]]
    llm = FakeLLMClient([{"groups": [{"group": 0, "decisions": [
        {"name": "Roma", "reason": "Italian name of Rome", "same": True},
        {"name": "Rome Airport", "reason": "an airport, not the city", "same": False},
        {"name": "WHO", "reason": "belongs to the other group", "same": True}],
        "canonical": "Rome Airport"}]}])                       # nhóm 1 (WHO) không được trả lời → không gộp
    canon = Canonicalizer(fake_embed(), llm)
    canon.run(kg, "entity")
    assert kg.facts == [["Rome", "has", "Colosseum", [], []], ["Rome Airport", "serves", "Italy", [], []],
                        ["WHO", "warns", "Italy", [], []], ["World Health Organization", "warns", "Countries", [], []]]
    assert ("entity", frozenset(("Rome", "Rome Airport"))) in canon.rejected
    assert kg.aliases == {"Rome": {"Roma"}}


def test_failed_cluster_call_leaves_kg_unchanged(tmp_path):
    kg, log = who_kg(), tmp_path / "m.jsonl"
    before = [list(f) for f in kg.facts]
    stats = Canonicalizer(fake_embed(), FakeLLMClient([]), log_path=log).run(kg, "entity")
    assert (stats["calls"], stats["errors"], stats["merged"]) == (1, 1, 0) and kg.facts == before
    assert "error" in json.loads(log.read_text(encoding="utf8").splitlines()[0])


def test_round_on_empty_kg_does_nothing():
    embed, llm = fake_embed(), FakeLLMClient([])
    stats = Canonicalizer(embed, llm).run_round(TemporalKG())
    assert [s["names"] for s in stats] == [0, 0] and embed.seen == [] and llm.n_calls == 0


def test_apply_mapping_dedups_drops_self_loops_and_moves_aliases():
    kg = TemporalKG()
    kg.facts = [["A", "r", "B", [], []], ["A2", "r", "B", [], []], ["A2", "r", "A", [], []],
                ["A", "r", "B", ["2020-01-01"], []]]
    kg.aliases = {"A2": {"A3"}}
    apply_mapping(kg, "entity", {"A2": "A"})
    assert kg.facts == [["A", "r", "B", [], []], ["A", "r", "B", ["2020-01-01"], []]]
    assert kg.aliases == {"A": {"A2", "A3"}}
    apply_mapping(kg, "relation", {"r": "rel"})
    assert [f[1] for f in kg.facts] == ["rel", "rel"] and kg.aliases == {"A": {"A2", "A3"}}
```

(Ghi chú: k = ceil(n/10) = 1 cụm với dữ liệu nhỏ này; enum của lệnh gọi gồm mọi tên trong các nhóm của cụm. Test đầu dùng `scale=3.0` để kiểm `run` tự chuẩn hóa vector. Ở test thứ hai cụm có 2 nhóm theo thứ tự tên đã sắp xếp: nhóm 0 = Rome/Roma/Rome Airport (mốc Rome, gần tâm nhất), nhóm 1 = WHO/World Health Organization; `canonical` "Rome Airport" không phải mốc hay tên được "yes" → quay về mốc "Rome"; quyết định "WHO" nằm trong nhóm 0 nhưng thuộc nhóm 1 → bị bỏ.)

- [ ] **Step 2: Chạy test, phải hỏng**

Run: `.venv/Scripts/python.exe -m pytest tests/test_canonicalize.py -q`
Expected: FAIL — `ImportError: cannot import name 'Canonicalizer'`

- [ ] **Step 3: Viết code** — trong `kgu/extract/canonicalize.py`, đổi khối import thành:

```python
from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, create_model
```

thêm sau docstring/import (trước `kmeans`):

```python
ENTITY_SYSTEM = """You clean the entity names of a knowledge graph built from news. Each GROUP has an ANCHOR name and \
CANDIDATE names that look similar, each followed by facts where it appears. For every candidate decide if it names \
exactly the same real-world entity as the anchor: write a short reason first, then "same": true or false. Related, \
broader, narrower, a part of, or a different instance (another place, person, number or date) is NOT the same. Then \
give "canonical": the clearest complete name among the anchor and the candidates you marked same."""

RELATION_SYSTEM = """You clean the relation names of a knowledge graph built from news. Each GROUP has an ANCHOR \
relation and CANDIDATE relations that look similar, each followed by facts that use it. For every candidate decide if \
it means exactly the same as the anchor in the same direction (head -> tail): write a short reason first, then \
"same": true or false. Broader, narrower, opposite or reversed relations are NOT the same. Then give "canonical": the \
clearest name among the anchor and the candidates you marked same."""
```

và thêm vào cuối file:

```python
def _collect(facts: list[list], kind: str) -> tuple[list[str], Counter, dict[str, list[str]]]:
    """→ (tên đã sắp xếp, số fact chứa tên, tối đa 3 fact mới nhất chứa tên)."""
    counts: Counter = Counter()
    context: dict[str, list[str]] = {}
    for f in reversed(facts):
        for name in ((f[0], f[2]) if kind == "entity" else (f[1],)):
            counts[name] += 1
            ctx = context.setdefault(name, [])
            if len(ctx) < 3:
                ctx.append(f"{f[0]} | {f[1]} | {f[2]}")
    return sorted(counts), counts, context


def _merge_model(names: list[str]) -> type[BaseModel]:
    lit = Literal[tuple(names)]
    decision = create_model("Decision", name=(lit, ...), reason=(str, ...), same=(bool, ...))
    group = create_model("Group", group=(int, ...), decisions=(list[decision], ...), canonical=(lit, ...))
    return create_model("Merges", groups=(list[group], ...))


def apply_mapping(kg, kind: str, mapping: dict[str, str]) -> None:
    """Đổi tên trong mọi fact, bỏ fact trùng hệt và vòng tự thân; entity: tên cũ (và bí danh của nó) thành bí danh của
    tên chung."""
    if not mapping:
        return
    new, seen = [], set()
    for f in kg.facts:
        if kind == "entity":
            f = [mapping.get(f[0], f[0]), f[1], mapping.get(f[2], f[2]), f[3], f[4]]
        else:
            f = [f[0], mapping.get(f[1], f[1]), f[2], f[3], f[4]]
        key = (f[0], f[1], f[2], tuple(f[3]), tuple(f[4]))
        if f[0].lower() != f[2].lower() and key not in seen:
            seen.add(key)
            new.append(f)
    kg.facts[:] = new
    if kind == "entity":
        for old, can in mapping.items():
            kg.aliases.setdefault(can, set()).update({old, *kg.aliases.pop(old, set())})


class Canonicalizer:
    def __init__(self, embed, llm, tau: float = 0.75, cluster_size: int = 10, log_path=None, seed: int = 0):
        self.embed, self.llm, self.tau, self.cluster_size, self.seed = embed, llm, tau, cluster_size, seed
        self.log_path = Path(log_path) if log_path else None
        self.rejected: set[tuple[str, frozenset]] = set()          # (kind, cặp mốc–ứng viên) đã bị "no"
        self.seen: dict[str, set[str]] = {"entity": set(), "relation": set()}
        self.round = 0

    def run_round(self, kg) -> list[dict]:
        self.round += 1
        return [self.run(kg, "entity"), self.run(kg, "relation")]

    def run(self, kg, kind: str) -> dict:
        names, counts, context = _collect(kg.facts, kind)
        stats = {"kind": kind, "names": len(names), "clusters": 0, "groups": 0, "calls": 0, "errors": 0, "merged": 0}
        if len(names) >= 2:
            X = np.asarray(self.embed([n.replace("_", " ") for n in names] if kind == "relation" else names),
                           dtype=float)
            X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
            labels = kmeans(X, math.ceil(len(names) / self.cluster_size), self.seed)
            blocked = {pair for k, pair in self.rejected if k == kind}
            by_cluster = candidate_groups(names, X, labels, counts, self.tau, blocked)
            stats.update(clusters=len(set(labels.tolist())), groups=sum(map(len, by_cluster.values())))
            mapping: dict[str, str] = {}
            for c, groups in by_cluster.items():
                if all(m in self.seen[kind] for g in groups for m in g["members"]):
                    continue                                       # cụm không có tên mới → đã xét ở lượt trước
                stats["calls"] += 1
                try:
                    out = self._ask(kind, groups, context)
                except Exception as e:                             # lệnh gọi hỏng → bỏ cụm này, KG giữ nguyên
                    stats["errors"] += 1
                    self._log({"round": self.round, "kind": kind, "cluster": c, "error": repr(e)[:300]})
                    continue
                mapping.update(self._decide(kind, c, groups, context, out))
            apply_mapping(kg, kind, mapping)
            stats["merged"] = len(mapping)
            self.seen[kind] |= set(mapping.values())
        self.seen[kind] |= set(names)
        return stats

    def _ask(self, kind: str, groups: list[dict], context: dict[str, list[str]]):
        names = sorted({m for g in groups for m in g["members"]})
        lines: list[str] = []
        for i, g in enumerate(groups):
            lines.append(f"GROUP {i}")
            for j, m in enumerate(g["members"]):
                lines.append(f"{'ANCHOR' if j == 0 else 'CANDIDATE'}: {m}")
                lines += [f"    {t}" for t in context[m]]
        system = ENTITY_SYSTEM if kind == "entity" else RELATION_SYSTEM
        return self.llm.complete_json(system, "\n".join(lines), _merge_model(names))

    def _decide(self, kind: str, cluster: int, groups: list[dict], context: dict[str, list[str]], out) -> dict[str, str]:
        answers: dict[int, object] = {}
        for g in out.groups:
            answers.setdefault(g.group, g)                         # trùng số nhóm → lấy lần đầu
        mapping: dict[str, str] = {}
        for i, g in enumerate(groups):
            a = answers.get(i)
            if a is None:
                continue
            anchor, cands = g["members"][0], g["members"][1:]
            decided: dict[str, tuple[bool, str]] = {}
            for d in a.decisions:                                  # bỏ quyết định cho mốc / tên ngoài nhóm
                if d.name in cands and d.name not in decided:
                    decided[d.name] = (d.same, d.reason)
            same = [n for n in cands if decided.get(n, (False, ""))[0]]
            canonical = a.canonical if a.canonical in (anchor, *same) else anchor
            for name, (ok, reason) in decided.items():
                if not ok:
                    self.rejected.add((kind, frozenset((anchor, name))))
                self._log({"round": self.round, "kind": kind, "cluster": cluster, "group": i, "name": name,
                           "anchor": anchor, "cosine": g["cos"][name], "same": ok, "reason": reason,
                           "canonical": canonical if ok else None,
                           "anchor_context": context[anchor], "name_context": context[name]})
            if same:
                mapping.update({n: canonical for n in (anchor, *same) if n != canonical})
        return mapping

    def _log(self, rec: dict) -> None:
        if self.log_path:
            with self.log_path.open("a", encoding="utf8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def sentence_embedder(model_name: str, device: str):
    """Embedder thật (sentence-transformers, chỉ dùng trên Kaggle): list[str] → ndarray đã chuẩn hóa."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)
    return lambda texts: model.encode(list(texts), batch_size=128, normalize_embeddings=True)
```

- [ ] **Step 4: Chạy test, phải pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_canonicalize.py -q`
Expected: `10 passed`

- [ ] **Step 5: Commit**

```bash
git add kgu/extract/canonicalize.py tests/test_canonicalize.py
git commit -m "feat: Canonicalizer — one LLM call per k-means cluster, yes/no + reason per candidate, rename with aliases

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Chỉ số gộp tên của ATOM (`kgu/eval/atom_merge.py`, `scripts/score_covid_merge.py`)

**Files:**
- Create: `kgu/eval/atom_merge.py`, `scripts/score_covid_merge.py`
- Test: `tests/test_atom_merge.py`

**Interfaces:**
- Consumes: `sentence_embedder` (Task 4) trong script; file KG `<out>.kg.json` (list fact `[h, r, t, ts, te]`) do `scripts/run_covid_nyt.py` ghi; `data/covid_nyt/2020_covid_nyt.jsonl` (mỗi dòng có `gold`).
- Produces: `similar_pairs(names, embed, threshold=0.8) -> int`; `kg_names(facts, kind) -> list[str]`; `gold_mentions(quintuples, kind) -> list[str]`; `resolution(kg_names, gold_mentions, embed, threshold=0.8, kind="entity") -> dict` với khóa `precision, recall, f1, kg_names, similar_pairs, gold_mentions, gold_unique, gold_merged`. CLI: `python scripts/score_covid_merge.py --data <jsonl> --kg tên=đường_dẫn.kg.json ... [--embedder ...] [--device cuda] [--threshold 0.8] --out merge_scores.json` → `{tên: {"entity": {...}, "relation": {...}}}`.

- [ ] **Step 1: Viết test hỏng**

`tests/test_atom_merge.py`:

```python
import numpy as np

from kgu.eval.atom_merge import gold_mentions, kg_names, resolution

E = {"A": [1, 0, 0], "A2": [1, 0, 0], "B": [0, 1, 0], "a": [1, 0, 0], "b": [0, 1, 0], "c": [0, 0, 1],
     "r": [1, 0, 0], "s": [0, 1, 0]}


def embed(texts):
    return np.array([E[t] for t in texts], dtype=float)


def test_entity_resolution_matches_hand_computation():
    r = resolution(["A", "A2", "B"], ["A", "A", "B", "b", "c"], embed, 0.8, "entity")
    # KG: 1 cặp gần trùng (A, A2); cần gộp = 5 lần xuất hiện − 3 tên KG = 2 → P = 1 − 1/2;
    # gold: 5 − 3 tên khác nhau (a, b, c) − 0 cặp gần trùng = 2 → R = 1 − 1/2
    assert (r["precision"], r["recall"], r["f1"], r["similar_pairs"], r["gold_merged"]) == (0.5, 0.5, 0.5, 1, 2)


def test_precision_when_nothing_to_merge_follows_atom_per_kind():
    assert resolution(["r", "s"], ["r", "s"], embed, kind="relation")["precision"] == 1.0
    assert resolution(["A", "B"], ["A", "B"], embed, kind="entity")["precision"] == 0.0
    assert resolution(["A", "B"], ["A", "B"], embed, kind="entity")["recall"] == 1.0


def test_names_and_mentions():
    facts = [["A", "r", "B", [], []], ["A", "s", "C", [], []]]
    assert kg_names(facts, "entity") == ["A", "B", "C"] and kg_names(facts, "relation") == ["r", "s"]
    gold = [("A", "r", "B", [], []), ("A", "r", "C", [], [])]
    assert gold_mentions(gold, "entity") == ["A", "A", "B", "C"] and gold_mentions(gold, "relation") == ["r", "r"]
```

- [ ] **Step 2: Chạy test, phải hỏng**

Run: `.venv/Scripts/python.exe -m pytest tests/test_atom_merge.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'kgu.eval.atom_merge'`

- [ ] **Step 3: Viết code**

`kgu/eval/atom_merge.py`:

```python
"""Chỉ số gộp tên (entity/relation resolution) của ATOM, cài lại theo evaluation/atom/merge/evaluate_atom_merge.py
(AuvaLab/itext2kg) với embedder truyền vào thay cho text-embedding-3-large:
- similar = số cặp tên khác nhau trong KG có cosine > ngưỡng (bản gốc 0,8);
- precision = 1 − similar / (số lần xuất hiện trong gold − số tên của KG); khi mẫu ≤ 0: entity → 0, relation → 1 nếu
  bằng nhau, ngược lại 0 (như bản gốc);
- recall = 1 − similar / (số lần xuất hiện − số tên gold khác nhau (không phân biệt hoa thường) − số cặp tên gold có
  cosine > ngưỡng); mẫu ≤ 0 → 1. Cả hai kẹp [0, 1]."""
from __future__ import annotations

import numpy as np


def similar_pairs(names: list[str], embed, threshold: float = 0.8) -> int:
    if len(names) < 2:
        return 0
    X = np.asarray(embed(list(names)), dtype=float)
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    iu = np.triu_indices(len(names), 1)
    return int(((X @ X.T)[iu] > threshold).sum())


def kg_names(facts: list[list], kind: str) -> list[str]:
    if kind == "entity":
        return sorted({f[0] for f in facts} | {f[2] for f in facts})
    return sorted({f[1] for f in facts})


def gold_mentions(quintuples: list, kind: str) -> list[str]:
    if kind == "entity":
        return [q[0] for q in quintuples] + [q[2] for q in quintuples]
    return [q[1] for q in quintuples]


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


def resolution(names: list[str], mentions: list[str], embed, threshold: float = 0.8, kind: str = "entity") -> dict:
    names = sorted(set(names))
    sim = similar_pairs(names, embed, threshold)
    expected = len(mentions) - len(names)
    if expected > 0:
        p = _clamp(1 - sim / expected)
    else:
        p = (1.0 if expected == 0 else 0.0) if kind == "relation" else 0.0
    uniq = sorted({m.lower() for m in mentions if m})
    merged = len(mentions) - len(uniq) - similar_pairs(uniq, embed, threshold) if len(uniq) >= 2 else 0
    r = 1.0 if merged <= 0 else _clamp(1 - sim / merged)
    f1 = 2 * p * r / (p + r) if p > 0 and r > 0 else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4), "kg_names": len(names),
            "similar_pairs": sim, "gold_mentions": len(mentions), "gold_unique": len(uniq), "gold_merged": merged}
```

`scripts/score_covid_merge.py`:

```python
"""Chỉ số gộp tên của ATOM (kgu.eval.atom_merge) trên KG cuối của từng biến thể, 2020-COVID-NYT (spec
docs/superpowers/specs/2026-10-15-condense-canonicalize-design.md §3). Gold = mọi 5-tuple gold của các nhóm ngày trong
--data. Nạp embedder → chạy trên Kaggle GPU (kaggle/covid_v2).

Dùng: python scripts/score_covid_merge.py --data covid.jsonl --kg v0=out/v0.jsonl.kg.json [--kg ...]
      [--embedder sentence-transformers/all-mpnet-base-v2] [--device cuda] [--threshold 0.8] --out merge_scores.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from kgu.eval.atom_merge import gold_mentions, kg_names, resolution
from kgu.extract.canonicalize import sentence_embedder


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--kg", action="append", default=[], help="tên=đường_dẫn.kg.json")
    ap.add_argument("--embedder", default="sentence-transformers/all-mpnet-base-v2")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threshold", type=float, default=0.8)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rows = [json.loads(line) for line in args.data.read_text(encoding="utf8").splitlines() if line.strip()]
    gold = [q for r in rows for q in r["gold"]]
    embed = sentence_embedder(args.embedder, args.device)
    out = {}
    for spec in args.kg:
        name, path = spec.split("=", 1)
        facts = json.loads(Path(path).read_text(encoding="utf8"))
        out[name] = {kind: resolution(kg_names(facts, kind), gold_mentions(gold, kind), embed, args.threshold, kind)
                     for kind in ("entity", "relation")}
        print(name, out[name], flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf8")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Chạy test, phải pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_atom_merge.py -q && .venv/Scripts/python.exe -m py_compile scripts/score_covid_merge.py`
Expected: `3 passed`, py_compile không in gì.

- [ ] **Step 5: Commit**

```bash
git add kgu/eval/atom_merge.py scripts/score_covid_merge.py tests/test_atom_merge.py
git commit -m "feat: ATOM entity/relation resolution metric with a local embedder, scored on final KGs

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Runner COVID-NYT có cờ v1/v2 và script kiểm tay quyết định gộp

**Files:**
- Modify: `scripts/run_covid_nyt.py` (`run_rows` thêm `extra`, `after`; `main` thêm cờ)
- Create: `scripts/audit_merges.py`
- Test: `tests/test_home_bench_io.py` (thêm test)

**Interfaces:**
- Consumes: `TemporalKG(condense=...)`, `kg.last_stats`, `kg.aliases` (Task 2); `Canonicalizer`, `sentence_embedder` (Task 4); log JSONL của `Canonicalizer` (Task 4).
- Produces: `run_rows(rows, step, out, extra=None, after=None)` — `extra() -> dict` ghép vào từng dòng output, `after(k)` gọi sau nhóm thứ k (1-based); CLI `run_covid_nyt.py --data --out [--limit] [--condense] [--canon-every N] [--tau 0.75] [--embedder ...] [--device cuda]` ghi `<out>`, `<out>.kg.json`, `<out>.aliases.json`, và khi bật gộp tên: `<out>.merges.jsonl`, `<out>.canon.json`. `audit_merges.sample(records, n=100, seed=0) -> list[dict]`, `audit_merges.summarize(rows) -> dict`; CLI `audit_merges.py --merges <jsonl> [--n 100] [--seed 0] --out audit.csv` hoặc `audit_merges.py --labels audit.csv`.

- [ ] **Step 1: Viết test hỏng** — thêm vào cuối `tests/test_home_bench_io.py`:

```python
def test_covid_runner_adds_row_stats_and_calls_after_each_group(tmp_path):
    m = _mod("run_covid_nyt")
    rows = [{"row": 0, "date": "2020-01-02", "text": "b"}, {"row": 1, "date": "2020-01-01", "text": "a"}]
    done = []
    out = tmp_path / "o.jsonl"
    m.run_rows(rows, lambda text, date: [], out, extra=lambda: {"calls": 2, "sentences": 3}, after=done.append)
    got = [json.loads(line) for line in out.read_text(encoding="utf8").splitlines()]
    assert [(g["row"], g["calls"], g["sentences"]) for g in got] == [(1, 2, 3), (0, 2, 3)] and done == [1, 2]


def test_audit_sample_is_stratified_deterministic_and_skips_errors():
    m = _mod("audit_merges")
    recs = [{"id": i, "same": i % 3 == 0} for i in range(300)] + [{"id": 300, "error": "x"}]
    a = m.sample(recs, n=10, seed=0)
    assert a == m.sample(recs, n=10, seed=0) and len(a) == 10 and sum(r["same"] for r in a) == 5
    assert [r["id"] for r in a] == sorted(r["id"] for r in a)


def test_audit_sample_fills_from_the_other_side_when_one_is_short():
    m = _mod("audit_merges")
    recs = [{"id": 0, "same": True}] + [{"id": i, "same": False} for i in range(1, 20)]
    a = m.sample(recs, n=10)
    assert len(a) == 10 and sum(r["same"] for r in a) == 1


def test_audit_summary_counts_only_labelled_rows():
    m = _mod("audit_merges")
    rows = [{"same": "True", "correct": "1"}, {"same": "True", "correct": "0"},
            {"same": "False", "correct": "1"}, {"same": "False", "correct": " "}]
    assert m.summarize(rows) == {"yes": {"n": 2, "correct": 1, "accuracy": 0.5},
                                 "no": {"n": 1, "correct": 1, "accuracy": 1.0}}
```

- [ ] **Step 2: Chạy test, phải hỏng**

Run: `.venv/Scripts/python.exe -m pytest tests/test_home_bench_io.py -q`
Expected: 4 test mới FAIL (`TypeError: run_rows() got an unexpected keyword argument 'extra'`, `FileNotFoundError ... audit_merges.py`), test cũ PASS.

- [ ] **Step 3: Viết code**

Trong `scripts/run_covid_nyt.py`, sửa docstring (thêm dòng sau dòng "Ra jsonl ..."):

```
Cờ (spec docs/superpowers/specs/2026-10-15-condense-canonicalize-design.md): --condense = v1 (rút câu đơn + trích bổ
sung); thêm --canon-every N = v2 (gộp tên k-means/LLM sau mỗi N nhóm ngày và sau nhóm cuối; log quyết định ở
<out>.merges.jsonl, thống kê lượt ở <out>.canon.json). Mỗi dòng output có thêm "calls" và thống kê của TemporalKG.
```

thêm `import time` đã có; thay `run_rows` và `main` bằng:

```python
def run_rows(rows: list[dict], step, out: Path, extra=None, after=None) -> None:
    """step(text, date) -> list 5-tuple; dùng chung cho mình (TemporalKG) và baseline (atom_covid, kggen_covid).
    extra() -> dict ghép vào dòng output; after(k) gọi sau nhóm ngày thứ k (đếm từ 1, theo thứ tự ngày)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf8") as f:
        for k, r in enumerate(sorted(rows, key=lambda r: (r["date"], r["row"])), 1):
            t0 = time.time()
            try:
                qs = step(r["text"], r["date"])
            except Exception as e:                   # lỗi một nhóm ngày → rỗng (tính 0 điểm), chạy tiếp
                print(f"!!! row {r['row']}: {e!r}"[:300], flush=True)
                qs = []
            rec = {"row": r["row"], "quintuples": [list(q) for q in qs], "seconds": round(time.time() - t0, 2)}
            if extra:
                rec.update(extra())
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if after:
                after(k)


def main() -> None:
    from kgu.llm import client_from_env

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--condense", action="store_true", help="v1: rút câu đơn + trích bổ sung")
    ap.add_argument("--canon-every", type=int, default=0, help="v2: gộp tên sau mỗi N nhóm ngày; 0 = tắt")
    ap.add_argument("--tau", type=float, default=0.75)
    ap.add_argument("--embedder", default="sentence-transformers/all-mpnet-base-v2")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    rows = sorted(_score.load_dataset(args.data), key=lambda r: (r["date"], r["row"]))[: args.limit]
    kg, llm = TemporalKG(condense=args.condense), client_from_env()
    canon, rounds, last = None, [], [0]
    if args.canon_every:
        from kgu.extract.canonicalize import Canonicalizer, sentence_embedder

        log = Path(f"{args.out}.merges.jsonl")
        log.unlink(missing_ok=True)
        canon = Canonicalizer(sentence_embedder(args.embedder, args.device), llm, tau=args.tau, log_path=log)

    def extra() -> dict:
        n, last[0] = llm.n_calls - last[0], llm.n_calls
        return {"calls": n, **kg.last_stats}

    def after(k: int) -> None:
        if canon and (k % args.canon_every == 0 or k == len(rows)):
            t0 = time.time()
            stats = canon.run_round(kg)
            rounds.append({"after_group": k, "seconds": round(time.time() - t0, 1), "stats": stats})
            print(f"gộp tên sau nhóm {k}: {stats}", flush=True)
            last[0] = llm.n_calls                    # lệnh gọi gộp tên không tính vào nhóm ngày sau

    run_rows(rows, lambda text, date: kg.update(text, date, llm), args.out, extra=extra, after=after)
    Path(f"{args.out}.kg.json").write_text(json.dumps(kg.facts, ensure_ascii=False), encoding="utf8")
    Path(f"{args.out}.aliases.json").write_text(
        json.dumps({k: sorted(v) for k, v in kg.aliases.items()}, ensure_ascii=False), encoding="utf8")
    if canon:
        Path(f"{args.out}.canon.json").write_text(json.dumps(rounds, ensure_ascii=False, indent=1), encoding="utf8")
    print(f"{len(rows)} nhóm ngày, KG cuối {len(kg.facts)} fact, {llm.n_calls} lệnh gọi LLM → {args.out}")
```

`scripts/audit_merges.py`:

```python
"""Kiểm tay quyết định gộp tên (spec docs/superpowers/specs/2026-10-15-condense-canonicalize-design.md §3): rút mẫu
từ <out>.merges.jsonl ra CSV (một nửa "yes", một nửa "no", seed cố định); người kiểm điền cột `correct` = 1 (quyết
định đúng) / 0 (sai); chạy lại với --labels để tính tỉ lệ đúng của "yes" và "no".

Dùng: python scripts/audit_merges.py --merges out/v2.jsonl.merges.jsonl --n 100 --out audit.csv
      python scripts/audit_merges.py --labels audit.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

FIELDS = ["id", "kind", "anchor", "name", "cosine", "same", "reason", "canonical", "anchor_context", "name_context",
          "correct"]


def sample(records: list[dict], n: int = 100, seed: int = 0) -> list[dict]:
    recs = [r for r in records if "same" in r]       # bỏ dòng lỗi lệnh gọi
    yes, no = [r for r in recs if r["same"]], [r for r in recs if not r["same"]]
    k_yes = min(len(yes), n // 2)
    k_no = min(len(no), n - k_yes)
    k_yes = min(len(yes), n - k_no)                  # một bên thiếu → lấy thêm từ bên kia
    rng = random.Random(seed)
    return sorted(rng.sample(yes, k_yes) + rng.sample(no, k_no), key=lambda r: r["id"])


def summarize(rows: list[dict]) -> dict:
    out = {}
    for label in ("yes", "no"):
        rs = [r for r in rows if (r["same"] == "True") == (label == "yes") and r["correct"].strip() in ("0", "1")]
        ok = sum(r["correct"].strip() == "1" for r in rs)
        out[label] = {"n": len(rs), "correct": ok, "accuracy": round(ok / len(rs), 3) if rs else None}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--merges", type=Path)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--labels", type=Path, help="CSV đã điền cột correct → in tỉ lệ đúng")
    args = ap.parse_args()
    if args.labels:
        with args.labels.open(encoding="utf8", newline="") as f:
            print(json.dumps(summarize(list(csv.DictReader(f))), ensure_ascii=False))
        return
    lines = args.merges.read_text(encoding="utf8").splitlines()
    records = [{"id": i, **json.loads(line)} for i, line in enumerate(lines) if line.strip()]
    picked = sample(records, args.n, args.seed)
    with args.out.open("w", encoding="utf8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in picked:
            w.writerow({**r, "anchor_context": " || ".join(r["anchor_context"]),
                        "name_context": " || ".join(r["name_context"]), "correct": ""})
    print(f"{len(picked)} quyết định ({sum(r['same'] for r in picked)} yes) → {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Chạy toàn bộ test, phải pass**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: toàn bộ pass (120 test cũ + 26 test mới = 146 passed).

- [ ] **Step 5: Commit**

```bash
git add scripts/run_covid_nyt.py scripts/audit_merges.py tests/test_home_bench_io.py
git commit -m "feat: COVID-NYT runner flags for condense (v1) and periodic canonicalization (v2); merge audit sampler

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Kernel Kaggle `kgu-covid-condense-canon` và lượt dev 20 nhóm ngày

**Files:**
- Create: `kaggle/covid_v2/run_covid_v2.py`, `kaggle/covid_v2/kernel-metadata.json`
- Create (kết quả): `results/kaggle/covid_v2_dev_r<N>.summary.json`, `results/kaggle/covid_v2_dev_r<N>_kernel.log`

**Interfaces:**
- Consumes: CLI `scripts/run_covid_nyt.py` (Task 6), `scripts/score_covid_nyt.py` (có sẵn), `scripts/score_covid_merge.py` (Task 5); dataset Kaggle `ngocnam2005/llama-server-cuda`.
- Produces: output kernel `/kaggle/working/out/`: `v0.jsonl`, `v1.jsonl`, `v2.jsonl` (+ `.kg.json`, `.aliases.json`, `v2.jsonl.merges.jsonl`, `v2.jsonl.canon.json`), `scores_covid.json`, `scores_covid_heldout.json` (khi LIMIT ≥ 21), `merge_scores.json`, `summary.json`, `logs/`.

- [ ] **Step 1: Viết kernel**

`kaggle/covid_v2/kernel-metadata.json`:

```json
{
  "id": "ngocnam2005/kgu-covid-condense-canon",
  "title": "kgu-covid-condense-canon",
  "code_file": "run_covid_v2.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": true,
  "dataset_sources": [
    "ngocnam2005/llama-server-cuda"
  ],
  "competition_sources": [],
  "kernel_sources": []
}
```

`kaggle/covid_v2/run_covid_v2.py`:

```python
"""Kaggle kernel: ablation trên 2020-COVID-NYT cho pipeline của mình (spec
docs/superpowers/specs/2026-10-15-condense-canonicalize-design.md): v0 = TemporalKG hiện tại, v1 = + rút câu đơn và
trích bổ sung, v2 = v1 + gộp tên k-means/LLM mỗi CANON_EVERY nhóm ngày. Gemma 4 E4B qua llama-server CUDA (GPU 0) như
kgu-home-benchmarks; embedder gộp tên all-mpnet-base-v2 trên GPU 1. Chấm scripts/score_covid_nyt.py (mọi nhóm + held-out
bỏ DEV_N nhóm đầu) và scripts/score_covid_merge.py (chỉ số gộp tên của ATOM trên KG cuối).
LIMIT = số nhóm ngày theo thứ tự ngày (20 = tập dev để chỉnh prompt/τ; None = đủ 122). VARIANTS chọn biến thể.
Đầu ra /kaggle/working/out/: <v>.jsonl (+ .kg.json, .aliases.json; v2 thêm .merges.jsonl, .canon.json),
scores_covid[_heldout].json, merge_scores.json, summary.json, logs/*.log, server.log.
"""
import json
import os
import shutil
import subprocess
import time
import urllib.request

LIMIT = 20
VARIANTS = ["v0", "v1", "v2"]
DEV_N = 20
CANON_EVERY, TAU = 10, 0.75
WORKERS = 4
KGU_REPO, KGU_BRANCH = "https://github.com/HgNam123456/kltn.git", "feat/core-pipeline"
GGUF_REPO, GGUF_FILE = "unsloth/gemma-4-E4B-it-qat-GGUF", "gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf"
MTP_FILE = "mtp-gemma-4-E4B-it.gguf"
SCORE_EMBEDDER, SCORE_THRESHOLD = "sentence-transformers/all-mpnet-base-v2", "0.7"
WORK = "/kaggle/working"
OUT, LOGS = f"{WORK}/out", f"{WORK}/out/logs"
KLTN = f"{WORK}/kltn"
SERVER = "http://127.0.0.1:8080"
T0 = time.time()


def sh(cmd: str, **kw) -> str:
    print(f"$ {cmd}", flush=True)
    return subprocess.run(cmd, shell=True, check=True, text=True, capture_output=kw.pop("capture", False), **kw).stdout


def find_dir(marker: str) -> str | None:
    for dirpath, dirnames, filenames in os.walk("/kaggle/input"):
        if marker in dirnames or marker in filenames:
            return dirpath
    return None


def read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


for d in (OUT, LOGS):
    os.makedirs(d, exist_ok=True)
summary: dict = {"limit": LIMIT, "variants": VARIANTS, "canon_every": CANON_EVERY, "tau": TAU, "runs": {},
                 "outputs": {}, "canon": {}, "notes": []}

# 1. GPU, code, gói
gpus = sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader", capture=True).strip().splitlines()
summary["gpus"] = gpus
sh(f"git clone -q --depth 1 -b {KGU_BRANCH} {KGU_REPO} {KLTN}")
summary["kltn_commit"] = sh(f"git -C {KLTN} rev-parse HEAD", capture=True).strip()
sh("pip install -q dateparser 'openai>=1'")

# 2. dữ liệu theo thứ tự ngày
rows = sorted(read_jsonl(f"{KLTN}/data/covid_nyt/2020_covid_nyt.jsonl"), key=lambda r: (r["date"], r["row"]))[:LIMIT]
COVID, COVID_HELDOUT = f"{WORK}/covid.jsonl", f"{WORK}/covid_heldout.jsonl"
for path, part in ((COVID, rows), (COVID_HELDOUT, rows[DEV_N:])):
    with open(path, "w", encoding="utf-8") as f:
        for r in part:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
summary["covid_n"], summary["heldout_n"] = len(rows), max(len(rows) - DEV_N, 0)

# 3. llama-server (giống kaggle/home_bench/run_home_bench.py), ghim GPU 0
from huggingface_hub import hf_hub_download  # noqa: E402

cached = find_dir("llama-server")
assert cached, "thiếu dataset ngocnam2005/llama-server-cuda"
bin_dir = f"{WORK}/llama-bin"
shutil.copytree(cached, bin_dir, dirs_exist_ok=True)
sh(f"chmod +x {bin_dir}/llama-server")
model = hf_hub_download(GGUF_REPO, GGUF_FILE, local_dir=f"{WORK}/models")
draft = hf_hub_download(GGUF_REPO, MTP_FILE, local_dir=f"{WORK}/models")
COMMON = f"{bin_dir}/llama-server -m {model} --alias local --host 127.0.0.1 --port 8080 -ngl 99 --jinja --temp 0"
ATTEMPTS = [
    COMMON + f" --parallel {WORKERS} --cont-batching --kv-unified-per-slot 8192 --reasoning off --flash-attn on"
             f" --spec-type draft-mtp --spec-draft-n-max 2 --model-draft {draft}",
    COMMON + f" --parallel {WORKERS} --cont-batching -c {8192 * WORKERS} --flash-attn on",
]


def healthy() -> bool:
    try:
        return b"ok" in urllib.request.urlopen(f"{SERVER}/health", timeout=3).read()
    except Exception:
        return False


server = None
for cmd in ATTEMPTS:
    print(f"$ {cmd}", flush=True)
    server = subprocess.Popen(cmd, shell=True, stdout=open(f"{OUT}/server.log", "ab"), stderr=subprocess.STDOUT,
                              env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"})
    for _ in range(180):
        if healthy() or server.poll() is not None:
            break
        time.sleep(2)
    if healthy():
        break
    print("server không lên, thử bộ cờ khác", flush=True)
    server.kill()
assert healthy(), "llama-server không khởi động được, xem server.log"
summary["server_flags"] = cmd.replace(COMMON, "")

env = {**os.environ, "PYTHONIOENCODING": "utf8", "TOKENIZERS_PARALLELISM": "false", "PYTHONPATH": KLTN,
       "KGU_LLM_BASE_URL": f"{SERVER}/v1", "KGU_LLM_MODEL": "local", "KGU_LLM_API_KEY": "none",
       "KGU_LLM_EXTRA_BODY": "{}", "KGU_LLM_MAX_TOKENS": "4096"}


def timed(run: str, cmd: str, n: int, **kw) -> bool:
    t0 = time.time()
    try:
        sh(f"{cmd} 2>&1 | tee {LOGS}/{run}.log; test ${{PIPESTATUS[0]}} -eq 0", executable="/bin/bash", **kw)
        ok = True
    except subprocess.CalledProcessError:
        print(f"!!! {run} lỗi, xem logs/{run}.log", flush=True)
        ok = False
    sec = round(time.time() - t0, 1)
    summary["runs"][run] = {"ok": ok, "seconds": sec, "sec_per_item": round(sec / max(n, 1), 2)}
    print(f"=== {run}: ok={ok} {sec}s, tổng {round((time.time() - T0) / 60)} phút", flush=True)
    return ok


# 4. biến thể
emb_device = "cuda:1" if len(gpus) >= 2 else "cuda:0"
FLAGS = {"v0": "", "v1": "--condense",
         "v2": f"--condense --canon-every {CANON_EVERY} --tau {TAU} --device {emb_device}"}
for v in VARIANTS:
    timed(v, f"python -u {KLTN}/scripts/run_covid_nyt.py --data {COVID} --out {OUT}/{v}.jsonl {FLAGS[v]}",
          len(rows), env=env)
server.kill()

# 5. chấm: trích xuất (mọi nhóm + held-out) và gộp tên
done = [v for v in VARIANTS if os.path.exists(f"{OUT}/{v}.jsonl")]
preds = " ".join(f"--pred {v}={OUT}/{v}.jsonl" for v in done)
for split, data in (("covid", COVID), ("covid_heldout", COVID_HELDOUT)):
    if not os.path.getsize(data):
        continue
    try:
        sh(f"set -o pipefail; python -u {KLTN}/scripts/score_covid_nyt.py --data {data} {preds} "
           f"--embedder {SCORE_EMBEDDER} --threshold {SCORE_THRESHOLD} --device cuda "
           f"--out {OUT}/scores_{split}.json 2>&1 | tee {LOGS}/score_{split}.log", executable="/bin/bash", env=env)
        summary[split] = json.load(open(f"{OUT}/scores_{split}.json"))["results"][SCORE_EMBEDDER][SCORE_THRESHOLD]
    except subprocess.CalledProcessError:
        print(f"!!! chấm {split} lỗi, xem logs/score_{split}.log", flush=True)
kgs = " ".join(f"--kg {v}={OUT}/{v}.jsonl.kg.json" for v in done if os.path.exists(f"{OUT}/{v}.jsonl.kg.json"))
try:
    sh(f"set -o pipefail; python -u {KLTN}/scripts/score_covid_merge.py --data {COVID} {kgs} "
       f"--embedder {SCORE_EMBEDDER} --device cuda --out {OUT}/merge_scores.json 2>&1 | tee {LOGS}/score_merge.log",
       executable="/bin/bash", env=env)
    summary["merge"] = json.load(open(f"{OUT}/merge_scores.json"))
except subprocess.CalledProcessError:
    print("!!! chấm gộp tên lỗi, xem logs/score_merge.log", flush=True)

# 6. thống kê output
for v in done:
    rs = read_jsonl(f"{OUT}/{v}.jsonl")
    n_q = sum(len(r["quintuples"]) for r in rs)
    summary["outputs"][v] = {
        "rows": len(rs), "empty": sum(not r["quintuples"] for r in rs), "avg_quintuples": round(n_q / max(len(rs), 1), 2),
        "calls": sum(r.get("calls", 0) for r in rs),
        "avg_sentences": round(sum(r.get("sentences", 0) for r in rs) / max(len(rs), 1), 2),
        "missing_before": sum(r.get("missing_before", 0) for r in rs),
        "missing_after": sum(r.get("missing_after", 0) for r in rs),
        "refill_sentences": sum(r.get("refill_sentences", 0) for r in rs),
        "errors": {k: sum(k in r for r in rs) for k in ("condense_error", "fill_error", "refill_error")}}
    if os.path.exists(f"{OUT}/{v}.jsonl.canon.json"):
        summary["canon"][v] = json.load(open(f"{OUT}/{v}.jsonl.canon.json"))

summary["total_minutes"] = round((time.time() - T0) / 60, 1)
json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(summary, indent=1, ensure_ascii=False), flush=True)
for d in (KLTN, f"{WORK}/models", f"{WORK}/llama-bin"):
    shutil.rmtree(d, ignore_errors=True)       # không đưa repo/model vào output
print("DONE", flush=True)
```

- [ ] **Step 2: Kiểm cú pháp, commit, push nhánh** (kernel clone nhánh `feat/core-pipeline` nên code Task 1–6 phải đã push)

```bash
.venv/Scripts/python.exe -m py_compile kaggle/covid_v2/run_covid_v2.py
git add kaggle/covid_v2
git commit -m "run: Kaggle kernel kgu-covid-condense-canon (v0/v1/v2 ablation on COVID-NYT, dev 20 groups)

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
git push origin feat/core-pipeline
```

- [ ] **Step 3: Đẩy kernel lượt dev (LIMIT = 20) và theo dõi**

```bash
cd kaggle/covid_v2 && PYTHONUTF8=1 kaggle kernels push -p .
```

Theo dõi bằng Monitor, báo người dùng mỗi ~5 phút:
`while true; do s=$(PYTHONUTF8=1 kaggle kernels status ngocnam2005/kgu-covid-condense-canon 2>&1 | tail -1 || true); echo "$(date +%H:%M) $s"; echo "$s" | grep -qiE "complete|error|cancel" && break; sleep 300; done`

Khi xong: `PYTHONUTF8=1 kaggle kernels output ngocnam2005/kgu-covid-condense-canon -p <scratchpad>/covid_v2_dev_r1`; chép `summary.json` → `results/kaggle/covid_v2_dev_r1.summary.json`, log kernel → `results/kaggle/covid_v2_dev_r1_kernel.log`.

- [ ] **Step 4: Đọc kết quả dev và quyết định chỉnh (tối đa 3 lượt dev, như lần trước)**

Đọc trong `summary.json` và các file output:
- `covid.v1.R` so với `covid.v0.R` (recall phải tăng); `covid.v1.F1` ≥ `covid.v0.F1` − 1.
- `outputs.v1.missing_after` / `missing_before` (bổ sung câu có tác dụng), `refill_sentences`, `calls` / 20 (mục tiêu 2–4 lệnh/nhóm).
- `v2.jsonl.merges.jsonl`: đọc 30 quyết định đầu. Nhiều "yes" sai → tăng τ thêm 0,05 hoặc sửa `ENTITY_SYSTEM`/`RELATION_SYSTEM`; hầu như không có nhóm (`canon.v2[*].stats[*].groups` ≈ 0) → giảm τ 0,05.
- `merge.v2` so với `merge.v1` (precision/recall gộp tên của ATOM phải tăng).

Mỗi lần chỉnh: sửa code/hằng số, chạy `.venv/Scripts/python.exe -m pytest -q`, commit (message nêu lượt dev và lý do), push, đẩy lại kernel (lượt r2, r3), lưu kết quả như Step 3. Ghi diễn biến từng lượt vào `docs/data-notes-emerge.md` mục mới "§18. Rút câu đơn + gộp tên (COVID-NYT)".

- [ ] **Step 5: Commit kết quả dev**

```bash
git add results/kaggle/covid_v2_dev_r*.summary.json results/kaggle/covid_v2_dev_r*_kernel.log docs/data-notes-emerge.md
git commit -m "results: covid_v2 dev runs (20 date groups) — condense/canonicalize tuning

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: Lượt đủ 122 nhóm, kiểm tay, cập nhật ghi chú và báo cáo

**Files:**
- Modify: `kaggle/covid_v2/run_covid_v2.py` (`LIMIT = None`), `docs/data-notes-emerge.md` (§18)
- Create (kết quả): `results/kaggle/covid_v2/` (summary.json, scores_covid*.json, merge_scores.json, v*.jsonl, v2.jsonl.merges.jsonl, v2.jsonl.canon.json, v*.jsonl.kg.json), `results/kaggle/covid_v2_kernel.log`, `results/covid_v2_audit.csv`

**Interfaces:**
- Consumes: mọi thứ ở Task 1–7.
- Produces: số chính thức v0/v1/v2 trên 122 và 102 nhóm held-out; file CSV để người dùng kiểm tay.

- [ ] **Step 1: Chạy đủ**

Sửa `LIMIT = 20` → `LIMIT = None` trong `kaggle/covid_v2/run_covid_v2.py`, commit ("run: covid_v2 full 122 date groups"), push, `kaggle kernels push`, theo dõi như Task 7 Step 3 (ước tính: v0 ~21 phút, v1 ~45, v2 ~60, chấm ~10 → ~2,5 giờ).

- [ ] **Step 2: Lấy kết quả**

`kaggle kernels output ngocnam2005/kgu-covid-condense-canon -p <scratchpad>/covid_v2_full`; chép các file nêu ở mục Files vào `results/kaggle/covid_v2/` và log kernel vào `results/kaggle/covid_v2_kernel.log`.

- [ ] **Step 3: Rút mẫu kiểm tay**

```bash
.venv/Scripts/python.exe scripts/audit_merges.py --merges results/kaggle/covid_v2/v2.jsonl.merges.jsonl --n 100 --out results/covid_v2_audit.csv
```

Expected: in `100 quyết định (50 yes) → results/covid_v2_audit.csv`. Gửi file cho người dùng điền cột `correct`; khi có nhãn: `.venv/Scripts/python.exe scripts/audit_merges.py --labels results/covid_v2_audit.csv`.

- [ ] **Step 4: Ghi kết quả**

Trong `docs/data-notes-emerge.md` §18 thêm bảng (122 nhóm và 102 held-out): F1/P/R/F1_t, 5-tuple/nhóm, lệnh gọi/nhóm, phút cho v0/v1/v2 (kèm số ATOM 2 bước 52,5 / KGGen 48,3 từ §17 để đối chiếu); bảng thống kê rút câu (câu/nhóm, mục thiếu trước/sau, câu trích bổ sung); bảng gộp tên (ER/RR precision/recall/F1 của ATOM, số tên entity/relation, số lượt, lệnh gọi, số tên đã gộp); kết quả kiểm tay khi có. So với tiêu chí thành công ở spec §1 và ghi rõ đạt/không đạt.

- [ ] **Step 5: Commit và cập nhật báo cáo**

```bash
git add kaggle/covid_v2/run_covid_v2.py results/kaggle/covid_v2 results/kaggle/covid_v2_kernel.log results/covid_v2_audit.csv docs/data-notes-emerge.md
git commit -m "results: condense (v1) and canonicalization (v2) on full COVID-NYT, held-out 102 groups

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
git push origin feat/core-pipeline
```

Cập nhật trang Notion báo cáo thực nghiệm (`3fac4cb5-99a9-81ac-9968-fc1291f80945`): thêm mục 6.x "COVID-NYT: rút câu đơn và gộp tên" với bảng v0/v1/v2 và bảng gộp tên; thêm dòng kernel vào nhật ký chạy.
