# So baseline trên benchmark sân nhà (WebNLG, 2020-COVID-NYT) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Chấm pipeline của mình và các baseline trên đúng benchmark mà paper baseline tự dùng — WebNLG (sân của EDC) và 2020-COVID-NYT (sân của ATOM/iText2KG) — tất cả cùng Gemma 4 E4B qua llama-server trên Kaggle, để bảng so sánh không còn là "đem baseline ra sân lạ".

**Architecture:** Hai benchmark chỉ đo trích xuất (không có KG trước, không có Deprecate), nên "pipeline của mình" ở đây là module Add chạy không cần KG Wikidata: WebNLG — bộ quan hệ lấy từ schema của dataset (thay cho bộ quan hệ lấy từ KG), ép bằng enum trong json_schema; COVID-NYT — KG tự tích lũy qua các nhóm ngày theo thứ tự thời gian, mỗi đoạn tin được cho xem fact đã lưu có nhắc thực thể của nó (localize) + bộ quan hệ đã dùng, rồi trả cả fact xác nhận lại (Exists) lẫn fact mới (Add) kèm t_start/t_end. Baseline chạy bằng chính code của paper (EDC gốc clear-nus/edc; ATOM trong itext2kg 1.1.0; KGGen 0.4.0). iText2KG bản gốc không chạy riêng: benchmark của nó không dùng lại được (15 tài liệu, chấm tay) và ATOM là bản kế nhiệm của cùng nhóm, chấm trên COVID-NYT; KARMA và SAC-KG không có benchmark công khai. Bộ chấm: script WebNLG của EDC (tất định); metric ATOM cài lại với embedding local, hiệu chỉnh ngưỡng bằng output GPT-4.1 có sẵn trong dataset để tái lập số trong paper ATOM.

**Tech Stack:** Python 3.11, pydantic, openai SDK (llama-server OpenAI-compatible), sentence-transformers (embedding chấm), dateparser, Kaggle CLI (T4), itext2kg 1.1.0 / langchain-openai, kg-gen 0.4.0, clear-nus/edc.

**Spec:** quyết định trong hội thoại 14/10 + `docs/data-notes-emerge.md` §14–16 (bảng EMERGE, chẩn đoán baseline thấp). Kế hoạch tổng: trang Notion "🎯 kế hoạch".

## Global Constraints

- LLM duy nhất cho MỌI hệ thống: Gemma 4 E4B GGUF `unsloth/gemma-4-E4B-it-qat-GGUF` / `gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf`, llama-server `--jinja --reasoning off`, 4 slot × 8192 ngữ cảnh (giống `kaggle/baselines/run_baselines.py`).
- Mọi thứ cần GPU (LLM, embedding chấm/hiệu chỉnh) chạy trên Kaggle T4 (memory feedback-use-kaggle-cli). Máy local chỉ viết code, unit test (không nạp model), phân tích file kết quả.
- Baseline chạy bằng code gốc, ghim commit/phiên bản; mọi chỗ phải vá để chạy với llama-server ghi rõ trong docstring kernel và notes.
- Cùng một embedder + ngưỡng chấm cho mọi hệ thống trên COVID-NYT; bài có gold mà hệ thống không ra gì tính MATCH = 0 (khác script ATOM là bỏ qua) — báo cả hai cách nếu chênh.
- WebNLG chấm bằng `evaluate/evaluation_script.py` của clear-nus/edc, không dùng `--max_length_diff`; báo số bài thực sự được chấm.
- Tên model ghi kết quả: `edc/gemma-4-e4b`, `mine/gemma-4-e4b`, `atom/gemma-4-e4b`, `atom-direct/gemma-4-e4b`, `kggen/gemma-4-e4b`.

## Review Focus

- Dòng output LLM không parse được / json rỗng → bài đó vẫn có dòng output (danh sách rỗng), không làm lệch thứ tự dòng của file WebNLG (script chấm ghép theo số dòng). Test ở Task 4.
- Ngày tháng ở nhiều định dạng (`2020-01-24`, `24-01-2020`, `['2020-01-28']`, `[]`, `None`) → so khớp ngày đúng, list rỗng = không có thời gian. Test ở Task 1.
- Quan hệ model trả không nằm trong schema WebNLG (dù có enum) hoặc chữ hoa/thường khác → bỏ fact, không crash. Test ở Task 3.
- KG tích lũy COVID phình to → khối STORED trong prompt bị cắt theo giới hạn, không vượt 8192 token. Test ở Task 3.
- Nhóm ngày có gold nhưng hệ thống ra rỗng → tính 0, không bị bỏ qua. Test ở Task 1.

---

## File Structure

| File | Trách nhiệm |
|---|---|
| `kgu/eval/atom_quality.py` (mới) | Metric ATOM thuần (không gọi embedder): ghép 1-1 tham lam theo ma trận cosine, MATCH/HALL/OM + biến thể thời gian, gộp macro theo bài |
| `scripts/convert_covid_nyt.py` (mới), `data/covid_nyt/2020_covid_nyt.jsonl` (mới, commit) | Đổi pickle phát hành trên HF sang JSONL **một lần** (pickle có thể chạy code tùy ý → chỉ nạp một lần, có ghi nguồn); mọi script khác đọc JSONL |
| `scripts/score_covid_nyt.py` (mới) | Đọc dataset + prediction, nhúng mỗi câu một lần cho mọi hệ thống, chấm nhiều embedder × ngưỡng một lượt; `--calibrate` chấm cột GPT-4.1 có sẵn để chọn ngưỡng |
| `kaggle/covid_calib/run_calib.py`, `kernel-metadata.json` (mới) | Kernel GPU nhỏ (không LLM): chạy `score_covid_nyt.py --calibrate` trên T4 |
| `kgu/extract/text_kg.py` (mới) | Pipeline của mình không cần KG Wikidata: `extract_schema` (WebNLG) và `TemporalKG.update` (COVID) |
| `scripts/run_webnlg.py` (mới) | Chạy `extract_schema` trên WebNLG, ghi file 1 dòng/câu theo định dạng `canon_kg.txt` của EDC |
| `scripts/run_covid_nyt.py` (mới) | Chạy `TemporalKG` qua các nhóm ngày theo thứ tự, ghi jsonl `{row, quintuples}` |
| `scripts/atom_covid.py` (mới) | Chạy ATOM (2 bước factoid → 5-tuple) và ATOM-direct bằng itext2kg 1.1.0 vá json_schema |
| `scripts/kggen_covid.py` (mới) | Chạy KGGen 0.4.0 trên từng nhóm ngày (không có thời gian) |
| `kaggle/home_bench/run_home_bench.py`, `kernel-metadata.json` (mới) | Kernel: dựng llama-server, chạy mọi hệ thống trên 2 benchmark, chấm, xuất bảng |
| `tests/test_atom_quality.py`, `tests/test_text_kg.py`, `tests/test_home_bench_io.py` (mới) | Test |
| `docs/data-notes-emerge.md` (sửa) | §17 kết quả + cách hiệu chỉnh |

---

### Task 1: Metric ATOM cài lại (thuần, có test)

**Files:**
- Create: `kgu/eval/atom_quality.py`
- Test: `tests/test_atom_quality.py`

**Interfaces:**
- Produces: `dates(v) -> frozenset[datetime.date]`, `match_row(pred: list[tuple], gold: list[tuple], sim: numpy.ndarray, threshold: float) -> dict[str, int]` (khóa `match, hall, om, match_t, om_t, hall_t, n_pred, n_gold`), `summarize(rows: list[dict]) -> dict[str, float]` (khóa `MATCH, HALL, OM, MATCH_t, P, R, F1, R_t, n_rows`).

- [ ] **Step 1: Viết test fail**

```python
import numpy as np

from kgu.eval.atom_quality import dates, match_row, summarize


def test_dates_accepts_list_string_and_day_first_formats():
    assert dates(["2020-01-28"]) == dates("2020-01-28") == dates("28-01-2020")
    assert dates([]) == dates(None) == dates("") == dates("None") == frozenset()


def test_match_row_is_greedy_one_to_one_above_threshold():
    pred = [("a", "r", "b", ["2020-01-01"], []), ("a", "r", "b2", [], [])]
    gold = [("a", "r", "b", ["2020-01-01"], []), ("c", "r", "d", [], [])]
    sim = np.array([[0.95, 0.80], [0.90, 0.10]])
    out = match_row(pred, gold, sim, threshold=0.7)
    # (0,0)=0.95 thắng trước; (1,0) bị chặn vì gold 0 đã dùng; (0,1) bị chặn vì pred 0 đã dùng
    assert out == {"match": 1, "hall": 1, "om": 1, "match_t": 1, "om_t": 0, "hall_t": 0, "n_pred": 2, "n_gold": 2}


def test_temporal_categories_follow_atom_priority():
    gold = [("a", "r", "b", ["2020-01-01"], [])]
    sim = np.array([[0.99]])
    missing = match_row([("a", "r", "b", [], [])], gold, sim, 0.7)
    extra = match_row([("a", "r", "b", ["2020-01-01"], ["2020-02-01"])], gold, sim, 0.7)
    assert (missing["om_t"], missing["match_t"]) == (1, 0)
    assert (extra["hall_t"], extra["match_t"]) == (1, 0)


def test_empty_prediction_counts_as_full_omission_not_skipped():
    gold = [("a", "r", "b", [], [])]
    out = match_row([], gold, np.zeros((0, 1)), 0.7)
    assert out["match"] == 0 and out["om"] == 1
    s = summarize([out, {"match": 1, "hall": 0, "om": 0, "match_t": 1, "om_t": 0, "hall_t": 0, "n_pred": 1, "n_gold": 1}])
    assert s["R"] == 0.5 and s["n_rows"] == 2
```

- [ ] **Step 2: Chạy test, thấy fail**

Run: `.venv/Scripts/python -m pytest tests/test_atom_quality.py -q`
Expected: FAIL `ModuleNotFoundError: kgu.eval.atom_quality`

- [ ] **Step 3: Cài đặt**

```python
"""Metric 5-tuple của ATOM (Lairgi et al., EACL 2026), cài lại từ evaluation/atom/quintuples_quality/
calculate_quintuples_quality.py của AuvaLab/itext2kg, tách khỏi embedder:
- mỗi bài (nhóm ngày): ghép dự đoán ↔ gold 1-1 tham lam theo cosine của "h r t" giảm dần, chỉ cặp > ngưỡng;
- MATCH = khớp / gold, HALL = dự đoán thừa / dự đoán, OM = gold sót / gold;
- trong cặp khớp: MATCH_t nếu t_start và t_end đều khớp; OM_t nếu dự đoán thiếu mốc gold có; HALL_t nếu dự đoán có
  mốc gold không có; còn lại OM_t. Biến thể thời gian chia cho số gold.
Khác bản gốc (ghi trong notes): ngày đọc thành tập date (list rỗng/None/"" = không có mốc; bản gốc so chuỗi repr của
list); bài có gold mà không có dự đoán tính MATCH = 0 thay vì bị bỏ qua.
"""
from __future__ import annotations

import datetime as dt

import dateparser
import numpy as np


def dates(v) -> frozenset[dt.date]:
    vals = v if isinstance(v, (list, tuple)) else [v]
    out = set()
    for x in vals:
        if x is None or str(x).strip().lower() in ("", "none", "null"):
            continue
        d = dateparser.parse(str(x).strip(), settings={"DATE_ORDER": "YMD", "PREFER_DAY_OF_MONTH": "first"})
        if d is None and "-" in str(x):                                     # "24-01-2020"
            d = dateparser.parse(str(x).strip(), settings={"DATE_ORDER": "DMY"})
        if d is not None:
            out.add(d.date())
    return frozenset(out)


def _t(q, i):
    return dates(q[i]) if len(q) > i else frozenset()


def match_row(pred: list[tuple], gold: list[tuple], sim: np.ndarray, threshold: float) -> dict[str, int]:
    pairs = sorted(((sim[i, j], i, j) for i in range(len(pred)) for j in range(len(gold)) if sim[i, j] > threshold),
                   reverse=True)
    used_p, used_g, out = set(), set(), dict.fromkeys(("match", "match_t", "om_t", "hall_t"), 0)
    for _, i, j in pairs:
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        out["match"] += 1
        ps, pe, gs, ge = _t(pred[i], 3), _t(pred[i], 4), _t(gold[j], 3), _t(gold[j], 4)
        if ps == gs and pe == ge:
            out["match_t"] += 1
        elif (not ps and gs) or (not pe and ge):
            out["om_t"] += 1
        elif (ps and not gs) or (pe and not ge):
            out["hall_t"] += 1
        else:
            out["om_t"] += 1
    out.update(hall=len(pred) - out["match"], om=len(gold) - out["match"], n_pred=len(pred), n_gold=len(gold))
    return out


def summarize(rows: list[dict]) -> dict[str, float]:
    rows = [r for r in rows if r["n_gold"]]
    mean = lambda f: float(np.mean([f(r) for r in rows])) if rows else 0.0          # noqa: E731
    p = mean(lambda r: 1 - r["hall"] / r["n_pred"] if r["n_pred"] else 0.0)
    rec = mean(lambda r: r["match"] / r["n_gold"])
    return {"MATCH": rec, "HALL": mean(lambda r: r["hall"] / r["n_pred"] if r["n_pred"] else 0.0),
            "OM": mean(lambda r: r["om"] / r["n_gold"]), "MATCH_t": mean(lambda r: r["match_t"] / r["n_gold"]),
            "P": p, "R": rec, "F1": 2 * p * rec / (p + rec) if p + rec else 0.0,
            "R_t": mean(lambda r: r["match_t"] / r["n_gold"]), "n_rows": len(rows)}
```

- [ ] **Step 4: Chạy test, thấy pass**

Run: `.venv/Scripts/python -m pip install -q dateparser && .venv/Scripts/python -m pytest tests/test_atom_quality.py -q`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add kgu/eval/atom_quality.py tests/test_atom_quality.py
git commit -m "feat: ATOM 5-tuple metric (greedy cosine matching, temporal categories), embedder-agnostic"
```

### Task 2: Bộ chấm COVID-NYT + hiệu chỉnh trên Kaggle GPU

**Files:**
- Create: `scripts/convert_covid_nyt.py`, `data/covid_nyt/2020_covid_nyt.jsonl`, `scripts/score_covid_nyt.py`, `kaggle/covid_calib/run_calib.py`, `kaggle/covid_calib/kernel-metadata.json`
- Test: `tests/test_home_bench_io.py` (hàm `load_predictions`, `score`, `pick`)

**Interfaces:**
- Consumes: `match_row`, `summarize` (Task 1).
- Produces: JSONL 1 dòng/nhóm ngày `{"row", "date", "text", "gold": [[h,r,t,ts,te],...], "calib": {cột_gpt41: [[...]], ...}}`; `load_dataset(path) -> list[dict]` (khóa `row, date, text, gold, calib`), `load_predictions(path) -> dict[int, list[tuple]]` (jsonl `{"row": int, "quintuples": [[h,r,t,ts,te], ...]}`), `score(rows, preds, emb: dict[str, ndarray], thresholds) -> dict[float, dict]`, `pick(results) -> dict` (embedder/ngưỡng gần số paper nhất), CLI `--data <jsonl> --pred name=path ... --embedder <name>... --threshold <x>... [--calibrate] --device cuda --out <json>`.

- [ ] **Step 0: Đổi pickle → JSONL một lần** — `scripts/convert_covid_nyt.py --pkl <2020-COVID-NYT.pkl> --out data/covid_nyt/2020_covid_nyt.jsonl`: `pandas.read_pickle` (comment ghi rõ: nguồn HF `lairgiyassir/2020-COVID-NYT`, giấy phép MIT, trùng byte với `datasets/atom/nyt_news/2020_nyt_COVID_last_version_ready.pkl` của AuvaLab/itext2kg; chỉ chạy một lần ở local, kernel không nạp pickle), giữ `date` (YYYY-MM-DD), `lead_paragraph_observation_date`, `quintuples_g_truth` và 6 cột `quintuples_gpt41*` (đổi tuple/list/Timestamp sang JSON thuần). Kiểm: 122 dòng, 3.918 gold.

- [ ] **Step 1: Viết test fail** — `tests/test_home_bench_io.py`:

```python
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _mod(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_load_predictions_tuples_and_keeps_empty_rows(tmp_path):
    p = tmp_path / "p.jsonl"
    p.write_text(json.dumps({"row": 0, "quintuples": [["a", "r", "b", [], []]]}) + "\n"
                 + json.dumps({"row": 3, "quintuples": []}) + "\n", encoding="utf8")
    assert _mod("score_covid_nyt").load_predictions(p) == {0: [("a", "r", "b", [], [])], 3: []}


def test_score_counts_empty_rows_and_reports_skip_variant():
    m = _mod("score_covid_nyt")
    rows = [{"row": 0, "gold": [("a", "r", "b", [], [])]}, {"row": 1, "gold": [("c", "r", "d", [], [])]}]
    emb = {"a r b": np.array([1.0, 0.0]), "c r d": np.array([0.0, 1.0])}
    out = m.score(rows, {0: [("a", "r", "b", [], [])]}, emb, [0.7, 0.9])
    assert out[0.7]["R"] == 0.5 and out[0.7]["R_skip"] == 1.0 and out[0.7]["n_rows_skip"] == 1


def test_pick_closest_to_paper_and_keeps_atom_above_direct():
    m = _mod("score_covid_nyt")
    res = {"e1": {"0.7": {"gpt41-direct": {"F1_skip": 0.50}, "gpt41-atom": {"F1_skip": 0.64}},
                  "0.8": {"gpt41-direct": {"F1_skip": 0.60}, "gpt41-atom": {"F1_skip": 0.55}}},
           "e2": {"0.7": {"gpt41-direct": {"F1_skip": 0.40}, "gpt41-atom": {"F1_skip": 0.70}}}}
    assert (m.pick(res)["embedder"], m.pick(res)["threshold"]) == ("e1", "0.7")
```

- [ ] **Step 2: Chạy, thấy fail** — `.venv/Scripts/python -m pytest tests/test_home_bench_io.py -q` → FileNotFoundError.

- [ ] **Step 3: Cài `scripts/score_covid_nyt.py`** — đọc JSONL; gom mọi câu "h r t" (gold + mọi hệ thống) rồi nhúng **một lần** mỗi embedder; `score(rows, preds, emb, thresholds) -> {ngưỡng: summarize(...) + khóa *_skip}` (`*_skip` = bỏ bài hệ thống ra rỗng, như script gốc; khóa thường = tính 0); `pick(results)` chọn (embedder, ngưỡng) giữ ATOM > direct và tổng |F1_skip − paper| nhỏ nhất; CLI `--embedder` và `--threshold` nhận nhiều giá trị, `--device cuda` mặc định; e5 thêm tiền tố `"query: "` cả hai phía. Mã đầy đủ: xem file trong repo (viết theo đúng mô tả trên).

- [ ] **Step 4: Chạy test** — `.venv/Scripts/python -m pytest tests/test_home_bench_io.py -q` → 3 passed (máy local chỉ chạy test, không nạp embedder).

- [ ] **Step 5: Hiệu chỉnh trên Kaggle GPU** — kernel `kaggle/covid_calib/` (`ngocnam2005/kgu-covid-calibration`, GPU T4, internet, không cần dataset): clone repo kltn nhánh `feat/core-pipeline`, `pip install dateparser`, chạy `score_covid_nyt.py --calibrate --embedder sentence-transformers/all-mpnet-base-v2 BAAI/bge-large-en-v1.5 intfloat/e5-large-v2 --threshold 0.6 0.65 0.7 0.75 0.8 0.85 0.9 --device cuda --out /kaggle/working/covid_calibration.json`. Push nhánh trước khi đẩy kernel. Kéo output về `results/kaggle/covid_calibration.json` + `results/kaggle/covid_calib_kernel.log`. Kernel chạy nền trong lúc làm Task 3–5. Nếu không cặp nào lệch ≤ 0,05 thì ghi rõ trong notes và dùng `all-mpnet-base-v2` @ 0,7 (cùng embedder với Completeness của EMERGE).

- [ ] **Step 6: Commit**

```bash
git add scripts/convert_covid_nyt.py data/covid_nyt/2020_covid_nyt.jsonl scripts/score_covid_nyt.py \
        tests/test_home_bench_io.py kaggle/covid_calib/
git commit -m "feat: COVID-NYT scorer (ATOM metric, local embedder) + Kaggle calibration kernel"
# sau khi kernel xong:
git add results/kaggle/covid_calibration.json results/kaggle/covid_calib_kernel.log
git commit -m "results: COVID-NYT scorer calibration against released GPT-4.1 outputs"
```

### Task 3: Pipeline của mình không cần KG Wikidata (`kgu/extract/text_kg.py`)

**Files:**
- Create: `kgu/extract/text_kg.py`
- Test: `tests/test_text_kg.py`

**Interfaces:**
- Consumes: `kgu.llm.LLMClient.complete_json(system, user, schema)`, `kgu.llm.FakeLLMClient`.
- Produces:
  - `extract_schema(text: str, relations: list[str], llm, examples: str = "") -> list[tuple[str, str, str]]` — 1 lệnh gọi/câu, quan hệ ép bằng enum (pydantic `Literal` tạo động), bỏ quan hệ ngoài schema (so không phân biệt hoa thường), bỏ fact head == tail.
  - `class TemporalKG` với `update(text: str, obs_date: str, llm) -> list[tuple[str, str, str, list[str], list[str]]]` — trả 5-tuple của đoạn này (Exists xác nhận lại + Add mới), cập nhật KG: fact mới được thêm, fact cũ bị báo "ended" được gán t_end = obs_date; thuộc tính `facts: list[list]`, `max_stored: int = 40`, `max_relations: int = 60`.

- [ ] **Step 1: Viết test fail**

```python
from kgu.extract.text_kg import TemporalKG, extract_schema
from kgu.llm import FakeLLMClient

RELS = ["birthPlace", "leader", "country"]


def test_extract_schema_keeps_schema_relations_case_insensitively_and_drops_rest():
    llm = FakeLLMClient([{"facts": [{"subject": "Ab", "relation": "birthplace", "object": "X"},
                                    {"subject": "Ab", "relation": "spouse", "object": "Y"},
                                    {"subject": "Ab", "relation": "country", "object": "Ab"}]}])
    assert extract_schema("Ab was born in X.", RELS, llm) == [("Ab", "birthPlace", "X")]


def test_extract_schema_prompt_lists_every_relation():
    llm = FakeLLMClient([{"facts": []}])
    extract_schema("t", RELS, llm)
    assert all(r in llm.calls[0][1] for r in RELS)


def test_temporal_kg_reuses_stored_facts_and_closes_ended_ones():
    kg = TemporalKG()
    first = FakeLLMClient([{"facts": [{"head": "Virus", "relation": "occurs_in", "tail": "China",
                                       "valid_from": "2020-01-24", "valid_until": ""}], "ended": []}])
    assert kg.update("Virus spreads in China.", "2020-01-28", first) == [("Virus", "occurs_in", "China", ["2020-01-24"], [])]
    second = FakeLLMClient([{"facts": [], "ended": [0]}])
    kg.update("The Virus is gone from China.", "2020-03-01", second)
    assert "[0] Virus | occurs_in | China" in second.calls[0][1]          # fact cũ có thực thể được nhắc → STORED
    assert kg.facts[0][4] == ["2020-03-01"]                              # bị báo ended → đóng t_end


def test_temporal_kg_caps_stored_block():
    kg = TemporalKG(max_stored=2)
    kg.facts = [["A", "r", f"B{i}", [], []] for i in range(5)]
    llm = FakeLLMClient([{"facts": [], "ended": []}])
    kg.update("A did things.", "2020-01-01", llm)
    assert llm.calls[0][1].count("| r |") == 2
```

- [ ] **Step 2: Chạy, thấy fail** — `.venv/Scripts/python -m pytest tests/test_text_kg.py -q` → ModuleNotFoundError.

- [ ] **Step 3: Cài đặt**

```python
"""Pipeline của mình khi không có KG Wikidata (benchmark trích xuất sân nhà của baseline):
- extract_schema: module Add với bộ quan hệ lấy từ schema dataset (WebNLG) thay cho bộ quan hệ của KG lân cận,
  ép bằng enum trong json_schema (constrained decoding của llama-server);
- TemporalKG: KG tự tích lũy qua tin tức theo thời gian. Mỗi đoạn: localize = fact đã lưu có head/tail xuất hiện
  trong đoạn (STORED) + quan hệ đã dùng (RELATIONS) → một lệnh gọi trả fact đoạn này nói (Exists + Add, kèm mốc
  thời gian) và fact đã lưu bị đoạn này nói là hết đúng (Deprecate → đóng t_end).
"""
from __future__ import annotations

from collections import Counter
from typing import Literal

from pydantic import BaseModel, create_model

SCHEMA_SYSTEM = """Extract the facts the text states as (subject, relation, object) triples, using ONLY the listed \
relations. Subjects and objects are names or values copied from the text. A fact counts only if the text says it \
(directly or by clear paraphrase). Give every such fact, no duplicates."""

TEMPORAL_SYSTEM = """You maintain a temporal knowledge graph built from dated news. You get one news passage with \
its observation date, STORED facts already in the graph that mention entities of this passage, and the RELATIONS the \
graph already uses.
1. List every fact the passage states as (head, relation, tail, valid_from, valid_until). Reuse the exact entity and \
relation names of STORED facts and RELATIONS when they mean the same thing; otherwise create a short snake_case \
relation. valid_from / valid_until are YYYY-MM-DD dates the passage gives or implies for when the fact starts / stops \
being true ("" if none; use the observation date for a state described as current).
2. In "ended", give the numbers of STORED facts the passage says are no longer true.

Example:
OBSERVATION DATE: 2020-03-02
STORED: [0] Italy | has_status | low_case_count
RELATIONS: has_status, occurs_in, causes
Passage: "Italy, which had few cases until last week, reported 1,694 cases on Sunday, most in Lombardy."
-> {"facts": [{"head": "Italy", "relation": "reports", "tail": "1,694 cases", "valid_from": "2020-03-01", \
"valid_until": ""}, {"head": "1,694 cases", "relation": "occurs_in", "tail": "Lombardy", "valid_from": "2020-03-01", \
"valid_until": ""}], "ended": [0]}"""


def _schema_model(relations: list[str]) -> type[BaseModel]:
    fact = create_model("Fact", subject=(str, ...), relation=(Literal[tuple(relations)], ...), object=(str, ...))
    return create_model("Facts", facts=(list[fact], ...))


def extract_schema(text: str, relations: list[str], llm, examples: str = "") -> list[tuple[str, str, str]]:
    by_lower = {r.lower(): r for r in relations}
    user = (f"{examples}\n\n" if examples else "") + f"RELATIONS: {', '.join(relations)}\nText: {text}"
    out: list[tuple[str, str, str]] = []
    for f in llm.complete_json(SCHEMA_SYSTEM, user, _schema_model(relations)).facts:
        r = by_lower.get(str(f.relation).strip().lower())
        s, o = f.subject.strip(), f.object.strip()
        if r and s and o and s.lower() != o.lower() and (s, r, o) not in out:
            out.append((s, r, o))
    return out


class TFact(BaseModel):
    head: str
    relation: str
    tail: str
    valid_from: str
    valid_until: str


class TOut(BaseModel):
    facts: list[TFact]
    ended: list[int]


def _ds(v: str) -> list[str]:
    return [v.strip()] if v and v.strip() else []


class TemporalKG:
    def __init__(self, max_stored: int = 40, max_relations: int = 60):
        self.facts: list[list] = []                    # [head, relation, tail, t_start(list), t_end(list)]
        self.max_stored, self.max_relations = max_stored, max_relations

    def _stored(self, text: str) -> list[int]:
        low = text.lower()
        hits = [i for i, f in enumerate(self.facts) if f[0].lower() in low or f[2].lower() in low]
        return hits[-self.max_stored:]                 # gần nhất trước khi cắt

    def update(self, text: str, obs_date: str, llm) -> list[tuple[str, str, str, list[str], list[str]]]:
        stored = self._stored(text)
        rels = [r for r, _ in Counter(f[1] for f in self.facts).most_common(self.max_relations)]
        lines = "\n".join(f"[{k}] {self.facts[i][0]} | {self.facts[i][1]} | {self.facts[i][2]}"
                          for k, i in enumerate(stored))
        user = (f"OBSERVATION DATE: {obs_date}\nSTORED:\n{lines or '(none)'}\n"
                f"RELATIONS: {', '.join(rels) or '(none)'}\nPassage: {text}")
        res = llm.complete_json(TEMPORAL_SYSTEM, user, TOut)
        for k in res.ended:
            if 0 <= k < len(stored) and not self.facts[stored[k]][4]:
                self.facts[stored[k]][4] = [obs_date]
        out = []
        for f in res.facts:
            q = (f.head.strip(), f.relation.strip(), f.tail.strip(), _ds(f.valid_from), _ds(f.valid_until))
            if q[0] and q[1] and q[2] and q not in out:
                out.append(q)
                if not any(g[:3] == list(q[:3]) for g in self.facts):
                    self.facts.append(list(q))
        return out
```

- [ ] **Step 4: Chạy test** — `.venv/Scripts/python -m pytest tests/test_text_kg.py -q` → 4 passed; cả bộ `pytest -q` vẫn xanh.

- [ ] **Step 5: Commit**

```bash
git add kgu/extract/text_kg.py tests/test_text_kg.py
git commit -m "feat: KG-free variants of our Add module (schema-constrained WebNLG, accumulating temporal KG for news)"
```

### Task 4: Runner của mình cho WebNLG và COVID-NYT

**Files:**
- Create: `scripts/run_webnlg.py`, `scripts/run_covid_nyt.py`
- Test: `tests/test_home_bench_io.py` (thêm)

**Interfaces:**
- Consumes: `extract_schema`, `TemporalKG` (Task 3); `load_dataset` (Task 2); `kgu.llm.client_from_env()` (đọc `KGU_LLM_BASE_URL`, `KGU_LLM_MODEL`, `KGU_LLM_API_KEY`).
- Produces:
  - `run_webnlg.py --input datasets/webnlg.txt --schema schemas/webnlg_schema.csv --examples few_shot_examples/webnlg/oie_few_shot_examples.txt --out <canon_kg.txt> [--limit N] [--workers 4]`; hàm `format_line(triples) -> str` ra đúng `str(list_of_lists)` như `canon_kg.txt` của EDC; lỗi LLM → dòng `[]`.
  - `run_covid_nyt.py --data data/covid_nyt/2020_covid_nyt.jsonl --out <jsonl> [--limit N]`; chạy tuần tự theo `date` tăng dần (KG tích lũy nên không song song); ghi `{"row", "quintuples", "seconds"}`; lưu KG cuối ở `<out>.kg.json`.

- [ ] **Step 1: Test fail**

```python
def test_webnlg_line_format_matches_edc_canon_kg():
    m = _mod("run_webnlg")
    assert m.format_line([("A", "leader", "B")]) == "[['A', 'leader', 'B']]"
    assert m.format_line([]) == "[]"


def test_webnlg_reader_keeps_line_order_and_reads_schema(tmp_path):
    m = _mod("run_webnlg")
    (tmp_path / "s.csv").write_text("leader,The leader of x\ncountry,Country of x\n", encoding="utf8")
    assert m.read_schema(tmp_path / "s.csv") == ["leader", "country"]
```

- [ ] **Step 2: Chạy, thấy fail.**

- [ ] **Step 3: Cài** — `run_webnlg.py`: đọc câu (1 dòng/câu), `ThreadPoolExecutor(workers)` gọi `extract_schema` (bọc try/except → `[]`), ghi theo đúng thứ tự dòng; `read_schema` dùng `csv.reader`, cột 0. `run_covid_nyt.py`: `rows = sorted(load_dataset(...), key=date)`, một `TemporalKG`, mỗi bài gọi `update(text, date, llm)` trong try/except (lỗi → `[]`), ghi jsonl ngay sau mỗi bài (flush) để kernel chết giữa chừng vẫn còn kết quả.

```python
# scripts/run_webnlg.py
"""Pipeline của mình (module Add, quan hệ ép theo schema) trên WebNLG test của EDC; ra file định dạng canon_kg.txt
để chấm bằng evaluate/evaluation_script.py của clear-nus/edc."""
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from kgu.extract.text_kg import extract_schema
from kgu.llm import client_from_env


def read_schema(path: Path) -> list[str]:
    with open(path, encoding="utf8") as f:
        return [row[0].strip() for row in csv.reader(f) if row and row[0].strip()]


def format_line(triples) -> str:
    return str([list(t) for t in triples])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--schema", type=Path, required=True)
    ap.add_argument("--examples", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    texts = args.input.read_text(encoding="utf8").splitlines()[: args.limit]
    rels = read_schema(args.schema)
    examples = args.examples.read_text(encoding="utf8") if args.examples else ""
    llm = client_from_env()

    def one(t: str):
        try:
            return extract_schema(t, rels, llm, examples)
        except Exception as e:                       # lỗi một câu → dòng rỗng, giữ thứ tự dòng
            print(f"!!! {e!r}", flush=True)
            return []

    with ThreadPoolExecutor(args.workers) as pool:
        lines = [format_line(r) for r in pool.map(one, texts)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf8")


if __name__ == "__main__":
    main()
```

```python
# scripts/run_covid_nyt.py
"""Pipeline của mình trên 2020-COVID-NYT: KG tích lũy theo thứ tự ngày (kgu.extract.text_kg.TemporalKG)."""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

from kgu.extract.text_kg import TemporalKG
from kgu.llm import client_from_env

_spec = importlib.util.spec_from_file_location("score_covid_nyt", Path(__file__).with_name("score_covid_nyt.py"))
_score = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_score)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    rows = sorted(_score.load_dataset(args.data), key=lambda r: (r["date"], r["row"]))[: args.limit]
    kg, llm = TemporalKG(), client_from_env()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf8") as f:
        for r in rows:
            t0 = time.time()
            try:
                qs = kg.update(r["text"], r["date"], llm)
            except Exception as e:
                print(f"!!! row {r['row']}: {e!r}", flush=True)
                qs = []
            f.write(json.dumps({"row": r["row"], "quintuples": [list(q) for q in qs],
                                "seconds": round(time.time() - t0, 2)}, ensure_ascii=False) + "\n")
            f.flush()
    Path(str(args.out) + ".kg.json").write_text(json.dumps(kg.facts, ensure_ascii=False), encoding="utf8")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Test pass + chạy khô** — `pytest -q` xanh; chạy `run_webnlg.py --limit 2` và `run_covid_nyt.py --limit 2` với server OpenAI giả (như `fake_llm.py` khi làm KARMA, trả `{"facts": [], "ended": []}`) để kiểm I/O end-to-end.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_webnlg.py scripts/run_covid_nyt.py tests/test_home_bench_io.py
git commit -m "feat: runners for our pipeline on WebNLG (EDC canon_kg format) and COVID-NYT (accumulating KG)"
```

### Task 5: Runner baseline trên COVID-NYT (ATOM, ATOM-direct, KGGen)

**Files:**
- Create: `scripts/atom_covid.py`, `scripts/kggen_covid.py`
- Test: `tests/test_home_bench_io.py` (thêm: `to_quintuple` của atom_covid)

**Interfaces:**
- Consumes: `load_dataset` (Task 2); itext2kg 1.1.0 (`LangchainOutputParser`, `itext2kg.atom.models.AtomicFact`, `RelationshipsExtractor`, `Prompt`); kg-gen 0.4.0.
- Produces: jsonl cùng định dạng Task 4 (`{"row", "quintuples", "seconds"}`); `atom_covid.py --mode atom|direct`.

- [ ] **Step 1: Test fail cho `to_quintuple(rel) -> tuple | None`**

```python
from types import SimpleNamespace as NS


def test_atom_relationship_to_quintuple_normalizes_dates_and_drops_broken():
    m = _mod("atom_covid")
    rel = NS(startNode=NS(name="Virus"), name="occurs_in", endNode=NS(name="China"), t_start="2020-01-24", t_end=[])
    assert m.to_quintuple(rel) == ("Virus", "occurs_in", "China", ["2020-01-24"], [])
    assert m.to_quintuple(NS(startNode=None, name="r", endNode=NS(name="x"), t_start=[], t_end=[])) is None
```

- [ ] **Step 2: Chạy, thấy fail.**

- [ ] **Step 3: Cài `atom_covid.py`** — y như `evaluation/atom/exhaustivity/factoids_extraction_nyt.py` + `quintuples_extraction_nyt_from_factoids.py` của itext2kg (mode `atom`: bước 1 `extract_information_as_json_for_context(AtomicFact, [paragraph])` → factoids; bước 2 `extract_information_as_json_for_context(output_data_structure=RelationshipsExtractor, contexts=factoids, system_query=Prompt.temporal_system_query(obs_timestamp=date) + Prompt.EXAMPLES.value)`; mode `direct`: bước 2 trên cả đoạn). LLM: lớp con `class ChatOpenAIJSON(ChatOpenAI)` ghi đè `with_structured_output(self, schema, **kw)` → `super().with_structured_output(schema, method="json_schema", **kw)` (giữ chữ "OpenAI" trong tên lớp để itext2kg nhận đúng provider), `base_url=KGU_LLM_BASE_URL`, `model="local"`, `temperature=0`. Embeddings: `HuggingFaceEmbeddings("sentence-transformers/all-mpnet-base-v2")` (chỉ để khởi tạo, bước trích không gọi). `kggen_covid.py`: `KGGen(model="hosted_vllm/local", api_base=..., api_key="none", temperature=0, max_tokens=4096).generate(input_data=paragraph)` → 5-tuple `(h, r, t, [], [])`. Cả hai: lỗi một bài → `[]`, ghi + flush từng dòng.

- [ ] **Step 4: Test pass** (`pytest -q`). Không chạy được thật trên máy local (không bật llama-server) → kiểm thật ở Task 6 bước chạy thử.

- [ ] **Step 5: Commit**

```bash
git add scripts/atom_covid.py scripts/kggen_covid.py tests/test_home_bench_io.py
git commit -m "feat: ATOM (factoids->quintuples, direct) and KGGen runners on COVID-NYT via llama-server"
```

### Task 6: Kernel Kaggle `kgu-home-benchmarks`

**Files:**
- Create: `kaggle/home_bench/run_home_bench.py`, `kaggle/home_bench/kernel-metadata.json`

**Interfaces:**
- Consumes: mọi script Task 2–5 (clone repo kltn nhánh `feat/core-pipeline`); dataset `ngocnam2005/llama-server-cuda`.
- Produces: `/kaggle/working/out/` — `webnlg/{edc,mine}.txt`, `webnlg/scores.json`, `covid/{mine,atom,atom-direct,kggen}.jsonl`, `covid/scores.json`, `summary.json` (giây/bài), log.

- [ ] **Step 1: Viết kernel** theo khung `kaggle/baselines/run_baselines.py` (chép phần dựng llama-server: binary cache, GGUF, `ATTEMPTS`, `healthy()`), thêm:
  1. `nvidia-smi --query-gpu=name,memory.total --format=csv` → `N_GPU`; llama-server ghim `CUDA_VISIBLE_DEVICES=0`.
  2. Clone `clear-nus/edc` (ghim commit HEAD lúc chạy, in ra log), `pip install nervaluate unidecode bs4 regex dateparser sentence-transformers`. Vá `edc/edc_framework.py::load_model` nhánh `sts`: `SentenceTransformer(model_name, trust_remote_code=True, device=EMB_DEVICE, model_kwargs={"torch_dtype": torch.float16})` (bản gốc nạp fp32 = 28 GB). `EMB_DEVICE = "cuda:1"` nếu `N_GPU >= 2`, embedder `intfloat/e5-mistral-7b-instruct` như paper; nếu 1 GPU thì `intfloat/multilingual-e5-large-instruct` trên `cuda:0` và ghi vào summary là sai khác so với paper.
  3. EDC trên WebNLG: `OPENAI_KEY=none OPENAI_BASE_URL=http://127.0.0.1:8080/v1 python run.py --oie_llm gpt-gemma --sd_llm gpt-gemma --sc_llm gpt-gemma --sc_embedder <EMB> --oie_few_shot_example_file_path ./few_shot_examples/webnlg/oie_few_shot_examples.txt --sd_few_shot_example_file_path ./few_shot_examples/webnlg/sd_few_shot_examples.txt --input_text_file_path <webnlg cắt LIMIT> --target_schema_path ./schemas/webnlg_schema.csv --output_dir <out>/edc_webnlg` (tên chứa "gpt" để EDC đi nhánh OpenAI; llama-server bỏ qua tên model).
  4. Mình trên WebNLG: `run_webnlg.py` với cùng file few-shot của EDC.
  5. Chấm WebNLG: `python evaluate/evaluation_script.py --edc_output <file> --reference evaluate/references/webnlg.txt` (cắt reference cùng LIMIT), lưu stdout; parse Partial/Strict/Exact F1 vào `scores.json`.
  6. COVID: tải `2020-COVID-NYT.pkl` từ HF; chạy `run_covid_nyt.py`, `atom_covid.py --mode atom`, `atom_covid.py --mode direct`, `kggen_covid.py` (itext2kg và kg-gen cài trong venv riêng `python -m venv --system-site-packages` để không đè gói của EDC).
  7. Tắt server, chấm COVID bằng `score_covid_nyt.py --device cuda` với embedder/ngưỡng đã chốt ở Task 2 (thêm `--calibrate` để số GPT-4.1 nằm cùng bảng).
  8. Mỗi bước bọc `timed()` như kernel baselines (lỗi một hệ thống không chặn các hệ thống sau); xóa repo/model khỏi output.
- [ ] **Step 2: Chạy thử** `LIMIT = 20` (20 câu WebNLG, 10 nhóm ngày COVID): đẩy kernel, đọc log, kiểm mỗi hệ thống có output khác rỗng, bộ chấm chạy hết, đo giây/bài để ước thời gian chạy đủ.
- [ ] **Step 3: Chạy đủ** `LIMIT = None` (1.165 câu WebNLG, 122 nhóm ngày). Nếu ước > 10 giờ: chia 2 kernel (WebNLG / COVID) bằng biến `BENCH`.
- [ ] **Step 4: Commit** kernel + `results/kaggle/home_bench_*.json` + log.

```bash
git add kaggle/home_bench/ results/kaggle/home_bench_*
git commit -m "feat: Kaggle kernel running EDC/ours on WebNLG and ATOM/KGGen/ours on COVID-NYT with Gemma 4"
```

### Task 7: Notes + bảng tổng

**Files:**
- Modify: `docs/data-notes-emerge.md` (thêm §17)

- [ ] **Step 1:** Ghi §17: cách chạy (phiên bản, vá), bảng hiệu chỉnh bộ chấm COVID (Task 2), bảng WebNLG (Partial/Strict/Exact P/R/F1: EDC Gemma, mình Gemma, kèm số EDC GPT-4 / Mistral-7B công bố để đối chiếu), bảng COVID (P/R/F1, MATCH_t, giây/bài: ATOM, ATOM-direct, KGGen, mình; kèm số GPT-4.1 của ATOM chấm lại bằng bộ chấm của mình), nhận xét đối chiếu với EMERGE §14–16.
- [ ] **Step 2:** Commit + push.

```bash
git add docs/data-notes-emerge.md
git commit -m "docs: home-benchmark comparison (WebNLG, COVID-NYT) with Gemma 4"
git push origin feat/core-pipeline
```
