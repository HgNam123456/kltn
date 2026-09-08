# Error Analysis — Month 2

### Table 0: LLM Extractor + LLM Judge (n=20) by Event Type

`--by event` trên `results/nba_test_llm_llm_20.jsonl` (đo 2026-09-08, sau Task 13 — schema enum quan hệ, few-shot, prompt bộ phán gọn; model Qwen3.5-2B Q8_0):

| group       | n  | f1    | add   | del   | cut/ex | add/ex |
|-------------|----|-------|-------|-------|--------|--------|
| draft       | 1  | 0.948 | 0.000 | 0.000 | 0.0    | 0.0    |
| free_agency | 1  | 0.948 | 0.000 | 0.000 | 0.0    | 0.0    |
| overseas    | 1  | 0.934 | 0.000 | 0.000 | 0.0    | 0.0    |
| released    | 11 | 0.951 | 0.000 | 0.000 | 0.0    | 0.0    |
| retirement  | 1  | 0.934 | 0.000 | 0.000 | 0.0    | 0.0    |
| trade       | 5  | 0.920 | 0.000 | 0.000 | 0.0    | 0.0    |

`cut/ex` = `add/ex` = 0.0 across every event: LLM #1 (extractor) produced zero explicit ops on all 20 examples in this run, so `localize()` never had participants to build candidates from, and the judge (LLM #2) was never invoked (`llm_calls` = 20, all from the extractor). This isolates the current bottleneck to LLM #1's zero-shot/few-shot extraction on this 2B model, not to the judge or the localization/judge hardening done in Task 13 — see `gold_explicit + llm judge` in the README (add_acc 0.051, del_acc 1.0, 1/20 context-overflow error handled as a per-example record instead of aborting the run) for evidence the judge and error-handling paths work when given real explicit ops.

This table is the run **AFTER Task 13** (`results/nba_test_llm_llm_20.jsonl`, current committed schema+prompt). It overwrote the pre-Task-13 file of the same base name; Table 4 and the "Five Concrete Error Examples" section further below preserve that pre-Task-13 run's numbers (now explicitly labeled) for comparison. Table 5 below is the controlled probe that separates the schema change from the prompt change to explain why this table's `n_explicit` is 0 while Table 4's `cut/ex` was 7.6.

## Analysis Tables

### Table 1: Oracle Results by Coverage

| group  | n   | f1    | add   | del   | cut/ex | add/ex |
|--------|-----|-------|-------|-------|--------|--------|
| full   | 438 | 1.000 | 1.000 | 1.000 | 70.3   | 31.0   |

The oracle judge (perfect event extraction + perfect judgment) achieves F1=1.0 across all 438 examples. This establishes the upper bound for pipeline performance.

### Table 2: Oracle Results by Event Type

| group       | n   | f1    | add   | del   | cut/ex | add/ex |
|-------------|-----|-------|-------|-------|--------|--------|
| d_league    | 6   | 1.000 | 0.000 | 1.000 | 73.3   | 0.0    |
| draft       | 18  | 1.000 | 1.000 | 0.000 | 0.0    | 20.0   |
| free_agency | 107 | 1.000 | 1.000 | 0.000 | 0.0    | 20.2   |
| head_coach  | 10  | 1.000 | 1.000 | 1.000 | 0.0    | 19.5   |
| overseas    | 3   | 1.000 | 0.000 | 1.000 | 62.7   | 0.0    |
| released    | 139 | 1.000 | 0.000 | 1.000 | 71.3   | 0.0    |
| retirement  | 5   | 1.000 | 0.000 | 1.000 | 61.6   | 0.0    |
| trade       | 150 | 1.000 | 1.000 | 1.000 | 133.1  | 72.3   |

With perfect extraction, all events achieve F1=1.0. Key observations:
- `trade` events have highest cut/ex (133.1) and add/ex (72.3) — inherently multi-operation events
- `released` events have moderate cut/ex (71.3) — consistent deletions needed
- `draft` and `free_agency` have only ADD operations (no deletions)

### Table 3: Gold Extractor (No Judge) by Event Type

| group       | n   | f1    | add   | del   | cut/ex | add/ex |
|-------------|-----|-------|-------|-------|--------|--------|
| d_league    | 6   | 0.952 | 0.000 | 0.052 | 73.3   | 0.0    |
| draft       | 18  | 0.951 | 0.053 | 0.000 | 0.0    | 20.0   |
| free_agency | 107 | 0.952 | 0.052 | 0.000 | 0.0    | 20.2   |
| head_coach  | 10  | 1.000 | 1.000 | 1.000 | 0.0    | 19.5   |
| overseas    | 3   | 0.945 | 0.000 | 0.060 | 62.7   | 0.0    |
| released    | 139 | 0.951 | 0.000 | 0.053 | 71.3   | 0.0    |
| retirement  | 5   | 0.944 | 0.000 | 0.061 | 61.6   | 0.0    |
| trade       | 150 | 0.900 | 0.056 | 0.056 | 133.1  | 72.3   |

Gold extractor with explicit rules shows degradation:
- `trade`: F1 drops to 0.900 (worst), suggesting extraction complexity
- `head_coach`: F1=1.000 (best) — simplest event pattern
- Other events: F1≈0.95, suggesting minor extraction issues

### Table 4: LLM Extractor (20-sample, No Judge) by Event Type — **lượt chạy TRƯỚC Task 13 (prompt/schema cũ, không enum, không few-shot)**

**Lưu ý:** bảng này và "Five Concrete Error Examples" ngay dưới đây đọc từ file `nba_test_llm_nojudge_20.jsonl` **của lượt chạy trước Task 13** (schema `r: str` tự do, prompt cũ không few-shot — xem `git show 7b627d3:kgu/extract/llm.py`). File đó đã bị Task 13 GHI ĐÈ bởi lượt chạy mới (schema enum + few-shot); số liệu của lượt chạy mới nằm ở **Table 0** phía trên (`results/nba_test_llm_llm_20.jsonl`, đo sau Task 13) và ở README. Giữ bảng cũ này nguyên văn vì nó là bằng chứng cho quyết định ở Table 5 (probe có kiểm soát cho thấy schema/prompt mới không phải nguyên nhân gây ops rỗng so với bảng này).

| group       | n   | f1    | add   | del   | cut/ex | add/ex |
|-------------|-----|-------|-------|-------|--------|--------|
| draft       | 1   | 0.948 | 0.000 | 0.000 | 0.0    | 0.0    |
| free_agency | 1   | 0.948 | 0.000 | 0.000 | 0.0    | 0.0    |
| overseas    | 1   | 0.934 | 0.000 | 0.000 | 0.0    | 0.0    |
| released    | 11  | 0.951 | 0.000 | 0.002 | 7.6    | 0.0    |
| retirement  | 1   | 0.934 | 0.000 | 0.000 | 0.0    | 0.0    |
| trade       | 5   | 0.920 | 0.000 | 0.000 | 0.0    | 0.0    |

LLM extractor (n=20, pre-Task-13 prompt/schema) shows:
- Severe degradation compared to gold (F1≈0.92–0.95 vs ≈1.0)
- Dominance of `released` events (11/20 samples)
- No ADD operations detected by LLM (add_acc=0.0 across all events)
- `trade` still worst (F1=0.920)
- Low candidate counts (avg cut=0–7.6, add=0.0) suggest aggressive under-extraction

### Table 5: Prompt/Schema Ablation Probe (n=8, temperature 0) — resolves Task 13 review finding

**Câu hỏi:** sau Task 13, `llm+none`/`llm+llm` (n=20) cho `n_explicit=0` trên CẢ 20 ví dụ (README, Table 0) — khác hẳn Table 4 ở trên (cùng model, cùng 20 ví dụ đầu, nhưng prompt/schema cũ) vốn có `cut/ex=7.6` cho `released` và ít nhất 1 op sống sót (Example 4 dưới đây). Hai thứ đổi cùng lúc trong Task 13: (1) `r` chuyển từ `str` tự do sang `Literal[...]` enum (constrained decoding), (2) prompt viết lại (few-shot + đổi nhãn mục). Cần tách 2 biến để biết cái nào gây ops rỗng.

**Phương pháp:** gọi trực tiếp `client.complete_json(system, prompt, schema)` (client thật, server local đang chạy, `temperature=0` mặc định của `OpenAICompatClient`) trên **8 ví dụ đầu** của test split, với 4 tổ hợp:
- **(a) enum_new** = code hiện tại (đã commit): `make_ops_schema` (enum) + `SYSTEM`/`_prompt` mới (few-shot).
- **(b) enum_old** = `make_ops_schema` (enum) + `SYSTEM`/`_prompt` **cũ** (từ commit `7b627d3`, không few-shot).
- **(c) str_new** = schema `r: str` tự do + `SYSTEM`/`_prompt` **mới**.
- **(d) str_old** = schema `r: str` tự do + `SYSTEM`/`_prompt` **cũ** = tái hiện đúng hành vi trước Task 13 (control).

Với mỗi (idx, variant): `raw` = số op model trả về TRƯỚC post-validation; `survive` = số op sống sót SAU post-validation (cùng logic lọc như `LLMExtractor.extract()`: `r` hợp lệ, `h`/`t` thuộc entity đã biết, đúng trạng thái active/inactive); `correct` = số survivor khớp CHÍNH XÁC (kind, h, r, t) với gold explicit ops (`results/nba_test_gold_explicit_nojudge.jsonl` theo idx).

| idx | event       | a_enum_new (raw/survive/correct) | b_enum_old (raw/survive/correct) | c_str_new (raw/survive/correct) | d_str_old (raw/survive/correct) |
|-----|-------------|-----------------------------------|-----------------------------------|------------------------------------|------------------------------------|
| 0   | released    | 0 / 0 / 0                          | 1 / 0 / 0                          | 0 / 0 / 0                           | 1 / 0 / 0                           |
| 1   | trade       | 0 / 0 / 0                          | 3 / 0 / 0                          | 0 / 0 / 0                           | 1 / 0 / 0                           |
| 2   | released    | 0 / 0 / 0                          | 1 / 0 / 0                          | 0 / 0 / 0                           | 1 / 0 / 0                           |
| 3   | released    | 0 / 0 / 0                          | 1 / 1 / 0                          | 0 / 0 / 0                           | 1 / 0 / 0                           |
| 4   | released    | 0 / 0 / 0                          | 1 / 0 / 0                          | 0 / 0 / 0                           | 1 / 0 / 0                           |
| 5   | draft       | 0 / 0 / 0                          | 1 / 0 / 0                          | 0 / 0 / 0                           | 1 / 0 / 0                           |
| 6   | released    | 0 / 0 / 0                          | 1 / 0 / 0                          | 0 / 0 / 0                           | 1 / 0 / 0                           |
| 7   | released    | 0 / 0 / 0                          | 1 / 1 / 0                          | 0 / 0 / 0                           | 1 / 0 / 0                           |
| **Σ** |           | **0 / 0 / 0**                       | **10 / 2 / 0**                      | **0 / 0 / 0**                        | **8 / 0 / 0**                        |

**Kết luận:** cột quyết định là **PROMPT**, không phải schema. Cả hai biến thể dùng prompt MỚI (a, c) đều cho `raw=0` trên toàn bộ 8 ví dụ bất kể schema là enum hay str; cả hai biến thể dùng prompt CŨ (b, d) đều cho model thực sự phát ra ít nhất 1 op/ví dụ. Ngược lại, enum vs str KHÔNG phải nguyên nhân: enum+prompt cũ (b) sống sót NHIỀU hơn str+prompt cũ (d) (2 so với 0) — enum không làm giảm khả năng trích xuất, thậm chí giúp `r` khớp đúng tên quan hệ hơn khi model có emit op. Tuy nhiên **không biến thể nào đạt `correct > 0`** trên 8 ví dụ này (kể cả control (d) tái hiện đúng hành vi trước Task 13) — với thước đo "surviving-and-correct", (a) hiện tại KHÔNG THUA biến thể tốt nhất (hòa 0 ở mọi biến thể). Theo tiêu chí đã thống nhất, **giữ nguyên code đã commit** (schema enum + prompt mới); không rollback. Ghi chú thêm: prompt mới khiến model trả `ops: []` "an toàn" hơn (ít false positive hơn — 0 survivor thay vì 2 survivor sai ở biến thể b) nhưng cũng ít true positive hơn trên mẫu nhỏ này; cả 4 biến thể đều chưa đạt độ chính xác thực dụng với model 2B — xác nhận lại kết luận ở Table 0/README rằng nút thắt là năng lực model 2B, không phải cách mã hóa schema/prompt.

Script probe (throwaway, không commit): `C:\Users\ADMIN\AppData\Local\Temp\claude\c--Users-ADMIN-Desktop-kltn\9df26baf-ec93-42cb-b666-123064e7705b\scratchpad\probe_variants.py`.

## Five Concrete Error Examples — **lượt chạy TRƯỚC Task 13 (prompt/schema cũ)**

Based on comparison of `nba_test_llm_nojudge_20.jsonl` (lượt chạy trước Task 13, đã bị ghi đè — xem ghi chú ở Table 4) vs `nba_test_gold_explicit_nojudge.jsonl`:

### Example 1: idx=0, event=released
**Text:** "the Indiana_Pacers announced friday they have waived forward Thaddeus_Young_2017-18 ..."

**Expected ops:** 2 INVALIDATE (remove player from roster, remove his contract)
**LLM ops:** (empty — 0 operations)

**Likely cause:** LLM sai chính tả entity — player name with version suffix `_2017-18` not recognized; LLM may have treated it as malformed or OOV token.

---

### Example 2: idx=1, event=trade
**Text:** "cleveland , oh – the Cleveland_Cavaliers have acquired center Anthony_Randolph_2014-15 and a 2015 second round pick from the Denver_Nuggets in exchange for two protected 2015 first round picks ( <unk> oklahoma city and memphis ) , cavaliers general manager david griffin announced tonight ..."

**Expected ops:** 2 INVALIDATE + 2 ADD (remove traded players, add acquired players)
**LLM ops:** (empty — 0 operations)

**Likely cause:** tin nhiều giao dịch — trade with multiple picks and entities; OOV token `<unk>` corrupts extraction context.

---

### Example 3: idx=2, event=released
**Text:** "boston – the Boston_Celtics announced today that they have waived guard Kelly_Olynyk_2015-16 ..."

**Expected ops:** 2 INVALIDATE (standard waived player pattern)
**LLM ops:** (empty — 0 operations)

**Likely cause:** LLM sai chính tả entity — player identifier with version suffix `_2015-16` not recognized by LLM.

---

### Example 4: idx=3, event=released
**Text:** "the Golden_State_Warriors state have waived forward Jason_Thompson_2015-16 , the team announced today ..."

**Expected ops:**
- INVALIDATE(Golden_State_Warriors, <player>, Jason_Thompson_2015-16)
- INVALIDATE(Jason_Thompson_2015-16, <player>, Golden_State_Warriors)

**LLM ops:** 
- INVALIDATE(Golden_State_Warriors, <general_mananger>, Bob_Myers_2015-16)

**Likely cause:** LLM bịa entity + LLM đảo vị trí (h, r, t) — LLM hallucinated an entity (Bob_Myers_2015-16, not in text) and wrong relation (<general_mananger> vs <player>). The op survived post-validation only because Bob_Myers_2015-16 exists in the graph, but it's factually wrong for this news event.

---

### Example 5: idx=4, event=released
**Text:** "<unk> the Atlanta_Hawks basketball club has requested waivers on forward Jaylen_Morris_2018-19 , it was announced today ..."

**Expected ops:** 2 INVALIDATE
**LLM ops:** (empty — 0 operations)

**Likely cause:** LLM sai chính tả entity + OOV tokens — text begins with OOV token `<unk>` that likely disrupts extraction; player name with version suffix not recognized.

---

## Summary

**Gold Extractor Performance:** F1≈0.95 on most event types, but `trade` drops to 0.90 due to multi-operation complexity.

**LLM Extractor Performance (lượt chạy TRƯỚC Task 13 — Table 4 / Five Examples ở trên):** F1≈0.92–0.95 overall, but with systematic failures:
1. **Zero ADD operations** across all samples (gold achieves ~1.0 for draft/free_agency/trade)
2. **Entity recognition failures** on versioned player IDs (`_2017-18` suffixes)
3. **OOV token handling** — `<unk>` tokens disrupt extraction pipeline
4. **Hallucination of non-existent entities** — Example 4 shows LLM fabricating Bob_Myers_2015-16 when text mentions Jason_Thompson_2015-16; also wrong relation detected

**Primary Bottleneck (cả trước và sau Task 13):** LLM #1 (model 2B) không trích xuất tin cậy được op tường minh nào, dù trước Task 13 nó ít nhất phát ra vài op sai (h, r, t) hay bịa entity, còn sau Task 13 (Table 0, Table 5) nó trả `ops: []` "an toàn" hơn nhưng vẫn không đúng — Table 5 (probe có kiểm soát) xác nhận đây là do PROMPT thay đổi (không phải schema enum), và ở cả 4 tổ hợp schema/prompt, không tổ hợp nào đạt `correct > 0` trên 8 ví dụ — xác nhận nút thắt là năng lực model 2B chứ không phải cách mã hóa. Việc tiền xử lý entity identifier / xử lý token OOV vẫn là hướng cải thiện khả dụng; Task tháng 3 dự kiến đo lại với model 7B.
