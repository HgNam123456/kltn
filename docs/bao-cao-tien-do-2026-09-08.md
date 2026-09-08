# Báo cáo tiến độ — Cập nhật knowledge graph từ bản tin bằng LLM

Ngày: 2026-09-08 (bản 2, sau khi đổi bộ phán sang phán theo mẫu quan hệ, 2 pha)
Nhánh: `feat/core-pipeline` (24 commit, 64 test pass, chưa merge vào `main`)
Spec gốc: trang Notion "🎯 kế hoạch" (bản sao `docs/superpowers/specs/2026-09-05-ke-hoach-notion.md`)
Plan thực thi: `docs/superpowers/plans/2026-09-07-kg-update-core-pipeline.md` (13 task, đã xong)

---

## 1. Tóm tắt điều hành

**Đã làm được gì.** Toàn bộ pipeline tháng 1–2 của lộ trình đã được code, test và review: đọc dataset NBAtransactions, đồ thị song thời gian (bi-temporal), executor áp ops, khoanh vùng cấu trúc 2 vế, bộ phán (oracle và LLM), bộ trích ops tường minh (gold và LLM), metric theo GUpdate, CLI đánh giá, script phân tích lỗi, và hạ tầng LLM local chạy trên GPU AMD 780M. Sau bản báo cáo đầu, bộ phán LLM được thiết kế lại: thay vì liệt kê từng cạnh ứng viên, nó gom ứng viên thành **mẫu quan hệ**, phán từng mẫu và ánh xạ phán quyết xuống mọi cạnh trong mẫu; chạy thành **2 pha** (pha CẮT giữ/bỏ, pha SINH thêm mới). Prompt giảm 3,5–4,6 lần, mục tiêu dưới 3.000 token trong spec đã đạt.

**Kết quả quan trọng nhất.** Trên toàn bộ split test (438 tin), với ops tường minh lấy từ dữ liệu vàng và bộ phán oracle, khoanh vùng 2 vế đạt **coverage = 1,0** và pipeline dựng lại KG-sau với **F1 = 1,0**. Kiến trúc "ops tường minh → khoanh vùng → phán" là đủ để giải bài toán trên miền này. Con số "chỉ 5,6% thay đổi được nói thẳng trong text" của paper GUpdate được tái lập chính xác (0,056).

**Chưa được gì.** Phần LLM thật chưa cho số so sánh được với GUpdate. Model Qwen3.5-2B chạy local không trích được op tường minh nào. Ở vai bộ phán, sau khi đổi sang phán theo mẫu, model 2B lần đầu cho phán quyết có nghĩa (một tin trade cắt đúng 30/30 và thêm đúng 44/44 cạnh đồng đội) nhưng vẫn chọn sai 1–2 trong 4 mẫu ở đa số tin, nên add_acc chỉ 0,05–0,29. Đây là giới hạn suy luận ngữ nghĩa quan hệ của model 2B, không còn là vấn đề prompt. Bước tiếp theo là thử model 4–8B (Gemma 4 E4B, Qwen2.5-7B) trên cùng prompt, giờ đã đủ ngắn để chạy được.

---

## 2. Mục tiêu theo spec và mức đạt

| Mục tiêu spec | Trạng thái |
|---|---|
| LLM #1 trích ops tường minh với constrained decoding | Code xong; constrained decoding hoạt động (JSON schema, enum quan hệ). Model 2B không đủ năng lực. |
| Executor bi-temporal, không xóa chỉ vô hiệu hóa | Xong, in-memory, có test lịch sử khoảng hiệu lực. Neo4j để plan tháng 3–4. |
| Khoanh vùng 2 vế, không hop, không luật miền | Xong. Coverage 1,0 trên 438 tin. |
| Bộ phán A (LLM) | Code xong, đã qua 3 thiết kế prompt (từng cạnh → theo mẫu 1 pha → theo mẫu 2 pha). Prompt < 3k token. Model 2B chưa đủ. |
| Bộ phán B/C (AnyBURL/ULTRA, lai) | Chưa, đúng lộ trình tháng 3. |
| Đo so KG-sau theo GUpdate | Xong, có bảng và mốc so sánh. |
| Bản gốc dữ liệu JSONL kèm provenance | Xong (`source`, `quote`, `conf` trên mỗi op; record JSONL mỗi tin). |
| Mốc tháng 1: F1 cao với ops vàng | **Đạt**, F1 = 1,0. |
| Mốc tháng 2: bảng 1 sơ bộ + phân tích lỗi | Đạt về công cụ; số LLM là cận dưới n=20 với model 2B. |

---

## 3. Quá trình làm

Cách làm: plan viết trước với code và test đầy đủ cho từng task. Mỗi task do một subagent implementer làm theo TDD, một reviewer độc lập chấm spec và chất lượng, có vòng sửa nếu cần; cuối cùng một review toàn nhánh và một đợt sửa gộp. Phần thiết kế lại bộ phán (sau review) làm trực tiếp, có test trước khi chạy trên model thật.

| Task | Nội dung | Review |
|---|---|---|
| 1 | Scaffold package `kgu`, kiểu `Op`/`Triple`, tải data, spike khảo sát dữ liệu | sạch |
| 2 | Loader NBAtransactions → `Example` có tên thật | sạch |
| 3 | `BiTemporalGraph` in-memory | sạch, 1 điểm hoãn |
| 4 | `diff` ops vàng, `apply_ops` executor, JSONL provenance | sạch |
| 5 | Metric F1 / add_acc / del_acc kiểu GUpdate | sạch |
| 6 | Khoanh vùng cấu trúc 2 vế `localize` | sạch |
| 7 | Protocol `Extractor`/`Judge`, `OracleJudge`, gold extractors | 1 vòng sửa |
| 8 | Pipeline end-to-end, coverage, CLI `run_eval` → mốc tháng 1 | 1 vòng sửa |
| 9 | `OpenAICompatClient` + `FakeLLMClient` | 1 vòng sửa |
| 10 | `LLMExtractor` (LLM #1) | sạch; phát hiện model 2B đảo vị trí (h, r, t) |
| 11 | `LLMJudge` bản 1: liệt kê từng cạnh ứng viên | sạch; phát hiện prompt vượt ctx 8k |
| 12 | `analyze_errors` + tài liệu phân tích lỗi tháng 2 | 1 vòng sửa |
| 13 | Enum quan hệ trong schema, few-shot, gom theo hub, ghi lỗi từng ví dụ, ctx 16k | 1 vòng sửa |
| Review toàn nhánh | 1 Critical (README mô tả bộ phán quá lạc quan), 6 Important, 11 Minor | đợt sửa gộp, re-review sạch |
| Bổ sung | `LLMJudge` bản 2: phán theo mẫu quan hệ, 2 pha; chuẩn hóa cạnh bằng chứng trong `localize` | 64 test, commit f23aebc |

Hạ tầng dựng kèm: llama.cpp bản Vulkan b10839 tại `tools/llama.cpp/`, model `models/Qwen3.5-2B-Q8_0.gguf` (1,87 GB), script `scripts/llm_server.ps1`, đo được sinh ~20 token/giây, đọc prompt ~106 token/giây trên Radeon 780M.

---

## 4. Hệ thống hiện tại

### 4.1 Cấu trúc mã

```
kgu/
  types.py          Triple = (h, r, t); Op(kind ADD|INVALIDATE, h, r, t, source, quote, conf)
  data/__init__.py  Example(idx, event, season, text, mentioned, before, after)
  data/nba.py       Vocab (entity/relation/token id→tên), load_split()
  graph.py          BiTemporalGraph: add / invalidate / active(at) / neighbors / edges_of / history
  ops.py            diff(before, after) → ops vàng; apply_ops(graph, ops, at); JSONL I/O
  localize.py       localize(graph, explicit_ops, at) → Localization(cut, add)
  extract/          Extractor protocol; GoldAllExtractor, GoldExplicitExtractor, LLMExtractor
  judge/            JudgeContext, Judge protocol; OracleJudge
  judge/llm.py      CutPattern, AddPattern, group_cut(), group_add(); LLMJudge 2 pha
  llm.py            OpenAICompatClient (json_schema → json_object fallback, retry), FakeLLMClient
  eval/metrics.py   Counts, score(), summarize()
  eval/coverage.py  coverage(): recall của khoanh vùng trên các op ngầm
  pipeline.py       run_example(ex, extractor, judge, relations) → PipelineResult
scripts/
  download_nba.py, inspect_nba.py, run_eval.py, analyze_errors.py, llm_server.ps1, llm_smoke.py
tests/              64 test, fixture thế giới nhỏ Messi/Barca/Inter/Pedri/Lautaro
```

Khoảng 2.100 dòng Python. Không có mã riêng cho miền NBA trong `kgu/` ngoài loader; tên quan hệ chỉ xuất hiện trong dữ liệu, fixture test và ví dụ few-shot.

### 4.2 Luồng dữ liệu, từ file đến số đo

```
data/raw/nba/NBAtransactions_test.json      (GUpdater: text = token id, subgraph = [h, r, t] id)
  + entity2id.txt / relation2id.txt / token2id.txt
        │  Vocab.load + load_split
        ▼
Example(text tiếng Anh, mentioned = entity được nhắc, before = set triple, after = set triple)
        │
        ▼
[t = 0]  graph = BiTemporalGraph.from_triples(before)
        │
        ▼
[1] Extractor.extract(ex, graph, at=0) → explicit_ops
        gold_all      : toàn bộ diff(before, after)                      (cận trên, không cần khoanh vùng)
        gold_explicit : diff nhưng chỉ op có cả h và t ∈ mentioned       (xấp xỉ "được nói thẳng")
        llm           : prompt(text + cạnh hiện có của entity) → JSON ops → hậu kiểm
        │
        ▼
[2] apply_ops(graph, explicit_ops, at=1)    INVALIDATE = đóng valid_to = 1; ADD = mở valid_from = 1
        │
        ▼
[3] localize(graph, explicit_ops, at=1)
        participants = mọi entity trong explicit_ops
        vế cắt : hub kề a hoặc b của cạnh bị vô hiệu, và kề ≥ 2 participant → mọi cạnh hub–participant
        vế sinh: với ADD (a, r, b): mọi n ∈ neighbors(b) \ neighbors(a) → cặp (a, n) + cạnh bằng chứng b–n
        │
        ▼
[3b] group_cut / group_add: gom ứng viên thành MẪU
        mẫu cắt : (participant p, quan hệ r, hướng)  → "(p, r, X)" hoặc "(X, r, p)", X chạy trên các hub
        mẫu thêm: (subject a, cạnh bằng chứng)        → "a ~ X, bằng chứng (X, r', b)"
        │
        ▼
[4] Judge.judge(JudgeContext(text, explicit_ops, loc, graph, at=1, relations)) → judged_ops
        none   : không phán
        oracle : nhìn KG-sau (cận trên của bộ phán)
        llm    : pha CẮT  → {invalidate: [i...]}          i = chỉ số mẫu; áp cho MỌI cạnh trong mẫu
                 pha SINH → {add: [{idx, r, direction}]}  áp cho MỌI X trong mẫu; r ép theo enum quan hệ
        │
        ▼
[5] apply_ops(graph, judged_ops, at=1);  pred_after = graph.active(1)
        │
        ▼
score(before, after, pred_after) → Counts(tp, fp, fn, add_hit, n_add, del_hit, n_del)
coverage(ex, explicit_ops, loc) → (số op ngầm nằm trong vùng ứng viên, tổng op ngầm)
        │
        ▼
results/<tên>.jsonl   (1 dòng / tin: counts, coverage, số ứng viên, ops có provenance, hoặc record lỗi)
results/<tên>.summary.json  (micro F1, add_acc, del_acc, coverage, judged_ratio, errors, thời gian)
        │
        ▼
scripts/analyze_errors.py --by event | n_gold | coverage
```

Quy ước thời gian: số nguyên, KG-trước là t = 0, sau khi áp tin là t = 1. Không xóa cạnh; lịch sử truy vấn được qua `graph.history(triple)`.

### 4.3 Prompt bộ phán trông thế nào

Tin idx=1 (Cavaliers lấy Anthony Randolph từ Nuggets). Phần đầu prompt là bản tin, 4 ops tường minh đã áp, entity tham gia và 4 quan hệ cho phép. Phần ứng viên, trước đây là 98 dòng liệt kê từng cạnh, giờ là:

```
MẪU CẮT (cạnh hiện có; mặc định KEEP):
  [C0] (X, <teammate>, Anthony_Randolph_2014-15) — 15 X: Aaron_Brooks_2014-15, Andre_Miller_2014-15, ...; X cũng nối với Denver_Nuggets
  [C1] (Anthony_Randolph_2014-15, <teammate>, X) — 15 X: ...; X cũng nối với Denver_Nuggets
  [C2] (X, <player>, Denver_Nuggets) — 15 X: ...; X cũng nối với Anthony_Randolph_2014-15
  [C3] (Denver_Nuggets, <player>, X) — 15 X: ...; X cũng nối với Anthony_Randolph_2014-15

MẪU THÊM (cặp chưa nối trong bối cảnh mới; mặc định bỏ qua):
  [A0] Anthony_Randolph_2014-15 ~ X — 21 X: Alonzo_Gee_2014-15, ...; bằng chứng: (X, <player>, Cleveland_Cavaliers)
  [A1] Anthony_Randolph_2014-15 ~ X — 1 X: David_Griffin_2014-15; bằng chứng: (X, <general_mananger>, Cleveland_Cavaliers)
  [A2] Anthony_Randolph_2014-15 ~ X — 1 X: Mike_Brown_2014-15; bằng chứng: (X, <head_coach>, Cleveland_Cavaliers)
  [A3] Cleveland_Cavaliers ~ X — 15 X: Aaron_Brooks_2014-15, ...; bằng chứng: (X, <teammate>, Anthony_Randolph_2014-15)
```

Đáp án đúng: invalidate [0, 1]; add A0 với r = `<teammate>` cả 2 hướng. Bộ phán trả về chỉ số mẫu, mã ánh xạ xuống 30 cạnh cắt và 42 cạnh thêm.

Kích thước prompt (system + user, tokenizer Qwen):

| Tin | Bản 1, từng cạnh | Bản 2, theo mẫu | Số mẫu |
|---|---|---|---|
| idx=1, trade 1 cầu thủ, 60 cắt + 38 thêm | 5.307 | 1.469 | 4 + 4 |
| idx=15, trade 4 cầu thủ, 192 cắt + 104 thêm | 14.742 (vượt ctx) | 3.200 | 12 + 14 |

Lý do bản 1 phình: tên entity kèm mùa giải tốn 12 token (`Anthony_Randolph_2014-15` → 12 mảnh), đồ thị lưu 2 chiều nên mỗi hub sinh 4 cạnh, và số ứng viên tỉ lệ với số cầu thủ trong tin nhân sĩ số đội.

### 4.4 Vận hành

```
# một lần
python -m venv .venv && .venv\Scripts\pip install -e ".[dev]"
python scripts/download_nba.py

# đánh giá không cần LLM (vài giây)
python scripts/run_eval.py --extractor gold_all      --judge none   --out results/nba_test_gold_all.jsonl
python scripts/run_eval.py --extractor gold_explicit --judge none   --out results/nba_test_gold_explicit_nojudge.jsonl
python scripts/run_eval.py --extractor gold_explicit --judge oracle --out results/nba_test_gold_explicit_oracle.jsonl

# LLM local
.\scripts\llm_server.ps1                      # llama-server, Vulkan, ctx 16384, cổng 8080
python scripts/llm_smoke.py                   # kiểm tra JSON schema + tốc độ
python scripts/run_eval.py --extractor gold_explicit --judge llm --limit 20 --out results/x.jsonl
python scripts/run_eval.py --extractor llm --judge llm --limit 20 --out results/y.jsonl

# phân tích
python scripts/analyze_errors.py results/x.jsonl --by event
python -m pytest -q -W error
```

Đổi model: `.\scripts\llm_server.ps1 -Model models\<file>.gguf`. Đổi endpoint: biến môi trường `KGU_LLM_BASE_URL`, `KGU_LLM_MODEL`, `KGU_LLM_API_KEY`, `KGU_LLM_EXTRA_BODY` (xem README).

---

## 5. Dữ liệu

**Nguồn.** NBAtransactions từ repo GUpdater (Tang et al., EMNLP 2019), tải trực tiếp từ GitHub bằng `scripts/download_nba.py`. Split test 438 tin (paper ghi 422; file thực tế 438), train 3.262, valid 399.

**Định dạng gốc.** Mỗi tin là object `{event, season, text, text_mentioned_entities, subgraph_before, subgraph_after}`. `text` là dãy token id, `subgraph_*` là danh sách `[h, r, t]` theo id. Ba file từ điển có dòng đầu là số đếm.

**Phát hiện khi khảo sát.**
- 4 quan hệ: `<player>`, `<teammate>`, `<head_coach>`, `<general_mananger>` (paper viết sai chính tả, giữ nguyên).
- `<player>` và `<teammate>` lưu **cả hai chiều**. Pipeline vì thế không giả định chiều; bộ phán có thể thêm 2 chiều.
- Entity cầu thủ gắn mùa giải: `Thaddeus_Young_2017-18`.
- Trung bình mỗi tin ở split test: 38,2 cạnh thêm, 39,7 cạnh xóa.
- Phân bố event: trade 150, released 139, free_agency 107, draft 18, head_coach 10, d_league 6, retirement 5, overseas 3.
- Text có ký tự lạ `�` và `<unk>` do tiền xử lý của paper.

**Ops vàng và "ops tường minh".** Ops vàng = `diff(before, after)`. Vì dataset không đánh dấu op nào được nói thẳng trong text, "ops tường minh" được xấp xỉ bằng các op có cả hai đầu thuộc `text_mentioned_entities`. Cách xấp xỉ này cho add/del acc = 0,056, trùng với "5,6%" paper báo cáo.

---

## 6. Kết quả và so sánh với GUpdate

### 6.1 Số của GUpdate (paper, split test)

| Hệ thống | F1 | Acc cạnh thêm | Acc cạnh xóa |
|---|---|---|---|
| GUpdate (GNN) | 0,9866 | 0,8926 | 0,8988 |
| Baseline IE thuần (IE + R-GCN) | 0,9429 | 0,4681 | 0,2448 |

### 6.2 Số của hệ thống này

Ba dòng đầu chạy trên toàn split test 438 tin; các dòng LLM chạy n=20 với Qwen3.5-2B.

| # | Cấu hình | F1 | add_acc | del_acc | coverage | Ghi chú |
|---|---|---|---|---|---|---|
| 1 | ops vàng đầy đủ, không phán | 1,000 | 1,000 | 1,000 | – | kiểm tra executor + metric |
| 2 | ops tường minh, không lan truyền | 0,926 | 0,056 | 0,056 | 1,0 | tương đương "IE thuần" |
| 3 | ops tường minh + khoanh vùng + oracle | 1,000 | 1,000 | 1,000 | 1,0 | cận trên của bộ phán |
| 4 | LLM #1 (2B), không phán | 0,937 | 0,000 | 0,000 | – | model trả rỗng |
| 5 | ops tường minh + judge bản 1 (từng cạnh) | 0,944 | 0,051 | 1,000* | 1,0 | cắt 1.329/1.332 ứng viên |
| 6 | ops tường minh + judge bản 2, theo mẫu, 1 pha | 0,923 | 0,287 | 1,000* | 1,0 | lần đầu có phán quyết đúng |
| 7 | ops tường minh + judge bản 2, theo mẫu, 2 pha | 0,925 | 0,051 | 0,775 | 1,0 | chọn mẫu thất thường |
| 8 | LLM #1 + judge (2B) | 0,937 | 0,000 | 0,000 | – | không có op vào |

\* del_acc 1,0 ở dòng 5 và 6 là vô nghĩa: bộ phán cắt cả 4 mẫu (dòng 6) hoặc gần hết ứng viên (dòng 5); vì coverage = 1,0 nên mọi cạnh cần xóa đều nằm trong tập bị cắt, trả giá bằng hàng trăm cạnh đúng bị xóa oan.

Mỗi dòng LLM 20 tin đều có 1 tin lỗi (vượt ctx ở bản 1, timeout 120 s ở bản 2 với tin trade 4 cầu thủ). Thời gian: bản 1 khoảng 60–95 s mỗi lệnh gọi, bản 2 khoảng 20 s.

### 6.3 So với GUpdate

**Cùng dataset, cùng split, cùng ba chỉ số.** Metric ở đây làm theo định nghĩa của GUpdate: cạnh thêm = KG-sau trừ KG-trước, cạnh xóa = KG-trước trừ KG-sau, F1 micro trên toàn bộ cạnh. Một khác biệt: universe tính F1 ở đây là `before ∪ gold_after ∪ pred_after`, nghĩa là mọi cạnh dự đoán sai ngoài tập cố định đều bị tính false positive, còn GUpdate chấm trên tập ứng viên cố định. Cách ở đây chặt hơn, F1 thấp hơn chút không có nghĩa kém hơn.

**Dòng 2 so với baseline IE của paper.** Cùng ý tưởng: chỉ ghi điều text nói, không lan truyền. F1 0,926 so với 0,943; add/del 0,056 so với 0,468/0,245. Baseline của paper còn có R-GCN học được nên lan truyền được một phần; ở đây là xấp xỉ tối giản. Cả hai cùng chứng minh không lan truyền thì gần như không cập nhật được.

**Dòng 3 so với GUpdate.** GUpdate đạt 0,89 trên cạnh thêm/xóa bằng GNN học end-to-end trên 3.262 tin train. Kiến trúc ở đây tách "tìm ở đâu" và "đổi cái gì": phần "tìm ở đâu" đã giải xong bằng khoanh vùng cấu trúc thuần túy, không học, coverage 1,0; phần "đổi cái gì" là bộ phán. Cận trên oracle 1,0 có nghĩa nếu bộ phán đúng thì hệ thống vượt GUpdate. Khoảng cách hiện tại nằm trọn ở bộ phán.

**Bài toán của bộ phán giờ nhỏ đến mức nào.** Sau khi gom theo mẫu, một tin "released" chỉ còn 4 mẫu cắt, một tin trade 1 cầu thủ còn 4 mẫu cắt + 4 mẫu thêm; tin trade lớn nhất 12 + 14. Bộ phán chỉ cần trả lời đúng "đồng đội có bị vô hiệu khi một người rời đội không" và "người mới có thành đồng đội của cả đội không". GUpdate phải học điều này từ 3.262 tin; ở đây kỳ vọng LLM biết sẵn. Model 2B chưa biết ổn định. Với dòng 6, trên tin idx=1 nó trả lời đúng hoàn toàn cả hai câu, nhưng ở các tin khác lại cắt nhầm mẫu "đồng đội cũ vẫn ở đội cũ" và thêm nhầm "đội mới nhận cả đồng đội cũ".

**Bảng so sánh gọn cho khóa luận** (tạm thời):

| | F1 | add | del | Cần học? |
|---|---|---|---|---|
| GUpdate | 0,987 | 0,893 | 0,899 | có, 3.262 tin |
| IE thuần (paper) | 0,943 | 0,468 | 0,245 | có |
| Hệ thống này, ops tường minh không lan truyền | 0,926 | 0,056 | 0,056 | không |
| Hệ thống này, cận trên oracle | 1,000 | 1,000 | 1,000 | không |
| Hệ thống này, LLM 2B tốt nhất (dòng 6, n=20) | 0,923 | 0,287 | 1,000* | không |

---

## 7. Những điểm chưa như kỳ vọng

1. **LLM #1 (Qwen3.5-2B) không trích được op.** Prompt cũ: model trả 1–3 op nhưng đảo vị trí (tên đội vào ô quan hệ, "guard" vào ô đuôi) hoặc bịa entity. Prompt mới có few-shot và enum: model trả rỗng. Đối chứng 4 biến thể (enum/str × prompt mới/cũ) trên 8 tin: prompt là thứ khiến model trả rỗng, enum không gây hại, nhưng không biến thể nào ra op đúng. Giới hạn năng lực model.

2. **Bộ phán 2B phán mẫu chưa ổn định.** Bản 1 suy biến cắt hết. Bản 2 có phán quyết đúng ở một số tin nhưng không nhất quán: 1 pha thì cắt cả mẫu "(X, player, đội cũ)" dù system prompt nói rõ giữ; 2 pha thì tin released chỉ cắt 1 trong 2 hướng teammate, và pha SINH chọn `<player>` thay vì `<teammate>`, thêm cả đồng đội cũ vào đội mới. Tách 2 pha không giúp model 2B, có thể do pha SINH mất ngữ cảnh phần cắt.

3. **Timeout ở tin trade lớn.** idx=15 (16 ops, 26 mẫu) vượt 120 s dù prompt 3.200 token. Cần nâng timeout hoặc chia mẫu thành lô.

4. **Tốc độ.** Khoảng 20 s một lệnh gọi, 2 lệnh gọi một tin trade. Toàn split 438 tin mất khoảng 3 giờ với model 2B; model 7B sẽ chậm hơn 2–3 lần.

5. **Điểm kỹ thuật hoãn lại có chủ đích** (ghi trong ledger): chưa kiểm tra thứ tự thời gian khi `add`/`invalidate`; op của bộ phán chưa mang `quote`/`conf` có nghĩa; chưa có test ghim ràng buộc "áp ops trước rồi mới khoanh vùng"; số lệnh gọi LLM của ví dụ bị lỗi không được đếm; README chưa cập nhật bộ phán bản 2.

---

## 8. Bước tiếp theo đề xuất

1. **Thử model mạnh hơn trên cùng prompt**: Gemma 4 E4B (nếu có GGUF và llama.cpp b10839 hỗ trợ) hoặc Qwen2.5-7B-Instruct Q4_K_M (~4,7 GB). Chạy dòng 6 và 7 với n=50, chọn 1 pha hay 2 pha theo số, rồi chạy toàn split cho bảng 1 chính thức.
2. **Cho pha SINH thấy kết quả pha CẮT** (nối tiếp thay vì độc lập), hoặc quay về 1 pha nếu model lớn làm tốt hơn.
3. **Chia lô mẫu và nâng timeout** cho tin trade nhiều cầu thủ.
4. **Plan tháng 3**: bộ phán B/C (AnyBURL/ULTRA lọc trước, LLM chỉ phán vùng xám), bảng 2 ablation, Neo4j projection.
5. **Merge nhánh** `feat/core-pipeline` vào `main` sau khi có số model 7B.
