# kltn

LLM-driven knowledge-graph update từ tin tức (đồ án tốt nghiệp). Đồ thị tri thức song-thời gian (bi-temporal), cập nhật bằng ops `ADD` / `INVALIDATE` được trích từ văn bản, khoanh vùng ứng viên bằng cấu trúc đồ thị, rồi phán bởi oracle/LLM.

## Cài đặt

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows; source .venv/bin/activate trên Linux/macOS
pip install -e ".[dev]"
python scripts/download_nba.py  # tải NBAtransactions vào data/raw/nba/
python -m pytest
```

## Kết quả sơ bộ (NBAtransactions test)

Chạy pipeline (`kgu/pipeline.py`) trên toàn bộ split test (438 ví dụ) với 3 cấu hình extractor/judge, đo bằng `kgu/eval/metrics.py` (F1, add_acc, del_acc kiểu GUpdate) và `kgu/eval/coverage.py` (độ phủ khoanh vùng cấu trúc trên các op ngầm không nằm trong `explicit_ops`).

| Cấu hình | f1 | add_acc | del_acc | coverage | avg_cut_cand | avg_add_cand |
|---|---|---|---|---|---|---|
| gold_all + none | 1.000 | 1.000 | 1.000 | 0.0 (0/0, không có op ngầm) | 6.5 | 2.5 |
| gold_explicit + none | 0.926 | 0.056 | 0.056 | 1.0 | 70.3 | 31.0 |
| gold_explicit + oracle | 1.000 | 1.000 | 1.000 | 1.0 | 70.3 | 31.0 |

Nhận xét:
- `gold_all + none` = 1.0 chứng minh loader + executor + metric đúng (trích toàn bộ diff before/after, không cần khoanh vùng).
- `gold_explicit + none` cho add_acc/del_acc ≈ 0,056 — trùng với con số "5,6% thay đổi được nói thẳng" của paper gốc, cho thấy cách xấp xỉ ops tường minh (cả 2 đầu cạnh đều được nhắc trong text) hợp lý; F1 0,926 gần baseline IE thuần (0,9429).
- `gold_explicit + oracle` có coverage = 1.0 — khoanh vùng cấu trúc 2 vế (cắt/thêm) bao phủ toàn bộ cạnh ngầm trên NBAtransactions; với oracle phán đúng, pipeline đạt lại F1 = 1.0.
- Mỗi tin có trung bình ~70 ứng viên cắt + ~31 ứng viên thêm — ghi chú cho việc thiết kế prompt bộ phán LLM (tháng 2).

Chạy lại:

```bash
python scripts/run_eval.py --extractor gold_all      --judge none   --out results/nba_test_gold_all.jsonl
python scripts/run_eval.py --extractor gold_explicit --judge none   --out results/nba_test_gold_explicit_nojudge.jsonl
python scripts/run_eval.py --extractor gold_explicit --judge oracle --out results/nba_test_gold_explicit_oracle.jsonl
```

Mỗi lệnh ghi 1 dòng JSON/ví dụ vào `--out` và một bản tổng hợp vào `<out>.summary.json` (in ra console luôn). Thư mục `results/` không được commit (`.gitignore`).

## Cấu hình LLM

`kgu.llm.client_from_env()` đọc 3 biến môi trường `KGU_LLM_BASE_URL` / `KGU_LLM_MODEL` / `KGU_LLM_API_KEY` để dựng một `OpenAICompatClient` (bất kỳ endpoint OpenAI-compatible nào). Có 3 lựa chọn:

```
# A. Local llama.cpp (Vulkan, AMD 780M) — MẶC ĐỊNH.  Khởi động:  .\scripts\llm_server.ps1
set KGU_LLM_BASE_URL=http://localhost:8080/v1   KGU_LLM_MODEL=local   KGU_LLM_API_KEY=none
#    Model khác: .\scripts\llm_server.ps1 -Model models\<file>.gguf  (GGUF Qwen2.5-7B-Instruct Q4_K_M ~4,7 GB cho số cuối)
# B. Colab + vLLM (Qwen2.5-7B-Instruct, GPU T4) qua ngrok/cloudflared:
set KGU_LLM_BASE_URL=https://<tunnel>/v1   KGU_LLM_MODEL=Qwen/Qwen2.5-7B-Instruct  KGU_LLM_API_KEY=none
# C. OpenRouter (hosted):
set KGU_LLM_BASE_URL=https://openrouter.ai/api/v1  KGU_LLM_MODEL=qwen/qwen-2.5-7b-instruct  KGU_LLM_API_KEY=<key>
```

Mặc định của `client_from_env()` là **A** (đã dựng sẵn 2026-09-07: `scripts/llm_server.ps1` + `models/Qwen3.5-2B-Q8_0.gguf` + `tools/llama.cpp/` bản Vulkan chạy trên AMD Radeon 780M; kiểm tra bằng `python scripts/llm_smoke.py`). Vì llama-server hỗ trợ `response_format={"type":"json_schema", ...}` (constrained decoding thật, thay cho outlines/xgrammar trong spec), `OpenAICompatClient.complete_json` gửi `json_schema` trước, và chỉ lùi về `json_object` nếu server báo lỗi. Đã đo trên máy dev (2026-09-07, Qwen3.5-2B Q8_0, Vulkan 780M): sinh ~20 tok/s, đọc prompt ~106 tok/s, JSON schema được tuân thủ 100% (constrained decoding). Ước tính split test 438 tin × 2 lệnh gọi ≈ 1–2 giờ với 2B.
