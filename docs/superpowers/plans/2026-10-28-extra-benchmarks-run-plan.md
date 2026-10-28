# Kế hoạch chạy 28/10: độ ổn định, độ chính xác, trùng lặp + 3 benchmark mới

Yêu cầu (28/10): làm A, B, C; chạy Tem-DocRED, NYT11-HRL, KnowledgeNet với mọi pipeline có code (mình, EDC, ATOM,
KGGen, KARMA) + phương pháp của paper gốc từng bộ nếu chạy được; nhiều kernel song song, chạy đủ, nhanh nhất có thể.
Giới hạn: Kaggle 2 kernel GPU đồng thời / tài khoản, ~30 giờ GPU / tuần → 2 làn GPU liên tục.

## Hạng mục
- **B. Độ ổn định (COVID-NYT)**: thêm lượt r2, r3 cho mình v2, ATOM, ATOM trích thẳng, KGGen (r1 = results/kaggle/covid_v2/v2.jsonl,
  results/kaggle/home_bench/covid/{atom,atom-direct,kggen}.jsonl). Chỉ số của paper ATOM (arXiv 2510.22590v2, Bảng 1):
  Jaccard tập 5-tuple giữa lượt 1 và lượt r (r = 2, 3), trung bình ± std theo nhóm; paper GPT-4.1: ATOM 0,552/0,534,
  trích thẳng 0,212/0,218. Thêm: Jaccard mềm (ghép embedding @0,7) và F1 từng lượt → mean ± std.
- **A. Độ chính xác**: Gemma kiểm từng triple có căn cứ trong văn bản nguồn (1 lệnh/văn bản, trả có/không từng triple);
  COVID-NYT (mọi hệ + output GPT-4.1 phát hành), MINE-1 (mình, KGGen-Gemma, 3 KG phát hành), 3 bộ mới. Claude kiểm tay
  mẫu 100 phán quyết để đo độ tin của judge.
- **C. Trùng lặp** (CPU, local): cặp tên gần trùng cos > 0,8 (metric gộp ATOM), triple trùng ý; mọi hệ, mọi bộ.
- **Tem-DocRED** (Sci Data 2025; Zenodo 10.5281/zenodo.14322300; test 348 văn bản / 2.483 bộ bốn, 33 quan hệ, 1 mốc
  thời gian): 5 hệ; gốc = LoRA Llama-3-8B (yaml của GTKG, QLoRA fp16 trên T4); chấm bộ bốn (h, r, t, time) khớp tên
  bất kỳ mention của thực thể gold + bản bỏ thời gian; số paper: Llama-3-8B 28,37 F1 (fine-tune), không fine-tune ~0.
- **NYT11-HRL** (test 369 câu / 370 triple, 12 quan hệ Freebase): 5 hệ; gốc HRL (Takanobu 2019) không có model sẵn →
  số công bố. Dữ liệu: Google Drive của HRL-RE (RAR5) → dataset Kaggle riêng tư (NYT thuộc LDC, không đưa lên repo).
- **KnowledgeNet** (EMNLP 2019; test giấu → fold 4: 1.022 văn bản / 2.708 fact, 15 thuộc tính): 5 hệ, chấm evaluator.py
  span_overlap (Text F1; Link F1 cần URI Wikidata); gốc = baseline 5 có model sẵn (README: dev span F1 0,704).

## Làn
| Làn | Việc |
|---|---|
| GPU 1 | `kgu-covid-repeat` (r2 + r3) → KnowledgeNet 5 hệ |
| GPU 2 | A trên output cũ → smoke 3 bộ mới → NYT11 → Tem-DocRED → LoRA Llama-3-8B |
| CPU | C (local); KnowledgeNet baseline 5 (kernel CPU) |

## Nhật ký
- 28/10 20:40: B chạy (kgu-covid-repeat-r2, -r3; commit 1ca5db9). Giờ thật: B xong ~23:20.
- 28/10 21:00: user chốt ABCD + F, bỏ KARMA, bỏ E (Tem-DocRED), không chạy lại EMERGE, chỉ dùng GPU Kaggle (không chạy
  tính toán trên máy). Code: D/A/C 244b43d (kgu-extra-nyt11-a/b, kgu-quality), F a38e47c (kgu-extra-knet-a/b; giới hạn
  trên gold-as-prediction 93,3 span F1), baseline 5 KnowledgeNet: kaggle/knet_baseline (kgu-knet-baseline).
- Thứ tự khi B xong: làn 1 smoke NYT11 (EXTRA_LIMIT=5, cả 4 hệ) → nyt11-a → knet-a → knet-baseline; làn 2 quality →
  nyt11-b → knet-b. Sau cùng: gộp bảng P/R/F1 mọi bộ (MINE: P = tỉ lệ được hỗ trợ của judge), notes, Notion.
- 28/10 21:30: user muốn chạy hết qua đêm kể cả khi phiên Claude dừng → thay chuỗi push từng kernel bằng 2 kernel làn
  (kaggle/night, run_lane.py: nhiều việc trong một kernel, mỗi việc clone code mới nhất của nhánh, giờ chót chung 11 h,
  mỗi hệ tự co thời gian để còn 1 h cho judge + chấm). Bỏ smoke (không ai xem giữa đêm; mỗi hệ/bước chấm đã bọc lỗi riêng).
  | Kernel | Việc | Ước tính |
  |---|---|---|
  | kgu-night-1 | NYT11 (mình, EDC, ATOM+EDC, KGGen+EDC) → KnowledgeNet ATOM+EDC | ~2,5 h + ~3,2 h |
  | kgu-night-2 | quality (A + C trên COVID-NYT / MINE-1, ~1.900 lượt judge) → KnowledgeNet mình, EDC, KGGen+EDC | ~1,5–2 h + ~4,5 h |
  | kgu-knet-baseline (CPU) | baseline 5 của paper KnowledgeNet, chạy lại | env ~1 h + evaluate ≤ 9 h |
  Kaggle không xếp hàng kernel GPU thứ 3 ("Maximum batch GPU session count of 2 reached") → kaggle/night/watch.ps1 chạy
  tách khỏi phiên (WMI), giữ máy thức, thử push mỗi 2 phút đến khi B nhả GPU (~23:20), rồi đợi mọi kernel xong, tải output
  về results/kaggle/{covid_repeat/r2,r3, night/lane1,lane2, knet_baseline} và dựng results/kaggle/night/REPORT.md
  (scripts/night_report.py). Dự kiến xong ~05:30. Máy ngủ / tắt giữa chừng → sáng chạy `bash kaggle/night/watch.sh fetch`.
  Baseline KnowledgeNet v1 lỗi (uv không có Python 3.7), v2 lỗi (gói phát hành lồng thư mục) → v3: micromamba + tự tìm
  models/, tmp/. NYT11 (NYT/LDC): .gitignore chặn mọi file có câu gốc trong results/kaggle/night/lane1/out/nyt11/.
