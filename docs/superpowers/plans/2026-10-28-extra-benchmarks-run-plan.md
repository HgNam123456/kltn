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
- 28/10 ~21:30: kế hoạch, bắt đầu B.
