# Ghi chú dữ liệu EMERGE (spike 2026-09-21)

Nguồn: HuggingFace `klimzaporojets/emerge-benchmark` (arXiv 2507.03617), bản 2026-08-03 có thêm baseline KG-aware.
Số liệu dưới đây đo trực tiếp trên 3.500 instance của `evaluation_set`, không lấy từ paper.

## 1. Cách dựng lại

```
python scripts/download_emerge.py            # 35 file JSONL, 167 MB -> data/raw/emerge/evaluation_set/
python scripts/build_emerge_subgraphs.py     # 7 file KG-trước thu gọn, 133 MB -> data/raw/emerge/kg_subsets/
python scripts/inspect_emerge.py             # bảng ở mục 4
```

Loader: `kgu/data/emerge.py` (`load_snapshot`, `load_kg_subset`, `to_example` đưa về khuôn before/after của pipeline NBA).

## 2. Instance KHÔNG kèm KG-trước

Mỗi instance chỉ có passage, mentions (QID + offset), ops vàng và prediction của 16 model. KG-trước phải lấy từ
`kg_snapshots/` (7 file TSV nén, 411–617 MB mỗi file, 26–35 triệu cạnh, 6 cột: h, r, t + 3 nhãn).

`build_emerge_subgraphs.py` stream thẳng từ HF, không lưu file gốc, chỉ giữ cạnh 1-hop quanh entity được nhắc trong
500 passage của snapshot đó: cạnh đi ra giữ hết, cạnh đi vào giữ tối đa 500 mỗi entity (United States có 945 nghìn
cạnh vào). Mỗi snapshot mất khoảng 35 giây, còn 255–327 nghìn cạnh (~19 MB). Không cần 22 GB hay 180 GB RAM.

Mỗi instance: trung vị ~700 cạnh KG-trước, ~650 entity lân cận (NBA: subgraph vài trăm cạnh, 4 quan hệ; ở đây 448 quan hệ).

## 3. Gold phải lọc theo đánh giá của LLM assessor

Mỗi triple vàng mang `llm_assessment` của nhiều LLM. Bộ chấm chính chủ chỉ tính triple được
`Meta-Llama-3.1-405B_prompt_v1` xác nhận (prompt `triple_assertion`, riêng Deprecate là `triple_deprecation`).
Loader tách sẵn `gold` (đã xác nhận) và `unverified`.

| Op | tất cả | đã xác nhận |
|---|---|---|
| Exists | 17.326 | 12.783 |
| Add | 2.735 | 2.277 |
| Mint+Add | 2.660 | 2.264 |
| Infer | 11.613 | **4.170** (36%) |
| Deprecate | 1.911 | 1.628 |

## 4. Kiểm chứng gold đã xác nhận với KG-trước (cả 7 snapshot)

| Op | n | có trong KG-trước | cả 2 đầu được nhắc | đầu xa nằm trong 1-hop |
|---|---|---|---|---|
| Exists | 12.783 | 0,993 | 1,000 | – |
| Add | 2.277 | 0,016 | 1,000 | – |
| Mint+Add | 2.264 | 0,000 | 1,000 | – |
| Infer | 4.170 | 0,000 | 0,000 | **0,398** |
| Deprecate | 1.628 | 0,912 | 0,981 | – |

- Dữ liệu nhất quán với snapshot: Exists có sẵn, Add chưa có, Deprecate có sẵn.
- ~9% Deprecate không có trong KG-trước đều là cặp `d-triples + e-triples`: cạnh được thêm SAU snapshot rồi gắn
  `end time` ngay (vd Governor of Wisconsin – officeholder – Scott Walker, thêm 06/01/2019, end time 07/01/2019).
- 73% Deprecate có qualifier `end time`; quan hệ chủ yếu: position held, head coach, officeholder, spouse,
  member of sports team.

## 5. Hệ quả cho thiết kế — khác NBA ở hai điểm gốc

**Deprecate trên EMERGE là tường minh, không phải hệ quả lan truyền.** 98% cạnh bị vô hiệu có cả hai đầu xuất hiện
trong passage. Ở NBA 94% cạnh xóa là hệ quả ngầm (teammate) nên vế CẮT của khoanh vùng là đòn chính; ở đây việc cần
làm là *tra KG-trước xem giữa các entity được nhắc đang có cạnh nào, rồi để LLM phán cạnh nào text nói đã hết hiệu lực*.
Lợi thế so với EDC+/KGGen/RAKG là được nhìn KG, không phải tam giác hàng xóm chung.

**Infer trên EMERGE phần lớn không suy ra được từ cấu trúc.** Infer = thuộc tính Wikidata của entity MỚI mà passage
không nói thẳng (instance of human, family name, occupation...). Chỉ 39,8% đầu xa nằm trong vùng 1-hop của các entity
được nhắc. Nhóm làm được bằng cấu trúc: country, country of citizenship, country of origin (0,72–0,86 trên snapshot
2019). Nhóm cần tri thức nền của LLM: instance of (human, album, film).

## 6. Bộ chấm Executable-R tự viết, đã đối chiếu với số công bố

`kgu/eval/emerge.py::executable_scores` = metric `cie_exact_match` của repo EMERGE: khớp chính xác bộ ba QID, tính
theo instance, trung bình trên các instance có gold đã xác nhận cho op đó, instance không dự đoán tính 0. Chạy CPU,
vài giây. Completeness và G-BERTScore-R cần sentence-transformers + GPU, chưa chạy.

`python scripts/emerge_baselines.py`, toàn bộ 3.500 instance (recall / precision):

| Model | Exists | Add | Deprecate | Số công bố (E-R) |
|---|---|---|---|---|
| KG-aware GPT-5.1 oracle | 0,503 / 0,614 | 0,396 / 0,370 | 0,358 / 0,361 | 50,3 / 39,6 / 36,9 |
| KG-aware GPT-5.1 kg_rag@32 | 0,194 / 0,201 | 0,125 / 0,071 | 0,112 / 0,088 | 19,4 / 12,5 / 11,2 |
| KG-aware GPT-5.1 oracle+kg_rag | 0,432 / 0,415 | 0,182 / 0,093 | 0,173 / 0,141 | 43,2 / 18,2 / 17,3 |
| **Tra KG, không LLM** (mọi ứng viên = Exists) | **0,994** / 0,679 | – | – | |
| Trần của bộ phán trên ứng viên | 0,994 / 0,998 | – | 0,901 / 0,908 | |

Khớp 8/9 ô với README của repo (Deprecate oracle lệch 1,1 điểm, chưa rõ nguyên nhân).

Ứng viên = cạnh KG-trước mà cả hai đầu được nhắc trong passage: trung vị 2, trung bình 5,8, max 229 mỗi instance.
Chỉ riêng bước tra này đã gần gấp đôi recall Exists của GPT-5.1 oracle, và giữ được 90% đáp án Deprecate trong một
danh sách rất ngắn.

Lưu ý công bằng: mình dùng trường `mentions` (QID từ hyperlink Wikipedia, là một phần của instance) làm entity
linking. Biến thể `oracle` còn được cho thẳng triple vàng; `kg_rag@32` thì tự truy hồi, không dùng `mentions`.
Setting của mình nằm giữa hai biến thể đó.

## 7. Bộ phán v0 trên tập dev 350 (Gemma 4 E4B local)

`python scripts/run_emerge.py --per-delta 10 --workers 4 --out results/emerge_dev350_gemma4.jsonl`

Tập dev = 10 instance đầu mỗi file delta. Một lệnh gọi/instance: passage + danh sách cạnh đang có trong KG, model xếp
từng cạnh vào `supported` / `deprecated` / bỏ qua (`kgu/judge/emerge.py`). 242/350 instance có ứng viên, 242 lệnh gọi,
703 giây, 0 lỗi JSON. Ô ghi recall / precision Executable-R trên cùng 350 instance.

| Hệ | Exists | Deprecate |
|---|---|---|
| KG-aware GPT-5.1 oracle | 0,532 / 0,651 | 0,352 / 0,352 |
| KG-aware GPT-5.1 kg_rag@32 | 0,188 / 0,202 | 0,088 / 0,065 |
| Tra KG, không LLM | 0,983 / 0,669 | – |
| Bộ phán v0 | 0,536 / 0,746 | 0,375 / 0,371 |
| Lai: Deprecate do LLM, ứng viên còn lại = Exists | 0,883 / 0,683 | 0,375 / 0,371 |

Lỗi Deprecate (145 cạnh vàng): 57 đúng, 48 bị xếp vào `supported`, 29 bị bỏ qua, 11 không nằm trong ứng viên.
105 cạnh phán Deprecate sai, trong đó 89 là gold Exists. 181 cạnh Exists vàng rơi ngoài ứng viên do cắt ở 40 cạnh/instance.
Chỉ 125 instance có gold Deprecate nên chênh 0,375 / 0,352 chưa có ý nghĩa thống kê.

## 8. Bộ phán v1 (2026-09-21): hai bước + mốc thời gian + few-shot

Chẩn đoán từ lỗi v0: gold Deprecate của EMERGE chỉ tính việc VỪA kết thúc quanh ngày chụp KG ("governor from 2011 to
2019", snapshot 01/2019). Việc kết thúc từ lâu ("previously played for IK Sirius") không phải Deprecate. v0 không biết
mốc thời gian nên vừa bỏ sót vừa phán thừa.

Thay đổi trong `kgu/judge/emerge.py`:
- Mỗi cạnh trả lời hai câu theo thứ tự: `discussed` (passage có nói tới không) → `ended` (có nói vừa kết thúc không).
- Prompt có `GRAPH DATE` (ngày snapshot) và `TODAY` (ngày delta); "vừa kết thúc" = từ khoảng một năm trước GRAPH DATE tới TODAY.
- Few-shot cho position held / officeholder (cả hai chiều) / head of government / head coach / spouse.
- Bỏ giới hạn 40 ứng viên, chia lô 12 cạnh mỗi lệnh gọi. `--batch-size 1` = mỗi cạnh một lệnh gọi có/không.
- Exists = mọi ứng viên không bị phán "ended" (`--exists all`).
- KHÔNG dùng được qualifier `end time`/`start time`: file KG snapshot chỉ có 6 cột (h, r, t + nhãn), qualifier chỉ
  nằm trên triple vàng, dùng là lộ đáp án.

`scripts/run_emerge.py` giờ in CI95 bootstrap (2.000 lần lấy mẫu lại theo instance) và hiệu recall ghép cặp so với
GPT-5.1 oracle. Dev 350, Gemma 4 E4B local:

| Hệ | Exists recall | Exists precision | Deprecate recall | Deprecate precision | lệnh gọi | s/instance |
|---|---|---|---|---|---|---|
| GPT-5.1 oracle | 0,532 [0,470; 0,596] | 0,651 | 0,352 [0,272; 0,440] | 0,352 | API | – |
| GPT-5.1 kg_rag@32 | 0,188 | 0,202 | 0,088 [0,040; 0,136] | 0,065 | API | – |
| Tra KG, không LLM (không cap) | 0,995 [0,991; 0,999] | 0,668 | – | – | 0 | ~0 |
| v0 (cap 40, một danh sách) | 0,536 | 0,746 | 0,375 | 0,371 | 242 | 2,0 |
| **v1 gom lô 12** | 0,877 [0,840; 0,910] | 0,693 | **0,652 [0,568; 0,736]** | 0,445 [0,376; 0,519] | 335 | 4,8 |

Hiệu recall ghép cặp v1 − GPT-5.1 oracle: Exists +0,345 [0,272; 0,415], Deprecate +0,300 [0,208; 0,388]. CI không chứa 0.

Lỗi còn lại của v1 (145 cạnh Deprecate vàng): 99 đúng, 22 phán "holds", 16 phán "unrelated", 8 ngoài ứng viên.
143 cạnh phán "ended" thừa (103 là gold Exists), 71 trong đó thuộc position held (P39) / officeholder (P1308).

### Phán từng cạnh kém hơn gom lô

`--batch-size 1` (2.183 lệnh gọi, 8,0 s/instance): Deprecate 0,524 [0,436; 0,612] / precision 0,377; Exists 0,887.
Hiệu recall Deprecate ghép cặp gom lô − từng cạnh: +0,128 [0,068; 0,196]. Gom lô vừa chuẩn hơn vừa nhanh hơn →
giữ gom lô 12. Giả thuyết (chưa kiểm chứng): nhìn các cạnh cạnh nhau giúp model so sánh hai chiều của cùng một chức
vụ, chức vụ cũ với chức vụ vừa hết.

### Xác nhận trên tập chưa nhìn

Prompt v1 được viết sau khi đọc các ca sai trên dev 350, nên chạy lại trên 350 instance khác (bài 11–20 của mỗi
delta, `--offset 10`), không sửa gì:

| Tập | Exists recall | Exists precision | Deprecate recall | Deprecate precision | s/instance |
|---|---|---|---|---|---|
| dev 350 (đã nhìn) | 0,877 | 0,693 | 0,652 [0,568; 0,736] | 0,445 | 4,8 |
| held-out 350 | 0,879 [0,843; 0,911] | 0,729 | 0,617 [0,531; 0,698] | 0,430 [0,360; 0,499] | 4,3 |
| GPT-5.1 oracle trên held-out | 0,524 | 0,639 | 0,346 [0,272; 0,427] | 0,348 | – |

Hiệu recall ghép cặp trên held-out, v1 − GPT-5.1 oracle: Exists +0,355 [0,286; 0,426], Deprecate +0,272 [0,180; 0,364].
Điểm gần như không tụt → không overfit vào tập dev.

## 9. Ngày 22/09: thử nâng precision Deprecate (v2, v3) và Add v0

Dev 350, Gemma 4 E4B, gom lô 12. Ô ghi recall / precision Executable-R.

| Bản | Thay đổi so với v1 | Exists | Deprecate | TP/FP cạnh Deprecate |
|---|---|---|---|---|
| v1 | – | 0,877 / 0,693 | 0,652 / 0,445 | 99 / 143 |
| v2 | luật cứng: ended CHỈ KHI passage ghi năm {Y-1}/{Y} | 0,918 / 0,694 | 0,560 / 0,401 | 86 / 97 |
| v3 | gợi ý mềm: "kết thúc {Y-2} trở về trước = đã lâu" | 0,891 / 0,699 | 0,644 / 0,451 | – |

- v2 bớt 46 phán thừa nhưng mất 13 ca đúng; vì metric tính theo bài (bài trống = precision 0) nên cả hai chỉ số tụt.
  Model 4B đọc luật "CHỈ KHI" thành "khi nghi ngờ thì đừng đánh dấu": "from 2011 to 2019" với GRAPH DATE 2019 vẫn trả holds.
- v3 không khác v1 trong khoảng nhiễu. Giữ v3 trong code (Exists nhỉnh hơn).
- **Trần dữ liệu của Deprecate trên dev 350:** 15/145 gold không xuất hiện trong passage (diff Wikidata: place of birth,
  screenwriter); 26/91 phán thừa chung của v1+v2 là chiều ngược của một gold Deprecate và gold chọn chiều không theo
  quy luật (P39→P1308 62 lần, ngược lại 35 lần). Bỏ hai nhóm này thì v1 tương đương recall ~0,75 / precision ~0,55.
- Lỗi thật còn lại: 64 cạnh gold Exists bị phán ended vì passage dùng thì quá khứ cho việc kết thúc từ lâu
  ("previously played for", "served as Taoiseach 2011–2017" với KG 2020). v2/v3 chưa sửa được nhóm này.

### Add v0 (`kgu/judge/emerge_add.py`, `--judge add`)

Ứng viên: cặp entity được nhắc (phủ 100% gold Add); quan hệ cho phép = quan hệ trên lân cận 1-hop của entity được nhắc
(phủ 83%, ~40 quan hệ/bài; tập theo kiểu entity chỉ phủ 58% nên không dùng). Một lệnh gọi/bài, model trả (số entity,
nhãn quan hệ, số entity), ánh xạ nhãn → PID.

| Hệ | Add recall | Add precision |
|---|---|---|
| GPT-5.1 oracle | 0,475 [0,388; 0,568] | 0,423 |
| GPT-5.1 kg_rag@32 | 0,076 | 0,059 |
| Add v0 | 0,221 [0,159; 0,290] | 0,125 |

Lỗi: sinh 1.576 dự đoán cho 273 gold (gấp 6). 95/230 ca sót là đúng cặp nhưng sai quan hệ (instance of thay vì sport,
occupation thay vì position held) hoặc ngược chiều (position held ↔ position holder). Một số entity thiếu nhãn KG.
Add v1: giới hạn 5 fact/bài, gợi ý chiều/từ vựng Wikidata, nhãn fallback từ chữ trong passage — đang chạy.

### Phạm vi chốt (22/09)
Chỉ xét tri thức được passage đề cập. **Infer bỏ hẳn** (định nghĩa là "không nói trong passage", 60% không có trong
KG lân cận). Deprecate ngoài passage (~10%) ghi là trần dữ liệu.

## 10. Ngày 23/09: số CHÍNH THỨC từ bộ chấm EMERGE (Kaggle GPU)

Pipeline: `scripts/export_emerge_predictions.py` xuất prediction v3 (Exists/Deprecate) theo format `kg-prompt`
→ dataset Kaggle `ngocnam2005/emerge-dev350-mine` → kernel `emerge-eval-dev350` (`kaggle/emerge_eval/run_eval.py`)
clone repo EMERGE, chạy `evaluation.s0x_evaluate_predictions` với config "fixed", không nạp KG snapshot, bỏ
entity_coverage. Bảng đầy đủ: `results/kaggle/dev350_v3_tables.txt`. Kaggle mount dataset ở
`/kaggle/input/datasets/<user>/<slug>/` (không phải `/kaggle/input/<slug>/`).

Dev 350, 11 baseline kèm sẵn cùng tập. C = Completeness, R = G-BERTScore-R (bảng chính của paper).

| Model | Exists C / R | Add C / R | Deprecate C / R |
|---|---|---|---|
| **mine/gemma4-e4b (v3, chưa Add)** | **92,8 / 91,3** | 0 / 0 | **69,0 / 70,6** |
| KG-aware GPT-5.1 oracle | 65,7 / 53,2 | 58,6 / 50,9 | 40,0 / 35,2 |
| KG-aware GPT-5.1 kg_rag@32 | 15,3 / 29,2 | 10,6 / 17,5 | 11,0 / 10,3 |
| EDC+ GPT-5.1 | 22,5 / 65,4 | 37,7 / 76,4 | 35,9 / 56,0 |
| EDC+ Mistral-Large | 17,3 / 53,9 | 26,7 / 69,9 | 31,0 / 52,2 |
| EDC+ Mistral-Small | 12,8 / 45,2 | 19,4 / 62,4 | 17,2 / 28,7 |
| RAKG Mistral-Small | 10,8 / 63,9 | 17,7 / 70,7 | 0 / 0 |
| ReLiK RE | 19,5 / 57,5 | 13,6 / 61,6 | 0 / 0 |
| REBEL | 10,3 / 41,1 | 7,7 / 51,1 | 0 / 0 |

QID exact-match chính thức (P / R / F1): mine Exists 64,7 / 82,6 / 69,5, Deprecate 44,1 / 62,9 / 49,2;
oracle Exists 60,3 / 49,3 / 52,5, Deprecate 34,4 / 34,4 / 34,4.

- Bộ chấm tự viết lệch nhẹ so với chính chủ: Exists recall 0,891 vs 82,6; Deprecate 0,644 vs 62,9; oracle Deprecate
  0,352 vs 34,4. Chưa rõ nguyên nhân (nghi khác cách gộp triple trùng giữa x/d hoặc lọc gold). Từ giờ báo số chính thức.
- Completeness của mình cao hơn G-BERTScore-R ở Exists nhưng ngược lại với EDC+: đúng như README nói, G-R thưởng
  giống nghĩa, C thưởng đúng dạng KG.

### Add v1 (23/09)
recall 0,213 / precision 0,122 — bằng v0. Tuân thủ "tối đa 5" (4,8 fact/bài) nhưng vẫn gấp 6 lần gold (1.668 / 273);
103/231 ca sót là sai quan hệ (55) hoặc ngược chiều (48). Prompt không sửa được → Add v2 phải chuẩn hóa bằng cấu trúc:
đảo chiều khi (t, r, h) hợp kiểu entity hơn (h, r, t) theo thống kê KG; ánh xạ nhãn quan hệ gần nghĩa về quan hệ mà
cặp kiểu (type(h), type(t)) thực sự mang trong KG. So công bằng: kg_rag@32 (không được cho relation type) 0,076 / 0,059.

## 11. Bộ chấm mềm viết lại (23/09): C và G-BERTScore-R chạy được ngoài repo EMERGE

Đọc `src/evaluation/scorers/completeness_scorer.py` và `scorers/misc/graph_matching.py` của EMERGE, viết lại thành
`kgu/eval/emerge_soft.py` + `scripts/score_emerge_soft.py`. Mục đích: mọi bản thử có ngay số **cùng thang với bảng
paper** (thay vì exact-match QID tự viết, vốn phạt nặng "đúng cặp, sai tên quan hệ / ngược chiều" mà bộ chấm chính chủ
cho gần đúng). Chạy trên Kaggle: kernel `kgu-score-emerge` (`kaggle/score/`, chấm file jsonl đã có qua dataset
`ngocnam2005/kgu-results`) và bước 6 của kernel `kgu-judge-emerge` (chấm ngay sau khi phán).

Cách chính chủ tính (đã đối chiếu code, không phải đoán từ paper):
- Chuỗi so khớp của một triple = `"head relation tail"` từ **nhãn**, chữ thường, `_` → space (`triple_labels`; QID không
  dùng). Gold chỉ lấy triple được assessor xác nhận; bài không có gold cho op đó bị bỏ qua; bài có gold mà không có dự
  đoán tính 0 (`score_empty_predictions_as_zero`).
- **Completeness (C)**: `all-mpnet-base-v2`, mỗi gold lấy cosine lớn nhất với mọi dự đoán cùng op, "phủ" nếu > 0,9
  (`completeness_threshold`). C = tỉ lệ gold được phủ, **gộp theo triple** trên toàn tập. **Không có precision** →
  sinh thừa không bị trừ; đây là lý do Add v1 được 32,6 dù exact-match recall chỉ 0,19.
- **G-BERTScore-R (G-R)**: BERTScore-F1 (`bert-base-uncased`, `idf=False`, không rescale) cho mọi cặp gold × dự đoán
  trong một bài, ghép 1-1 bằng Hungarian (`linear_sum_assignment`, maximize), R = tổng điểm cặp đã ghép / số gold,
  **trung bình theo bài**. G-P = tổng / số dự đoán (paper không báo).
- Hệ quả khi đọc bảng: C thưởng đúng dạng KG (nhãn quan hệ Wikidata), G-R thưởng giống nghĩa; cả hai đều là recall.

Kiểm chứng: chấm lại v3 + Add v1 và ba baseline kg-aware bằng bản viết lại, so với bảng chính thức mục 10.

Kết quả kiểm chứng (kernel `kgu-score-emerge` v1, 115 giây kể cả cài thư viện; `results/kaggle/score_v1_kernel.log`),
bản viết lại / chính thức, C / G-R:

| Hệ | Exists | Add | Deprecate |
|---|---|---|---|
| mine v3 + Add v1 | 92,8 / 91,3 · 92,8 / 91,3 | 32,6 / 74,0 · 32,6 / 74,1 | 69,0 / 70,6 · 69,0 / 70,6 |
| oracle | 65,7 / 53,2 · 65,7 / 53,2 | 59,0 / 50,9 · 58,6 / 50,9 | 40,0 / 35,2 · 40,0 / 35,2 |
| kg_rag@32 | 15,3 / 29,2 · 15,3 / 29,2 | 10,6 / 17,4 · 10,6 / 17,5 | 11,0 / 10,3 · 11,0 / 10,3 |

Lệch tối đa 0,4 điểm (Add oracle; nghi do chính chủ gộp triple trùng khác một chút), còn lại khớp đến 0,1. Kết luận:
dùng bộ chấm mềm cho mọi lần so bản nội bộ; kernel `emerge-eval-dev350` chỉ để chốt số đưa vào khóa luận.
Bộ chấm mềm cũng cho Mint+Add (oracle 38,5 / 43,5; mình 0 vì chưa làm).

## 12. Add v2 (23/09): chuẩn hóa cấu trúc, không thêm lệnh gọi LLM

Soi 231 ca sót của Add v1: 128 model không sinh gì cho cặp đó (sót thật: "killed in Baghdad" → place of death,
ngôn ngữ nói…), 103 đúng cặp nhưng ngược chiều hoặc sai quan hệ. Ý định ban đầu là dùng kiểu entity (P31) để sửa,
nhưng KG-trước thu gọn chỉ có P31 cho 65% entity được nhắc (nhiều entity dùng P279, snapshot EMERGE cũng đã lược) →
thống kê theo kiểu quá thưa (10k/270k cạnh có kiểu ở cả hai đầu). Thay bằng ba luật chỉ dựa trên cạnh 1-hop sẵn có
(`KGShape` trong `kgu/judge/emerge_add.py`, bật mặc định trong `run_emerge.py --judge add`, tắt bằng `--no-add-norm`;
áp lên jsonl cũ bằng `scripts/normalize_emerge_add.py`):

1. **Đảo chiều theo cạnh của chính entity**: (h, r, t) → (t, r, h) nếu t đã phát ra r / h đã nhận r nhiều hơn chiều
   xuôi (phim phát ra `cast member`, diễn viên không).
2. **Sửa quan hệ**: nếu r chưa từng xuất hiện ở cả hai đầu mà có đúng một quan hệ r' h đã phát ra và t đã nhận, đổi
   sang r'.
3. **Thêm chiều nghịch đảo**: gold Wikidata lưu cả hai chiều (`has part(s)`/`part of`, `position held`/`position
   holder`, `father`/`child`…). Rút 53 cặp nghịch đảo từ chính KG-trước (cặp (h,r,t) & (t,r',h) cùng có ≥ 30 lần và
   ≥ 30% số cạnh của r) và tự bổ sung chiều còn lại. Không lộ gold.

Dev 350, exact-match (bộ chấm tự viết, thuần CPU):

| Bản | TP / dự đoán | micro R / P | R / P theo bài |
|---|---|---|---|
| Add v1 | 42 / 1.668 | 0,154 / 0,025 | 0,213 / 0,122 |
| + đảo chiều | 51 / 1.667 | 0,187 / 0,031 | 0,231 / 0,127 |
| + sửa quan hệ | 60 / 1.663 | 0,220 / 0,036 | 0,238 / 0,132 |
| **+ nghịch đảo = Add v2** | **65 / 1.798** | **0,238 / 0,036** | **0,274 / 0,139** |

TP +55% không tốn lệnh gọi nào; precision không giảm. Điểm mềm (C, G-R) của Add v2 chấm trên Kaggle: xem bên dưới.
Lưu ý cho khóa luận: cả C lẫn G-R đều là recall (không phạt thừa) → phải báo thêm G-P / exact P để trung thực.

Điểm mềm (kernel `kgu-score-emerge` v2, `results/kaggle/emerge_dev350_v3_addv{1,2}.soft.json`), dev 350, Add:

| Bản | C | G-R | G-P |
|---|---|---|---|
| Add v1 | 32,6 | 74,0 | 38,2 |
| **Add v2** | **36,6** | **75,9** | 37,5 |
| GPT-5.1 oracle (được cho relation type) | 59,0 | 50,9 | 45,1 |
| EDC+ GPT-5.1 (chính thức) | 37,7 | 76,4 | – |
| EDC+ Mistral-Small (chính thức) | 19,4 | 62,4 | – |

Add v2 chạm EDC+ GPT-5.1 về C và G-R với model 4B; G-P gần như không đổi (thêm chiều nghịch đảo không làm loãng).
Tham chiếu G-P của mình ở op khác: Exists 71,7 (oracle 65,1), Deprecate 49,1 (oracle 35,2).

## 13. Hạ tầng Kaggle chạy trọn vẹn (23/09, kernel `kgu-judge-emerge` v3)

Lỗi hai lượt đầu: CMake không thấy `CUDA::cuda_driver` — ảnh Kaggle không có `/usr/local/cuda/lib64/stubs/`, driver
thật ở `/usr/local/nvidia/lib64/libcuda.so` → dò file rồi truyền `-DCUDA_cuda_driver_LIBRARY`. Build sm75 ~20 phút.
Server lên với đủ cờ như máy local (4 slot, MTP draft, flash-attn). Binary đã lưu thành dataset
`ngocnam2005/llama-server-cuda` (`kaggle/dataset_llama/`), kernel tự nhận → các lần sau bỏ bước build.

Đối chiếu Kaggle T4 với máy local (dev 350, chấm mềm; `results/kaggle/emerge_dev350_v3_addv2_kaggle.soft.json`):

| | Exists C / G-R | Add v2 C / G-R | Deprecate C / G-R | s/bài (Deprecate · Add) |
|---|---|---|---|---|
| Kaggle T4 | 92,0 / 90,7 | 37,0 / 75,0 | 69,7 / 70,8 | **1,07 · 1,35** |
| Máy local | 92,8 / 91,3 | 36,6 / 75,9 | 69,0 / 70,6 | 4,8 · 6,2 |

Lệch ≤ 0,9 điểm (nhiễu speculative/batch), nhanh gấp 4–5 lần → toàn bộ 3.500 bài ước ~70 phút/lượt trên Kaggle.

## 14. Baseline chạy lại bằng Gemma 4 E4B (12/10): so cùng model

Mục đích: tách phần hơn nhờ pipeline khỏi phần hơn nhờ model. Kernel `kgu-baselines-gemma4` v2 (`kaggle/baselines/`)
chạy code EMERGE ghim commit `f688910` với llama-server Gemma 4 E4B (cùng GGUF/cờ với kernel judge), dev 350:
- **EDC+ ICL / ZS**: config `edc_plus_{icl,zs}_gpt_5_1` giữ nguyên prompt + generation profile, chỉ đổi khối `llm`
  (`backend http`, `mode chat`, `model local`). Không canonicalize (giống bản GPT-5.1) → không cần e5-mistral-7b,
  chỉ tải `documents.jsonl` của 7 snapshot.
- **KGGen**: `kg-gen==0.4.0`, `model=hosted_vllm/local`, temp 0 / chunk 10240 như config Mistral (bản GPT-5.1 bị
  kg-gen ép temp 1.0), `max_tokens` 4096 cho vừa 8192 ngữ cảnh/slot. Wrapper ghi khóa `kg-gen-<model>` → đổi tên
  chuẩn trước khi merge.
- Gộp bằng `merge.s0x_merge_predictions` của EMERGE; 350/350 bài có prediction, không bài rỗng; prediction cũ giữ
  nguyên (đã so từng bài). Dataset `emerge-dev350-mine` đã thêm 3 model; chấm bằng kernel `emerge-eval-dev350` v5.

Số chính thức (C / G-BERTScore-R), dev 350, bảng đầy đủ `results/kaggle/dev350_baselines_gemma4_tables.txt`.
s/bài = wall / 350 trên T4, 4 slot song song (`results/kaggle/baselines_gemma4.summary.json`).

| Model | Exists | Add | Mint+Add | Deprecate | s/bài |
|---|---|---|---|---|---|
| **mine/gemma4-e4b** (v3 + Add v1) | **92,8 / 91,3** | 32,6 / 74,1 | – | **69,0 / 70,6** | 1,07 + 1,35 |
| EDC+ ICL GPT-5.1 | 22,5 / 65,4 | **37,7 / 76,4** | 31,4 / 76,6 | 35,9 / 56,0 | – |
| EDC+ ICL Gemma 4 | 11,4 / 59,9 | 12,8 / 68,9 | 17,7 / 68,5 | 4,1 / 10,3 | 2,13 |
| EDC+ ZS GPT-5.1 | 14,6 / 63,4 | 15,0 / 71,0 | 18,6 / 67,9 | 0,0 / 0,6 | – |
| EDC+ ZS Gemma 4 | 9,1 / 58,3 | 11,4 / 67,1 | 11,9 / 64,7 | 0,7 / 2,7 | 1,82 |
| KGGen GPT-5.1 | 13,3 / 62,8 | 14,3 / 68,1 | 13,7 / 68,2 | 0 / 0 | – |
| KGGen Gemma 4 | 9,5 / 60,4 | 11,0 / 66,7 | 12,8 / 67,6 | 0 / 0 | 2,82 |

- Đổi GPT-5.1 → Gemma 4 làm EDC+ ICL mất nặng nhất (Add C 37,7 → 12,8; Deprecate 35,9 → 4,1): Gemma gần như không
  sinh action DEPRECATE (lượt thử 70 bài: 6 triple so với 67 của GPT-5.1, cùng prompt). EDC+ ZS và KGGen chỉ mất
  1–7 điểm C — hai bản này với GPT-5.1 vốn đã thấp.
- Cùng Gemma 4 E4B, pipeline của mình hơn baseline Gemma tốt nhất: Exists C +81 điểm, Deprecate C +65 điểm, Add C
  +20 điểm, tốc độ tương đương (2,4 s/bài cho cả Exists/Deprecate + Add so với 1,8–2,8 s/bài). Phần hơn GPT-5.1
  baseline vì vậy không đến từ model mạnh hơn.
- Dòng mine trong bảng chính thức dùng Add **v1** (bản đã export vào dataset); Add v2 chấm mềm là 36,6 / 75,9 (§12).
- Infer (ee-kg) ngoài phạm vi, không đưa vào bảng. KARMA (bản gốc 03/2025): §16; SAC-KG không có code → bỏ.

## 15. Vì sao baseline Gemma thấp (13/10): không phải lỗi chạy

Kiểm phần chạy: không request nào bị cắt ngữ cảnh (dài nhất 5.726 / 8.192 token, ~1.400 lượt gọi, không lỗi HTTP);
99,5% triple EDC+ đủ 4 phần với action hợp lệ, merge không bỏ triple nào, 350/350 bài parse được.

So từng gold với dự đoán bằng chữ (`scripts/diagnose_emerge_baselines.py`), tỉ lệ trên số gold, dev 350:

| | Exists: nối đúng 2 thực thể | … và đúng nhãn quan hệ | Add: nối đúng 2 thực thể | Deprecate: số dự đoán | Quan hệ là nhãn Wikidata |
|---|---|---|---|---|---|
| EDC+ ICL GPT-5.1 | 38,4% | 15,2% | 61,5% | 399 | 70,5% |
| EDC+ ICL Gemma 4 | 26,4% | 3,1% | 47,6% | 38 | 22,0% |
| EDC+ ZS GPT-5.1 | 34,6% | 3,8% | 53,1% | 6 | 17,8% |
| EDC+ ZS Gemma 4 | 26,6% | 0,0% | 44,3% | 16 | 1,6% |
| KGGen GPT-5.1 | 34,9% | 0,0% | 44,0% | 0 | 1,6% |
| KGGen Gemma 4 | 29,7% | 0,0% | 44,3% | 0 | 0,9% |
| KARMA Gemma 4 | 19,1% | 0,9% | 34,8% | 0 | 14,8% |

1. **Tên quan hệ** là thứ C phạt nặng nhất: C chỉ tính gold đã phủ khi chuỗi "head relation tail" có cosine > 0,9 →
   phải gần trùng nhãn Wikidata. GPT-5.1 bắt chước được nhãn Wikidata trong few-shot (70,5%), Gemma thì không (22%,
   ghi `nationality`, `plays for` thay cho `country of citizenship`, `member of sports team`). Vì vậy G-R chỉ giảm
   5–7 điểm trong khi C giảm một nửa. Bước canonicalize của EDC (EMERGE bỏ) sinh ra để sửa đúng chỗ này.
2. **Trích ít fact hơn**: Gemma nối đúng 2 thực thể ít hơn GPT-5.1 ~12 điểm ở EDC+ ICL; KGGen gần ngang.
3. **Deprecate**: Gemma hầu như không dùng action DEPRECATE (38 so với 399 của GPT-5.1, gold 145); few-shot chỉ có 7
   ví dụ DEPRECATE trên 31.
4. **Trần của đề bài với OIE**: gộp cả 7 lượt chạy chỉ nối được 57,1% gold Exists (Add 74,7%). Phần còn lại là fact
   nền không có trong câu chữ (`AC Milan – sport – association football`), phải tra KG mới có → baseline OIE nào cũng
   thấp ở Exists, kể cả GPT-5.1 (C 22,5); pipeline của mình tra KG nên được 92,8.

Cách báo trong khóa luận: mốc chính là bản GPT-5.1 (số chính chủ); bản Gemma là phép so cùng model; báo cả C lẫn G-R
và nói rõ C thưởng đúng dạng nhãn Wikidata.

## 16. KARMA (bản gốc 82d68ba) bằng Gemma 4 (13/10)

Code KARMA 03/2025 (một file `karma_pipeline.py`) gắn cứng miền y sinh: Reader giữ đoạn khi điểm "liên quan tri thức
y sinh" > 0,2, NER tìm bệnh/thuốc/gene, quan hệ mẫu treats/causes. Conflict Resolution chỉ bắt 4 cặp quan hệ y sinh
đối nghĩa (cùng head, tail) và chỉ quyết định giữ hay bỏ triple **mới**, không xóa triple cũ → KARMA không sinh được
Deprecate trên EMERGE. Wrapper `scripts/karma_emerge.py` (test `tests/test_karma_emerge.py`): mỗi bài một pipeline mới,
KG rỗng; hai biến thể — `karma-biomed` chạy nguyên văn, `karma` chỉ thay 15 cụm chỉ miền trong prompt (`GENERAL_PATCH`:
y sinh → bách khoa, loại thực thể, bộ quan hệ mẫu → `member of`, `employer`, `educated at`…), few-shot trong system
prompt giữ nguyên. Kernel `kgu-baselines-gemma4` v4, chấm bằng `emerge-eval-dev350` v6
(`results/kaggle/dev350_karma_gemma4_tables.txt`, `results/kaggle/karma_gemma4.summary.json`).

| | Exists C / R | Add C / R | Mint+Add C / R | Deprecate | Bài rỗng | triple/bài | s/bài |
|---|---|---|---|---|---|---|---|
| KARMA (đổi miền) | 7,0 / 50,9 | 15,4 / 61,9 | 11,5 / 62,5 | 0 / 0 | 13 | 5,3 | 7,67 |
| KARMA nguyên văn (y sinh) | 0 / 0 | 0 / 0,9 | 0 / 0,6 | 0 / 0 | 345 | 0,05 | 1,83 |

- Bản nguyên văn: Reader bỏ hết đoạn ở 341/350 bài → không dùng được ngoài miền y sinh.
- Bản đổi miền: Add C 15,4 cao nhất trong các baseline Gemma nhờ bộ quan hệ mẫu gần nhãn Wikidata (14,8% quan hệ
  trùng nhãn, KGGen 0,9%); nhưng Exists/G-R thấp nhất vì Summarizer nén đoạn còn 2–4 câu và Evaluator lọc bớt
  (5,7 → 5,3 triple/bài) → nối đúng 2 thực thể chỉ 19,1% gold Exists. Chậm nhất: ~7 + 2×số triple lượt gọi mỗi bài.

## 17. So trên benchmark sân nhà của baseline (14/10): WebNLG (EDC), 2020-COVID-NYT (ATOM)

Không paper baseline nào đo cập nhật KG (§ benchmark của 5 paper), nên ngoài EMERGE còn chấm trên đúng benchmark mà
baseline tự dùng, mọi hệ thống cùng Gemma 4 E4B qua llama-server (kernel `kgu-home-benchmarks`,
`kaggle/home_bench/run_home_bench.py`). Hai benchmark chỉ đo trích xuất (không KG sẵn, không Deprecate) → "mình" là
module Add chạy không cần KG Wikidata (`kgu/extract/text_kg.py`):
- WebNLG: bộ quan hệ lấy từ schema của dataset (159 quan hệ + mô tả) thay cho quan hệ của KG lân cận, ép bằng enum
  trong json_schema; một lệnh gọi/câu, cùng file few-shot OIE của EDC (`scripts/run_webnlg.py`).
- COVID-NYT: KG tự tích lũy theo thứ tự ngày (`TemporalKG`): mỗi nhóm ngày thấy fact đã lưu có nhắc thực thể của nó
  + quan hệ đã dùng, một lệnh gọi trả fact của đoạn (Exists + Add, kèm t_start/t_end) và fact cũ bị báo hết đúng
  (đóng t_end) (`scripts/run_covid_nyt.py`).

Baseline chạy bằng code gốc:
- EDC: clear-nus/edc @ 02aad17 (OIE → schema definition → canonicalization, không refinement), embedder
  e5-mistral-7b-instruct như paper (fp16, GPU thứ hai). Vá: tên model chứa "gpt" (nhánh OpenAI) + OPENAI_BASE_URL;
  import `e5_mistral_utils` (chỉ cho refinement) bọc try/except vì `huggingface_hub.Repository` đã bị gỡ; lỗi gốc ở
  nhánh OpenAI — khối giải phóng model cuối canonicalization xóa verify model chưa từng nạp → bọc điều kiện.
- ATOM (itext2kg 1.1.0, venv Python 3.11): `scripts/atom_covid.py` làm y script đánh giá của họ — 2 bước (factoid →
  5-tuple, mỗi factoid một lệnh gọi) và trích thẳng. Vá: `with_structured_output(method="json_schema")` (issue #44) và
  đưa JSON schema kèm mô tả field vào prompt như OpenAI làm với response_format — llama-server chỉ dùng schema để ép
  ngữ pháp, không vá thì Gemma không thấy hướng dẫn chính của ATOM (nằm trong mô tả field). `return_exceptions=True`
  để một lệnh gọi hỏng chỉ mất factoid đó.
- KGGen 0.4.0: `generate()` trên từng nhóm ngày, `hosted_vllm/local`, temperature 0, max_tokens 4096; không có thời
  gian → t_start/t_end rỗng.

Bộ chấm:
- WebNLG: `evaluate/evaluation_script.py` của EDC (nervaluate 0.1.8 — bản mới đổi API). Script lặng lẽ bỏ dòng không
  parse được (cả gold) → báo số dòng thực sự được chấm.
- COVID-NYT: metric ATOM cài lại (`kgu/eval/atom_quality.py`, `scripts/score_covid_nyt.py`): ghép 1-1 tham lam theo
  cosine "h r t" > ngưỡng; MATCH/HALL/OM; MATCH_t khi cả t_start, t_end khớp (ngày đọc thành tập date). Embedding
  OpenAI của họ thay bằng local, chọn bằng cách chấm 6 cột GPT-4.1 phát hành trong dataset (kernel
  `kgu-covid-calibration`, `results/kaggle/covid_calibration.json`): all-mpnet-base-v2 @ 0,7 (cùng embedder với C của
  EMERGE, cùng ngưỡng gốc của ATOM) tái lập Table 2: trích thẳng F1 0,520 (paper 0,504), ATOM 0,633 (0,638); thời gian
  0,236 (0,220), 0,326 (0,313). Bài có gold mà hệ thống ra rỗng tính 0 (script gốc bỏ qua; báo cả *_skip).

Dữ liệu: bản phát hành có 122 nhóm ngày (01–06/2020), 3.918 gold — paper ghi 274 nhóm / 7.210; gold do Claude soạn. 20 nhóm đầu (theo ngày) đã xem ở 3 lượt chạy thử khi sửa prompt của mình (tách fact nguyên tử; chỉ gán ngày
khi văn bản nói tới thời gian, luật quy đổi ngày tương đối lấy từ schema ATOM = quy ước gold; max_tokens chặn output
lặp) → báo thêm số trên 102 nhóm held-out. Prompt WebNLG không sửa sau lượt thử. Few-shot WebNLG của EDC (dùng chung)
lấy từ chính tập test (câu 1, 2, …) — như paper EDC.

### Kết quả (kernel `kgu-home-benchmarks` v5, 14–15/10; `results/kaggle/home_bench/`)

WebNLG test (1.165 câu, cả 1.165 dòng đều được chấm; EDC 7 dòng rỗng, mình 0):

| Gemma 4 E4B | Partial P / R / F1 | Strict F1 | Exact F1 | Lệnh gọi LLM / câu | Phút |
|---|---|---|---|---|---|
| EDC (OIE → SD → SC, e5-mistral-7b) | 69,8 / 72,1 / **70,6** | 63,9 | 66,8 | ≥ 2 (+ verify từng quan hệ) | 50 |
| Mình (Add, quan hệ ép theo schema) | 73,7 / 75,8 / **74,4** | 68,3 | 71,1 | 1 | 20 |

→ Mình hơn EDC +3,8 Partial / +4,4 Strict / +4,3 Exact F1 trên chính benchmark của EDC, cùng model, ít lệnh gọi hơn.
Cùng ý với §15: chọn quan hệ trong schema ngay lúc trích (constrained decoding) thay cho trích tự do rồi canonicalize.

2020-COVID-NYT (metric ATOM, all-mpnet-base-v2 @ 0,7; nhóm ngày ra rỗng tính 0). 122 nhóm / 102 nhóm held-out
(bỏ 20 nhóm dev):

| | F1 | P | R | F1 held-out | F1_t | MATCH_t | HALL_t | 5-tuple/nhóm | % có mốc t |
|---|---|---|---|---|---|---|---|---|---|
| GPT-4.1 trích thẳng (cột phát hành) | 49,5 | 65,6 | 39,8 | 47,4 | 23,8 | 19,1 | 13,8 | | |
| GPT-4.1 ATOM (cột phát hành) | 63,3 | 56,3 | 72,3 | 64,2 | 32,9 | 37,2 | 21,3 | | |
| ATOM 2 bước, Gemma | **52,5** | 67,4 | **43,0** | **52,6** | 13,7 | 11,1 | 23,1 | 20,5 | 89 |
| KGGen, Gemma | 48,3 | 72,1 | 36,3 | 47,6 | **26,4** | **19,6** | 0 | 13,7 | 0 |
| Mình, Gemma | 45,4 | **76,3** | 32,3 | 43,1 | 10,9 | 7,7 | 14,9 | 11,3 | 80 |
| ATOM trích thẳng, Gemma | 38,3 | 74,3 | 25,8 | 36,0 | 12,3 | 8,3 | 9,5 | 8,7 | 81 |

(gold: 32,1 5-tuple/nhóm, 34,2% có mốc thời gian.)

- Cùng kiểu một lượt gọi mỗi đoạn, mình hơn ATOM trích thẳng +7,1 F1 (held-out +7,1); precision cao nhất bảng
  (76,3, hơn cả GPT-4.1). Thua ATOM 2 bước −7,1 và KGGen −2,9 vì recall: mình 11,3 fact/nhóm, ATOM 20,5, gold 32,1.
  Tách factoid rồi gọi LLM cho từng factoid (ATOM: 10,3 factoid/nhóm → ~11 lệnh gọi/nhóm, 49 phút) là kỹ thuật tăng recall bổ trợ, ghép
  được vào bước trích của mình; benchmark này không đo phần cập nhật (Exists/Deprecate với KG có sẵn) mà pipeline
  của mình nhắm tới.
- F1 thời gian thưởng việc KHÔNG gán ngày: 65,8% gold không có mốc, MATCH_t tính rỗng = rỗng là khớp → KGGen (không
  sinh thời gian) dẫn đầu F1_t. ATOM và mình gán mốc cho 80–90% fact → HALL_t cao. Báo F1_t kèm chú thích này.
- Gemma kéo ATOM từ 63,3 (GPT-4.1) xuống 52,5; trích thẳng 49,5 → 38,3.
- Lỗi: mình mất 1/122 nhóm (output JSON dài quá 4.096 token ở nhóm 38), ATOM trích thẳng 1/122.
- Thời gian kernel 420 phút, trong đó 230 phút là script chấm WebNLG của EDC chạy trên output EDC (chấm output của mình chỉ 8
  phút; nguyên nhân: script thử mọi hoán vị ghép triple, n! với n = max(số dự đoán, số gold) — xem REBEL/Wiki-NRE dưới);
  các hệ thống: mình WebNLG 20, EDC 50, mình COVID 21, ATOM trích thẳng 23, ATOM 49, KGGen 15 phút.

### REBEL và Wiki-NRE (kernel `kgu-edc-rebel-wikinre`, 15–16/10; `results/kaggle/edc_bench/`)

Hai benchmark còn lại của paper EDC, bản trong repo EDC @ 02aad17: mỗi bộ 1.000 câu, schema 196 (REBEL) / 45 (Wiki-NRE)
quan hệ. Cách chạy y WebNLG (`kaggle/edc_bench/run_edc_bench.py`): mình = `scripts/run_webnlg.py` với schema + file
few-shot OIE của EDC cho từng bộ; EDC = OIE → SD → SC, e5-mistral-7b. Prompt REBEL của mình ~28k ký tự (196 quan hệ
kèm mô tả) → mỗi slot 12.288 token. Không sửa prompt sau lượt chạy thử 30 câu (lượt đó chỉ để dò hạ tầng).

Chấm: script gốc của EDC thử mọi hoán vị (n!) → 30 câu REBEL của EDC mất 10,9 phút, câu 12+ triple không bao giờ
xong. `scripts/edc_eval_fast.py` chỉ thay bước tìm hoán vị bằng gán tối ưu (Hungarian) chọn đúng hoán vị script gốc
chọn (n ≤ 7 vẫn duyệt đủ); kiểm toàn văn stdout với script gốc trên WebNLG 1.165 câu (cả EDC và mình) và 30 câu
REBEL/Wiki-NRE: trùng hoàn toàn, trừ 1 ulp ở dòng "Full triple scores" của sklearn trên WebNLG (kernel
`kgu-edc-eval-check`, `results/kaggle/edc_eval_check.summary.json`). Chấm 1.000 câu còn 0,3–0,8 phút.

| Gemma 4 E4B | Partial P / R / F1 | Strict F1 | Exact F1 | Triple/câu | Câu rỗng | Phút |
|---|---|---|---|---|---|---|
| REBEL, EDC | 48,8 / 49,8 / **49,2** | 44,4 | 46,2 | 5,49 | 0 | 61 |
| REBEL, mình | 54,9 / 55,9 / **55,3** | 50,7 | 52,1 | 4,41 | 1 | 21 |
| Wiki-NRE, EDC | 62,0 / 62,5 / **62,2** | 60,8 | 61,3 | 3,77 | 4 | 41 |
| Wiki-NRE, mình | 66,0 / 66,4 / **66,2** | 65,1 | 65,3 | 3,60 | 0 | 15 |

(cả 1.000 dòng mỗi bộ đều được chấm; log không có lỗi/retry.)

→ Mình hơn EDC trên cả ba benchmark của EDC: WebNLG +3,8, REBEL +6,0, Wiki-NRE +4,0 Partial F1 (Strict +4,4 / +6,3 /
+4,3), nhanh gấp ~3 lần, 1 lệnh gọi/câu. Khoảng cách lớn nhất ở REBEL (schema lớn nhất): chọn quan hệ trong schema
ngay lúc trích bằng enum thay cho trích tự do rồi canonicalize về 196 quan hệ.

Cách báo trong khóa luận: bảng "sân nhà" là bằng chứng phụ cho EMERGE — trên WebNLG, REBEL, Wiki-NRE mình hơn EDC; trên COVID-NYT
(chỉ đo trích xuất) mình ngang nhóm baseline một lượt (hơn ATOM trích thẳng, dưới KGGen 3 điểm), dưới ATOM 2 bước do
recall; nói rõ quy trình dev/held-out và chú thích F1_t.

## 18. EMERGE toàn bộ 3.500 bài (15–16/10): số CHÍNH THỨC

Chạy: mình = kernel `kgu-judge-emerge-full` v2 (20 bài/delta mỗi tiến trình — v1 hết RAM sau 150 bài vì mỗi job chép
bảng nhãn KG của snapshot; khởi động lại llama-server nếu chết), Exists/Deprecate v3 + Add v2. EDC+ ICL và KGGen bằng
Gemma = kernel `kgu-baselines-full` (EDC+ ZS, KARMA chỉ có số dev350 — quota GPU). Ghép bằng
`scripts/export_emerge_predictions.py` vào evaluation_set ra từ kernel baseline → dataset `ngocnam2005/emerge-full-mine`
→ kernel `emerge-eval-full` (`kaggle/emerge_eval_full/`, bộ chấm chính chủ, config `…_zs_fixed_with_kg`, không nạp KG).
EDC+ Gemma có `predicted_triples` rỗng ở mọi bài (cả dev350): bộ chấm đọc `predicted_triples_oie` — không phải lỗi.
Kết quả: `results/kaggle/emerge_full/` (`eval_full_tables.txt`).

C / G-R (x = Exists, e = Add, ee = Mint+Add, ee-kg = Infer, d = Deprecate):

| | Exists | Add | Mint+Add | Infer | Deprecate | s/bài |
|---|---|---|---|---|---|---|
| Mình (Gemma) | **91,3 / 90,7** | 39,3 / 75,2 | 0 / 0 | 0 / 0 | **63,3 / 67,4** | 2,28 |
| EDC+ ICL (Gemma) | 12,5 / 59,2 | 19,9 / 69,8 | 19,0 / 68,8 | 13,4 / 64,2 | 6,3 / 11,4 | 1,93 |
| KGGen (Gemma) | 10,2 / 59,1 | 15,5 / 66,7 | 12,9 / 65,7 | 1,9 / 56,4 | 0 / 0 | 2,52 |
| KG-aware oracle GPT-5.1 | 60,4 / 50,3 | 46,5 / 42,9 | 43,2 / 47,6 | 50,3 / 75,9 | 37,4 / 35,8 | |
| EDC+ ICL GPT-5.1 | 25,7 / 66,3 | 40,2 / 77,5 | 35,6 / 77,3 | 21,9 / 72,4 | 34,8 / 53,2 | |
| EDC+ ZS GPT-5.1 | 15,5 / 64,3 | 22,6 / 71,5 | 18,9 / 69,6 | 5,8 / 52,5 | 1,5 / 3,2 | |
| KGGen GPT-5.1 | 12,8 / 62,7 | 15,2 / 68,1 | 13,3 / 67,0 | 3,3 / 59,1 | 0 / 0 | |

- Chấm lại prediction GPT-5.1 phát hành cho ĐÚNG từng số EDC+ GPT-5.1 và KGGen GPT-5.1 trong Bảng 3 paper EMERGE v2
  (arXiv 2507.03617v2, 07/04/2026, cùng 3.500 bài) → số chạy lại và số công bố so trực tiếp được. Oracle không có
  trong paper; README repo ghi Deprecate 38,4 / 36,9 (mình chấm 37,4 / 35,8), các thao tác khác trùng.
- Cùng Gemma: mình hơn EDC+ ICL +78,8 / +19,4 / +57,0 C (Exists / Add / Deprecate). Hơn EDC+ GPT-5.1 +65,6 Exists,
  +28,5 Deprecate; Add −0,9. Hơn oracle +30,9 Exists, +25,9 Deprecate; Add −7,2. Mint+Add = 0 (chưa mint thực thể).
- QID exact match (P/R/F1): mình Exists 65,0/81,4/69,6, Add 14,4/24,9/16,7, Deprecate 40,0/57,6/44,5; oracle
  57,0/46,8/49,7, 32,6/34,8/32,5, 34,2/33,9/34,0.
- Điểm mềm (`emerge_full_v3_addv2_kaggle.soft.json`) lệch số chính thức ≤ 0,9 (Add 40,2/75,9 vs 39,3/75,2).
- So với dev350 (số chính thức, Add v1): Exists 92,8 → 91,3; Deprecate 69,0 → 63,3 (dev350 chỉ 145 gold Deprecate).

## 19. COVID-NYT: rút câu đơn + gộp tên (v0/v1/v2, 16/10; kernel `kgu-covid-condense-canon` v4 @6bdeb80)

Spec/plan: `docs/superpowers/specs/2026-10-15-condense-canonicalize-design.md`. v0 = bản cũ (1 lệnh/nhóm); v1 = rút
câu đơn (condense + bù ý thiếu) rồi trích theo câu đánh số (+ bù câu chưa có fact); v2 = v1 + cứ 10 nhóm ngày gộp tên
(k-means trên all-mpnet, nhóm cosine ≥ 0,75 cùng chữ số, một lệnh gọi/cụm, LLM trả có/không + lý do).
3 lượt dev trên 20 nhóm đầu (`results/kaggle/covid_v2_dev_r{1,2,3}*`): r2 bỏ header "Observation date" + cấm ngày/
snake_case làm thực thể + prompt gộp chặt hơn; r3 sửa regex header (2/122 đoạn dính liền văn bản → r2 xóa sạch đoạn).
v0 riêng 3 lượt dev: 49,6 / 50,8 / 52,6 → nhiễu giữa lượt ~±1,5 F1 trên 20 nhóm.

Toàn bộ 122 nhóm (`results/kaggle/covid_v2/`), metric ATOM all-mpnet @ 0,7, nhóm rỗng tính 0:

| | F1 | P | R | F1 held-out (102) | 5-tuple/nhóm | F1_t | lệnh gọi | phút |
|---|---|---|---|---|---|---|---|---|
| v0 | 42,5 | 74,8 | 29,6 | 41,0 | 10,5 | 8,3 | 124 | 20 |
| v1 | 49,4 | 64,5 | 40,0 | 48,3 | 17,1 | 22,0 | 349 | 44 |
| v2 | **50,1** | 65,7 | 40,4 | **48,9** | 16,8 | 22,7 | 354 + 312 gộp | 57 |

- v2 − v0 = +7,6 F1 (held-out +7,9). v0 lượt này 42,5, lượt `kgu-home-benchmarks` v5 là 45,4 → dao động giữa lượt ~3
  F1 trên 122 nhóm; so baseline (ATOM 52,5, KGGen 48,3 từ lượt khác) cần nói rõ.
- Condense: 17,3 câu/nhóm; mục có số/tên riêng bị sót 201 → 24 sau lệnh bù; 21 câu cần lệnh bù fact. Rỗng: v2 2/122
  (nhóm 38: JSON lặp, có ở cả v0; nhóm 50).
- Gộp tên: 583 quyết định (488 thực thể, 95 quan hệ), "có" 134 (103 + 31), 29 chỉ khác hoa/thường; 0 lỗi lệnh gọi.
  Metric gộp kiểu ATOM (cặp cosine > 0,8): thực thể F1 v1 85,9 → v2 91,5 (cặp gần trùng 597 → 366); quan hệ 95,8 → 97,1.
  Sai thấy được: "new coronavirus" → "coronavirus outbreak", "SoftBank" → "news about SoftBank". Mẫu 100 quyết định
  (50 có) để kiểm tay: `results/kaggle/covid_v2_audit.csv` (cột `correct`; chấm: `scripts/audit_merges.py --labels`).
- Kiểm mẫu 100 quyết định (17/10; người gán là Claude Opus 5.5 theo yêu cầu, không phải người; lý do từng ca sai/khó ở
  cột `note`). Đúng khi gộp hai tên chỉ khác bề mặt (hoa/thường, số ít/nhiều của danh từ chỉ loại, chức danh, mạo từ,
  thì của cùng động từ); sai khi gộp cá thể vào nhóm, đảo chủ động/bị động, bỏ thể tương lai/thói quen, hoặc hai khái
  niệm khác nhau. "Có" đúng 37/50 (74%; 9/9 ca chỉ khác hoa/thường đều đúng), "không" đúng 46/50 (92%). 13 ca gộp sai:
  cá thể → nhóm 5 (Officer → Officers, expert → Experts, speaker, EMS worker, government), khái niệm khác 4 ("news about
  SoftBank" → SoftBank, threat ↔ fears, scourge, Hong Kong police ↔ authorities), đảo chủ động/bị động 2
  (canceled/have_been_canceled, announced/was_announced), bỏ thể 2 (will_open → opened, usually_). 4 ca bỏ sót: hai
  triple trùng "road-test requirement", hai nút dateline, birthday/birthdays, report/report_on. Ước lượng trên cả 583
  quyết định: ~35 gộp sai trong 134 "có", ~36 bỏ sót trong 449 "không" (mẫu 50 mỗi bên, sai số ±12 điểm ở 95%).

### 18b. Điều kiện `mentions` = hyperlink Wikipedia (17/10)

`mentions` của EMERGE là hyperlink trong đoạn Wikipedia (`target_entity` = trang được link), không phải nhãn gold soạn
tay; pipeline dùng chúng để chọn thực thể và tra cạnh KG-trước. KG con (`kg_subsets`) cũng chỉ chứa lân cận 1-hop của các
thực thể được hyperlink. Thử biến thể không hyperlink trên đủ 3.500 bài (CPU, không LLM): thực thể chủ đề
(`anchor_page_qid`) + hàng xóm 1-hop của nó có nhãn KG khớp văn bản → bắt 20,4% hyperlink (precision 65,5%), ứng viên
1,9/bài (hyperlink: 5,8); phủ 16,0% gold Exists (hyperlink: 99,3%), 32,0% Deprecate (91,2%), 8,7% Add có cả hai đầu được
liên kết; anchor chỉ có cạnh trong KG con ở 63,4% bài; chỉ 21,8% gold Exists dính tới anchor. → Biến thể không hyperlink
cần bộ liên kết trên toàn Wikidata; EMERGE tự làm việc tương tự bằng ReLiK (`kg_rag@32`, 13,8 C Exists). Quyết định 17/10:
không chạy, báo rõ điều kiện (pipeline nằm giữa `kg_rag@32` và oracle về thông tin đầu vào).

## 20. MINE-1 (benchmark của KGGen, 17/10; kernel `kgu-mine` v1–v3)

100 bài luận × 15 fact (HF `josancamon/kg-gen-MINE-evaluation-dataset`, 101 hàng, hàng id 106 bài rỗng). Mỗi bài dựng
một KG; mỗi fact truy xuất bằng đúng hàm của kg-gen 0.4.0 (`KGGen.retrieve`, all-MiniLM-L6-v2, k=8, sâu 2), judge dspy
`EvaluateResponse` chép nguyên văn → 1 nếu ngữ cảnh chứa fact; điểm = trung bình theo bài. Judge là Gemma 4 E4B
(paper dùng judge OpenAI) nên chấm lại cả ba KG phát hành trong dataset để đo lệch.

- Mình: `scripts/run_mine.py` (TemporalKG condense + gộp tên như COVID-NYT v2, mỗi đoạn một lượt `update`). KGGen 0.4.0
  bằng Gemma (`scripts/kggen_mine.py`) KHÔNG gom cụm: lượt 1 bật gom cụm thì Gemma trả nhãn ngoài danh sách cho phép,
  vòng thử lại của KGGen chạy ~10 giờ cho 6 bài, kernel bị cắt ở 12 giờ (các lượt KGGen trên COVID-NYT/EMERGE cũng không
  gom cụm).
- **Dataset lệch hàng**: ở ~80% hàng, cột KG + `*_responses` + `*_accuracy` là của bài ở hàng trước (KG hàng i chứa thực
  thể của bài i−1, `correct_answer` trong responses là fact của bài i−1). Chấm thẳng theo hàng thì KG phát hành chỉ được
  ~10%. `score_mine.realign` gán mỗi KG về bài có `generated_queries` trùng ≥ nửa số `correct_answer` (fallback: từ trong
  tên thực thể); kggen gán 96 + 2, graphrag 84, openie 95 + 2; bỏ 4–5 (trùng đích, khớp thua).

Kết quả (`results/kaggle/mine/v2` cho mình + KGGen-Gemma, `v3` cho KG phát hành; bảng `v3/tables.json` từ
`scripts/mine_tables.py`):

| | 100 bài | trên bài có KG KGGen phát hành (95) | judge gốc (dataset, đã ghép) | paper |
|---|---|---|---|---|
| Mình (Gemma 4 E4B) | **85,4** | **85,4** | | |
| KGGen + Gemma 4 E4B, không gom cụm | 75,9 | 76,3 | | |
| KGGen phát hành | 68,7 | 72,4 | 66,6 | 66,07 |
| GraphRAG phát hành (84 bài có KG) | 37,5 | 44,6 (trên 84) | 47,5 | 47,80 |
| OpenIE phát hành (95 bài có KG) | 44,9 | 47,2 (trên 95) | 29,1 | 29,84 |

Trên tập 84 bài của GraphRAG: mình 84,7, KGGen-Gemma 74,2.

- Judge Gemma lệch không đều: KGGen +5,8, GraphRAG −2,9, OpenIE +18,1 so judge gốc; đồng ý từng fact 78 / 85 / 71%.
  Ngữ cảnh truy xuất của mình (kg-gen 0.4.0) khác `retrieved_context` trong dataset ở gần mọi fact (bản kg-gen khác) → lệch
  gộp cả truy xuất lẫn judge, không tách được. Thứ tự KGGen > hai hệ còn lại giữ nguyên; GraphRAG/OpenIE đảo chỗ.
- Metric chỉ đo recall theo ngữ cảnh truy xuất: không phạt cạnh thừa/sai, KG to và tên thực thể dài có lợi. Mình 71,6 quan
  hệ, 84,2 thực thể/bài, 3,4 từ/thực thể, ngữ cảnh 1.996 ký tự/fact; KGGen-Gemma 46,9 / 37,4 / 1,7 / 1.458; KGGen phát
  hành 71,6 / 101,4 / 1,3 / 1.366; OpenIE 267 quan hệ, 4.490 ký tự (judge Gemma dễ hơn hẳn trên ngữ cảnh dài này).
- Chi phí: mình 178 giây/bài, KGGen-Gemma 34 giây/bài (cùng llama-server 4 slot); 1 bài KG mình có lỗi lệnh gọi ở một
  đoạn (vẫn có KG).
