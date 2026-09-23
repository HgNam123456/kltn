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
