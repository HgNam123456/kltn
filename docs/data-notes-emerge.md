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
