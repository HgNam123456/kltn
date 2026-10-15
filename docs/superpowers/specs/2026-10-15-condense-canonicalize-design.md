# Rút câu đơn trước khi trích + gộp tên bằng k-means/LLM cho KG tích lũy (COVID-NYT)

Ngày 15/10/2026. Áp dụng trước cho biến thể `TemporalKG` (2020-COVID-NYT); EMERGE và WebNLG/REBEL/Wiki-NRE để sau.

## 1. Mục tiêu

Kết quả 15/10 (`docs/data-notes-emerge.md` §17): trên COVID-NYT pipeline của mình có precision cao nhất (76,3) nhưng
recall thấp (32,3; ATOM 2 bước 43,0), trích 11,3 fact/nhóm ngày so với 32,1 của gold. KG tích lũy còn để tên do LLM
tự đặt, nên cùng một thực thể/quan hệ có thể mang nhiều tên.

Hai thay đổi:
1. **Rút câu đơn**: viết lại đoạn tin thành các câu đơn, mỗi câu một ý, tự đứng được, đủ ý, rồi mới trích triple.
   Mục tiêu tăng recall mà giữ chi phí thấp hơn nhiều so với ATOM (~11 lệnh gọi/nhóm ngày).
2. **Gộp tên**: định kỳ phân cụm k-means tên entity rồi tên relation theo embedding; trong mỗi cụm chọn các nhóm
   tên gần nhau vượt ngưỡng; một lệnh gọi LLM mỗi cụm quyết định yes/no + lý do cho từng ứng viên; gộp về một tên chung.

Tiêu chí thành công (đo trên 102 nhóm held-out, cùng Gemma 4 E4B, cùng bộ chấm):
- v1 (rút câu) có recall cao hơn v0 (bản hiện tại) và F1 không thấp hơn v0.
- v2 (v1 + gộp tên) có F1 không thấp hơn v1 quá 1 điểm, và chỉ số gộp tên của ATOM tốt hơn v1.
- Kiểm tay 100 quyết định gộp: ít nhất 90% quyết định "yes" đúng.

## 2. Luồng xử lý một nhóm ngày

```
đoạn tin ──condense──▶ câu đơn [1..n] ──(thiếu mục?)──▶ bổ sung câu
        ──extract (STORED, RELATIONS, câu đánh số)──▶ fact có chỉ số câu + ended
        ──câu không có fact?──▶ extract lần 2 chỉ trên các câu đó
        ──▶ output nhóm ngày (5-tuple) + cập nhật KG
mỗi N nhóm ngày và cuối cùng: gộp tên entity, rồi gộp tên relation
```

### 2.1. Rút câu đơn (`kgu/extract/condense.py`)

- `condense(text, llm) -> list[str]`: 1 lệnh gọi, output `{"sentences": [str]}`.
  Luật trong prompt: mỗi câu đúng một ý; câu tự đứng được (thay đại từ bằng tên); giữ nguyên số liệu, tên riêng và
  cụm thời gian như trong đoạn ("on Sunday", "last week" — việc quy đổi ngày vẫn do bước trích làm); không bỏ ý, không
  thêm ý. Ví dụ trong prompt: "A, a teacher, joined a course at B." → "A is a teacher." / "A joined a course at B."
- `missing_items(text, sentences) -> list[str]`: các số (so theo chữ số, bỏ dấu phẩy/chấm) và cụm từ viết hoa có trong
  đoạn nhưng không có trong câu nào (so không phân biệt hoa thường; bỏ một danh sách từ chức năng như The, In, But).
  Tối đa 15 mục.
- Có mục thiếu → 1 lệnh gọi bổ sung: đưa đoạn gốc, các câu đã có và danh sách mục thiếu, yêu cầu thêm câu đơn cho các
  ý chứa chúng. Số mục còn thiếu sau bổ sung được trả về để ghi log.
- `condense` lỗi (LLM/parse) → `TemporalKG` quay về trích trên đoạn gốc như v0.

### 2.2. Trích trên câu đánh số và bổ sung (`kgu/extract/text_kg.py`)

- `TemporalKG(condense=False, ...)`: `condense=False` giữ nguyên hành vi v0 (để chạy lại v0 và các test cũ).
- Khi `condense=True`: phần "Passage" của prompt thay bằng danh sách câu `[1] … [n]`; mỗi fact có thêm trường
  `sentence` (số câu nguồn). STORED, RELATIONS, `ended`, luật quy đổi ngày giữ nguyên.
- Câu không có fact nào trỏ về → 1 lệnh gọi trích nữa chỉ trên các câu đó (giữ số câu gốc, cùng STORED/RELATIONS);
  `ended` của lệnh bổ sung bị bỏ qua. Lệnh bổ sung lỗi → bỏ qua.
- Thống kê mỗi nhóm ngày (số câu, mục thiếu trước/sau bổ sung, số câu phải trích bổ sung, số lệnh gọi) trả về để
  script ghi vào output.

### 2.3. Gộp tên (`kgu/extract/canonicalize.py`)

Mỗi lượt chạy cho `kind="entity"` rồi `kind="relation"`.
1. **Gom tên**: entity = mọi head/tail; relation = mọi tên quan hệ (đổi `_` thành dấu cách khi nhúng). Mỗi tên kèm
   số fact chứa nó và tối đa 3 fact gần nhất làm ngữ cảnh.
2. **Nhúng + k-means**: embedder truyền vào (`embed(list[str]) -> ndarray` đã chuẩn hóa; kernel dùng
   sentence-transformers/all-mpnet-base-v2). k = ceil(số tên / 10). k-means++ viết bằng numpy, seed cố định (không
   thêm phụ thuộc scikit-learn).
3. **Nhóm ứng viên trong cụm**: nối cặp tên có cosine ≥ τ (mặc định 0,75, chọn lại trên 20 nhóm dev) và cùng tập
   chữ số (`1,694 Cases` không bao giờ cặp với `1,700 Cases`); bỏ cặp đã bị "no" ở lượt trước. Mỗi thành phần liên
   thông ≥ 2 tên là một nhóm; tên mốc = tên gần tâm nhóm nhất (hòa → nhiều fact hơn). Nhóm quá 8 tên → giữ mốc + 7 tên
   cosine cao nhất với mốc.
4. **Chỉ gọi LLM cho cụm có nhóm chứa tên mới** (tên chưa có ở lượt trước).
5. **Một lệnh gọi mỗi cụm**: prompt liệt kê từng nhóm (mốc + ứng viên, mỗi tên kèm fact ngữ cảnh). Output
   `{"groups": [{"group": i, "decisions": [{"name", "reason", "same"}], "canonical"}]}`; `name`, `canonical` ép enum
   theo tên trong cụm; `reason` đứng trước `same`. Entity: `same` = cùng một thực thể ngoài đời (không phải liên quan,
   rộng hơn, hẹp hơn, bộ phận). Relation: cùng nghĩa và cùng chiều head→tail.
   Code kiểm lại: quyết định cho tên ngoài nhóm bị bỏ; `canonical` không phải mốc hay tên được "yes" → dùng mốc.
6. **Áp dụng**: đổi tên trong mọi fact; bỏ fact trùng hệt (cùng h, r, t, t_start, t_end); tên cũ thành bí danh của
   tên chung (`kg.aliases[canonical]`). `TemporalKG._stored` khớp đoạn tin với tên chung hoặc bí danh.
7. **Nhớ quyết định**: cặp (mốc, ứng viên) đã "no" không hỏi lại; cặp "yes" đã gộp.
8. **Log**: mỗi quyết định một dòng trong `<out>.merges.jsonl`: round, kind, cluster, group, name, anchor, cosine,
   same, reason, canonical.
9. Lệnh gọi của một cụm lỗi → bỏ qua cụm đó (KG không đổi), ghi log.

Lịch: `scripts/run_covid_nyt.py --canon-every N` (mặc định N = 10) chạy gộp sau mỗi N nhóm ngày và sau nhóm cuối.
Output của nhóm ngày đã ghi không viết lại; các nhóm sau tự dùng tên chung vì LLM được dặn dùng lại tên trong STORED.

## 3. Đánh giá

Ba biến thể chạy trong cùng một kernel Kaggle (Gemma 4 E4B, llama-server như `kgu-home-benchmarks`):
v0 = bản hiện tại (chạy lại để cùng môi trường), v1 = v0 + rút câu + bổ sung, v2 = v1 + gộp tên.

- **Trích xuất**: `scripts/score_covid_nyt.py` (all-mpnet-base-v2 @ 0,7) trên 122 nhóm và 102 nhóm held-out: F1, P, R,
  F1_t, 5-tuple/nhóm, số lệnh gọi, thời gian.
- **Rút câu**: số câu/nhóm, tỉ lệ mục thiếu trước/sau bổ sung, số câu phải trích bổ sung.
- **Gộp tên**: cài lại chỉ số của ATOM (`evaluation/atom/merge/evaluate_atom_merge.py`) trong `kgu/eval/atom_merge.py`
  với embedder local, ngưỡng 0,8 như bản gốc:
  - `similar` = số cặp tên khác nhau trong KG cuối có cosine > 0,8;
  - ER precision = 1 − similar / (số lần xuất hiện entity trong gold − số entity của KG), kẹp [0, 1];
  - ER recall = 1 − similar / (số lần xuất hiện − số entity gold khác nhau (không phân biệt hoa thường) − số cặp
    entity gold có cosine > 0,8), kẹp [0, 1]; RR tương tự cho relation.
  Báo cho v0, v1, v2, kèm số tên entity/relation khác nhau. Số trong paper ATOM chỉ để tham khảo (GPT-4.1, embedding
  OpenAI).
- **Kiểm tay**: `scripts/audit_merges.py` rút 100 quyết định của v2 (50 yes, 50 no, seed 0) ra CSV có cột nhãn để
  người kiểm điền; chạy lại với `--labels` để tính tỉ lệ đúng của "yes" và "no".

Quy trình: chỉnh prompt và τ chỉ trên 20 nhóm dev (giống lần trước); số chính báo trên 102 nhóm held-out.

## 4. Ngoài phạm vi

- EMERGE, WebNLG/REBEL/Wiki-NRE (làm sau nếu v1/v2 có ích).
- Viết lại output đã chấm theo tên chung; gộp entity khác loại; dựng lại KG đã merge của ATOM để so trực tiếp.

## 5. Kiểm thử

Unit test (pytest, `FakeLLMClient`, embedder giả) cho: tách câu, phát hiện mục thiếu và gọi bổ sung, quay về v0 khi
rút câu lỗi, gán fact về câu nguồn, trích bổ sung đúng câu thiếu, k-means tất định, nhóm theo τ, chặn cặp khác số,
không hỏi lại cặp "no", tên chung không hợp lệ → mốc, đổi tên + bỏ trùng + bí danh, `_stored` khớp qua bí danh, chỉ số
gộp tên của ATOM trên ví dụ nhỏ tính tay.
