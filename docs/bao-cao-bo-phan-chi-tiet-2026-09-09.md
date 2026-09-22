# Báo cáo chi tiết: 6 thay đổi của bộ phán LLM (bản Gemma 4)

Ngày: 2026-09-09 · Nhánh `feat/core-pipeline`, HEAD 7b82f9a
Mã: `kgu/judge/llm.py`, `scripts/probe_judge.py`, `scripts/run_eval.py`
Bổ sung cho mục 3 của `docs/bao-cao-gemma4-2026-09-08.md`. Mọi prompt trong bài này là prompt thật, render bằng chính code hiện tại trên tin idx=1 của split test (không gọi LLM).

---

## 0. Bộ phán làm gì, và "mẫu" là gì

Sau khi áp các thay đổi tường minh (ví dụ "Randolph rời Nuggets, sang Cavaliers"), bước khoanh vùng (`localize`) đưa ra hai danh sách cạnh ứng viên:

- **vế cắt**: cạnh hiện có, có thể không còn đúng (đồng đội cũ của Randolph, đội cũ của Randolph, ...)
- **vế sinh**: cặp entity chưa nối, có thể cần nối (Randolph với 21 cầu thủ Cavaliers, với GM, với HLV, ...)

Tin idx=1 có **60 cạnh cắt và 38 cạnh thêm**. Bản 1 của bộ phán liệt kê từng cạnh, prompt 5.307 token, model 2B cắt sạch. Bản 2 gom cạnh thành **mẫu**: các cạnh chỉ khác nhau ở một entity X thì gộp lại thành một dòng, X chạy trên một nhóm. Tin idx=1 còn **4 mẫu cắt + 4 mẫu thêm**:

```
[C0] (X, <teammate>, Anthony_Randolph)   15 X = 15 đồng đội cũ ở Nuggets
[C1] (Anthony_Randolph, <teammate>, X)   15 X, chiều ngược lại
[C2] (X, <player>, Denver_Nuggets)       15 X = 15 cầu thủ Nuggets vẫn ở lại
[C3] (Denver_Nuggets, <player>, X)       15 X, chiều ngược lại

[A0] Randolph ~ X, bằng chứng (X, <player>, Cavaliers)          21 X = cầu thủ Cavaliers
[A1] Randolph ~ X, bằng chứng (X, <general_mananger>, Cavaliers) 1 X = David Griffin
[A2] Randolph ~ X, bằng chứng (X, <head_coach>, Cavaliers)       1 X = Mike Brown
[A3] Cavaliers ~ X, bằng chứng (X, <teammate>, Randolph)        15 X = đồng đội cũ ở Nuggets
```

Đáp án vàng: cắt C0, C1 (đồng đội cũ hết là đồng đội). Giữ C2, C3 (đồng đội cũ vẫn thuộc Nuggets). Thêm A0 với `<teammate>` cả 2 chiều. Bỏ A1, A2, A3.

Bộ phán chỉ cần trả lời đúng **hai câu**: "đồng đội có mất khi một người rời đội không" (C0/C1 cắt, C2/C3 giữ) và "người mới có thành đồng đội của cả đội không" (A0 thêm, A3 bỏ). Mọi thay đổi dưới đây là để model 4B trả lời hai câu đó ổn định.

Điểm khó nằm ở chỗ: C0 và C2 **giống hệt nhau về hình thức**. Cùng một X (Aaron Brooks), cùng "X không tham gia", chỉ khác quan hệ `<teammate>` với `<player>`. Model phải hiểu rằng đồng đội là quan hệ phái sinh từ "cùng đội", còn "thuộc đội" là quan hệ gốc.

---

## 1. Thêm trường `reasoning` vào schema, đứng trước quyết định

**Trước.** Schema pha CẮT là `{"invalidate": [int]}`. Model bị constrained decoding ép trả về ngay danh sách số, không có chỗ suy nghĩ. Với model nhỏ, token đầu tiên sinh ra đã là quyết định.

**Sau.** Schema thành `{"reasoning": str, "invalidate": [int]}` và `{"reasoning": str, "add": [...]}`. Thứ tự trường quan trọng: `reasoning` đứng trước nên model bắt buộc viết 1–2 câu lập luận rồi mới điền số. Đây là chain-of-thought rút gọn, nhét vào JSON để vẫn parse được.

```python
def make_cut_schema():
    return create_model("CutOut", reasoning=(str, ""), invalidate=(list[int], []))
```

Few-shot trong system prompt cũng đổi theo, cho model thấy reasoning trông thế nào:

```
Kết quả đúng: {"reasoning": "C0 đúng: Pedri vẫn thuộc Barca, plays_for độc lập (0%).
                             C1 sai: đồng đội luôn đi kèm cùng CLB (100%), Messi đã rời Barca.
                             C2 đúng: bạn thân không phụ thuộc CLB (40%, không phải quy luật).",
               "invalidate": [1]}
```

**Tác dụng đo.** Kết hợp với "tương tự" (mục 2), cut teammate từ 42/44 lên 44/44, cut player sai từ 39/44 xuống 30/44, add từ 4/11 lên 9/11. Không tách riêng được phần của reasoning vì hai thay đổi được thử cùng lúc. Chi phí: mỗi lệnh gọi sinh thêm khoảng 30–60 token.

---

## 2. Bằng chứng "tương tự" cho mẫu thêm

Đây là thay đổi quan trọng nhất cho pha SINH.

**Vấn đề.** Với mẫu A0 (Randolph ~ 21 cầu thủ Cavaliers), model phải tự quyết định **quan hệ gì** và **chiều nào**. Qwen 2B thường chọn `<player>` (sai, vì player là cầu thủ–đội) hoặc chỉ một chiều. Gemma 4 với prompt v1 cũng chỉ đúng 4/11.

**Ý tưởng.** Trong KG-trước, các cầu thủ Cavaliers **đang có sẵn** quan hệ với nhau. Randolph vừa vào Cavaliers với cùng vai `<player>` như họ. Vậy cứ đếm: những entity S cùng vai với Randolph (cũng có cạnh `(S, <player>, Cavaliers)`) đang có quan hệ gì với các X trong mẫu? Kết quả in thẳng vào prompt:

```
[A0] Anthony_Randolph_2014-15 ~ X — nhóm 21 X: Alonzo_Gee_2014-15, Anderson_Varejao_2014-15, ...
  - bằng chứng: (X, <player>, Cleveland_Cavaliers)
  - tương tự: (S, <teammate>, X) 100%, (X, <teammate>, S) 100%
[A1] Anthony_Randolph_2014-15 ~ X — nhóm 1 X: David_Griffin_2014-15
  - bằng chứng: (X, <general_mananger>, Cleveland_Cavaliers)
  - tương tự: không có cạnh S–X nào
[A2] Anthony_Randolph_2014-15 ~ X — nhóm 1 X: Mike_Brown_2014-15
  - bằng chứng: (X, <head_coach>, Cleveland_Cavaliers)
  - tương tự: không có cạnh S–X nào
```

Đọc: "21 cầu thủ Cavaliers hiện tại, 100% cặp có `<teammate>` cả hai chiều. Randolph vào cùng vai, nên cũng cần đúng các cạnh đó." Còn A1: "không cầu thủ Cavaliers nào có cạnh với GM, nên Randolph cũng không." Model không phải đoán quan hệ nữa, chỉ cần chép.

**Cách tính** (`_analogy`, `kgu/judge/llm.py`):

1. Tìm op ADD tường minh nối subject với `other` (Randolph, `<player>`, Cavaliers).
2. Siblings S = mọi entity có cạnh cùng loại tới `other` (tối đa 30).
3. Với mỗi quan hệ r trong danh sách và mỗi chiều, đếm tỷ lệ cặp (S, X) đang có cạnh. Chỉ in khi tỷ lệ ≥ 50%.
4. Không có gì đạt 50% thì in "không có cạnh S–X nào".

Không có luật miền nào ("cầu thủ cùng đội là đồng đội") được viết vào code. Toàn bộ là đếm cạnh đang tồn tại, nên chuyển sang miền khác (công ty, trường học) vẫn chạy.

**Tác dụng đo.** Pha SINH từ 4/11 lên 9/11 (cùng với reasoning), rồi 11/11 khi hỏi mỗi mẫu một lệnh gọi. add_acc pipeline từ 0,420 lên 0,824.

---

## 3. Mẫu cắt viết dạng fact cụ thể, khối 3–4 dòng

**Trước (prompt v1)**, mỗi mẫu là một dòng trừu tượng với biến X:

```
[C0] (X, <teammate>, Anthony_Randolph_2014-15) — 15 X: Aaron_Brooks_2014-15, Andre_Miller_2014-15, ...; X cũng nối với Denver_Nuggets
[C2] (X, <player>, Denver_Nuggets) — 15 X: Aaron_Brooks_2014-15, Andre_Miller_2014-15, ...; X cũng nối với Anthony_Randolph_2014-15
```

Model phải tự giải: X là ai, X có liên quan gì đến thay đổi, quan hệ này có phụ thuộc thay đổi không. Gemma 4 v1 cắt nhầm C2 ở 39/44 tin.

**Sau**, mỗi mẫu thành một câu hỏi cụ thể "fact này còn đúng không?", kèm ba dòng bối cảnh:

```
[C0] "Aaron_Brooks_2014-15 <teammate> Anthony_Randolph_2014-15" — đại diện cho nhóm 15 X: Aaron_Brooks_2014-15, Andre_Miller_2014-15, ...
  - X = Aaron_Brooks_2014-15: không tham gia, không rời đi
  - Anthony_Randolph_2014-15 vừa tách khỏi Denver_Nuggets (qua <player>)
  - trong KG: 100% cặp <teammate> cùng chung hàng xóm qua <player>
[C2] "Aaron_Brooks_2014-15 <player> Denver_Nuggets" — đại diện cho nhóm 15 X: Aaron_Brooks_2014-15, Andre_Miller_2014-15, ...
  - X = Aaron_Brooks_2014-15: không tham gia, không rời đi
  - Denver_Nuggets vừa tách khỏi Anthony_Randolph_2014-15 (qua <player>)
  - trong KG: 0% cặp <player> cùng chung hàng xóm qua <player>
```

Bốn thành phần, mỗi cái nhắm một lỗi đã thấy:

| Dòng | Nhắm lỗi gì | Từ đâu |
|---|---|---|
| Fact cụ thể với X đại diện (`Aaron_Brooks <teammate> Randolph`) | Model không xử lý được biến X trừu tượng | Lấy hub đầu tiên của mẫu |
| "X không tham gia, không rời đi" | Model tưởng X cũng chuyển đội, cắt cả `(X, player, Nuggets)` | X không nằm trong ops tường minh |
| "P vừa tách khỏi Q (qua r0)" | Model không nối được fact với thay đổi vừa áp | Op INVALIDATE tường minh có chứa participant, với Q là co-anchor của mẫu |
| "trong KG: N% cặp r cùng chung hàng xóm qua r0" | Model không biết r có phụ thuộc r0 không | Đếm trên KG-trước, xem dưới |

**Đồng xuất hiện** (`_cooccur`): lấy mọi cạnh `(u, r, v)` trong KG-trước. Đếm tỷ lệ cặp mà u và v **cùng chung một hàng xóm** qua r0. Với r = `<teammate>`, r0 = `<player>`: 100% cặp đồng đội cùng có chung một đội. Với r = `<player>`, r0 = `<player>`: 0%, vì cầu thủ và đội không "cùng thuộc" một đội thứ ba. Con số này nói "teammate là quan hệ đi kèm cùng-đội, player thì không", và system prompt dạy model đọc: N cao thì fact phụ thuộc cạnh vừa vô hiệu, thường sai; N thấp thì r độc lập, thường giữ.

**Tác dụng đo.** Đây là chỗ khó nhất, và bảng probe cho thấy các cách diễn đạt **kéo nhau**:

| Biến thể | cut teammate đúng | cut player SAI |
|---|---|---|
| v1 dòng dài | 42/44 | 39/44 |
| + fact cụ thể "còn đúng?" | 28/44 | 11/44 |
| + "P vừa tách khỏi Q" + quy ước trạng thái | 40/44 | 18/44 |
| + "X không tham gia, không rời đi" | 28/44 | 4/44 |
| + đồng xuất hiện N% | 30/44 | 26/44 |
| + khối 3–4 dòng, mỗi mẫu một lệnh gọi | 44/44 | 21/44 |

Nhấn "X không rời đi" thì player được giữ (4/44 sai) nhưng model giữ luôn teammate (28/44). Nhấn "P tách khỏi Q" thì teammate cắt tốt (40/44) nhưng player cắt nhầm nhiều hơn (18/44). Không biến thể nào đạt cả hai. Biến thể "P vừa tách khỏi Q" là bản đang dùng cho số pipeline 0,973.

---

## 4. Chia lô `batch_size`, mặc định mỗi lệnh gọi một mẫu

**Trước.** Cả pha CẮT là một lệnh gọi với 4 mẫu (idx=1), hoặc 12 mẫu (tin trade 4 cầu thủ idx=15). Model phải giữ nhiều mẫu trong đầu và trả về nhiều chỉ số cùng lúc. Tin idx=15 timeout 120 s.

**Sau.** `LLMJudge(batch_size=1)`: mỗi mẫu một lệnh gọi, chỉ số `[C0]` là cục bộ trong lô. Prompt cắt của idx=1 từ 2.218 ký tự (4 mẫu) còn 1.239 ký tự (1 mẫu). `batch_size=None` giữ hành vi cũ, `--judge-batch N` trên CLI.

```
MẪU CẮT (mỗi mẫu một fact đại diện; phán quyết áp cho cả nhóm X; mặc định KEEP):
[C0] "Aaron_Brooks_2014-15 <teammate> Anthony_Randolph_2014-15" — đại diện cho nhóm 15 X: ...
  - X = Aaron_Brooks_2014-15: không tham gia, không rời đi
  - Anthony_Randolph_2014-15 vừa tách khỏi Denver_Nuggets (qua <player>)
  - trong KG: 100% cặp <teammate> cùng chung hàng xóm qua <player>

Sau bản tin, fact nào KHÔNG còn đúng? Trả về invalidate = các số i của [Ci] đó.
```

**Tác dụng đo.** Tốt nhất ở teammate (44/44) và add (11/11), nhưng player vẫn 21/44 sai. Giá: 154 lệnh gọi cho 20 tin thay vì khoảng 40, thời gian 10,6 phút thay vì 2 phút. Với continuous batching 4 slot, chi phí này chịu được cho n=20 nhưng toàn split 438 tin ước 4 giờ.

Một quan sát: hỏi riêng từng mẫu làm model **mất so sánh**. Khi thấy C0 và C2 cạnh nhau, model có thể nhận ra "hai cái này khác nhau ở quan hệ". Hỏi riêng thì mỗi mẫu tự đứng, và mẫu player một mình trông rất giống mẫu teammate một mình.

---

## 5. Cổng cấu trúc `--judge-gate` (phương án C, đã code, đo sơ bộ trong bài này)

**Ý tưởng.** Hai con số đo được ở mục 2 và 3 (đồng xuất hiện và tương tự) đã nói khá rõ mẫu nào đáng nghi. Vậy lọc trước bằng cấu trúc, LLM chỉ phán vùng xám:

- Mẫu cắt: đồng xuất hiện (r, r0) **dưới ngưỡng** → KEEP luôn, không hỏi. Trên ngưỡng → hỏi LLM.
- Mẫu thêm: **không có** dòng "tương tự" nào đạt 50% → bỏ qua luôn. Có → hỏi LLM.

```python
keep_cut = [p for p in cut if self._cut_dependency(p, ctx) >= self.gate]
keep_add = [p for p in add if any("%" in a for a in self._analogy(p, ctx))]
```

Trên idx=1 với `gate=0.5`: dependency của 4 mẫu cắt là `[1.0, 1.0, 0.0, 0.0]`, nên C2 và C3 (player) KEEP không hỏi, chỉ C0 và C1 (teammate) lên LLM. Mẫu thêm chỉ A0 lên LLM, A1/A2/A3 bỏ.

**Đo mới hôm nay** (không gọi LLM, chỉ so cổng với đáp án vàng, n=20):

| Loại mẫu | Đáp án vàng | Cổng 0,5 quyết định | Số mẫu |
|---|---|---|---|
| cắt `<player>` | giữ | KEEP, không hỏi | 44/44 |
| cắt `<teammate>` | cắt | đẩy lên LLM | 44/44 |
| thêm, có cạnh đúng | thêm | đẩy lên LLM | 11/11 |
| thêm, không có | bỏ | bỏ, không hỏi | 29/29 |

Cổng **tách đúng 100%** trên 20 tin. Hệ quả:

- Mẫu player, chỗ duy nhất Gemma 4 dao động (4/44 đến 39/44 sai), **không bao giờ đến LLM nữa**. Lỗi player kỳ vọng về 0.
- LLM chỉ còn thấy 44 mẫu teammate (đang đúng 44/44 khi hỏi riêng) và 11 mẫu thêm (đang đúng 11/11).
- Số lệnh gọi từ 128 mẫu xuống 55 mẫu, giảm 57%.

Nếu số này giữ được khi chạy thật, pipeline sẽ tiến sát cận trên oracle (1,0) trên n=20.

**Cảnh báo cần ghi vào khóa luận.** Trên miền NBA với 4 quan hệ, cổng mạnh tới mức LLM gần như chỉ còn xác nhận: mọi mẫu qua cổng đều là "cắt" hoặc "thêm". Điều này không có nghĩa LLM thừa. Miền có quan hệ đồng xuất hiện 40–60% (bạn thân, cố vấn) mới là vùng xám thật, và ở đó cổng không quyết được. Nhưng phải nói thẳng: trên dataset này, đóng góp của LLM sau cổng chưa được chứng minh tách biệt. Ablation "cổng + luôn cắt/luôn thêm, không LLM" là thí nghiệm bắt buộc.

---

## 6. `scripts/probe_judge.py`: chấm ở mức mẫu

**Vấn đề.** Trước đây muốn biết một prompt tốt hơn không thì phải chạy full pipeline rồi nhìn F1/add_acc. Số này gộp mọi thứ nên không biết prompt hỏng ở mẫu nào, và chạy 2–8 phút mỗi vòng.

**Sau.** Script chạy từng tin đến bước khoanh vùng, gom mẫu, gọi LLM đúng như bộ phán thật, rồi **suy đáp án vàng cho từng mẫu từ KG-sau**:

- mẫu cắt là "cần cắt" nếu đa số cạnh thành viên không còn trong KG-sau
- mẫu thêm có đáp án là tập (r, chiều) mà đa số thành viên có trong KG-sau

In bảng TP/FN/FP/TN theo loại mẫu (quan hệ + chiều). Dạng bảng như sau, số điền lại từ biến thể cuối trong báo cáo Gemma 4 (44 teammate đúng, 21/44 player sai, 11/11 add), không phải bản in nguyên của một lần chạy:

```
mẫu                                    TP   FN   FP   TN
add  via <player> neighbor_first       11    0    0    x
add  via <head_coach> ...               0    0    0    x
add  via <teammate> ...                 0    0    0    x
cut  <player> neighbor_first            0    0   ~10  ~12   ← chỗ hỏng
cut  <player> subject_first             0    0   ~11  ~11
cut  <teammate> neighbor_first         22    0    0    0
cut  <teammate> subject_first          22    0    0    0
add: đúng cả (r, direction) khi gold có: 11 / 11
```

Nhìn một bảng biết ngay: teammate và add xong, player còn 21 FP. Mọi con số 44/44, 11/11, 21/44 trong bảng thử prompt ở mục 3 và 4 đều từ script này. Một vòng thử prompt 2 phút (cả pha một lệnh gọi) đến 10 phút (mỗi mẫu một lệnh gọi) với 4 luồng.

Lưu ý: bảng có ±2 mẫu lệch giữa hai lần chạy cùng cấu hình vì continuous batching không cho kết quả bit-exact.

---

## 7. Tổng hợp: cái gì kéo số lên

| Thay đổi | Nhắm vào | Tác dụng đo được |
|---|---|---|
| 1. `reasoning` | model quyết định quá sớm | (đo chung với 2) |
| 2. tương tự | pha SINH chọn sai quan hệ/chiều | add 4/11 → 9/11 → 11/11; add_acc 0,42 → 0,82 |
| 3. fact cụ thể + tách khỏi + đồng xuất hiện | pha CẮT nhầm player | teammate 44/44; player còn dao động 4–39/44 |
| 4. batch_size=1 | prompt dài, timeout | teammate và add lên tối đa; chậm 5 lần |
| 5. cổng cấu trúc | player | chưa chạy LLM; cổng tách đúng 100% trên n=20 |
| 6. probe_judge | vòng thử chậm, mù | mỗi vòng 2–10 phút, thấy đúng loại mẫu hỏng |

Chuỗi nhân quả: probe (6) cho thấy pha SINH hỏng vì chọn quan hệ → tương tự (2) sửa xong. Probe cho thấy pha CẮT hỏng ở player → thử 7 cách diễn đạt (3, 4), không cách nào ổn → hai con số cấu trúc đã tính sẵn trong (2) và (3) đủ mạnh để làm cổng (5).

---

## 8. Việc tiếp theo, theo thứ tự

1. Chạy `run_eval.py --judge-gate 0.5 --judge-batch 1 --limit 20` để có số pipeline thật với cổng. Kỳ vọng add_acc và del_acc đều trên 0,95.
2. Chạy ablation "cổng, không LLM" (cổng qua thì cắt/thêm luôn) để tách đóng góp của LLM.
3. Thử `gate` ở 0,3 và 0,7 để biết ngưỡng có nhạy không.
4. n=50 rồi toàn split.
