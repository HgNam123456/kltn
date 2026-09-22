# Báo cáo: chạy toàn split test 438 tin với Gemma 4 E4B + cổng cấu trúc

Ngày: 2026-09-09 · Nhánh `feat/core-pipeline`, HEAD 7b82f9a
Bổ sung cho `docs/bao-cao-gemma4-2026-09-08.md` (n=20) và `docs/bao-cao-bo-phan-chi-tiet-2026-09-09.md` (6 thay đổi của bộ phán).
Kết quả: `results/nba_test_goldexp_gemma4_gate_438.jsonl` và `.summary.json`.

---

## 1. Kết luận ngắn

- **Lần đầu chạy ngang GUpdate**: toàn split test, 438 tin, không tin nào lỗi. Cấu hình: ops tường minh vàng, bộ phán Gemma 4 E4B, mỗi mẫu một lệnh gọi, cổng cấu trúc 0,5.
- **F1 0,993 và del_acc 0,997 vượt GUpdate** (0,987 / 0,899). **add_acc 0,840 kém GUpdate 0,05** (0,893).
- Khoảng cách add_acc nằm gọn ở **38/150 tin trade nhiều cầu thủ**. Toàn bộ 2.646 cạnh thêm bị thiếu đều là `<teammate>`. Nguyên nhân đã xác định bằng cách đọc `reasoning` của model: với trade 12–24 ops tường minh, model bị header ops kéo đi và bỏ qua khối mẫu ngay bên dưới. Không phải lỗi cổng, không phải timeout.
- Thời gian 2 giờ 4 phút, 1.428 lệnh gọi, trùng khớp con số đếm trước bằng code (không có retry).

---

## 2. Lệnh và cấu hình

```
.\scripts\llm_server.ps1        # Gemma 4 E4B QAT Q4_K_XL + MTP, flash-attn, 4 slot
python scripts/run_eval.py --extractor gold_explicit --judge llm --workers 4 \
    --judge-batch 1 --judge-gate 0.5 --out results/nba_test_goldexp_gemma4_gate_438.jsonl
```

Summary:

```
f1 0.9928  precision 0.9985  recall 0.9872
add_acc 0.8398  del_acc 0.9971
n 438  n_scored 438  errors 0  coverage 1.0
avg_cut_cand 70.3  avg_add_cand 31.0  judged_ratio 0.673
llm_calls 1428  seconds 7427
```

`judged_ratio` 0,673: cổng chặn 33% ứng viên không cần hỏi LLM.

---

## 3. So với GUpdate

Cùng dataset NBAtransactions, cùng split test, cùng ba chỉ số theo định nghĩa GUpdate. Paper ghi 422 tin, file thực tế trong repo GUpdater có 438; không xác định được 16 tin lệch.

| | F1 | add_acc | del_acc | học từ dữ liệu? | n |
|---|---|---|---|---|---|
| GUpdate (GNN) | 0,987 | 0,893 | 0,899 | có, 3.262 tin train | 422 |
| IE thuần + R-GCN (paper) | 0,943 | 0,468 | 0,245 | có | 422 |
| Mình, ops tường minh không lan truyền | 0,926 | 0,056 | 0,056 | không | 438 |
| Mình, cận trên oracle | 1,000 | 1,000 | 1,000 | không | 438 |
| Mình, Qwen3.5-2B tốt nhất | 0,923 | 0,287 | 1,000\* | không | 20 |
| Mình, Gemma 4 E4B, n=20 (báo cáo trước) | 0,973 | 0,824 | 0,920 | không | 20 |
| **Mình, Gemma 4 E4B + cổng 0,5** | **0,993** | 0,840 | **0,997** | không | **438** |

\* del_acc 1,0 của Qwen là do cắt hết, vô nghĩa.

Hai điểm khác biệt khi so:

- Universe tính F1 của mình là `before ∪ gold_after ∪ pred_after`, mọi cạnh dự đoán sai đều bị tính false positive. GUpdate chấm trên tập ứng viên cố định. Cách của mình chặt hơn, nên F1 0,993 không bị thổi phồng.
- GUpdate học end-to-end từ 3.262 tin train. Hệ thống này không học: khoanh vùng là cấu trúc thuần túy, cổng là hai con số đếm trên KG-trước, LLM không fine-tune.

---

## 4. Kết quả theo loại sự kiện

```
group              n      f1     add     del  cut/ex  add/ex
d_league           6   1.000   0.000   1.000    73.3     0.0
draft             18   1.000   1.000   0.000     0.0    20.0
free_agency      107   1.000   0.992   0.000     0.0    20.2
head_coach        10   1.000   1.000   1.000     0.0    19.5
overseas           3   1.000   0.000   1.000    62.7     0.0
released         139   1.000   0.000   0.996    71.3     0.0
retirement         5   1.000   0.000   1.000    61.6     0.0
trade            150   0.986   0.778   0.998   133.1    72.3
```

Các ô `0.000` ở add hoặc del là loại sự kiện không có cạnh thêm hoặc không có cạnh xóa (mẫu số 0), không phải lỗi. Đọc bảng: **7/8 loại sự kiện đạt F1 1,000**. Released (139 tin), free_agency (107 tin), draft, head_coach đều gần như hoàn hảo. Toàn bộ khoảng cách nằm ở trade.

---

## 5. Phân tích lỗi trade

### 5.1 Thiếu gì

Soi 150 tin trade trong JSONL, so `explicit_ops + judged_ops` với `after − before`:

| | Số |
|---|---|
| Cạnh thêm vàng trong trade | 11.912 |
| Thiếu | 2.646 (22%) |
| Thừa | 280 (2%) |
| Trade đúng tuyệt đối | 112/150 |
| Trade có thiếu | 38/150 |

Toàn bộ 2.646 cạnh thiếu là `<teammate>`, và đều là cạnh giữa một cầu thủ vừa chuyển đội với đồng đội mới. Mọi tin thiếu đều là trade nhiều cầu thủ (12–24 ops tường minh, tức 3–6 cầu thủ chuyển). Trade 1 cầu thủ (4 ops) không thiếu.

Các tin thiếu nặng nhất:

| idx | ops tường minh | cạnh thêm vàng | thiếu | ADD bộ phán trả về |
|---|---|---|---|---|
| 235 | 24 | 178 | 164 | 88 |
| 243 | 16 | 158 | 150 | 0 |
| 236 | 16 | 158 | 150 | 0 |
| 205 | 16 | 158 | 150 | 0 |
| 42 | 20 | 180 | 144 | 26 |

### 5.2 Cổng có chặn nhầm không

Không. Tin idx 243 (Cavaliers lấy Favors, Murphy, Clark từ Jazz; đổi Seth Curry): 118 cạnh thêm gom thành 14 mẫu. 4 mẫu có đáp án vàng (mỗi cầu thủ chuyển ~ cầu thủ đội mới) đều có "tương tự" 90–100% và **đều qua cổng**. 10 mẫu còn lại (GM, HLV, đồng đội cũ theo sang) đúng là cần bỏ và cổng bỏ. Số lệnh gọi của tin là 12 = 8 mẫu cắt + 4 mẫu thêm, không retry.

Vậy 4 lệnh gọi pha SINH đã đến LLM và LLM trả về `add=[]` cho cả 4.

### 5.3 Model lập luận gì

Gọi lại mẫu A1 của idx 243 hai lần (temperature 0), kết quả giống hệt:

Prompt phần mẫu:
```
[A0] Derrick_Favors_2014-15 ~ X — nhóm 20 X: Alonzo_Gee_2014-15, Anderson_Varejao_2014-15, ...
  - bằng chứng: (X, <player>, Cleveland_Cavaliers)
  - tương tự: (S, <teammate>, X) 90%, (X, <teammate>, S) 90%
```

Model trả về `add=[]` với reasoning:

> Cleveland_Cavaliers đã mua Derrick_Favors, Erik_Murphy, và Ian_Clark, do đó họ trở thành đồng đội của các cầu thủ này. Vì các cầu thủ này vừa gia nhập Cleveland_Cavaliers, chúng ta cần thêm quan hệ đồng đội hai chiều với các cầu thủ khác đang ở Cleveland_Cavaliers (nếu có trong KG, **nhưng không có thông tin về các cầu thủ khác tại Cleveland_Cavaliers** ngoài những người được đề cập). [...] Vì không có thông tin về các cầu thủ khác tại Cleveland_Cavaliers, không có bằng chứng nối X với các cầu thủ khác tại Cleveland_Cavaliers để thêm quan hệ đồng đội.

Model biết luật đúng ("cần thêm đồng đội hai chiều"), nhưng nói "không có thông tin về cầu thủ khác tại Cavaliers" **trong khi khối mẫu ngay dưới liệt kê 20 X với bằng chứng (X, player, Cavaliers)**. Nó lập luận trên 16 dòng "THAY ĐỔI TƯỜNG MINH ĐÃ ÁP" ở header và không đọc khối mẫu. Mẫu A10 (Seth Curry ~ 15 cầu thủ Jazz, tương tự 100%) cũng bị bỏ với lý do tương tự, thậm chí reasoning nói về Cavaliers chứ không nói về Jazz.

Với trade 1 cầu thủ, header chỉ 4 dòng, model đọc đúng khối mẫu (idx 1: 44/44 cạnh thêm đúng). Đây là vấn đề **phân bổ chú ý** của model 4B khi header dài, không phải thiếu bằng chứng.

Một chi tiết phụ: "tương tự" ở đây là 90% chứ không phải 100% vì siblings S được lấy tại `at=AFTER`, gồm cả hai cầu thủ vừa vào cùng lúc (Murphy, Clark) chưa có cạnh teammate nào. Không phải nguyên nhân chính, nhưng nên lấy siblings tại `at=BEFORE` để con số sạch.

---

## 6. Cách sửa đề xuất, theo thứ tự rẻ dần

1. **Header pha SINH chỉ in ops liên quan đến subject của mẫu.** Với `batch_size=1`, mẫu "Favors ~ X" chỉ cần 2 dòng ops của Favors (rời Jazz, vào Cavaliers), không cần 16 dòng. Model hết bị nhiễu, prompt ngắn hơn. Sửa `_header` trong `kgu/judge/llm.py`, thêm tham số subject.
2. **Viết mẫu thêm dạng fact cụ thể** như pha CẮT đã làm: `"Alonzo_Gee <player> Cleveland_Cavaliers" đang đúng; Derrick_Favors vừa vào Cleveland_Cavaliers cùng vai <player>`. Mục 3 của báo cáo bộ phán cho thấy fact cụ thể giúp model 4B hơn biến X trừu tượng.
3. **Siblings tại `at=BEFORE`** để "tương tự" không bị pha loãng bởi cầu thủ vào cùng lúc.
4. Fallback cấu trúc: mẫu thêm có "tương tự" ≥ 90% mà LLM bỏ → ghi log để đếm, chưa tự thêm. Chỉ dùng nếu 1–3 không đủ, vì nó làm mờ vai trò của LLM.

Cách kiểm tra: `probe_judge.py` trên riêng 38 tin trade lỗi (danh sách idx lấy từ JSONL), khoảng 10 phút, không cần chạy lại toàn split. Nếu add đúng, chạy lại toàn split 2 giờ.

---

## 7. Điểm cần ghi rõ trong khóa luận

1. **Ablation "cổng, không LLM" chưa chạy.** Trên miền 4 quan hệ này, cổng tách đúng 100% mẫu trên n=20. Cần đo xem nếu mẫu qua cổng thì cắt/thêm luôn, không hỏi LLM, số ra bao nhiêu. Nếu ngang bằng, đóng góp của LLM trên dataset này chưa được chứng minh tách biệt. Lỗi ở mục 5 lại cho thấy ngược lại: chính LLM đang kéo add_acc xuống, còn cổng thì đúng.
2. **LLM #1 (trích ops tường minh) chưa chạy với Gemma 4.** Mọi số trong bài này dùng ops tường minh vàng. Số end-to-end thật (LLM trích + LLM phán) chưa có.
3. 438 tin so với 422 của paper; universe F1 chặt hơn.
4. Batching song song không bit-exact; hai lần chạy cùng cấu hình có thể lệch vài mẫu.

---

## 8. Việc tiếp theo

1. Sửa header và render mẫu thêm (mục 6.1–6.3), probe trên 38 trade lỗi.
2. Chạy lại toàn split nếu probe tốt. Kỳ vọng add_acc trên 0,95.
3. Chạy ablation cổng-không-LLM trên toàn split (không tốn LLM, vài phút).
4. Chạy LLM #1 với Gemma 4, n=50, rồi end-to-end.
5. Cập nhật README, Notion, merge nhánh.
