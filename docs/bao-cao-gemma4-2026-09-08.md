# Báo cáo: chuyển sang Gemma 4 E4B, tối ưu server, bộ phán theo mẫu

Ngày: 2026-09-08 (đêm) · Nhánh `feat/core-pipeline`, HEAD 7b82f9a, 68 test pass
Bổ sung cho bản "Báo cáo tiến độ 2026-09-08" (`docs/bao-cao-tien-do-2026-09-08.md`). Chỉ ghi những gì thay đổi từ bản đó.

---

## 1. Kết luận ngắn

- **Đổi model 2B → Gemma 4 E4B** là bước ngoặt. Với ops tường minh vàng và bộ phán LLM, trên 20 tin: **F1 0,973, add_acc 0,824, del_acc 0,920**, không tin nào lỗi. GUpdate: 0,987 / 0,893 / 0,899. Cạnh xóa đã vượt GUpdate, cạnh thêm còn kém 0,07, F1 kém 0,014.
- **Ba tối ưu server bạn yêu cầu đều đã bật**: flash attention, speculative decoding (đầu MTP của Gemma 4), continuous batching 4 slot. 20 tin chạy 133–168 giây thay vì 465 giây với model 2B, dù model lớn gấp đôi và mỗi tin gọi 2 lần thay vì 1.
- **Bộ phán chỉ còn một điểm yếu có hệ thống**: mẫu "đồng đội cũ vẫn thuộc đội cũ" bị cắt nhầm. Model trả lời đúng 44/44 mẫu teammate và 11/11 mẫu thêm, nhưng dao động từ 4/44 đến 39/44 sai ở mẫu player tùy cách diễn đạt prompt. Đã thử 7 biến thể prompt, không biến thể nào ổn định cả hai. Đã viết sẵn cổng cấu trúc (phương án C trong spec) để lọc mẫu này trước khi hỏi LLM, chưa chạy đánh giá vì bạn yêu cầu dừng.

---

## 2. Model và server

**Model.** `unsloth/gemma-4-E4B-it-qat-GGUF`, file `gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf` (4,2 GB) và đầu dự đoán đa token `mtp-gemma-4-E4B-it.gguf` (60 MB). Chọn bản QAT (quantization-aware training) vì chất lượng gần BF16 mà chạy 4-bit, nhanh nhất trên Vulkan. llama.cpp b10839 nhận diện kiến trúc Gemma 4 và MTP không cần cập nhật.

**Server** (`scripts/llm_server.ps1`, mặc định mới):

| Tối ưu | Flag | Tác dụng đo được |
|---|---|---|
| Flash attention | `--flash-attn on` | Đã bật từ trước; KV cache gọn, đọc prompt nhanh hơn |
| Speculative decoding | `--spec-type draft-mtp --model-draft mtp-*.gguf --spec-draft-n-max 2` | Sinh **35 tok/s** so với 20 tok/s của Qwen 2B không speculative; model 4B mà nhanh hơn 2B |
| Continuous batching | `--parallel 4 --cont-batching --kv-unified-per-slot 8192` | 4 tin xử lý đồng thời; runner gửi 4 luồng (`run_eval --workers 4`) |

Đo smoke test: đọc prompt 110 tok/s, sinh 35 tok/s. Ngay lệnh gọi đầu Gemma 4 đã trích đúng op "Pacers waived Young" mà Qwen 2B chưa bao giờ làm được.

**Runner.** `scripts/run_eval.py --workers N`: mỗi luồng có extractor và judge riêng (client riêng) nên bộ đếm lệnh gọi từng tin không lẫn; thứ tự ghi JSONL giữ nguyên. Thêm `KGU_LLM_TIMEOUT` (mặc định 300 s), `--judge-batch`, `--judge-gate`.

Lưu ý: batching song song không cho kết quả bit-exact giữa các lần chạy; probe cùng cấu hình lệch ±2 mẫu. So sánh biến thể cần nhìn xu hướng, không nhìn một con số.

---

## 3. Bộ phán: những gì đã đổi

Bản trước (từng cạnh) → bản 2 (mẫu, 2 pha) đã có trong báo cáo cũ. Hôm nay thêm:

1. **`reasoning` trong schema**, đứng trước quyết định, để model suy luận ngắn rồi mới trả chỉ số.
2. **Bằng chứng "tương tự" cho mẫu thêm**: đếm trong KG hiện có xem các entity S cùng vai với subject (cùng loại cạnh tới đội mới) đang có quan hệ gì với X. Ví dụ "(S, teammate, X) 95%". Không luật miền, chỉ đếm cạnh. Đây là thứ đưa pha SINH từ 4/11 lên 11/11.
3. **Mẫu cắt dạng fact cụ thể**: thay "(X, player, Nuggets)" bằng khối 3–4 dòng:
   ```
   [C2] "Aaron_Brooks <player> Denver_Nuggets" — đại diện cho nhóm 15 X: ...
     - X = Aaron_Brooks: không tham gia, không rời đi
     - Denver_Nuggets vừa tách khỏi Anthony_Randolph (qua <player>)
     - trong KG: 0% cặp <player> cùng chung hàng xóm qua <player>
   ```
   Dòng cuối là **đồng xuất hiện**: tỷ lệ cặp có quan hệ r mà cũng chung một hàng xóm qua quan hệ vừa bị vô hiệu. Trong KG này 100% cặp teammate chung đội, 0% cặp player. Đây là tín hiệu "r có phụ thuộc vào thay đổi không", đo từ KG-trước.
4. **Chia lô** (`batch_size`, mặc định 1): mỗi lệnh gọi một mẫu, theo đề xuất của bạn. Prompt ngắn nhất, chỉ số cục bộ.
5. **Cổng cấu trúc** (`gate`, tùy chọn, chưa đánh giá): mẫu cắt có đồng xuất hiện dưới ngưỡng → KEEP không hỏi LLM; mẫu thêm không có "tương tự" → bỏ qua. LLM chỉ phán vùng xám. Đây là phương án C trong spec.
6. `scripts/probe_judge.py`: chấm ở mức mẫu so với đáp án oracle, in bảng TP/FN/FP/TN theo loại mẫu. Một vòng thử prompt mất 2–10 phút thay vì chạy full pipeline.

---

## 4. Kết quả

### 4.1 Pipeline (gold_explicit + LLM judge, n=20, split test)

| Cấu hình | F1 | add_acc | del_acc | lỗi | giây |
|---|---|---|---|---|---|
| Qwen3.5-2B, mẫu 1 pha (báo cáo cũ, tốt nhất) | 0,923 | 0,287 | 1,000* | 1 | 466 |
| Gemma 4 E4B, mẫu 2 pha, prompt v1 | 0,939 | 0,420 | 0,952 | 0 | 133 |
| **Gemma 4 E4B, + reasoning + tương tự + fact cụ thể + tách khỏi** | **0,973** | **0,824** | **0,920** | 0 | 168 |
| GUpdate (paper) | 0,987 | 0,893 | 0,899 | | |

\* del_acc 1,0 của 2B là do cắt hết, vô nghĩa.

### 4.2 Nhật ký thử prompt ở mức mẫu (probe, n=20; 44 mẫu teammate, 44 mẫu player, 11 mẫu thêm có đáp án)

| Biến thể | cut teammate đúng | cut player SAI | add đúng (r, hướng) | thời gian |
|---|---|---|---|---|
| v1 dòng dài, không reasoning | 42/44 | 39/44 | 4/11 | 2 phút |
| + reasoning + tương tự | 44/44 | 30/44 | 9/11 | |
| + fact cụ thể "còn đúng?" | 28/44 | 11/44 | 9/11 | |
| + "P vừa tách khỏi Q" + quy ước trạng thái | 40/44 | 18/44 | 10/11 | → pipeline 0,973 |
| + "X = ... không tham gia, không rời đi" | 28/44 | 4/44 | 10/11 | |
| + đồng xuất hiện N% | 30/44 | 26/44 | 8/11 | |
| + khối 3–4 dòng, **mỗi mẫu một lệnh gọi** | **44/44** | 21/44 | **11/11** | 10,6 phút |

Đọc bảng: pha SINH đã giải xong (11/11) nhờ bằng chứng "tương tự". Mẫu teammate cũng xong khi hỏi riêng từng mẫu (44/44). Mẫu player là chỗ model dao động: mọi cách diễn đạt nhấn "X không rời đi" thì giữ được player nhưng lại giữ nhầm teammate, và ngược lại. Model 4B chưa tách bạch được hai câu hỏi này khi chúng có hình thức giống hệt nhau về cấu trúc.

Chia lô mỗi mẫu một lệnh gọi cho kết quả tốt nhất ở teammate và add nhưng chậm gấp 5 (154 lệnh gọi cho 20 tin). Toàn split 438 tin ước 4 giờ.

---

## 5. So với GUpdate, cập nhật

| | F1 | add | del | học từ dữ liệu? |
|---|---|---|---|---|
| GUpdate | 0,987 | 0,893 | 0,899 | có, 3.262 tin |
| IE thuần (paper) | 0,943 | 0,468 | 0,245 | có |
| Mình, không lan truyền | 0,926 | 0,056 | 0,056 | không |
| Mình, cận trên oracle | 1,000 | 1,000 | 1,000 | không |
| Mình, Qwen 2B tốt nhất (n=20) | 0,923 | 0,287 | 1,000* | không |
| **Mình, Gemma 4 E4B tốt nhất (n=20)** | **0,973** | **0,824** | **0,920** | không |

Lần đầu có số LLM đứng được cạnh GUpdate. Khoảng cách còn lại nằm gọn ở một loại mẫu, và có hai đường xử lý rõ ràng: cổng cấu trúc (đã code, chưa đo) hoặc model lớn hơn.

Lưu ý so sánh: n=20 của mình so với 438 của paper, và universe F1 của mình chặt hơn. Cần chạy toàn split trước khi đưa vào khóa luận.

---

## 6. Chưa xong / rủi ro

1. **Cổng cấu trúc chưa đánh giá.** Code và test có (68 test), chưa chạy trên model. Kỳ vọng: loại mẫu player khỏi LLM → cut player sai về 0, số lệnh gọi giảm khoảng 60%.
2. **Chỉ mới n=20.** Cần n=50 rồi toàn split; thời gian 1–4 giờ tùy batch.
3. **LLM #1 (trích ops) chưa chạy lại với Gemma 4.** Smoke test cho thấy triển vọng, chưa có số.
4. **Chưa cập nhật README và báo cáo tiến độ chính** với cấu hình Gemma 4.
5. Nhánh chưa merge.

---

## 7. Lệnh để chạy tiếp

```
.\scripts\llm_server.ps1                                              # Gemma 4 + MTP, 4 slot
python scripts/probe_judge.py --limit 20 --workers 4 --batch 1        # chấm mức mẫu
python scripts/run_eval.py --extractor gold_explicit --judge llm --limit 20 --workers 4 \
    --judge-batch 1 --judge-gate 0.5 --out results/x.jsonl            # pipeline, phương án C
python scripts/run_eval.py --extractor llm --judge llm --limit 20 --workers 4 --out results/y.jsonl
```
