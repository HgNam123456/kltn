# Kế hoạch (bản sao từ Notion, trang "🎯 kế hoạch" — sửa lần cuối 2026-09-05)

Nguồn: https://app.notion.com/p/3d2c4cb599a981d086c4f87bca8e362e (trong "Update data for knowledge graph")

# Đề tài
**Nghiên cứu phương pháp ứng dụng mô hình ngôn ngữ lớn trong bài toán tự động cập nhật cơ sở dữ liệu (knowledge graph)**

Hệ thống nhận **bản tin + KG hiện tại** → LLM trích các thao tác thay đổi tường minh → khoanh vùng cấu trúc để tìm các cạnh bị ảnh hưởng gián tiếp → bộ phán quyết định KEEP / INVALIDATE / ADD → áp lên KG theo cơ chế bi-temporal (không xóa, chỉ vô hiệu hóa có dấu thời gian).

Mục tiêu: **general — dùng cho nhiều miền data**, nạp miền mới không phải viết code/luật tay.

# Kiến trúc pipeline (bản chốt sau các vòng tranh luận)
```
Tin mới
  │
  ▼
[1] LLM #1 (Qwen2.5-7B + constrained decoding)
    → ops TƯỜNG MINH trên fact gốc
    vd: INVALIDATE chơi_cho(Messi, Barca)
        ADD        chơi_cho(Messi, Inter)
  │
  ▼
[2] Executor bi-temporal (Neo4j)
    áp ops TRƯỚC → cấu trúc mới hiện hình trong graph
    INVALIDATE = đóng dấu valid_to, KHÔNG xóa
  │
  ▼
[3] KHOANH VÙNG CẤU TRÚC (2 vế, không cần luật/schema)
    • Vế CẮT:  cạnh (a–b) bị vô hiệu → ứng viên = hàng xóm CHUNG của a và b
               (entity nối trực tiếp với ≥ 2 entity tham gia sự kiện)
    • Vế SINH: cạnh (a–b) mới thêm → ứng viên = hàng xóm của b chưa nối với a
               (thành viên của "bối cảnh mới": Lautaro, HLV Inter...)
  │
  ▼
[4] BỘ PHÁN (3 phương án → so sánh thực nghiệm)
    A. LLM #2: 1 lệnh gọi duy nhất, phán KEEP/INVALIDATE/ADD cho cả danh sách
    B. Link prediction INDUCTIVE: AnyBURL (luật + confidence) / ULTRA (zero-shot)
       → ngưỡng conf ≥ 0.9 thì áp, vùng xám đẩy cho LLM
    C. Lai A+B
  │
  ▼
[5] Áp toàn bộ vào KG → so với KG-sau (gold) để đo
```

# Các quyết định thiết kế đã chốt (và lý do)
- **Không giới hạn lan truyền bằng số hop** — khoanh vùng theo cấu trúc sự kiện: hàng xóm chung (tam giác trên cạnh bị đổi) cho vế cắt + lân cận bối cảnh mới cho vế sinh. Lọc "≥ 2 kết nối tới entity tham gia" loại được false positive một-chân (nhà tài trợ, sân vận động, người thân).
- **Khoanh vùng chỉ trả lời "nhìn ở đâu", chưa trả lời "đổi cái gì"** → bắt buộc có bộ phán per cạnh (ví dụ Pedri: teammate–Messi phải cắt, chơi_cho–Barca phải giữ, bạn_thân–Messi phải giữ — 3 cạnh cùng vùng, 3 số phận).
- **Áp ops tường minh TRƯỚC khi chạy link prediction** — để predictor đọc được cạnh Messi–Inter mới.
- **Bẫy embedding transductive:** TransE/RotatE học sẵn sẽ "nghĩ" Messi vẫn thuộc Barca (embedding cũ) → phải dùng predictor inductive/path-based (AnyBURL, ULTRA) đọc cấu trúc tại thời điểm suy diễn. Đây cũng là lý do không dùng KGE tĩnh làm bộ phán.
- **Link prediction trả về độ hợp lý, không phải sự thật** → chỉ chấm trong vùng ứng viên + ngưỡng conf cao + vùng xám đi qua LLM. Không bao giờ thêm top-k mù quáng.
- **Không xóa, chỉ vô hiệu hóa** (valid_from / valid_to) → truy vấn được lịch sử ("tháng 5 Messi đá cho đội nào?"), audit được, học theo Zep/Graphiti.
- **Fact quá khứ không được vô hiệu** (ghi_672_bàn_cho(Messi, Barca) đúng mãi mãi) — bộ phán phải phân biệt fact trạng thái vs fact sự kiện lịch sử.
- **Ngoài phạm vi:** tự đoán sự kiện thế giới chưa xảy ra (ai thay vị trí Messi ở Barca) = ảo giác có tổ chức, không làm; chỉ ghi điều văn bản khẳng định + điều suy ra được từ cấu trúc.

# Dataset & mốc so sánh
## 1. NBAtransactions — miền neo, có baseline công bố
Nguồn: paper GUpdate — Tang et al., "Learning to Update Knowledge Graphs by Reading News", EMNLP 2019 · https://aclanthology.org/D19-1265/ · code + data: https://github.com/esddse/GUpdater
- 4.100 cặp (tin chuyển nhượng NBA 2010–2019, KG-trước, KG-sau)
- Chia sẵn: **3.261 train / 417 valid / 422 test**
- Trung bình mỗi tin: 29,1 từ · **17,07 cạnh thêm + 18,37 cạnh xóa**
- Chỉ **5,6%** thay đổi cạnh được nói thẳng trong text → 94,4% là hệ quả phải suy ra — đây chính là lý do cần khoanh vùng + bộ phán

## Số của GUpdate phải so (mục tiêu: tiệm cận hoặc vượt)
| Chỉ số | GUpdate (GNN, 2019) | Baseline trích xuất thuần (IE + R-GCN) |
|---|---|---|
| F1 tái dựng đồ thị | **0,9866** | 0,9429 |
| Accuracy cạnh THÊM | **0,8926** | 0,4681 |
| Accuracy cạnh XÓA | **0,8988** | 0,2448 |

Đọc bảng: khoảng cách khổng lồ ở cột xóa (0,90 vs 0,24) chứng minh: không có cơ chế lan truyền thì trích xuất giỏi mấy cũng vô dụng với bài toán update.

## 2. WikiFactDiff — đánh giá tính general đa miền
- Orange Labs, LREC-COLING 2024 · https://aclanthology.org/2024.lrec-main.1532/ · arXiv 2403.14364
- GitHub: https://github.com/Orange-OpenSource/WikiFactDiff · HuggingFace: Orange/WikiFactDiff
- Ghi tiến hóa fact thật giữa 2 bản dump Wikidata → cập nhật đa miền thật
- Vốn làm cho knowledge editing trong LLM → **tái mục đích cho KG update = điểm mới phụ**
- ⚠️ Việc cần làm trước: tải thử, kiểm tra dựng được cặp (văn bản mô tả thay đổi, KG-trước, KG-sau) không

## 3. Dataset tiếng Việt tự xây (tùy chọn, điểm cộng)
- ~200–300 tin chuyển nhượng bóng đá từ báo VN, gán KG-trước/sau bán tự động (LLM nháp + duyệt tay)
- Chứng minh Bootstrap chạy được trên **ngôn ngữ mới**, không chỉ miền mới

# Kế hoạch thực nghiệm
**Bảng 1 — so sánh hệ thống (trên NBAtransactions test)**
- GUpdate (số từ paper)
- Baseline IE thuần (số từ paper)
- LLM zero-shot (không khoanh vùng, bắt LLM xuất hết) — cận dưới
- Pipeline đầy đủ: LLM ops + khoanh vùng + bộ phán A/B/C (3 dòng)

**Bảng 2 — so sánh bộ phán (ablation trung tâm)**
- Vế cắt: LLM phán vs không lan truyền
- Vế sinh: LLM vs AnyBURL vs ULTRA vs lai
- Đo: acc cạnh thêm / cạnh xóa / F1 / số lệnh gọi LLM / thời gian

**Bảng 3 — tính general (leave-one-domain-out)**
- Phát triển toàn bộ trên NBA → chạy nguyên trạng trên WikiFactDiff + tiếng Việt, đo độ suy giảm
- Metric riêng: **chi phí nạp miền mới** (phút người + số lệnh gọi LLM, mục tiêu: 0 dòng code)

**Fine-tune (nếu kịp) — chú ý để giữ tính general**
- KHÔNG QLoRA riêng trên NBA (dạy model miền NBA = phản mục tiêu)
- Instruction-tune trên hỗn hợp miền, test trên miền bị giấu — dạy định dạng nhiệm vụ, không dạy nội dung miền
- Nhãn sinh tự động từ diff KG-trước/sau, không gán tay

# Công cụ
- **LLM:** Qwen2.5-7B-Instruct (đa ngôn ngữ) · QLoRA 4-bit qua Unsloth trên Colab T4
- **JSON sạch:** constrained decoding bằng `outlines` hoặc `xgrammar`
- **KG store:** Neo4j Community (hoặc KuzuDB embedded lúc dev) · cạnh mang valid_from/valid_to · constraint unique id trước khi nạp
- **Link prediction:** AnyBURL (CPU, vài phút) · ULTRA checkpoint (zero-shot, không train)
- **Bản gốc dữ liệu:** mọi triple + ops lưu JSONL kèm provenance (src, quote, conf) — Neo4j chỉ là bản chiếu, xóa dựng lại được trong vài phút

# Lộ trình 4 tháng
| Tháng | Việc | Mốc kiểm chứng |
|---|---|---|
| 1 | Dựng KG NBA + executor bi-temporal + khoanh vùng 2 vế (thuần code, chưa LLM). Chạy với **ops vàng** lấy từ diff | Nếu F1 cao với ops vàng → kiến trúc đúng |
| 2 | LLM #1 zero/few-shot + bộ phán LLM → số baseline đầu tiên; phân tích lỗi theo loại | Có bảng 1 sơ bộ |
| 3 | Bộ phán B (AnyBURL/ULTRA) + ablation; fine-tune nếu kịp | Có bảng 2 |
| 4 | WikiFactDiff + dataset tiếng Việt + demo Streamlit + viết báo cáo | Có bảng 3, demo chạy được |

# Rủi ro & đường lui
- LLM 7B xuất ops kém → few-shot + retry + constrained decoding; tệ nữa thì fine-tune
- WikiFactDiff không dựng được cặp trước/sau → tự tạo snapshot Wikidata cho 2–3 loại quan hệ; hoặc chấp nhận 2 miền
- Không kịp fine-tune → zero/few-shot + khoanh vùng + so GUpdate đã đủ một khóa luận trọn vẹn
- Dataset tiếng Việt tốn công gán → cắt được mà không thủng cốt truyện

# Ghi chú kỹ thuật bổ sung (khảo sát repo GUpdater, 2026-09-07)
- `data/NBAtransactions_{train,valid,test}.json`: list các object `{event, season, text, text_mentioned_entities, subgraph_before, subgraph_after}`; `subgraph_*` là list `[h, r, t]` theo id; `text` đã map sang token id.
- `data/entity2id.txt`, `data/token2id.txt`: dòng đầu là số đếm, các dòng sau `name<TAB>id`. `data/relation2id.txt`: 4 quan hệ `<general_mananger> 0, <player> 1, <teammate> 2, <head_coach> 3`.
- Metric GUpdate: added = after − before, deleted = before − after, unchanged = before ∩ after; F1 tính trên toàn bộ triple với nhãn 1 cho added/unchanged, 0 cho deleted.
- Máy dev: Windows 11, Python 3.11.7, không có GPU NVIDIA, có Java 11 (đủ chạy AnyBURL). LLM 7B phải chạy qua endpoint OpenAI-compatible (Colab+vLLM, OpenRouter/DashScope) hoặc Ollama CPU.
