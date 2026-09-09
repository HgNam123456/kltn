# Báo cáo dataset NBAtransactions (GUpdater)

Ngày: 2026-09-09 · Số liệu đo trực tiếp trên file trong `data/raw/nba/` (3 split, 4.100 tin), không lấy từ paper.
Script đo: `scripts/inspect_nba.py` và các script khảo sát trong phiên làm việc này.

---

## 1. Trả lời nhanh bốn câu hỏi

**Dataset có gì.** 4.100 tin chuyển nhượng NBA (train 3.263, valid 399, test 438), mỗi tin kèm một đồ thị con KG-trước và KG-sau. Từ vựng: 33 đội, 6.658 entity "người + mùa giải" (1.879 người thật, 9 mùa 2010-11 đến 2018-19), 4 quan hệ, 7.417 token. Tin dài trung bình 29 token, nhắc tới trung bình 2,7 entity.

**KG ban đầu có gì.** Chỉ là **ảnh chụp đội hình** của 1 đến 3 đội trong một mùa: mỗi đội gồm danh sách cầu thủ (`<player>`, hai chiều), mọi cặp cầu thủ trong đội nối `<teammate>` (hai chiều, đồ thị đầy đủ), một HLV trưởng và một GM. Không có gì khác: không thống kê, không ngày tháng, không lịch sử, không cạnh giữa hai đội. Trung bình 476–498 cạnh, 28 entity, trong đó 89% là cạnh teammate.

**Lượng thay đổi mỗi tin.** Trung bình 33–38 cạnh thêm và 36–40 cạnh xóa, tức khoảng 13–14% đồ thị con đổi sau một tin. Gần như toàn bộ là cạnh teammate (94%), phần còn lại là `<player>` (5%) và một ít `<head_coach>`. Chỉ 5,6% số thay đổi có cả hai đầu được nhắc trong text; 94,4% phải suy ra.

**KG ban đầu đã chứa các update trước chưa?** **Không, theo nghĩa chuỗi thời gian.** KG-trước không phải trạng thái tích lũy sau các tin trước đó. Bằng chứng: 4.100 tin chỉ có 735 đồ thị KG-trước khác nhau; 11 tin "Pelicans waived X" trong cùng mùa dùng đúng một KG-trước và chỉ khác tên X. Mỗi tin là một bài toán độc lập, xuất phát từ cùng một ảnh chụp đội hình của mùa đó. Cái "có vẻ giống chuỗi" (2.076 tin có KG-trước trùng KG-sau của tin khác) là hệ quả của việc dữ liệu được sinh bằng cách thay tên vào khuôn tin, tạo ra các cặp thả/ký đối xứng, không phải dòng thời gian thật. Chi tiết ở mục 6.

---

## 2. Nguồn và định dạng

Nguồn: repo GUpdater (Tang et al., "Learning to Update Knowledge Graphs by Reading News", EMNLP 2019), tải bằng `scripts/download_nba.py`. Bốn file:

| File | Nội dung |
|---|---|
| `NBAtransactions_{train,valid,test}.json` | danh sách tin, mỗi tin một object |
| `entity2id.txt` | 6.691 entity, dòng đầu là số đếm |
| `relation2id.txt` | 4 quan hệ |
| `token2id.txt` | 7.417 token của text |

Mỗi tin:

```
{
  "event": "trade",                      # 8 loại
  "season": "2014-15",
  "text": [7380, 12, 7319, ...],         # token id; entity trong text là entity id, không phải token id
  "text_mentioned_entities": [2048, 12, 0, 6542],
  "subgraph_before": [[h, r, t], ...],   # id
  "subgraph_after":  [[h, r, t], ...]
}
```

Text giải mã là tiếng Anh viết thường đã tách token, tên entity giữ nguyên dạng `Anthony_Randolph_2014-15`. Có ký tự lạ `�` và `<unk>` do tiền xử lý của paper (ví dụ tên cầu thủ hiếm bị thay bằng `<unk>`).

---

## 3. Từ vựng

**Entity: 6.691.**
- 33 đội: 30 đội hiện tại cộng tên cũ (Charlotte Bobcats, New Jersey Nets, New Orleans Hornets).
- 6.658 entity người gắn mùa: `Tên_Mùa`. Cùng một người ở hai mùa là hai entity khác nhau (`Pablo_Prigioni_2014-15` và `Pablo_Prigioni_2015-16`). 1.879 người thật, 1.307 người xuất hiện ở từ 2 mùa trở lên. Mỗi mùa 590–795 entity.
- Không có entity nào khác: không sân, không thành phố, không giải đấu.

**Quan hệ: 4.** Phân bố trong KG-trước:

| Quan hệ | Tỷ lệ cạnh | Ý nghĩa | Chiều |
|---|---|---|---|
| `<teammate>` | 88,9% | hai cầu thủ cùng đội | lưu cả hai chiều |
| `<player>` | 10,0% | cầu thủ thuộc đội | lưu cả hai chiều: (đội, player, cầu thủ) và (cầu thủ, player, đội) |
| `<head_coach>` | 0,5% | HLV trưởng của đội | hai chiều |
| `<general_mananger>` | 0,5% | GM của đội; paper viết sai chính tả, giữ nguyên | hai chiều |

Không có quan hệ giữa cầu thủ và HLV/GM, không có quan hệ giữa hai đội.

---

## 4. Giải phẫu một KG-trước

Tin test idx=1: "Cleveland_Cavaliers have acquired center Anthony_Randolph_2014-15 ... from the Denver_Nuggets". KG-trước 742 cạnh, 43 entity:

| Đội | Cầu thủ | Cạnh teammate | Cạnh player | HLV | GM |
|---|---|---|---|---|---|
| Cleveland Cavaliers | 21 | 420 = 21×20 | 42 | Mike Brown | David Griffin |
| Denver Nuggets | 16 | 240 = 16×15 | 32 | Brian Shaw | Tim Connelly |

Cộng 4 cạnh HLV/GM mỗi đội = 742. Không có cạnh nào nối hai đội với nhau. Công thức tổng quát cho một đội n cầu thủ: n(n−1) teammate + 2n player + 4 = n² + n + 4 cạnh. Đội 15 người là 244 cạnh, đội 21 người là 466 cạnh. Đó là lý do đồ thị con dao động từ 160 đến 1.256 cạnh, và tại sao 89% cạnh là teammate.

Số đội trong đồ thị con: 1 đội (66%), 2 đội (32%), 3 đội (1,5%). Tin released/free_agency/draft chỉ có 1 đội; tin trade có 2 đội, hiếm khi 3.

Mùa của entity luôn trùng mùa của tin: kiểm tra 4.100 tin, không có ngoại lệ.

---

## 5. Lượng thông tin cập nhật

### 5.1 Theo split

| Split | n | cạnh KG-trước (TB) | thêm (TB) | xóa (TB) | % đồ thị đổi | tin không thêm gì | tin không xóa gì |
|---|---|---|---|---|---|---|---|
| train | 3.263 | 476 | 33,2 | 36,1 | 13,2% | 1.273 (39%) | 967 (30%) |
| valid | 399 | 489 | 37,7 | 38,5 | 13,9% | 137 | 123 |
| test | 438 | 498 | 38,2 | 39,7 | 13,9% | 153 | 125 |

Không tin nào có KG-sau bằng KG-trước. Trung vị thêm/xóa đều là 34, tức đúng bằng một cầu thủ vào/ra đội 17 người (2 cạnh player + 2×16 cạnh teammate = 34).

### 5.2 Theo loại sự kiện (trung bình toàn bộ 4.100 tin)

| Sự kiện | Số tin | Thêm | Xóa | Bản chất |
|---|---|---|---|---|
| trade | 1.245 | 75,0 | 74,4 | 1 hoặc nhiều cầu thủ đổi đội, cả hai đội đều đổi |
| released | 1.382 | 0 | 37,3 | một cầu thủ rời đội, chỉ xóa |
| free_agency | 1.078 | 38,4 | 0 | một cầu thủ vào đội, chỉ thêm |
| draft | 137 | 37,7 | 0 | như free_agency |
| overseas | 85 | 0 | 35,2 | như released |
| d_league | 39 | 0 | 38,3 | như released |
| retirement | 57 | 0 | 33,0 | như released |
| head_coach | 77 | 2,0 | 2,0 | đổi HLV: 2 cạnh xóa, 2 cạnh thêm |

Vì vậy về cấu trúc chỉ có ba kiểu bài toán: **rời đội** (5 loại sự kiện, 38% tin), **vào đội** (2 loại, 30%), **trade** (30%, là rời + vào ở hai đội), cộng một nhóm nhỏ đổi HLV (2%).

### 5.3 Theo quan hệ

Trên toàn bộ dữ liệu: 132.302 cạnh teammate thêm, 142.248 xóa; 7.548 player thêm, 8.288 xóa; 154 head_coach thêm và xóa. GM không bao giờ đổi. Tức 94% mọi thay đổi là cạnh teammate, chính là phần "lan truyền" mà tin không bao giờ nói ra.

### 5.4 Tường minh và ngầm

Đếm số op (thêm hoặc xóa) có cả hai đầu nằm trong `text_mentioned_entities`: 5,6% ở cả ba split (test: 1.924/34.130). Con số này trùng khớp với paper. Với tin "Pacers waived Young": text nói 2 cạnh (Young–Pacers hai chiều), KG-sau xóa 32 cạnh; 30 cạnh teammate còn lại là ngầm.

Entity mới: 31% tin có đúng 1 entity không có trong KG-trước (cầu thủ được ký hoặc draft); 69% tin không có entity mới. Với draft và free_agency, cầu thủ vào đội **không bao giờ** có mặt trong KG-trước, nên hệ thống phải tạo entity mới từ text.

---

## 6. KG-trước có phải trạng thái sau các update trước không?

Đây là câu hỏi quan trọng nhất về bản chất dataset, và câu trả lời là **không**.

**Bằng chứng 1: rất ít KG-trước khác nhau.** 4.100 tin nhưng chỉ 735 đồ thị KG-trước phân biệt. 3.693 tin (90%) dùng chung KG-trước với ít nhất một tin khác. Nếu dữ liệu là chuỗi thời gian thật, mỗi tin sẽ có KG-trước riêng vì đội hình thay đổi sau mỗi giao dịch.

**Bằng chứng 2: khuôn tin lặp lại với tên thay đổi.** 1.499 khuôn tin (thay entity bằng chỗ trống) cho 4.100 tin; 243 khuôn dùng từ 5 lần trở lên. Ví dụ khuôn "new orleans – the Pelicans announced today the team has waived point guard ⟨E⟩" xuất hiện 22 lần: 11 lần mùa 2015-16, 11 lần mùa 2016-17, mỗi mùa cùng một KG-trước, chỉ khác tên cầu thủ bị thải. Trong đó có "Pelicans waived Anthony_Davis_2016-17", điều không xảy ra ngoài đời. Tương tự "Knicks waived Carmelo_Anthony_2014-15", "Clippers waived Jamal_Crawford_2013-14". 2.765/4.100 tin (67%) có một tin "anh em" cùng khuôn và cùng KG-trước.

Kết luận hợp lý từ bằng chứng: dataset được **tăng cường bằng cách thay tên cầu thủ trong đội hình vào khuôn tin thật**. KG-trước là ảnh chụp đội hình của mùa, dùng chung cho mọi tin tổng hợp từ khuôn đó.

**Bằng chứng 3: "chuỗi" thực ra là chu trình.** 2.076 tin có KG-trước trùng KG-sau của một tin khác. Nhưng khi lần theo, chuỗi tạo thành vòng lặp: tin "thải X" cho KG-sau không có X, trùng KG-trước của tin "ký X", mà KG-sau của tin này lại trùng KG-trước của tin "thải X". 1.643 liên kết như vậy còn bắc qua ranh giới train/valid/test. Đây là hệ quả của việc sinh cả tin rời và tin vào cho cùng một cầu thủ trên cùng ảnh chụp, không phải dòng thời gian.

**Bằng chứng 4: đội hình khác nhau trong cùng mùa là do sinh dữ liệu.** Nhóm Sacramento Kings 2010-11 có 34 tin, KG-trước khác nhau một cầu thủ tùy tin: tin "acquired Hilton Armstrong" thì KG-trước không có Armstrong, tin "acquired Michael Finley" thì có Armstrong nhưng không có Finley. Tức KG-trước được suy ngược từ KG-sau bằng cách bỏ cầu thủ được nhắc, trên cùng một đội hình gốc.

**Ý nghĩa cho bài toán:**
- Mỗi tin là bài toán độc lập "cho ảnh chụp đội hình và một tin, dựng ảnh chụp mới". Không cần và không thể dùng lịch sử giữa các tin. Thiết kế bi-temporal của mình vì thế chỉ có hai mốc t=0 và t=1 trên dataset này; giá trị của bi-temporal sẽ thể hiện khi đưa vào chuỗi tin thật ở tháng 3–4.
- Vì mỗi tin đứng độc lập, đánh giá "cập nhật nhiều tin liên tiếp" không thể làm trên dataset này.

---

## 7. Rò rỉ train/test và hệ quả cho so sánh với GUpdate

397/438 tin test (91%) có KG-trước **xuất hiện nguyên vẹn trong train**. 247 đồ thị KG-trước có mặt ở hơn một split. Hai tin cùng khuôn "Knicks waived ⟨E⟩" nằm một ở train, một ở test.

Với mô hình học end-to-end như GUpdate, điều này có nghĩa GNN đã thấy đúng đồ thị đó, đúng khuôn tin đó ở train, chỉ khác tên cầu thủ. Con số 0,89 add/del của GUpdate vì thế phần nào phản ánh khả năng ghi nhớ đồ thị và khuôn tin, không chỉ khả năng tổng quát. Đây là nhận xét quan trọng cho khóa luận: hệ thống của mình **không học từ train**, nên không hưởng lợi từ rò rỉ này; nếu đạt số tương đương thì so sánh có lợi cho mình về mặt tổng quát hóa.

---

## 8. Điều dataset không có, và giới hạn khi dùng cho khóa luận

- Không có nhãn "op nào được nói thẳng trong text". Mình xấp xỉ bằng `text_mentioned_entities`, cho đúng 5,6% như paper.
- Không có ngày tháng, chỉ có mùa. Không có thứ tự tin.
- Không có quan hệ nào ngoài 4 loại, và teammate là quan hệ suy diễn thuần cấu trúc (cùng đội thì là teammate). Bài toán "lan truyền" trên dataset này vì thế có lời giải cấu trúc gần như hoàn toàn: cận trên oracle 1,0 của mình chứng minh điều đó.
- Tin tổng hợp làm ngôn ngữ nghèo: 1.499 khuôn cho 4.100 tin, nhiều tin phi thực tế. LLM trích ops chỉ cần nhận dạng khuôn, không cần hiểu sâu.
- Đội hình 15–21 người tạo đồ thị con 244–466 cạnh mỗi đội, là lý do prompt phình khi liệt kê từng cạnh.

Tóm lại: NBAtransactions là một dataset tốt để kiểm chứng cơ chế "ops tường minh → khoanh vùng → phán" và so sánh trực tiếp với GUpdate, nhưng là dataset dễ về ngôn ngữ, gần như tổng hợp, và không có chiều thời gian thật. Khóa luận nên nói rõ điều này và bổ sung một miền có chuỗi tin thật cho phần bi-temporal.
