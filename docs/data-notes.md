# NBAtransactions — ghi nhận từ spike (2026-09-07)

## Cấu trúc
- keys mỗi example: `event, season, subgraph_after, subgraph_before, text, text_mentioned_entities`
- `text`: list token id (giải mã ra câu tiếng Anh)
- `text_mentioned_entities`: list entity id

## Hướng quan hệ (quan trọng cho khoanh vùng)
- `<player>`: head = team OR player, tail = player OR team (cả 2 chiều: `(Indiana_Pacers, <player>, Thaddeus_Young_2017-18)` và `(Anthony_Randolph_2014-15, <player>, Cleveland_Cavaliers)`)
- `<teammate>`: có lưu cả 2 chiều (a,teammate,b) và (b,teammate,a)? **có** (ví dụ: `(Aaron_Brooks_2017-18, <teammate>, Thaddeus_Young_2017-18)` và ngược lại)
- `<head_coach>`: head = team hoặc player, tail = ngược lại
- `<general_mananger>`: head = team hoặc player, tail = ngược lại

## Thống kê split test
- số example: 438
- avg added: 38.20 triples/example
- avg deleted: 39.72 triples/example
- phân bố event: trade (150), released (139), free_agency (107), draft (18), head_coach (10), d_league (6), retirement (5), overseas (3)

## Ví dụ 1: Released (Thaddeus Young)
```
event: released | season: 2017-18
text: the Indiana_Pacers announced friday they have waived forward Thaddeus_Young_2017-18 .
mentioned: ['Thaddeus_Young_2017-18', 'Indiana_Pacers']
|before| = 276 |after| = 244
DELETED: Aaron_Brooks_2017-18 <teammate> Thaddeus_Young_2017-18, Al_Jefferson_2017-18 <teammate> Thaddeus_Young_2017-18, C.J._Miles_2017-18 <teammate> Thaddeus_Young_2017-18, ..., Indiana_Pacers <player> Thaddeus_Young_2017-18
```
Khi một cầu thủ bị release, tất cả quan hệ của anh ta với đội cũ (player, teammate) bị xóa.
