# Thống kê dữ liệu bão đã công bố

Tính từ 10 tệp thời tiết và `data/bao_manh/bang_tra/dem.npz`. Mẫu là cạnh có hướng theo thời gian, có tương quan; các tỷ lệ dưới đây là mô tả mô phỏng, không phải ước lượng từ quan sát đứt liên kết thực địa.

## Mười cơn bão

| Cơn | Khoảng UTC | Giờ bão | Cửa sổ | Mưa max (mm/h) | Gió giật max (m/s) | Điểm gió >15 m/s (max %) |
|---|---|---:|---:|---:|---:|---:|
| prapiroon_2024 | 2024-07-21T16:00 → 2024-07-24T00:00 | 6 | 6 | 30.7 | 21.7 | 17.57 |
| soulik_2024 | 2024-09-17T18:00 → 2024-09-21T22:00 | 6 | 7 | 37.3 | 23.4 | 13.51 |
| trami_2024 | 2024-10-26T07:00 → 2024-10-27T19:00 | 13 | 6 | 43.0 | 36.2 | 33.78 |
| wutip_2025 | 2025-06-10T20:00 → 2025-06-13T19:00 | 8 | 8 | 41.9 | 24.2 | 31.08 |
| wipha_2025 | 2025-07-19T17:00 → 2025-07-23T00:00 | 33 | 15 | 29.9 | 33.8 | 60.81 |
| kajiki_2025 | 2025-08-24T12:00 → 2025-08-26T06:00 | 19 | 8 | 26.1 | 34.8 | 55.41 |
| nongfa_2025 | 2025-08-29T15:00 → 2025-08-30T20:00 | 6 | 3 | 23.2 | 22.4 | 39.19 |
| bualoi_2025 | 2025-09-26T14:00 → 2025-09-30T04:00 | 34 | 16 | 33.2 | 38.7 | 52.70 |
| matmo_2025 | 2025-10-06T08:00 → 2025-10-07T08:00 | 1 | 2 | 19.4 | 15.6 | 2.70 |
| kalmaegi_2025 | 2025-11-06T00:00 → 2025-11-07T06:00 | 3 | 4 | 28.9 | 33.2 | 21.62 |

“Giờ bão” đạt tiêu chí ≥5% điểm mưa >11 mm/h hoặc ≥20% điểm gió giật >15 m/s; đây là tiêu chí chọn cửa sổ, không phải cấp bão chính thức. “Điểm … max” là tỷ lệ lớn nhất tại một giờ. Mô tả sáu chiều, công thức, cấu hình và giới hạn nằm trong [BANG_TRA_RUI_RO.md](BANG_TRA_RUI_RO.md).

## Giông ở đầu mặt đất theo cơn

| Cơn | Tổng mẫu | Lần đứt | Đứt (%) | Mẫu có giông mặt đất | Giông / mọi mẫu (%) | Giông / mẫu băng >0 (%) |
|---|---:|---:|---:|---:|---:|---:|
| prapiroon_2024 | 2,125,464 | 5,335 | 0.251 | 3,640 | 0.171 | 1.177 |
| soulik_2024 | 2,459,418 | 11,466 | 0.466 | 18,494 | 0.752 | 5.431 |
| trami_2024 | 2,116,650 | 4,258 | 0.201 | 620 | 0.029 | 0.206 |
| wutip_2025 | 2,837,134 | 5,944 | 0.210 | 6,154 | 0.217 | 1.481 |
| wipha_2025 | 5,165,844 | 8,564 | 0.166 | 3,742 | 0.072 | 0.598 |
| kajiki_2025 | 2,727,406 | 6,018 | 0.221 | 5,128 | 0.188 | 1.677 |
| nongfa_2025 | 1,040,704 | 2,346 | 0.225 | 1,600 | 0.154 | 1.207 |
| bualoi_2025 | 5,500,480 | 17,444 | 0.317 | 21,768 | 0.396 | 3.312 |
| matmo_2025 | 684,006 | 4,800 | 0.702 | 8,184 | 1.196 | 10.412 |
| kalmaegi_2025 | 1,431,496 | 1,470 | 0.103 | 1,402 | 0.098 | 0.635 |

Matmo có 8.184 mẫu giông mặt đất: **1,196% mọi mẫu** hoặc **10,412% mẫu băng >0**. Gộp chín cơn còn lại tương ứng **0,246%** và **1,891%**. Các tỷ lệ 5,7%/1,4% không tái lập được với hai mẫu số này; cần nêu rõ tập con/mẫu số nếu sử dụng. Số đếm hỗ trợ giữ Matmo để so sánh, chưa chứng minh riêng giông là nguyên nhân của toàn bộ chênh lệch giữa các cơn.

## Theo băng tần khi có giông mặt đất

| Băng | Mẫu có giông | Lần đứt | Tỷ lệ đứt (%) |
|---|---:|---:|---:|
| 0 GHz | 0 | 0 | không có mẫu |
| 5 GHz | 5,690 | 2,444 | 42.95 |
| 12 GHz | 44,862 | 20,254 | 45.15 |
| 20 GHz | 12,852 | 6,346 | 49.38 |
| 28 GHz | 7,328 | 3,726 | 50.85 |

Tổng 70.732 mẫu có giông (0,271% toàn bộ), 32.770 lần đứt: tỷ lệ đứt 46,33%, chiếm 48,44% tổng 67.645 lần đứt. Giông là đặc trưng của luật mô phỏng (proxy CAPE/mưa và xác suất sét làm tắt nút). Các con số không chứng minh rủi ro từng ô đều bằng 0,5 hoặc chứng minh GNN đã học tắt; cần đánh giá riêng nhóm giông và ablation đặc trưng trước khi kết luận.

## Ô rỗng và tái lập

Tổng 364/3.000 ô có mẫu trên 10 cơn; 351 ô có mẫu train. Nhóm gió UAV >15 m/s và nhóm giông UAV=True đều có n=0. Đây là hai mức giá trị rỗng do UAV đã tắt bị chặn trước khi đếm, không phải hai trục của schema bị mất. Giữ đủ sáu chiều và thứ tự ô hiện tại. Khi thay ngưỡng/luật hạ cánh, kết luận ô rỗng có thể thay đổi.

Seed phụ thuộc thứ tự cơn/cửa sổ: `100000 + 1000*j + k`. Giữ danh mục nguyên thứ tự hoặc dùng seed đã lưu trong `meta.json:cases` khi tái lập tập con. Không diễn giải số mẫu tương quan là số thí nghiệm độc lập.
