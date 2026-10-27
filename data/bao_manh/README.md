# Dữ liệu bão mạnh dùng ngay

Bộ dữ liệu nhỏ này được lưu trực tiếp trong Git. Sau khi clone/pull repo, Nhật
và Trung có thể nạp bảng tra mà không cần chạy lại quá trình mô phỏng.
Các đường dẫn ví dụ dưới đây tính từ thư mục gốc repo.

| Tệp | Nội dung |
|---|---|
| `danh_muc.json` | Danh mục 10 cơn, 75 cửa sổ mô phỏng, giờ UTC và hash thời tiết |
| `<ten_bao>.npz` | 10 tệp thời tiết đã cắt, mỗi trường thời tiết 2-D có 74 cột địa điểm |
| `bang_tra/bang_tra.npz` | Bảng 3.000 ô đã học, xác suất sống và số mẫu hỗ trợ |
| `bang_tra/dem.npz` | Số mẫu/tổng số lần sống theo từng cơn và ô |
| `bang_tra/meta.json` | Cấu hình, seed, split và hash các đầu vào/bảng |
| `bang_tra/validation.json` | Điểm validation theo cơn và gộp |
| `SHA256SUMS` | SHA-256 cho toàn bộ tệp dữ liệu và metadata trong bộ này |

## Nạp bảng

```python
from src.ngau_nhien.ma_tran_rui_ro import nap_bang, ma_tran_online

bang = nap_bang('data/bao_manh/bang_tra/bang_tra.npz')
print(bang['p_song'].shape)  # (3000,)

# Khi đã có g (MangGoc), Wn/q (sinh_kich_ban) và mốc quyết định t:
# o = t - 9
# past = {k: q[k][:o+1] for k in ('R', 'gust', 'giong', 'tat')}
# r = ma_tran_online(bang, g, Wn[o], past, t, tuoi=9, delta=3)
```

Tuổi thông tin 180 giây, cửa sổ giữ liên kết 60 giây. Dùng `valid` để lọc cạnh
trước khi đọc `risk_edge`; NaN là cạnh không dùng được. Xem
[hướng dẫn đầy đủ](../../dataset/BANG_TRA_RUI_RO.md) cho model và visualization.

## Kiểm tra và chạy lại

Kiểm tra nguyên vẹn từ gốc repo:

```bash
(cd data/bao_manh && sha256sum -c SHA256SUMS)
```

Đánh giá bảng đã có trên một cửa sổ Yagi; thư mục đầu ra phải mới hoặc rỗng:

```bash
python -m src.ngau_nhien.danh_gia_bang_rui_ro \
  --table-dir data/bao_manh/bang_tra \
  --out /tmp/pbl4-yagi-lookup-eval
```

Đếm và học lại từ các tệp thời tiết đã commit:

```bash
python -m src.ngau_nhien.ma_tran_rui_ro \
  --catalog data/bao_manh/danh_muc.json \
  --out /tmp/pbl4-bang-tra-rebuild
```

`source_path` trong danh mục ghi đường dẫn ở máy tạo dữ liệu để lưu nguồn gốc;
không cần có thư mục `/workspace` trên máy người dùng. Bộ đếm tìm tệp thời tiết
bằng trường `tep` tương đối với danh mục. Danh mục, bảng và các hash đầu vào giữ
nguyên; `meta.json` được bổ sung `applicability` để nêu rõ phạm vi sử dụng.

## Phạm vi

26.088.602 mẫu cạnh theo thời gian: 20.414.992 train và 5.673.610 validation.
Train gồm 7 cơn, validation gồm Prapiroon/Trami/Kalmaegi; Yagi không tham gia
học bảng. Chỉ 351/3.000 ô có mẫu train; bảng có làm trơn cho ô ít/thiếu mẫu.

Đây là dữ liệu thời tiết dùng cho mô phỏng và số đếm sự cố do mô phỏng tạo ra,
không phải bản ghi đứt liên kết ngoài thực địa. Bộ tái tạo có 75 cửa sổ, không
phải bộ 81 cửa sổ được mô tả trong tài liệu ban đầu. Các hạn chế hình học v5,
thời tiết và nhãn được ghi trong hướng dẫn đầy đủ.

Bảng chỉ áp dụng cho `xung_dot=False, tai=False` và các tham số trong
`bang_tra/meta.json` (trường `applicability`). Nếu thí nghiệm bật xung đột/tải
động, cần đếm và học lại trước khi tuyên bố xác suất phù hợp với cấu hình đó.
Prior 20 đang cố định; chưa chọn trên val và chưa huấn luyện lại bằng cả 10 cơn.
