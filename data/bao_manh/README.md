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
| `bang_tra/validation.json` | Chỉ số gộp/theo cơn, băng tần, hỗ trợ train và giông mặt đất |
| `SHA256SUMS` | SHA-256 cho toàn bộ tệp dữ liệu và metadata trong bộ này |

## Nạp bảng

```python
import json
from src.ngau_nhien.kich_ban import ThamSo
from src.ngau_nhien.ma_tran_rui_ro import nap_bang, ma_tran_online

with open('data/bao_manh/bang_tra/meta.json', encoding='utf-8') as f:
    meta = json.load(f)
p = ThamSo(**meta['parameters'])  # dùng chính p này khi gọi sinh_kich_ban
bang = nap_bang('data/bao_manh/bang_tra/bang_tra.npz', tham_so=p)
print(bang['p_song'].shape)  # (3000,)

# Khi đã có g (MangGoc), Wn/q (sinh_kich_ban) và mốc quyết định t:
# o = t - 9
# past = {k: q[k][:o+1] for k in ('R', 'gust', 'giong', 'tat')}
# r = ma_tran_online(bang, g, Wn[o], past, t, tuoi=9, delta=3, tham_so=p)
```

Tuổi thông tin 180 giây, cửa sổ giữ liên kết 60 giây. Dùng `valid` để lọc cạnh
trước khi đọc `risk_edge`; NaN là cạnh không dùng được. Xem
[hướng dẫn đầy đủ](../../dataset/BANG_TRA_RUI_RO.md) cho model và visualization.

`tham_so` bắt buộc là cấu hình **thực tế của kịch bản**, không phải một bản cấu
hình tham chiếu khác với cấu hình đang mô phỏng. Với kịch bản đã có, truyền đối
tượng `ThamSo` của kịch bản đó vào cả hai hàm. Loader kiểm tra `meta.json`, hash
và `contract_json` nhúng trong NPZ; online kiểm tra lại hợp đồng mỗi lần gọi.
Thiếu hợp đồng hoặc lệch bất kỳ tham số nào đều báo lỗi, kể cả tham số đang tắt.
API không tự thay cấu hình kịch bản để ép cho khớp bảng.

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

Chỉ tạo lại **báo cáo từ số đếm**, không sinh mô phỏng hoặc học lại bảng:

```bash
python -m src.ngau_nhien.bao_cao_rui_ro \
  --dem data/bao_manh/bang_tra/dem.npz \
  --bang data/bao_manh/bang_tra/bang_tra.npz \
  --out data/bao_manh/bang_tra/validation.json
```

Lệnh này chỉ ghi tệp `--out`; khi thay báo cáo đã commit phải cập nhật SHA-256
của nó trong `SHA256SUMS`. `meta.json` không tham chiếu báo cáo nên không đổi.

`validation.json` giữ các khóa cũ `<cơn>/pooled.lookup` và
`constant_train_baseline`, bổ sung BSS, AUPRC, số ca đứt và các phân nhóm.
Baseline Brier là cùng một xác suất sống cố định học từ **toàn bộ train** cho
mọi nhóm; baseline AUPRC là tỷ lệ đứt trong **nhóm đang báo cáo**. AUPRC tính cho
lớp đứt, gộp đồng điểm và cộng precision × phần tăng recall.

Nhóm dưới 1.000 mẫu val **hoặc** dưới 30 ca đứt có `mau_it=true`, BSS/AUPRC
để `null`. Các trường `denominators` mô tả mẫu số; `failure_share_percent` chia
cho tổng **11.063 ca đứt trên ba cơn val** trong bản hiện tại. Hỗ trợ dùng
`n_train`, không dùng số mẫu val. Không bootstrap theo cạnh hoặc báo khoảng
tin cậy từ ba cơn này. Xem bảng kết quả trong
[hướng dẫn](../../dataset/BANG_TRA_RUI_RO.md#báo-cáo-validation-từ-số-đếm).

`source_path` trong danh mục ghi đường dẫn ở máy tạo dữ liệu để lưu nguồn gốc;
không cần có thư mục `/workspace` trên máy người dùng. Bộ đếm tìm tệp thời tiết
bằng trường `tep` tương đối với danh mục. Danh mục và số đếm giữ nguyên. Bảng
được bổ sung `contract_json` nên hash NPZ đã đổi, nhưng xác suất không đổi;
hash tương ứng trong `meta.json` và `SHA256SUMS` đã cập nhật. Bảng xuất cũ thiếu
hợp đồng bị từ chối; pull cả NPZ lẫn metadata mới, không chỉ chép riêng code.

**Seed phụ thuộc thứ tự:** `seed = 100000 + 1000*j + k`, với `j` là chỉ số cơn
trong `danh_muc.json`, `k` là chỉ số cửa sổ trong cơn (đều bắt đầu từ 0).
Thêm/xóa/đảo thứ tự cơn hoặc cửa sổ có thể đổi seed. Muốn tái lập một phần,
dùng đúng `(storm, h0, seed)` trong `bang_tra/meta.json:cases`; không đánh lại
`j,k` từ 0 cho danh mục con. CLI chạy toàn danh mục hiện dùng công thức này;
chưa có chế độ tự phát lại danh sách `cases`.

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

## Lưu ý cho Nhật trước khi dùng GNN

- Có 364/3.000 ô có mẫu trên cả 10 cơn, 351 ô trên 7 cơn train. Hai **nhóm giá
  trị** `gió UAV >15 m/s` và `giông UAV=True` có số mẫu bằng 0: nút đã tắt tại
  quan sát bị loại khỏi các cạnh ứng viên. Đây là hệ quả của cấu hình hiện tại,
  không phải thiếu tệp. Vẫn giữ schema sáu chiều để không đổi chỉ số ô.
- Giông mặt đất: 70.732 mẫu (0,271%), 32.770 lần đứt, tỷ lệ đứt 46,33%; chiếm
  48,44% tổng 67.645 lần đứt. Tỷ lệ đứt theo băng 5/12/20/28 GHz lần lượt
  42,95% / 45,15% / 49,38% / 50,85%. Đây là tương quan với luật giông/sét của
  mô phỏng, chưa chứng minh tác động thực địa hoặc GNN chắc chắn học tắt.
  Cần báo cáo riêng nhóm giông và thử bỏ đặc trưng giông để kiểm tra mức phụ
  thuộc; không diễn giải mọi rủi ro cao là do suy hao mưa. Giá trị mỗi ô sau
  làm trơn không nhất thiết bằng 0,5.
- Giữ Matmo: 684.006 mẫu, 4.800 lần đứt (0,702%). Tỷ lệ mẫu có giông mặt đất
  cần nêu rõ mẫu số; xem [thống kê theo cơn](../../dataset/BAO_MANH_THONG_KE.md).

Mô tả 10 cơn, sáu chiều và các lưu ý được lưu trong Markdown; không đưa DOCX
trở lại repo.
