# Bảng tra bão mạnh và ma trận rủi ro online

**Dữ liệu đã được lưu trong repo:** [data/bao_manh](../data/bao_manh/README.md)
gồm 10 tệp thời tiết, danh mục, bảng tra đã học, số đếm, metadata và SHA-256.
Clone/pull là có thể nạp bảng; không cần chạy lại các lệnh sinh dữ liệu bên dưới.

Pipeline `storm-lookup-v1` triển khai bảng 3.000 ô trong tài liệu
`Du_lieu_bao_manh.docx` của Huy. Bảng học từ sự cố **mô phỏng** với thời tiết
74 điểm và mạng nền v5, không phải xác suất đứt liên kết được đo ngoài thực địa.

## Ý nghĩa ma trận

Tại thời điểm quyết định `t`, bộ định tuyến chỉ biết trạng thái ở `o = t − 9`
(cũ 180 giây). Cần giữ liên kết liên tục từ `t` đến `t + 3` (60 giây).
Một bước bằng 20 giây. Có hai thành phần cần dùng cùng nhau:

- `valid[i,j]`: liên kết tồn tại tại `t` và toàn bộ cửa sổ theo lịch quỹ đạo,
  trạng thái quan sát không báo cạnh hỏng, và hai đầu không có nút đã biết bị tắt.
- `risk_edge[i,j] = 1 − p_song[cell_id[i,j]]`: xác suất mô phỏng liên kết
  không hoạt động tại ít nhất một mốc trong `[t,t+3]`, trên các cạnh dự báo hợp lệ.
  Các đặc trưng băng tần, mưa, gió và giông thể hiện tác động bão lên từng liên kết.

Như vậy, điều kiện kết nối giữa các nút được xử lý bằng `valid` và lịch quỹ đạo;
rủi ro còn lại được ước lượng bằng bảng thời tiết. Cạnh chắc chắn không dùng
được có `valid=False`, `risk_edge=NaN`; **không đổi NaN thành 0 rồi cho phép đi qua**.
Đối với cạnh mới xuất hiện theo quỹ đạo, chưa có phép đo suy hao ở thời điểm quan
sát nên dùng hệ số nền 1, nhưng vẫn chặn đầu nút đã tắt. Đây là chỗ đã sửa lỗi
vệ tinh nối trở lại một UAV đang được biết là hạ cánh.

Hình học tương lai từ quỹ đạo là đầu vào dự báo cho phép. Thời tiết và trạng thái
nút tương lai chỉ được dùng để tạo nhãn/chấm điểm. Hàm online chạy được với lịch
sử thời tiết bị cắt ngay tại `o`; không nhận mạng thực tế tương lai.

Nút đã tắt tại `o` bị chặn. Nút còn hoạt động tại `o` nhưng tắt sau đó, kể cả
trong lúc giữ tuyến `[t,t+3]`, vẫn tạo nhãn đứt: đây chính là rủi ro cần dự đoán.

## Đặc trưng và làm trơn

| Chiều | Nhóm |
|---|---|
| Băng tần | 0 (không suy hao mưa), 5, 12, 20, 28 GHz |
| Mưa tại điểm đại diện liên kết | 10 khoảng, biên 0,5; 2; 5; 8; 11; 15; 20; 30; 50 mm/h; bằng biên vào khoảng trên |
| Xu hướng mưa 15 phút | Giảm hơn 1 mm/h; thay đổi trong [-1,1]; tăng hơn 1 mm/h |
| Gió giật tại đầu UAV | Không có UAV; <10; [10,13); [13,15]; >15 m/s |
| Giông tại đầu UAV | Có/không |
| Giông tại đầu ground | Có/không; ground gồm gateway và user |

Tích số nhóm là `5×10×3×5×2×2 = 3000`. Khi có hai đầu UAV, lấy gió lớn nhất;
giông lấy OR trên các đầu thuộc loại tương ứng. Điểm đại diện và băng tần theo
`MangGoc`, không tự gán lại đường truyền trong bảng tra.

Chỉ số đếm **train** tham gia học. Với `s` lần sống và `n` mẫu:

```text
p_band = (s_band + 1) / (n_band + 2)
p_band_rain = (s_band_rain + 20*p_band) / (n_band_rain + 20)
p_cell = (s_cell + 20*p_band_rain) / (n_cell + 20)
```

Số 20 là tham số làm trơn cố định, chưa tối ưu trên Yagi. `support` là số mẫu
ở ô chi tiết; `support_level` cho biết mức có dữ liệu: 3=ô, 2=băng×mưa,
1=băng, 0=chưa thấy băng (prior 0,5), -1=cạnh không hợp lệ. Mức hỗ trợ không
phải khoảng tin cậy và số mẫu không phải số quan sát độc lập.

## Tạo dữ liệu và tái lập

Chạy từ gốc repo với Python có các thư viện trong `requirements.txt`:

```bash
python -m src.ngau_nhien.bao_manh \
  --out /workspace/pbl4-data/bao-manh

python -m src.ngau_nhien.ma_tran_rui_ro \
  --catalog /workspace/pbl4-data/bao-manh/danh_muc.json \
  --out /workspace/pbl4-data/bang-tra-bao

python -m src.ngau_nhien.danh_gia_bang_rui_ro \
  --table-dir /workspace/pbl4-data/bang-tra-bao \
  --out /workspace/pbl4-data/yagi-lookup-eval
```

Các lệnh từ chối ghi đè thư mục có dữ liệu. Muốn đổi cấu hình, dùng thư mục mới.
Không cần sinh lại `oracle_dataset_v5.npz` khi chỉ đổi thời tiết; phải đếm/học
lại bảng và tạo lại dữ liệu bão phụ thuộc cấu hình đã đổi. Những bộ sinh cũ dùng
mạng dự báo có lỗi UAV cần tạo lại nếu muốn áp dụng bản sửa này.

Danh mục cắt 10 khoảng thời tiết theo tài liệu, lấy cửa sổ gần 3 giờ 20 phút,
bắt đầu cách nhau 3 giờ và đủ giờ thứ 4 để nội suy. Chọn cửa sổ có giờ mà ít nhất
5% điểm có mưa >11 mm/h hoặc 20% điểm có gió giật >15 m/s. Đây là tiêu chí chọn
kịch bản của mô phỏng, không phải phân cấp bão khí tượng.

Từ tệp hiện có tạo được **75 cửa sổ**, theo thứ tự Prapiroon/Soulik/Trami/Wutip/
Wipha/Kajiki/Nongfa/Bualoi/Matmo/Kalmaegi: **6/7/6/8/15/8/3/16/2/4**.
Tài liệu ghi 81 nhưng thiếu danh sách giờ bắt đầu đầy đủ; không tuyên bố tái lập
chính xác bộ 81 đó. `danh_muc.json` lưu giờ, nguồn và hash của từng tệp đã cắt.

- Train: Soulik, Wutip, Wipha, Kajiki, Nongfa, Bualoi, Matmo.
- Validation: Prapiroon, Trami, Kalmaegi. Không dùng số đếm các cơn này để học.
- Yagi: giữ ngoài cả hai tập trên, chỉ đánh giá bảng đã đóng băng.
- Khi đếm train/validation, dùng 85 mốc thuộc split **train hình học v5**, đủ
  lịch sử và cửa sổ, lấy mỗi 3 phần tử sau lọc. Chia theo cơn để không đưa hai
  cửa sổ chồng nhau của cùng cơn sang train và validation.
- Bật mưa, giông và hạ cánh UAV; dùng mức `khắc nghiệt` của bộ mô phỏng.
  Tắt xung đột ngẫu nhiên và biến động tải, nên không tái sử dụng số đếm cũ
  14.274.022 mẫu trong tài liệu. Các tham số chính xác nằm trong `meta.json`.

**Phạm vi áp dụng:** trường `applicability` trong `meta.json` yêu cầu
`xung_dot=False, tai=False`, cùng các tham số, hình học, tuổi thông tin và horizon
đã ghi. Khi bật xung đột hoặc tải động trong thí nghiệm định tuyến, cần đếm/học
lại theo cấu hình đó trước khi coi xác suất là đã hiệu chuẩn cho thí nghiệm.
Trong bộ mô phỏng hiện tại, tải chủ yếu đổi trọng số/độ trễ; nhãn sống chỉ kiểm
tra `Wn > 0`, không phải xác suất vi phạm SLA hay nghẽn mạng. Không diễn giải
bảng này là dự báo mọi dạng thất bại do tải.

Số 81 kịch bản và 14.274.022 mẫu trong bản DOCX cũ đã hết hạn: lần đếm hiện tại
dùng danh mục 75 cửa sổ công khai, chặn UAV đã tắt trong dự báo, tắt xung đột/tải
động và lọc mốc đủ lịch sử. Không có đủ đầu ra cũ để tách riêng đóng góp của từng
thay đổi vào chênh lệch số mẫu. Hướng dẫn hiện hành được lưu trong tài liệu Markdown này.

Các tệp bảng: `dem.npz` chứa số lần sống/tổng theo cơn và ô; `bang_tra.npz`
chứa xác suất và hỗ trợ học; `validation.json` chứa điểm từng cơn;
`meta.json` chứa seed, split, cấu hình và hash. `dem.npz` được ghi từng cơn;
chỉ coi lần chạy hoàn tất khi có `meta.json`.

## Sử dụng cho model và visualization

```python
import json
from src.ngau_nhien.kich_ban import ThamSo
from src.ngau_nhien.ma_tran_rui_ro import nap_bang, ma_tran_online

with open('data/bao_manh/bang_tra/meta.json', encoding='utf-8') as f:
    meta = json.load(f)
p = ThamSo(**meta['parameters'])  # chính cấu hình truyền cho sinh_kich_ban
bang = nap_bang('data/bao_manh/bang_tra/bang_tra.npz', tham_so=p)
o = t - 9
past = {k: q[k][:o+1] for k in ('R', 'gust', 'giong', 'tat')}
r = ma_tran_online(bang, g, Wn[o], past, t, tuoi=9, delta=3, tham_so=p)
# Nhật: chỉ tạo graph từ r['valid']; gắn risk_edge, support vào cạnh.
# Trung: vẽ màu risk_edge trên valid; tô riêng ô thiếu mẫu/support thấp.
```

`g` là `MangGoc`; `Wn,q` là đầu ra `sinh_kich_ban`. `ma_tran_online` trả các
ma trận `[N,N]`: `risk_edge`, `valid`, `cell_id`, `support`, `support_level`,
`W_forecast`. Thứ tự nút theo v5. Đổi tuổi thông tin hoặc horizon khác với bảng
sẽ báo lỗi, cần học lại cho cấu hình đó.

Cả hai hàm bắt buộc nhận `tham_so` của kịch bản thật. `nap_bang` đối chiếu hash,
hợp đồng cấu hình `contract_json` trong NPZ và `meta.json`; `ma_tran_online`
kiểm tra lại toàn bộ `ThamSo` với hợp đồng nhúng và `applicability` ở mỗi lần
gọi. Thiếu metadata/hợp đồng hoặc lệch cấu hình thì dừng, không chỉ cảnh báo.
API không suy ra cấu hình từ ma trận quan sát: người gọi phải truyền đúng đối
tượng đã dùng để mô phỏng. Bản NPZ cũ thiếu hợp đồng cần thay bằng bản mới cùng
metadata trong repo. Thêm hợp đồng không đổi xác suất hoặc số đếm.

Bộ đánh giá Yagi xuất `matrices.npz` với các ma trận xếp theo thời gian và
`base_snapshot_id`, `node_order`. Trường `target_survives_oracle` là nhãn hậu
kiểm, **không đưa vào đầu vào model**. `evaluation.json` lưu điểm và hash nguồn.
Hàm đánh giá mặc định dùng một cửa sổ bắt đầu `2024-09-07T06:00 UTC`, seed 2026,
48 mốc test v5 đủ lịch sử. Cần thêm các cửa sổ/seed để kết luận toàn đợt Yagi.

Rủi ro từng cạnh không tự cho ra xác suất chính xác của cả tuyến: nhiều cạnh có
thể cùng hỏng do một vùng bão. `1 − product(1-risk_edge)` chỉ là xấp xỉ độc lập.
Đánh giá cả tuyến bằng diễn biến mô phỏng chung hoặc nhãn ensemble ở
[BAO_RUI_RO.md](BAO_RUI_RO.md). Bộ ensemble đó có định nghĩa cửa sổ riêng (180 s,
không gồm hiện tại), không ghép trực tiếp nhãn với bảng này (60 s, gồm hiện tại).

## Kết quả và giới hạn lần chạy mặc định

Đếm được **26.088.602 mẫu cạnh có hướng theo thời gian**, gồm 20.414.992 train và
5.673.610 validation. Chỉ **351/3.000 ô** có mẫu train. Một số tổ hợp đặc trưng
không thể xuất hiện do loại liên kết; số ô trống không đồng nghĩa tất cả đều
cần bổ sung dữ liệu. Các cửa sổ chồng lấn và hai chiều cạnh có tương quan.

Trên cả 10 cơn có **364 ô** có mẫu. Hai nhóm gió UAV >15 m/s và giông UAV=True
đều không có mẫu sau khi loại nút đã tắt tại quan sát; đây là các trạng thái
rỗng do cách chọn ứng viên, không phải mất dữ liệu. Xem
[thống kê bão và giông](BAO_MANH_THONG_KE.md) cùng lưu ý cho GNN trong
[README dữ liệu](../data/bao_manh/README.md).

Brier validation: **0,001552**, so với **0,001947** khi dùng một xác suất sống
cố định học từ train (càng thấp càng tốt). Tỷ lệ sống validation là 99,805%,
vì vậy không dùng accuracy cao để kết luận mô hình tốt. Xem thêm kết quả Yagi
và từng băng tần trong `evaluation.json`; phần lớn cạnh là ISL ít chịu mưa.

Trên một cửa sổ Yagi nói trên: 187.888 mẫu cạnh, Brier **0,001296** so với
**0,001564** của baseline cố định; log-loss **0,004859** so với **0,011985**.
Khoảng 90% mẫu thuộc nhóm không suy hao mưa. Tỷ lệ mất liên kết ở các nhóm
12/20/28 GHz lần lượt khoảng 0,68% / 5,88% / 8,76%; cần đọc theo từng nhóm
thay vì chỉ nhìn điểm gộp. Không có băng tần hoàn toàn chưa gặp trong train.

Hình học v5 dùng TLE 2026 kết hợp thời tiết 2024–2025: đây là thí nghiệm giả định,
không tái dựng mạng lịch sử. Giới hạn kiểm tra che khuất Trái Đất của ISL trong
v5 vẫn còn. Thời tiết lưới, gió gần mặt đất, giông suy từ proxy và các ngưỡng hỏng
là giả định mô phỏng; bảng chưa được hiệu chuẩn bằng sự cố liên kết ngoài thực địa.

## Các thí nghiệm tiếp theo chưa thực hiện trong bản phát hành này

- Chọn prior trên validation rồi học lại cả 10 cơn: cần lưu riêng bảng cuối và
  bảng dùng chấm validation. Điểm validation hiện tại thuộc bảng 7 cơn, không
  được gán cho bảng học lại trên 10 cơn.
- Báo cáo theo băng tần/gió/giông và bootstrap theo cơn; không bootstrap từng
  cạnh như mẫu độc lập. Hiện validation chỉ có 3 cơn, nên suy luận bất định còn hạn chế.
- Kiểm tra đơn điệu theo mưa ở 20/28 GHz khi giữ cố định các chiều khác; phân
  biệt ô thiếu mẫu và hiệu ứng chỉ chọn cạnh còn sống với vi phạm mô hình vật lý.
- Đánh giá Yagi nhiều cửa sổ và seed; chưa dùng Yagi để chọn prior.
- So sánh có/không Matmo trước khi bỏ. Matmo hiện đóng góp 684.006 mẫu, trong đó
  4.800 mẫu không sống suốt cửa sổ; một giờ đạt tiêu chí bão không có nghĩa là
  không đóng góp dữ liệu. Giữ nguyên split hiện tại để bảo toàn kết quả đã công bố.
