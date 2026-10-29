# Dữ liệu bão mạnh và ma trận rủi ro đứt tuyến — v1

Để dùng **bảng tra học từ nhiều cơn bão làm đầu vào online**, xem
[BANG_TRA_RUI_RO.md](BANG_TRA_RUI_RO.md). Tài liệu hiện tại mô tả nhãn ensemble
hậu kiểm; hai bộ có định nghĩa cửa sổ khác nhau.

Đầu vào: mạng nền v5 (224 nút), thời tiết mở rộng 74 điểm. Tệp `oracle_dataset_v5.npz` chỉ là mạng nền; lệnh dưới tạo **bộ đánh giá Yagi** có tác động mưa, gió/giông. Giữ nguyên mọi bộ train/val/test cũ.

## Sinh dữ liệu

Chạy từ thư mục gốc repo, Python đã cài `requirements.txt` hoặc môi trường CPU tương đương:

```bash
python -m src.ngau_nhien.tao_du_lieu_bao \
  --out-dir /workspace/pbl4-data/yagi-risk \
  --start 2024-09-07T06:00 --snapshots 600 --stride 20 \
  --delta 9 --ensemble 16 --pairs 32 --seed 42
```

- 600 bước × 20 s: gần 3 giờ 20 phút thời tiết; lưu 30 mốc, cách nhau 400 s.
- Mỗi mốc có 32 cặp gateway/user cố định, tổng 960 truy vấn. Bao gồm cả cặp không còn đường đi; UAV không được tính làm nhu cầu người dùng trong bộ này.
- Cửa sổ rủi ro: 9 bước = 180 s, **không gồm bước hiện tại**. Không cắt ngắn cửa sổ ở cuối.
- Một thế giới quan sát và 16 thế giới ensemble có seed độc lập, đều tái lập từ seed gốc.
- Chỉ bật tác động thời tiết; không thêm sự cố an ninh hoặc biến động tải để làm bão có vẻ nặng hơn.
- Từ chối ghi đè thư mục có dữ liệu. Mỗi cấu hình dùng thư mục mới.

Stress test tăng mưa/gió, có nhãn `synthetic_stress=true`:

```bash
python -m src.ngau_nhien.tao_du_lieu_bao \
  --out-dir /workspace/pbl4-data/yagi-risk-stress \
  --start 2024-09-07T06:00 --ensemble 16 \
  --rain-scale 1.5 --gust-scale 1.25
```

Hệ số 1,5 và 1,25 là thiết lập stress test, **không phải** cường độ Yagi đã đo hay xác suất đã hiệu chuẩn. Nguồn thời tiết thực tế vẫn là dữ liệu lưới, không phải cảm biến trên UAV. Ngưỡng hạ cánh và giông là giả định của mô phỏng hiện có.

## Định nghĩa ma trận rủi ro

Với mỗi liên kết (i,j) và mốc t:

```
R_edge[t,i,j] = số thế giới có cạnh sống ở t và đứt ít nhất một lần trong (t,t+Δ]
               / số thế giới có cạnh sống ở t
```

Đơn vị là xác suất trong [0,1], ước lượng từ ensemble hữu hạn. Có thể đứt vì quỹ đạo hoặc thời tiết; đây là rủi ro tổng, không phải chỉ rủi ro do mưa. Không loại trừ việc phục hồi trước khi hết cửa sổ: chỉ cần từng đứt một lần là sự kiện đã xảy ra.

- Rủi ro cả tuyến được đếm **trực tiếp trên cùng các thế giới** và điều kiện tuyến còn hoạt động ở t. Không dùng `1 − ∏(1 − R_edge)` vì các cạnh chịu cùng bão/nút hỏng phụ thuộc nhau.
- Tuyến dùng để chấm là tuyến ngắn nhất trên thế giới quan sát ở t; giữ nguyên tuyến trong cửa sổ. Nhãn này không đo lợi ích tái định tuyến.
- `NaN` cùng `count=0`: không có thế giới hợp lệ để ước lượng. **Không** thay NaN bằng 0 và coi cạnh đó an toàn. Đường chéo luôn không hợp lệ; cạnh đang hỏng trong mọi thế giới không có xác suất "từ sống thành đứt".
- Đọc `count`: với 16 mẫu độc lập, sai số chuẩn cực đại của xác suất Bernoulli xấp xỉ 0,125; mẫu có điều kiện có thể còn ít hơn. Đây là nhãn thử nghiệm, cần tăng ensemble và kiểm độ hội tụ trước kết luận nghiên cứu.

**Cảnh báo rò rỉ thông tin:** các trường `_oracle` dùng diễn biến thời tiết tương lai và mô phỏng hậu kiểm. Chúng là **nhãn/đối chứng**, không được ghép vào feature của mô hình online. Chúng điều kiện theo trạng thái cạnh/tuyến sống hiện tại và cùng quỹ đạo thời tiết, chưa điều kiện theo toàn bộ trạng thái ngẫu nhiên quan sát. Muốn dự báo nhân quả phải xây estimator chỉ dùng thông tin có tại t, rồi so với nhãn này.

## Schema `test.npz`

Ký hiệu S: số mốc lưu; N: số nút; M: số truy vấn; L: chiều dài tuyến tối đa.

| Trường | Shape | Vai trò |
|---|---|---|
| `W_snapshots` | S,N,N | Ma trận trễ hiện tại của thế giới quan sát, ms; -1 = không có cạnh |
| `node_extra` | S,N,8 | Các đặc trưng hiện tại như loader ngẫu nhiên cũ; không có nhãn tương lai |
| `node_order`, `node_kind`, `node_layer` | N | Tên, loại và tầng nút |
| `base_snapshot_id` | S | Mốc tương ứng trong mạng nền |
| `weather_time_utc` | S | Thời gian thời tiết tuyệt đối, UTC |
| `risk_edge_oracle` | S,N,N | Nhãn xác suất đứt liên kết; NaN nếu không hợp lệ |
| `risk_edge_count` | S,N,N | Số thế giới hợp lệ trong mẫu số |
| `query_snapshot_id` | M | Chỉ số cục bộ 0..S-1 của truy vấn |
| `query_source`, `query_target` | M | Chỉ số nguồn/đích, cùng các cặp cho mọi mốc |
| `query_reachable` | M | Có đường trong thế giới quan sát tại t không |
| `current_path_indices` | M,L | Tuyến ngắn nhất hiện tại; -1 padding; toàn -1 khi không có đường |
| `risk_route_oracle` | M | Rủi ro cả tuyến có điều kiện, đo trực tiếp qua ensemble |
| `risk_route_count` | M | Mẫu số hợp lệ của nhãn rủi ro tuyến |
| `route_failed_observed` | M | Tuyến bị đứt trong tương lai thực hiện: 0/1; -1 khi không có tuyến tại t |

Không dùng trực tiếp `SagsinData` để tránh nó âm thầm bỏ các truy vấn không có đường. Đây là schema đánh giá rủi ro mới, không phải thay thế không điều kiện cho nhãn next-hop.

`meta.json` ghi schema, mục đích holdout, tham số vật lý, thời gian, horizon, seed quan sát/ensemble, hệ số stress, SHA-256 các đầu vào và đầu ra. Tệp đầu vào gốc không bị sửa.

## Bàn giao cho Nhật và Trung

```python
import numpy as np
with np.load('/workspace/pbl4-data/yagi-risk/test.npz', allow_pickle=False) as d:
    k = 0
    W = d['W_snapshots'][k]
    R = d['risk_edge_oracle'][k]
    # Chỉ vẽ cạnh hiện tại tồn tại VÀ có ước lượng hợp lệ.
    mask = (W > 0) & (d['risk_edge_count'][k] > 0)
    edge_risk_for_plot = np.where(mask, R, np.nan)
    # Mẫu số gồm cả truy vấn không còn tuyến, không lọc trước khi báo service rate.
    service_rate = d['query_reachable'].mean()
```

- Nhật: có thể học dự báo `_oracle` từ W/feature hiện tại, giữ loss mask hợp lệ và báo chất lượng theo số mẫu. Bộ này chứa Yagi nên mặc định **chỉ đánh giá**; train/val lấy sự kiện/ngày ngoài Yagi, không chia ngẫu nhiên các hàng của bộ này.
- Trung: có thể tô cạnh theo rủi ro, hiển thị cửa sổ 180 s, số mẫu và trạng thái không có dữ liệu; phân biệt cạnh đang đứt với cạnh còn sống nhưng rủi ro cao.
- Muốn rủi ro "không còn bất kỳ đường nào giữa s–d" cần đánh giá reachability tương lai trên từng thế giới; `risk_route_oracle` hiện đo đứt **tuyến cố định**, không phải mất mọi đường thay thế.

## Giới hạn đang giữ nguyên

Hình học v5 thuộc TLE 2026 trong khi thời tiết Yagi thuộc 2024; đây là mạng giả định chịu thời tiết Yagi. Lỗi kiểm tra che khuất ISL của mạng nền đã được ghi nhận trong lần đánh giá trước, chưa sửa trong thay đổi này. Không suy từ bộ dữ liệu này ra xác suất hỏng mạng thực tế tại Yagi 2024. Gió 10 m, nội suy mưa giờ, AR(1), luật giông và ngưỡng UAV cần được hiệu chuẩn/kiểm độ nhạy khi dùng cho nghiên cứu.
