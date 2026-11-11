# Kết quả TN10

- KET_QUA_10.md: báo cáo, bảng so sánh và kết luận H10.1–H10.3.
- ket_qua_10.json: toàn bộ số liệu tổng hợp dùng tạo báo cáo.
- hinh_10.png: hình kết quả gồm ba biểu đồ.
- mo_hinh10/: toàn bộ checkpoint .pt, nhật ký học *_nhat_ky.json,
  cấu hình suy luận chon_buoc_*.json và dấu hoàn thành .xong.
- tn10/chon.json: kết quả các bước chọn cấu hình; tn10/kiem_thu_cheo/:
  số liệu kiểm thử chéo được lưu lại.
- logs/: log toàn bộ TN10, log từng mô hình, trạng thái hoàn thành
  và thông tin điều chỉnh hiệu năng trong lần chạy.
- manifest_sha256.json: danh sách tệp và SHA-256 đã kiểm tra sau khi chép.

## Checkpoint cuối để sử dụng

- DM10: mo_hinh10/d10_s0.pt, d10_s1.pt, d10_s2.pt (3 seed), S=32.
- Đối chứng DM6r: mo_hinh10/r6_s0.pt, r6_s1.pt, r6_s2.pt, S=8.
- Đi kèm các file chon_buoc_d10_s*.json và chon_buoc_r6_s*.json.
- Các file e*_p*.pt dùng kiểm thử chéo; e*_day_s*.pt là checkpoint
  nguồn của mô hình cuối. Nhật ký học dùng các tên e* này.

## Ghi chú lần chạy

TN10 hoàn tất. Bước 39 đã chạy lại thành công với file kiem_dinh_boi_9.py
gốc do người dùng cung cấp. Số liệu JSON khớp báo cáo trước khi thay file.
Log giữ cả lỗi thiếu mô-đun ban đầu và các lần chạy lại thành công.
HighQoS được bật trong quá trình chạy; các phép đo thời gian trước và sau
thay đổi chịu ảnh hưởng của chế độ CPU và tải hệ thống.

Đây là bản sao các tệp kết quả, không phải bản sao toàn bộ dự án.
Không gồm venv, mã nguồn hay dữ liệu .pkl trong data/bao_da_tuyen.
Muốn tính lại toàn bộ thống kê cần giữ thêm dữ liệu .pkl và mã nguồn dự án.

Log tn10.log đã rút gọn các khối JSON thành dòng tóm tắt; log gốc: logs/archive/tn10_before_cleanup_20261111_105539.log.
