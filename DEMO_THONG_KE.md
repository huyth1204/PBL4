# Xem demo TN6 và TN11

1. Chọn **Code → Download ZIP** trên nhánh `demo`, rồi giải nén toàn bộ.
2. Mở `ban_do_3d.html` bằng Chrome hoặc Edge để xem bản đồ TN6.
3. Bấm **Thống kê TN6 / TN11** trên bản đồ để mở bộ biểu đồ.

Có thể mở thẳng `results/bao_da_tuyen/demo_thong_ke/index.html`, rồi bấm
**Chọn mô hình & xem đường học** để xem 21 mô hình TN11. Hai trang thống kê
chạy offline, không cần Python hoặc GPU. GitHub hiển thị mã nguồn HTML;
cần tải về và mở bằng trình duyệt.

- Hình 01: thống kê TN6 gốc.
- Hình 02–09: kết quả TN11, đường học, chẩn đoán overfitting, giới hạn kết luận
  underfitting, chia nhóm 4-fold, kết quả từng fold, bootstrap và độ nhạy.
- PNG/SVG: dùng cho slide; `csv/`: số liệu vẽ hình.
- [Kịch bản demo và giới hạn kết luận](results/bao_da_tuyen/demo_thong_ke/HUONG_DAN_DEMO.md).

TN6 chưa có nhật ký học và CV của đúng run gốc trong nguồn đã kiểm tra.
Không dùng đường học TN11 làm bằng chứng của TN6. Không coi việc dừng sớm
là bằng chứng rằng mô hình không underfit.

Script `scripts/xuat_demo_thong_ke.py` dùng để tái xuất khi có đầy đủ kết quả,
nhật ký và cache TN11 gốc; xem hướng dẫn bên trên. Nhánh demo chỉ kèm bộ
trình bày đã xuất và các bảng nguồn tổng hợp, không kèm checkpoint/cache.
