# Xem demo TN6 và TN11

1. Chọn **Code → Download ZIP** trên nhánh `demo`, rồi giải nén toàn bộ.
2. Mở `ban_do_3d.html` bằng Chrome hoặc Edge để xem bản đồ TN6.
3. Bấm **Chạy so sánh TN6 / TN11** trên bản đồ.
4. Chọn bộ dữ liệu, thước đo, phương pháp A và đối chứng B, rồi bấm **Chạy so sánh**.
5. Xem kết quả vừa tính và tải **CSV tổng hợp / CSV chi tiết / JSON / SVG**.

Nút này **phân tích lại kết quả đã lưu**, không huấn luyện hoặc suy luận lại.
TN11 tính mới CI bằng bootstrap ghép cặp theo cửa sổ. TN6 tính chênh lệch từ
bảng gốc và giữ CI gốc khi có; không bịa CI mới nếu thiếu dữ liệu từng cửa sổ.

Chỉ cần gửi **một file `results/bao_da_tuyen/demo_thong_ke/chay_so_sanh.html`**
để chạy so sánh offline; dữ liệu và mã đã nhúng trong file. Muốn dùng các liên
kết sang đường học, 9 biểu đồ và hướng dẫn thì gửi cả thư mục `demo_thong_ke`.

Nút **Thống kê TN6 / TN11** mở bộ 9 hình đã xuất trước.

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

Để cập nhật dữ liệu trang so sánh từ một repo đã chạy xong TN11:

```sh
python -m scripts.xuat_demo_so_sanh --source-root /duong/dan/repo_TN11
node --test test/test_demo_so_sanh.cjs
```

Bộ xuất dùng thư viện chuẩn Python. Cache gốc `cham_tep`, `cv_tep`,
`ket_qua_11.json`, `phan.json`, `kiem_tra_du_lieu.json` phải có ở repo nguồn.
Trang nhúng tổng và số lượng từng cửa sổ/dải tải; đủ để tái lấy mẫu cửa sổ
cho trung bình ghép cặp. Không thay các cửa sổ thành mẫu độc lập theo từng bản ghi.
