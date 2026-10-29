"""Thí nghiệm 2/4 — chọn trọng số score S = α·tốc độ + (1 − α)·rủi ro đứt theo giai đoạn thời tiết, chạy từ đầu.

    python chay_tu_dau/2_score_toc_do_dam_bao.py          # đầy đủ, khoảng 30–40 phút
    python chay_tu_dau/2_score_toc_do_dam_bao.py --thu    # chạy thử (4 cửa sổ mỗi thiết lập), vài phút

Tốc độ = % chậm hơn tuyến nhanh nhất; rủi ro = xác suất đứt dự báo. Chọn tuyến bằng Dijkstra theo S, không có mô hình học.
Các bước, cho mỗi thiết lập E1 (mức "thường", tuổi 0, Δ 9) và E2 ("khắc nghiệt", tuổi 9, Δ 3), như thí nghiệm 5:
  1) Mô phỏng 24 cửa sổ thời tiết ngày train, đếm xác suất sống của liên kết; mô phỏng 12 cửa sổ ngày val, với mỗi truy vấn
     tính trễ và xác suất đứt của các tuyến ứng viên   (src/ngau_nhien/thu_trong_so.py)
  2) Quét α = 0; 0,1; …; 1 và chấm theo giai đoạn bão / nhẹ   (src/ngau_nhien/phan_tich_trong_so.py)
Không dùng dữ liệu Yagi. Kết quả: ket_qua_tu_chay/score/.
"""
import argparse

from _chung import GOC, bang, buoc, doc, kiem_dau_vao, so, thu_muc, tieu_de

TS = {"E1": ["--muc", "thường", "--tuoi", "0", "--delta", "9"],
      "E2": ["--muc", "khắc nghiệt", "--tuoi", "9", "--delta", "3"]}
A = ("0.0", "0.1", "0.2", "0.5", "0.8", "1.0")
GD = (("gd15_bão", "bão"), ("gd15_nhẹ", "nhẹ"))

ap = argparse.ArgumentParser()
ap.add_argument("--thu", action="store_true", help="chạy thử nhanh để kiểm tra máy")
a = ap.parse_args()
kiem_dau_vao("data/thoi_tiet_vn_mr.npz", "dataset/oracle_dataset_v5.npz", "dataset/v5/split_info.json")
D = thu_muc("score", a.thu)
tieu_de("SCORE CÂN BẰNG TỐC ĐỘ VÀ MỨC ĐẢM BẢO",
        "Khi bão và khi trời nhẹ, nên đặt α (trọng số tốc độ) bao nhiêu? α = 1 là chỉ xét tốc độ.", a.thu)

for e, ts in TS.items():
    so_cs = "1,1,1,1" if a.thu else "16,8,8,4"
    n_val = sum(map(int, so_cs.split(",")[2:]))
    buoc(f"{e}: mô phỏng cửa sổ train + val", ["src.ngau_nhien.thu_trong_so", *ts, "--so-cua-so", so_cs, "--seed", "61",
                                               "--buoc-fit", "3", "--out", D / e],
         D / "nhat_ky" / f"{e}_thu_trong_so.log", xong=D / e / "tam" / f"val_{n_val - 1}.npz")
    buoc(f"{e}: quét α và chấm", ["src.ngau_nhien.phan_tich_trong_so", "--vao", D / e],
         D / "nhat_ky" / f"{e}_phan_tich.log")

# ---- in kết quả
print("\nKẾT QUẢ (G = % chi phí trễ vượt hậu kiểm; cập nhật giai đoạn 15 phút)")
for e in TS:
    d = doc(D / e / "tong_hop.json")
    print(f"{e} ({'thông tin mới' if e == 'E1' else 'thông tin cũ 3 phút'}), {d['so_truy_van']} truy vấn")
    hang = []
    for k, ten in GD:
        g = d["theo_giai_doan"][k]
        hang.append([f"{ten}: G (%)"] + [so(g["ti_le"][x]["G_%"], 1) for x in A] + [so(g["it_dut_nhat_trong_ung_vien"]["G_%"], 1)])
        hang.append([f"{ten}: số truy vấn đứt"] + [str(g["ti_le"][x]["so_truy_van_dut"]) for x in A] +
                    [str(g["it_dut_nhat_trong_ung_vien"]["so_truy_van_dut"])])
    bang(["", *[f"α={x[:3].replace('.', ',')}" for x in A], "biết trước*"], hang)
print("* biết trước = trong các ứng viên chọn tuyến ít đứt nhất theo hậu kiểm (không làm được thật, chỉ để so).")
print("Đọc kết quả: giai đoạn bão, α nhỏ (nghiêng về đảm bảo) có giảm số truy vấn đứt mà không tăng trễ nhiều không;")
print("giai đoạn nhẹ, α nào cho G nhỏ nhất. Kết quả gốc: bão α ≈ 0,1, nhẹ α 0,5–0,8 → nhóm dùng 1/9 và 8/2.\n")

# ---- so với kết quả gốc
if not a.thu:
    print("So với kết quả gốc results/thu_trong_so/ (chạy 27/10, TRƯỚC khi nhóm sửa dự báo quỹ đạo; mã hiện tại đã có bản sửa,")
    print("nên E1 (tuổi 0) phải khớp, E2 có thể lệch nhẹ):")
    hang = []
    for e in TS:
        moi, cu = doc(D / e / "tong_hop.json"), GOC / "results/thu_trong_so" / e / "tong_hop.json"
        if not cu.exists():
            continue
        cu = doc(cu)
        for k, ten in GD:
            m, c = moi["theo_giai_doan"][k]["ti_le"], cu["theo_giai_doan"][k]["ti_le"]
            lech = max(abs(m[x]["G_%"] - c[x]["G_%"]) for x in m)
            hang.append([f"{e} {ten}", "khớp" if lech < 1e-9 else f"lệch tối đa {so(lech, 3)} điểm G",
                         f"{m['0.1']['so_truy_van_dut']} / {c['0.1']['so_truy_van_dut']}"])
    bang(["", "G mọi α", "đứt ở α=0,1 (mới / gốc)"], hang)
