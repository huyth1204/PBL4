"""
so_do_huong_moi.py — Sơ đồ hướng đề tài mới (thời tiết cực đoan + định tuyến khẩn cấp đa luồng) để trình giảng viên.

    python bao_cao/so_do_huong_moi.py        # → bao_cao/ket_qua/so_do_huong_moi.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "TeX Gyre Termes", "DejaVu Serif"]})
DEN, XAM = "#000000", "#555555"
NEN = {"du_lieu": "#eef4fb", "mo_phong": "#fdf3e6", "bai_toan": "#f3f3f3", "dm": "#e8f5ee", "dc": "#f7f7f7",
       "danh_gia": "#fff8dc", "da_co": "#ffffff"}


def hop(ax, x, y, w, h, tieu_de, dong=(), nen="#ffffff", vien=DEN, nang=False, co=10.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.015",
                                fc=nen, ec=vien, lw=1.6 if nang else 1.0))
    ax.text(x + w / 2, y + h - 0.022, tieu_de, ha="center", va="top", fontsize=co, fontweight="bold", color=DEN)
    for i, d in enumerate(dong):
        ax.text(x + w / 2, y + h - 0.058 - i * 0.03, d, ha="center", va="top", fontsize=co - 1.5, color=DEN)


def mui_ten(ax, a, b, nhan=None, net="-"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=14, lw=1.2, color=DEN, linestyle=net,
                                 shrinkA=2, shrinkB=2))
    if nhan:
        ax.text((a[0] + b[0]) / 2 + 0.01, (a[1] + b[1]) / 2, nhan, fontsize=8.5, color=XAM, va="center", style="italic")


def main():
    fig = plt.figure(figsize=(11, 8.6))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.985, "Hướng đề tài mới: định tuyến khẩn cấp đa luồng trong SAGSIN dưới thời tiết cực đoan",
            ha="center", va="top", fontsize=14, fontweight="bold")

    # Cột trái: kết quả đã có → lý do chuyển hướng
    ax.text(0.15, 0.935, "Đã có (nền tảng)", ha="center", fontsize=11, fontweight="bold", color=XAM)
    hop(ax, 0.02, 0.79, 0.26, 0.125, "GNN + DM đơn luồng", ["trùng tối ưu 98,3% (softmax 98,3%)",
                                                            "chậm hơn Dijkstra hàng chục lần"], NEN["da_co"])
    hop(ax, 0.02, 0.635, 0.26, 0.125, "Thông tin trễ bị cũ", ["mô hình học có lợi khi cũ > 1 phút",
                                                             "softmax ≥ DM, DM chưa có ý nghĩa"], NEN["da_co"])
    hop(ax, 0.02, 0.48, 0.26, 0.125, "Mưa trôi + lịch sử", ["lịch sử không giúp",
                                                            "trần lý thuyết chỉ +0,01–0,07 R²"], NEN["da_co"])
    hop(ax, 0.02, 0.265, 0.26, 0.175, "Kết luận dẫn tới hướng mới",
        ["Một quyết định đơn lẻ (next-hop):", "DM không có lợi thế riêng.", "Lợi thế lý thuyết của DM: đầu ra",
         "tổ hợp, nhiều chiều, phụ thuộc nhau."], "#f3f3f3", nang=True)
    for y0, y1 in ((0.784, 0.766), (0.629, 0.611), (0.474, 0.446)):
        mui_ten(ax, (0.15, y0), (0.15, y1))

    # Cột phải: luồng chính
    hop(ax, 0.34, 0.80, 0.30, 0.125, "Dữ liệu thật", ["Quỹ đạo Starlink (TLE) · ERA5 mưa 2024–2025",
                                                      "CAPE (giông) · gió 850/700 hPa (Open-Meteo)"], NEN["du_lieu"])
    hop(ax, 0.67, 0.80, 0.31, 0.125, "Sự kiện thật: bão Yagi 7–8/9/2024",
        ["Hải Phòng: mưa 208 mm/ngày, giật 42 m/s", "9.235 trạm BTS mất kết nối [1]"], NEN["du_lieu"])
    hop(ax, 0.34, 0.615, 0.64, 0.145, "Bộ mô phỏng SAGSIN Việt Nam dưới thời tiết cực đoan (73 nút)",
        ["Mưa → suy hao ITU-R P.838 → giảm dung lượng / đứt liên kết  ·  giông, gió → UAV hạ cánh",
         "Mạng mặt đất hỏng → lưu lượng khẩn cấp dồn sang vệ tinh / HAPS  ·  trễ hàng đợi M/M/1 theo tải"],
        NEN["mo_phong"])
    mui_ten(ax, (0.49, 0.80), (0.49, 0.76))
    mui_ten(ax, (0.825, 0.80), (0.825, 0.76))
    hop(ax, 0.34, 0.445, 0.64, 0.135, "Bài toán: định tuyến đồng thời K = 20–40 luồng khẩn cấp",
        ["Mỗi luồng 8 tuyến ứng viên (Yen)  ·  tuyến của luồng này làm nghẽn luồng kia",
         "Mục tiêu: trễ trung bình theo lưu lượng nhỏ nhất  ·  ba mức tải: nhẹ / vừa / nặng"], NEN["bai_toan"])
    mui_ten(ax, (0.66, 0.615), (0.66, 0.58))
    hop(ax, 0.34, 0.255, 0.30, 0.155, "DM đa luồng (đề xuất)",
        ["Khử nhiễu ĐỒNG THỜI lựa chọn", "của mọi luồng (attention giữa luồng)", "Lấy 16 mẫu, chọn mẫu tốt nhất"],
        NEN["dm"], nang=True)
    hop(ax, 0.68, 0.255, 0.30, 0.155, "Đối chứng",
        ["Dijkstra độc lập · Dijkstra tuần tự", "Tìm kiếm cục bộ (1 lần, 10 lần)",
         "Softmax cùng kiến trúc · cận dưới Frank–Wolfe"], NEN["dc"])
    mui_ten(ax, (0.495, 0.445), (0.495, 0.41))
    mui_ten(ax, (0.825, 0.445), (0.825, 0.41))
    hop(ax, 0.34, 0.035, 0.64, 0.185, "Đánh giá trên bão Yagi (giữ riêng, huấn luyện trên các cơn bão khác)",
        ["Tiêu chí chốt TRƯỚC khi chạy (đã lưu trong git):",
         "C1, C2: tốt hơn Dijkstra tuần tự và softmax, khoảng tin cậy 95% dưới 0",
         "C3: cách lời giải tốt nhất ≤ 2%   ·   C4: nhanh hơn tìm kiếm cục bộ 10 lần ≥ 10×",
         "Báo đầy đủ kể cả khi DM không đạt"], NEN["danh_gia"], nang=True)
    mui_ten(ax, (0.495, 0.255), (0.56, 0.22))
    mui_ten(ax, (0.825, 0.255), (0.76, 0.22))
    mui_ten(ax, (0.285, 0.36), (0.335, 0.49), nhan=None, net="--")
    ax.text(0.02, 0.012, "[1] Effects of Typhoon Yagi in Vietnam (Wikipedia). Số liệu mưa, gió: ERA5 qua Open-Meteo, "
            "dữ liệu của dự án.", fontsize=8, color=XAM)

    ra = Path(__file__).resolve().parent / "ket_qua" / "so_do_huong_moi.png"
    ra.parent.mkdir(exist_ok=True)
    fig.savefig(ra, dpi=200)
    print(ra)


if __name__ == "__main__":
    main()
