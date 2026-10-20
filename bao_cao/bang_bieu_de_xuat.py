"""
bang_bieu_de_xuat.py — Biểu đồ tổng hợp bằng chứng cho đề xuất chuyển hướng đề tài.
Mọi số đọc từ tệp kết quả của dự án (không gõ tay), trừ các số ghi rõ nguồn trong chú thích.

    python bao_cao/bang_bieu_de_xuat.py     # → bao_cao/ket_qua/bieu_do_bang_chung.png
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

GOC = Path(__file__).resolve().parents[1]
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "TeX Gyre Termes", "DejaVu Serif"],
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 11})
MAU = {"Dijkstra": "#8a8984", "Softmax": "#eb6834", "DM": "#2a78d6", "Dijkstra khử nhiễu": "#b8b6af"}


def nhan_cot(ax, cot, dinh_dang="{:.1f}%", dich=1.0):
    for c in cot:
        ax.text(c.get_x() + c.get_width() / 2, c.get_height() + dich, dinh_dang.format(c.get_height()).replace(".", ","),
                ha="center", va="bottom", fontsize=10)


def main():
    ss = json.load(open(GOC / "results" / "so_sanh_dm_softmax.json", encoding="utf-8"))
    nd = json.load(open(GOC / "results" / "quy_mo" / "nhieu_do.json", encoding="utf-8"))["373"]["0.4"]
    a = json.load(open(GOC / "results" / "do_cu" / "phan_tich.json", encoding="utf-8"))["hieu"]

    fig, axs = plt.subplots(2, 2, figsize=(12, 8.2))

    # (a) đơn luồng
    ax = axs[0, 0]
    dm = ss["do_chinh_xac"]["DM, nhãn đơn"]
    sm = ss["do_chinh_xac"]["Softmax, nhãn đơn"]
    v = [100.0, sm["rollout"]["trung_khop_duong_oracle"] * 100, dm["rollout"]["trung_khop_duong_oracle"] * 100]
    cot = ax.bar(["Dijkstra\n(là đáp án)", "Softmax", "DM"], v, color=[MAU["Dijkstra"], MAU["Softmax"], MAU["DM"]], width=0.6)
    nhan_cot(ax, cot, dich=0.1)
    ax.set_ylim(95, 100.8)
    ax.set_ylabel("% trùng đường tối ưu")
    ax.set_title("(a) Đơn luồng, điều kiện thường: DM ngang softmax", loc="left", fontsize=12)
    ax.text(0.98, 0.97, f"Thời gian mỗi đường (cùng lần đo):\nsoftmax {sm['ms_tung_duong']:.1f} ms, DM {dm['ms_tung_duong']:.1f} ms"
            .replace(".", ","), transform=ax.transAxes, ha="right", va="top", fontsize=9.5, color="#333")

    # (b) số đo sai 40 %
    ax = axs[0, 1]
    ten = ["Dijkstra trên trễ đo", "Dijkstra khử nhiễu (biết σ)", "DM học với nhiễu"]
    v = [(nd[t]["tre_so_voi_toi_uu"] - 1) * 100 for t in ten]
    cot = ax.bar(["Dijkstra\ntrên số đo", "Dijkstra\nkhử nhiễu", "DM học\nvới nhiễu"], v,
                 color=[MAU["Dijkstra"], MAU["Dijkstra khử nhiễu"], MAU["DM"]], width=0.6)
    nhan_cot(ax, cot, "+{:.1f}%", 0.4)
    ax.set_ylabel("trễ vượt tối ưu (%)")
    ax.set_title("(b) Số đo trễ sai 40%, mạng 373 nút: DM tốt nhất*", loc="left", fontsize=12)
    ax.text(0.98, 0.9, "* chưa so với softmax học cùng nhiễu", transform=ax.transAxes, ha="right", fontsize=9.5,
            color="#333")

    # (c) tuyến dự phòng
    ax = axs[1, 0]
    nt = ss["nhieu_tuyen"]
    v = [nt["DM, nhãn nhiều tuyến"]["η=1 (DDPM)"]["co_san_du_phong"] * 100,
         nt["Softmax, nhãn nhiều tuyến"]["T=1"]["co_san_du_phong"] * 100,
         nt["Softmax, nhãn nhiều tuyến"]["T=2"]["co_san_du_phong"] * 100]
    cot = ax.bar(["DM\n(η = 1)", "Softmax\n(T = 1)", "Softmax\n(T = 2)"], v,
                 color=[MAU["DM"], MAU["Softmax"], MAU["Softmax"]], width=0.6)
    nhan_cot(ax, cot)
    ax.set_ylim(0, 85)
    ax.set_ylabel("% tình huống hỏng liên kết\ncó sẵn tuyến dự phòng")
    ax.set_title("(c) Sinh nhiều tuyến dự phòng: softmax hơn DM", loc="left", fontsize=12)

    # (d) thông tin cũ
    ax = axs[1, 1]
    tuoi = ["0", "1", "3", "9", "15"]
    nhan = ["0", "20 s", "1 phút", "3 phút", "5 phút"]
    ax.axhline(0, color="#555", lw=1)
    for j, m in enumerate(("DM", "Softmax")):
        h = np.array([a[t][m]["G"]["hieu"] for t in tuoi])
        lo = np.array([a[t][m]["G"]["ci95"][0] for t in tuoi])
        hi = np.array([a[t][m]["G"]["ci95"][1] for t in tuoi])
        ax.errorbar(np.arange(5) + (j - 0.5) * 0.18, h, yerr=[h - lo, hi - h], fmt="o" if m == "DM" else "^",
                    color=MAU[m], capsize=3, label=m)
    ax.set_xticks(range(5))
    ax.set_xticklabels(nhan)
    ax.set_xlabel("Độ cũ của thông tin trễ")
    ax.set_ylabel("G: mô hình − Dijkstra + lịch quỹ đạo\n(điểm %, âm = mô hình tốt hơn)")
    ax.set_title("(d) Thông tin cũ: chỉ softmax tốt hơn có ý nghĩa", loc="left", fontsize=12)
    ax.legend(frameon=False, loc="lower left")

    fig.suptitle("Bằng chứng từ thực nghiệm: với quyết định đơn luồng, DM chưa có lợi thế riêng",
                 x=0.01, ha="left", fontsize=14, fontweight="bold")
    fig.tight_layout()
    ra = GOC / "bao_cao" / "ket_qua" / "bieu_do_bang_chung.png"
    fig.savefig(ra, dpi=200)
    print(ra)


if __name__ == "__main__":
    main()


def hai_do_thi():
    """Hai đồ thị nhỏ cho bản đề xuất gọn (Word): do_thi_1_dm_softmax.png, do_thi_2_thong_tin_cu.png."""
    ss = json.load(open(GOC / "results" / "so_sanh_dm_softmax.json", encoding="utf-8"))
    nd = json.load(open(GOC / "results" / "quy_mo" / "nhieu_do.json", encoding="utf-8"))["373"]["0.4"]
    a = json.load(open(GOC / "results" / "do_cu" / "phan_tich.json", encoding="utf-8"))["hieu"]
    kq = GOC / "bao_cao" / "ket_qua"

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.1))
    ten = ["Dijkstra trên trễ đo", "Dijkstra khử nhiễu (biết σ)", "DM học với nhiễu"]
    cot = axs[0].bar(["Dijkstra\ntrên số đo", "Dijkstra\nkhử nhiễu", "DM"], [(nd[t]["tre_so_voi_toi_uu"] - 1) * 100 for t in ten],
                     color=[MAU["Dijkstra"], MAU["Dijkstra khử nhiễu"], MAU["DM"]], width=0.6)
    nhan_cot(axs[0], cot, "+{:.1f}%", 0.4)
    axs[0].set_ylabel("trễ vượt tối ưu (%) — thấp là tốt")
    axs[0].set_ylim(0, 33)
    axs[0].set_title("(a) Số đo trễ sai 40%, 373 nút: DM tốt nhất", loc="left", fontsize=12)
    nt = ss["nhieu_tuyen"]
    v = [nt["DM, nhãn nhiều tuyến"]["η=1 (DDPM)"]["co_san_du_phong"] * 100,
         nt["Softmax, nhãn nhiều tuyến"]["T=1"]["co_san_du_phong"] * 100,
         nt["Softmax, nhãn nhiều tuyến"]["T=2"]["co_san_du_phong"] * 100]
    cot = axs[1].bar(["DM", "Softmax\n(T = 1)", "Softmax\n(T = 2)"], v, color=[MAU["DM"], MAU["Softmax"], MAU["Softmax"]], width=0.6)
    nhan_cot(axs[1], cot)
    axs[1].set_ylim(0, 85)
    axs[1].set_ylabel("% tình huống có sẵn tuyến dự phòng\n— cao là tốt")
    axs[1].set_title("(b) Sinh tuyến dự phòng: softmax hơn DM", loc="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(kq / "do_thi_1_dm_softmax.png", dpi=200)

    fig, ax = plt.subplots(figsize=(11, 4.1))
    tuoi, nhan = ["0", "1", "3", "9", "15"], ["0", "20 giây", "1 phút", "3 phút", "5 phút"]
    ax.axhspan(0, 11, color="#8a8984", alpha=0.08)
    ax.axhspan(-16, 0, color="#2a78d6", alpha=0.05)
    ax.text(2.5, 10.4, "vùng Dijkstra + lịch quỹ đạo tốt hơn", ha="center", va="top", fontsize=10, color="#555")
    ax.text(2.5, -15.4, "vùng mô hình học tốt hơn", ha="center", va="bottom", fontsize=10, color="#2a78d6")
    ax.axhline(0, color="#555", lw=1)
    for j, m in enumerate(("DM", "Softmax")):
        h = np.array([a[t][m]["G"]["hieu"] for t in tuoi])
        lo = np.array([a[t][m]["G"]["ci95"][0] for t in tuoi])
        hi = np.array([a[t][m]["G"]["ci95"][1] for t in tuoi])
        x = np.arange(5) + (j - 0.5) * 0.16
        ax.errorbar(x, h, yerr=[h - lo, hi - h], fmt="o-" if m == "DM" else "^-", color=MAU[m], capsize=3, label=m, lw=1.2)
    ax.set_xticks(range(5))
    ax.set_xticklabels(nhan)
    ax.set_xlabel("Độ cũ của thông tin trễ khi bộ định tuyến quyết định")
    ax.set_ylabel("chênh lệch chi phí G (điểm %)\nso với Dijkstra + lịch quỹ đạo")
    ax.set_ylim(-16, 11)
    ax.set_title("Thông tin cũ từ khoảng 1 phút, mô hình học bắt đầu có lợi; chỉ softmax ở 3 phút có ý nghĩa thống kê",
                 loc="left", fontsize=12)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(kq / "do_thi_2_thong_tin_cu.png", dpi=200)
    print(kq / "do_thi_1_dm_softmax.png", kq / "do_thi_2_thong_tin_cu.png")


if __name__ == "__main__" and "--hai" in __import__("sys").argv:
    hai_do_thi()
