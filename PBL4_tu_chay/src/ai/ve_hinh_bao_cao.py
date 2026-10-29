"""
Vẽ hình cho báo cáo từ các file kết quả trong results/.

    python src/ai/ve_hinh_bao_cao.py --results results --out results/hinh
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
                     "font.size": 10.5, "axes.edgecolor": INK2, "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2})


def clean(ax):
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def save(fig, path):
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Đã lưu", path)


def hinh_kien_truc(out):
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 36)
    ax.axis("off")

    def box(x, y, w, h, text, fill="#f3f2ee"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                    fc=fill, ec=INK2, lw=1))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9.5, color=INK)

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12, color=INK2, lw=1))

    box(1, 13, 14, 10, "Snapshot SAGSIN\nma trận trễ W\nloại nút, tầng")
    box(19, 13, 14, 10, "Đặc trưng nút\n+ cờ nút đích")
    box(37, 13, 15, 10, "GNN 4 lớp\n2 kênh gom tin\n(trung bình, theo trễ)")
    box(57, 22, 20, 11, "Diffusion (DM)\nnhiễu → khử nhiễu 1–2 bước\ndự đoán y0, CFG", fill="#e3eefa")
    box(57, 3, 20, 11, "Softmax (đối chứng)\nchấm điểm ứng viên\n→ xác suất", fill="#fbe8df")
    box(82, 13, 16, 10, "Next-hop\n(chỉ ứng viên hợp lệ)\nlặp tới đích")
    arrow(15.8, 18, 18.2, 18)
    arrow(33.8, 18, 36.2, 18)
    arrow(52.8, 19.5, 56.2, 26)
    arrow(52.8, 16.5, 56.2, 9)
    arrow(77.8, 27.5, 81.2, 20)
    arrow(77.8, 8.5, 81.2, 16)
    ax.text(44.5, 10.5, "chạy 1 lần cho cả đường", ha="center", fontsize=8.5, color=INK2)
    save(fig, out / "h1_kien_truc.png")


def hinh_dong_gop(rd, out):
    def nh(tag):
        return json.loads((rd / f"{tag}_results.json").read_text())["test_nexthop"]["mo_hinh"] * 100
    series = [("DM cũ (dự đoán nhiễu, MSE)", ["eps_0lop", "eps_4lop"], BLUE),
              ("DM mới (dự đoán y0, cross-entropy)", ["dmx0_0lop", "dmx0_4lop"], ORANGE),
              ("Softmax (không diffusion)", ["clsA_73", "clsB_73"], AQUA)]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    x, w = [0, 1.25], 0.36
    for k, (name, tags, c) in enumerate(series):
        if not all((rd / f"{t}_results.json").exists() for t in tags):
            continue
        vals = [nh(t) for t in tags]
        pos = [i + (k - 1) * w for i in x]
        bars = ax.bar(pos, vals, width=w, color=c, edgecolor="white", linewidth=2, label=name, zorder=3)
        for r in bars:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 1.2, f"{r.get_height():.1f}%",
                    ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(["Không GNN (0 lớp)", "Có GNN (4 lớp)"])
    ax.set_ylim(0, 112)
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel("Chọn đúng next-hop trên test (%)")
    ax.legend(frameon=False, fontsize=9, loc="upper left", ncol=1)
    clean(ax)
    save(fig, out / "h2_dong_gop.png")


def hinh_so_buoc(rd, out):
    g = json.loads((rd / "so_sanh_gnn.json").read_text())["bien_the"]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    steps = [1, 2, 5, 20]
    for (name, c) in (("Có GNN (4 lớp)", ORANGE), ("Không GNN (0 lớp)", BLUE)):
        grid = g[name]["val_luoi"]
        vals = [grid[f"{n}b_w1"] * 100 for n in steps]
        ax.plot(range(len(steps)), vals, color=c, lw=2, marker="o", ms=7, label=name, zorder=3)
        for i, v in enumerate(vals):
            ax.text(i, v + 1.2, f"{v:.1f}", ha="center", fontsize=8.5)
    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels([f"{n} bước" for n in steps])
    ax.set_ylim(70, 103)
    ax.set_ylabel("Next-hop trên val, w = 1 (%)")
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    clean(ax)
    save(fig, out / "h3_so_buoc.png")


OFFSETS = {
    ("nhãn đơn", "DM", "DDIM tất định"): (2, 13), ("nhãn đơn", "DM", "η=1 (DDPM)"): (-8, 8),
    ("nhãn đơn", "DM", "η=1, w=0.5"): (7, 5), ("nhãn đơn", "Softmax", "T=1"): (9, -11),
    ("nhãn đơn", "Softmax", "T=2"): (8, -4), ("nhãn đơn", "Softmax", "T=3"): (-9, 4),
    ("nhãn nhiều tuyến", "DM", "DDIM tất định"): (8, 3), ("nhãn nhiều tuyến", "DM", "η=1 (DDPM)"): (8, -13),
    ("nhãn nhiều tuyến", "DM", "η=1, w=0.5"): (8, -4), ("nhãn nhiều tuyến", "Softmax", "T=1"): (-9, 5),
    ("nhãn nhiều tuyến", "Softmax", "T=2"): (-9, 5), ("nhãn nhiều tuyến", "Softmax", "T=3"): (-9, 5),
}


def hinh_du_phong(rd, out):
    r = json.loads((rd / "so_sanh_dm_softmax.json").read_text())["nhieu_tuyen"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, nhan in zip(axes, ("nhãn đơn", "nhãn nhiều tuyến")):
        for name, c, mk in ((f"DM, {nhan}", ORANGE, "o"), (f"Softmax, {nhan}", AQUA, "s")):
            pts = r[name]
            xs = [(v["tre_du_phong_so_Dijkstra_chay_lai"] - 1) * 100 for v in pts.values()]
            ys = [v["co_san_du_phong"] * 100 for v in pts.values()]
            ax.plot(xs, ys, color=c, lw=1.5, marker=mk, ms=8, label=name.split(",")[0], zorder=3,
                    markeredgecolor="white", markeredgewidth=1.5)
            for (lab, _), xx, yy in zip(pts.items(), xs, ys):
                off = OFFSETS.get((nhan, name.split(",")[0], lab), (7, 5))
                ax.annotate(lab, (xx, yy), textcoords="offset points", xytext=off, fontsize=8, color=INK2,
                            ha="right" if off[0] < 0 else "left",
                            bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.9), zorder=4)
        ax.set_title(f"Học bằng {nhan}", loc="left", fontsize=10.5)
        ax.set_xlabel("Đường dự phòng trễ hơn Dijkstra chạy lại (%)")
        ax.set_xlim(-0.3, 7)
        clean(ax)
    axes[0].set_ylabel("Có sẵn đường dự phòng khi hỏng 1 liên kết (%)")
    axes[0].set_ylim(0, 90)
    axes[0].legend(frameon=False, fontsize=9, loc="upper right")
    save(fig, out / "h4_du_phong.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/hinh")
    a = ap.parse_args()
    rd, out = Path(a.results), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    hinh_kien_truc(out)
    hinh_dong_gop(rd, out)
    hinh_so_buoc(rd, out)
    hinh_du_phong(rd, out)


if __name__ == "__main__":
    main()
