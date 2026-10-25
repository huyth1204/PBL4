"""
tien_ich.py — Các hàm vẽ và kiểm tra dùng chung cho các tệp huong_dan/buoc_*.py.

Mọi kết quả đều hiện bằng hình hoặc bảng màu (✅ / ❌), không in chữ ra terminal.
"""
from __future__ import annotations

import contextlib
import importlib
import io
import sys
import time
from pathlib import Path

import numpy as np

GOC = Path(__file__).resolve().parents[1]
if str(GOC) not in sys.path:
    sys.path.insert(0, str(GOC))

# Console Windows mặc định cp1252 -> in chữ có dấu là UnicodeEncodeError.
# Ép stdout/stderr sang UTF-8 ngay khi nạp module, trước mọi lệnh print.
for _stream in (sys.stdout, sys.stderr):
    _reconfigure = getattr(_stream, "reconfigure", None)
    if _reconfigure is not None:
        try:
            _reconfigure(encoding="utf-8")
        except (ValueError, OSError):
            pass

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

DEMO = GOC / "huong_dan" / "du_lieu_demo"
THOI_TIET = GOC / "data" / "thoi_tiet_vn.npz"
MAU = {"xanh": "#2a78d6", "cam": "#eb6834", "luc": "#1baf7a", "vang": "#eda100", "xam": "#8a8984", "den": "#0b0b0b"}
TANG = {"gateway": 0.0, "user": 0.0, "uav": 1.0, "haps": 2.0, "leo": 3.0}
TEN_TANG = {0.0: "Mặt đất", 1.0: "UAV (2 km)", 2.0: "HAPS (20 km)", 3.0: "Vệ tinh LEO (550 km)"}
KY_HIEU = {"gateway": ("s", MAU["den"]), "user": ("o", MAU["den"]), "uav": ("^", MAU["luc"]),
           "haps": ("D", MAU["vang"]), "leo": (".", MAU["xam"])}

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "TeX Gyre Termes", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#52514e", "figure.dpi": 110})


# ------------------------------------------------------------------ kiểm tra (✅ / ❌)
def _trong_notebook():
    try:
        from IPython import get_ipython
        return get_ipython() is not None and "IPKernelApp" in get_ipython().config
    except Exception:  # noqa: BLE001
        return False


def ve_bang_kiem_tra(ket_qua, tieu_de="Kết quả kiểm tra"):
    """Bảng kiểm tra vẽ bằng matplotlib (dùng khi chạy file .py trong PyCharm): hàng xanh = đạt, đỏ = lỗi."""
    n = max(len(ket_qua), 1)
    so_dat = sum(k["dat"] for k in ket_qua)
    fig = plt.figure(figsize=(12, 0.42 * n + 1.0))
    ax = fig.add_axes([0.01, 0.02, 0.98, 0.98 - 0.55 / (0.42 * n + 1.0)])
    ax.set_xlim(0, 1)
    ax.set_ylim(n, 0)
    ax.axis("off")
    for i, k in enumerate(ket_qua):
        ax.add_patch(FancyBboxPatch((0.0, i + 0.08), 1.0, 0.84, boxstyle="round,pad=0,rounding_size=0.01",
                                    fc="#eaf7f1" if k["dat"] else "#fdecea", ec="none"))
        ax.text(0.018, i + 0.5, "✔" if k["dat"] else "✘", color="#1baf7a" if k["dat"] else "#d62728", fontsize=15,
                va="center", ha="center", fontfamily="DejaVu Sans", fontweight="bold")
        ax.text(0.04, i + 0.5, k["ten"], va="center", fontsize=10.5)
        ct = str(k.get("chi_tiet", ""))
        ax.text(0.47, i + 0.5, ct if len(ct) < 95 else ct[:92] + "…", va="center", fontsize=9.5, color="#52514e")
        if "giay" in k:
            ax.text(0.995, i + 0.5, f"{k['giay']:.1f} s", va="center", ha="right", fontsize=9, color="#8a8984")
    mau = "#1baf7a" if so_dat == len(ket_qua) else "#d62728"
    fig.suptitle(f"{tieu_de}: {so_dat}/{len(ket_qua)} đạt", x=0.01, ha="left", fontsize=13, fontweight="bold", color=mau)
    try:
        fig.canvas.manager.set_window_title(tieu_de)
    except Exception:  # noqa: BLE001
        pass
    return fig


def hien_thi(ham, tieu_de):
    """Chạy một bước: vẽ các hình của bước đó, thêm bảng kiểm tra, rồi mở cửa sổ hình (PyCharm: tab Plots)."""
    kq = ham()
    bang_kiem_tra(kq, tieu_de)
    if not _trong_notebook():
        plt.show()
    return kq


def bang_kiem_tra(ket_qua, tieu_de="Kết quả kiểm tra"):
    """ket_qua: list các dict {ten, dat, chi_tiet, giay}. Notebook: bảng HTML màu; file .py: bảng vẽ bằng matplotlib."""
    so_dat = sum(k["dat"] for k in ket_qua)
    if _trong_notebook():
        from IPython.display import HTML, display
        hang = "".join(
            f"<tr style='background:{'#eaf7f1' if k['dat'] else '#fdecea'}'>"
            f"<td style='font-size:18px;text-align:center'>{'✅' if k['dat'] else '❌'}</td>"
            f"<td style='text-align:left'>{k['ten']}</td><td style='text-align:left;color:#52514e'>{k.get('chi_tiet', '')}</td>"
            f"<td style='text-align:right;color:#8a8984'>{k.get('giay', 0):.1f} s</td></tr>" for k in ket_qua)
        mau = "#1baf7a" if so_dat == len(ket_qua) else "#eb6834"
        display(HTML(f"<div style='font-family:Times New Roman,serif'><b style='font-size:15px'>{tieu_de}: "
                     f"<span style='color:{mau}'>{so_dat}/{len(ket_qua)} đạt</span></b>"
                     f"<table style='border-collapse:collapse;margin-top:6px;font-size:14px'>"
                     f"<tr style='background:#f3f2ee'><th></th><th style='text-align:left'>Kiểm tra</th>"
                     f"<th style='text-align:left'>Chi tiết</th><th>Thời gian</th></tr>{hang}</table></div>"))
    else:
        ve_bang_kiem_tra(ket_qua, tieu_de)
        print(f"{tieu_de}: {so_dat}/{len(ket_qua)} đạt")
    return so_dat == len(ket_qua)


def kiem_tra(ten, ham):
    """Chạy ham() -> (dat: bool, chi_tiet: str). Lỗi Python cũng được ghi lại thành ❌."""
    t0 = time.time()
    try:
        dat, chi_tiet = ham()
    except Exception as e:  # noqa: BLE001
        dat, chi_tiet = False, f"{type(e).__name__}: {e}"
    return {"ten": ten, "dat": bool(dat), "chi_tiet": chi_tiet, "giay": time.time() - t0}


from test.tien_ich_test import BoQua  # noqa: E402


def chay_file_test(ten_module):
    """Chạy mọi hàm test_* trong một file test của dự án, trả bảng ✅ / ❌ / ⏭."""
    m = importlib.import_module(ten_module)
    kq = []
    for ten, f in sorted(vars(m).items()):
        if not (ten.startswith("test_") and callable(f)):
            continue
        t0, buf = time.time(), io.StringIO()
        nhan = ten.replace("test_", "").replace("_", " ")
        try:
            with contextlib.redirect_stdout(buf):
                f()
            # "bỏ qua" trong chính HÀM test hoặc in ra khi chạy đều là SKIP, không phải đạt:
            # trước đây nhánh return sớm bị tính là ✅ nên bộ test xanh giả.
            ghi = buf.getvalue().strip()
            bo_qua = "bỏ qua" in ghi or "bỏ qua" in (f.__doc__ or "")
            kq.append({"ten": nhan, "dat": not bo_qua, "chi_tiet": ghi if bo_qua else "",
                       "giay": time.time() - t0})
        except BoQua as e:
            kq.append({"ten": nhan, "dat": False, "chi_tiet": f"bỏ qua: {e}",
                       "giay": time.time() - t0})
        except Exception as e:  # noqa: BLE001
            kq.append({"ten": nhan, "dat": False,
                       "chi_tiet": f"{type(e).__name__}: {e}", "giay": time.time() - t0})
    return kq


# ------------------------------------------------------------------ dữ liệu
def doc_demo(n):
    from src.ai.data_v2 import SagsinData
    return SagsinData(DEMO / f"n{n}.npz")


def toa_do(data):
    """x của mỗi nút: vĩ độ thật cho nút mặt đất / trên không; vệ tinh lấy trung bình các nút nó nối tới."""
    tt = np.load(THOI_TIET, allow_pickle=True)
    vi_do = {str(n): float(la) for n, la in zip(tt["site_names"], tt["lat"])}
    ten = [str(x) for x in data.node_order]
    x = np.array([vi_do.get(t, np.nan) for t in ten])
    y = np.array([TANG[str(k)] for k in data.kind])
    A = (data.W.max(0) > 0)
    for _ in range(6):
        for i in np.flatnonzero(np.isnan(x)):
            ke = np.flatnonzero(A[i] & ~np.isnan(x))
            if len(ke):
                x[i] = x[ke].mean()
    x[np.isnan(x)] = np.nanmean(x)
    rng = np.random.default_rng(0)
    ve_tinh = y == 3.0
    x[ve_tinh] += rng.normal(0, 0.35, ve_tinh.sum())
    y = y + np.where(ve_tinh, rng.uniform(-0.28, 0.28, len(y)), 0.0)
    return x, y


def ve_mang(data, s, tuyen=(), canh_to=(), ax=None, tieu_de="", ve_canh=None, ghi_ten=True):
    """Vẽ mạng SAGSIN theo tầng (trục ngang: vĩ độ Nam → Bắc).
    tuyen: [(đường đi, màu, nhãn)], canh_to: [(tập cạnh (u, v), màu, nhãn, kiểu nét)] để tô cạnh đặc biệt."""
    ax = ax or plt.gca()
    x, y = toa_do(data)
    W = data.W[s]
    ve_canh = data.N <= 100 if ve_canh is None else ve_canh
    if ve_canh:
        iu, iv = np.nonzero(np.triu(W > 0, 1))
        for u, v in zip(iu, iv):
            ax.plot([x[u], x[v]], [y[u], y[v]], color="#d9d8d3", lw=0.4, zorder=1)
    for canh, mau, nhan, kieu in canh_to:
        for k, (u, v) in enumerate(canh):
            ax.plot([x[u], x[v]], [y[u], y[v]], color=mau, lw=1.2, ls=kieu, zorder=2, label=nhan if k == 0 else None)
    for loai, (mk, mau) in KY_HIEU.items():
        m = np.array([str(k) == loai for k in data.kind])
        ax.scatter(x[m], y[m], marker=mk, s=6 if loai == "leo" else 34, color=mau, zorder=3,
                   label={"gateway": "Gateway", "user": "Người dùng", "uav": "UAV", "haps": "HAPS", "leo": "Vệ tinh"}[loai])
    for p, mau, nhan in tuyen:
        if not p:
            continue
        for k, (u, v) in enumerate(zip(p[:-1], p[1:])):
            ax.add_patch(FancyArrowPatch((x[u], y[u]), (x[v], y[v]), arrowstyle="-|>", mutation_scale=11, lw=2.2,
                                         color=mau, zorder=4, label=nhan if k == 0 else None, shrinkA=4, shrinkB=4))
        if ghi_ten:
            for i in (p[0], p[-1]):
                ax.annotate(str(data.node_order[i]), (x[i], y[i]), xytext=(0, -13), textcoords="offset points",
                            ha="center", fontsize=8)
    ax.set_yticks(list(TEN_TANG))
    ax.set_yticklabels(list(TEN_TANG.values()))
    ax.set_xlabel("Vĩ độ (°B): Nam  →  Bắc")
    ax.set_ylim(-0.6, 3.6)
    ax.set_title(tieu_de, loc="left", fontsize=11)
    ax.legend(loc="upper left", fontsize=8, ncol=3, frameon=False, bbox_to_anchor=(0, -0.13))
    return ax


def tre(W, p):
    return float(sum(W[u, v] for u, v in zip(p[:-1], p[1:])))


# ------------------------------------------------------------------ mô hình
def tai_mo_hinh(ten):
    from test.danh_gia_quy_mo import load
    return load(GOC / ten)


def dinh_tuyen_dm(m, data, idx, n_steps=2):
    from test.danh_gia_quy_mo import chay_mo_hinh
    return chay_mo_hinh(m, data, np.asarray(idx), 1.0, n_steps, 200)


def diem_ung_vien(m, data, i, n_steps=10):
    """Điểm DM cho các ứng viên next-hop ở bước đầu của truy vấn i, sau từng bước khử nhiễu (để vẽ quá trình)."""
    import torch
    from src.ai.data_v2 import valid_mask
    from src.ai.train_gnn import encode_pair
    s, p = data.sid[i], data.paths[i]
    W = data.W[s:s + 1]
    dst = np.array([p[-1]])
    hc, hu = encode_pair(m, W, data.static_feat, dst)
    rows = W[:, p[0]]
    vis = np.zeros((1, data.N), bool)
    vis[0, p[0]] = True
    mask = valid_mask(rows, dst, data.can_relay, vis)
    cur, dt, R, M = (torch.from_numpy(a) for a in (np.array([p[0]]), dst, rows, mask))
    torch.manual_seed(0)
    y = torch.randn(1, data.N)
    steps = torch.linspace(m.T - 1, 0, n_steps).long()
    on = torch.ones(1, dtype=torch.bool)
    trang_thai = [y[0].numpy().copy()]
    with torch.no_grad():
        for k, t in enumerate(steps):
            tb = torch.full((1,), int(t), dtype=torch.long)
            ab = m.alpha_bar[int(t)]
            out = m.den(y, tb, hc, cur, dt, R, M, on, None)
            y0 = m._to_x0(out, y, ab, M)
            if k < n_steps - 1:
                ab_n = m.alpha_bar[int(steps[k + 1])]
                eps = (y - ab.sqrt() * y0) / (1 - ab).sqrt()
                y = ab_n.sqrt() * y0 + (1 - ab_n).sqrt() * eps
            else:
                y = y0
            trang_thai.append(y[0].numpy().copy())
    return np.flatnonzero(mask[0]), np.array(trang_thai), p


def ten_ngan(ten):
    ten = str(ten)
    return "SL-" + ten.split("-")[1].split(" ")[0] if ten.startswith("STARLINK") else ten


# ------------------------------------------------------------------ sơ đồ khối
def ve_hop(ax, x, y, chu, mau="#f3f2ee", vien="#52514e", w=2.6, h=0.9, co=9.5):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.05,rounding_size=0.12",
                                fc=mau, ec=vien, lw=1.1))
    ax.text(x, y, chu, ha="center", va="center", fontsize=co)


def ve_mui_ten(ax, a, b, chu="", lech=(0, 0.12)):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=13, lw=1.3, color="#52514e"))
    if chu:
        ax.text((a[0] + b[0]) / 2 + lech[0], (a[1] + b[1]) / 2 + lech[1], chu, ha="center", fontsize=9, color="#52514e")


def ve_chuoi(buoc, ax=None, mau=None, w=2.1):
    """Sơ đồ chuỗi các bước nằm ngang."""
    ax = ax or plt.gca()
    for k, chu in enumerate(buoc):
        ve_hop(ax, k * (w + 0.7), 0, chu, mau=(mau or {}).get(k, "#f3f2ee"), w=w, h=1.2, co=9)
        if k:
            ve_mui_ten(ax, ((k - 1) * (w + 0.7) + w / 2 + 0.05, 0), (k * (w + 0.7) - w / 2 - 0.05, 0))
    ax.set_xlim(-w / 2 - 0.2, (len(buoc) - 1) * (w + 0.7) + w / 2 + 0.2)
    ax.set_ylim(-0.9, 0.9)
    ax.axis("off")
    return ax
