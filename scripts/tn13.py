"""
tn13.py — Thí nghiệm 13 (results/bao_da_tuyen/TIEU_CHI_13.md): tỷ lệ 7/3 trong J có tốt nhất không, hay kết luận không phụ
thuộc tỷ lệ? Thước đo nằm NGOÀI J: đường đánh đổi vật lý (mất dịch vụ %, trễ ms) và điểm gối của nó.

    python -m scripts.tn13 toi-da        # chạy hết (cần dữ liệu + mô hình TN11)

Từng bước (chạy lại an toàn — phần đã xong tự bỏ qua):
    sinh-a        sinh lại val / val6 (4–5 luồng): thầy theo 11 mục tiêu (w ∈ {0,3 … 1,0}, J vật lý X ∈ {1, 3, 10, 30}) trên
                  cùng 32 thế giới nhãn + Dijkstra phạt trên lưới β, γ (16 bộ) → data/bao_da_tuyen/tn13/
    kiem-a        tình huống sinh lại có trùng bản gốc không (cảnh báo, không dừng)
    phan-tich-a   đường đánh đổi của thầy, điểm gối theo cửa sổ / gộp, luật rẽ nhánh → tn13/chon_b.json (2 mục tiêu)
    sinh-b        gán nhãn lại tập học theo 2 mục tiêu đã chọn; sinh lại 11 bộ kiểm tra (không thầy) có lưới Dijkstra phạt
    kiem-b        như kiem-a cho dữ liệu giai đoạn B
    huan-luyen-b  DM13 theo từng mục tiêu, 3 seed (cấu hình DM11, dừng sớm theo J của chính mục tiêu đó) → mo_hinh13/
    cham-b        11 bộ kiểm tra: DM theo 7/3 (DM11), DM13 × 2 mục tiêu; Dijkstra phạt mọi bộ β, γ → tn13/cham_tep/
    phan-tich     → tn13/KET_QUA_13.md, ket_qua_13.json, hinh_13_*.png; gói results/bao_da_tuyen/TN13_ket_qua.zip
Log: logs/tn13.log; mỗi mô hình logs/tn13_<tên>.log.
"""

from __future__ import annotations

import argparse
import glob
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

GOC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GOC))
from scripts import tn11 as t11  # noqa: E402
from scripts import tn12 as t12  # noqa: E402

KQ, DL, LOG = t11.KQ, t11.DL, t11.LOG
RA, M13, M11, DL13 = KQ / "tn13", KQ / "mo_hinh13", t11.M11, DL / "tn13"
BO_45, TEN_BO, DO, SEEDS = t11.BO_45, t11.TEN_BO, t11.DO, t12.SEEDS
W_DS = ("w0.3", "w0.5", "w0.6", "w0.7", "w0.8", "w0.9", "w1")
X_DS = ("x1", "x3", "x10", "x30")
MT = W_DS + X_DS
VAL = {"val": "val_k4_5", "val6": "val6_k4_5"}
HOC = {"train10": "train10_k4_5", "train6_10": "train6_10_k4_5", "x11": "x11_k4_5"}
GIUA = ("w0.6", "w0.7", "w0.8")                       # "quanh 7/3"
NGUONG_PHANG = (0.3, 1.0)                             # điểm % mất DV, ms trễ — đổi w 0,5 → 0,9 nhỏ hơn cả hai: phẳng
NGUONG_GOI = 0.05                                     # khoảng cách tới dây cung (đã chuẩn hoá) tối thiểu của điểm gối
SAN_KHOANG = (0.05, 0.05)                             # khoảng tối thiểu (ms, điểm %) để chuẩn hoá một đường
B_BOOT = 2000


def _w(o):
    return float(o[1:])


def py(*a):
    return t11.py(*a)


def sinh(tap, thu_muc_ra, a, *them):
    d = Path(thu_muc_ra) / (VAL.get(tap) or HOC.get(tap) or t11.thu_muc_bo(tap, "4-5"))
    if (d / ".xong").exists():
        return
    print(f"  sinh {d.relative_to(DL)} {' '.join(them)}", flush=True)
    t11.chay(py("-m", "src.nhieu_luong.tao_du_lieu_dt", "--tap", tap, "--K", "4-5", "--tien-trinh", str(a.tien_trinh),
                "--ra", str(thu_muc_ra), *them))


# ============================================================================================ giai đoạn A
def lenh_sinh_a(a):
    from src.nhieu_luong.tao_du_lieu_dt import MUC_TIEU_13
    assert tuple(MUC_TIEU_13) == MT, (MUC_TIEU_13, MT)
    for tap in VAL:
        sinh(tap, DL13, a, "--muc-tieu-them", ",".join(MT), "--luoi-dphat")


def _ly_do_lech(x, y, kiem_nhan):
    """Lý do bản ghi sinh lại khác bản gốc (None = khớp): mốc, mức tải, số luồng, cạnh của kho, tải luồng, nhãn."""
    if x["t"] != y["t"]:
        return "moc"
    if x["muc"] != y["muc"]:
        return "muc_tai"
    if x["gon"].K != y["gon"].K:
        return "so_luong"
    if not np.array_equal(x["gon"].U, y["gon"].U):
        return "canh_kho"
    if not np.allclose(np.asarray(x["gon"].lam, float), np.asarray(y["gon"].lam, float)):
        return "tai_luong"
    if kiem_nhan is not None and not kiem_nhan(x, y):
        return "nhan"
    return None


def _so_khop(goc, moi, kiem_nhan, dem_ly_do):
    """So từng bản ghi theo thứ tự; số bản ghi khác nhau thì phần dư tính là lệch."""
    n = 0
    for x, y in zip(goc, moi):
        ly_do = _ly_do_lech(x, y, kiem_nhan)
        if ly_do is None:
            n += 1
        else:
            dem_ly_do[ly_do] = dem_ly_do.get(ly_do, 0) + 1
    if len(goc) != len(moi):
        dem_ly_do["so_ban_ghi"] = dem_ly_do.get("so_ban_ghi", 0) + abs(len(goc) - len(moi))
    return len(goc) == len(moi), n


def kiem_tai_lap(cap, ten):
    """cap = [(thư mục gốc, thư mục sinh lại, hàm so nhãn | None)] → tỷ lệ bản ghi khớp; ghi tn13/kiem_tai_lap_<ten>.json."""
    ra, tong, khop = {}, 0, 0
    for goc, moi, kn in cap:
        tg = sorted(glob.glob(str(goc / "kb_*.pkl")))
        if not tg:
            ra[str(moi.name)] = "không có bản gốc để so"
            continue
        n_d, k_d, lech_so, ly_do = 0, 0, 0, {}
        for f in tg:
            g = Path(f)
            m = moi / g.name
            if not m.exists():
                continue
            x, y = pickle.load(open(g, "rb")), pickle.load(open(m, "rb"))
            ok, k = _so_khop(x, y, kn, ly_do)
            n_d, k_d, lech_so = n_d + len(x), k_d + k, lech_so + (not ok)
        ra[f"{moi.parent.name}/{moi.name}"] = {"so_ban_ghi": n_d, "khop": k_d, "tep_lech_so_ban_ghi": lech_so,
                                               "ly_do_lech": ly_do}
        if ly_do:
            print(f"    {moi.parent.name}/{moi.name}: khớp {k_d}/{n_d}; lệch do {ly_do}", flush=True)
        tong, khop = tong + n_d, khop + k_d
    ti_le = khop / tong if tong else float("nan")
    RA.mkdir(parents=True, exist_ok=True)
    (RA / f"kiem_tai_lap_{ten}.json").write_text(json.dumps({"ti_le_khop": ti_le, "chi_tiet": ra}, ensure_ascii=False,
                                                            indent=1), encoding="utf-8")
    print(f"  tái lập {ten}: {khop}/{tong} bản ghi khớp bản gốc ({ti_le:.1%})"
          + ("" if ti_le >= 0.99 else " — CẢNH BÁO: dưới 99%, xem kiem_tai_lap_" + ten + ".json"), flush=True)
    return ti_le


def lenh_kiem_a(a):
    nhan = lambda x, y: x.get("nhan") is not None and y.get("nhan") is not None and np.array_equal(x["nhan"], y["nhan"])
    kiem_tai_lap([(DL / d, DL13 / d, nhan) for d in VAL.values()], "a")


def _doc_a():
    """Bản ghi val giai đoạn A → (mảng theo cửa sổ). Trả dict: cum [n], mat/tre [n, |MT|] của thầy, lưới Dijkstra."""
    rs, cum = [], []
    for i, d in enumerate(VAL.values()):
        for f in sorted(glob.glob(str(DL13 / d / "kb_*.pkl"))):
            for r in pickle.load(open(f, "rb")):
                rs.append(r)
                cum.append(i * 1000 + int(r["kb"]))
    if not rs:
        raise SystemExit("chưa có dữ liệu giai đoạn A — chạy sinh-a")
    luoi = sorted(k for k in rs[0]["ens"] if k.startswith("dp_"))
    lay = lambda ten, m: np.array([r["ens"][ten][m] for r in rs], float)
    ten_mt = list(rs[0]["J_mt"]["_ten"])
    return {"cum": np.array(cum), "n": len(rs), "luoi": luoi, "ten_mt": ten_mt,
            "thay": {o: (lay(f"thay_{o}", "mat_dv_%"), lay(f"thay_{o}", "tre_chinh_ms")) for o in MT},
            "dp": {g: (lay(g, "mat_dv_%"), lay(g, "tre_chinh_ms")) for g in luoi},
            "J_dp": {g: np.array([r["J_mt"][g] for r in rs], float) for g in luoi},
            "goc_73": (lay("thay", "mat_dv_%"), lay("thay", "tre_chinh_ms"))}


def diem_goi(diem):
    """diem = [(w, trễ, mất DV)] theo w tăng. Trả (w của điểm gối | None, khoảng cách). Chuẩn hoá hai trục về [0, 1]; dây cung
    nối hai đầu (w nhỏ nhất, lớn nhất); điểm gối = điểm trong có khoảng cách lớn nhất về phía điểm lý tưởng (trễ thấp, mất
    thấp). Đường quá hẹp (dưới SAN_KHOANG) hoặc không điểm nào vượt NGUONG_GOI → None."""
    w = [p[0] for p in diem]
    x, y = np.array([p[1] for p in diem]), np.array([p[2] for p in diem])
    if np.ptp(x) < SAN_KHOANG[0] or np.ptp(y) < SAN_KHOANG[1]:
        return None, 0.0
    x, y = (x - x.min()) / np.ptp(x), (y - y.min()) / np.ptp(y)
    x0, y0, x1, y1 = x[0], y[0], x[-1], y[-1]
    L = np.hypot(x1 - x0, y1 - y0)
    if L < 1e-9:
        return None, 0.0
    z = ((x1 - x0) * (y - y0) - (y1 - y0) * (x - x0)) / L
    huong = -np.sign((x1 - x0) * (0 - y0) - (y1 - y0) * (0 - x0)) or 1.0     # dấu của phía điểm lý tưởng (0, 0)
    d = -huong * z
    d[0] = d[-1] = -np.inf
    j = int(np.argmax(d))
    return (w[j], float(d[j])) if d[j] > NGUONG_GOI else (None, float(d[j]))


def _trung_binh_cum(v, cum, chon=None):
    u = np.unique(cum) if chon is None else chon
    return np.array([v[cum == k].mean() for k in u])


def phan_tich_a(D):
    cum = D["cum"]
    u = np.unique(cum)
    mat = {o: _trung_binh_cum(D["thay"][o][0], cum) for o in MT}             # [số cửa sổ]
    tre = {o: _trung_binh_cum(D["thay"][o][1], cum) for o in MT}
    dem = np.array([(cum == k).sum() for k in u])
    gop = lambda v, idx=slice(None): float((v[idx] * dem[idx]).sum() / dem[idx].sum())
    duong = {o: {"mat_dv": gop(mat[o]), "tre": gop(tre[o])} for o in MT}
    goi_cs = [diem_goi([(_w(o), tre[o][i], mat[o][i]) for o in W_DS])[0] for i in range(len(u))]
    co_goi = [g for g in goi_cs if g is not None]
    ti_le_giua = (sum(g in {_w(o) for o in GIUA} for g in co_goi) / len(co_goi)) if co_goi else 0.0
    goi_gop, kc_gop = diem_goi([(_w(o), duong[o]["tre"], duong[o]["mat_dv"]) for o in W_DS])
    rng = np.random.default_rng(0)
    goi_bs = []
    for _ in range(B_BOOT):
        j = rng.integers(0, len(u), len(u))
        goi_bs.append(diem_goi([(_w(o), gop(tre[o], j), gop(mat[o], j)) for o in W_DS])[0])
    dm_mat = duong["w0.5"]["mat_dv"] - duong["w0.9"]["mat_dv"]
    dm_tre = duong["w0.9"]["tre"] - duong["w0.5"]["tre"]
    phang = abs(dm_mat) < NGUONG_PHANG[0] and abs(dm_tre) < NGUONG_PHANG[1]
    # Dijkstra phạt: bộ β, γ có J trung bình nhỏ nhất theo từng mục tiêu (trên val)
    i_mt = {o: D["ten_mt"].index(o) for o in MT}
    dp_chon = {o: min(D["luoi"], key=lambda g: (float(D["J_dp"][g][:, i_mt[o]].mean()), g)) for o in MT}
    dp_diem = {g: {"mat_dv": float(D["dp"][g][0].mean()), "tre": float(D["dp"][g][1].mean())} for g in D["luoi"]}
    # tỷ giá ngầm quanh 7/3: ms trễ đổi một điểm % mất DV giữa w0.6 và w0.8
    dmat = duong["w0.6"]["mat_dv"] - duong["w0.8"]["mat_dv"]
    ty_gia = (duong["w0.8"]["tre"] - duong["w0.6"]["tre"]) / dmat if dmat > 1e-9 else float("nan")
    # luật rẽ nhánh (TIEU_CHI_13.md)
    giua = {_w(o) for o in GIUA}
    if phang:
        nhanh, b = "khong_phu_thuoc", ["w0.5", "w1"]
    elif goi_gop in giua and ti_le_giua >= 2 / 3:
        nhanh, b = "73_tot_nhat", ["w0.5", "w0.9"]
    elif goi_gop in giua:
        nhanh, b = "gan_73_khong_on_dinh", ["w0.5", "w0.9"]
    elif goi_gop is None:
        nhanh, b = "khong_diem_goi", ["w0.5", "w1"]
    else:
        ws = f"w{goi_gop:g}"
        xs_ = np.array([duong[o]["tre"] for o in W_DS])
        ys_ = np.array([duong[o]["mat_dv"] for o in W_DS])
        sx, sy = max(np.ptp(xs_), 1e-9), max(np.ptp(ys_), 1e-9)
        xstar = min(X_DS, key=lambda o: np.hypot((duong[o]["tre"] - duong[ws]["tre"]) / sx,
                                                 (duong[o]["mat_dv"] - duong[ws]["mat_dv"]) / sy))
        nhanh, b = "khac", [ws, xstar]
    return {"so_cua_so": int(len(u)), "so_ban_ghi": int(D["n"]), "duong_thay": duong,
            "thay_goc_73": {"mat_dv": float(D["goc_73"][0].mean()), "tre": float(D["goc_73"][1].mean())},
            "goi_theo_cua_so": goi_cs, "so_cua_so_co_goi": len(co_goi), "ti_le_goi_quanh_73": ti_le_giua,
            "goi_gop": goi_gop, "khoang_cach_goi_gop": kc_gop,
            "goi_bootstrap": {str(k): sum(g == k for g in goi_bs) / B_BOOT for k in sorted({g for g in goi_bs},
                                                                                         key=lambda v: (v is None, v or 0))},
            "phang": {"mat_0.5_tru_0.9": dm_mat, "tre_0.9_tru_0.5": dm_tre, "phang": phang},
            "ty_gia_quanh_73_ms_moi_diem": ty_gia, "dp_chon": dp_chon, "dp_diem": dp_diem,
            "nhanh": nhanh, "muc_tieu_b": b}


def ve_a(A):
    plt = t11._plt()
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    for g, p in A["dp_diem"].items():
        ax.plot(p["tre"], p["mat_dv"], "s", color="#bbbbbb", ms=5)
    ax.plot([], [], "s", color="#bbbbbb", ms=5, label="Dijkstra phạt (16 bộ β, γ)")
    xs = [A["duong_thay"][o]["tre"] for o in W_DS]
    ys = [A["duong_thay"][o]["mat_dv"] for o in W_DS]
    ax.plot(xs, ys, "-o", color="#2a78d6", lw=1.8, ms=6, mec="white", mew=1.2, label="thầy theo w")
    for o, x_, y_ in zip(W_DS, xs, ys):
        ax.annotate(o.replace("w", "w = "), (x_, y_), textcoords="offset points", xytext=(5, 4), fontsize=7.5)
    for o in X_DS:
        p = A["duong_thay"][o]
        ax.plot(p["tre"], p["mat_dv"], "D", color="#eb6834", ms=6, mec="white", mew=1.2)
        ax.annotate(o.replace("x", "X = ") + " ms", (p["tre"], p["mat_dv"]), textcoords="offset points",
                    xytext=(5, -10), fontsize=7.5, color="#eb6834")
    if A["goi_gop"] is not None:
        p = A["duong_thay"][f"w{A['goi_gop']:g}"]
        ax.plot(p["tre"], p["mat_dv"], "o", ms=13, mfc="none", mec="#e34948", mew=1.6, label="điểm gối (gộp)")
    ax.set_xlabel("trễ tuyến chính (ms)")
    ax.set_ylabel("mất dịch vụ (%)")
    ax.set_title(f"TN13 giai đoạn A (val) — đường đánh đổi của thầy · nhánh: {A['nhanh']}", loc="left", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RA / "hinh_13_a.png", bbox_inches="tight")
    plt.close(fig)


def lenh_phan_tich_a(a):
    A = phan_tich_a(_doc_a())
    RA.mkdir(parents=True, exist_ok=True)
    (RA / "ket_qua_a.json").write_text(json.dumps(A, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    (RA / "chon_b.json").write_text(json.dumps({"nhanh": A["nhanh"], "muc_tieu_b": A["muc_tieu_b"],
                                                "dp_chon": A["dp_chon"]}, ensure_ascii=False, indent=1), encoding="utf-8")
    ve_a(A)
    print(f"  giai đoạn A: điểm gối gộp w = {A['goi_gop']}, {A['ti_le_goi_quanh_73']:.0%} cửa sổ có gối quanh 7/3, "
          f"phẳng = {A['phang']['phang']} → nhánh {A['nhanh']}, học lại {A['muc_tieu_b']}", flush=True)


# ============================================================================================ giai đoạn B
def _chon_b():
    t = RA / "chon_b.json"
    if not t.exists():
        raise SystemExit("chưa có tn13/chon_b.json — chạy phan-tich-a")
    return json.loads(t.read_text(encoding="utf-8"))


def lenh_sinh_b(a):
    cb = _chon_b()
    for o in cb["muc_tieu_b"]:
        for tap in HOC:
            sinh(tap, DL13 / o, a, "--muc-tieu", o, "--chi-nhan")
    for b in BO_45:
        sinh(b, DL13 / "kiem_tra", a, "--khong-thay", "--luoi-dphat")


def lenh_kiem_b(a):
    cb = _chon_b()
    nhan = lambda x, y: ("nhan_goc" not in y) or np.array_equal(x["nhan"], y["nhan_goc"])
    dp = lambda x, y: abs(x["ens"]["dphat"]["mat_dv_%"] - y["ens"]["dphat"]["mat_dv_%"]) < 1e-9
    cap = [(DL / d, DL13 / o / d, nhan) for o in cb["muc_tieu_b"] for d in HOC.values()]
    cap += [(DL / t11.thu_muc_bo(b, "4-5"), DL13 / "kiem_tra" / t11.thu_muc_bo(b, "4-5"), dp) for b in BO_45]
    kiem_tai_lap(cap, "b")


def lenh_huan_luyen_b(a):
    cb = _chon_b()
    viec = [(f"dm13_{o}_s{s}", t12.lenh_hoc("dm", 18, f"dm13_{o}_s{s}", s, a, ["--muc-tieu", o]))
            for s in SEEDS for o in cb["muc_tieu_b"]]
    t12.hang_doi(viec, a.song_song, M13, "tn13")


def _cham_tep(job):
    import torch

    from src.nhieu_luong import mo_hinh_dt as md
    from src.nhieu_luong.cham_lai_dt import giong_ban_ghi
    from src.nhieu_luong.cham_mo_hinh_dt import _tg
    tep_goc, tep_moi, mts = job
    torch.set_num_threads(1)
    goc = pickle.load(open(tep_goc, "rb"))
    moi = pickle.load(open(tep_moi, "rb")) if Path(tep_moi).exists() else [None] * len(goc)
    ra = []
    for r, m in zip(goc, moi):
        gon = r["gon"]
        tgw, sd, x = _tg(gon, r["cham"]), giong_ban_ghi(r), t11._co_ban(r)
        x["khop"] = bool(m is not None and m["t"] == r["t"] and m["muc"] == r["muc"])
        if x["khop"]:
            x["luoi"] = {g: [float(v["mat_dv_%"]), float(v["tre_chinh_ms"])] for g, v in m["ens"].items()
                         if g.startswith("dp_")}
        for o in mts:
            for s in SEEDS:
                net = t11._net(str(M11), f"dm11_s{s}") if o == "w0.7" else t11._net(str(M13), f"dm13_{o}_s{s}")
                a = md.chon_dm(net, gon, t12.CFG["S"], 5, torch.Generator().manual_seed(sd))
                x[f"dm_{o}_s{s}"] = t11._do(gon, tgw, a)
        ra.append(x)
    return tep_goc, ra


def lenh_cham_b(a):
    cb = _chon_b()
    mts = ["w0.7"] + [o for o in cb["muc_tieu_b"] if o != "w0.7"]
    thieu = [f"dm13_{o}_s{s}" for o in mts if o != "w0.7" for s in SEEDS if not (M13 / f"dm13_{o}_s{s}.xong").exists()]
    if thieu:
        raise SystemExit(f"chưa xong mô hình TN13: {thieu}")
    viec = []
    for b in BO_45:
        d = t11.thu_muc_bo(b, "4-5")
        viec += [(f"{d}__{Path(t).stem}.json", (t, str(DL13 / "kiem_tra" / d / Path(t).name), mts))
                 for t in sorted(glob.glob(str(DL / d / "kb_*.pkl")))]
    print(f"  chấm bộ kiểm tra: {len(viec)} tệp, mục tiêu {mts}", flush=True)
    t11._chay_tep(_cham_tep, viec, a.tien_trinh, RA / "cham_tep")


# ============================================================================================ phân tích cuối
def _bao_duoi(diem):
    """Đường bao dưới (Pareto: trễ thấp và mất thấp) của các điểm (trễ, mất) → hàm nội suy mất theo trễ."""
    p = sorted(diem)
    pareto = []
    for x, y in p:
        if not pareto or y < pareto[-1][1]:
            pareto.append((x, y))
    xs, ys = np.array([q[0] for q in pareto]), np.array([q[1] for q in pareto])
    return lambda t: float(np.interp(t, xs, ys))          # ngoài hai đầu: giữ giá trị đầu mút


def _khoang_cach_bao(rs, kb, o, luoi, rng):
    """Mất DV của DM_o trừ đường bao Dijkstra (16 bộ) tại cùng trễ; CI bootstrap theo cửa sổ (tính lại đường bao mỗi lần)."""
    u = np.unique(kb)
    dm_m = t11._ho(rs, f"dm_{o}", SEEDS, 0)
    dm_t = t11._ho(rs, f"dm_{o}", SEEDS, 4)
    G = {g: (np.array([r["luoi"][g][0] for r in rs]), np.array([r["luoi"][g][1] for r in rs])) for g in luoi}
    tong = lambda v: np.array([v[kb == k].sum() for k in u])
    dem = np.array([(kb == k).sum() for k in u])
    T = {"dm": (tong(dm_m), tong(dm_t)), **{g: (tong(G[g][0]), tong(G[g][1])) for g in luoi}}

    def kc(j):
        n = dem[j].sum()
        bao = _bao_duoi([(T[g][1][j].sum() / n, T[g][0][j].sum() / n) for g in luoi])
        return T["dm"][0][j].sum() / n - bao(T["dm"][1][j].sum() / n)
    tat = np.arange(len(u))
    bs = [kc(rng.integers(0, len(u), len(u))) for _ in range(B_BOOT)]
    return [kc(tat), [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]]


def phan_tich_b(T, cb, A):
    mts = ["w0.7"] + [o for o in cb["muc_tieu_b"] if o != "w0.7"]
    rng = np.random.default_rng(0)
    bo = {}
    for b, rs in T.items():
        rs = [r for r in rs if r.get("khop")]
        if not rs:
            continue
        kb = np.array([r["kb"] for r in rs])
        luoi = sorted(rs[0]["luoi"])
        x = {"n": len(rs)}
        for o in mts:
            g = cb["dp_chon"][o]
            dm_m, dm_t = t11._ho(rs, f"dm_{o}", SEEDS, 0), t11._ho(rs, f"dm_{o}", SEEDS, 4)
            dp_m, dp_t = np.array([r["luoi"][g][0] for r in rs]), np.array([r["luoi"][g][1] for r in rs])
            x[o] = {"dm": [float(dm_m.mean()), float(dm_t.mean())], "dp": [float(dp_m.mean()), float(dp_t.mean())],
                    "dp_bo": g, "hieu_mat": t11._bs(dm_m - dp_m, kb), "hieu_tre": t11._bs(dm_t - dp_t, kb),
                    "duoi_bao": _khoang_cach_bao(rs, kb, o, luoi, rng)}
        x["luoi"] = {g: [float(np.mean([r["luoi"][g][0] for r in rs])), float(np.mean([r["luoi"][g][1] for r in rs]))]
                     for g in luoi}
        bo[b] = x
    n = len(bo)
    can = max(1, int(np.ceil(8 / 11 * n)))
    h2 = {o: [b for b, x in bo.items() if x[o]["hieu_mat"][1][1] < 0] for o in mts}
    h3 = {o: [b for b, x in bo.items() if x[o]["duoi_bao"][1][1] < 0] for o in mts}
    # gộp mọi bộ: điểm DM theo mục tiêu
    gop = {}
    for o in mts:
        m = np.concatenate([t11._ho([r for r in rs if r.get("khop")], f"dm_{o}", SEEDS, 0) for rs in T.values()])
        t_ = np.concatenate([t11._ho([r for r in rs if r.get("khop")], f"dm_{o}", SEEDS, 4) for rs in T.values()])
        gop[o] = {"mat_dv": float(m.mean()), "tre": float(t_.mean())}
    cum = np.concatenate([BO_45.index(b) * 1000 + np.array([r["kb"] for r in rs if r.get("khop")]) for b, rs in T.items()])
    hieu73 = {}
    for o in mts[1:]:
        dm_o = np.concatenate([t11._ho([r for r in rs if r.get("khop")], f"dm_{o}", SEEDS, 0) for rs in T.values()])
        dm_7 = np.concatenate([t11._ho([r for r in rs if r.get("khop")], "dm_w0.7", SEEDS, 0) for rs in T.values()])
        tr_o = np.concatenate([t11._ho([r for r in rs if r.get("khop")], f"dm_{o}", SEEDS, 4) for rs in T.values()])
        tr_7 = np.concatenate([t11._ho([r for r in rs if r.get("khop")], "dm_w0.7", SEEDS, 4) for rs in T.values()])
        hieu73[o] = {"hieu_mat": t11._bs(dm_o - dm_7, cum), "hieu_tre": t11._bs(tr_o - tr_7, cum)}
    nhanh = cb["nhanh"]
    if nhanh == "khong_phu_thuoc" or nhanh == "khong_diem_goi":
        dat4 = all(-NGUONG_PHANG[0] < v["hieu_mat"][1][0] and v["hieu_mat"][1][1] < NGUONG_PHANG[0]
                   and abs(v["hieu_tre"][0]) < NGUONG_PHANG[1] for v in hieu73.values())
        h4 = {"kieu": "DM không phụ thuộc tỷ lệ", "hieu_so_voi_73": hieu73, "dat": dat4}
    elif nhanh in ("73_tot_nhat", "gan_73_khong_on_dinh"):
        g_, d_ = diem_goi([(_w(o), gop[o]["tre"], gop[o]["mat_dv"]) for o in sorted(mts, key=_w)])
        h4 = {"kieu": "7/3 là điểm gối của chính sách DM", "goi": g_, "khoang_cach": d_, "hieu_so_voi_73": hieu73,
              "dat": g_ == 0.7}
    else:
        h4 = {"kieu": "mô tả: DM theo w* và X* so với 7/3", "hieu_so_voi_73": hieu73, "dat": None}
    return {"muc_tieu": mts, "bo": bo, "gop": gop, "so_bo_can": can,
            "H13.2": {"bo_dat": h2, "dat": all(len(v) >= can for v in h2.values())},
            "H13.3": {"bo_dat": h3, "dat": all(len(v) >= can for v in h3.values())}, "H13.4": h4}


def ve_b(B):
    plt = t11._plt()
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    mau = {"w0.7": "#2a78d6"}
    khac = iter(("#eb6834", "#1baf7a", "#6250d6"))
    luoi = {}
    for x in B["bo"].values():
        for g, (m, t_) in x["luoi"].items():
            luoi.setdefault(g, []).append((t_, m))
    pts = [(np.mean([p[0] for p in v]), np.mean([p[1] for p in v])) for v in luoi.values()]
    ax.plot([p[0] for p in pts], [p[1] for p in pts], "s", color="#bbbbbb", ms=5, label="Dijkstra phạt (16 bộ β, γ)")
    bao = sorted(pts)
    par = []
    for p in bao:
        if not par or p[1] < par[-1][1]:
            par.append(p)
    ax.plot([p[0] for p in par], [p[1] for p in par], "-", color="#999999", lw=1.2)
    for o in B["muc_tieu"]:
        c = mau.get(o) or next(khac)
        p = B["gop"][o]
        ax.plot(p["tre"], p["mat_dv"], "o", color=c, ms=9, mec="white", mew=1.5, label=f"DM học theo {o}")
    ax.set_xlabel("trễ tuyến chính (ms), 11 bộ gộp")
    ax.set_ylabel("mất dịch vụ (%), 11 bộ gộp")
    ax.set_title("TN13 giai đoạn B (bộ kiểm tra) — DM theo từng mục tiêu so với đường bao Dijkstra phạt",
                 loc="left", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RA / "hinh_13_b.png", bbox_inches="tight")
    plt.close(fig)


def viet_md(A, B):
    ci = t11._ci
    L = ["# Thí nghiệm 13 — kết quả (sinh tự động bởi `scripts/tn13.py`)", "",
         "Tiêu chí: TIEU_CHI_13.md. Mất DV bằng %, trễ bằng ms; hiệu bằng điểm % / ms, CI 95% bootstrap theo cửa sổ.", "",
         f"## 1. Giai đoạn A — đường đánh đổi của thầy trên val ({A['so_cua_so']} cửa sổ, {A['so_ban_ghi']} tình huống)", "",
         "| Mục tiêu | Mất DV (%) | Trễ (ms) | Dijkstra phạt chọn theo mục tiêu này |", "|---|---:|---:|---|"]
    for o in MT:
        p = A["duong_thay"][o]
        L.append(f"| {o} | {p['mat_dv']:.2f} | {p['tre']:.2f} | {A['dp_chon'][o]} |")
    L += ["", f"- Thầy gốc (nhãn TN11, 7/3): mất DV {A['thay_goc_73']['mat_dv']:.2f}%, trễ {A['thay_goc_73']['tre']:.2f} ms.",
          f"- Phẳng? mất(0,5) − mất(0,9) = {A['phang']['mat_0.5_tru_0.9']:+.2f} điểm; trễ(0,9) − trễ(0,5) = "
          f"{A['phang']['tre_0.9_tru_0.5']:+.2f} ms → {'PHẲNG' if A['phang']['phang'] else 'không phẳng'}.",
          f"- Điểm gối gộp: w = {A['goi_gop']} (khoảng cách {A['khoang_cach_goi_gop']:.3f}); bootstrap theo cửa sổ: "
          f"{A['goi_bootstrap']}.",
          f"- Theo cửa sổ: {A['so_cua_so_co_goi']}/{A['so_cua_so']} cửa sổ có điểm gối; {A['ti_le_goi_quanh_73']:.0%} trong số đó "
          "ở w ∈ {0,6; 0,7; 0,8}.",
          f"- Tỷ giá ngầm quanh 7/3 (thầy, w 0,6 → 0,8): {A['ty_gia_quanh_73_ms_moi_diem']:.2f} ms cho 1 điểm % mất DV.",
          f"- **Nhánh: {A['nhanh']}** → học lại DM theo {A['muc_tieu_b']}. Hình: `hinh_13_a.png`.", ""]
    if B:
        L += ["## 2. Giai đoạn B — bộ kiểm tra, 4–5 luồng, trung bình 3 seed", ""]
        for o in B["muc_tieu"]:
            L += [f"### DM học theo {o}", "", "| Bộ | DM: mất / trễ | Dijkstra phạt (bộ β, γ chọn trên val): mất / trễ | "
                  "Hiệu mất DV | Hiệu trễ | DM − đường bao Dijkstra tại cùng trễ |", "|---|---|---|---|---|---|"]
            for b, x in B["bo"].items():
                y = x[o]
                L.append(f"| {TEN_BO[b]} | {y['dm'][0]:.2f} / {y['dm'][1]:.1f} | {y['dp'][0]:.2f} / {y['dp'][1]:.1f} "
                         f"({y['dp_bo']}) | {ci(y['hieu_mat'])} | {ci(y['hieu_tre'])} | {ci(y['duoi_bao'])} |")
            L.append("")
        L += [f"Gộp 11 bộ: " + "; ".join(f"DM {o}: mất {p['mat_dv']:.2f}%, trễ {p['tre']:.2f} ms" for o, p in B["gop"].items())
              + ". Hình: `hinh_13_b.png`.", "", "## 3. Giả thuyết", "",
              f"- **H13.2 (DM hơn Dijkstra phạt ở mọi mục tiêu): {'ĐẠT' if B['H13.2']['dat'] else 'KHÔNG ĐẠT'}** — "
              f"số bộ CI dưới 0 cần ≥ {B['so_bo_can']}: " + "; ".join(f"{o}: {len(v)}" for o, v in B["H13.2"]["bo_dat"].items()),
              f"- **H13.3 (DM nằm dưới đường bao Dijkstra): {'ĐẠT' if B['H13.3']['dat'] else 'KHÔNG ĐẠT'}** — "
              + "; ".join(f"{o}: {len(v)}" for o, v in B["H13.3"]["bo_dat"].items()),
              f"- **H13.4 ({B['H13.4']['kieu']}): " + ("MÔ TẢ" if B["H13.4"]["dat"] is None else
                                                      "ĐẠT" if B["H13.4"]["dat"] else "KHÔNG ĐẠT") + "** — "
              + "; ".join(f"{o} − 7/3: mất {ci(v['hieu_mat'])}, trễ {ci(v['hieu_tre'])}"
                          for o, v in B["H13.4"]["hieu_so_voi_73"].items())
              + (f"; điểm gối của DM: w = {B['H13.4'].get('goi')}" if "goi" in B["H13.4"] else "")]
    (RA / "KET_QUA_13.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def lenh_phan_tich(a):
    A = json.loads((RA / "ket_qua_a.json").read_text(encoding="utf-8")) if (RA / "ket_qua_a.json").exists() else None
    if A is None:
        raise SystemExit("chưa có ket_qua_a.json — chạy phan-tich-a")
    T = {b: rs for b in BO_45 if (rs := t11._gop(RA / "cham_tep", t11.thu_muc_bo(b, "4-5")))}
    B = phan_tich_b(T, _chon_b(), A) if T else None
    (RA / "ket_qua_13.json").write_text(json.dumps({"A": A, "B": B}, ensure_ascii=False, indent=1, default=float),
                                        encoding="utf-8")
    if B:
        ve_b(B)
        print("  " + "; ".join(f"{k}: {'MÔ TẢ' if B[k]['dat'] is None else 'ĐẠT' if B[k]['dat'] else 'KHÔNG ĐẠT'}"
                               for k in ("H13.2", "H13.3", "H13.4")), flush=True)
    viet_md(A, B)
    t12.dong_goi("TN13", RA, M13, "tn13")


# ============================================================================================ chạy
def lenh_toi_da(a):
    p = t11.tai_nguyen(a.ram, a.vram)
    print(f"máy: {p['cpu']} luồng CPU, ngân sách RAM {p['ram']} GB, GPU: {p['gpu']}", flush=True)
    print(f"→ sinh {p['sinh']} tiến trình; học {p['hoc']} mô hình cùng lúc × {p['luong']} luồng ({p['thiet_bi']}); "
          f"chấm {p['cham']} tiến trình", flush=True)
    a.song_song, a.luong_cpu, a.thiet_bi = p["hoc"], p["luong"], p["thiet_bi"]
    buoc = [("sinh-a", lenh_sinh_a, "sinh"), ("kiem-a", lenh_kiem_a, None), ("phan-tich-a", lenh_phan_tich_a, None),
            ("sinh-b", lenh_sinh_b, "sinh"), ("kiem-b", lenh_kiem_b, None), ("huan-luyen-b", lenh_huan_luyen_b, None),
            ("cham-b", lenh_cham_b, "cham"), ("phan-tich", lenh_phan_tich, None)]
    for ten, f, tt in buoc:
        if tt:
            a.tien_trinh = p[tt]
        print(f"=== {ten} {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
        f(a)


VIEC = {"sinh-a": lenh_sinh_a, "kiem-a": lenh_kiem_a, "phan-tich-a": lenh_phan_tich_a, "sinh-b": lenh_sinh_b,
        "kiem-b": lenh_kiem_b, "huan-luyen-b": lenh_huan_luyen_b, "cham-b": lenh_cham_b, "phan-tich": lenh_phan_tich,
        "toi-da": lenh_toi_da}


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("viec", choices=list(VIEC))
    ap.add_argument("--tien-trinh", type=int, default=4)
    ap.add_argument("--song-song", type=int, default=2)
    ap.add_argument("--luong-cpu", type=int, default=4)
    ap.add_argument("--thiet-bi", default="auto", choices=["cpu", "auto", "cuda"])
    ap.add_argument("--ram", type=float, default=20.0, help="ngân sách RAM (GB) cho toi-da")
    ap.add_argument("--vram", type=float, default=8.0, help="ngân sách VRAM (GB) cho toi-da")
    ap.add_argument("--tn11", default=None, help="thư mục repo đã chạy TN11: chép mô hình + dữ liệu TN11 còn thiếu")
    a = ap.parse_args()
    t12.mo_log("tn13", sys.argv[1:])
    if a.tn11:
        t12.lay_tn11(a.tn11)
    VIEC[a.viec](a)
    print("xong", flush=True)


if __name__ == "__main__":
    main()
