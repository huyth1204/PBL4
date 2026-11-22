"""
tn15b.py — Thí nghiệm 15b (results/bao_da_tuyen/TIEU_CHI_15B.md, bản 2): kiểm chứng lại các kết luận của TN15 trên bộ dữ liệu
MỚI (thời tiết 2017–2021 chưa dùng ở TN nào), luật phân tích chốt trước, CÓ học lại: thêm seed DM x10 (s3, s4), DM 7/3 học lại
cùng đường ống với x10 (dm13_w0.7 s0–4), quét X = 3 (dm13_x3 s0–2).

    python -m scripts.tn15b toi-da                  # chạy hết trên máy GPU của nhóm (~6–7 giờ)
    python -m scripts.tn15b toi-da --thu <thư mục>  # chạy thử trên dữ liệu CŨ (nam9, ít mốc), KHÔNG gán nhãn / học

Từng bước (chạy lại an toàn — phần đã xong tự bỏ qua):
    dong-bang   ghi mã băm của tiêu chí + code → tn15b/dong_bang.json
    gan-nhan    gán nhãn lại tập học theo w0.7 và x3 → data/bao_da_tuyen/tn13/<mục tiêu>/ (như TN15 với x10)
    kiem-nhan   tình huống gán nhãn lại có trùng tập gốc → tn15b/kiem_tai_lap.json
    hoc         dm13_x10_s3..4, dm13_w0.7_s0..4, dm13_x3_s0..2 → results/bao_da_tuyen/mo_hinh13/
    sinh        bộ kiểm chứng mới: A (4–5 luồng, kho cũ, thầy 7/3 + x10, lưới Dijkstra), B (kho nhanh), K10 (10 luồng)
    nhom        dựng nhóm rủi ro cho B (như TN14) — cần cho bước vá hết tuyến
    cham        chấm A / B / K10 từng tệp → tn15b/cham_tep/
    phan-tich   → tn15b/KET_QUA_15B.md, ket_qua_15b.json, hinh_15b.png; gói results/bao_da_tuyen/TN15B_ket_qua.zip
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import pickle
import shutil
import subprocess
import sys
import time
import zipfile
from math import erf, sqrt
from pathlib import Path

import numpy as np

GOC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GOC))
from scripts import tn11 as t11  # noqa: E402

KQ, DL, LOG = GOC / "results" / "bao_da_tuyen", GOC / "data" / "bao_da_tuyen", GOC / "logs"
RA, D15, DL13, M13 = KQ / "tn15b", DL / "tn15b", DL / "tn13", KQ / "mo_hinh13"
# thư mục có thể chứa trọng số, theo thứ tự thử (máy người dùng có bản giải nén *_may_nhom; máy nhóm có mo_hinh13)
CHO_MH = {"dm": [KQ / "mo_hinh11"], "x10": [KQ / "tn15_may_nhom" / "mo_hinh13", M13], "w07": [M13], "x3": [M13],
          "w05": [KQ / "tn13_may_nhom" / "mo_hinh13", M13], "w09": [KQ / "tn13_may_nhom" / "mo_hinh13", M13]}
TEN_MH = {"dm": "dm11_s{}", "x10": "dm13_x10_s{}", "w07": "dm13_w0.7_s{}", "x3": "dm13_x3_s{}",
          "w05": "dm13_w0.5_s{}", "w09": "dm13_w0.9_s{}"}
SEED_MH = {"dm": (0, 1, 2), "x10": (0, 1, 2, 3, 4), "w07": (0, 1, 2, 3, 4), "x3": (0, 1, 2), "w05": (0, 1, 2), "w09": (0, 1, 2)}
HOC_MOI = [("x10", 3), ("x10", 4)] + [("w0.7", s) for s in range(5)] + [("x3", s) for s in range(3)]
NHAN_MOI = ("w0.7", "x3")
HOC = {"train10": "train10_k4_5", "train6_10": "train6_10_k4_5", "x11": "x11_k4_5"}     # như tn13.HOC
S, NB = 32, 5
MAT, A_, B_, TRE = 0, 2, 3, 4                       # chỉ số trong t11.DO
TAP_A, TAP_K10 = ("bao15b", "ngay15b"), ("k15b",)
BUOC_MOC, THU = None, False
NGUONG, KHONG_KEM, NB_BOOT, AA = 100.0, 0.10, 4000, 7919
TIEU_CHI = KQ / "TIEU_CHI_15B.md"
DONG_BANG = [TIEU_CHI, GOC / "scripts" / "tn15b.py", GOC / "src" / "nhieu_luong" / "tao_du_lieu_dt.py"]
Phi = lambda z: 0.5 * (1 + erf(z / sqrt(2)))


def dat_thu(thu):
    """--thu: chạy thử trên dữ liệu cũ (nam9), ít mốc, ghi vào thư mục riêng; không gán nhãn / học (mô hình mới thay tạm
    bằng mô hình có sẵn, chỉ để thử code)."""
    global RA, D15, TAP_A, TAP_K10, BUOC_MOC, THU
    RA, D15 = Path(thu) / "tn15b", Path(thu) / "data"
    TAP_A, TAP_K10, BUOC_MOC, THU = ("nam9",), ("nam9",), 16, True


def ghi(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.mkdir(parents=True, exist_ok=True)
    with open(LOG / ("tn15b_thu.log" if THU else "tn15b.log"), "a", encoding="utf-8") as f:
        f.write(s + "\n")


def bam(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _tep_mh(k, s):
    ten = TEN_MH[k].format(s) + ".pt"
    return next((d / ten for d in CHO_MH[k] if (d / ten).exists()), None)


def _net(k, s):
    """Mạng của mô hình k, seed s. Chạy thử: mô hình mới chưa học → thay tạm bằng mô hình có sẵn (chỉ để thử code)."""
    t = _tep_mh(k, s)
    if t is None and THU:
        k, s = {"w07": "dm", "x3": "x10"}.get(k, k), s % 3
        t = _tep_mh(k, s)
    if t is None:
        raise SystemExit(f"thiếu trọng số {TEN_MH[k].format(s)}.pt")
    return t11._net(str(t.parent), t.stem)


# ============================================================================================ đóng băng
def lenh_dong_bang(a):
    RA.mkdir(parents=True, exist_ok=True)
    t = RA / "dong_bang.json"
    if t.exists():
        ghi("  đã đóng băng trước:", json.loads(t.read_text(encoding="utf-8"))["luc"])
        return
    t.write_text(json.dumps({"luc": time.strftime("%Y-%m-%d %H:%M:%S"),
                             "bam": {str(p.relative_to(GOC)): bam(p) for p in DONG_BANG}}, indent=1), encoding="utf-8")
    ghi("  đóng băng:", t)


# ============================================================================================ chạy lệnh con
def _chay(cmd):
    ghi("  $", " ".join(str(c) for c in cmd[3:]))
    p = subprocess.Popen(cmd, cwd=GOC, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                         errors="replace", env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    for dong in p.stdout:
        ghi("    " + dong.rstrip())
    if p.wait():
        raise SystemExit(f"lỗi khi chạy: {cmd}")


def _tdt(tap, K, goc, co, tien_trinh, buoc=None):
    th = Path(goc) / (f"{tap}_k{K.replace('-', '_')}" if "-" in K else f"{tap}_k{K}")
    if (th / ".xong").exists():
        ghi(f"  {th}: đã có")
        return
    cmd = t11.py("-m", "src.nhieu_luong.tao_du_lieu_dt", "--tap", tap, "--K", K, "--tien-trinh", str(tien_trinh),
                 "--ra", str(goc), *co)
    if buoc:
        cmd += ["--buoc-moc", str(buoc)]
    _chay(cmd)


# ============================================================================================ gán nhãn + học
def lenh_gan_nhan(a):
    for mt in NHAN_MOI:
        for tap in HOC:
            if THU:
                ghi(f"  (chạy thử, bỏ qua) gán nhãn {tap} theo {mt} → {DL13 / mt}")
                continue
            _tdt(tap, "4-5", DL13 / mt, ["--muc-tieu", mt, "--chi-nhan"], a.tien_trinh_sinh)


def lenh_kiem_nhan(a):
    if THU:
        ghi("  (chạy thử, bỏ qua) kiểm tái lập nhãn")
        return
    from scripts import tn13 as t13
    nhan = lambda x, y: ("nhan_goc" not in y) or np.array_equal(x["nhan"], y["nhan_goc"])
    cap = [(DL / d, DL13 / mt / d, nhan) for mt in NHAN_MOI for d in HOC.values()]
    t13.kiem_tai_lap(cap, "tn15b")
    RA.mkdir(parents=True, exist_ok=True)
    shutil.copy(t13.RA / "kiem_tai_lap_tn15b.json", RA / "kiem_tai_lap.json")


def lenh_hoc(a):
    from scripts import tn12 as t12
    thieu = [str(DL13 / mt / d) for mt in NHAN_MOI for d in HOC.values() if not (DL13 / mt / d / ".xong").exists()]
    if THU:
        for mt, s in HOC_MOI:
            ghi("  (chạy thử, bỏ qua) học", " ".join(t12.lenh_hoc("dm", 18, f"dm13_{mt}_s{s}", s, a, ["--muc-tieu", mt])[3:]))
        return
    if thieu or not (DL13 / "x10" / HOC["train10"] / ".xong").exists():
        raise SystemExit(f"chưa đủ nhãn: {thieu or 'tn13/x10 (của TN15)'} — chạy bước gan-nhan / kiểm dữ liệu TN15")
    viec = [(f"dm13_{mt}_s{s}", t12.lenh_hoc("dm", 18, f"dm13_{mt}_s{s}", s, a, ["--muc-tieu", mt])) for mt, s in HOC_MOI]
    t12.hang_doi(viec, a.song_song, M13, "tn15b")


# ============================================================================================ sinh bộ kiểm chứng mới
def lenh_sinh(a):
    for tap in TAP_A:
        _tdt(tap, "4-5", D15 / "A", ["--muc-tieu", "x10", "--luoi-dphat"], a.tien_trinh_sinh, BUOC_MOC)
    for tap in TAP_A:
        _tdt(tap, "4-5", D15 / "B", ["--khong-thay", "--kho-nhanh"], a.tien_trinh_sinh, BUOC_MOC)
    for tap in TAP_K10:
        _tdt(tap, "10", D15 / "K10", ["--khong-thay"], a.tien_trinh_sinh, BUOC_MOC)


# ============================================================================================ nhóm rủi ro (B)
def _nhom_tep(job):
    from scripts import tn14 as t14
    t14.NHOM = Path(job[2])
    return t14._nhom_tep(job[:2])


def lenh_nhom(a):
    from multiprocessing import Pool
    viec = [(f"{tap}_k4_5", t, str(D15 / "nhom")) for tap in TAP_A
            for t in sorted(glob.glob(str(D15 / "B" / f"{tap}_k4_5" / "kb_*.pkl")))]
    con = [v for v in viec if not (D15 / "nhom" / v[0] / Path(v[1]).name).exists()]
    ghi(f"  nhóm rủi ro B: {len(viec)} tệp, còn {len(con)}")
    if con:
        s = time.time()
        with Pool(a.tien_trinh_sinh) as pool:
            for k, _ in enumerate(pool.imap_unordered(_nhom_tep, con)):
                if k % 10 == 0 or k == len(con) - 1:
                    ghi(f"    {k + 1}/{len(con)} tệp ({time.time() - s:.0f} s)")
    ok = [0, 0]
    for d, t, _ in viec:
        x = pickle.load(open(D15 / "nhom" / d / Path(t).name, "rb"))
        ok[0], ok[1] = ok[0] + sum(z["ok"] for z in x), ok[1] + len(x)
    RA.mkdir(parents=True, exist_ok=True)
    (RA / "nhom_khop.json").write_text(json.dumps(ok), encoding="utf-8")
    ghi(f"  nhóm khớp {ok[0]}/{ok[1]}")


# ============================================================================================ chấm
def tre_thuc(gon, tgw, a):
    """Trễ THỰC (thăm dò): tuyến ĐANG DÙNG trong từng thế giới chấm (tuyến sống đầu tiên của tập → có chuyển sang dự
    phòng), dung lượng còn trống tệ nhất trong cửa sổ du của thế giới đó, tải nền như lúc quyết định (C' = du / (1 − ρ0));
    chỉ tính các (luồng, thế giới) còn phục vụ (không (a), không (b)). Trả [trễ trung bình λ-trọng, % > 100 ms]."""
    from src.nhieu_luong import da_tuyen as dt
    from src.nhieu_luong.bai_toan import h_hang_doi
    sol = gon.tu_lua_chon(a)
    cd = [tgw._dang_dung(k, sol[k]) for k in range(gon.K)]
    F = sum(gon.lam[k] * cd[k][1] for k in range(gon.K))
    ovl = tgw._qua_tai(F)
    du = tgw.Cw[:, 0]
    rho0 = np.minimum(gon.rho0[None], 0.999)
    Cp = np.maximum(du, 1e-9) / (1 - rho0)
    lat = gon.prop[None] + dt.PKT / Cp * (1 + h_hang_doi(rho0 + F / Cp)) + dt.PROC
    tong = dem = qua = 0.0
    for k in range(gon.K):
        co, inc = cd[k]
        phuc_vu = co & ~(ovl & inc).any(1)
        d = (lat * inc).sum(1)
        tong += gon.lam[k] * d[phuc_vu].sum()
        dem += gon.lam[k] * phuc_vu.sum()
        qua += gon.lam[k] * (phuc_vu & (d > NGUONG)).sum()
    return [float(tong / dem) if dem else float("nan"), float(qua / dem * 100) if dem else float("nan")]


def _x_eff_va_bang(gon, r):
    """(X_eff λ-trọng trung bình hình học, bảng rủi ro 32 thế giới để chọn seed) — chỉ thông tin lúc quyết định."""
    from src.nhieu_luong import da_tuyen as dt
    B = dt.TheGioi.tu_bang(gon, np.random.default_rng([int(r["kb"]), int(r["t"]), 99]))
    HR = np.maximum(B.Hmax - B.Hmin, dt.SAN_H)
    X = (0.7 / 0.3) * gon.drange / (100.0 * HR)
    lam = np.asarray(gon.lam, float)
    return float(np.exp((np.log(np.maximum(X, 1e-3)) * lam).sum() / lam.sum())), B


def _co_so(r):
    return {"kb": int(r["kb"]), "t": int(r["t"]), "muc": float(r["muc"]), "bao": r["bao"]}


def _cham_A(job):
    import torch

    from src.nhieu_luong import mo_hinh_dt as md
    from src.nhieu_luong.cham_lai_dt import giong_ban_ghi
    from src.nhieu_luong.cham_mo_hinh_dt import _tg
    tep = job[0]
    torch.set_num_threads(1)
    G = lambda sd: torch.Generator().manual_seed(sd)
    ra = []
    for r in pickle.load(open(tep, "rb")):
        gon = r["gon"]
        tgw, sd = _tg(gon, r["cham"]), giong_ban_ghi(r)
        do = lambda a: t11._do(gon, tgw, a)
        x = {**_co_so(r), "dp": [float(r["ens"]["dp_50_50"][m]) for m in t11.DO],
             "luoi": {g: [float(v[m]) for m in t11.DO] for g, v in r["ens"].items() if g.startswith("dp_")},
             "thay_x10": do(r["nhan"]), "thay_73": do(r["nhan_goc"]),
             "thay_x10_ens": [float(r["ens"]["thay"][m]) for m in t11.DO]}
        x["X_eff"], Bng = _x_eff_va_bang(gon, r)
        Bng.muc_tieu = ("x", 10.0)
        a_ktn = []
        for k in ("dm", "x10", "w07", "x3", "w05", "w09"):
            for s in SEED_MH[k]:
                net = _net(k, s)
                a = md.chon_dm(net, gon, S, NB, G(sd))
                x[f"{k}_s{s}"] = do(a)
                if k in ("dm", "x10", "w07"):
                    x[f"thuc_{k}_s{s}"] = tre_thuc(gon, tgw, a)
                    ak = md.chon_sua_dan(net, gon, S, NB, G(sd))
                    x[f"{k}_ktn_s{s}"] = do(ak)
                    if k == "x10":
                        a_ktn.append(ak)
                if k == "x10":
                    x[f"x10_nhanh_s{s}"] = do(md.chon_sua_dan(net, gon, 8, NB, G(sd)))
                    x[f"x10_aa_s{s}"] = do(md.chon_dm(net, gon, S, NB, G(sd + AA)))
        # chọn seed lúc quyết định: lời giải x10 ktn có J x10 nhỏ nhất trên bảng rủi ro (chỉ thông tin lúc quyết định)
        x["chon_seed"] = int(np.argmin([float(Bng.cham(gon.tu_lua_chon(a))["J"]) for a in a_ktn]))
        ra.append(x)
    return tep, ra


def _cham_B(job):
    import torch

    from src.nhieu_luong import da_tuyen as dt
    from src.nhieu_luong import hau_xu_ly as hx
    from src.nhieu_luong import mo_hinh_dt as md
    from src.nhieu_luong.cham_lai_dt import giong_ban_ghi
    from src.nhieu_luong.cham_mo_hinh_dt import _tg
    tep, tep_nhom = job
    torch.set_num_threads(1)
    G = lambda sd: torch.Generator().manual_seed(sd)
    ra = []
    for r, z in zip(pickle.load(open(tep, "rb")), pickle.load(open(tep_nhom, "rb"))):
        if int(r["t"]) != z["t"] or r["muc"] != z["muc"]:
            raise SystemExit(f"nhóm lệch bản ghi: {tep}")
        gon = r["gon"]
        tgw, sd = _tg(gon, r["cham"]), giong_ban_ghi(r)
        do = lambda a: t11._do(gon, tgw, a)
        x = {**_co_so(r), "ok_nhom": bool(z["ok"])}
        for k in ("dm", "w07"):
            for s in SEED_MH[k]:
                x[f"{k}_s{s}"] = do(md.chon_dm(_net(k, s), gon, S, NB, G(sd)))
        for s in SEED_MH["x10"]:
            a = md.chon_sua_dan(_net("x10", s), gon, S, NB, G(sd))
            x[f"x10_ktn_s{s}"] = do(a)
            Bv = dt.TheGioi.tu_bang_nhom(gon, z.get("nhom") if z["ok"] else None, np.random.default_rng([sd, 14]), 1.0, 64)
            x[f"de_xuat_s{s}"] = do(hx.va_het_tuyen(Bv, a, so_vong=2))
        ra.append(x)
    return tep, ra


def _cham_K10(job):
    import torch

    from src.nhieu_luong import mo_hinh_dt as md
    from src.nhieu_luong.cham_lai_dt import giong_ban_ghi
    from src.nhieu_luong.cham_mo_hinh_dt import _tg
    tep = job[0]
    torch.set_num_threads(1)
    G = lambda sd: torch.Generator().manual_seed(sd)
    ra = []
    for r in pickle.load(open(tep, "rb")):
        gon = r["gon"]
        tgw, sd = _tg(gon, r["cham"]), giong_ban_ghi(r)
        x = {**_co_so(r), "dp": [float(r["ens"]["dphat"][m]) for m in t11.DO]}
        for k in ("dm", "x10", "w07"):
            for s in SEED_MH[k]:
                net = _net(k, s)
                x[f"{k}_s{s}"] = t11._do(gon, tgw, md.chon_dm(net, gon, S, NB, G(sd)))
                if k == "x10":
                    x[f"x10_ktn_s{s}"] = t11._do(gon, tgw, md.chon_sua_dan(net, gon, S, NB, G(sd)))
        ra.append(x)
    return tep, ra


def lenh_cham(a):
    thieu = [TEN_MH[k].format(s) for k in TEN_MH for s in SEED_MH[k] if _tep_mh(k, s) is None]
    if thieu and not THU:
        raise SystemExit(f"thiếu trọng số: {thieu} — chạy bước hoc")
    for nhom, ham, goc, K in (("A", _cham_A, D15 / "A", "4_5"), ("B", _cham_B, D15 / "B", "4_5"),
                              ("K10", _cham_K10, D15 / "K10", "10")):
        taps = TAP_K10 if nhom == "K10" else TAP_A
        viec = []
        for tap in taps:
            d = f"{tap}_k{K}"
            for t in sorted(glob.glob(str(goc / d / "kb_*.pkl"))):
                job = (t, str(D15 / "nhom" / d / Path(t).name)) if nhom == "B" else (t,)
                viec.append((f"{nhom}__{d}__{Path(t).stem}.json", job))
        ghi(f"  chấm {nhom}: {len(viec)} tệp")
        t11._chay_tep(ham, viec, a.tien_trinh, RA / "cham_tep")


# ============================================================================================ phân tích
def _nap(nhom):
    T = []
    for f in sorted(glob.glob(str(RA / "cham_tep" / f"{nhom}__*.json"))):
        tap = Path(f).name.split("__")[1]
        for x in json.loads(Path(f).read_text(encoding="utf-8")):
            x["_cum"] = f"{tap}:{x['kb']}"
            T.append(x)
    return T


class Bo:
    """Mảng theo phương pháp × seed của một nhóm dữ liệu + bootstrap hai tầng theo cửa sổ và seed."""

    def __init__(self, T):
        self.T = T
        self.cum = np.array([x["_cum"] for x in T])
        self.u, self.inv = np.unique(self.cum, return_inverse=True)
        self.dem = np.bincount(self.inv)

    def m(self, k, j=None, ham=None):
        """[số seed, N] (phương pháp không seed: [1, N]) của cột j / hàm ham."""
        ten = [f"{k}_s{i}" for i in range(10) if f"{k}_s{i}" in self.T[0]] or [k]
        X = np.array([[x[t] for x in self.T] for t in ten], float)          # [n_seed, N, 5]
        return ham(X) if ham else X[..., j]

    def boot(self, v1, v0, nguong=0.0, cap=False, seed=1):
        """v1, v0: [n_seed, N]. Hiệu trung bình (v1 − v0) với bootstrap cửa sổ + seed (cap=True: cùng cặp seed)."""
        rg = np.random.default_rng(seed)
        s1 = np.array([np.bincount(self.inv, r, minlength=len(self.u)) for r in v1])
        s0 = np.array([np.bincount(self.inv, r, minlength=len(self.u)) for r in v0])
        diem = v1.mean() - v0.mean()
        bs = np.empty(NB_BOOT)
        for b in range(NB_BOOT):
            w = np.bincount(rg.integers(0, len(self.u), len(self.u)), minlength=len(self.u))
            n = (w * self.dem).sum()
            i1 = rg.integers(0, len(v1), len(v1))
            i0 = i1 if cap else rg.integers(0, len(v0), len(v0))
            bs[b] = (s1[i1] @ w).mean() / n - (s0[i0] @ w).mean() / n
        se = bs.std(ddof=1)
        return {"diem": float(diem), "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "se": float(se),
                "p_nho_hon": float(Phi((diem - nguong) / se)) if se > 0 else float(not diem < nguong),
                "p_lon_hon": float(1 - Phi((diem - nguong) / se)) if se > 0 else float(not diem > nguong)}


def _duoi(X, ng=NGUONG):
    return (X[..., TRE] > ng).astype(float) * 100


def hoi_quy(bo, T, k1="x10", k0="w07", seed=2):
    """P5: hệ số log X_eff trong hồi quy hiệu mất (k1 − k0, trung bình seed) ~ 1 + log X_eff + log trễ Dijkstra; bootstrap
    cửa sổ + seed."""
    v1, v0 = bo.m(k1, MAT), bo.m(k0, MAT)
    lx = np.log(np.maximum([x["X_eff"] for x in T], 1e-3))
    lt = np.log(np.maximum([x["dp"][TRE] for x in T], 1e-3))
    Xd = np.column_stack([np.ones(len(lx)), lx, lt])
    hs = lambda y, idx: np.linalg.lstsq(Xd[idx], y[idx], rcond=None)[0]
    tat = np.arange(len(lx))
    diem = hs(v1.mean(0) - v0.mean(0), tat)
    rg = np.random.default_rng(seed)
    nhom = [np.flatnonzero(bo.inv == g) for g in range(len(bo.u))]
    bs = []
    for _ in range(1000):
        idx = np.concatenate([nhom[g] for g in rg.integers(0, len(bo.u), len(bo.u))])
        y = v1[rg.integers(0, len(v1), len(v1))].mean(0) - v0[rg.integers(0, len(v0), len(v0))].mean(0)
        bs.append(hs(y, idx))
    bs = np.array(bs)
    se = bs[:, 1].std(ddof=1)
    yt = bo.m(k1, TRE).mean(0) - bo.m(k0, TRE).mean(0)
    return {"he_so_log_X": float(diem[1]), "ci": [float(np.percentile(bs[:, 1], 2.5)), float(np.percentile(bs[:, 1], 97.5))],
            "se": float(se), "p_lon_hon": float(1 - Phi(diem[1] / se)) if se > 0 else 1.0,
            "he_so_log_tre_dp": float(diem[2]), "he_so_log_X_hieu_tre": float(hs(yt, tat)[1])}


def holm(ps):
    k = sorted(ps, key=lambda t: ps[t])
    m, ra, lon = len(k), {}, 0.0
    for i, t in enumerate(k):
        lon = max(lon, min(1.0, (m - i) * ps[t]))
        ra[t] = lon
    return ra


def lenh_phan_tich(a):
    ra = {"dong_bang": kiem_bang(),
          "kiem_tai_lap": json.loads((RA / "kiem_tai_lap.json").read_text(encoding="utf-8"))
          if (RA / "kiem_tai_lap.json").exists() else None}
    TA, TB, TK = _nap("A"), _nap("B"), _nap("K10")
    if not TA:
        raise SystemExit("chưa có bản ghi chấm A")
    A = Bo(TA)
    ra["du_lieu"] = {"A": {"tinh_huong": len(TA), "cua_so": int(len(A.u))}, "B": {"tinh_huong": len(TB)},
                     "K10": {"tinh_huong": len(TK)}, "seed": {k: list(v) for k, v in SEED_MH.items()}}
    ra["kiem_thay"] = float(np.nanmax(np.abs(A.m("thay_x10", MAT) - A.m("thay_x10_ens", MAT))))
    mat = lambda bo, k: bo.m(k, MAT)
    duoi = lambda bo, k, ng=NGUONG: bo.m(k, ham=lambda X: _duoi(X, ng))
    tre = lambda bo, k: bo.m(k, TRE)
    dp = lambda bo, j: bo.m("dp", j)
    P = {}
    P["P1"] = A.boot(mat(A, "x10"), mat(A, "w07"), KHONG_KEM)
    P["P2"] = A.boot(duoi(A, "x10"), duoi(A, "w07"))
    P["P3"] = A.boot(mat(A, "w07_ktn"), mat(A, "w07"), cap=True)
    P["P4"] = A.boot(mat(A, "x10_ktn"), mat(A, "x10"), cap=True)
    P["P5"] = hoi_quy(A, TA)
    P["P6"] = A.boot(mat(A, "x10"), dp(A, MAT))
    p7m = A.boot(mat(A, "thay_x10"), mat(A, "thay_73"), KHONG_KEM)
    p7d = A.boot(duoi(A, "thay_x10"), duoi(A, "thay_73"))
    P["P7"] = {"mat": p7m, "duoi": p7d, "p_nho_hon": max(p7m["p_nho_hon"], p7d["p_nho_hon"])}
    Bb = Bo(TB) if TB else None
    if Bb:
        p8m = Bb.boot(mat(Bb, "de_xuat"), mat(Bb, "dm"))
        p8d = Bb.boot(duoi(Bb, "de_xuat"), duoi(Bb, "dm"))
        P["P8"] = {"mat": p8m, "duoi": p8d, "p_nho_hon": max(p8m["p_nho_hon"], p8d["p_nho_hon"])}
    ps = {k: (v["p_lon_hon"] if k == "P5" else v["p_nho_hon"]) for k, v in P.items()}
    hp = holm(ps)
    for k in P:
        P[k]["p_holm"] = hp[k]
        P[k]["dat"] = bool(hp[k] < 0.05)
    ra["chinh"] = P
    aam, aad = A.boot(mat(A, "x10_aa"), mat(A, "x10"), cap=True), A.boot(duoi(A, "x10_aa"), duoi(A, "x10"), cap=True)
    ra["P0"] = {"mat": aam, "duoi": aad,
                "khong_y_nghia": bool(aam["ci"][0] <= 0 <= aam["ci"][1] and aad["ci"][0] <= 0 <= aad["ci"][1])}

    # ------------------------------------------------------------ chỉ số phụ
    Sx = {}
    Sx["S0"] = {"mat": A.boot(mat(A, "x10"), mat(A, "dm"), KHONG_KEM), "duoi": A.boot(duoi(A, "x10"), duoi(A, "dm")),
                "tre": A.boot(tre(A, "x10"), tre(A, "dm")), "ktn_dm": A.boot(mat(A, "dm_ktn"), mat(A, "dm"), cap=True)}
    Sx["S1"] = {"tre_tb": A.boot(tre(A, "x10"), tre(A, "w07")),
                "tre_trung_vi": [float(np.median(tre(A, "w07"))), float(np.median(tre(A, "x10")))],
                "duoi_50": A.boot(duoi(A, "x10", 50), duoi(A, "w07", 50)),
                "duoi_200": A.boot(duoi(A, "x10", 200), duoi(A, "w07", 200))}
    Sx["S2"] = A.boot(mat(A, "x10"), mat(A, "w07"))
    Xe = np.array([x["X_eff"] for x in TA])
    nh = np.digitize(Xe, [3, 10, 30])
    Sx["S3"] = {"he_so_log_X_hieu_tre": P["P5"]["he_so_log_X_hieu_tre"], "nhom": {}}
    for g, ten in enumerate(["< 3", "3–10", "10–30", "> 30"]):
        msk = nh == g
        if msk.sum() >= 30:
            con = Bo([x for x, mm in zip(TA, msk) if mm])
            Sx["S3"]["nhom"][ten] = {"ti_le_%": float(msk.mean() * 100), "mat": con.boot(mat(con, "x10"), mat(con, "w07")),
                                     "tre": float(tre(con, "x10").mean() - tre(con, "w07").mean())}
    from scipy import stats
    dB = A.m("x10", B_).mean(0) - A.m("w07", B_).mean(0)
    hi = (tre(A, "w07") > NGUONG).mean(0) > 0
    Sx["S4"] = {"phan_hieu_b_tu_ca_duoi_w07_%": float(dB[hi].sum() / dB.sum() * 100) if abs(dB.sum()) > 1e-12 else None,
                "spearman_hieu_b_hieu_tre": float(stats.spearmanr(dB, tre(A, "x10").mean(0) - tre(A, "w07").mean(0)).statistic)}
    Sx["S5"] = {w: {"mat": A.boot(mat(A, "x10"), mat(A, w)), "duoi": A.boot(duoi(A, "x10"), duoi(A, w)),
                    "tre": A.boot(tre(A, "x10"), tre(A, w))} for w in ("w05", "w07", "w09")}
    Sx["S6"] = {"tre_dm_x10_tru_thay_x10": A.boot(tre(A, "x10"), tre(A, "thay_x10")),
                "duoi_dm_x10_tru_thay_x10": A.boot(duoi(A, "x10"), duoi(A, "thay_x10")),
                "mat_dm_x10_tru_thay_x10": A.boot(mat(A, "x10"), mat(A, "thay_x10")),
                "tre_thay_x10_tru_dp": A.boot(tre(A, "thay_x10"), dp(A, TRE)),
                "tre_dm_x10_tru_dp": A.boot(tre(A, "x10"), dp(A, TRE))}
    r6 = Sx["S6"]["tre_dm_x10_tru_thay_x10"]["ci"]
    Sx["S6"]["doc"] = ("khoảng còn lại do mục tiêu" if -0.5 <= r6[0] and r6[1] <= 0.5 else
                       "DM chậm hơn thầy (một phần do bắt chước)" if r6[1] > 0.5 and r6[0] > -0.5 else "chưa kết luận được")
    if Bb:
        Sx["S7"] = {"va": Bb.boot(mat(Bb, "de_xuat"), mat(Bb, "x10_ktn"), cap=True),
                    "de_xuat_tru_w07": {"mat": Bb.boot(mat(Bb, "de_xuat"), mat(Bb, "w07")),
                                        "duoi": Bb.boot(duoi(Bb, "de_xuat"), duoi(Bb, "w07"))}}
    chon = np.array([x["chon_seed"] for x in TA])
    xk = mat(A, "x10_ktn")
    v_chon = xk[chon, np.arange(len(chon))][None]
    Sx["S8"] = {"chon_tru_tb": A.boot(v_chon, xk), "can_tren_biet_truoc": float(xk.min(0).mean() - xk.mean())}
    Sx["S9"] = A.boot(mat(A, "x10_nhanh"), mat(A, "x10_ktn"), KHONG_KEM, cap=True)
    th = lambda k, j: np.array([[x[f"thuc_{k}_s{s}"][j] for x in TA] for s in SEED_MH[k]], float)
    Sx["S10"] = {"tre_thuc": A.boot(th("x10", 0), th("w07", 0)), "duoi_thuc": A.boot(th("x10", 1), th("w07", 1)),
                 "muc": {k: [float(np.nanmean(th(k, 0))), float(np.nanmean(th(k, 1)))] for k in ("dm", "w07", "x10")}}
    if TK:
        K = Bo(TK)
        Sx["S11"] = {"mat": K.boot(mat(K, "x10"), mat(K, "w07"), KHONG_KEM), "duoi": K.boot(duoi(K, "x10"), duoi(K, "w07")),
                     "ktn_x10": K.boot(mat(K, "x10_ktn"), mat(K, "x10"), cap=True),
                     "x10_tru_dm11": K.boot(mat(K, "x10"), mat(K, "dm"), KHONG_KEM), "x10_tru_dp": K.boot(mat(K, "x10"), dp(K, MAT))}
    loai = np.array([c.startswith("bao") for c in A.u])
    dw = np.array([((mat(A, "x10").mean(0) - mat(A, "w07").mean(0))[A.inv == g]).mean() for g in range(len(A.u))])
    tw = np.array([((duoi(A, "x10").mean(0) - duoi(A, "w07").mean(0))[A.inv == g]).mean() for g in range(len(A.u))])
    sd_ = lambda v, m: float(v[m].std(ddof=1)) if m.sum() > 1 else None
    Sx["S12"] = {"sd_cua_so_mat": {"bao": sd_(dw, loai), "ngay": sd_(dw, ~loai)},
                 "sd_cua_so_duoi": {"bao": sd_(tw, loai), "ngay": sd_(tw, ~loai)},
                 "tn15_2017_21_tham_chieu": {"mat": 0.22, "duoi": 0.82}}
    luoi = sorted(TA[0]["luoi"])
    tb_luoi = {g: float(np.mean([x["luoi"][g][MAT] for x in TA])) for g in luoi}
    g_tot = min(tb_luoi, key=tb_luoi.get)
    Sx["S13"] = {"dp_tot_nhat": g_tot, "mat": A.boot(mat(A, "x10"), np.array([[x["luoi"][g_tot][MAT] for x in TA]]))}
    Sx["S14"] = {"mat": A.boot(mat(A, "w07"), mat(A, "dm")), "duoi": A.boot(duoi(A, "w07"), duoi(A, "dm")),
                 "tre": A.boot(tre(A, "w07"), tre(A, "dm"))}
    s14 = Sx["S14"]
    Sx["S14"]["doc"] = ("không có nhiễu đường ống đáng kể" if -0.10 <= s14["mat"]["ci"][0] and s14["mat"]["ci"][1] <= 0.10
                        and -0.5 <= s14["duoi"]["ci"][0] and s14["duoi"]["ci"][1] <= 0.5 else "CÓ thể có nhiễu đường ống")
    Sx["S15"] = {"mat_x3_tru_x10": A.boot(mat(A, "x3"), mat(A, "x10"), KHONG_KEM),
                 "duoi_x3_tru_x10": A.boot(duoi(A, "x3"), duoi(A, "x10")),
                 "tre_x3_tru_x10": A.boot(tre(A, "x3"), tre(A, "x10")), "tre_x3_tru_dp": A.boot(tre(A, "x3"), dp(A, TRE)),
                 "mat_x3_tru_w07": A.boot(mat(A, "x3"), mat(A, "w07"))}
    s15 = Sx["S15"]
    Sx["S15"]["doc"] = {"x_nho_hon_tot_hon": bool(s15["mat_x3_tru_x10"]["ci"][1] < KHONG_KEM and s15["duoi_x3_tru_x10"]["ci"][1] < 0),
                        "dong_khoang_tre_dijkstra": bool(s15["tre_x3_tru_dp"]["ci"][0] <= 0 <= s15["tre_x3_tru_dp"]["ci"][1])}
    ra["phu"] = Sx
    ds = ("dm", "dm_ktn", "w07", "w07_ktn", "x10", "x10_ktn", "x10_nhanh", "x3", "w05", "w09", "thay_73", "thay_x10", "dp")
    ra["trung_binh"] = {k: {"mat": float(mat(A, k).mean()), "a": float(A.m(k, A_).mean()), "b": float(A.m(k, B_).mean()),
                            "tre": float(tre(A, k).mean()), "duoi": float(duoi(A, k).mean()), "so_seed": int(len(mat(A, k)))}
                        for k in ds}
    if Bb:
        ra["trung_binh_B"] = {k: {"mat": float(mat(Bb, k).mean()), "duoi": float(duoi(Bb, k).mean())}
                              for k in ("dm", "w07", "x10_ktn", "de_xuat")}
        ra["nhom_khop"] = json.loads((RA / "nhom_khop.json").read_text()) if (RA / "nhom_khop.json").exists() else None
    RA.mkdir(parents=True, exist_ok=True)
    (RA / "ket_qua_15b.json").write_text(json.dumps(ra, ensure_ascii=False, indent=1), encoding="utf-8")
    bao_cao(ra)
    ve(ra)
    if not THU:
        goi()
    ghi("  " + "; ".join(f"{k}: {'ĐẠT' if v['dat'] else 'KHÔNG ĐẠT'}" for k, v in P.items())
        + f"; P0 (A/A) {'không có ý nghĩa — tốt' if ra['P0']['khong_y_nghia'] else 'CÓ Ý NGHĨA — CI quá hẹp'}")


def kiem_bang():
    t = RA / "dong_bang.json"
    if not t.exists():
        return {"canh_bao": "chưa đóng băng"}
    cu = json.loads(t.read_text(encoding="utf-8"))
    doi = [p for p, h in cu["bam"].items() if (GOC / p).exists() and bam(GOC / p) != h]
    return {"luc": cu["luc"], "tep_doi_sau_dong_bang": doi}


def _ci(d, n=2):
    return f"{d['diem']:+.{n}f} [{d['ci'][0]:+.{n}f}; {d['ci'][1]:+.{n}f}]"


def bao_cao(ra):
    P, Sx = ra["chinh"], ra["phu"]
    L = ["# Thí nghiệm 15b — kết quả (sinh tự động bởi `scripts/tn15b.py`)", "",
         "Tiêu chí: TIEU_CHI_15B.md (bản 2, có học lại). Bộ kiểm chứng mới (thời tiết 2017–2021 chưa dùng). Hiệu bằng điểm % "
         "(trễ: ms); CI 95% bootstrap hai tầng (cửa sổ thời tiết + seed), p một phía, họ P1–P8 hiệu chỉnh Holm. "
         "w0.7 = DM 7/3 học lại cùng đường ống với x10.", ""]
    db = ra["dong_bang"]
    L += [f"Đóng băng: {db.get('luc', '—')}; tệp đổi sau đóng băng: {db.get('tep_doi_sau_dong_bang', db.get('canh_bao'))}.",
          f"Kiểm tái lập nhãn: {ra['kiem_tai_lap'].get('ti_le_khop') if ra['kiem_tai_lap'] else '—'}.",
          f"Dữ liệu: A {ra['du_lieu']['A']['tinh_huong']} tình huống / {ra['du_lieu']['A']['cua_so']} cửa sổ; "
          f"B {ra['du_lieu']['B']['tinh_huong']}; K10 {ra['du_lieu']['K10']['tinh_huong']}. "
          f"Kiểm chấm thầy x10 (thế giới chấm thu gọn so với bộ sinh): lệch tối đa {ra['kiem_thay']:.3g} điểm.", ""]
    ten = {"dm": "DM11 (TN11, máy khác)", "dm_ktn": "DM11 ktn", "w07": "DM 7/3 cùng đường ống (w0.7)", "w07_ktn": "w0.7 ktn",
           "x10": "DM x10", "x10_ktn": "DM x10 ktn", "x10_nhanh": "DM x10 ktn 5×8", "x3": "DM x3", "w05": "DM w0,5",
           "w09": "DM w0,9", "thay_73": "thầy 7/3", "thay_x10": "thầy x10", "dp": "Dijkstra phạt dp_50_50"}
    L += ["## 1. Trung bình (A, 4–5 luồng)", "", "| Phương pháp | seed | mất % | (a) | (b) | trễ ms | đuôi % |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for k, v in ra["trung_binh"].items():
        L.append(f"| {ten[k]} | {v['so_seed']} | {v['mat']:.2f} | {v['a']:.2f} | {v['b']:.2f} | {v['tre']:.1f} | {v['duoi']:.2f} |")
    L += ["", "## 2. Giả thuyết chính (Holm)", "", "| Mã | Phép thử | Ước lượng [CI 95%] | p | p Holm | Kết quả |",
          "|---|---|---|---|---|---|"]
    mo = {"P1": "mất x10 − w0.7 < +0,10", "P2": "đuôi x10 − w0.7 < 0", "P3": "mất w0.7 ktn − w0.7 < 0",
          "P4": "mất x10 ktn − x10 < 0", "P5": "hệ số log X_eff (hiệu mất x10 − w0.7) > 0", "P6": "mất x10 − Dijkstra phạt < 0",
          "P7": "thầy x10 − thầy 7/3: mất < +0,10 và đuôi < 0", "P8": "đề xuất − DM11: mất < 0 và đuôi < 0"}
    for k, v in P.items():
        if k == "P5":
            uoc, p = f"{v['he_so_log_X']:+.3f} [{v['ci'][0]:+.3f}; {v['ci'][1]:+.3f}]", v["p_lon_hon"]
        elif k in ("P7", "P8"):
            uoc, p = f"mất {_ci(v['mat'])}; đuôi {_ci(v['duoi'])}", v["p_nho_hon"]
        else:
            uoc, p = _ci(v), v["p_nho_hon"]
        L.append(f"| {k} | {mo[k]} | {uoc} | {p:.2g} | {v['p_holm']:.2g} | {'**ĐẠT**' if v['dat'] else 'KHÔNG ĐẠT'} |")
    p0 = ra["P0"]
    L += ["", f"**P0 (A/A, phải không có ý nghĩa):** mất {_ci(p0['mat'])}, đuôi {_ci(p0['duoi'])} → "
          f"{'không có ý nghĩa: CI đáng tin' if p0['khong_y_nghia'] else '**CÓ Ý NGHĨA: CI quá hẹp, đọc lại mọi kết luận**'}.", ""]
    s0 = Sx["S0"]
    L += ["## 3. Chỉ số phụ", "", "| Mã | Chỉ số | Ước lượng [CI 95%] |", "|---|---|---|",
          f"| S0 | lặp lại TN15 với DM11: mất / đuôi / trễ x10 − DM11; ktn DM11 | {_ci(s0['mat'])} / {_ci(s0['duoi'])} / "
          f"{_ci(s0['tre'], 1)}; {_ci(s0['ktn_dm'])} |",
          f"| S1 | trễ trung bình x10 − w0.7 (ms) | {_ci(Sx['S1']['tre_tb'], 1)} |",
          f"| S1 | trễ trung vị w0.7 → x10 | {Sx['S1']['tre_trung_vi'][0]:.2f} → {Sx['S1']['tre_trung_vi'][1]:.2f} |",
          f"| S1 | đuôi > 50 ms / > 200 ms | {_ci(Sx['S1']['duoi_50'])} / {_ci(Sx['S1']['duoi_200'])} |",
          f"| S2 | mất x10 − w0.7 < 0 | {_ci(Sx['S2'])}, p = {Sx['S2']['p_nho_hon']:.2g} |"]
    for ten_n, v in Sx["S3"]["nhom"].items():
        L.append(f"| S3 | X_eff {ten_n} ({v['ti_le_%']:.0f}%): mất / trễ | {_ci(v['mat'])} / {v['tre']:+.1f} ms |")
    L.append(f"| S3 | hệ số log X_eff của hiệu trễ | {Sx['S3']['he_so_log_X_hieu_tre']:+.2f} |")
    s4 = Sx["S4"]
    pb = "—" if s4["phan_hieu_b_tu_ca_duoi_w07_%"] is None else f"{s4['phan_hieu_b_tu_ca_duoi_w07_%']:.0f}%"
    L.append(f"| S4 | phần hiệu (b) từ ca w0.7 có đuôi; Spearman (hiệu b, hiệu trễ) | {pb}; {s4['spearman_hieu_b_hieu_tre']:+.2f} |")
    for w in ("w05", "w07", "w09"):
        v = Sx["S5"][w]
        L.append(f"| S5 | x10 − {w}: mất / đuôi / trễ | {_ci(v['mat'])} / {_ci(v['duoi'])} / {_ci(v['tre'], 1)} |")
    v = Sx["S6"]
    L.append(f"| S6 | DM x10 − thầy x10: trễ / đuôi / mất | {_ci(v['tre_dm_x10_tru_thay_x10'], 1)} / "
             f"{_ci(v['duoi_dm_x10_tru_thay_x10'])} / {_ci(v['mat_dm_x10_tru_thay_x10'])} → {v['doc']} |")
    L.append(f"| S6 | trễ − Dijkstra: thầy x10 / DM x10 | {_ci(v['tre_thay_x10_tru_dp'], 1)} / {_ci(v['tre_dm_x10_tru_dp'], 1)} |")
    if "S7" in Sx:
        v = Sx["S7"]
        L.append(f"| S7 | vá trên x10 ktn (B); đề xuất − w0.7: mất / đuôi | {_ci(v['va'])}; {_ci(v['de_xuat_tru_w07']['mat'])} / "
                 f"{_ci(v['de_xuat_tru_w07']['duoi'])} |")
    L.append(f"| S8 | chọn seed − trung bình (x10 ktn) | {_ci(Sx['S8']['chon_tru_tb'])}; cận trên biết trước "
             f"{Sx['S8']['can_tren_biet_truoc']:+.2f} |")
    L.append(f"| S9 | x10 ktn 5×8 − 5×32 (< +0,10) | {_ci(Sx['S9'])} |")
    s10 = Sx["S10"]
    L.append(f"| S10 | trễ thực x10 − w0.7 (ms) / % > 100 ms | {_ci(s10['tre_thuc'], 1)} / {_ci(s10['duoi_thuc'])} |")
    if "S11" in Sx:
        v = Sx["S11"]
        L.append(f"| S11 | K10: mất x10 − w0.7 / đuôi / ktn x10 / x10 − DM11 / x10 − Dijkstra | {_ci(v['mat'])} / {_ci(v['duoi'])} / "
                 f"{_ci(v['ktn_x10'])} / {_ci(v['x10_tru_dm11'])} / {_ci(v['x10_tru_dp'])} |")
    v = Sx["S12"]
    so = lambda d: " / ".join("—" if d[k] is None else f"{d[k]:.2f}" for k in ("bao", "ngay"))
    L.append(f"| S12 | SD giữa cửa sổ (bão / ngày): hiệu mất; hiệu đuôi | {so(v['sd_cua_so_mat'])}; {so(v['sd_cua_so_duoi'])} "
             f"(TN15 2017–21: 0,22; 0,82) |")
    L.append(f"| S13 | x10 − Dijkstra tốt nhất sau khi xem ({Sx['S13']['dp_tot_nhat']}) | {_ci(Sx['S13']['mat'])} |")
    v = Sx["S14"]
    L.append(f"| S14 | nhiễu đường ống: w0.7 − DM11, mất / đuôi / trễ | {_ci(v['mat'])} / {_ci(v['duoi'])} / {_ci(v['tre'], 1)} → "
             f"{v['doc']} |")
    v = Sx["S15"]
    L.append(f"| S15 | quét X: x3 − x10 mất / đuôi / trễ; x3 − Dijkstra trễ; x3 − w0.7 mất | {_ci(v['mat_x3_tru_x10'])} / "
             f"{_ci(v['duoi_x3_tru_x10'])} / {_ci(v['tre_x3_tru_x10'], 1)}; {_ci(v['tre_x3_tru_dp'], 1)}; {_ci(v['mat_x3_tru_w07'])} → "
             f"X nhỏ hơn tốt hơn: {v['doc']['x_nho_hon_tot_hon']}; đóng khoảng trễ Dijkstra: {v['doc']['dong_khoang_tre_dijkstra']} |")
    if "trung_binh_B" in ra:
        L += ["", "B (kho nhanh): " + ", ".join(f"{k} mất {v['mat']:.2f} / đuôi {v['duoi']:.2f}" for k, v in ra["trung_binh_B"].items())
              + f"; nhóm khớp {ra.get('nhom_khop')}."]
    L += ["", "Hình: `hinh_15b.png`."]
    (RA / "KET_QUA_15B.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def ve(ra):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    P, Sx = ra["chinh"], ra["phu"]
    trai = [("P1 mất x10 − w0.7", P["P1"]), ("S0 mất x10 − DM11", Sx["S0"]["mat"]), ("P3 mất w0.7 ktn − đầy đủ", P["P3"]),
            ("P4 mất x10 ktn − đầy đủ", P["P4"]), ("P6 mất x10 − Dijkstra", P["P6"]), ("P7 mất thầy x10 − thầy 7/3", P["P7"]["mat"])]
    if "P8" in P:
        trai.append(("P8 mất đề xuất − DM11", P["P8"]["mat"]))
    trai += [("S14 mất w0.7 − DM11 (đường ống)", Sx["S14"]["mat"]), ("S15 mất x3 − x10", Sx["S15"]["mat_x3_tru_x10"]),
             ("P0 mất A/A (phải chứa 0)", ra["P0"]["mat"])]
    phai = [("P2 đuôi x10 − w0.7", P["P2"]), ("S0 đuôi x10 − DM11", Sx["S0"]["duoi"]), ("P7 đuôi thầy x10 − thầy 7/3", P["P7"]["duoi"])]
    if "P8" in P:
        phai.append(("P8 đuôi đề xuất − DM11", P["P8"]["duoi"]))
    phai += [("S1 đuôi > 50 ms", Sx["S1"]["duoi_50"]), ("S1 đuôi > 200 ms", Sx["S1"]["duoi_200"]),
             ("S15 đuôi x3 − x10", Sx["S15"]["duoi_x3_tru_x10"]), ("P0 đuôi A/A (phải chứa 0)", ra["P0"]["duoi"])]
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.6))
    for a, ds, tieu, nguong in ((ax[0], trai, "mất dịch vụ (điểm %)", KHONG_KEM), (ax[1], phai, "đuôi trễ > ngưỡng (điểm %)", None)):
        for i, (ten, v) in enumerate(ds[::-1]):
            mau = "#888888" if ten.startswith(("P0", "S")) else "#eb6834"
            a.errorbar(v["diem"], i, xerr=[[max(0.0, v["diem"] - v["ci"][0])], [max(0.0, v["ci"][1] - v["diem"])]], fmt="o",
                       color=mau, ecolor=mau, capsize=3, lw=1.5, ms=6)
        a.set_yticks(range(len(ds)), [t for t, _ in ds[::-1]], fontsize=8.5)
        a.axvline(0, color="#444444", lw=0.8)
        if nguong:
            a.axvline(nguong, color="#c4302b", lw=0.8, ls="--")
            a.text(nguong, len(ds) - 0.4, " ngưỡng không kém", color="#c4302b", fontsize=7.5)
        a.set_xlabel(tieu)
        a.grid(axis="x", alpha=0.3)
    fig.suptitle("TN15b — bộ kiểm chứng mới, CI 95% hai tầng (cửa sổ thời tiết + seed); cam = giả thuyết chính, xám = phụ / A/A",
                 fontsize=10.5, x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(RA / "hinh_15b.png", dpi=150)
    plt.close(fig)


def goi():
    z = KQ / "TN15B_ket_qua.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as f:
        for p in RA.rglob("*"):
            if p.is_file():
                f.write(p, p.relative_to(KQ))
        for p in LOG.glob("tn15b*.log"):
            if p.name != "tn15b_thu.log":
                f.write(p, f"logs/{p.name}")
        for mt, s in HOC_MOI:
            for duoi_tep in ("_nhat_ky.json", ".pt"):
                p = M13 / f"dm13_{mt}_s{s}{duoi_tep}"
                if p.exists():
                    f.write(p, f"mo_hinh13/{p.name}")
        f.write(TIEU_CHI, TIEU_CHI.relative_to(KQ))
    ghi(f"  đã gói → {z} ({z.stat().st_size / 1e6:.1f} MB)")


# ============================================================================================ main
BUOC = [("dong-bang", lenh_dong_bang), ("gan-nhan", lenh_gan_nhan), ("kiem-nhan", lenh_kiem_nhan), ("hoc", lenh_hoc),
        ("sinh", lenh_sinh), ("nhom", lenh_nhom), ("cham", lenh_cham), ("phan-tich", lenh_phan_tich)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lenh", choices=[b for b, _ in BUOC] + ["toi-da"])
    ap.add_argument("--thu", default=None, help="chạy thử trên dữ liệu cũ (nam9) ở thư mục này, không gán nhãn / học")
    ap.add_argument("--tien-trinh", type=int, default=None, help="số tiến trình chấm (mặc định: tự chọn)")
    ap.add_argument("--tien-trinh-sinh", type=int, default=None, help="số tiến trình sinh / gán nhãn (mặc định: tự chọn)")
    ap.add_argument("--song-song", type=int, default=None, help="số mô hình học cùng lúc (mặc định: tự chọn)")
    ap.add_argument("--luong-cpu", type=int, default=None)
    ap.add_argument("--thiet-bi", default=None, choices=["cpu", "auto", "cuda"])
    ap.add_argument("--ram", type=float, default=20.0, help="ngân sách RAM (GB)")
    ap.add_argument("--vram", type=float, default=8.0, help="ngân sách VRAM (GB)")
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if a.thu:
        dat_thu(a.thu)
    p = t11.tai_nguyen(a.ram, a.vram)
    a.tien_trinh = a.tien_trinh or p["cham"]
    a.tien_trinh_sinh = a.tien_trinh_sinh or p["sinh"]
    a.song_song = a.song_song or p["hoc"]
    a.luong_cpu = a.luong_cpu or p["luong"]
    a.thiet_bi = a.thiet_bi or p["thiet_bi"]
    ghi(f"máy: {p['cpu']} luồng CPU, RAM {p['ram']} GB, GPU {p['gpu']} → sinh {a.tien_trinh_sinh} tiến trình; học "
        f"{a.song_song} mô hình × {a.luong_cpu} luồng ({a.thiet_bi}); chấm {a.tien_trinh} tiến trình")
    buoc = BUOC if a.lenh == "toi-da" else [b for b in BUOC if b[0] == a.lenh]
    for ten, f in buoc:
        ghi(f"=== {ten} {time.strftime('%Y-%m-%d %H:%M:%S')}")
        f(a)
    ghi("xong")


if __name__ == "__main__":
    main()
