"""
tn12.py — Thí nghiệm 12 (results/bao_da_tuyen/TIEU_CHI_12.md): cơ chế khuếch tán có thật là yếu tố tạo độ chính xác?
Thang bậc, mỗi bậc thêm MỘT thành phần, chấm trên 11 bộ kiểm tra của TN11 (4–5 luồng, 3 seed):

    0 Dijkstra phạt · 1 TKCB trên bảng rủi ro · 2 SM11 đoán 1 lần · 3 SM11 chọn trong 32 mẫu · 4 DM11 1 bước × 32 chuỗi
    5 DM11 5 bước × 1 chuỗi · 6 DM11 đầy đủ (5 × 32) · 7 DM không phối hợp · 8 DM lịch nhiễu tuyến tính
    9 softmax chống quá khớp — và quét số bước {1, 2, 3, 5, 10} × số chuỗi {1, 8, 32} của DM11 (seed 0).

    python -m scripts.tn12 toi-da                 # chạy hết (cần mô hình TN11 ở results/bao_da_tuyen/mo_hinh11/)

Từng bước (chạy lại an toàn — phần đã xong tự bỏ qua):
    huan-luyen   dm12k (không phối hợp), dm12l (lịch tuyến tính): 3 seed mỗi họ; softmax chống quá khớp: 2 cấu hình ở
                 seed 0, chọn theo J val, rồi seed 1, 2 của cấu hình được chọn → mo_hinh12/
    cham         mọi bậc + phép quét trên 11 bộ kiểm tra → tn12/cham_tep/
    phan-tich    → tn12/KET_QUA_12.md, ket_qua_12.json, hinh_12_*.png; gói results/bao_da_tuyen/TN12_ket_qua.zip
Log: logs/tn12.log; mỗi mô hình logs/tn12_<tên>.log. Không cần sinh dữ liệu mới.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

GOC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GOC))
from scripts import tn11 as t11  # noqa: E402

KQ, DL, LOG = t11.KQ, t11.DL, t11.LOG
RA, M12, M11 = KQ / "tn12", KQ / "mo_hinh12", t11.M11
BO_45, CUC_DOAN, TEN_BO, DO = t11.BO_45, t11.CUC_DOAN, t11.TEN_BO, t11.DO
SEEDS, CFG = (0, 1, 2), {"epochs": 120, "H": 128, "dung_som": 20, "S": 32}
# biến thể học lại: tên họ → (kiểu, cờ của huan_luyen_dt)
BIEN_THE = {"dm12k": ("dm", ["--khong-phoi-hop"]), "dm12l": ("dm", ["--lich", "tuyen_tinh"]),
            "sm12a": ("softmax", ["--dropout", "0.1", "--weight-decay", "1e-4"]),
            "sm12b": ("softmax", ["--dropout", "0.3", "--weight-decay", "1e-3"])}
BAC = [("Dijkstra phạt", "dphat", False), ("TKCB trên bảng rủi ro", "online_tkcb", False),
       ("SM11 · đoán 1 lần", "sm_1", True), ("SM11 · chọn trong 32 mẫu", "sm_32", True),
       ("DM11 · 1 bước × 32 chuỗi", "dm_1x32", True), ("DM11 · 5 bước × 1 chuỗi", "dm_5x1", True),
       ("DM11 · đầy đủ 5 × 32", "dm_5x32", True), ("DM không phối hợp", "dm12k", True),
       ("DM lịch nhiễu tuyến tính", "dm12l", True), ("Softmax chống quá khớp", "sm12", True)]
QUET_BUOC, QUET_S = (1, 2, 3, 5, 10), (1, 8, 32)
CAP_DM11 = ((5, 32), (1, 32), (5, 1))                       # các cấu hình giải mã DM11 chấm cho cả 3 seed
NGUONG_TUONG_DUONG = 0.3                                    # điểm % — H12.5


def py(*a):
    return t11.py(*a)


# dữ liệu TN11 mà TN12 / TN13 đọc: tập học (TN13 so tái lập), val, 11 bộ kiểm tra 4–5 luồng
DL_TN11 = ("train10_k4_5", "train6_10_k4_5", "x11_k4_5", "val_k4_5", "val6_k4_5") + tuple(
    t11.thu_muc_bo(b, "4-5") for b in BO_45)


TEP_GOI_TN11 = "TN11_cho_TN12_TN13.zip"


def goi_tn11(a=None):
    """Chạy trên MÁY ĐÃ CHẠY TN11: gói mô hình mo_hinh11/, kết quả tn11/ và dữ liệu DL_TN11 thành một zip (đường dẫn tương
    đối results/bao_da_tuyen/…, data/bao_da_tuyen/…) để chép sang máy khác."""
    import zipfile
    tep = KQ / TEP_GOI_TN11
    cap = [(M11, "results/bao_da_tuyen/mo_hinh11"), (KQ / "tn11", "results/bao_da_tuyen/tn11")]
    cap += [(DL / d, f"data/bao_da_tuyen/{d}") for d in DL_TN11]
    thieu = [d.name for d, _ in cap if not d.exists()]
    if not (M11 / "dm11_s0.pt").exists():
        raise SystemExit(f"không thấy mô hình TN11 ở {M11} — chạy lệnh này trên máy đã chạy TN11")
    so = 0
    with zipfile.ZipFile(tep, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for d, ten in cap:
            for f in sorted(d.rglob("*")) if d.exists() else []:
                if f.is_file() and not f.name.endswith(".tam"):
                    z.write(f, f"{ten}/{f.relative_to(d).as_posix()}")
                    so += 1
    print(f"  đã gói {so} tệp → {tep} ({tep.stat().st_size / 1e9:.2f} GB)" + (f"; không có: {thieu}" if thieu else ""),
          flush=True)


def lay_tn11(nguon):
    """--tn11 <thư mục đã chạy TN11 | tệp TN11_cho_TN12_TN13.zip>: chép sang repo này những gì CÒN THIẾU — mô hình
    mo_hinh11/, kết quả tn11/ (cham_tep để kiểm lặp lại) và các thư mục dữ liệu DL_TN11. Tệp đã có thì giữ nguyên; chạy lại an
    toàn."""
    import shutil
    g = Path(nguon).resolve()
    if g.suffix.lower() == ".zip":
        import zipfile
        goc_tuong_doi = {"results/bao_da_tuyen/": KQ, "data/bao_da_tuyen/": DL}
        so = 0
        with zipfile.ZipFile(g) as z:
            for ten in z.namelist():
                tien_to = next((p for p in goc_tuong_doi if ten.startswith(p)), None)
                if tien_to is None or ten.endswith("/"):
                    continue
                dich = goc_tuong_doi[tien_to] / ten[len(tien_to):]
                if not dich.exists():
                    dich.parent.mkdir(parents=True, exist_ok=True)
                    with z.open(ten) as nguon_f, open(dich, "wb") as dich_f:
                        shutil.copyfileobj(nguon_f, dich_f)
                    so += 1
        print(f"  lấy dữ liệu TN11 từ {g.name}: giải nén {so} tệp còn thiếu", flush=True)
        return
    if g == GOC.resolve():
        return
    kq_g, dl_g = g / "results" / "bao_da_tuyen", g / "data" / "bao_da_tuyen"
    if not (kq_g / "mo_hinh11").exists():
        raise SystemExit(f"--tn11: không thấy {kq_g / 'mo_hinh11'}")
    cap = [(kq_g / "mo_hinh11", M11), (kq_g / "tn11", KQ / "tn11")]
    cap += [(dl_g / d, DL / d) for d in DL_TN11]
    so, dung_luong, thieu = 0, 0, []
    for nguon_d, dich_d in cap:
        if not nguon_d.exists():
            thieu.append(nguon_d.name)
            continue
        for f in nguon_d.rglob("*"):
            dich = dich_d / f.relative_to(nguon_d)
            if f.is_file() and not dich.exists():
                dich.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, dich)
                so, dung_luong = so + 1, dung_luong + f.stat().st_size
    print(f"  lấy dữ liệu TN11 từ {g}: chép {so} tệp ({dung_luong / 1e9:.2f} GB)"
          + (f"; nguồn không có: {thieu}" if thieu else ""), flush=True)


# ============================================================================================ huấn luyện (dùng chung TN13)
def hang_doi(viec, song_song, thu_muc_mh, tien_to):
    """viec = [(tên, lệnh)]; luôn giữ song_song mô hình chạy; bỏ qua mô hình đã có <tên>.xong."""
    viec = [v for v in viec if not (Path(thu_muc_mh) / f"{v[0]}.xong").exists()]
    LOG.mkdir(parents=True, exist_ok=True)
    chay, loi = [], []
    while viec or chay:
        while viec and len(chay) < song_song and not loi:
            ten, lenh = viec.pop(0)
            tep_log = LOG / f"{tien_to}_{ten}.log"
            print(f"  {time.strftime('%H:%M:%S')} huấn luyện {ten} → {LOG.name}/{tep_log.name}", flush=True)
            chay.append((ten, subprocess.Popen(lenh, cwd=GOC, stderr=subprocess.STDOUT,
                                               stdout=open(tep_log, "w", encoding="utf-8"))))
        if not chay:
            break
        time.sleep(5)
        for ten, p in list(chay):
            if p.poll() is not None:
                chay.remove((ten, p))
                if p.returncode:
                    loi.append(ten)
                    print(f"  LỖI {ten} — xem {LOG.name}/{tien_to}_{ten}.log", flush=True)
                else:
                    print(f"  {time.strftime('%H:%M:%S')} xong {ten}", flush=True)
        if loi and not chay:
            break
    if loi:
        raise SystemExit(f"huấn luyện lỗi: {loi} — chạy lại đúng lệnh (mô hình xong rồi được giữ)")


def lenh_hoc(kieu, tn, ten, seed, a, co=()):
    return py("-m", "src.nhieu_luong.huan_luyen_dt", kieu, "--thi-nghiem", str(tn), "--ten", ten, "--them", "--seed",
              str(seed), "--epochs", str(CFG["epochs"]), "--H", str(CFG["H"]), "--dung-som", str(CFG["dung_som"]),
              "--luong-cpu", str(a.luong_cpu), "--thiet-bi", a.thiet_bi, *co)


def nhat_ky(thu_muc, ten):
    t = Path(thu_muc) / f"{ten}_nhat_ky.json"
    return json.loads(t.read_text(encoding="utf-8")) if t.exists() else None


def sm12_chon():
    """Cấu hình softmax chống quá khớp được chọn: J val tốt nhất của seed 0 (luật ghi trước)."""
    J = {h: nhat_ky(M12, f"{h}_s0") for h in ("sm12a", "sm12b")}
    if any(j is None for j in J.values()):
        return None
    return min(J, key=lambda h: (J[h]["tot_nhat_J_val"], h))


def lenh_huan_luyen(a):
    thieu = [f"{h}_s{s}" for h in ("dm11", "sm11") for s in SEEDS if not (M11 / f"{h}_s{s}.pt").exists()]
    if thieu:
        raise SystemExit(f"thiếu mô hình TN11 ở {M11}: {thieu}")
    viec = [(f"{h}_s{s}", lenh_hoc(BIEN_THE[h][0], 17, f"{h}_s{s}", s, a, BIEN_THE[h][1]))
            for s in SEEDS for h in ("dm12k", "dm12l")]
    viec += [(f"{h}_s0", lenh_hoc(BIEN_THE[h][0], 17, f"{h}_s0", 0, a, BIEN_THE[h][1])) for h in ("sm12a", "sm12b")]
    hang_doi(viec, a.song_song, M12, "tn12")
    h = sm12_chon()
    print(f"  softmax chống quá khớp: chọn {h} ({' '.join(BIEN_THE[h][1])}) theo J val seed 0", flush=True)
    RA.mkdir(parents=True, exist_ok=True)
    (RA / "chon_sm12.json").write_text(json.dumps({"chon": h, "co": BIEN_THE[h][1]}, ensure_ascii=False), encoding="utf-8")
    hang_doi([(f"{h}_s{s}", lenh_hoc("softmax", 17, f"{h}_s{s}", s, a, BIEN_THE[h][1])) for s in (1, 2)],
             a.song_song, M12, "tn12")


# ============================================================================================ chấm
def _gio(f):
    s = time.perf_counter()
    a = f()
    return a, (time.perf_counter() - s) * 1e3


def _cham_tep(job):
    import torch

    from src.nhieu_luong import mo_hinh_dt as md
    from src.nhieu_luong.cham_lai_dt import giong_ban_ghi
    from src.nhieu_luong.cham_mo_hinh_dt import _tg
    tep, sm12, quet = job
    torch.set_num_threads(1)
    import pickle
    ra = []
    G = lambda sd: torch.Generator().manual_seed(sd)
    for r in pickle.load(open(tep, "rb")):
        gon = r["gon"]
        tgw, sd, x = _tg(gon, r["cham"]), giong_ban_ghi(r), t11._co_ban(r)
        ens = r.get("ens", {})
        x["online_tkcb"] = [float(ens["online_tkcb"][m]) for m in DO] if "online_tkcb" in ens else [float("nan")] * len(DO)
        luu = lambda k, a, ms: x.update({k: t11._do(gon, tgw, a), f"ms_{k}": ms})
        for s in SEEDS:
            dm = t11._net(str(M11), f"dm11_s{s}")
            for nb, S in CAP_DM11:
                luu(f"dm_{nb}x{S}_s{s}", *_gio(lambda: md.chon_dm(dm, gon, S, nb, G(sd))))
            sm = t11._net(str(M11), f"sm11_s{s}")
            luu(f"sm_1_s{s}", *_gio(lambda: md.chon_softmax(sm, gon)))
            luu(f"sm_32_s{s}", *_gio(lambda: md.chon_softmax_mau(sm, gon, 32, G(sd))))
            for h in ("dm12k", "dm12l"):
                n = t11._net(str(M12), f"{h}_s{s}")
                luu(f"{h}_s{s}", *_gio(lambda: md.chon_dm(n, gon, CFG["S"], 5, G(sd))))
            n = t11._net(str(M12), f"{sm12}_s{s}")
            luu(f"sm12_s{s}", *_gio(lambda: md.chon_softmax(n, gon)))
        if quet:                                                # DM11 seed 0: số bước × số chuỗi
            dm = t11._net(str(M11), "dm11_s0")
            for nb in QUET_BUOC:
                for S in QUET_S:
                    k = f"quet_{nb}x{S}"
                    if (nb, S) in CAP_DM11:
                        x[k], x[f"ms_{k}"] = x[f"dm_{nb}x{S}_s0"], x[f"ms_dm_{nb}x{S}_s0"]
                    else:
                        luu(k, *_gio(lambda: md.chon_dm(dm, gon, S, nb, G(sd))))
        ra.append(x)
    return tep, ra


def lenh_cham(a):
    sm12 = sm12_chon()
    thieu = [f"{h}_s{s}" for h in ("dm12k", "dm12l", sm12 or "sm12?") for s in SEEDS if not (M12 / f"{h}_s{s}.xong").exists()]
    if thieu:
        raise SystemExit(f"chưa xong mô hình TN12: {thieu}")
    viec = [(f"{t11.thu_muc_bo(b, '4-5')}__{Path(t).stem}.json", (t, sm12, True))
            for b in BO_45 for t in sorted(glob.glob(str(DL / t11.thu_muc_bo(b, "4-5") / "kb_*.pkl")))]
    if not viec:
        raise SystemExit(f"không thấy bộ kiểm tra TN11 ở {DL}")
    print(f"  chấm bộ kiểm tra: {len(viec)} tệp", flush=True)
    t11._chay_tep(_cham_tep, viec, a.tien_trinh, RA / "cham_tep")


# ============================================================================================ phân tích
def _tai():
    return {b: rs for b in BO_45 if (rs := t11._gop(RA / "cham_tep", t11.thu_muc_bo(b, "4-5")))}


def _gia_tri(rs, khoa, co_seed, j=0):
    return t11._ho(rs, khoa, SEEDS, j) if co_seed else t11._cot(rs, khoa, j)


def _ci_dat(ci, chieu):
    """chieu = -1: CI dưới 0; +1: CI trên 0; 0: CI nằm trong ±NGUONG_TUONG_DUONG."""
    lo, hi = ci[1]
    return hi < 0 if chieu < 0 else lo > 0 if chieu > 0 else (-NGUONG_TUONG_DUONG < lo and hi < NGUONG_TUONG_DUONG)


def phan_tich_bo(T):
    bo, gop = {}, {}
    for b, rs in T.items():
        kb = np.array([r["kb"] for r in rs])
        v = {ten: _gia_tri(rs, k, cs) for ten, k, cs in BAC}
        thanh_phan = {ten: [float(_gia_tri(rs, k, cs, j).mean()) for j in (2, 3, 4)] for ten, k, cs in BAC}
        d = lambda x, y: t11._bs(v[x] - v[y], kb)
        bo[b] = {"n": len(rs), "trung_binh": {k: float(x.mean()) for k, x in v.items()},
                 "a_b_tre": thanh_phan,
                 "vs_dphat": {k: t11._bs(x - v["Dijkstra phạt"], kb) for k, x in v.items() if k != "Dijkstra phạt"},
                 "H12.1": d("DM11 · đầy đủ 5 × 32", "DM11 · 1 bước × 32 chuỗi"),
                 "H12.2": d("DM11 · đầy đủ 5 × 32", "SM11 · chọn trong 32 mẫu"),
                 "H12.3": d("DM không phối hợp", "DM11 · đầy đủ 5 × 32"),
                 "H12.4": d("Softmax chống quá khớp", "DM11 · đầy đủ 5 × 32"),
                 "H12.5": d("DM lịch nhiễu tuyến tính", "DM11 · đầy đủ 5 × 32"),
                 "lap_lai_buoc_vs_chuoi": d("DM11 · 5 bước × 1 chuỗi", "DM11 · 1 bước × 32 chuỗi")}
        for ten, k, cs in BAC:                                  # gộp mọi bộ (cụm = bộ × cửa sổ) cho tổng hợp / hình
            for j in (0, 3):
                gop.setdefault((ten, j), []).append(_gia_tri(rs, k, cs, j))
        gop.setdefault("cum", []).append(BO_45.index(b) * 1000 + kb)
    cum = np.concatenate(gop.pop("cum"))
    tong = {ten: {"mat_dv": t11._bs(np.concatenate(gop[(ten, 0)]), cum),
                  "qua_tai": float(np.concatenate(gop[(ten, 3)]).mean())} for ten, _, _ in BAC}
    l7, l6 = np.concatenate(gop[("DM không phối hợp", 0)]), np.concatenate(gop[("DM11 · đầy đủ 5 × 32", 0)])
    b7, b6 = np.concatenate(gop[("DM không phối hợp", 3)]), np.concatenate(gop[("DM11 · đầy đủ 5 × 32", 3)])
    ti_le_qua_tai = float((b7 - b6).mean() / (l7 - l6).mean()) if abs((l7 - l6).mean()) > 1e-9 else float("nan")
    return bo, tong, ti_le_qua_tai


def phan_tich_quet(T):
    rs = [r for v in T.values() for r in v]
    if not rs or "quet_1x1" not in rs[0]:
        return {}
    return {f"{nb}x{S}": {"mat_dv": float(np.mean([r[f"quet_{nb}x{S}"][0] for r in rs])),
                          "tre": float(np.mean([r[f"quet_{nb}x{S}"][4] for r in rs])),
                          "ms": float(np.median([r[f"ms_quet_{nb}x{S}"] for r in rs]))}
            for nb in QUET_BUOC for S in QUET_S}


def doc_hoc():
    ra = {}
    for f in sorted(M12.glob("*_nhat_ky.json")):
        j = json.loads(f.read_text(encoding="utf-8"))
        nk = j["nhat_ky"]
        b = min(nk, key=lambda r: r["J_val"])
        ra[f.name.replace("_nhat_ky.json", "")] = {
            "so_epoch": len(nk), "epoch_tot": b["epoch"], "dung_som_o": j.get("dung_som_o"),
            "loss_train_tot": b["loss_train"], "loss_val_tot": b["loss_val"],
            "ti_le_val_train": b["loss_val"] / b["loss_train"], "J_val_tot": b["J_val"], "J_thay": b["J_thay_val"],
            "chua_toi": len(nk) >= CFG["epochs"] and b["epoch"] >= 0.9 * CFG["epochs"],
            "nk": [[r["epoch"], r["loss_train"], r["loss_val"], r["J_val"]] for r in nk]}
    return ra


def kiem_lap_lai(T):
    """DM11 5 × 32 của TN12 phải trùng DM11 của TN11 (cùng mô hình, cùng hạt giống) nếu có tn11/cham_tep."""
    c11 = KQ / "tn11" / "cham_tep"
    if not c11.exists():
        return None
    lech, n = 0.0, 0
    for b, rs in T.items():
        r11 = t11._gop(c11, t11.thu_muc_bo(b, "4-5"))
        if len(r11) != len(rs):
            return {"loi": f"{b}: số bản ghi khác ({len(rs)} vs {len(r11)})"}
        for x, y in zip(rs, r11):
            for s in SEEDS:
                lech = max(lech, abs(x[f"dm_5x32_s{s}"][0] - y[f"dm11_s{s}"][0]))
                n += 1
    return {"so_so_sanh": n, "lech_toi_da": lech}


def gia_thuyet(bo, ti_le_qt, hoc, sm12):
    dem = lambda k, chieu: [b for b, x in bo.items() if _ci_dat(x[k], chieu)]
    n = len(bo)
    can = max(1, int(np.ceil(8 / 11 * n)))
    h1, h2, h3 = dem("H12.1", -1), dem("H12.2", -1), dem("H12.3", +1)
    h4 = dem("H12.4", +1)
    sm_hoc = [hoc.get(f"{sm12}_s{s}", {}) for s in SEEDS]
    sm_qua_khop = [f"{sm12}_s{s}" for s, x in zip(SEEDS, sm_hoc) if not x or x["ti_le_val_train"] > 1.25]
    tuong_duong, cos_tot, tt_tot = dem("H12.5", 0), dem("H12.5", +1), dem("H12.5", -1)
    kl5 = ("tương đương (lịch nhiễu không quan trọng)" if len(tuong_duong) >= can else
           "cos² tốt hơn" if len(cos_tot) >= can else "tuyến tính tốt hơn" if len(tt_tot) >= can else "không kết luận")
    return {"so_bo_can": can, "so_bo": n,
            "H12.1": {"bo_dat": h1, "dat": len(h1) >= can},
            "H12.2": {"bo_dat": h2, "dat": len(h2) >= can},
            "H12.3": {"bo_dat": h3, "ti_le_qua_tai": ti_le_qt, "dat": len(h3) >= can and ti_le_qt >= 0.6},
            "H12.4": {"bo_dat": h4, "softmax_chon": sm12, "van_qua_khop": sm_qua_khop,
                      "dat": len(h4) >= can and not sm_qua_khop},
            "H12.5": {"tuong_duong": tuong_duong, "cos_tot_hon": cos_tot, "tuyen_tinh_tot_hon": tt_tot,
                      "ket_luan": kl5, "dat": kl5 != "không kết luận"}}


# -------------------------------------------------------------------------------------------- biểu đồ
MAU_BAC = ["#888888", "#eda100", "#1baf7a", "#008300", "#e87ba4", "#6250d6", "#2a78d6", "#e34948", "#eb6834", "#1baf7a"]


def ve_thang(tong):
    plt = t11._plt()
    ten = [t for t, _, _ in BAC]
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    for i, t in enumerate(ten):
        m, (lo, hi) = tong[t]["mat_dv"]
        ax.plot([lo, hi], [i, i], color=MAU_BAC[i], lw=2.2)
        ax.plot([m], [i], "o", color=MAU_BAC[i], ms=7, mec="white", mew=1.5)
        ax.text(hi + 0.08, i, f"{m:.2f}", va="center", fontsize=8, color="#333333")
    ax.set_yticks(range(len(ten)), ten)
    ax.invert_yaxis()
    ax.set_xlabel("mất dịch vụ (%), 11 bộ gộp, CI 95% theo cửa sổ — sang trái là tốt hơn")
    ax.set_title("TN12 — thang bậc: mỗi bậc thêm một thành phần", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(RA / "hinh_12_thang.png", bbox_inches="tight")
    plt.close(fig)


def ve_quet(quet):
    if not quet:
        return
    plt = t11._plt()
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for S, mau in zip(QUET_S, ("#e87ba4", "#eda100", "#2a78d6")):
        xs = [quet[f"{nb}x{S}"]["ms"] for nb in QUET_BUOC]
        ys = [quet[f"{nb}x{S}"]["mat_dv"] for nb in QUET_BUOC]
        ax.plot(xs, ys, "-o", color=mau, lw=1.8, ms=6, mec="white", mew=1.2, label=f"{S} chuỗi")
        for nb, x_, y_ in zip(QUET_BUOC, xs, ys):
            ax.annotate(f"{nb} bước", (x_, y_), textcoords="offset points", xytext=(4, 4), fontsize=7, color="#555555")
    ax.set_xscale("log")
    ax.set_xlabel("thời gian quyết định, trung vị (ms, thang log)")
    ax.set_ylabel("mất dịch vụ (%), 11 bộ gộp")
    ax.set_title("TN12 — DM11 (seed 0): số bước khử nhiễu × số chuỗi", loc="left", fontsize=10)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RA / "hinh_12_quet.png", bbox_inches="tight")
    plt.close(fig)


def ve_hoc(hoc):
    if not hoc:
        return
    plt = t11._plt()
    ho = sorted({k.rsplit("_s", 1)[0] for k in hoc})
    fig, ax = plt.subplots(len(ho), 2, figsize=(10, 2.8 * len(ho)), squeeze=False)
    for i, h in enumerate(ho):
        for j, (k, x) in enumerate(sorted((k, x) for k, x in hoc.items() if k.rsplit("_s", 1)[0] == h)):
            e, mau = np.array(x["nk"]), t11.MAU_PHAN[j % 4]
            ax[i, 0].plot(e[:, 0], e[:, 1], color=mau, lw=1.5, label=f"{k} · train")
            ax[i, 0].plot(e[:, 0], e[:, 2], color=mau, lw=1.5, ls="--", label=f"{k} · val")
            ax[i, 1].plot(e[:, 0], e[:, 3], color=mau, lw=1.5, label=k)
            ax[i, 1].plot([x["epoch_tot"]], [x["J_val_tot"]], "o", color=mau, ms=6, mec="white", mew=1.5)
        ax[i, 0].set_title(f"{h} · loss (liền = train, gạch = val)", loc="left", fontsize=9)
        ax[i, 1].set_title(f"{h} · J val (chấm = checkpoint giữ lại)", loc="left", fontsize=9)
        ax[i, 0].legend(fontsize=6, ncol=2)
        ax[i, 1].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(RA / "hinh_12_hoc.png", bbox_inches="tight")
    plt.close(fig)


# -------------------------------------------------------------------------------------------- báo cáo
def viet_md(ra):
    ci = t11._ci
    L = ["# Thí nghiệm 12 — kết quả (sinh tự động bởi `scripts/tn12.py`)", "",
         "Tiêu chí: TIEU_CHI_12.md. Mất DV bằng %, hiệu bằng điểm %, CI 95% bootstrap theo cửa sổ; 4–5 luồng, trung bình "
         "3 seed (bậc có mô hình). Bậc 0–1 không học.", "", "## 1. Thang bậc — mất dịch vụ (%) theo bộ", "",
         "| Bậc | " + " | ".join(TEN_BO[b] for b in ra["bo"]) + " | Gộp 11 bộ | Quá tải (b) gộp |",
         "|---|" + "---:|" * (len(ra["bo"]) + 2)]
    for ten, _, _ in BAC:
        m, (lo, hi) = ra["tong"][ten]["mat_dv"]
        L.append(f"| {ten} | " + " | ".join(f"{x['trung_binh'][ten]:.2f}" for x in ra["bo"].values())
                 + f" | {m:.2f} [{lo:.2f}; {hi:.2f}] | {ra['tong'][ten]['qua_tai']:.2f} |")
    L += ["", "Hình: `hinh_12_thang.png`.", "", "## 2. Giả thuyết — hiệu theo bộ (điểm %, CI 95%)", "",
          "| Bộ | H12.1: đầy đủ − 1 bước | H12.2: đầy đủ − SM 32 mẫu | H12.3: không phối hợp − đầy đủ | "
          "H12.4: SM chống quá khớp − đầy đủ | H12.5: tuyến tính − cos² | 5 bước × 1 − 1 bước × 32 |",
          "|---|---|---|---|---|---|---|"]
    for b, x in ra["bo"].items():
        L.append(f"| {TEN_BO[b]} | {ci(x['H12.1'])} | {ci(x['H12.2'])} | {ci(x['H12.3'])} | {ci(x['H12.4'])} | "
                 f"{ci(x['H12.5'])} | {ci(x['lap_lai_buoc_vs_chuoi'])} |")
    gt = ra["gia_thuyet"]
    L += ["", f"Đạt = CI đúng chiều ở ≥ {gt['so_bo_can']}/{gt['so_bo']} bộ. H12.3 còn cần phần quá tải ≥ 60% phần chênh: "
          f"{ra['ti_le_qua_tai']:.0%}.", ""]
    for k in ("H12.1", "H12.2", "H12.3", "H12.4", "H12.5"):
        v = gt[k]
        L.append(f"- **{k}: {'ĐẠT' if v['dat'] else 'KHÔNG ĐẠT'}** — " + ", ".join(f"{kk} = {vv}" for kk, vv in v.items()
                                                                                 if kk != "dat"))
    L += ["", "## 3. So với Dijkstra phạt (điểm %, CI 95%)", "", "| Bậc | " + " | ".join(TEN_BO[b] for b in ra["bo"]) + " |",
          "|---|" + "---|" * len(ra["bo"])]
    for ten, _, _ in BAC[1:]:
        L.append(f"| {ten} | " + " | ".join(ci(x["vs_dphat"][ten]) for x in ra["bo"].values()) + " |")
    if ra["quet"]:
        L += ["", "## 4. Quét số bước × số chuỗi (DM11 seed 0, 11 bộ gộp)", "", "| Bước \\ chuỗi | "
              + " | ".join(f"{S} chuỗi: mất DV / trễ ms / thời gian ms" for S in QUET_S) + " |", "|---|" + "---|" * len(QUET_S)]
        for nb in QUET_BUOC:
            L.append(f"| {nb} | " + " | ".join(
                f"{ra['quet'][f'{nb}x{S}']['mat_dv']:.2f} / {ra['quet'][f'{nb}x{S}']['tre']:.1f} / "
                f"{ra['quet'][f'{nb}x{S}']['ms']:.0f}" for S in QUET_S) + " |")
        L += ["", "Hình: `hinh_12_quet.png`."]
    L += ["", "## 5. Học (mô hình mới)", "", "| Mô hình | Epoch chạy | Epoch tốt nhất | val / train | J val (thầy) | Cờ |",
          "|---|---:|---:|---:|---|---|"]
    for k, x in ra["hoc"].items():
        co = ("CHƯA TỚI " if x["chua_toi"] else "") + ("QUÁ KHỚP? " if x["ti_le_val_train"] > 1.25 else "")
        L.append(f"| {k} | {x['so_epoch']} | {x['epoch_tot']} | {x['ti_le_val_train']:.2f} | {x['J_val_tot']:.3f} "
                 f"({x['J_thay']:.3f}) | {co or 'ổn'} |")
    L += ["", f"Softmax chống quá khớp được chọn: {ra['sm12']}. Hình: `hinh_12_hoc.png`.", "",
          f"Kiểm lặp lại DM11 5 × 32 so với TN11: {ra['lap_lai']}."]
    (RA / "KET_QUA_12.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def lenh_phan_tich(a):
    T = _tai()
    if not T:
        raise SystemExit("chưa có kết quả chấm — chạy bước cham")
    bo, tong, ti_le = phan_tich_bo(T)
    hoc, sm12 = doc_hoc(), sm12_chon()
    ra = {"cau_hinh": CFG, "bo": bo, "tong": tong, "ti_le_qua_tai": ti_le, "quet": phan_tich_quet(T),
          "hoc": {k: {kk: vv for kk, vv in x.items() if kk != "nk"} for k, x in hoc.items()}, "sm12": sm12,
          "lap_lai": kiem_lap_lai(T)}
    ra["gia_thuyet"] = gia_thuyet(bo, ti_le, hoc, sm12)
    RA.mkdir(parents=True, exist_ok=True)
    (RA / "ket_qua_12.json").write_text(json.dumps(ra, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    ve_thang(tong)
    ve_quet(ra["quet"])
    ve_hoc(hoc)
    viet_md(ra)
    print("  " + "; ".join(f"{k}: {'ĐẠT' if v['dat'] else 'KHÔNG ĐẠT'}" for k, v in ra["gia_thuyet"].items()
                           if k.startswith("H")), flush=True)
    dong_goi("TN12", RA, M12, "tn12")


def dong_goi(ten, ra, mh, tien_to):
    """Một tệp để gửi lại: kết quả + bản ghi chấm (tệp đệm) + đường học + trọng số mô hình mới + log."""
    import zipfile
    tep = KQ / f"{ten}_ket_qua.zip"
    gom = [(f, Path(ra.name) / f.relative_to(ra)) for f in sorted(ra.rglob("*")) if f.is_file()]
    gom += [(f, Path(mh.name) / f.name) for f in sorted(mh.glob("*")) if f.suffix in (".json", ".pt", ".xong")]
    gom += [(f, Path("logs") / f.name) for f in sorted(LOG.glob(f"{tien_to}*.log"))]
    with zipfile.ZipFile(tep, "w", zipfile.ZIP_DEFLATED) as z:
        for f, t in gom:
            z.write(f, str(t))
    print(f"  đã gói {len(gom)} tệp → {tep} ({tep.stat().st_size / 1e6:.1f} MB) — gửi tệp này", flush=True)


# ============================================================================================ chạy
def lenh_toi_da(a):
    p = t11.tai_nguyen(a.ram, a.vram)
    print(f"máy: {p['cpu']} luồng CPU, ngân sách RAM {p['ram']} GB, GPU: {p['gpu']}", flush=True)
    print(f"→ học {p['hoc']} mô hình cùng lúc × {p['luong']} luồng ({p['thiet_bi']}); chấm {p['cham']} tiến trình", flush=True)
    a.song_song, a.luong_cpu, a.thiet_bi, a.tien_trinh = p["hoc"], p["luong"], p["thiet_bi"], p["cham"]
    for ten, f in (("huan-luyen", lenh_huan_luyen), ("cham", lenh_cham), ("phan-tich", lenh_phan_tich)):
        print(f"=== {ten} {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
        f(a)


def mo_log(ten, argv):
    if not os.environ.get("PBL4_KHONG_LOG"):
        LOG.mkdir(parents=True, exist_ok=True)
        f = open(LOG / f"{ten}.log", "a", encoding="utf-8", buffering=1)
        sys.stdout, sys.stderr = t11._Tee(sys.stdout, f), t11._Tee(sys.stderr, f)
        print(f"\n##### {time.strftime('%Y-%m-%d %H:%M:%S')}  python -m scripts.{ten} {' '.join(argv)}", flush=True)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("viec", choices=["huan-luyen", "cham", "phan-tich", "toi-da", "goi-tn11"])
    ap.add_argument("--tien-trinh", type=int, default=4)
    ap.add_argument("--song-song", type=int, default=2)
    ap.add_argument("--luong-cpu", type=int, default=4)
    ap.add_argument("--thiet-bi", default="auto", choices=["cpu", "auto", "cuda"])
    ap.add_argument("--ram", type=float, default=20.0, help="ngân sách RAM (GB) cho toi-da")
    ap.add_argument("--vram", type=float, default=8.0, help="ngân sách VRAM (GB) cho toi-da")
    ap.add_argument("--tn11", default=None, help="thư mục repo đã chạy TN11: chép mô hình + dữ liệu TN11 còn thiếu")
    a = ap.parse_args()
    mo_log("tn12", sys.argv[1:])
    if a.tn11:
        lay_tn11(a.tn11)
    {"huan-luyen": lenh_huan_luyen, "cham": lenh_cham, "phan-tich": lenh_phan_tich, "toi-da": lenh_toi_da,
     "goi-tn11": goi_tn11}[a.viec](a)
    print("xong", flush=True)


if __name__ == "__main__":
    main()
