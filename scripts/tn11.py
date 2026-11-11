"""
tn11.py — Thí nghiệm 11 (results/bao_da_tuyen/TIEU_CHI_11.md): DM bền dưới thời tiết cực đoan, quy trình như sản phẩm.

    python -m scripts.tn11 toi-da                 # chạy hết: ~20 GB RAM, GPU 8 GB (tự dò; không GPU thì dùng CPU)
    python -m scripts.tn11 toi-da --ram 24        # nới / siết ngân sách RAM (GB)

Từng bước (chạy lại an toàn — phần đã xong tự bỏ qua):
    sinh         sinh tập còn thiếu: x11 (5% cực đoan, có thầy), xc11 / xy11 (kiểm tra cực đoan mới), c7wh / y7wh (TN7a)…
    chuan-bi     làm sạch + kiểm tra rò rỉ (tn11/kiem_tra_du_lieu.json) → chia 4 phần THEO NHÓM bão / ngày (tn11/phan.json)
    huan-luyen   DM11 (A + x11), DM10c (chỉ A), SM11 (softmax, A + x11): 4 mô hình kiểm thử chéo + 3 mô hình cuối mỗi họ,
                 dừng sớm theo J val → mo_hinh11/
    cv           kiểm thử chéo: mỗi tình huống học chấm bằng mô hình KHÔNG học nó (và một mô hình ĐÃ học nó, đo học thuộc)
    cham         bộ kiểm tra (4–5 luồng: 3 seed; 10 / 20 luồng: seed 0): Dijkstra phạt, Dijkstra (Yen), SM11, DM10c,
                 DM11, Lai(·), thời gian
    phan-tich    → tn11/KET_QUA_11.md, ket_qua_11.json, hinh_11_*.png (đường học, kiểm thử chéo, kiểm tra, thời gian…)
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

GOC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GOC))
KQ = Path(os.environ.get("PBL4_KQ", GOC / "results" / "bao_da_tuyen"))
RA, M11 = KQ / "tn11", KQ / "mo_hinh11"
DL = Path(os.environ.get("PBL4_DL", GOC / "data" / "bao_da_tuyen"))
LOG = Path(os.environ.get("PBL4_LOG", GOC / "logs"))

TAP_HOC = {"train10_k4_5": "train10", "train6_10_k4_5": "train6_10", "x11_k4_5": "x11"}
TAP_VAL = {"val_k4_5": "val", "val6_k4_5": "val6"}
BO_45 = ("c6", "yagi5", "c6_b", "yagi5_b", "ngay9", "nam9", "bao9", "c7wh", "y7wh", "xc11", "xy11")
BO_LON = ("c6", "yagi5", "c6_b", "yagi5_b", "c7wh", "y7wh")
CUC_DOAN = ("c7wh", "y7wh", "xc11", "xy11")
THUONG = tuple(b for b in BO_45 if b not in CUC_DOAN)
TEN_BO = {"c6": "TG A · C ngày giông", "yagi5": "TG A · Yagi", "c6_b": "TG B · C ngày giông", "yagi5_b": "TG B · Yagi",
          "ngay9": "ngày giông mới", "nam9": "ngày xấu 2017–21", "bao9": "đợt mạnh 2017–21",
          "c7wh": "C cực đoan WH (TN7a)", "y7wh": "Yagi cực đoan WH (TN7a)",
          "xc11": "C cực đoan ×1,3–1,5", "xy11": "Yagi cực đoan ×1,3–1,5"}
HO = {"dm11": {"kieu": "dm", "tn": 15}, "dm10c": {"kieu": "dm", "tn": 16}, "sm11": {"kieu": "softmax", "tn": 15}}
SO_PHAN, SEEDS, SO_BUOC, TAU, ALPHA = 4, (0, 1, 2), 5, 1.1, 0.025
DO = ("mat_dv_%", "J", "khong_tuyen_song_%", "du_phong_qua_tai_%", "tre_chinh_ms")      # thứ tự cột của mọi thước đo
CFG = {"epochs": 120, "H": 128, "dung_som": 20, "S": 32}


def thu_muc_bo(bo, K):
    return f"{bo}_k{K.replace('-', '_')}"


def py(*a):
    return [sys.executable, "-W", "ignore", *a]


def chay(lenh):
    """Chạy tiến trình con; mọi dòng đầu ra đi qua print → vào cả màn hình lẫn logs/tn11.log."""
    env = {**os.environ, "PBL4_KHONG_LOG": "1", "PYTHONIOENCODING": "utf-8"}
    p = subprocess.Popen(lenh, cwd=GOC, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         encoding="utf-8", errors="replace", bufsize=1)
    for dong in p.stdout:
        print(dong.rstrip("\n"), flush=True)
    if p.wait():
        raise SystemExit(f"lỗi (mã {p.returncode}): {' '.join(lenh[3:])}")


class _Tee:
    """Ghi đồng thời ra màn hình và tệp log."""

    def __init__(self, *dich):
        self.dich = dich

    def write(self, x):
        for d in self.dich:
            d.write(x)
        return len(x)

    def flush(self):
        for d in self.dich:
            d.flush()


# ============================================================================================ sinh dữ liệu
def lenh_sinh(a):
    can = [(t, "4-5", True) for t in (*TAP_HOC.values(), *TAP_VAL.values())]
    can += [(b, "4-5", False) for b in BO_45] + [(b, K, False) for b in BO_LON for K in ("10", "20")]
    for tap, K, thay in can:
        d = DL / thu_muc_bo(tap, K)
        if (d / ".xong").exists():
            continue
        lenh = py("-m", "src.nhieu_luong.tao_du_lieu_dt", "--tap", tap, "--K", K, "--tien-trinh", str(a.tien_trinh))
        print(f"  sinh {d.name}{'' if thay else ' (không thầy)'}", flush=True)
        chay(lenh + ([] if thay else ["--khong-thay"]))


# ============================================================================================ làm sạch + chia phần
def lenh_chuan_bi(a):
    from datetime import date

    from src.nhieu_luong import tao_du_lieu_rr as td
    from src.nhieu_luong.kiem_tra_du_lieu import bao_cao_thu_muc
    from src.nhieu_luong.tao_du_lieu_dt import NAM_9, cac_cua_so_tap, cua_so_x11, he_so_11, thoi_tiet_9
    td._khoi_tao()
    RA.mkdir(parents=True, exist_ok=True)
    # 1. làm sạch / thống kê
    tm = {d: True for d in (*TAP_HOC, *TAP_VAL)}
    tm |= {thu_muc_bo(b, "4-5"): False for b in BO_45} | {thu_muc_bo(b, K): False for b in BO_LON for K in ("10", "20")}
    bc = {}
    for d, nhan in tm.items():
        if (DL / d).exists():
            bc[d] = bao_cao_thu_muc(DL / d, can_nhan=nhan)
            print(f"  {d}: {bc[d]['so_ban_ghi']} bản ghi, lỗi {bc[d]['so_ban_ghi_loi']}, trùng {bc[d]['so_trung']}", flush=True)
    # 2. rò rỉ theo ngày: cửa sổ val / kiểm tra cùng ngày hoặc kề ngày (±1) với cửa sổ học → báo, và phân tích độ nhạy
    def ngay(tap):
        tt = thoi_tiet_9() if tap in NAM_9 else td._tt
        return [date.fromisoformat(str(tt.time_utc[h0])[:10]) for _, h0, _ in cac_cua_so_tap(tap)]
    hoc = set(ngay("train")) | set(ngay("train6"))
    ro_ri = {}
    for tap in ("val", "val6", "c6", "yagi5", "ngay9", "nam9", "bao9"):
        ng = ngay(tap)
        kb = [i for i, x in enumerate(ng) if any(abs((x - h).days) <= 1 for h in hoc)]
        ro_ri[tap] = {"kb": kb, "ngay": [str(ng[i]) for i in kb],
                      "trung_ngay": [str(ng[i]) for i in kb if ng[i] in hoc]}
        print(f"  rò rỉ {tap}: " + ("KHÔNG" if not kb else f"{len(kb)} cửa sổ kề ngày học: {', '.join(ro_ri[tap]['ngay'])}"
                                     f" (trùng đúng ngày: {len(ro_ri[tap]['trung_ngay'])})"), flush=True)
    # 3. cửa sổ cực đoan và hệ số
    x11 = [{"kb": kb, "ten": ten, "h0": int(h0), "nguon": nguon, "do_du_doi": dd, "k": he_so_11("x11", kb)}
           for kb, (ten, h0, _, nguon, dd) in enumerate(cua_so_x11())]
    k_test = {t: [he_so_11(t, kb) for kb in range(len(cac_cua_so_tap(t)))] for t in ("xc11", "xy11")}
    # 4. chia phần theo NHÓM (cơn bão ở train, ngày ở train6), tham lam cân bằng từng nửa; x11 theo cửa sổ nguồn
    phan, nhom_cua = {}, {}
    for d, tap in (("train10_k4_5", "train"), ("train6_10_k4_5", "train6")):
        nhom = {}
        for kb, (ten, _, _) in enumerate(cac_cua_so_tap(tap)):
            nhom.setdefault(ten if tap == "train" else ten[:14], []).append(kb)
        co, phan[d] = [0] * SO_PHAN, {}
        for ten in sorted(nhom, key=lambda n: (-len(nhom[n]), n)):
            f = int(np.argmin(co))
            co[f] += len(nhom[ten])
            for kb in nhom[ten]:
                phan[d][str(kb)] = f
                nhom_cua[(d, kb)] = ten
        print(f"  phần {d}: số cửa sổ mỗi phần {co}, {len(nhom)} nhóm", flush=True)
    nguon_d = {"train": "train10_k4_5", "train6": "train6_10_k4_5"}
    phan["x11_k4_5"] = {str(x["kb"]): phan[nguon_d[x["nguon"][0]]][str(x["nguon"][1])] for x in x11}
    phan["_nhom"] = {d: {kb: nhom_cua[(d, int(kb))] for kb in phan[d]} for d in nguon_d.values()}
    (RA / "phan.json").write_text(json.dumps(phan, ensure_ascii=False, indent=1), encoding="utf-8")
    (RA / "kiem_tra_du_lieu.json").write_text(json.dumps({"thu_muc": bc, "ro_ri": ro_ri, "x11": x11, "k_kiem_tra": k_test},
                                                         ensure_ascii=False, indent=1, default=str), encoding="utf-8")


# ============================================================================================ huấn luyện
def ds_mo_hinh():
    viec = [(f"{h}_p{f}", h, f, f) for h in HO for f in range(SO_PHAN)]          # kiểm thử chéo trước
    return viec + [(f"{h}_s{s}", h, s, None) for h in HO for s in SEEDS]       # rồi mô hình cuối


def lenh_huan_luyen(a, chi=None):
    """Hàng đợi: luôn giữ a.song_song mô hình đang chạy. chi: chỉ các tên này."""
    viec = [v for v in ds_mo_hinh() if (chi is None or v[0] in chi) and not (M11 / f"{v[0]}.xong").exists()]
    LOG.mkdir(parents=True, exist_ok=True)
    chay, loi = [], []
    while viec or chay:
        while viec and len(chay) < a.song_song and not loi:
            ten, h, seed, phan = viec.pop(0)
            lenh = py("-m", "src.nhieu_luong.huan_luyen_dt", HO[h]["kieu"], "--thi-nghiem", str(HO[h]["tn"]), "--ten", ten,
                      "--them", "--seed", str(seed), "--epochs", str(CFG["epochs"]), "--H", str(CFG["H"]),
                      "--dung-som", str(CFG["dung_som"]), "--luong-cpu", str(a.luong_cpu), "--thiet-bi", a.thiet_bi,
                      *([] if phan is None else ["--bo-phan", str(phan)]))
            print(f"  {time.strftime('%H:%M:%S')} huấn luyện {ten} → {LOG.name}/tn11_{ten}.log", flush=True)
            chay.append((ten, subprocess.Popen(lenh, cwd=GOC, stderr=subprocess.STDOUT,
                                               stdout=open(LOG / f"tn11_{ten}.log", "w", encoding="utf-8"))))
        if not viec and not chay:
            break
        if loi and not chay:
            break
        time.sleep(5)
        for ten, p in list(chay):
            if p.poll() is not None:
                chay.remove((ten, p))
                if p.returncode:
                    loi.append(ten)
                    print(f"  LỖI {ten} — xem {LOG.name}/tn11_{ten}.log", flush=True)
                else:
                    print(f"  {time.strftime('%H:%M:%S')} xong {ten}", flush=True)
        if loi and not chay:
            break
    if loi:
        raise SystemExit(f"huấn luyện lỗi: {loi} — chạy lại đúng lệnh (mô hình xong rồi được giữ)")


# ============================================================================================ chấm (dùng chung)
_NET = {}


def _net(mh, ten):
    from src.nhieu_luong import mo_hinh_dt as md
    if (mh, ten) not in _NET:
        _NET[(mh, ten)] = md.nap(ten, Path(mh))
    return _NET[(mh, ten)]


def _do(gon, tgw, a):
    c = tgw.cham(gon.tu_lua_chon(a))
    return [float(c[k]) for k in DO]


def _chon(net, gon, seed, S):
    import torch

    from src.nhieu_luong import mo_hinh_dt as md
    s = time.perf_counter()
    if net.__class__.__name__ == "MangDT" and getattr(net, "_kieu", "dm") == "dm":
        a = md.chon_dm(net, gon, S, SO_BUOC, torch.Generator().manual_seed(seed))
    else:
        a = md.chon_softmax(net, gon)
    return a, (time.perf_counter() - s) * 1e3


def _lai(gon, r, a_dm):
    from src.nhieu_luong import da_tuyen as dt
    s = time.perf_counter()
    a = dt.tkcb(dt.TheGioi.tu_bang(gon, np.random.default_rng([int(r["kb"]), int(r["t"]), 4])), a_dm)[0]
    return a, (time.perf_counter() - s) * 1e3


def _co_ban(r):
    from src.nhieu_luong.kiem_tra_du_lieu import dai_tai
    ra = {"kb": int(r["kb"]), "t": int(r["t"]), "dai": dai_tai(r["muc"])}
    for k in ("dphat", "yen", "thay"):
        if k in r.get("ens", {}):
            ra[k] = [float(r["ens"][k][m]) for m in DO]
    ra["ms_dphat"] = float(r.get("tg_ms", {}).get("dphat", np.nan))
    return ra


def _nap_kieu(mh, ten, kieu):
    n = _net(mh, ten)
    n._kieu = kieu
    return n


def _cv_tep(job):
    """Tập học: mô hình phần KHÔNG học tình huống (ngoài phần) + mô hình phần ĐÃ học nó (trong phần) — DM11 / SM11."""
    import torch

    from src.nhieu_luong.cham_lai_dt import giong_ban_ghi, rho_hat
    from src.nhieu_luong.cham_mo_hinh_dt import _tg
    tep, d, mh, phan_tep, S = job
    torch.set_num_threads(1)
    bang = json.loads(Path(phan_tep).read_text(encoding="utf-8"))[d]
    ra = []
    for r in pickle.load(open(tep, "rb")):
        gon, f = r["gon"], int(bang[str(int(r["kb"]))])
        tgw, sd, x = _tg(gon, r["cham"]), giong_ban_ghi(r), _co_ban(r)
        x["rho"] = rho_hat(gon)
        for h in HO:
            for nhan, ff in ((h, f), (f"{h}_da_hoc", (f + 1) % SO_PHAN)):
                if nhan.endswith("_da_hoc") and h == "dm10c":
                    continue
                a, _ = _chon(_nap_kieu(mh, f"{h}_p{ff}", HO[h]["kieu"]), gon, sd, S)
                x[nhan] = _do(gon, tgw, a)
                if HO[h]["kieu"] == "dm" and not nhan.endswith("_da_hoc"):
                    x[f"{h}_lai"] = _do(gon, tgw, _lai(gon, r, a)[0]) if x["rho"] >= TAU else x[nhan]
        ra.append(x)
    return tep, ra


def _cham_tep(job):
    """Bộ kiểm tra: mọi phương pháp trên thế giới chấm đã lưu; thời gian từng phương pháp; TKCB 10 lần mỗi 4 mốc (C5)."""
    import torch

    from src.nhieu_luong.cham_lai_dt import giong_ban_ghi, rho_hat
    from src.nhieu_luong.cham_mo_hinh_dt import _tg
    from src.nhieu_luong.tao_du_lieu_dt import tkcb10
    tep, mh, seeds, S = job
    torch.set_num_threads(1)
    recs = pickle.load(open(tep, "rb"))
    cac_t = sorted({int(r["t"]) for r in recs})
    ra = []
    for r in recs:
        gon = r["gon"]
        tgw, sd, x = _tg(gon, r["cham"]), giong_ban_ghi(r), _co_ban(r)
        x["rho"] = rho_hat(gon)
        for s in seeds:
            for h in HO:
                a, ms = _chon(_nap_kieu(mh, f"{h}_s{s}", HO[h]["kieu"]), gon, sd, S)
                x[f"{h}_s{s}"], x[f"ms_{h}_s{s}"] = _do(gon, tgw, a), ms
                if HO[h]["kieu"] == "dm":
                    if x["rho"] >= TAU:
                        al, ms2 = _lai(gon, r, a)
                        x[f"{h}_lai_s{s}"], x[f"ms_{h}_lai_s{s}"] = _do(gon, tgw, al), ms + ms2
                    else:
                        x[f"{h}_lai_s{s}"], x[f"ms_{h}_lai_s{s}"] = x[f"{h}_s{s}"], ms
        x["ms_tkcb10"] = (tkcb10(gon, np.random.default_rng([int(r["kb"]), int(r["t"]), 3]))[1]
                          if cac_t.index(int(r["t"])) % 4 == 0 else None)
        ra.append(x)
    return tep, ra


def _chay_tep(ham, viec, tt, thu_dem):
    """Chạy ham trên từng tệp, lưu đệm từng tệp (chạy lại không làm lại)."""
    thu_dem.mkdir(parents=True, exist_ok=True)
    con = [v for v in viec if not (thu_dem / v[0]).exists()]
    if con:
        s = time.time()
        with Pool(tt) as pool:
            for k, (tep, ra) in enumerate(pool.imap_unordered(ham, [v[1] for v in con])):
                ten = next(v[0] for v in con if v[1][0] == tep)
                (thu_dem / ten).write_text(json.dumps(ra), encoding="utf-8")
                if k % 10 == 0 or k == len(con) - 1:
                    print(f"    {k + 1}/{len(con)} tệp ({time.time() - s:.0f} s)", flush=True)
    return {v[0]: json.loads((thu_dem / v[0]).read_text(encoding="utf-8")) for v in viec}


def lenh_cv(a):
    thieu = [v[0] for v in ds_mo_hinh() if v[3] is not None and not (M11 / f"{v[0]}.xong").exists()]
    if thieu:
        raise SystemExit(f"chưa xong mô hình kiểm thử chéo: {thieu}")
    viec = [(f"{d}__{Path(t).stem}.json", (t, d, str(M11), str(RA / "phan.json"), CFG["S"]))
            for d in TAP_HOC for t in sorted(glob.glob(str(DL / d / "kb_*.pkl")))]
    print(f"  kiểm thử chéo: {len(viec)} tệp", flush=True)
    _chay_tep(_cv_tep, viec, a.tien_trinh, RA / "cv_tep")


def lenh_cham(a):
    thieu = [v[0] for v in ds_mo_hinh() if v[3] is None and not (M11 / f"{v[0]}.xong").exists()]
    if thieu:
        raise SystemExit(f"chưa xong mô hình cuối: {thieu}")
    viec = []
    for b, K in [(b, "4-5") for b in BO_45] + [(b, K) for b in BO_LON for K in ("10", "20")]:
        seeds = SEEDS if K == "4-5" else (0,)
        viec += [(f"{thu_muc_bo(b, K)}__{Path(t).stem}.json", (t, str(M11), seeds, CFG["S"]))
                 for t in sorted(glob.glob(str(DL / thu_muc_bo(b, K) / "kb_*.pkl")))]
    print(f"  chấm bộ kiểm tra: {len(viec)} tệp", flush=True)
    _chay_tep(_cham_tep, viec, a.tien_trinh, RA / "cham_tep")


# ============================================================================================ phân tích
def _bs(v, kb, seed=0):
    from scripts.tn10 import bootstrap
    return bootstrap(np.asarray(v, float), np.asarray(kb), seed)


def _gop(thu_dem, tien_to):
    """Gộp các tệp đệm có tên bắt đầu bằng tien_to → danh sách bản ghi (theo thứ tự tệp)."""
    ra = []
    for f in sorted(glob.glob(str(thu_dem / f"{tien_to}__*.json"))):
        ra += json.loads(Path(f).read_text(encoding="utf-8"))
    return ra


def _cot(rs, khoa, j=0):
    return np.array([r[khoa][j] for r in rs], float)


def _ho(rs, h, seeds, j=0):
    """Trung bình theo seed của họ h (khoá h_s{seed}) cho từng bản ghi."""
    return np.mean([[r[f"{h}_s{s}"][j] for r in rs] for s in seeds], 0)


def doc_hoc():
    """Đường học + cờ học chưa tới / học thuộc của mọi mô hình."""
    ra = {}
    for ten, h, seed, phan in ds_mo_hinh():
        t = M11 / f"{ten}_nhat_ky.json"
        if not t.exists():
            continue
        j = json.loads(t.read_text(encoding="utf-8"))
        nk = j["nhat_ky"]
        b = min(nk, key=lambda r: r["J_val"])
        ra[ten] = {"ho": h, "phan": phan, "so_epoch": len(nk), "epoch_tot": b["epoch"], "dung_som_o": j.get("dung_som_o"),
                   "loss_train_tot": b["loss_train"], "loss_val_tot": b["loss_val"],
                   "ti_le_val_train": b["loss_val"] / b["loss_train"], "J_val_tot": b["J_val"], "J_thay": b["J_thay_val"],
                   "chua_toi": len(nk) >= CFG["epochs"] and b["epoch"] >= 0.9 * CFG["epochs"],
                   "nk": [[r["epoch"], r["loss_train"], r["loss_val"], r["J_val"]] for r in nk]}
    return ra


def phan_tich_cv():
    rs = []
    for i, d in enumerate(TAP_HOC):
        for r in _gop(RA / "cv_tep", d):
            r["nhom_tap"], r["cum"] = d, i * 10_000 + r["kb"]
            rs.append(r)
    if not rs:
        return None, None
    cum = np.array([r["cum"] for r in rs])
    phan = {"train (7 cơn bão)": "train10_k4_5", "train6 (60 ngày)": "train6_10_k4_5", "x11 (cực đoan)": "x11_k4_5"}
    pp = ["dphat", "yen", "thay", "sm11", "dm10c", "dm10c_lai", "dm11", "dm11_lai"]
    ra = {}
    for ten, d in {**phan, "A (thường)": None}.items():
        m = np.array([(r["nhom_tap"] == d) if d else (r["nhom_tap"] != "x11_k4_5") for r in rs])
        sub = [r for r, x in zip(rs, m) if x]
        x = {p: float(_cot(sub, p).mean()) for p in pp}
        x["n"] = len(sub)
        x["dm11_tru_dphat"] = _bs(_cot(sub, "dm11_lai") - _cot(sub, "dphat"), cum[m])
        x["dm11_tru_dm10c"] = _bs(_cot(sub, "dm11_lai") - _cot(sub, "dm10c_lai"), cum[m])
        x["dm11_tru_sm11"] = _bs(_cot(sub, "dm11") - _cot(sub, "sm11"), cum[m])
        for h in ("dm11", "sm11"):
            x[f"{h}_da_hoc"] = float(_cot(sub, f"{h}_da_hoc").mean())
            x[f"{h}_khoang_hoc_thuoc"] = _bs(_cot(sub, h) - _cot(sub, f"{h}_da_hoc"), cum[m])
        x["G_dm11"] = float((_cot(sub, "dm11") - _cot(sub, "thay")).mean())
        ra[ten] = x
    return ra, rs


def _tai(thu_dem=None):
    """{(bộ, K): [bản ghi]} của bộ kiểm tra."""
    ra = {}
    for b, K in [(b, "4-5") for b in BO_45] + [(b, K) for b in BO_LON for K in ("10", "20")]:
        rs = _gop(RA / "cham_tep", thu_muc_bo(b, K))
        if rs:
            ra[(b, K)] = rs
    return ra


GOC_BO = {"c6": "c6", "c6_b": "c6", "c7wh": "c6", "xc11": "c6", "yagi5": "yagi5", "yagi5_b": "yagi5", "y7wh": "yagi5",
          "xy11": "yagi5", "ngay9": "ngay9", "nam9": "nam9", "bao9": "bao9"}       # bộ → bộ gốc có cùng thứ tự cửa sổ


def phan_tich_kiem_tra(T, ro_ri=None):
    from scripts.kiem_dinh_boi_9 import bh, holm, p_mot_phia
    bo45 = {}
    for b in BO_45:
        rs = T.get((b, "4-5"))
        if not rs:
            continue
        kb = np.array([r["kb"] for r in rs])
        dp = _cot(rs, "dphat")
        f = {"Dijkstra phạt": dp, "Dijkstra (Yen)": _cot(rs, "yen"), "SM11": _ho(rs, "sm11", SEEDS),
             "Lai(DM10c)": _ho(rs, "dm10c_lai", SEEDS), "Lai(DM11)": _ho(rs, "dm11_lai", SEEDS),
             "DM11 thuần": _ho(rs, "dm11", SEEDS)}
        if all("thay" in r for r in rs):
            f["Thầy"] = _cot(rs, "thay")
        x = {"n": len(rs), "trung_binh": {k: float(v.mean()) for k, v in f.items()},
             "hieu_vs_dphat": {k: _bs(v - dp, kb) for k, v in f.items() if k != "Dijkstra phạt"},
             "dm11_tru_dm10c": _bs(f["Lai(DM11)"] - f["Lai(DM10c)"], kb),
             "dm11_tru_sm11": _bs(f["DM11 thuần"] - f["SM11"], kb),
             "a_b": {k: [float(np.mean(_ho(rs, h, SEEDS, 2) if h else _cot(rs, "dphat", 2))),
                         float(np.mean(_ho(rs, h, SEEDS, 3) if h else _cot(rs, "dphat", 3)))]
                     for k, h in (("Dijkstra phạt", None), ("Lai(DM10c)", "dm10c_lai"), ("Lai(DM11)", "dm11_lai"),
                                  ("SM11", "sm11"))}}
        bo_kb = sorted(set(kb.tolist()) & set((ro_ri or {}).get(GOC_BO[b], {}).get("kb", [])))
        if bo_kb and len(set(kb.tolist())) > len(bo_kb):   # độ nhạy: bỏ cửa sổ kề ngày học
            m = ~np.isin(kb, bo_kb)
            x["bo_ro_ri"] = {"so_cua_so_bo": len(bo_kb),
                             "Lai(DM11) − Dijkstra phạt": _bs(f["Lai(DM11)"][m] - dp[m], kb[m]),
                             "Lai(DM11) − Lai(DM10c)": _bs(f["Lai(DM11)"][m] - f["Lai(DM10c)"][m], kb[m]),
                             "DM11 − SM11": _bs(f["DM11 thuần"][m] - f["SM11"][m], kb[m])}
        bo45[b] = x
    # ô tiêu chí so với Dijkstra phạt (mô hình seed 0)
    rng = np.random.default_rng(0)
    o = {}
    for ten, khoa in (("Lai(DM11)", "dm11_lai_s0"), ("Lai(DM10c)", "dm10c_lai_s0"), ("SM11", "sm11_s0")):
        cac = []
        for (b, K), rs in T.items():
            kb, dai = np.array([r["kb"] for r in rs]), np.array([r["dai"] for r in rs])
            ms_k = khoa.replace("_s0", "") if False else khoa
            for j in range(3):
                m = dai == j
                if m.sum() < 10:
                    continue
                dJ = _cot([r for r, x in zip(rs, m) if x], khoa, 1) - _cot([r for r, x in zip(rs, m) if x], "dphat", 1)
                dL = _cot([r for r, x in zip(rs, m) if x], khoa, 0) - _cot([r for r, x in zip(rs, m) if x], "dphat", 0)
                pJ, pL = p_mot_phia(dJ, kb[m], rng)[0], p_mot_phia(dL, kb[m], rng)[0]
                c5 = True
                if K != "4-5":
                    t10 = [r["ms_tkcb10"] for r, x in zip(rs, m) if x and r["ms_tkcb10"] is not None]
                    tl = [r[f"ms_{ms_k}"] for r, x in zip(rs, m) if x and r["ms_tkcb10"] is not None]
                    c5 = bool(t10) and float(np.median(tl)) < float(np.median(t10))
                cac.append({"bo": b, "K": K, "dai": j, "n": int(m.sum()), "C2": _bs(dJ, kb[m]), "C4": _bs(dL, kb[m]),
                            "p": max(pJ, pL), "dat_C5": c5})
        p = np.array([c["p"] for c in cac], float)
        c5 = np.array([c["dat_C5"] for c in cac], bool)
        for c, th, hm, b_ in zip(cac, (p <= ALPHA) & c5, holm(p) & c5, bh(p) & c5):
            c.update(thang=bool(th), thang_holm=bool(hm), thang_bh=bool(b_), te_hon=c["C4"][1][0] > 0)
        o[ten] = cac
    tom = {ten: {"so_o": len(c), "thang": sum(x["thang"] for x in c), "thang_holm": sum(x["thang_holm"] for x in c),
                 "te_hon": sum(x["te_hon"] for x in c),
                 "holm_cuc_doan": sum(x["thang_holm"] for x in c if x["bo"] in CUC_DOAN),
                 "holm_thuong": sum(x["thang_holm"] for x in c if x["bo"] not in CUC_DOAN)} for ten, c in o.items()}
    # thời gian (trung vị ms) theo quy mô
    tg = {}
    for K in ("4-5", "10", "20"):
        rs = [r for (b, KK), v in T.items() if KK == K for r in v]
        if not rs:
            continue
        tg[K] = {"Dijkstra phạt": float(np.nanmedian([r["ms_dphat"] for r in rs])),
                 "SM11": float(np.median([r["ms_sm11_s0"] for r in rs])),
                 "DM11 thuần": float(np.median([r["ms_dm11_s0"] for r in rs])),
                 "Lai(DM11)": float(np.median([r["ms_dm11_lai_s0"] for r in rs])),
                 "TKCB 10 lần": float(np.median([r["ms_tkcb10"] for r in rs if r["ms_tkcb10"] is not None]))}
    return bo45, o, tom, tg


def _doi_khi_bo_ro_ri(bo45, ds, khoa, khoa_bo, dieu_kien):
    """Bộ nào đổi kết luận khi bỏ cửa sổ kề ngày học."""
    return [b for b in ds if "bo_ro_ri" in bo45[b] and dieu_kien(bo45[b][khoa] if isinstance(khoa, str) else khoa(bo45[b]))
            != dieu_kien(bo45[b]["bo_ro_ri"][khoa_bo])]


def gia_thuyet(bo45, tom, hoc, cv):
    sat = lambda ci, ng=0.0: (ng - 0.1 < ci[1][1] < ng) or (ng < ci[1][0] < ng + 0.1)
    cd = [b for b in CUC_DOAN if b in bo45]
    th = [b for b in THUONG if b in bo45]
    giam = [b for b in cd if bo45[b]["dm11_tru_dm10c"][1][1] < 0]
    hai = [b for b in th if bo45[b]["dm11_tru_dm10c"][1][0] > 0]
    h1 = {"giam_cuc_doan": giam, "hai_thuong": hai, "dat": len(giam) >= 2 and not hai,
          "sat_nguong": [b for b in cd + th if sat(bo45[b]["dm11_tru_dm10c"])],
          "doi_khi_bo_ro_ri": _doi_khi_bo_ro_ri(bo45, cd, "dm11_tru_dm10c", "Lai(DM11) − Lai(DM10c)",
                                                lambda ci: ci[1][1] < 0)}
    hon_sm = [b for b in bo45 if bo45[b]["dm11_tru_sm11"][1][1] < 0]
    h2 = {"so_bo_hon": len(hon_sm), "so_bo": len(bo45), "dat": len(hon_sm) >= 8}
    hon_dp = [b for b in cd if bo45[b]["hieu_vs_dphat"]["Lai(DM11)"][1][1] < 0]
    t11, t10 = tom.get("Lai(DM11)", {}), tom.get("Lai(DM10c)", {})
    h3 = {"hon_dphat_cuc_doan": hon_dp,
          "doi_khi_bo_ro_ri": _doi_khi_bo_ro_ri(bo45, cd, lambda x: x["hieu_vs_dphat"]["Lai(DM11)"],
                                                "Lai(DM11) − Dijkstra phạt", lambda ci: ci[1][1] < 0), "te_hon": t11.get("te_hon"), "holm_cuc_doan_DM11": t11.get("holm_cuc_doan"),
          "holm_cuc_doan_DM10c": t10.get("holm_cuc_doan"),
          "dat": len(hon_dp) == len(cd) == 4 and t11.get("te_hon") == 0
          and t11.get("holm_cuc_doan", 0) >= t10.get("holm_cuc_doan", 0)}
    cvm = [x for x in hoc.values() if x["phan"] is not None]
    chua = [k for k, x in hoc.items() if x["phan"] is not None and x["chua_toi"]]
    thuoc_loss = [k for k, x in hoc.items() if x["phan"] is not None and x["ti_le_val_train"] > 1.25]
    thuoc_cv = []
    if cv:
        for p_ in ("A (thường)", "x11 (cực đoan)"):
            for h in ("dm11", "sm11"):
                if cv[p_][f"{h}_khoang_hoc_thuoc"][0] > 1.0:
                    thuoc_cv.append(f"{h} / {p_}")
    h4 = {"so_mo_hinh": len(cvm), "chua_toi": chua, "hoc_thuoc_loss": thuoc_loss, "hoc_thuoc_cv": thuoc_cv,
          "dat": bool(cvm) and not chua and not thuoc_loss and not thuoc_cv}
    return {"H11.1": h1, "H11.2": h2, "H11.3": h3, "H11.4": h4}


# -------------------------------------------------------------------------------------------- biểu đồ
MAU = {"Lai(DM11)": "#2a78d6", "Dijkstra phạt": "#eb6834", "SM11": "#1baf7a", "Lai(DM10c)": "#eda100",
       "DM11 thuần": "#e87ba4", "Dijkstra (Yen)": "#008300", "Thầy": "#6250d6", "TKCB 10 lần": "#e34948"}
MAU_PHAN = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
TEN_HO = {"dm11": "DM11 (A + 5% cực đoan)", "dm10c": "DM10c (chỉ A)", "sm11": "SM11 softmax (A + 5% cực đoan)"}


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#888888", "axes.labelcolor": "#333333",
                         "xtick.color": "#555555", "ytick.color": "#555555", "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.8,
                         "legend.frameon": False, "figure.dpi": 150})
    return plt


def ve_hoc(hoc, cuoi=False):
    plt = _plt()
    from matplotlib.ticker import MaxNLocator
    fig, ax = plt.subplots(len(HO), 2, figsize=(11, 3.1 * len(HO)), squeeze=False)
    for i, h in enumerate(HO):
        cac = [(k, x) for k, x in hoc.items() if x["ho"] == h and ((x["phan"] is None) == cuoi)]
        for j, (k, x) in enumerate(sorted(cac)):
            e = np.array(x["nk"])
            mau = MAU_PHAN[j % 4]
            nhan = f"seed {k.split('_s')[-1]}" if cuoi else f"phần {x['phan']}"
            ax[i, 0].plot(e[:, 0], e[:, 1], color=mau, lw=1.6, label=f"{nhan} · train")
            ax[i, 0].plot(e[:, 0], e[:, 2], color=mau, lw=1.6, ls="--", label=f"{nhan} · val")
            ax[i, 1].plot(e[:, 0], e[:, 3], color=mau, lw=1.6, label=nhan)
            ax[i, 1].plot([x["epoch_tot"]], [x["J_val_tot"]], "o", color=mau, ms=6, mec="white", mew=1.5)
            ax[i, 0].axvline(x["epoch_tot"], color=mau, lw=0.8, alpha=0.5)
        if cac:
            ax[i, 1].axhline(cac[0][1]["J_thay"], color=MAU["Thầy"], lw=1.2, ls=":", label="thầy")
        ax[i, 0].set_title(f"{TEN_HO[h]} · loss", loc="left", fontsize=9)
        ax[i, 1].set_title(f"{TEN_HO[h]} · J val", loc="left", fontsize=9)
        ax[i, 0].set_ylabel("loss")
        ax[i, 1].set_ylabel("J val (thấp hơn là tốt)")
        for c in (0, 1):
            ax[i, c].set_xlabel("epoch")
            ax[i, c].xaxis.set_major_locator(MaxNLocator(integer=True))
        ax[i, 0].legend(fontsize=7, ncol=2)
        ax[i, 1].legend(fontsize=7, ncol=3)
    fig.suptitle(f"TN11 — đường học {'mô hình cuối (toàn bộ dữ liệu)' if cuoi else 'kiểm thử chéo 4 phần theo nhóm'}. "
                 "Loss: liền = train, gạch = val, vạch dọc = epoch giữ lại. J val: chấm = checkpoint giữ lại, "
                 "chấm chấm = thầy.", x=0.01, ha="left", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(RA / f"hinh_11_hoc{'_cuoi' if cuoi else ''}.png", bbox_inches="tight")
    plt.close(fig)


def ve_cv(cv):
    plt = _plt()
    phan = ["train (7 cơn bão)", "train6 (60 ngày)", "x11 (cực đoan)"]
    pp = [("Dijkstra phạt", "dphat"), ("SM11", "sm11"), ("Lai(DM10c)", "dm10c_lai"), ("Lai(DM11)", "dm11_lai"),
          ("Thầy", "thay")]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for i, (ten, k) in enumerate(pp):
        x = np.arange(3) + (i - 2) * 0.12
        ax[0].plot(x, [cv[p][k] for p in phan], "o", color=MAU[ten], ms=7, mec="white", mew=1.2, label=ten)
    ax[0].set_xticks(range(3), phan)
    ax[0].set_ylabel("mất DV % (mô hình không học tình huống)")
    ax[0].set_title("(a) kiểm thử chéo theo nhóm bão / ngày", loc="left", fontsize=9)
    ax[0].legend(fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    for i, (h, ten) in enumerate((("dm11", "DM11 thuần"), ("sm11", "SM11"))):
        x = np.arange(3) + (i - 0.5) * 0.3
        ngoai = [cv[p][h] for p in phan]
        trong = [cv[p][f"{h}_da_hoc"] for p in phan]
        for xx, a_, b_ in zip(x, ngoai, trong):
            ax[1].plot([xx, xx], [a_, b_], color=MAU[ten], lw=2, alpha=0.6)
        ax[1].plot(x, ngoai, "o", color=MAU[ten], ms=7, mec="white", mew=1.2, label=f"{ten} · chưa học")
        ax[1].plot(x, trong, "D", color=MAU[ten], ms=6, mfc="white", mew=1.6, label=f"{ten} · đã học")
    ax[1].set_xticks(range(3), phan)
    ax[1].set_ylabel("mất DV %")
    ax[1].set_title("(b) chưa học so với đã học — đoạn nối ngắn = không học thuộc", loc="left", fontsize=9)
    ax[1].legend(fontsize=7, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    fig.savefig(RA / "hinh_11_kiem_thu_cheo.png", bbox_inches="tight")
    plt.close(fig)


def ve_kiem_tra(bo45):
    plt = _plt()
    bo = [b for b in (*THUONG, *CUC_DOAN) if b in bo45]
    pp = ["Lai(DM11)", "Lai(DM10c)", "DM11 thuần", "SM11"]
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(bo) + 1.6))
    for i, ten in enumerate(pp):
        for j, b in enumerate(bo):
            m, (lo, hi) = bo45[b]["hieu_vs_dphat"][ten]
            y = j + (i - 1.5) * 0.18
            ax.plot([lo, hi], [y, y], color=MAU[ten], lw=2)
            ax.plot([m], [y], "o", color=MAU[ten], ms=5, mec="white", mew=1, label=ten if j == 0 else None)
    ax.axvline(0, color="#333333", lw=1)
    n_th = sum(b in THUONG for b in bo)
    ax.axhspan(n_th - 0.5, len(bo) - 0.5, color="#f3f0ff", zorder=0)
    ax.text(ax.get_xlim()[1], n_th - 0.4, "cực đoan ", ha="right", va="bottom", fontsize=8, color="#555555")
    ax.set_yticks(range(len(bo)), [TEN_BO[b] for b in bo])
    ax.invert_yaxis()
    ax.set_xlabel("hiệu mất DV so với Dijkstra phạt (điểm %, CI 95%); âm = tốt hơn")
    ax.legend(fontsize=8, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.06))
    ax.set_title("TN11 — bộ kiểm tra, 4–5 luồng (trung bình 3 seed)", loc="left", fontsize=10, pad=28)
    fig.tight_layout()
    fig.savefig(RA / "hinh_11_kiem_tra.png", bbox_inches="tight")
    plt.close(fig)


def ve_thoi_gian(tg):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    K = [k for k in ("4-5", "10", "20") if k in tg]
    for ten in ("Dijkstra phạt", "SM11", "DM11 thuần", "Lai(DM11)", "TKCB 10 lần"):
        ax.plot(range(len(K)), [tg[k][ten] for k in K], "o-", color=MAU[ten], lw=2, ms=6, mec="white", mew=1, label=ten)
    ax.set_yscale("log")
    ax.set_xticks(range(len(K)), [f"{k} luồng" for k in K])
    ax.set_ylabel("thời gian quyết định, trung vị (ms, thang log)")
    ax.set_title("TN11 — thời gian theo quy mô (C5)", loc="left", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RA / "hinh_11_thoi_gian.png", bbox_inches="tight")
    plt.close(fig)


def ve_du_lieu(kt):
    plt = _plt()
    tm = kt["thu_muc"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 0.28 * len(tm) + 1.8))
    ten = list(tm)
    n = [tm[d]["so_ban_ghi"] for d in ten]
    loi = [tm[d]["so_ban_ghi_loi"] + tm[d]["so_trung"] for d in ten]
    ax[0].barh(range(len(ten)), n, color="#5b5b5b")
    for i, (a_, b_) in enumerate(zip(n, loi)):
        ax[0].text(a_, i, f"  {a_}" + (f" (loại {b_})" if b_ else ""), va="center", fontsize=7, color="#333333")
    ax[0].set_yticks(range(len(ten)), ten, fontsize=7)
    ax[0].invert_yaxis()
    ax[0].set_xlabel("số tình huống")
    ax[0].set_title("(a) số tình huống từng tập (loại = lỗi + trùng)", loc="left", fontsize=9)
    nhom = [("học thường", [d for d in ("train10_k4_5", "train6_10_k4_5") if d in tm], "#5b5b5b"),
            ("học cực đoan x11", ["x11_k4_5"], "#5b5b5b"),
            ("kiểm tra cực đoan xc11 / xy11", ["xc11_k4_5", "xy11_k4_5"], "#5b5b5b")]
    for j, (nh, ds, mau) in enumerate(nhom):
        v = [tm[d]["thong_ke"]["R"]["tb"] for d in ds if d in tm and "R" in tm[d]["thong_ke"]]
        p = [tm[d]["thong_ke"]["R"]["p99"] for d in ds if d in tm and "R" in tm[d]["thong_ke"]]
        if v:
            ax[1].bar(j - 0.18, np.mean(v), 0.34, color=mau, label="trung bình" if j == 0 else None)
            ax[1].bar(j + 0.18, np.mean(p), 0.34, color=mau, alpha=0.4, label="phân vị 99" if j == 0 else None)
    ax[1].set_xticks(range(3), ["học thường", "học cực đoan", "kiểm tra cực đoan"])
    ax[1].set_ylabel("mưa lớn nhất trên cạnh (mm/h)")
    ax[1].set_title("(b) độ lệch phân bố: mưa lớn nhất trên cạnh mỗi tình huống", loc="left", fontsize=9)
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RA / "hinh_11_du_lieu.png", bbox_inches="tight")
    plt.close(fig)


# -------------------------------------------------------------------------------------------- báo cáo
def _ci(x, nd=2):
    return f"{x[0]:+.{nd}f} [{x[1][0]:+.{nd}f}; {x[1][1]:+.{nd}f}]"


def lenh_phan_tich(a):
    kt = json.loads((RA / "kiem_tra_du_lieu.json").read_text(encoding="utf-8")) if (RA / "kiem_tra_du_lieu.json").exists() \
        else {"thu_muc": {}, "ro_ri": {}, "x11": [], "k_kiem_tra": {}}
    hoc = doc_hoc()
    cv, _ = phan_tich_cv()
    T = _tai()
    bo45, o, tom, tg = phan_tich_kiem_tra(T, kt.get("ro_ri"))
    gt = gia_thuyet(bo45, tom, hoc, cv)
    RA.mkdir(parents=True, exist_ok=True)
    ra = {"cau_hinh": CFG, "kiem_tra_du_lieu": {k: v for k, v in kt.items() if k != "thu_muc"},
          "hoc": {k: {kk: vv for kk, vv in x.items() if kk != "nk"} for k, x in hoc.items()},
          "kiem_thu_cheo": cv, "kiem_tra": bo45, "o": o, "tom_tat_o": tom, "thoi_gian": tg, "gia_thuyet": gt}
    (RA / "ket_qua_11.json").write_text(json.dumps(ra, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    if kt["thu_muc"]:
        ve_du_lieu(kt)
    if hoc:
        ve_hoc(hoc)
        ve_hoc(hoc, cuoi=True)
    if cv:
        ve_cv(cv)
    if bo45:
        ve_kiem_tra(bo45)
    if tg:
        ve_thoi_gian(tg)
    viet_md(ra, kt)
    print("  " + "; ".join(f"{k}: {'ĐẠT' if v['dat'] else 'KHÔNG ĐẠT'}" for k, v in gt.items()), flush=True)
    dong_goi()


def dong_goi():
    """Một tệp để gửi lại: kết quả, biểu đồ, kiểm tra dữ liệu, phần, đường học mọi mô hình, mọi log TN11 (không kèm
    tệp đệm chấm cv_tep / cham_tep, có thể tạo lại)."""
    import zipfile
    tep = KQ / "TN11_ket_qua.zip"
    gom = [(f, Path("tn11") / f.name) for f in sorted(RA.iterdir()) if f.is_file()]
    gom += [(f, Path("mo_hinh11") / f.name) for f in sorted(M11.glob("*_nhat_ky.json"))]
    gom += [(f, Path("logs") / f.name) for f in sorted(LOG.glob("tn11*.log"))]
    with zipfile.ZipFile(tep, "w", zipfile.ZIP_DEFLATED) as z:
        for f, ten in gom:
            z.write(f, str(ten))
    so = {k: sum(str(t).startswith(k) for _, t in gom) for k in ("tn11", "mo_hinh11", "logs")}
    print(f"  đã gói {len(gom)} tệp ({so['tn11']} kết quả, {so['mo_hinh11']} đường học, {so['logs']} log) → {tep}",
          flush=True)


def _dong_ro_ri(tap, v):
    if not v["kb"]:
        return f"{tap}: không"
    return f"{tap}: {len(v['kb'])} cửa sổ kề ({', '.join(v['ngay'])}), trùng đúng ngày {len(v['trung_ngay'])}"


def viet_md(ra, kt):
    L = ["# Thí nghiệm 11 — kết quả (sinh tự động bởi `scripts/tn11.py`)", "",
         "Tiêu chí: TIEU_CHI_11.md. Mất DV bằng %, hiệu bằng điểm %, CI 95% bootstrap theo cửa sổ. Lai(·) = DM nếu ρ̂ < 1,1, "
         "ngược lại DM + TKCB.", "", "## 1. Dữ liệu: làm sạch, rò rỉ, cực đoan", "",
         "| Tập | Tình huống | Lỗi (loại) | Trùng | Thấp / giữa / cao | Mưa lớn nhất trên cạnh: TB / p99 |", "|---|---:|---|---:|---|---|"]
    for d, x in kt["thu_muc"].items():
        R = x["thong_ke"].get("R", {})
        L.append(f"| {d} | {x['so_ban_ghi']} | {x['so_ban_ghi_loi']} {x['loi'] or ''} | {x['so_trung']} | "
                 f"{' / '.join(str(x['dai_tai'][k]) for k in ('0', '1', '2'))} | "
                 f"{R.get('tb', float('nan')):.1f} / {R.get('p99', float('nan')):.1f} |")
    L += ["", "- Rò rỉ (cửa sổ val / kiểm tra cùng ngày hoặc kề ngày ±1 với cửa sổ học): " +
          "; ".join(_dong_ro_ri(k, v) for k, v in kt["ro_ri"].items()) + ". Bộ kiểm tra có cửa sổ kề ngày học được "
          "phân tích lại khi bỏ các cửa sổ đó (mục 4).",
          "- Cửa sổ cực đoan x11 (đỉnh, hệ số k): " + "; ".join(
              f"{x['ten']} (k = {x['k']:.2f}, dữ dội {x['do_du_doi']:.2f})" for x in kt["x11"]) + ".",
          "- k của bộ kiểm tra cực đoan: " + "; ".join(
              f"{t}: {min(v):.2f}–{max(v):.2f}" for t, v in kt["k_kiem_tra"].items() if v) + ".",
          "- Hình: `hinh_11_du_lieu.png`.", "", "## 2. Học: chống học chưa tới / học thuộc", "",
          "| Mô hình | Epoch chạy | Epoch tốt nhất | Dừng sớm | Loss train / val (tốt nhất) | val / train | J val (thầy) | Cờ |",
          "|---|---:|---:|---|---|---:|---|---|"]
    for k, x in ra["hoc"].items():
        co = ("CHƯA TỚI " if x["chua_toi"] else "") + ("HỌC THUỘC? " if x["ti_le_val_train"] > 1.25 else "")
        L.append(f"| {k} | {x['so_epoch']} | {x['epoch_tot']} | {x['dung_som_o'] or '–'} | {x['loss_train_tot']:.3f} / "
                 f"{x['loss_val_tot']:.3f} | {x['ti_le_val_train']:.2f} | {x['J_val_tot']:.3f} ({x['J_thay']:.3f}) | "
                 f"{co or 'ổn'} |")
    L += ["", "Hình: `hinh_11_hoc.png` (kiểm thử chéo), `hinh_11_hoc_cuoi.png` (mô hình cuối).", ""]
    cv = ra["kiem_thu_cheo"]
    if cv:
        L += ["## 3. Kiểm thử chéo 4 phần theo nhóm bão / ngày", "",
              "| Phần dữ liệu | n | Dijkstra phạt | Yen | SM11 | Lai(DM10c) | Lai(DM11) | Thầy | Lai(DM11) − Dijkstra phạt | "
              "Lai(DM11) − Lai(DM10c) | DM11 − SM11 | DM11 chưa học − đã học | SM11 chưa học − đã học |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---|"]
        for p_, x in cv.items():
            L.append(f"| {p_} | {x['n']} | {x['dphat']:.2f} | {x['yen']:.2f} | {x['sm11']:.2f} | {x['dm10c_lai']:.2f} | "
                     f"{x['dm11_lai']:.2f} | {x['thay']:.2f} | {_ci(x['dm11_tru_dphat'])} | {_ci(x['dm11_tru_dm10c'])} | "
                     f"{_ci(x['dm11_tru_sm11'])} | {_ci(x['dm11_khoang_hoc_thuoc'])} | {_ci(x['sm11_khoang_hoc_thuoc'])} |")
        L += ["", "Hình: `hinh_11_kiem_thu_cheo.png`.", ""]
    L += ["## 4. Bộ kiểm tra, 4–5 luồng (trung bình 3 seed)", "",
          "| Bộ | Dijkstra phạt | Yen | SM11 | Lai(DM10c) | Lai(DM11) | DM11 thuần | Lai(DM11) − Dijkstra phạt | "
          "Lai(DM11) − Lai(DM10c) | DM11 − SM11 |", "|---|---:|---:|---:|---:|---:|---:|---|---|---|"]
    for b, x in ra["kiem_tra"].items():
        t = x["trung_binh"]
        L.append(f"| {TEN_BO[b]} | {t['Dijkstra phạt']:.2f} | {t['Dijkstra (Yen)']:.2f} | {t['SM11']:.2f} | "
                 f"{t['Lai(DM10c)']:.2f} | {t['Lai(DM11)']:.2f} | {t['DM11 thuần']:.2f} | "
                 f"{_ci(x['hieu_vs_dphat']['Lai(DM11)'])} | {_ci(x['dm11_tru_dm10c'])} | {_ci(x['dm11_tru_sm11'])} |")
    ro = [(b, x["bo_ro_ri"]) for b, x in ra["kiem_tra"].items() if "bo_ro_ri" in x]
    if ro:
        L += ["", "**Độ nhạy — bỏ cửa sổ kề ngày học:**", ""]
        L += [f"- {TEN_BO[b]} (bỏ {r['so_cua_so_bo']} cửa sổ): Lai(DM11) − Dijkstra phạt {_ci(r['Lai(DM11) − Dijkstra phạt'])}; "
              f"Lai(DM11) − Lai(DM10c) {_ci(r['Lai(DM11) − Lai(DM10c)'])}; DM11 − SM11 {_ci(r['DM11 − SM11'])}" for b, r in ro]
    L += ["", "**Thành phần mất DV ở bộ cực đoan** — (a) không còn tuyến sống / (b) dự phòng quá tải:", ""]
    for b in CUC_DOAN:
        if b in ra["kiem_tra"]:
            L.append(f"- {TEN_BO[b]}: " + "; ".join(f"{k} {v[0]:.2f} / {v[1]:.2f}" for k, v in ra["kiem_tra"][b]["a_b"].items()))
    L += ["", "Hình: `hinh_11_kiem_tra.png`.", "", "## 5. Ô tiêu chí so với Dijkstra phạt (mô hình seed 0; Holm theo họ mọi ô)",
          "", "| Mô hình | Số ô | Thắng | Holm | Holm ở cực đoan | Holm ở thường | Tệ hơn |", "|---|---:|---:|---:|---:|---:|---:|"]
    for k, x in ra["tom_tat_o"].items():
        L.append(f"| {k} | {x['so_o']} | {x['thang']} | {x['thang_holm']} | {x['holm_cuc_doan']} | {x['holm_thuong']} | "
                 f"{x['te_hon']} |")
    if ra["thoi_gian"]:
        L += ["", "## 6. Thời gian quyết định (trung vị, ms)", "", "| Quy mô | " + " | ".join(next(iter(ra["thoi_gian"].values())))
              + " |", "|---|" + "---:|" * len(next(iter(ra["thoi_gian"].values())))]
        for K, x in ra["thoi_gian"].items():
            L.append(f"| {K} luồng | " + " | ".join(f"{v:.0f}" for v in x.values()) + " |")
        L += ["", "Hình: `hinh_11_thoi_gian.png`."]
    L += ["", "## 7. Giả thuyết", ""]
    for k, v in ra["gia_thuyet"].items():
        L.append(f"- **{k}: {'ĐẠT' if v['dat'] else 'KHÔNG ĐẠT'}** — " +
                 ", ".join(f"{kk} = {vv}" for kk, vv in v.items() if kk != "dat"))
    (RA / "KET_QUA_11.md").write_text("\n".join(L) + "\n", encoding="utf-8")


# ============================================================================================ chế độ tối đa
def tong_ram_gb():
    try:
        import ctypes

        class MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] + \
                       [(n, ctypes.c_ulonglong) for n in ("tong", "trong", "a", "b", "c", "d", "e")]
        m = MS()
        m.dwLength = ctypes.sizeof(MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return m.tong / 2**30
    except Exception:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30


def tai_nguyen(ram, vram):
    """Ngân sách theo RAM (GB, mặc định 20) và VRAM (GB). Đo trên máy người viết: tiến trình sinh ~3,9 GB; chấm ~0,3–0,6 GB;
    học ~1,5 GB (CPU) / ~3 GB (GPU, thêm thư viện CUDA); VRAM ~1,4 GB mỗi mô hình (ngữ cảnh CUDA + mạng H 128 + lô)."""
    cpu = os.cpu_count() or 4
    ram = min(ram, tong_ram_gb() - 2)
    gpu, ten = False, "không có"
    try:
        import torch
        if torch.cuda.is_available() and vram > 0:                       # --vram 0: bỏ GPU, học bằng CPU
            pr = torch.cuda.get_device_properties(0)
            gpu, ten, vram = True, pr.name, min(vram, pr.total_memory / 2**30)
    except Exception:
        pass
    p = {"cpu": cpu, "ram": round(ram, 1), "gpu": ten, "sinh": max(1, min(cpu, int((ram - 2) // 4)))}
    if gpu:
        p.update(hoc=max(1, min(8, int(vram // 1.4), int((ram - 3) // 3))), luong=2, thiet_bi="auto")
    else:
        p.update(hoc=max(1, min(cpu // 5, int((ram - 3) // 1.5))), thiet_bi="cpu")
        p["luong"] = max(1, cpu // p["hoc"])
    p["cham"] = max(1, min(cpu - 1, int((ram - 3) // 0.8)))
    return p


def lenh_toi_da(a):
    p = tai_nguyen(a.ram, a.vram)
    print(f"máy: {p['cpu']} luồng CPU, ngân sách RAM {p['ram']} GB, GPU: {p['gpu']}", flush=True)
    print(f"→ sinh {p['sinh']} tiến trình; học {p['hoc']} mô hình cùng lúc × {p['luong']} luồng ({p['thiet_bi']}); "
          f"chấm {p['cham']} tiến trình", flush=True)
    a.tien_trinh = p["sinh"]
    buoc("sinh", lenh_sinh, a)
    buoc("chuan-bi", lambda _: chay(py("-m", "scripts.tn11", "chuan-bi")), a)
    a.song_song, a.luong_cpu, a.thiet_bi = p["hoc"], p["luong"], p["thiet_bi"]
    buoc("huan-luyen", lenh_huan_luyen, a)
    a.tien_trinh = p["cham"]
    buoc("cv", lenh_cv, a)
    buoc("cham", lenh_cham, a)
    buoc("phan-tich", lenh_phan_tich, a)


def buoc(ten, f, a):
    print(f"=== {ten} {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    f(a)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("viec", choices=["sinh", "chuan-bi", "huan-luyen", "cv", "cham", "phan-tich", "toi-da"])
    ap.add_argument("--tien-trinh", type=int, default=4)
    ap.add_argument("--song-song", type=int, default=2)
    ap.add_argument("--luong-cpu", type=int, default=4)
    ap.add_argument("--thiet-bi", default="auto", choices=["cpu", "auto", "cuda"])
    ap.add_argument("--ram", type=float, default=20.0, help="ngân sách RAM (GB) cho toi-da")
    ap.add_argument("--vram", type=float, default=8.0, help="ngân sách VRAM (GB) cho toi-da")
    ap.add_argument("--epochs", type=int, default=CFG["epochs"])
    ap.add_argument("--H", type=int, default=CFG["H"])
    ap.add_argument("--dung-som", type=int, default=CFG["dung_som"])
    ap.add_argument("--S", type=int, default=CFG["S"])
    a = ap.parse_args()
    CFG.update(epochs=a.epochs, H=a.H, dung_som=a.dung_som, S=a.S)
    if not os.environ.get("PBL4_KHONG_LOG"):
        LOG.mkdir(parents=True, exist_ok=True)
        f = open(LOG / "tn11.log", "a", encoding="utf-8", buffering=1)
        sys.stdout, sys.stderr = _Tee(sys.stdout, f), _Tee(sys.stderr, f)
        print(f"\n##### {time.strftime('%Y-%m-%d %H:%M:%S')}  python -m scripts.tn11 {' '.join(sys.argv[1:])}", flush=True)
    {"sinh": lenh_sinh, "chuan-bi": lenh_chuan_bi, "huan-luyen": lenh_huan_luyen, "cv": lenh_cv, "cham": lenh_cham,
     "phan-tich": lenh_phan_tich, "toi-da": lenh_toi_da}[a.viec](a)
    print("xong", flush=True)


if __name__ == "__main__":
    main()
