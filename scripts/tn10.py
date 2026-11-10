"""
tn10.py — Thí nghiệm 10 "học tới nơi" (results/bao_da_tuyen/TIEU_CHI_10.md): 4 bước chọn bằng KIỂM THỬ CHÉO 4 phần trên
tập học (chia theo cửa sổ), rồi chấm bộ kiểm tra MỘT lần. Gọi từ scripts.chay_tn10 theo thứ tự; mỗi lệnh đọc / ghi
results/bao_da_tuyen/tn10/chon.json và tự bỏ qua phần đã xong, nên chạy lại an toàn.

    python -m scripts.tn10 huan-luyen --buoc 1      # bước 1: 4 cấu hình × 4 phần
    python -m scripts.tn10 chon --buoc 1            #   chọn cấu hình theo mất DV kiểm thử chéo (DM, S = 8)
    python -m scripts.tn10 chon --buoc 2            # bước 2: chọn S ∈ {8, 16, 32}
    python -m scripts.tn10 chon --buoc 3a           # bước 3: thầy 64 so với 32 thế giới nhãn (val10_t32 / val10_t64)
    python -m scripts.tn10 sinh --buoc 3            #   nếu qua: nhãn thầy 64 cho tập học A
    python -m scripts.tn10 huan-luyen --buoc 3      #   nếu qua: 4 phần trên nhãn thầy 64
    python -m scripts.tn10 chon --buoc 3b
    python -m scripts.tn10 sinh --buoc 4            # bước 4: nhãn thầy 64 cho tập học B (nếu bước 3 chọn thầy 64)
    python -m scripts.tn10 huan-luyen --buoc 4      #   4 phần trên A + B
    python -m scripts.tn10 chon --buoc 4            #   chọn theo kiểm thử chéo trên A + B
    python -m scripts.tn10 huan-luyen --buoc day    # DM10 và DM6r trên TOÀN BỘ tập học, 3 seed mỗi họ
    python -m scripts.tn10 chon --buoc chot         # chép mô hình → mo_hinh10/{r6,d10}_s*.pt + số chuỗi S
    python -m scripts.tn10 cham                     # chấm Lai(DM6r), Lai(DM10) trên bộ kiểm tra

Tên cấu hình: e<epoch>h<H>[_t64][_ab]. Mô hình kiểm thử chéo: <cấu hình>_p<phần> (seed = phần); mô hình cuối:
<cấu hình>_day_s<seed>. DM6r = e20h64 (cấu hình của DM6) học lại trên cùng tập học → đối chứng cùng dữ liệu.
Mọi mẫu DM dùng hạt giống theo bản ghi (cham_lai_dt.giong_ban_ghi) → không phụ thuộc số tiến trình.
"""

from __future__ import annotations

import argparse
import glob
import json
import pickle
import re
import shutil
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

GOC = Path(__file__).resolve().parents[1]
KQ = GOC / "results" / "bao_da_tuyen"
RA = KQ / "tn10"
M10 = KQ / "mo_hinh10"
DL = GOC / "data" / "bao_da_tuyen"
LOG = GOC / "logs"

GOC_HO = "e20h64"                                     # cấu hình của DM6
UNG_VIEN_1 = ("e20h64", "e60h64", "e60h128", "e120h128")
CAC_S = (8, 16, 32)
SO_BUOC_KHU, SO_PHAN = 5, 4
NGUONG = 0.3            # nhận một thay đổi nếu mất DV của DM (kiểm thử chéo) giảm ≥ 0,3 điểm
NGUONG_S = 0.1          # bước 2: S nhỏ nhất trong khoảng 0,1 điểm so với S tốt nhất
NGUONG_A = 0.1          # bước 4: không được tệ hơn 0,1 điểm trên phần A
TAP_A, TAP_B = ("train10_k4_5", "train6_10_k4_5"), ("train_b10_k4_5", "train6_b10_k4_5")
TEST_45 = ("c6", "yagi5", "c6_b", "yagi5_b", "ngay9", "nam9", "bao9")
TEST_LON = ("c6", "yagi5", "c6_b", "yagi5_b")
B_BOOT = 10_000


def doc_chon():
    tep = RA / "chon.json"
    return json.loads(tep.read_text(encoding="utf-8")) if tep.exists() else {}


def ghi_chon(c):
    RA.mkdir(parents=True, exist_ok=True)
    (RA / "chon.json").write_text(json.dumps(c, ensure_ascii=False, indent=1), encoding="utf-8")


def cau_hinh(ten):
    """e60h128_t64_ab → (epoch, H, thí nghiệm của huan_luyen_dt: 10 A / 11 A thầy 64 / 12 A+B / 13 A+B thầy 64)."""
    m = re.match(r"e(\d+)h(\d+)", ten)
    return int(m.group(1)), int(m.group(2)), 10 + ("_t64" in ten) + 2 * ("_ab" in ten)


# ----------------------------------------------------------------------------------------------- huấn luyện
def huan_luyen(viec, song_song, luong):
    """viec: [(tên mô hình, cấu hình, seed, phần bỏ ra hoặc None)] — bỏ qua mô hình đã có .xong."""
    viec = [v for v in viec if not (M10 / f"{v[0]}.xong").exists()]
    LOG.mkdir(exist_ok=True)
    for i in range(0, len(viec), song_song):
        chay = []
        for ten_mh, ch, seed, phan in viec[i:i + song_song]:
            E, H, tn = cau_hinh(ch)
            lenh = [sys.executable, "-W", "ignore", "-m", "src.nhieu_luong.huan_luyen_dt", "dm", "--thi-nghiem", str(tn),
                    "--ten", ten_mh, "--them", "--seed", str(seed), "--epochs", str(E), "--H", str(H), "--luong-cpu",
                    str(luong), *([] if phan is None else ["--bo-phan", str(phan)])]
            print(f"  {time.strftime('%H:%M:%S')} huấn luyện {ten_mh} → logs/tn10_{ten_mh}.log", flush=True)
            chay.append((ten_mh, subprocess.Popen(lenh, cwd=GOC, stderr=subprocess.STDOUT,
                                                  stdout=open(LOG / f"tn10_{ten_mh}.log", "w", encoding="utf-8"))))
        for ten_mh, p in chay:
            if p.wait() != 0:
                raise SystemExit(f"huấn luyện {ten_mh} lỗi — xem logs/tn10_{ten_mh}.log")
            print(f"  {time.strftime('%H:%M:%S')} xong {ten_mh}", flush=True)


def kiem_thu_cheo(ch):
    return [(f"{ch}_p{f}", ch, f, f) for f in range(SO_PHAN)]


def lenh_huan_luyen(a):
    c, ss = doc_chon(), a.song_song
    if a.buoc == "1":
        huan_luyen([v for ch in UNG_VIEN_1 for v in kiem_thu_cheo(ch)], ss, a.luong_cpu)
    elif a.buoc == "3":
        if not c.get("buoc3a", {}).get("gan_nhan_lai"):
            print("bước 3: thầy 64 không qua ngưỡng → không huấn luyện thêm")
            return
        huan_luyen(kiem_thu_cheo(c["buoc2"]["ho"] + "_t64"), ss, a.luong_cpu)
    elif a.buoc == "4":
        huan_luyen(kiem_thu_cheo(c["buoc3b"]["ho"] + "_ab"), ss, a.luong_cpu)          # ~2,5 GB RAM mỗi mô hình
    elif a.buoc == "day":
        ch = c["buoc4"]["ho"]
        viec = [(f"{GOC_HO}_day_s{k}", GOC_HO, k, None) for k in range(3)]
        if ch != GOC_HO:
            viec += [(f"{ch}_day_s{k}", ch, k, None) for k in range(3)]
        huan_luyen(viec, ss, a.luong_cpu)


# ----------------------------------------------------------------------------------------------- sinh dữ liệu
def sinh(tap, tt):
    lenh = [sys.executable, "-W", "ignore", "-m", "src.nhieu_luong.tao_du_lieu_dt", "--tap", tap, "--K", "4-5",
            "--tien-trinh", str(tt)]
    print("  " + " ".join(lenh[3:]), flush=True)
    subprocess.run(lenh, cwd=GOC, check=True)


def lenh_sinh(a):
    c = doc_chon()
    if a.buoc == "3":
        if not c.get("buoc3a", {}).get("gan_nhan_lai"):
            print("bước 3: thầy 64 không qua ngưỡng → không gắn lại nhãn")
            return
        for t in ("train10_t64", "train6_10_t64"):
            sinh(t, a.tien_trinh)
    elif a.buoc == "4":
        if "_t64" not in c["buoc3b"]["ho"]:
            print("bước 4: nhãn thầy 32 → dùng train_b10, train6_b10")
            return
        for t in ("train_b10_t64", "train6_b10_t64"):
            sinh(t, a.tien_trinh)


# ----------------------------------------------------------------------------------------------- kiểm thử chéo
_NET = {}


def _danh_gia_tep(job):
    """Mỗi bản ghi của một tệp tập học được chấm bằng mô hình của phần KHÔNG học nó (thế giới CHẤM đã lưu)."""
    import torch
    from src.nhieu_luong import mo_hinh_dt as md
    from src.nhieu_luong.cham_lai_dt import giong_ban_ghi
    from src.nhieu_luong.cham_mo_hinh_dt import _tg
    from src.nhieu_luong.huan_luyen_dt import phan_cua
    tep, ch, S, mh = job
    torch.set_num_threads(1)
    thu_muc, ra = Path(tep).parent.name, []
    for r in pickle.load(open(tep, "rb")):
        ten = f"{ch}_p{phan_cua(thu_muc, r['kb'])}"
        if (mh, ten) not in _NET:
            _NET[(mh, ten)] = md.nap(ten, Path(mh))
        a = md.chon_dm(_NET[(mh, ten)], r["gon"], S, SO_BUOC_KHU, torch.Generator().manual_seed(giong_ban_ghi(r)))
        c = _tg(r["gon"], r["cham"]).cham(r["gon"].tu_lua_chon(a))
        ra.append([c["mat_dv_%"], c["J"], r["ens"]["thay"]["mat_dv_%"], r["ens"]["dphat"]["mat_dv_%"]])
    return tep, ra


def cac_tep(thu_mucs):
    return [t for d in thu_mucs for t in sorted(glob.glob(str(DL / d / "kb_*.pkl")))]


def danh_gia(ch, S, thu_mucs, tt):
    """[n, 4] (mất DV DM, J DM, mất DV thầy, mất DV Dijkstra phạt) theo thứ tự bản ghi; lưu đệm ở tn10/kiem_thu_cheo/."""
    tep_dem = RA / "kiem_thu_cheo" / f"{ch}_S{S}_{'+'.join(thu_mucs)}.json"
    if tep_dem.exists():
        return np.array(json.loads(tep_dem.read_text(encoding="utf-8")))
    teps = cac_tep(thu_mucs)
    if not teps:
        raise SystemExit(f"thiếu tập học có thế giới chấm: {thu_mucs}")
    s = time.time()
    with Pool(tt) as pool:
        kq = dict(pool.imap_unordered(_danh_gia_tep, [(t, ch, S, str(M10)) for t in teps]))
    x = [v for t in teps for v in kq[t]]
    tep_dem.parent.mkdir(parents=True, exist_ok=True)
    tep_dem.write_text(json.dumps(x), encoding="utf-8")
    print(f"  kiểm thử chéo: {ch} S = {S} trên {'+'.join(thu_mucs)}: mất DV {np.mean([v[0] for v in x]):.3f} "
          f"({len(x)} bản ghi, {time.time() - s:.0f} s)", flush=True)
    return np.array(x)


def nhan_ban_ghi(thu_mucs):
    """Mã cửa sổ (tập × cửa sổ) và phần của từng bản ghi, cùng thứ tự với danh_gia."""
    from src.nhieu_luong.huan_luyen_dt import phan_cua
    ma, ph = [], []
    for i, d in enumerate(thu_mucs):
        for t in sorted(glob.glob(str(DL / d / "kb_*.pkl"))):
            for r in pickle.load(open(t, "rb")):
                ma.append(i * 10_000 + int(r["kb"]))
                ph.append(phan_cua(d, r["kb"]))
    return np.array(ma), np.array(ph)


def bootstrap(v, kb, seed=0):
    u = np.unique(kb)
    tong, dem = np.array([v[kb == k].sum() for k in u]), np.array([(kb == k).sum() for k in u])
    j = np.random.default_rng(seed).integers(0, len(u), (B_BOOT, len(u)))
    m = tong[j].sum(1) / dem[j].sum(1)
    return [float(v.mean()), [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]]


def epoch_tot(ch):
    """Epoch tốt nhất (theo J val) / số epoch của từng mô hình kiểm thử chéo — xem đã học tới nơi chưa."""
    ra = []
    for f in range(SO_PHAN):
        tep = M10 / f"{ch}_p{f}_nhat_ky.json"
        if tep.exists():
            nk = json.loads(tep.read_text(encoding="utf-8"))["nhat_ky"]
            ra.append([min(nk, key=lambda r: r["J_val"])["epoch"], len(nk)])
    return ra


def tom_tat(ch, S, thu_mucs, tt, nhan):
    X = danh_gia(ch, S, thu_mucs, tt)
    kb, ph = nhan
    return {"cau_hinh": ch, "S": S, "mat_dv": float(X[:, 0].mean()), "J": float(X[:, 1].mean()),
            "G": float((X[:, 0] - X[:, 2]).mean()), "dm_tru_dphat": float((X[:, 0] - X[:, 3]).mean()),
            "theo_phan": [float(X[ph == f, 0].mean()) for f in range(SO_PHAN)], "epoch_tot": epoch_tot(ch)}, X


def so_sanh(moi, cu, S, thu_mucs, tt, nhan):
    tm, Xm = tom_tat(moi, S, thu_mucs, tt, nhan)
    tc, Xc = tom_tat(cu, S, thu_mucs, tt, nhan)
    return {"moi": tm, "cu": tc, "hieu": bootstrap(Xm[:, 0] - Xc[:, 0], nhan[0]),
            "hieu_J": bootstrap(Xm[:, 1] - Xc[:, 1], nhan[0])}


def lenh_chon(a):
    c, tt = doc_chon(), a.tien_trinh
    if a.buoc == "1":
        nh = nhan_ban_ghi(TAP_A)
        ss = {ch: so_sanh(ch, GOC_HO, 8, TAP_A, tt, nh) for ch in UNG_VIEN_1 if ch != GOC_HO}
        tot = min(ss, key=lambda ch: ss[ch]["moi"]["mat_dv"])
        c["buoc1"] = {"so_sanh": ss, "goc": tom_tat(GOC_HO, 8, TAP_A, tt, nh)[0],
                      "ho": tot if ss[tot]["hieu"][0] <= -NGUONG else GOC_HO,
                      "quy_tac": f"cấu hình tốt nhất nếu giảm ≥ {NGUONG} điểm so với e20h64 (kiểm thử chéo, S = 8)"}
    elif a.buoc == "2":
        ch, nh = c["buoc1"]["ho"], nhan_ban_ghi(TAP_A)
        bang = {S: tom_tat(ch, S, TAP_A, tt, nh)[0]["mat_dv"] for S in CAC_S}
        S = min(s for s in CAC_S if bang[s] <= min(bang.values()) + NGUONG_S)
        c["buoc2"] = {"ho": ch, "bang": bang, "S": S, "quy_tac": f"S nhỏ nhất trong khoảng {NGUONG_S} điểm so với S tốt nhất"}
    elif a.buoc == "3a":
        R = {t: [r for f in cac_tep([f"{t}_k4_5"]) for r in pickle.load(open(f, "rb"))] for t in ("val10_t32", "val10_t64")}
        if not R["val10_t32"] or len(R["val10_t32"]) != len(R["val10_t64"]):
            raise SystemExit("thiếu hoặc lệch dữ liệu val10_t32 / val10_t64")
        khop = all(x["t"] == y["t"] and x["muc"] == y["muc"] and x["kb"] == y["kb"] and np.allclose(x["gon"].lam, y["gon"].lam)
                   for x, y in zip(R["val10_t32"], R["val10_t64"]))
        v = {t: np.array([r["ens"]["thay"]["mat_dv_%"] for r in rr]) for t, rr in R.items()}
        h = bootstrap(v["val10_t64"] - v["val10_t32"], np.array([r["kb"] for r in R["val10_t32"]]))
        c["buoc3a"] = {"cung_tinh_huong": bool(khop), "thay32": float(v["val10_t32"].mean()),
                       "thay64": float(v["val10_t64"].mean()), "hieu": h,
                       "gan_nhan_lai": bool(khop and h[0] <= -NGUONG and h[1][1] < 0),
                       "quy_tac": f"gắn lại nhãn nếu thầy 64 giảm ≥ {NGUONG} điểm và CI cận trên < 0"}
    elif a.buoc == "3b":
        ch, S = c["buoc2"]["ho"], c["buoc2"]["S"]
        if not c["buoc3a"]["gan_nhan_lai"]:
            c["buoc3b"] = {"ho": ch, "bo_qua": True}
        else:
            ss = so_sanh(ch + "_t64", ch, S, TAP_A, tt, nhan_ban_ghi(TAP_A))
            c["buoc3b"] = {"so_sanh": ss, "ho": ch + "_t64" if ss["hieu"][0] <= -NGUONG else ch}
    elif a.buoc == "4":
        ch, S = c["buoc3b"]["ho"], c["buoc2"]["S"]
        ab = so_sanh(ch + "_ab", ch, S, TAP_A + TAP_B, tt, nhan_ban_ghi(TAP_A + TAP_B))
        sa = so_sanh(ch + "_ab", ch, S, TAP_A, tt, nhan_ban_ghi(TAP_A))
        c["buoc4"] = {"so_sanh_AB": ab, "so_sanh_A": sa,
                      "ho": ch + "_ab" if ab["hieu"][0] <= -NGUONG and sa["hieu"][0] <= NGUONG_A else ch,
                      "quy_tac": f"nhận nếu A + B giảm ≥ {NGUONG} và phần A không tệ hơn {NGUONG_A} điểm"}
    elif a.buoc == "chot":
        ch, S = c["buoc4"]["ho"], c["buoc2"]["S"]
        for ho_, nguon, s_ in (("r6", GOC_HO, 8), ("d10", ch, S)):
            for k in range(3):
                shutil.copyfile(M10 / f"{nguon}_day_s{k}.pt", M10 / f"{ho_}_s{k}.pt")
                (M10 / f"chon_buoc_{ho_}_s{k}.json").write_text(json.dumps(
                    {"S": s_, "so_buoc": SO_BUOC_KHU, "nguon": f"{nguon}_day_s{k}.pt", "quy_tac": "thí nghiệm 10"},
                    ensure_ascii=False, indent=1), encoding="utf-8")
        c["chot"] = {"ho": ch, "S": S, "trung_DM6r": ch == GOC_HO and S == 8}
    ghi_chon(c)
    k = "chot" if a.buoc == "chot" else f"buoc{a.buoc}"
    print(json.dumps(c.get(k), ensure_ascii=False, indent=1)[:4000], flush=True)


# ----------------------------------------------------------------------------------------------- chấm kiểm tra
def cham(tap, K, ten, khoa, tkcb10, tt):
    lenh = [sys.executable, "-W", "ignore", "-m", "src.nhieu_luong.cham_lai_dt", "--tap", tap, "--K", K, "--mo-hinh",
            str(M10), "--ten", ten, "--khoa", khoa, "--khoa-tkcb10", tkcb10, "--tien-trinh", str(tt), "--giong-ban-ghi"]
    print(f"  {time.strftime('%H:%M:%S')} chấm {khoa} trên {tap} {K} luồng", flush=True)
    subprocess.run(lenh, cwd=GOC, check=True)


def lenh_cham(a):
    c = doc_chon()
    if "chot" not in c:
        raise SystemExit("chưa chốt (python -m scripts.tn10 chon --buoc chot)")
    ho = [("r6", "")] + ([] if c["chot"]["trung_DM6r"] else [("d10", "")])
    mo_hinh = [(f"{h}_s{k}", h if k == 0 else f"{h}_s{k}") for h, _ in ho for k in range(3)]
    for tap in TEST_45:
        for ten, khoa in mo_hinh:
            cham(tap, "4-5", ten, khoa, KHOA_T10 if khoa == "r6" and tap in TEST_LON else "", a.tien_trinh)
    for K in ("10", "20"):
        for tap in TEST_LON:
            for ten, khoa in mo_hinh:
                if khoa in ("r6", "d10"):
                    cham(tap, K, ten, khoa, KHOA_T10 if khoa == "r6" else "", a.tien_trinh)


KHOA_T10 = "online_tkcb10_6"


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("viec", choices=["huan-luyen", "sinh", "chon", "cham"])
    ap.add_argument("--buoc", default="1", choices=["1", "2", "3", "3a", "3b", "4", "day", "chot"])
    ap.add_argument("--tien-trinh", type=int, default=3)
    ap.add_argument("--song-song", type=int, default=2, help="số mô hình huấn luyện cùng lúc (~1,2 GB RAM, A + B ~2,5 GB)")
    ap.add_argument("--luong-cpu", type=int, default=4, help="số luồng PyTorch mỗi mô hình")
    a = ap.parse_args()
    {"huan-luyen": lenh_huan_luyen, "sinh": lenh_sinh, "chon": lenh_chon, "cham": lenh_cham}[a.viec](a)


if __name__ == "__main__":
    main()
