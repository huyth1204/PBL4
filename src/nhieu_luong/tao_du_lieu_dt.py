"""
tao_du_lieu_dt.py — Sinh dữ liệu và chấm thí nghiệm 4 "tập tuyến dự phòng" (results/bao_da_tuyen/TIEU_CHI_4.md).

    python -m src.nhieu_luong.tao_du_lieu_dt --tap train --K 4-5                 # nhãn thầy cho DM
    python -m src.nhieu_luong.tao_du_lieu_dt --tap val --K 4-5                   # nhãn + chấm (chọn checkpoint)
    python -m src.nhieu_luong.tao_du_lieu_dt --tap val --K 20 --buoc-moc 4       # chạy thử quy mô
    python -m src.nhieu_luong.tao_du_lieu_dt --tap test --K 10 --mo-hinh         # Yagi, chấm cả DM / softmax

Mỗi cửa sổ ghi data/bao_da_tuyen/<tập>_k<K>/kb_XXXX.pkl: danh sách bản ghi (một mốc × một mức tải) gồm
  gon        bài toán thu gọn về kho tuyến (đầu vào của DM) + đặc trưng cạnh / ứng viên chỉ dùng thông tin ở o
  nhan       lời giải của thầy: TKCB trên J tính bằng 32 thế giới NHÃN (rẽ nhánh tại o)
  ens / qs   (val, test) số đo mọi phương pháp trên 32 thế giới CHẤM — seed độc lập với bộ nhãn, không phương pháp
             nào thấy — và trên thế giới quan sát
  cham       (val) thế giới chấm thu gọn trên kho: sống [W, U] và dung lượng còn trống nhỏ nhất trong cửa sổ [W, U]
             (ρ + F/C ≥ 1 ở một mốc nào đó ⇔ F ≥ min C·(1 − ρ)), để chọn checkpoint DM
Cùng cơn, cửa sổ, split hình học, seed kịch bản như thí nghiệm 3.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import os
import pickle
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from src.ngau_nhien.kich_ban import sinh_kich_ban
from src.ngau_nhien.ma_tran_rui_ro import ma_tran_online
from src.nhieu_luong import da_tuyen as dt
from src.nhieu_luong import tao_du_lieu_rr as td
from src.nhieu_luong import the_gioi_rr as tg
from src.nhieu_luong.bai_toan import MUC_TAI
from src.nhieu_luong.bai_toan_rr import DELTA, TUOI
from src.ngau_nhien.thoi_tiet import GIONG

GOC = Path(__file__).resolve().parents[2]
RA = GOC / "data" / "bao_da_tuyen"
BETA_GAMMA = (10.0, 10.0)            # Dijkstra phạt: chỉnh trên val ở lần chạy thử (chay_thu_val.json)
SO_TG_NHAN = SO_TG_CHAM = 32
SO_DAU_THAY = 6                      # điểm xuất phát của thầy
SO_DO = ("J", "mat_dv_%", "khong_tuyen_song_%", "du_phong_qua_tai_%", "tre_chinh_ms")
LOAI_LEN = {"leo": 0, "uav": 1, "haps": 2}

_mo_hinh = None


def so_nguon(K):
    """3 nguồn ở K = 4–5 (như thí nghiệm 3); K lớn hơn giữ khoảng 1,5 luồng mỗi nguồn."""
    return 3 if K <= 5 else int(np.ceil(K / 1.5))


def phan_tich_K(s):
    a, _, b = s.partition("-")
    return int(a), int(b or a)


def _seeds(seed, t, bo, W):
    return [int(x.generate_state(1)[0]) for x in np.random.SeedSequence([seed, t, bo]).spawn(W)]


BAN_KINH_GIONG = 100.0          # km — thí nghiệm 6: giông lân cận UAV
_CAPE = {}


def cape_luc(h0, o, tep=None):
    """[S] CAPE (J/kg) của giờ chứa mốc o — giờ hiện tại, không phải giờ tới. tep: tệp thời tiết của tập (thí nghiệm 9:
    2017–2021); mặc định thời tiết 2024–2025."""
    tep = str(tep or td.THOI_TIET)
    if tep not in _CAPE:
        with np.load(tep, allow_pickle=False) as z:
            _CAPE[tep] = z["cape"].astype(np.float32)
    return _CAPE[tep][h0 + int(o * 20 / 3600)]


def dac_trung_uav(g, q, o, cape_o):
    """Thí nghiệm 6, cho mỗi nút (0 / False nếu không phải UAV có điểm thời tiết): có giông trong bán kính 100 km ở o;
    điểm của UAV có giông ở mốc nào đó trong 1 giờ trước o; CAPE giờ hiện tại / 2000 (chặn ở 2). Chỉ quan sát hiện tại
    và quá khứ — không dùng thời tiết giờ tới (mô phỏng dùng đúng số đó làm sự thật)."""
    s = g.node_site
    sp = np.maximum(s, 0)
    uav = (np.array(g.kind) == "uav") & (s >= 0)
    lan_can = ((g.kc_diem <= BAN_KINH_GIONG) & q["giong"][o][None]).any(1)
    gio_qua = q["giong"][max(0, o - 180):o + 1].any(0)
    return (uav & lan_can[sp], uav & gio_qua[sp], np.where(uav, np.clip(cape_o[sp] / 2000, 0, 2), 0.0))


def dac_trung_canh(g, q, mg, U, o, cape_o=None):
    """Đặc trưng cạnh chỉ từ quan sát ở o (và 15 phút trước): băng tần, mưa, xu hướng mưa, gió giật / giông ở đầu UAV,
    giông ở đầu mặt đất — cùng loại thông tin với bảng rủi ro, nhưng để thô cho mạng tự học. cape_o (thí nghiệm 6): thêm
    đặc trưng giông quanh UAV."""
    u, v = mg.eu[U], mg.ev[U]
    them = {}
    if cape_o is not None:
        lc, g1, cp = dac_trung_uav(g, q, o, cape_o)
        them = {"giong_lan_can_uav": lc[u] | lc[v], "giong_1h_uav": g1[u] | g1[v],
                "cape_uav": np.maximum(cp[u], cp[v]).astype(np.float32)}
    site = g.site[u, v]
    R = np.where(site >= 0, q["R"][o][np.maximum(site, 0)], 0.0)
    R0 = np.where(site >= 0, q["R"][o - 45][np.maximum(site, 0)], 0.0)
    kind = np.array(g.kind)
    ns = g.node_site
    uav = lambda n: (kind[n] == "uav") & (ns[n] >= 0)
    dat = lambda n: np.isin(kind[n], ["gateway", "user"]) & (ns[n] >= 0)
    gust = lambda n: np.where(uav(n), q["gust"][o][np.maximum(ns[n], 0)], 0.0)
    giong = lambda n: q["giong"][o][np.maximum(ns[n], 0)]
    return {"band": g.band[u, v].astype(np.int8), "R": R.astype(np.float32), "dR": (R - R0).astype(np.float32),
            "gust": np.maximum(gust(u), gust(v)).astype(np.float32),
            "giong_dat": ((dat(u) & giong(u)) | (dat(v) & giong(v))),
            "giong_uav": ((uav(u) & giong(u)) | (uav(v) & giong(v))),
            "mat_dat": np.isin(kind[u], ["gateway", "user"]) | np.isin(kind[v], ["gateway", "user"]), **them}


def dac_trung_uv(g, th, k):
    """[số kho, 5]: loại chặng đầu (LEO / UAV / HAPS, one-hot), thuộc lời giải Suurballe, số loại đường lên của nguồn."""
    kt = th["kho"][th["cap"][k]]
    ra = np.zeros((len(kt["kho"]), 5), np.float32)
    for i, r in enumerate(kt["kho"]):
        j = LOAI_LEN.get(g.kind[r[1]])
        if j is not None:
            ra[i, j] = 1
        ra[i, 3] = i in kt["i_suurballe"]
        ra[i, 4] = kt["so_loai"] / 3
    return ra


def thay(bt, En, dau, rng):
    """TKCB trên J ensemble nhãn từ nhiều điểm xuất phát; trả (lựa chọn, J)."""
    while len(dau) < SO_DAU_THAY:
        dau.append(np.array([rng.integers(len(O)) for O in bt.opts]))
    return min((dt.tkcb(En, a0, rng) for a0 in dau), key=lambda x: x[1])


def tkcb10(bt, rng):
    """Mốc thời gian của C5: TKCB 10 lần trên J bảng (tham lam + 9 điểm ngẫu nhiên). Trả (lựa chọn, ms)."""
    s = time.perf_counter()
    B10 = dt.TheGioi.tu_bang(bt, rng)
    dau = [dt.tham_lam(B10)] + [np.array([rng.integers(len(O)) for O in bt.opts]) for _ in range(9)]
    a, _ = min((dt.tkcb(B10, a0, rng) for a0 in dau), key=lambda x: x[1])
    return a, (time.perf_counter() - s) * 1e3


def anh_xa(bt, sol):
    """Lời giải (chỉ số bảng) → lựa chọn trong kho nếu mọi luồng đều chọn trong kho, ngược lại None."""
    a = [dt.lua_chon_cua(bt, k, [bt.bang[k][j] for j in sol[k]]) for k in range(bt.K)]
    return None if any(x is None for x in a) else np.array(a)


def _mot_cua_so(job):
    tap, K_rg, i, ten, h0, seed, moc, tep, mo_hinh, co_thay, lien_tuc = job
    if td._g is None:
        td._khoi_tao()
    if Path(tep).exists():
        return tep, 0, 0.0
    g, p = td._g, td._p                    # p: cấu hình hợp đồng của bảng rủi ro — bộ định tuyến luôn dùng p này
    tt, p_sim = moi_truong(tap, i)         # thế giới mô phỏng (thí nghiệm 7a / 11: đã stress)
    if mo_hinh:
        _nap_mo_hinh()
    t_bd = time.time()
    Wn, q = sinh_kich_ban(g, tt, h0, seed, p_sim)
    pl = tg.phat_lai(g, tt, h0, seed, p_sim)
    rng = np.random.default_rng([seed, K_rg[1]])
    ban_ghi = []
    for so, t in enumerate(moc):
        t = int(t)
        o = t - TUOI
        past = {k: q[k][:o + 1] for k in ("R", "gust", "giong", "tat")}
        r = ma_tran_online(td._bang, g, Wn[o], past, t, TUOI, DELTA, tham_so=p)
        th = dt.tao_tinh_huong(g, q, r, t, rng, so_luong=K_rg, so_nguon=so_nguon(K_rg[1]))
        if th is None:
            continue
        mg = th["mg"]
        s = time.perf_counter()
        Y = dt.yen_m(th)
        tg_yen = time.perf_counter() - s
        S = dt.suurballe_m(th)
        n_nhan = TAP_10[tap][1] if tap in TAP_10 else SO_TG_NHAN      # thí nghiệm 10: 64 thế giới nhãn (32 đầu trùng)
        sn, sc = _seeds(seed, t, 0, n_nhan), _seeds(seed, t, 1, SO_TG_CHAM)
        cac_muc = (np.random.default_rng([seed, t, 5]).uniform(*TAI_LIEN_TUC, 3).tolist() if lien_tuc
                   else list(MUC_TAI))                    # thí nghiệm 5: 3 hệ số tải bốc đều, cùng luồng
        for so_muc, muc in enumerate(cac_muc):
            lam = th["lam"] * (MUC_TAI[muc] if isinstance(muc, str) else muc) / th["u"]
            s = time.perf_counter()
            DP = dt.dijkstra_phat(th, lam, *BETA_GAMMA)
            tg_dp = time.perf_counter() - s
            bt = dt.BaiToan(th, muc, [Y[k] + DP[k] for k in range(len(lam))])
            gon = dt.BaiToanGon(bt)
            gon.dac_trung_canh = dac_trung_canh(g, q, mg, gon.U, o, cape_luc(h0, o, tep_thoi_tiet(tap)))
            gon.dac_trung_uv = [dac_trung_uv(g, th, k) for k in range(bt.K)]
            En = dt.TheGioi.ensemble(bt, g, pl, p_sim, sn) if co_thay else None
            s = time.perf_counter()
            B = dt.TheGioi.tu_bang(bt, np.random.default_rng([seed, t, 2]))
            a_on, _ = dt.tkcb(B, dt.tham_lam(B))
            tg_on = time.perf_counter() - s
            sol = {"yen": bt.loi_giai(Y), "suurballe": bt.loi_giai(S), "dphat": bt.loi_giai(DP),
                   "online_tkcb": bt.tu_lua_chon(a_on)}
            dau = [a_on] + [a for a in (anh_xa(bt, sol["suurballe"]), anh_xa(bt, sol["dphat"])) if a is not None]
            r_thay = np.random.default_rng([seed, t, 6, so_muc]) if tap in TAP_10 else rng   # TN10: thầy không chạm rng
            a_thay, J_thay = thay(bt, En, dau, r_thay) if co_thay else (None, float("nan"))
            rec = {"gon": gon, "nhan": None if a_thay is None else a_thay.astype(np.int16), "J_nhan": float(J_thay),
                   "kb": i, "bao": ten,
                   "t": t, "muc": muc, "K": bt.K, "nguon": th["nguon"],
                   "tg_ms": {"yen": tg_yen * 1e3, "dphat": tg_dp * 1e3, "online_tkcb": tg_on * 1e3}}
            if tap not in TAP_TRAIN:
                Ec = dt.TheGioi.ensemble(bt, g, pl, p_sim, sc)
                Q = dt.TheGioi.quan_sat(bt, Wn, q)
                if co_thay:
                    sol["thay"] = bt.tu_lua_chon(a_thay)
                if mo_hinh:
                    for kieu in _mo_hinh:
                        s = time.perf_counter()
                        a = _mo_hinh[kieu](gon)
                        rec["tg_ms"][kieu] = (time.perf_counter() - s) * 1e3
                        sol[kieu] = bt.tu_lua_chon(a)
                if so % 4 == 0 and tap == "val":                 # test: đo cùng DM trong cham_mo_hinh_dt (C5)
                    rec["tg_ms"]["online_tkcb10"] = tkcb10(bt, np.random.default_rng([seed, t, 3]))[1]
                rec["ens"], rec["qs"] = {}, {}
                for ten_pp, x in sol.items():
                    for ten_tg, tgw in (("ens", Ec), ("qs", Q)):
                        c = tgw.cham(x)
                        rec[ten_tg][ten_pp] = {k: c[k] for k in SO_DO}
                for ten_tg, tgw in (("cham", Ec), ("cham_qs", Q)):
                    du = np.maximum((tgw.Cw * (1 - tgw.rw)).min(1), 1e-9)
                    rec[ten_tg] = {"song": tgw.song[:, gon.cot], "du": du[:, gon.cot].astype(np.float32)}
            ban_ghi.append(rec)
    tam = str(tep) + ".tam"
    with open(tam, "wb") as f:
        pickle.dump(ban_ghi, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tam, tep)
    return tep, len(ban_ghi), time.time() - t_bd


def _nap_mo_hinh():
    global _mo_hinh
    if _mo_hinh is None:
        from src.nhieu_luong.mo_hinh_dt import ham_chon
        _mo_hinh = ham_chon()


TAI_LIEN_TUC = (0.5, 1.3)
# Thí nghiệm 9 (results/bao_da_tuyen/TIEU_CHI_9.md) — bậc thang độ lạ, mỗi bậc đổi MỘT yếu tố so với bậc trước; mọi tập
# có thầy, 13 cửa sổ, 4–5 luồng. hoc9: đúng tình huống đã học (cửa sổ + seed của train6); seed9: seed mới; hinh9: thêm
# split hình học test; ngay9: ngày mới 2024–2025 (cửa sổ C); nam9 / bao9: thời tiết 2017–2021 (ngày xấu / đợt mạnh).
TAP_9 = {"hoc9": "train", "seed9": "train", "hinh9": "test", "ngay9": "test", "nam9": "test", "bao9": "test"}
NAM_9 = ("nam9", "bao9")
SO_CUA_SO_9 = 13
HINH_HOC = {"val5": "val", "yagi5": "test", "mua5": "test",        # tập của thí nghiệm 5–7 → split hình học v5
            "train6": "train", "val6": "val", "c6": "test", "train7s": "train", "train7x": "train",
            **{f"{b}7{s}": "test" for b in ("c", "y") for s in ("w", "h", "wh")}, **TAP_9}
SEED_6 = {"c6": 720000, "val6": 710000, "train6": 700000, "train7x": 730000}
TAP_TRAIN = {"train", "train6", "train7s", "train7x", "train_b", "train6_b"}
# Thí nghiệm 7a — thế giới cực đoan giả lập (nhãn synthetic_stress, không phải cường độ đo được). Bộ định tuyến không
# biết: bảng rủi ro và mọi phương pháp vẫn dùng cấu hình hợp đồng td._p; chỉ thế giới mô phỏng / chấm bị stress.
STRESS = {"w": (1.5, 1.25, {}), "h": (1.0, 1.0, {"gio_giat_max": 12.0, "p_set": 0.2, "set_tb": 6.0}),
          "wh": (1.5, 1.25, {"gio_giat_max": 12.0, "p_set": 0.2, "set_tb": 6.0})}
TAP_7 = {f"{b}7{s}": ({"c": "c6", "y": "yagi5"}[b], s) for b in ("c", "y") for s in ("w", "h", "wh")}
# Thí nghiệm 8 — "thế giới B" (results/bao_da_tuyen/TIEU_CHI_8.md): THỜI TIẾT THẬT giữ nguyên, chỉ đổi mô hình hỏng hóc.
# Mỗi tập B dùng ĐÚNG cửa sổ và seed của tập gốc (thế giới A) → hai thế giới chỉ khác nhau ở cách hạ tầng hỏng.
TAP_8 = {"train_b": "train", "train6_b": "train6", "val_b": "val", "val6_b": "val6", "c6_b": "c6", "yagi5_b": "yagi5"}
THE_GIOI_B = {"uav_set_pmax": 0.30, "uav_set_L": 15.0, "dat_set_pmax": 0.05, "dat_set_L": 5.0, "set_d_max": 50.0,
              "p_set": 0.0}                         # p_set = 0: bỏ "sét ngắt trạm ~1 phút" cũ, thay bằng hỏng theo khoảng cách
UAV_12 = {"so": 5, "nguong_12": 12.0, "nguong_15": 15.0, "vi_do_bao": 16.0, "gio_bao": 17.2}
# Thí nghiệm 10 (results/bao_da_tuyen/TIEU_CHI_10.md): tên tập → (tập gốc, số thế giới NHÃN, có lưu thế giới CHẤM).
# Cửa sổ, seed, mốc và thế giới mô phỏng như tập gốc (train_b… là thế giới B của TN8). Thầy dùng bộ sinh ngẫu nhiên RIÊNG
# default_rng([seed, t, 6, số thứ tự mức tải]) → tình huống không phụ thuộc thầy: các tập cùng gốc có ĐÚNG cùng tình huống,
# nên thầy 32 / 64 so cặp được và mô hình học trên nhãn thầy 64 chấm được trên thế giới chấm của tập thầy 32.
#   train10, train6_10 (A), train_b10, train6_b10 (B): tập học có thế giới chấm → kiểm thử chéo 4 phần theo cửa sổ
#   *_t64: cùng tình huống, nhãn thầy 64 thế giới (chỉ nhãn); val10_t32 / val10_t64: kiểm tra nhanh thầy 64 trên val
TAP_10 = {"train10": ("train", 32, True), "train6_10": ("train6", 32, True),
          "train_b10": ("train_b", 32, True), "train6_b10": ("train6_b", 32, True),
          "train10_t64": ("train", 64, False), "train6_10_t64": ("train6", 64, False),
          "train_b10_t64": ("train_b", 64, False), "train6_b10_t64": ("train6_b", 64, False),
          "val10_t32": ("val", 32, True), "val10_t64": ("val", 64, True)}
TAP_TRAIN |= {t for t, (_, _, cham) in TAP_10.items() if not cham}          # chỉ nhãn → không lưu thế giới chấm
# Thí nghiệm 11 (results/bao_da_tuyen/TIEU_CHI_11.md): dữ liệu CỰC ĐOAN kiểu đỉnh bão — mưa, gió giật và xác suất sét
# ngắt trạm (p_set) × k, k ~ U(1,3; 1,5) bốc RIÊNG từng cửa sổ (default_rng([SEED_11, thứ tự tập, kb])). Như TN7a, bộ định
# tuyến không biết k (bảng rủi ro dùng tham số hợp đồng td._p) nhưng thấy quan sát mưa / gió đã tăng.
#   x11: tập học (thầy + thế giới chấm, cho kiểm thử chéo) — SO_X11 cửa sổ đỉnh (≈ 5% của 119 cửa sổ train + train6),
#        mỗi cơn bão / ngày tối đa một cửa sổ, seed mới; cùng phần kiểm thử chéo với cửa sổ nguồn.
#   xc11, xy11: bộ kiểm tra — đúng cửa sổ của c6 / yagi5, seed mới, k riêng từng cửa sổ.
TAP_11 = {"x11": "train", "xc11": "c6", "xy11": "yagi5"}
HE_SO_11, SEED_11, SO_X11 = (1.3, 1.5), 1100, 6


def he_so_11(tap, kb):
    return float(np.random.default_rng([SEED_11, list(TAP_11).index(tap), int(kb)]).uniform(*HE_SO_11))


def do_du_doi(tt, h0, so_gio):
    """Độ dữ dội của cửa sổ [h0, h0 + so_gio): đỉnh theo giờ của trung bình các điểm (mưa / 20 + gió giật / 25, mỗi phần
    chặn ở 1, + có giông)."""
    gio = slice(h0, h0 + so_gio)
    x = np.minimum(tt.precip[gio] / 20, 1) + np.minimum(tt.gust[gio] / 25, 1) + np.isin(tt.code[gio], GIONG)
    return float(x.mean(1).max())


def cua_so_x11():
    """[(tên, h0, seed, (tập nguồn, kb nguồn), độ dữ dội)] — SO_X11 cửa sổ dữ dội nhất của train + train6, mỗi nhóm (cơn
    bão ở train, ngày ở train6) tối đa một cửa sổ. Tất định."""
    so_gio = int(np.ceil(td._g.T * td._p.dt_s / 3600)) + 1
    ung = []
    for nguon in ("train", "train6"):
        for kb, (ten, h0, _) in enumerate(cac_cua_so_tap(nguon)):
            nhom = ten if nguon == "train" else ten[:14]                    # 'soulik_2024' / 'xau_2025-05-14'
            ung.append((do_du_doi(td._tt, h0, so_gio), nhom, nguon, kb, ten, h0))
    ra, da = [], set()
    for dd, nhom, nguon, kb, ten, h0 in sorted(ung, key=lambda x: (-x[0], x[2], x[3])):
        if nhom not in da and len(ra) < SO_X11:
            da.add(nhom)
            ra.append((f"cd_{ten}", h0, 1_100_000 + len(ra), (nguon, kb), dd))
    return ra
THOI_TIET_KHI_HAU = GOC / "data" / "thoi_tiet_vn_2017_2021_1.npz"
_moi_truong = {}


def loai_uav(g=None):
    """Thí nghiệm 8: hai loại UAV. VÙNG BÃO = vĩ độ ≥ 16°N (Bắc Bộ, Bắc Trung Bộ — nơi phần lớn bão đổ bộ): mọi UAV dùng
    loại chịu gió 15 m/s. Ngoài vùng bão, 5 UAV có ÍT ngày gió giật ≥ 17,2 m/s (cấp 8) nhất trong khí hậu 2017–2021 dùng
    loại 12 m/s; còn lại 15 m/s. Khí hậu 2017–2021 nằm ngoài thời gian kiểm tra 2024–2025, nên bố trí không nhìn trước dữ
    liệu kiểm tra. (Không xếp theo gió giật thuần: ERA5 cho gió nội đô Hải Phòng thấp quanh năm dù đây là vùng bão đổ bộ.)
    Trả ((tên, ngưỡng), ...) theo thứ tự nút."""
    g = g or td._g
    kh = np.load(THOI_TIET_KHI_HAU, allow_pickle=False)
    ten_diem = [str(s) for s in kh["site_names"]]
    ngay = np.array([str(t)[:10] for t in kh["time_utc"]])
    uav = [g.names[n] for n in range(g.N) if g.kind[n] == "uav"]
    ngoai = []
    for u in uav:
        i = ten_diem.index(u)
        if float(kh["lat"][i]) < UAV_12["vi_do_bao"]:
            ngoai.append((len(set(ngay[kh["gust"][:, i] >= UAV_12["gio_bao"]])), u))
    loai_12 = {u for _, u in sorted(ngoai)[:UAV_12["so"]]}
    return tuple((u, UAV_12["nguong_12"] if u in loai_12 else UAV_12["nguong_15"]) for u in uav)


def moi_truong(tap, kb=None):
    """(thời tiết, tham số) để MÔ PHỎNG thế giới của tập `tap`: thường là td._tt, td._p; tập 7a: mưa × hệ số, gió giật
    × hệ số, UAV / trạm mặt đất dễ hỏng hơn; tập 8 (thế giới B): thời tiết thật, UAV hai loại, hỏng theo khoảng cách
    tới giông; tập 9 năm mới: thời tiết thật 2017–2021, tham số như thế giới A."""
    if tap in NAM_9:
        return thoi_tiet_9(), td._p
    if tap in TAP_11:                                  # thí nghiệm 11: k riêng từng cửa sổ
        if kb is None:
            raise ValueError(f"{tap}: cần kb để bốc hệ số cực đoan")
        if (tap, kb) not in _moi_truong:
            k = he_so_11(tap, kb)
            tt = copy.copy(td._tt)
            tt.precip, tt.gust = td._tt.precip * k, td._tt.gust * k
            _moi_truong[(tap, kb)] = (tt, dataclasses.replace(td._p, p_set=min(1.0, td._p.p_set * k)))
        return _moi_truong[(tap, kb)]
    if tap in TAP_10:
        return moi_truong(TAP_10[tap][0])
    if tap in TAP_8:
        if tap not in _moi_truong:
            _moi_truong[tap] = (td._tt, dataclasses.replace(td._p, hong_b=True, uav_gio_nguong=loai_uav(), **THE_GIOI_B))
        return _moi_truong[tap]
    if tap not in TAP_7:
        return td._tt, td._p
    if tap not in _moi_truong:
        mua, gio, ts = STRESS[TAP_7[tap][1]]
        tt = copy.copy(td._tt)
        tt.precip, tt.gust = td._tt.precip * mua, td._tt.gust * gio
        _moi_truong[tap] = (tt, dataclasses.replace(td._p, **ts))
    return _moi_truong[tap]
def thoi_tiet_9():
    """Thời tiết 2017–2021 (cùng 74 điểm, cùng nguồn mưa / gió Open-Meteo ERA5; giông theo luật CAPE) — thí nghiệm 9."""
    if "_tt9" not in _moi_truong:
        from src.ngau_nhien.thoi_tiet import ThoiTiet
        _moi_truong["_tt9"] = ThoiTiet(str(THOI_TIET_KHI_HAU))
    return _moi_truong["_tt9"]


def tep_thoi_tiet(tap):
    return THOI_TIET_KHI_HAU if tap in NAM_9 else td.THOI_TIET


LOAI_MUA = ("2024-09-03", "2024-09-13")                             # Yagi (loại khỏi bộ mưa lớn mới)


def cua_so_mua_lon(tt, so=13):
    """Thí nghiệm 5, bộ B (luật chốt trong TIEU_CHI_5.md): giờ mưa lớn = ≥ 5% điểm có mưa > 11 mm/h; mỗi ngày ngoài 10
    cơn đã dùng (±1 ngày) và ngoài Yagi lấy cửa sổ (bắt đầu mỗi 3 giờ) có giờ mưa lớn trong [h, h+4] với mức mưa cao
    nhất; xếp ngày theo mức mưa giảm dần, bỏ ngày cách ngày đã chọn ≤ 1 ngày, lấy `so` ngày. Trả h0 theo thời gian."""
    import json
    T = [str(x) for x in tt.time_utc]
    mua = (tt.precip > 11).mean(1) * 100
    ngay = np.array([x[:10] for x in T]).astype("datetime64[D]")
    loai = (ngay >= np.datetime64(LOAI_MUA[0])) & (ngay <= np.datetime64(LOAI_MUA[1]))
    mot = np.timedelta64(1, "D")
    for b in json.loads((GOC / "data" / "bao_manh" / "danh_muc.json").read_text(encoding="utf-8"))["bao"]:
        loai |= (ngay >= np.datetime64(b["start_utc"][:10]) - mot) & (ngay <= np.datetime64(b["end_utc"][:10]) + mot)
    ung = []
    for d in np.unique(ngay):
        idx = np.flatnonzero(ngay == d)
        if loai[idx].any() or idx[-1] + 5 >= len(T):
            continue
        tot = None
        for h in idx[::3]:
            if (mua[h:h + 5] >= 5).any() and not loai[h:h + 5].any() and (tot is None or mua[h:h + 5].max() > tot[1]):
                tot = (int(h), float(mua[h:h + 5].max()))
        if tot:
            ung.append((d, *tot))
    chon = []
    for d, h, s in sorted(ung, key=lambda x: -x[2]):
        if all(abs((d - c[0]).astype(int)) > 1 for c in chon):
            chon.append((d, h, s))
        if len(chon) == so:
            break
    return sorted(h for _, h, _ in chon)


def ngay_xau(tt, loai_them=()):
    """Thí nghiệm 6 (luật chốt trong TIEU_CHI_6.md): ngày ngoài 10 cơn đã dùng (±1), ngoài Yagi và ngoài các ngày của bộ B
    (±1), có giờ với ≥ 3% điểm mưa > 11 mm/h, hoặc ≥ 20% điểm gió giật > 15 m/s, hoặc ≥ 10% điểm có mã giông. Mỗi ngày một
    cửa sổ bắt đầu ở giờ chia hết cho 3 chứa giờ nặng nhất. Xáo bằng default_rng(2026): 13 → c6, 16 → val6, 60 → train6.
    loai_them (thí nghiệm 9, năm 2017–2021): chỉ số giờ của các đợt bão đã lấy — loại thêm ngày của chúng (±1).
    Trả {tập: [(ngày, h0)]}."""
    import json
    from src.ngau_nhien.thoi_tiet import GIONG
    T = [str(x) for x in tt.time_utc]
    mua, gio = (tt.precip > 11).mean(1), (tt.gust > 15).mean(1)
    giong = np.isin(tt.code, GIONG).mean(1)
    xau = (mua >= .03) | (gio >= .20) | (giong >= .10)
    nang = mua + gio + giong
    ngay = np.array([x[:10] for x in T]).astype("datetime64[D]")
    mot = np.timedelta64(1, "D")
    loai = (ngay >= np.datetime64(LOAI_MUA[0])) & (ngay <= np.datetime64(LOAI_MUA[1]))
    for b in json.loads((GOC / "data" / "bao_manh" / "danh_muc.json").read_text(encoding="utf-8"))["bao"]:
        loai |= (ngay >= np.datetime64(b["start_utc"][:10]) - mot) & (ngay <= np.datetime64(b["end_utc"][:10]) + mot)
    for h in [*cua_so_mua_lon(tt), *loai_them]:
        loai |= np.abs((ngay - ngay[h]).astype(int)) <= 1
    ds = []
    for d in np.unique(ngay):
        idx = np.flatnonzero(ngay == d)
        if loai[idx].any() or idx[-1] + 5 >= len(T) or not xau[idx].any():
            continue
        hm = idx[np.argmax(nang[idx])]
        ds.append((str(d), int(max(idx[0], (hm // 3) * 3))))
    perm = np.random.default_rng(2026).permutation(len(ds))
    return {"c6": [ds[i] for i in perm[:13]], "val6": [ds[i] for i in perm[13:29]], "train6": [ds[i] for i in perm[29:89]],
            "train7x": [ds[i] for i in perm[89:149]]}         # thí nghiệm 7b: 60 ngày tiếp theo


def dot_bao_17(tt, so=SO_CUA_SO_9):
    """Thí nghiệm 9, bộ bão năm mới (2017–2021, không có danh mục bão): giờ mạnh theo luật chọn cửa sổ của bao_manh
    (≥ 5% điểm mưa > 11 mm/h hoặc ≥ 20% điểm gió giật > 15 m/s); giờ mạnh cách nhau ≤ 24 giờ gộp thành một đợt; cường
    độ = max theo giờ của max(tỉ lệ mưa / 0,05, tỉ lệ gió / 0,20); lấy `so` đợt mạnh nhất. Trả [(giờ mạnh nhất, các giờ
    mạnh)] theo thời gian."""
    cd = np.maximum((tt.precip > 11).mean(1) / 0.05, (tt.gust > 15).mean(1) / 0.20)
    dot = []
    for h in np.flatnonzero(cd >= 1):
        if dot and h - dot[-1][-1] <= 24:
            dot[-1].append(int(h))
        else:
            dot.append([int(h)])
    dot = [d for d in dot if (max(d) // 3) * 3 + 5 < tt.H]
    dot = sorted(dot, key=lambda d: -cd[d].max())[:so]
    return sorted((d[int(np.argmax(cd[d]))], d) for d in dot)


def cua_so_9(tap):
    """Thí nghiệm 9: [(tên, h0, seed)] của bậc `tap` (xem TAP_9)."""
    if tap in ("hoc9", "seed9", "hinh9"):
        goc = cac_cua_so_tap("train6")
        chon = sorted(np.random.default_rng(2027).choice(len(goc), SO_CUA_SO_9, replace=False))
        them = {"hoc9": 0, "seed9": 5000, "hinh9": 6000}[tap]
        return [(goc[i][0], goc[i][1], goc[i][2] + them) for i in chon]
    if tap == "ngay9":
        return [(ten, h0, seed + 5000) for ten, h0, seed in cac_cua_so_tap("c6")]
    tt = thoi_tiet_9()
    T = [str(x) for x in tt.time_utc]
    dot = dot_bao_17(tt)
    if tap == "bao9":
        return [("bao17_" + T[hm][:10], (hm // 3) * 3, 991000 + i) for i, (hm, _) in enumerate(dot)]
    ds = ngay_xau(tt, loai_them=[h for _, d in dot for h in d])["c6"]       # 13 ngày đầu của hoán vị, như bộ C
    return [("xau17_" + d, h, 990000 + i) for i, (d, h) in enumerate(ds)]


def cac_cua_so_tap(tap):
    """[(tên, h0, seed)] — thêm các tập của thí nghiệm 5, 6 (seed mới) vào các tập cũ."""
    if tap in TAP_9:
        return cua_so_9(tap)
    if tap == "x11":
        return [x[:3] for x in cua_so_x11()]
    if tap in TAP_11:                                  # xc11 / xy11: cửa sổ của bộ kiểm tra gốc, seed mới
        return [(ten, h0, seed + SEED_11) for ten, h0, seed in cac_cua_so_tap(TAP_11[tap])]
    if tap in TAP_10:
        return cac_cua_so_tap(TAP_10[tap][0])
    if tap in TAP_8:                                   # thí nghiệm 8: đúng cửa sổ + seed của tập gốc (thế giới A)
        return cac_cua_so_tap(TAP_8[tap])
    if tap in SEED_6:
        return [("xau_" + d, h, SEED_6[tap] + i) for i, (d, h) in enumerate(ngay_xau(td._tt)[tap])]
    if tap in TAP_7:                                   # 7a: cùng cửa sổ với C / Yagi, seed mới
        return [(ten, h0, seed + 800) for ten, h0, seed in cac_cua_so_tap(TAP_7[tap][0])]
    if tap == "train7s":                               # 7b: cửa sổ bão train, seed mới
        return [(ten, h0, seed + 900) for ten, h0, seed in td.cac_cua_so("train")]
    if tap == "mua5":
        return [("mua_" + td._T[h][:10], h, 600000 + i) for i, h in enumerate(cua_so_mua_lon(td._tt))]
    if tap in ("val5", "yagi5"):
        them = 77 if tap == "val5" else 500
        return [(ten, h0, seed + them) for ten, h0, seed in td.cac_cua_so(HINH_HOC[tap])]
    return td.cac_cua_so(tap)


def thu_muc(tap, K_rg):
    return RA / f"{tap}_k{K_rg[0]}" if K_rg[0] == K_rg[1] else RA / f"{tap}_k{K_rg[0]}_{K_rg[1]}"


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--tap", choices=["train", "val", "test", *HINH_HOC, *TAP_8, *TAP_10, *TAP_11], required=True)
    ap.add_argument("--K", default="4-5")
    ap.add_argument("--buoc-moc", type=int, default=None, help="lấy mỗi n mốc (mặc định như thí nghiệm 3)")
    ap.add_argument("--tien-trinh", type=int, default=3)
    ap.add_argument("--mo-hinh", action="store_true", help="chấm cả DM và softmax (results/bao_da_tuyen/mo_hinh)")
    ap.add_argument("--khong-thay", action="store_true", help="bỏ thầy (chỉ chấm, ví dụ quy mô ngoài tiêu chí)")
    a = ap.parse_args()
    K_rg = phan_tich_K(a.K)
    td._khoi_tao_nhe()
    goc = TAP_10[a.tap][0] if a.tap in TAP_10 else a.tap  # thí nghiệm 10: như tập gốc
    goc = TAP_8.get(goc, goc)                             # thí nghiệm 8: mốc / tải như tập gốc của thế giới A
    goc = TAP_11.get(goc, goc)                            # thí nghiệm 11: mốc / tải như tập gốc
    moc = td.cac_moc(HINH_HOC.get(goc, goc))
    if a.buoc_moc:
        moc = moc[::a.buoc_moc]
    ra = thu_muc(a.tap, K_rg)
    ra.mkdir(parents=True, exist_ok=True)
    if a.khong_thay and a.tap in TAP_TRAIN:
        raise SystemExit("train cần nhãn thầy")
    lien_tuc = goc in HINH_HOC                            # thí nghiệm 5: tải liên tục
    jobs = [(a.tap, K_rg, i, ten, h0, seed, moc, ra / f"kb_{i:04d}.pkl", a.mo_hinh, not a.khong_thay, lien_tuc)
            for i, (ten, h0, seed) in enumerate(cac_cua_so_tap(a.tap))]
    con = [j for j in jobs if not Path(j[7]).exists()]
    print(f"{a.tap} K={a.K}: {len(jobs)} cửa sổ × {len(moc)} mốc, còn {len(con)}", flush=True)
    t0, n = time.time(), 0
    with Pool(a.tien_trinh, initializer=td._khoi_tao) as pool:
        for k, (tep, so, tgian) in enumerate(pool.imap_unordered(_mot_cua_so, con)):
            n += so
            print(f"  {k + 1}/{len(con)}  {Path(tep).name}  {so} bản ghi  {tgian:.0f} s  (tổng {n}, {time.time() - t0:.0f} s)",
                  flush=True)
    (ra / ".xong").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
    print("xong", flush=True)


if __name__ == "__main__":
    main()
