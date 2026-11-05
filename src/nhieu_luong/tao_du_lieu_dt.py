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
_CAPE = None


def cape_luc(h0, o):
    """[S] CAPE (J/kg) của giờ chứa mốc o — giờ hiện tại, không phải giờ tới."""
    global _CAPE
    if _CAPE is None:
        with np.load(td.THOI_TIET, allow_pickle=False) as z:
            _CAPE = z["cape"].astype(np.float32)
    return _CAPE[h0 + int(o * 20 / 3600)]


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
    tt, p_sim = moi_truong(tap)            # thế giới mô phỏng (thí nghiệm 7a: đã stress)
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
        sn, sc = _seeds(seed, t, 0, SO_TG_NHAN), _seeds(seed, t, 1, SO_TG_CHAM)
        cac_muc = (np.random.default_rng([seed, t, 5]).uniform(*TAI_LIEN_TUC, 3).tolist() if lien_tuc
                   else list(MUC_TAI))                    # thí nghiệm 5: 3 hệ số tải bốc đều, cùng luồng
        for muc in cac_muc:
            lam = th["lam"] * (MUC_TAI[muc] if isinstance(muc, str) else muc) / th["u"]
            s = time.perf_counter()
            DP = dt.dijkstra_phat(th, lam, *BETA_GAMMA)
            tg_dp = time.perf_counter() - s
            bt = dt.BaiToan(th, muc, [Y[k] + DP[k] for k in range(len(lam))])
            gon = dt.BaiToanGon(bt)
            gon.dac_trung_canh = dac_trung_canh(g, q, mg, gon.U, o, cape_luc(h0, o))
            gon.dac_trung_uv = [dac_trung_uv(g, th, k) for k in range(bt.K)]
            En = dt.TheGioi.ensemble(bt, g, pl, p_sim, sn) if co_thay else None
            s = time.perf_counter()
            B = dt.TheGioi.tu_bang(bt, np.random.default_rng([seed, t, 2]))
            a_on, _ = dt.tkcb(B, dt.tham_lam(B))
            tg_on = time.perf_counter() - s
            sol = {"yen": bt.loi_giai(Y), "suurballe": bt.loi_giai(S), "dphat": bt.loi_giai(DP),
                   "online_tkcb": bt.tu_lua_chon(a_on)}
            dau = [a_on] + [a for a in (anh_xa(bt, sol["suurballe"]), anh_xa(bt, sol["dphat"])) if a is not None]
            a_thay, J_thay = thay(bt, En, dau, rng) if co_thay else (None, float("nan"))
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
HINH_HOC = {"val5": "val", "yagi5": "test", "mua5": "test",        # tập của thí nghiệm 5–7 → split hình học v5
            "train6": "train", "val6": "val", "c6": "test", "train7s": "train", "train7x": "train",
            **{f"{b}7{s}": "test" for b in ("c", "y") for s in ("w", "h", "wh")}}
SEED_6 = {"c6": 720000, "val6": 710000, "train6": 700000, "train7x": 730000}
TAP_TRAIN = {"train", "train6", "train7s", "train7x"}
# Thí nghiệm 7a — thế giới cực đoan giả lập (nhãn synthetic_stress, không phải cường độ đo được). Bộ định tuyến không
# biết: bảng rủi ro và mọi phương pháp vẫn dùng cấu hình hợp đồng td._p; chỉ thế giới mô phỏng / chấm bị stress.
STRESS = {"w": (1.5, 1.25, {}), "h": (1.0, 1.0, {"gio_giat_max": 12.0, "p_set": 0.2, "set_tb": 6.0}),
          "wh": (1.5, 1.25, {"gio_giat_max": 12.0, "p_set": 0.2, "set_tb": 6.0})}
TAP_7 = {f"{b}7{s}": ({"c": "c6", "y": "yagi5"}[b], s) for b in ("c", "y") for s in ("w", "h", "wh")}
_moi_truong = {}


def moi_truong(tap):
    """(thời tiết, tham số) để MÔ PHỎNG thế giới của tập `tap`: thường là td._tt, td._p; tập 7a: mưa × hệ số, gió giật
    × hệ số, UAV / trạm mặt đất dễ hỏng hơn."""
    if tap not in TAP_7:
        return td._tt, td._p
    if tap not in _moi_truong:
        mua, gio, ts = STRESS[TAP_7[tap][1]]
        tt = copy.copy(td._tt)
        tt.precip, tt.gust = td._tt.precip * mua, td._tt.gust * gio
        _moi_truong[tap] = (tt, dataclasses.replace(td._p, **ts))
    return _moi_truong[tap]
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


def ngay_xau(tt):
    """Thí nghiệm 6 (luật chốt trong TIEU_CHI_6.md): ngày ngoài 10 cơn đã dùng (±1), ngoài Yagi và ngoài các ngày của bộ B
    (±1), có giờ với ≥ 3% điểm mưa > 11 mm/h, hoặc ≥ 20% điểm gió giật > 15 m/s, hoặc ≥ 10% điểm có mã giông. Mỗi ngày một
    cửa sổ bắt đầu ở giờ chia hết cho 3 chứa giờ nặng nhất. Xáo bằng default_rng(2026): 13 → c6, 16 → val6, 60 → train6.
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
    for h in cua_so_mua_lon(tt):
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


def cac_cua_so_tap(tap):
    """[(tên, h0, seed)] — thêm các tập của thí nghiệm 5, 6 (seed mới) vào các tập cũ."""
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
    ap.add_argument("--tap", choices=["train", "val", "test", *HINH_HOC], required=True)
    ap.add_argument("--K", default="4-5")
    ap.add_argument("--buoc-moc", type=int, default=None, help="lấy mỗi n mốc (mặc định như thí nghiệm 3)")
    ap.add_argument("--tien-trinh", type=int, default=3)
    ap.add_argument("--mo-hinh", action="store_true", help="chấm cả DM và softmax (results/bao_da_tuyen/mo_hinh)")
    ap.add_argument("--khong-thay", action="store_true", help="bỏ thầy (chỉ chấm, ví dụ quy mô ngoài tiêu chí)")
    a = ap.parse_args()
    K_rg = phan_tich_K(a.K)
    td._khoi_tao_nhe()
    moc = td.cac_moc(HINH_HOC.get(a.tap, a.tap))
    if a.buoc_moc:
        moc = moc[::a.buoc_moc]
    ra = thu_muc(a.tap, K_rg)
    ra.mkdir(parents=True, exist_ok=True)
    if a.khong_thay and a.tap in TAP_TRAIN:
        raise SystemExit("train cần nhãn thầy")
    lien_tuc = a.tap in HINH_HOC                          # thí nghiệm 5: tải liên tục
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
