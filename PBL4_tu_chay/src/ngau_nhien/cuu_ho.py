"""
cuu_ho.py — TN6+4 (results/cuu_ho/TIEU_CHI_6.md): 2 UAV cứu hộ đổi chỗ mỗi 15 phút + định tuyến theo score khi bão.

Mạng nền v5 mô phỏng bằng sinh_kich_ban (mức "khắc nghiệt", bật xung đột). UAV cứu hộ KHÔNG nằm trong mạng nền: liên kết
của chúng được dựng thêm (cuu_ho_hinh_hoc.py) và chịu cùng mưa, gây nhiễu, tải, luật hạ cánh như sinh_kich_ban, dùng đúng
các chuỗi ngẫu nhiên của kịch bản (mưa tại điểm thời tiết, gây nhiễu tại điểm, tải của nút mạng nền).

    python -m src.ngau_nhien.cuu_ho --tap val  --out results/cuu_ho/val  --bao-dir <danh mục bão của nhóm> --bang <bang_tra.npz>
    python -m src.ngau_nhien.cuu_ho --tap yagi --out results/cuu_ho/yagi --bang <bang_tra.npz>
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.sparse.csgraph import dijkstra as cs_dijkstra

from src.ngau_nhien.cuu_ho_hinh_hoc import LUAT, LUAT_5GHZ, ecef, hinh_hoc, tre, ung_vien
from src.ngau_nhien.khoang_loi import MUC
from src.ngau_nhien.kich_ban import ALT, C_KM_MS, H_R, PKT, PROC, R_E, MangGoc, ThamSo, _haversine, sinh_kich_ban, \
    suy_hao_mua
from src.ngau_nhien.ma_tran_rui_ro import RAIN_BINS, SHAPE, ma_tran_online, nap_bang
from src.ngau_nhien.thoi_tiet import ThoiTiet
from src.ngau_nhien.thu_trong_so import giai_doan

TUOI, DELTA, CHU_KY, LICH_SU = 9, 3, 45, 45          # 3 phút, 1 phút, 15 phút, 15 phút (ảnh chụp 20 s)
TAM_BAY_KM = 25.0 * CHU_KY * 20 / 1000               # 25 m/s × 15 phút = 22,5 km
ALPHA = {0: 0.8, 1: 0.1}                             # giai đoạn nhẹ / bão
LAMBDA = [0.0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.85, 0.999]
PP_DAT = ("khong", "gan_nut_yeu", "vet_can")
ROUTER = ("quy_dao", "score")
BANG_TAN = np.array([0, 5, 12, 20, 28])
YAGI = [f"2024-09-0{7 + i * 3 // 24}T{i * 3 % 24:02d}:00" for i in range(16)]
VAL_BAO = ("prapiroon_2024", "trami_2024", "kalmaegi_2025")


class CuuHo:
    """Dữ liệu tĩnh của các ứng viên (không phụ thuộc kịch bản)."""

    def __init__(self, raw, tt0, luat=None):
        self.luat = LUAT if luat is None else luat
        self.kind = np.array([str(k) for k in raw["node_kind"]])
        self.N = len(self.kind)
        self.lat, self.lon, self.ten = ung_vien(raw)
        self.K = len(self.lat)
        self.G, self.C = hinh_hoc(raw, self.lat, self.lon, self.luat)                      # [T,K,N], [K,N]
        self.T = self.G.shape[0]
        self.site = np.array([int(np.argmin(_haversine(la, lo, tt0.lat, tt0.lon))) for la, lo in zip(self.lat, self.lon)])
        self.kc = _haversine(self.lat[:, None], self.lon[:, None], self.lat[None], self.lon[None])  # [K,K] km
        self.Pc = ecef(self.lat, self.lon, np.full(self.K, 2.5))
        X0 = raw["node_latlon_alt"][0]
        khac = np.flatnonzero(self.kind != "leo")
        # tải của UAV cứu hộ = tải của nút mạng nền (không phải vệ tinh) gần nhất: cùng nhiễu, cùng đợt tăng tải do sự cố
        self.gan = np.array([khac[np.argmin(_haversine(la, lo, X0[khac, 0], X0[khac, 1]))] for la, lo in zip(self.lat, self.lon)])
        k = self.kind
        self.band = np.where(k == "leo", 12, 5).astype(int)                          # UAV–LEO 12 GHz, còn lại 5 GHz
        self.band[k == "haps"] = 5
        self.dat = np.isin(k, ["user", "gateway"])
        self.uav = k == "uav"
        h_v = np.array([ALT.get(x, 0.0) for x in k])
        self.h1, self.h2 = np.minimum(ALT["uav"], h_v), np.maximum(ALT["uav"], h_v)
        self.co_dinh = PKT / self.C * (1 + 0.3 / 0.7) + PROC                        # [K,N]

    def ls_lg(self, prop):
        """Ls, LG như MangGoc, từ trễ lan truyền prop [..., N]."""
        d = prop * C_KM_MS
        r1, r2 = R_E + self.h1, R_E + self.h2
        with np.errstate(divide="ignore", invalid="ignore"):
            s = (r2 ** 2 - r1 ** 2 - d ** 2) / (2 * r1 * d)
        sin_el = np.clip(np.nan_to_num(s, nan=1.0), np.sin(np.radians(1.0)), 1.0)
        thap = self.h2 < H_R
        Ls = np.where(thap, d, np.maximum(H_R - self.h1, 0) / sin_el)
        LG = np.where(thap, d, Ls * np.sqrt(1 - sin_el ** 2))
        return Ls, LG


class KichBan:
    """Một kịch bản: mạng nền đã mô phỏng + cách tính liên kết UAV cứu hộ thực và dự báo."""

    def __init__(self, ch: CuuHo, g, tt, Wn, q, p, node_site):
        self.ch, self.g, self.tt, self.Wn, self.q, self.p = ch, g, tt, Wn, q, p
        self.ns = node_site
        self.ha_canh = q["giong"][:, ch.site] | (q["gust"][:, ch.site] > p.gio_giat_max)      # [T,K]
        self.nhip = 1 + p.bien_do_ngay * np.cos(2 * np.pi * (q["gio_vn"] - p.gio_dinh) / 24)  # [T]

    def site_lk(self, c, chieu):
        """Điểm thời tiết quyết định mưa trên liên kết c→v (chieu 0) hoặc v→c (chieu 1), như MangGoc."""
        ch = self.ch
        s = np.full(ch.N, ch.site[c])
        s[ch.dat] = self.ns[ch.dat]                         # có đầu mặt đất: lấy điểm của đầu mặt đất
        if chieu == 1:
            s[ch.uav] = self.ns[ch.uav]                     # UAV–UAV: lấy nút đứng trước trong cặp (u, v)
        return s

    def thuc(self, c, ts):
        """Trễ thực (−1 = đứt) của liên kết c→v và v→c ở các ảnh chụp ts: [len(ts), N] × 2."""
        ch, p, q = self.ch, self.p, self.q
        ts = np.asarray(ts)
        Gt = ch.G[ts, c].astype(np.float64)
        co = Gt > 0
        prop = np.where(co, Gt - ch.co_dinh[c], 0.0)
        Ls, LG = ch.ls_lg(prop)
        C = ch.C[c].astype(np.float64)
        ra = []
        for chieu in (0, 1):
            s = self.site_lk(c, chieu)
            R = q["R"][ts][:, s]
            A = suy_hao_mua(R, ch.band[None], Ls, LG)
            J = q["J"][ts][:, s]
            snr = p.snr0_db - A - J
            dut = snr < p.snr_min_db
            C2 = C * np.log2(1 + 10 ** (snr / 10)) / np.log2(1 + 10 ** (p.snr0_db / 10))
            C2 = np.where(dut, C, np.maximum(C2, 1e-3))
            L = q["tai"][ts][:, [ch.gan[c]]] if chieu == 0 else q["tai"][ts]
            rho = p.rho_tb * L * C / C2
            with np.errstate(divide="ignore"):
                hs = np.where(rho < 1, np.minimum(rho / (1 - rho), p.k_dem), p.k_dem)
            W = prop + PKT / C2 + PKT / C2 * hs + PROC
            song = co & ~dut & ~self.ha_canh[ts, c][:, None] & ~q["tat"][ts]
            ra.append(np.where(song, W, -1.0))
        return ra

    def du_bao(self, cs, t, o, o_tai):
        """W dự báo (0 = không dùng) c→v, v→c cho các ứng viên cs, quyết định ở t, quan sát ở o.
        o_tai[i]: UAV có đang ở cs[i] lúc o không (có số đo thì dùng hệ số đo như du_bao_quy_dao, không thì hệ số 1)."""
        ch = self.ch
        cs = np.asarray(cs)
        Gw = ch.G[t:t + DELTA + 1][:, cs]                                                 # [Δ+1, k, N]
        hop = (Gw > 0).all(0) & ~self.ha_canh[o, cs][:, None] & ~self.q["tat"][o][None]
        tb = ch.G[t + 1:t + DELTA + 1][:, cs].mean(0)
        f_cv, f_vc = np.ones(hop.shape), np.ones(hop.shape)
        for i, c in enumerate(cs):
            if o_tai[i]:
                g0 = ch.G[o, c]
                wcv, wvc = self.thuc(c, [o])
                with np.errstate(divide="ignore", invalid="ignore"):
                    f_cv[i] = np.where(g0 > 0, np.maximum(wcv[0], 0) / g0, 1.0)
                    f_vc[i] = np.where(g0 > 0, np.maximum(wvc[0], 0) / g0, 1.0)
        return np.where(hop, tb * f_cv, 0.0), np.where(hop, tb * f_vc, 0.0)

    def o_rui_ro(self, cs, o):
        """Ô của bảng tra cho liên kết c→v, v→c (cùng đặc trưng như o_dac_trung của nhóm)."""
        ch, q = self.ch, self.q
        ra = []
        for chieu in (0, 1):
            o_list = []
            for c in cs:
                s = self.site_lk(c, chieu)
                mua = q["R"][o, s]
                xu = mua - q["R"][o - LICH_SU, s]
                xu = np.where(xu < -1, 0, np.where(xu > 1, 2, 1))
                gc = q["gust"][o, ch.site[c]]
                gv = np.where(ch.uav, q["gust"][o, self.ns], 0.0)
                gio = np.maximum(gc, gv)
                gb = np.where(gio < 10, 1, np.where(gio < 13, 2, np.where(gio <= 15, 3, 4)))
                tc = q["giong"][o, ch.site[c]]
                tu = (tc | (q["giong"][o, self.ns] & ch.uav)).astype(int)
                tg = (q["giong"][o, self.ns] & ch.dat).astype(int)
                b = np.searchsorted(BANG_TAN, ch.band)
                o_list.append(np.ravel_multi_index((b, np.digitize(mua, RAIN_BINS), xu, gb, tu, tg), SHAPE))
            ra.append(np.array(o_list))
        return ra


def dijkstra_duong(C, s, d, chan):
    """Đường ngắn nhất s→d trên ma trận chi phí C (inf = không có cạnh); chan = nút không được làm trung gian."""
    C = C.copy()
    b = chan.copy()
    b[d] = False
    C[:, b] = np.inf
    dist, pred = cs_dijkstra(C, directed=True, indices=s, return_predecessors=True)
    if not np.isfinite(dist[d]):
        return None
    r, v = [], d
    while v >= 0:
        r.append(int(v))
        v = pred[v]
    return r[::-1]


def chon_score(Wf, rr, s, d, chan, alpha):
    """Tuyến nhanh nhất (quy_dao) và tuyến theo score S (ratio: chậm hơn tuyến nhanh nhất, xác suất đứt)."""
    Cw = np.where(Wf > 0, Wf, np.inf)
    r0 = dijkstra_duong(Cw, s, d, chan)
    if r0 is None:
        return None, None
    e0 = (np.array(r0[:-1]), np.array(r0[1:]))
    Ds = Wf[e0].sum()
    lg = np.where(Wf > 0, -np.log(np.clip(1 - rr, 1e-9, 1)), np.inf)
    uv = [r0]
    for lam in LAMBDA[1:]:
        r = dijkstra_duong((1 - lam) * Cw / Ds + lam * lg, s, d, chan)
        if r is not None and r not in uv:
            uv.append(r)
    D = np.array([Wf[np.array(r[:-1]), np.array(r[1:])].sum() for r in uv])
    P = np.array([1 - np.prod(1 - rr[np.array(r[:-1]), np.array(r[1:])]) for r in uv])
    S = alpha * (D / D.min() - 1) + (1 - alpha) * P
    return r0, uv[int(np.argmin(S))]


def phi(Wf, rr, chan, leo, w):
    """Φ̂ = Σ w_e · P(nút e nối được tới vệ tinh), P = tuyến sống tốt nhất theo bảng tra (Dijkstra trên −log p)."""
    n = len(Wf)
    M = np.full((n + 1, n + 1), np.inf)
    M[:n, :n] = np.where(Wf > 0, -np.log(np.clip(1 - rr, 1e-9, 1)) + 1e-6, np.inf)
    M[:n, :n][:, chan] = np.inf                       # nút không được làm trung gian: chặn cạnh đi vào
    M[leo, n] = 1e-6  # scipy coi cạnh < ~1e-8 là không có cạnh
    dist = cs_dijkstra(M.T, directed=True, indices=n)
    return float((w * np.exp(-dist[:len(w)])).sum())


def chay_kich_ban(ch, raw, tt, h0, seed, ids, bang, cap, w_e, ten_kb):
    p = ThamSo(**MUC["khắc nghiệt"])
    names, kinds = raw["node_order"], raw["node_kind"]
    g = MangGoc(raw["W_snapshots"], kinds, names, tt)
    Wn, q = sinh_kich_ban(g, tt, h0, seed, p)
    kb = KichBan(ch, g, tt, Wn, q, p, g.node_site)
    N, K = ch.N, ch.K
    chan = np.r_[ch.kind == "user", False, False]
    leo = np.flatnonzero(ch.kind == "leo")
    toa = raw["node_latlon_alt"]
    vi_tri = {m: [None, None] for m in PP_DAT}
    rec = {k: [] for k in ("t", "s", "d", "gd")}
    for m in PP_DAT:
        for r in ROUTER:
            rec[f"{m}|{r}|phuc_vu"], rec[f"{m}|{r}|tre"] = [], []
        rec[f"{m}|hau_kiem"] = []
    nhat_ky = []

    def ghep(base_W, base_r, cs, t, o, o_tai):
        """Ma trận dự báo + rủi ro đã ghép 2 ô UAV cứu hộ (ô trống = không có cạnh)."""
        W = np.zeros((N + 2, N + 2))
        R = np.full((N + 2, N + 2), np.nan)
        W[:N, :N], R[:N, :N] = base_W, base_r
        thuc = [i for i, c in enumerate(cs) if c is not None]
        if thuc:
            cc = [cs[i] for i in thuc]
            wcv, wvc = kb.du_bao(cc, t, o, [o_tai[i] for i in thuc])
            ocv, ovc = kb.o_rui_ro(cc, o)
            for j, i in enumerate(thuc):
                W[N + i, :N], W[:N, N + i] = wcv[j], wvc[j]
                R[N + i, :N] = np.where(wcv[j] > 0, 1 - bang["p_song"][ocv[j]], np.nan)
                R[:N, N + i] = np.where(wvc[j] > 0, 1 - bang["p_song"][ovc[j]], np.nan)
            if len(cc) == 2 and ch.kc[cc[0], cc[1]] <= ch.luat["uav"][0] and not kb.ha_canh[o, cc].any():
                d = np.linalg.norm(ch.Pc[cc[0]] - ch.Pc[cc[1]])
                W[N, N + 1] = W[N + 1, N] = tre(d, ch.luat["uav"][2])
                # UAV cứu hộ – UAV cứu hộ: băng 5 GHz, mưa và xu hướng ở điểm của UAV thứ nhất, gió/giông của cả hai
                s0 = ch.site[cc[0]]
                mua, xu = q["R"][o, s0], q["R"][o, s0] - q["R"][o - LICH_SU, s0]
                gmax = q["gust"][o, ch.site[cc]].max()
                o_cc = np.ravel_multi_index((1, int(np.digitize(mua, RAIN_BINS)), 0 if xu < -1 else (2 if xu > 1 else 1),
                                             1 if gmax < 10 else (2 if gmax < 13 else (3 if gmax <= 15 else 4)),
                                             int(q["giong"][o, ch.site[cc]].any()), 0), SHAPE)
                R[N, N + 1] = R[N + 1, N] = 1 - bang["p_song"][o_cc]
        return W, R

    def thuc_ghep(cs, ts):
        """Trễ thực đã ghép UAV cứu hộ ở các ảnh chụp ts: [len(ts), N+2, N+2] (−1 = đứt)."""
        W = -np.ones((len(ts), N + 2, N + 2))
        W[:, :N, :N] = Wn[ts]
        thuc = [(i, c) for i, c in enumerate(cs) if c is not None]
        for i, c in thuc:
            wcv, wvc = kb.thuc(c, ts)
            W[:, N + i, :N], W[:, :N, N + i] = wcv, wvc
        if len(thuc) == 2 and ch.kc[cs[0], cs[1]] <= ch.luat["uav"][0]:
            song = ~kb.ha_canh[ts][:, cs].any(1)
            d = np.linalg.norm(ch.Pc[cs[0]] - ch.Pc[cs[1]])
            W[:, N, N + 1] = W[:, N + 1, N] = np.where(song, tre(d, ch.luat["uav"][2]), -1.0)
        return W

    cac_moc = list(range(TUOI + LICH_SU, ch.T - DELTA, CHU_KY))
    for tau in cac_moc:
        o = tau - TUOI
        past = {k: q[k][:o + 1] for k in ("R", "gust", "giong", "tat")}
        on = ma_tran_online(bang, g, Wn[o], past, tau, TUOI, DELTA)
        bW, bR = on["W_forecast"], on["risk_edge"]
        # ---- tầng chậm: đặt 2 UAV cứu hộ
        bay_duoc = ~kb.ha_canh[o]
        vi_tri_cu = {m: list(v) for m, v in vi_tri.items()}
        P0W, P0R = ghep(bW, bR, [None, None], tau, o, [False, False])
        for m in PP_DAT:
            cu = vi_tri[m]
            if m == "khong":
                continue
            tam = [np.ones(K, bool) if c is None else ch.kc[c] <= TAM_BAY_KM for c in cu]
            if m == "gan_nut_yeu":
                n = N + 2
                M = np.full((n + 1, n + 1), np.inf)
                M[:n, :n] = np.where(P0W > 0, -np.log(np.clip(1 - P0R, 1e-9, 1)) + 1e-6, np.inf)
                M[:n, :n][:, chan] = np.inf
                M[leo, n] = 1e-6  # scipy coi cạnh < ~1e-8 là không có cạnh
                Pe = np.exp(-cs_dijkstra(M.T, directed=True, indices=n)[:N])
                dich = np.argsort(-(w_e * (1 - Pe)))
                moi = list(cu)
                dung = set()
                wcv_all, _ = kb.du_bao(np.arange(K), tau, o, [False] * K)
                for i in range(2):
                    for e in dich[:20]:
                        if w_e[e] == 0 or e in dung:
                            continue
                        ok = np.flatnonzero(tam[i] & bay_duoc & (wcv_all[:, e] > 0))
                        ok = [c for c in ok if c not in moi[:i]]
                        if ok:
                            moi[i] = int(min(ok, key=lambda c: _haversine(ch.lat[c], ch.lon[c], toa[o, e, 0], toa[o, e, 1])))
                            dung.add(e)
                            break
                vi_tri[m] = moi
            else:   # vet_can
                f = lambda cs: phi(*ghep(bW, bR, list(cs), tau, o, [cs[i] is not None and cs[i] == cu[i] for i in range(2)]),
                                   chan, leo, w_e)
                if cu[0] is None:
                    ung = np.flatnonzero(bay_duoc)
                    c1 = max(ung, key=lambda c: f([int(c), None]))
                    c2 = max([c for c in ung if c != c1], key=lambda c: f([int(c1), int(c)]))
                    vi_tri[m] = [int(c1), int(c2)]
                else:
                    R1 = [int(c) for c in np.flatnonzero(tam[0] & bay_duoc)] or [cu[0]]
                    R2 = [int(c) for c in np.flatnonzero(tam[1] & bay_duoc)] or [cu[1]]
                    cap_xet = [(a, b) for a in R1 for b in R2 if a != b]
                    # hoà Φ̂ thì ưu tiên bay ít nhất (đứng yên nếu không có gì để cứu)
                    vi_tri[m] = list(max(cap_xet, key=lambda cs: (round(f(cs), 9), -ch.kc[cs[0], cu[0]] - ch.kc[cs[1], cu[1]])))
            nhat_ky.append((tau, m, list(vi_tri[m])))
        # ---- tầng nhanh: định tuyến các truy vấn trong 15 phút tới
        for t in [x for x in ids if tau <= x < tau + CHU_KY and x + DELTA < ch.T]:
            ot = t - TUOI
            past = {k: q[k][:ot + 1] for k in ("R", "gust", "giong", "tat")}
            on = ma_tran_online(bang, g, Wn[ot], past, t, TUOI, DELTA)
            ts = np.arange(t, t + DELTA + 1)
            gds = {}
            for s, d in cap.get(t, []):
                gds[(s, d)] = giai_doan(tt, g, h0, ot, CHU_KY, toa[ot], s, d)
                rec["t"].append(t); rec["s"].append(s); rec["d"].append(d); rec["gd"].append(gds[(s, d)])
            for m in PP_DAT:
                cs = vi_tri[m]
                # có số đo của UAV cứu hộ ở vị trí hiện tại lúc quan sát ot không (đã tới chỗ mới, hoặc đứng yên)
                tai = [c is not None and (ot >= tau or vi_tri_cu[m][i] == c) for i, c in enumerate(cs)]
                Wf, Rf = ghep(on["W_forecast"], on["risk_edge"], cs, t, ot, tai)
                Wr = thuc_ghep(cs, ts)
                song_cs = (Wr > 0).all(0)
                Wh = np.where(song_cs, Wr.mean(0), np.inf)
                for s, d in cap.get(t, []):
                    r0, rs = chon_score(Wf, Rf, s, d, chan, ALPHA[gds[(s, d)]])
                    for ro, r in (("quy_dao", r0), ("score", rs)):
                        if r is None:
                            rec[f"{m}|{ro}|phuc_vu"].append(0); rec[f"{m}|{ro}|tre"].append(np.nan)
                            continue
                        e = (np.array(r[:-1]), np.array(r[1:]))
                        v = Wr[:, e[0], e[1]]
                        ok = bool((v > 0).all())
                        rec[f"{m}|{ro}|phuc_vu"].append(int(ok))
                        rec[f"{m}|{ro}|tre"].append(float(v.sum(1).mean()) if ok else np.nan)
                    rec[f"{m}|hau_kiem"].append(int(dijkstra_duong(Wh, s, d, chan) is not None))
    out = {k: np.array(v) for k, v in rec.items()}
    out["nhat_ky"] = np.array(json.dumps([(int(a), b, [None if x is None else int(x) for x in c]) for a, b, c in nhat_ky]))
    out["kich_ban"] = np.array(ten_kb)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tap", choices=["val", "yagi"], required=True)
    ap.add_argument("--raw", default="dataset/oracle_dataset_v5.npz")
    ap.add_argument("--split", default="dataset/v5/split_info.json")
    ap.add_argument("--thoi-tiet", default="data/thoi_tiet_vn_mr.npz")
    ap.add_argument("--bao-dir", default=None, help="thư mục danh mục bão của nhóm (bao_manh.py --out), cho tap val")
    ap.add_argument("--bang", required=True, help="bang_tra.npz của ma_tran_rui_ro.py")
    ap.add_argument("--chi", type=int, default=None, help="chỉ chạy n kịch bản đầu (thử)")
    ap.add_argument("--duong-len-5ghz", action="store_true", help="phiên bản 2, hướng 2: đường lên 5 GHz tầm xa (LUAT_5GHZ)")
    ap.add_argument("--chi-mat-dat", action="store_true", help="phiên bản 2, hướng 3: chỉ truy vấn có cả hai đầu là user/gateway")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    raw = np.load(a.raw)
    kind_all = np.array([str(k) for k in raw["node_kind"]])
    tt0 = ThoiTiet(a.thoi_tiet)
    bang = nap_bang(a.bang)
    split = json.load(open(a.split, encoding="utf-8"))["snapshot_ids"]
    ids = sorted(split["val" if a.tap == "val" else "test"])
    cap = {}
    for sid, row in zip(raw["snapshot_id"], raw["y_paths_indices"]):
        if int(sid) in ids:
            r = [int(x) for x in row if x >= 0]
            if a.chi_mat_dat and not (kind_all[r[0]] in ("user", "gateway") and kind_all[r[-1]] in ("user", "gateway")):
                continue
            cap.setdefault(int(sid), []).append((r[0], r[-1]))
    dem = np.zeros(len(raw["node_kind"]))
    for row in raw["y_paths_indices"]:
        r = [int(x) for x in row if x >= 0]
        dem[r[0]] += 1; dem[r[-1]] += 1
    if a.chi_mat_dat:
        dem[~np.isin(kind_all, ["user", "gateway"])] = 0
    w_e = dem / dem.sum()
    t0 = time.time()
    ch = CuuHo(raw, tt0, LUAT_5GHZ if a.duong_len_5ghz else None)
    print(f"{ch.K} vị trí ứng viên, hình học {time.time() - t0:.0f}s", flush=True)
    kbs = []
    if a.tap == "yagi":
        T = [str(x) for x in tt0.time_utc]
        kbs = [(f"yagi_{i:02d}", tt0, T.index(s), 900000 + i) for i, s in enumerate(YAGI)]
    else:
        dm = json.load(open(Path(a.bao_dir) / "danh_muc.json", encoding="utf-8"))["bao"]
        for j, b in enumerate([x for x in dm if x["ten"] in VAL_BAO]):
            tt = ThoiTiet(str(Path(a.bao_dir) / b["tep"]))
            assert list(tt.names) == list(tt0.names)
            kbs += [(f"{b['ten']}_{k:02d}", tt, h0, 800000 + 100 * j + k) for k, h0 in enumerate(b["gio_bat_dau_kich_ban"])]
    meta = {"tap": a.tap, "so_ung_vien": ch.K, "luat": ch.luat, "chi_mat_dat": a.chi_mat_dat, "tam_bay_km": TAM_BAY_KM, "alpha": ALPHA, "lambda": LAMBDA,
            "kich_ban": [(k, int(h), int(s)) for k, _, h, s in kbs], "bang": a.bang}
    json.dump(meta, open(out / "meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    for ten, tt, h0, seed in kbs[:a.chi]:
        f = out / f"{ten}.npz"
        if f.exists():
            continue
        t1 = time.time()
        r = chay_kich_ban(ch, raw, tt, h0, seed, ids, bang, cap, w_e, ten)
        np.savez_compressed(f, **r)
        pv = {k.split("|")[0] + "|" + k.split("|")[1]: float(np.mean(v)) for k, v in r.items() if k.endswith("phuc_vu")}
        print(f"{ten}: {len(r['t'])} truy vấn, {time.time() - t1:.0f}s | " +
              " ".join(f"{k} {v * 100:.1f}%" for k, v in pv.items()) +
              " | hậu kiểm " + " ".join(f"{m} {r[m + '|hau_kiem'].mean() * 100:.1f}%" for m in PP_DAT),
              flush=True)
    print("XONG_CUU_HO", flush=True)


if __name__ == "__main__":
    main()
