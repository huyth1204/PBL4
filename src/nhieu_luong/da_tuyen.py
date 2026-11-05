"""
da_tuyen.py — Thí nghiệm 4: mỗi luồng một TẬP tuyến (tuyến chính + dự phòng) lúc bão trên mạng SAGSIN v5
(xem results/bao_da_tuyen/TIEU_CHI_4.md).

  kho tuyến     ≤ 12 ứng viên mỗi cặp nguồn – đích: m tuyến Suurballe, tuyến nhanh nhất của từng loại đường lên,
                Yen theo trễ và theo rủi ro (hành lang như thí nghiệm 3; tuyến Suurballe luôn được giữ)
  số tuyến m    min(3, số đường rời cạnh s → d, số loại đường lên của nguồn + 1)
  mất dịch vụ   trong một thế giới: luồng có tuyến chính đứt chuyển sang dự phòng đầu tiên còn sống (mọi luồng chuyển
                đồng thời); luồng mất dịch vụ nếu (a) không còn tuyến sống, hoặc (b) tuyến đang dùng có liên kết ρ ≥ 1 ở
                bất kỳ mốc nào trong [t, t+3] — kể cả luồng không chuyển nhưng bị luồng khác chuyển sang làm quá tải
  J             Σ λ_k/Σλ · (0,7·Ĥ_k + 0,3·T̂_k); Ĥ = P(mất dịch vụ) chuẩn hoá min–max trên các tập của luồng khi chỉ có nó
                (sàn 0,01); T̂ = tốc độ tuyến chính có tải mọi tuyến chính (chuẩn hoá như thí nghiệm 3)
Thế giới: ensemble rẽ nhánh tại o (the_gioi_rr) để làm nhãn / chấm; bốc độc lập theo bảng rủi ro để ước lượng online.
"""

from __future__ import annotations

import itertools

import networkx as nx
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import maximum_flow

from src.ngau_nhien.kich_ban import PKT, PROC, loai_lien_ket
from src.nhieu_luong import the_gioi_rr as tg
from src.nhieu_luong.bai_toan import MUC_TAI, diem_bao, h_hang_doi
from src.nhieu_luong.bai_toan_rr import DELTA, HANH_LANG, SAN_D, TUOI, W_RR, DoThi, yen

SO_KHO, M_MAX, SAN_H = 12, 3, 0.01
SO_TG, SO_TG_BANG = 16, 32               # thế giới ensemble (nhãn / chấm) và thế giới bốc từ bảng (online)
BETA = (0.0, 2.0, 10.0, 50.0)            # ms trên một đơn vị −ln(1 − rủi ro)
GAMMA = (0.0, 2.0, 10.0, 50.0)           # ms phạt mỗi cạnh chung nhóm rủi ro
CAM = 1e9                                # trọng số cạnh hết dung lượng dự phòng


def tre(prop, C, rho0, F):
    return prop + PKT / C * (1 + h_hang_doi(rho0 + F / C)) + PROC


# ---------------------------------------------------------------- mạng lúc quyết định
class Mang:
    """Cạnh dùng được ở t (như thí nghiệm 3) + thông tin online quan sát ở o, trên mọi cạnh."""

    def __init__(self, g, q, r_online, t):
        o = t - TUOI
        C, rho0 = q["C2"][o].astype(float), q["rho"][o].astype(float)
        valid = r_online["valid"] & (rho0 < 1.0)
        np.fill_diagonal(valid, False)
        self.g, self.t, self.o = g, t, o
        self.eu, self.ev = np.nonzero(valid)
        self.can_relay = np.array([k != "user" for k in g.kind])
        self.dt = DoThi(g.N, self.can_relay, self.eu, self.ev)
        self.risk = np.clip(r_online["risk_edge"][self.eu, self.ev].astype(float), 0.0, 1 - 1e-6)
        self.prop = g.prop[t:t + DELTA + 1][:, self.eu, self.ev].mean(0)
        self.C, self.rho0 = C[self.eu, self.ev], rho0[self.eu, self.ev]
        self.d0 = tre(self.prop, self.C, self.rho0, 0.0)
        self.nlr = -np.log1p(-self.risk)
        self.w_rr = self.nlr + 1e-5 * self.d0 + 1e-12
        self.trong = self.C * (1 - self.rho0)
        self.nhom, self.khoa_nhom = nhom_rui_ro(g, self.eu, self.ev)

    def nhom_cua(self, e, bo_nut=()):
        """Vector bool [số nhóm]: các nhóm rủi ro của các cạnh e, bỏ nhóm nút trong bo_nut (nguồn, đích)."""
        v = np.asarray(self.nhom[e].sum(0)).ravel() > 0
        for n in bo_nut:
            j = self.khoa_nhom.get(("nut", int(n)))
            if j is not None:
                v[j] = False
        return v


def nhom_rui_ro(g, eu, ev):
    """Nhóm rủi ro chung của từng cạnh: (điểm thời tiết, băng tần) nếu chịu mưa; mỗi nút có thể tắt (mặt đất: sét,
    UAV: hạ cánh) là một nhóm. Trả (ma trận thưa bool [E, số nhóm], {khoá: cột})."""
    khoa, hang, cot = {}, [], []
    for e, (u, v) in enumerate(zip(eu, ev)):
        ks = []
        if g.site[u, v] >= 0 and g.band[u, v] > 0:
            ks.append(("mua", int(g.site[u, v]), int(g.band[u, v])))
        ks += [("nut", int(n)) for n in (u, v) if g.kind[n] in ("uav", "gateway", "user") and g.node_site[n] >= 0]
        for k in ks:
            hang.append(e)
            cot.append(khoa.setdefault(k, len(khoa)))
    M = sp.csr_matrix((np.ones(len(hang)), (hang, cot)), shape=(len(eu), len(khoa)))
    return M, khoa


# ---------------------------------------------------------------- kho tuyến
def khoa_loai(g, s, v):
    return loai_lien_ket(g.kind[s], g.kind[v])[0] + "-" + g.kind[v]


def so_duong_roi(mg, s, d):
    """Số đường rời cạnh s → d trên toàn mạng (min-cut), theo luật chuyển tiếp."""
    i = mg.dt._giu(s)
    M = sp.csr_matrix((np.ones(len(i), np.int32), (mg.eu[i], mg.ev[i])), shape=(mg.dt.N, mg.dt.N))
    return int(maximum_flow(M, int(s), int(d)).flow_value)


def _bo_vong(r):
    ra, vi_tri = [], {}
    for n in r:
        if n in vi_tri:
            ra = ra[:vi_tri[n] + 1]
            vi_tri = {x: i for i, x in enumerate(ra)}
        else:
            vi_tri[n] = len(ra)
            ra.append(n)
    return ra


def suurballe(mg, s, d, m):
    """≤ m tuyến rời cạnh s → d có tổng trễ nền nhỏ nhất (luồng chi phí nhỏ nhất, sức chứa 1 mỗi cạnh)."""
    G = nx.DiGraph()
    for e in mg.dt._giu(s):
        G.add_edge(int(mg.eu[e]), int(mg.ev[e]), capacity=1, weight=int(round(mg.d0[e] * 1000)))
    if s not in G or d not in G:
        return []
    while m >= 1:
        nx.set_node_attributes(G, 0, "demand")
        G.nodes[s]["demand"], G.nodes[d]["demand"] = -m, m
        try:
            luong = nx.min_cost_flow(G)
            break
        except nx.NetworkXUnfeasible:
            m -= 1
    if m < 1:
        return []
    ra_khoi = {}
    for u, nb in luong.items():
        for v, f in nb.items():
            if f > 0:
                ra_khoi.setdefault(u, []).append(v)
    tuyen = []
    for _ in range(m):
        r = [s]
        while r[-1] != d:
            r.append(ra_khoi[r[-1]].pop())
        tuyen.append(_bo_vong(r))
    return sorted(tuyen, key=lambda r: float(mg.d0[mg.dt.canh_cua(r)].sum()))


def kho_tuyen(mg, s, d):
    """Kho ≤ 12 tuyến của cặp (s, d). Trả dict(kho, i_suurballe, m, so_loai, min_cut) hoặc None."""
    g = mg.g
    L = yen(mg.dt, s, d, mg.d0, 4)
    if not L:
        return None
    gia = lambda r: float(mg.d0[mg.dt.canh_cua(r)].sum())
    han = HANH_LANG[0] * gia(L[0]) + HANH_LANG[1]
    L = [r for r in L if gia(r) <= han]
    R = [r for r in yen(mg.dt, s, d, mg.w_rr, 4) if gia(r) <= han]
    ra_s = np.flatnonzero(mg.eu == s)
    loai = {}
    for e in ra_s:
        loai.setdefault(khoa_loai(g, s, mg.ev[e]), []).append(int(e))
    theo_loai = []
    for es in loai.values():
        cam = {(int(mg.eu[e]), int(mg.ev[e])) for e in ra_s if e not in es}
        r = mg.dt.duong(s, d, mg.d0, cam)
        if r is not None and gia(r) <= han:
            theo_loai.append(r)
    mc = so_duong_roi(mg, s, d)
    m = max(1, min(M_MAX, mc, len(loai) + 1))
    S = suurballe(mg, s, d, m)
    kho, thay = [], set()

    def them(r):
        if len(kho) < SO_KHO and tuple(r) not in thay:
            thay.add(tuple(r))
            kho.append(r)

    for r in S:
        them(r)
    for r in theo_loai:
        them(r)
    for i in range(4):
        for ds in (L, R):
            if i < len(ds):
                them(ds[i])
    return {"kho": kho, "i_suurballe": [kho.index(r) for r in S], "m": int(min(m, len(kho))),
            "so_loai": len(loai), "min_cut": mc}


def lua_chon(d0_kho, m):
    """Mọi tập m tuyến có thứ tự: tuyến chính, rồi dự phòng xếp theo trễ nền tăng dần. [n, m] chỉ số trong kho."""
    thu_tu = np.argsort(d0_kho, kind="stable")
    ra = [(c,) + B for c in range(len(d0_kho)) for B in itertools.combinations([i for i in thu_tu if i != c], m - 1)]
    return np.array(ra, np.int64).reshape(len(ra), m)


# ---------------------------------------------------------------- tạo tình huống
def tao_tinh_huong(g, q, r_online, t, rng, so_luong=(4, 5), so_nguon=3):
    """Một bộ K luồng ở t với kho tuyến từng cặp; None nếu không đủ luồng. Mức tải gắn sau (bai_toan_muc)."""
    mg = Mang(g, q, r_online, t)
    diem = diem_bao(g, q, mg.o)
    gw_all = [n for n in range(g.N) if g.kind[n] == "gateway"]
    toi = {}

    def toi_cua(n):
        if n not in toi:
            toi[n] = mg.dt.toi_duoc(n)
        return toi[n]

    nguon = []
    for n in np.argsort(-diem):
        if len(nguon) == so_nguon or diem[n] <= 0:
            break
        if any(toi_cua(n)[x] for x in gw_all if x != n and x not in nguon):
            nguon.append(int(n))
    gw = [x for x in gw_all if x not in nguon and any(toi_cua(s)[x] for s in nguon)]
    nguon = np.array([s for s in nguon if any(toi_cua(s)[x] for x in gw)], int)
    if len(nguon) == 0 or not gw:
        return None
    K = int(rng.integers(so_luong[0], so_luong[1] + 1))
    pn = (diem[nguon] + 0.5) / (diem[nguon] + 0.5).sum()
    cap, kho, src, dst, lam = {}, [], [], [], []
    for _ in range(20 * K):
        if len(src) == K:
            break
        s = int(rng.choice(nguon, p=pn))
        dich = [x for x in gw if x != s and toi_cua(s)[x]]
        if not dich:
            continue
        d = int(rng.choice(dich))
        if (s, d) not in cap:
            kt = kho_tuyen(mg, s, d)
            cap[(s, d)] = None if kt is None else len(kho)
            if kt is not None:
                kho.append(kt)
        if cap[(s, d)] is None:
            continue
        src.append(s)
        dst.append(d)
        lam.append(float(rng.lognormal(0.0, 0.5)))
    if len(src) < K:
        return None
    th = {"mg": mg, "src": np.array(src), "dst": np.array(dst), "lam": np.array(lam), "kho": kho,
          "cap": np.array([cap[(s, d)] for s, d in zip(src, dst)]), "t": int(t),
          "nguon": [g.names[n] for n in nguon], "diem_nguon": [float(diem[n]) for n in nguon]}
    # hệ số mức tải: như thí nghiệm 3, mọi luồng đi tuyến nhanh nhất trong kho
    F0 = np.zeros(len(mg.eu))
    for k, c in enumerate(th["cap"]):
        nhanh = min(kho[c]["kho"], key=lambda r: float(mg.d0[mg.dt.canh_cua(r)].sum()))
        F0[mg.dt.canh_cua(nhanh)] += lam[k]
    th["u"] = float(np.max(F0[F0 > 0] / mg.trong[F0 > 0]))
    return th


# ---------------------------------------------------------------- đối chứng họ Dijkstra
def yen_m(th):
    """m tuyến ngắn nhất (Yen) cho mỗi luồng — Dijkstra 'yếu'."""
    mg = th["mg"]
    return [yen(mg.dt, int(s), int(d), mg.d0, th["kho"][c]["m"]) for s, d, c in zip(th["src"], th["dst"], th["cap"])]


def suurballe_m(th):
    return [[th["kho"][c]["kho"][i] for i in th["kho"][c]["i_suurballe"]] for c in th["cap"]]


def dijkstra_phat(th, lam, beta, gamma):
    """Luồng lớn trước. Tuyến chính: Dijkstra theo trễ có tải + β·(−ln(1 − rủi ro)). Dự phòng: Dijkstra lặp, phạt γ
    trên cạnh chung nhóm rủi ro với các tuyến đã chọn của luồng (bỏ nhóm của chính nguồn / đích), tránh cạnh đã dùng,
    chỉ đi cạnh còn đủ dung lượng sau khi trừ tải các tuyến chính và phần giữ chỗ của các luồng có tuyến chính chung
    nhóm rủi ro với luồng này (có thể chuyển mạch cùng lúc). Không có đường đủ dung lượng thì nhận đường ít vi phạm nhất."""
    mg = th["mg"]
    E = len(mg.eu)
    F = np.zeros(E)
    giu = []                                         # (cạnh các tuyến dự phòng, λ, nhóm rủi ro của tuyến chính)
    ra = [None] * len(lam)
    for k in np.argsort(-lam, kind="stable"):
        s, d = int(th["src"][k]), int(th["dst"][k])
        m = th["kho"][th["cap"][k]]["m"]
        r1 = mg.dt.duong(s, d, tre(mg.prop, mg.C, mg.rho0, F + lam[k]) + beta * mg.nlr + 1e-12)
        e1 = mg.dt.canh_cua(r1)
        G1 = mg.nhom_cua(e1, (s, d))
        tuyen, dung, G_dung = [r1], np.zeros(E, bool), G1.copy()
        dung[e1] = True
        con = mg.trong - F
        for e_b, l_b, G_b in giu:
            if (G_b & G1).any():
                con[e_b] -= l_b
        e_du = []
        for _ in range(1, m):
            pen = np.asarray(mg.nhom @ G_dung.astype(float)).ravel() > 0
            w = mg.d0 + beta * mg.nlr + gamma * pen + 1e3 * dung + CAM * (con < lam[k]) + 1e-12
            r = mg.dt.duong(s, d, w)
            if r is None or any(r == x for x in tuyen):
                break
            e = mg.dt.canh_cua(r)
            tuyen.append(r)
            dung[e] = True
            G_dung |= mg.nhom_cua(e, (s, d))
            e_du.append(e)
        F[e1] += lam[k]
        if e_du:
            giu.append((np.unique(np.concatenate(e_du)), lam[k], G1))
        ra[k] = tuyen
    return ra


# ---------------------------------------------------------------- bài toán ở một mức tải + bộ chấm
class BaiToan:
    """Một tình huống ở một mức tải. Bảng tuyến mỗi luồng = kho (dùng cho lựa chọn) + tuyến của đối chứng (thêm)."""

    def __init__(self, th, muc, them=None, cac_luong=None):
        mg = th["mg"]
        self.th, self.muc = th, muc
        idx = np.arange(len(th["src"])) if cac_luong is None else np.asarray(cac_luong)
        self.idx = idx
        self.lam = th["lam"][idx] * (MUC_TAI[muc] if isinstance(muc, str) else float(muc)) / th["u"]   # tên mức hoặc hệ số
        self.K = len(idx)
        bang = []
        for j, k in enumerate(idx):
            kt = th["kho"][th["cap"][k]]
            ds = list(kt["kho"])
            for r in (them[j] if them is not None else []):
                if not any(r == x for x in ds):
                    ds.append(r)
            bang.append(ds)
        self.so_kho = [len(th["kho"][th["cap"][k]]["kho"]) for k in idx]
        canh = [[mg.dt.canh_cua(r) for r in ds] for ds in bang]
        self.U = np.unique(np.concatenate([e for c in canh for e in c]))
        loc = -np.ones(len(mg.eu), np.int64)
        loc[self.U] = np.arange(len(self.U))
        self.bang = bang
        self.inc = []
        for c in canh:
            M = np.zeros((len(c), len(self.U)), bool)
            for i, e in enumerate(c):
                M[i, loc[e]] = True
            self.inc.append(M)
        self.prop, self.C, self.rho0 = mg.prop[self.U], mg.C[self.U], mg.rho0[self.U]
        self.risk = mg.risk[self.U]
        self.m = np.array([th["kho"][th["cap"][k]]["m"] for k in idx])
        self.opts = []
        dmin, drange = np.zeros(self.K), np.zeros(self.K)
        for j in range(self.K):
            M = self.inc[j][:self.so_kho[j]]
            d0 = (tre(self.prop, self.C, self.rho0, self.lam[j]) * M).sum(1)       # chỉ luồng j trên tải nền
            self.opts.append(lua_chon(d0, self.m[j]))
            dmin[j] = d0.min()
            drange[j] = max(d0.max() - d0.min(), SAN_D)
        self.dmin, self.drange = dmin, drange

    def chi_so(self, k, r):
        """Chỉ số của tuyến r (danh sách nút) trong bảng của luồng k."""
        return next(i for i, x in enumerate(self.bang[k]) if x == r)

    def loi_giai(self, tuyen):
        """[danh sách tuyến mỗi luồng] → [mảng chỉ số trong bảng]."""
        return [np.array([self.chi_so(k, r) for r in rs], np.int64) for k, rs in enumerate(tuyen)]

    def tu_lua_chon(self, a):
        return [self.opts[k][a[k]] for k in range(self.K)]


class BaiToanGon:
    """Bài toán rút gọn về kho tuyến (cho DM, để lưu): đủ thuộc tính để TheGioi chấm các lời giải chọn trong kho.
    cot: các cột (cạnh) của BaiToan đầy đủ mà kho dùng."""

    def __init__(self, bt, dac_trung_canh=None):
        self.cot = np.unique(np.concatenate([np.flatnonzero(M[:n].any(0)) for M, n in zip(bt.inc, bt.so_kho)]))
        self.inc = [M[:n][:, self.cot] for M, n in zip(bt.inc, bt.so_kho)]
        self.U = bt.U[self.cot]
        self.K, self.so_kho = bt.K, list(bt.so_kho)
        self.lam, self.m, self.opts, self.dmin, self.drange = bt.lam, bt.m, bt.opts, bt.dmin, bt.drange
        self.prop, self.C, self.rho0, self.risk = (a[self.cot] for a in (bt.prop, bt.C, bt.rho0, bt.risk))
        self.dac_trung_canh = dac_trung_canh

    def tu_lua_chon(self, a):
        return [self.opts[k][a[k]] for k in range(self.K)]

    def the_gioi(self, song, Cw, rw):
        """Thế giới trên các cạnh của bài toán đầy đủ → TheGioi trên kho."""
        return TheGioi(self, song[:, self.cot], Cw[:, :, self.cot], rw[:, :, self.cot])


class TheGioi:
    """Một bộ W thế giới trên các cạnh U của bài toán: song [W, U] (sống suốt cửa sổ), Cw / rw [W, S, U]."""

    def __init__(self, bt, song, Cw, rw):
        self.bt, self.song, self.Cw, self.rw = bt, song, Cw, rw
        self.W = len(song)
        chet = (~song).astype(np.int32)
        self.alive = [(chet @ M.T.astype(np.int32)) == 0 for M in bt.inc]                 # [W, số tuyến]
        self.Hmin, self.Hmax = np.zeros(bt.K), np.ones(bt.K)
        for k in range(bt.K):
            H = self._mat_mot_minh(k)
            self.Hmin[k], self.Hmax[k] = H.min(), H.max()

    @classmethod
    def ensemble(cls, bt, g, pl, p, seeds):
        mg = bt.th["mg"]
        song, C2, rho = tg.the_gioi(g, pl, mg.o, mg.t, DELTA, mg.eu[bt.U], mg.ev[bt.U], seeds, p)
        return cls(bt, song, C2, rho)

    @classmethod
    def quan_sat(cls, bt, Wn, q):
        mg = bt.th["mg"]
        u, v, t = mg.eu[bt.U], mg.ev[bt.U], mg.t
        song = (Wn[t:t + DELTA + 1][:, u, v] > 0).all(0)[None]
        return cls(bt, song, q["C2"][t:t + DELTA + 1][:, u, v][None].astype(float),
                   q["rho"][t:t + DELTA + 1][:, u, v][None].astype(float))

    @classmethod
    def tu_bang(cls, bt, rng, W=SO_TG_BANG):
        """Ước lượng online: cạnh hỏng độc lập theo bảng rủi ro, dung lượng / tải nền = quan sát ở o."""
        song = rng.random((W, len(bt.U))) >= bt.risk[None]
        return cls(bt, song, np.broadcast_to(bt.C, (W, 1, len(bt.U))), np.broadcast_to(bt.rho0, (W, 1, len(bt.U))))

    def _qua_tai(self, F):
        """F [W, ..., U] → bool cùng shape: liên kết có ρ ≥ 1 ở một mốc nào đó."""
        sh = (self.W,) + (1,) * (F.ndim - 2) + (len(self.bt.U),)
        ovl = np.zeros(F.shape, bool)
        for s in range(self.Cw.shape[1]):
            ovl |= self.rw[:, s].reshape(sh) + F / self.Cw[:, s].reshape(sh) >= 1
        return ovl

    def _dang_dung(self, k, tuyen):
        """Luồng k với các tuyến `tuyen` (chỉ số bảng): (có tuyến sống [W], cạnh tuyến đang dùng [W, U])."""
        al = self.alive[k][:, tuyen]
        co = al.any(1)
        return co, self.bt.inc[k][tuyen[al.argmax(1)]] & co[:, None]

    def _mat_mot_minh(self, k):
        O = self.bt.opts[k]
        al = self.alive[k][:, O]                                                         # [W, n, m]
        co = al.any(-1)
        inc = self.bt.inc[k][O[np.arange(len(O))[None], al.argmax(-1)]] & co[..., None]  # [W, n, U]
        lost = ~co | (self._qua_tai(self.bt.lam[k] * inc) & inc).any(-1)
        return lost.mean(0)

    def _J(self, H, d, lam):
        Hn = (H - self.Hmin) / np.maximum(self.Hmax - self.Hmin, SAN_H)
        T = (d - self.bt.dmin) / self.bt.drange
        return ((W_RR * Hn + (1 - W_RR) * T) * lam).sum(-1) / lam.sum()

    def cham(self, sol, chon=None):
        """Số đo của một lời giải (mảng chỉ số bảng mỗi luồng); chon [K] bool: chỉ tính các luồng được chọn."""
        bt = self.bt
        chon = np.ones(bt.K, bool) if chon is None else chon
        lam = np.where(chon, bt.lam, 0.0)
        co, inc = zip(*[self._dang_dung(k, sol[k]) if chon[k] else (np.ones(self.W, bool),
                        np.zeros((self.W, len(bt.U)), bool)) for k in range(bt.K)])
        F = sum(lam[k] * inc[k] for k in range(bt.K))
        ovl = self._qua_tai(F)
        a = np.array([~c for c in co])                                                   # [K, W]
        b = np.array([(ovl & i).any(1) for i in inc]) & ~a
        H = (a | b).mean(1)
        Fp = sum(lam[k] * bt.inc[k][sol[k][0]] for k in range(bt.K) if chon[k])
        lat = tre(bt.prop, bt.C, bt.rho0, Fp)
        d = np.array([(lat * bt.inc[k][sol[k][0]]).sum() for k in range(bt.K)])
        J = float(self._J(H, d, lam))
        return {"J": J, "mat_dv_%": float((H * lam).sum() / lam.sum() * 100),
                "khong_tuyen_song_%": float((a.mean(1) * lam).sum() / lam.sum() * 100),
                "du_phong_qua_tai_%": float((b.mean(1) * lam).sum() / lam.sum() * 100),
                "tre_chinh_ms": float((d * lam).sum() / lam.sum()), "H": H}

    def J_lua_chon(self, a, k, chon=None):
        """J của mọi lựa chọn của luồng k (các luồng khác giữ theo a, chỉ số lựa chọn). Trả [n]."""
        bt = self.bt
        chon = np.ones(bt.K, bool) if chon is None else chon.copy()
        chon[k] = True
        lam = np.where(chon, bt.lam, 0.0)
        O = bt.opts[k]
        n = len(O)
        khac = [j for j in range(bt.K) if j != k and chon[j]]
        dd = {j: self._dang_dung(j, bt.opts[j][a[j]]) for j in khac}
        F = np.zeros((self.W, len(bt.U)))
        for j in khac:
            F += lam[j] * dd[j][1]
        al = self.alive[k][:, O]
        co = al.any(-1)
        inc = bt.inc[k][O[np.arange(n)[None], al.argmax(-1)]] & co[..., None]            # [W, n, U]
        ovl = self._qua_tai(F[:, None] + lam[k] * inc)
        H = np.zeros((n, bt.K))
        H[:, k] = (~co | (ovl & inc).any(-1)).mean(0)
        for j in khac:
            H[:, j] = (~dd[j][0][:, None] | (ovl & dd[j][1][:, None]).any(-1)).mean(0)
        P = [bt.inc[j][bt.opts[j][a[j]][0]] for j in range(bt.K)]
        Fp = sum(lam[j] * P[j] for j in khac) + lam[k] * bt.inc[k][O[:, 0]]              # [n, U]
        lat = tre(bt.prop, bt.C, bt.rho0, Fp)
        d = np.zeros((n, bt.K))
        d[:, k] = (lat * bt.inc[k][O[:, 0]]).sum(1)
        for j in khac:
            d[:, j] = (lat * P[j]).sum(1)
        return self._J(H, d, lam)


# ---------------------------------------------------------------- chọn trong kho theo J
def tham_lam(tgw):
    bt = tgw.bt
    a = np.zeros(bt.K, np.int64)
    chon = np.zeros(bt.K, bool)
    for k in np.argsort(-bt.lam, kind="stable"):
        a[k] = int(np.argmin(tgw.J_lua_chon(a, k, chon)))
        chon[k] = True
    return a


def tkcb(tgw, a0, rng=None, max_vong=20):
    """Tìm kiếm cục bộ: lần lượt từng luồng đổi sang lựa chọn làm J nhỏ nhất."""
    a = np.array(a0, np.int64).copy()
    J = tgw.cham(tgw.bt.tu_lua_chon(a))["J"]
    for _ in range(max_vong):
        doi = False
        for k in (rng.permutation(tgw.bt.K) if rng is not None else range(tgw.bt.K)):
            Js = tgw.J_lua_chon(a, k)
            j = int(np.argmin(Js))
            if Js[j] < J - 1e-12:
                a[k], J, doi = j, float(Js[j]), True
        if not doi:
            break
    return a, J


def lua_chon_cua(bt, k, tuyen):
    """Chỉ số lựa chọn của luồng k ứng với tập tuyến (thứ tự dự phòng không tính); None nếu không có trong kho."""
    try:
        ids = [bt.chi_so(k, r) for r in tuyen]
    except StopIteration:
        return None
    if max(ids) >= bt.so_kho[k] or len(ids) != bt.m[k]:
        return None
    O = bt.opts[k]
    hit = np.flatnonzero((O[:, 0] == ids[0]) & np.array([set(o[1:]) == set(ids[1:]) for o in O]))
    return int(hit[0]) if len(hit) else None


def duyet_het(tgw):
    """Lời giải tối ưu bằng duyệt hết (chỉ dùng để kiểm tra trên bài toán nhỏ)."""
    bt = tgw.bt
    tot, gia = None, np.inf
    for head in itertools.product(*[range(len(O)) for O in bt.opts[:-1]]):
        a = np.array(list(head) + [0])
        Js = tgw.J_lua_chon(a, bt.K - 1)
        j = int(np.argmin(Js))
        if Js[j] < gia:
            gia, tot = float(Js[j]), np.array(list(head) + [j])
    return tot, gia
