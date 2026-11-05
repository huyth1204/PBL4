"""
bai_toan_rr.py — Thí nghiệm 3: 4–5 luồng khẩn cấp lúc bão mạnh trên mạng SAGSIN v5 (224 nút), chấm bằng điểm
70% rủi ro hỏng liên kết + 30% tốc độ tuyến (xem results/bao_rr/TIEU_CHI_3.md).

Bộ định tuyến quyết định ở t, chỉ biết trạng thái ở o = t − 9 và lịch quỹ đạo; tuyến phải giữ trong [t, t+3].
    p_r = 1 − Π_e (1 − risk_e)                                  rủi ro tuyến (bảng tra bão cuối, ma_tran_online)
    d_r = Σ_e prop_e + PKT/C_e·(1 + h(ρ_e + F_e/C_e)) + PROC      trễ tuyến khi có tải F của mọi luồng
    H_k = (p − min p) / max(max p − min p, 0,01)               chuẩn hoá trên các ứng viên của luồng k
    T_k = (d(F) − min d⁰) / max(max d⁰ − min d⁰, 1 ms)          d⁰ = trễ khi chỉ có luồng k; không chặn trên
    J   = Σ_k λ_k/Σλ · (0,7·H_k + 0,3·T_k)                      càng thấp càng tốt
Một tình huống lưu ở dạng gọn (chỉ các cạnh nằm trên tuyến ứng viên): đủ để tính J, tối ưu chính xác (duyệt hết
≤ 8⁵ tổ hợp) và kết quả thực tế (cạnh có sống suốt [t, t+3] trong mô phỏng không, trễ thật ở t).
"""

from __future__ import annotations

import heapq
import itertools

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import breadth_first_order
from scipy.sparse.csgraph import dijkstra as _dijkstra

from src.ngau_nhien.kich_ban import PKT, PROC
from src.nhieu_luong.bai_toan import MUC_TAI, SO_UNG_VIEN, diem_bao, h_hang_doi

W_RR = 0.7                       # trọng số rủi ro hỏng liên kết; tốc độ = 1 − W_RR
SAN_P, SAN_D = 0.01, 1.0         # sàn của khoảng chuẩn hoá: 1 điểm phần trăm rủi ro, 1 ms trễ
TUOI, DELTA = 9, 3               # tuổi thông tin / cửa sổ giữ tuyến của bảng rủi ro
MOI_BEN = SO_UNG_VIEN // 2       # 4 ứng viên theo trễ + 4 theo rủi ro
HANH_LANG = (3.0, 20.0)          # ứng viên: trễ nền ≤ 3·(trễ nhanh nhất) + 20 ms


# ---------------------------------------------------------------- đồ thị và Yen (scipy)
class DoThi:
    """Cạnh có hướng dùng được eu → ev; Dijkstra giữ luật chuyển tiếp (chỉ nguồn được đi ra từ nút user)."""

    def __init__(self, N, can_relay, eu, ev):
        self.N, self.can_relay = int(N), np.asarray(can_relay, bool)
        self.eu, self.ev = np.asarray(eu, np.int64), np.asarray(ev, np.int64)
        self.eid = -np.ones((self.N, self.N), np.int64)
        self.eid[self.eu, self.ev] = np.arange(len(self.eu))

    def _giu(self, s, cam_canh=(), cam_nut=()):
        giu = self.can_relay[self.eu] | (self.eu == s)
        if cam_nut:
            n = np.fromiter(cam_nut, np.int64)
            giu &= ~np.isin(self.eu, n) & ~np.isin(self.ev, n)
        if cam_canh:
            e = np.array([self.eid[a, b] for a, b in cam_canh], np.int64)
            giu[e[e >= 0]] = False
        return np.flatnonzero(giu)

    def duong(self, s, d, w, cam_canh=(), cam_nut=()):
        """Đường ngắn nhất s → d theo trọng số cạnh w (> 0); None nếu không tới được."""
        i = self._giu(s, cam_canh, cam_nut)
        M = sp.csr_matrix((w[i], (self.eu[i], self.ev[i])), shape=(self.N, self.N))
        dist, pred = _dijkstra(M, directed=True, indices=int(s), return_predecessors=True)
        if not np.isfinite(dist[d]):
            return None
        r = [int(d)]
        while r[-1] != s:
            r.append(int(pred[r[-1]]))
        return r[::-1]

    def toi_duoc(self, s):
        """[N] nút tới được từ s theo luật chuyển tiếp."""
        i = self._giu(s)
        M = sp.csr_matrix((np.ones(len(i)), (self.eu[i], self.ev[i])), shape=(self.N, self.N))
        ra = np.zeros(self.N, bool)
        ra[breadth_first_order(M, int(s), directed=True, return_predecessors=False)] = True
        return ra

    def canh_cua(self, r):
        return self.eid[np.array(r[:-1]), np.array(r[1:])]


def yen(dt: DoThi, s, d, w, k):
    """Tối đa k đường đơn ngắn nhất theo w (thuật toán Yen), giữ luật chuyển tiếp."""
    dau = dt.duong(s, d, w)
    if dau is None:
        return []
    A, B, thay = [dau], [], {tuple(dau)}
    while len(A) < k:
        cuoi = A[-1]
        for i in range(len(cuoi) - 1):
            spur, goc = cuoi[i], cuoi[:i + 1]
            if i > 0 and not dt.can_relay[spur]:
                continue
            cam = {(p[i], p[i + 1]) for p in A if len(p) > i + 1 and p[:i + 1] == goc}
            nhanh = dt.duong(spur, d, w, cam, set(goc[:-1]))
            if nhanh is None:
                continue
            r = goc[:-1] + nhanh
            if tuple(r) not in thay:
                thay.add(tuple(r))
                heapq.heappush(B, (float(w[dt.canh_cua(r)].sum()), r))
        if not B:
            break
        A.append(heapq.heappop(B)[1])
    return A


def ung_vien(dt, s, d, w_tre, w_rr):
    """≤ 8 tuyến: xen kẽ Yen theo trễ và theo rủi ro (bỏ trùng, tối đa 4 + 4, bù nếu một bên thiếu), chỉ giữ tuyến
    trong hành lang trễ nền ≤ 3·(trễ nhanh nhất) + 20 ms (như thí nghiệm 2). Trả (danh sách tuyến, chỉ số tuyến an
    toàn nhất trong các ứng viên). Ứng viên 0 = tuyến nhanh nhất."""
    L = yen(dt, s, d, w_tre, SO_UNG_VIEN)
    if not L:
        return [], 0
    gia = lambda r, w: float(w[dt.canh_cua(r)].sum())
    han = HANH_LANG[0] * gia(L[0], w_tre) + HANH_LANG[1]
    L = [r for r in L if gia(r, w_tre) <= han]
    R = [r for r in yen(dt, s, d, w_rr, SO_UNG_VIEN) if gia(r, w_tre) <= han]
    ra, thay = [], set()

    def them(r):
        if len(ra) < SO_UNG_VIEN and tuple(r) not in thay:
            thay.add(tuple(r))
            ra.append(r)

    for i in range(MOI_BEN):
        for ds in (L, R):
            if i < len(ds):
                them(ds[i])
    for i in range(MOI_BEN, SO_UNG_VIEN):
        for ds in (L, R):
            if i < len(ds):
                them(ds[i])
    return ra, int(np.argmin([gia(r, w_rr) for r in ra]))


# ---------------------------------------------------------------- tạo tình huống
def tao_tinh_huong(g, Wn, q, r_online, t, rng, so_luong=(4, 5), so_nguon=3, cache=None):
    """Một bộ K luồng ở mốc quyết định t; trả {mức tải: tình huống dạng gọn} (cùng luồng, khác hệ số) hoặc None.
    r_online = ma_tran_online(...) ở t (chỉ dùng quan sát tới o = t − 9)."""
    N, o = g.N, t - TUOI
    C, rho0 = q["C2"][o].astype(float), q["rho"][o].astype(float)
    valid = r_online["valid"] & (rho0 < 1.0)                     # tải nền ≥ 1: hết dung lượng trống (như thí nghiệm 1)
    np.fill_diagonal(valid, False)
    eu, ev = np.nonzero(valid)
    can_relay = np.array([k != "user" for k in g.kind])
    dt = DoThi(N, can_relay, eu, ev)
    risk = np.clip(r_online["risk_edge"][eu, ev].astype(float), 0.0, 1 - 1e-6)
    prop = g.prop[t:t + DELTA + 1][:, eu, ev].mean(0)          # truyền lan theo quỹ đạo (biết trước)
    Ce, re = C[eu, ev], rho0[eu, ev]
    d0 = prop + PKT / Ce * (1 + h_hang_doi(re)) + PROC
    w_rr = -np.log1p(-risk) + 1e-5 * d0 + 1e-12
    diem = diem_bao(g, q, o)
    gw_all = [n for n in range(N) if g.kind[n] == "gateway"]
    toi = {}

    def toi_cua(n):
        if n not in toi:
            toi[n] = dt.toi_duoc(n)
        return toi[n]

    nguon, bo_qua = [], []
    for n in np.argsort(-diem):
        if len(nguon) == so_nguon or diem[n] <= 0:
            break
        if any(toi_cua(n)[x] for x in gw_all if x != n and x not in nguon):
            nguon.append(int(n))
        else:
            bo_qua.append(g.names[n])
    gw = [x for x in gw_all if x not in nguon and any(toi_cua(s)[x] for s in nguon)]
    nguon = np.array([s for s in nguon if any(toi_cua(s)[x] for x in gw)], int)
    if len(nguon) == 0 or not gw:
        return None
    K = int(rng.integers(so_luong[0], so_luong[1] + 1))
    pn = (diem[nguon] + 0.5) / (diem[nguon] + 0.5).sum()
    cache = {} if cache is None else cache
    src, dst, lam, uv, i_at = [], [], [], [], []
    for _ in range(20 * K):
        if len(src) == K:
            break
        s = int(rng.choice(nguon, p=pn))
        dich = [x for x in gw if x != s and toi_cua(s)[x]]
        if not dich:
            continue
        d = int(rng.choice(dich))
        if (s, d) not in cache:
            cache[(s, d)] = ung_vien(dt, s, d, d0, w_rr)
        tuyen, j = cache[(s, d)]
        if not tuyen:
            continue
        src.append(s)
        dst.append(d)
        lam.append(float(rng.lognormal(0.0, 0.5)))
        uv.append(tuyen)
        i_at.append(j)
    if len(src) < K:
        return None
    # dạng gọn: chỉ các cạnh nằm trên ứng viên
    canh_g = [[dt.canh_cua(r) for r in u] for u in uv]
    dung = np.unique(np.concatenate([e for u in canh_g for e in u]))
    loc = -np.ones(len(eu), np.int64)
    loc[dung] = np.arange(len(dung))
    Lmax = max(len(e) for u in canh_g for e in u)
    canh = -np.ones((K, SO_UNG_VIEN, Lmax), np.int16)
    hop_le = np.zeros((K, SO_UNG_VIEN), bool)
    for k, u in enumerate(canh_g):
        for c, e in enumerate(u):
            canh[k, c, :len(e)] = loc[e]
            hop_le[k, c] = True
    gu, gv = eu[dung], ev[dung]
    tuong_lai = Wn[t:t + DELTA + 1][:, gu, gv] > 0
    x = {"e": (gu * N + gv).astype(np.int32), "prop": prop[dung], "C": Ce[dung], "rho0": re[dung],
         "risk": risk[dung],
         "song_thuc": tuong_lai.all(0), "prop_thuc": g.prop[t, gu, gv].astype(float),
         "C_thuc": q["C2"][t, gu, gv].astype(float), "rho_thuc": q["rho"][t, gu, gv].astype(float),
         "src": np.array(src, np.int16), "dst": np.array(dst, np.int16), "canh": canh, "hop_le": hop_le,
         "i_an_toan": np.array(i_at, np.int8), "t": int(t),
         "meta": {"nguon": [g.names[n] for n in nguon], "nguon_bo_qua": bo_qua,
                  "diem_nguon": [float(diem[n]) for n in nguon], "so_canh_valid": int(len(eu)),
                  "dich": [g.names[n] for n in dst]}}
    x["p_uv"] = _rui_ro_uv(x)
    lam = np.array(lam)
    F0 = tai(x, np.zeros((1, K), int), lam)[0]                   # mọi luồng đi tuyến nhanh nhất
    trong = x["C"] * (1 - x["rho0"])
    u = float(np.max(F0[F0 > 0] / trong[F0 > 0]))
    ra = {}
    for ten, muc in MUC_TAI.items():
        y = dict(x, lam=lam * muc / u, muc=ten)
        chuan_hoa(y)
        ra[ten] = y
    return ra


def _rui_ro_uv(x):
    """[K, 8] rủi ro tuyến p = 1 − Π(1 − risk_e) (0 cho ứng viên không tồn tại)."""
    m = x["canh"] >= 0
    lr = np.where(m, np.log1p(-x["risk"][np.maximum(x["canh"], 0)]), 0.0).sum(-1)
    return np.where(x["hop_le"], -np.expm1(lr), 0.0)


def chuan_hoa(x):
    """Thêm H [K,8], d0 [K,8] (trễ khi chỉ có luồng k), dmin, drange [K] — phụ thuộc λ nên tính theo từng mức tải."""
    hl = x["hop_le"]
    p = x["p_uv"]
    pmin = np.where(hl, p, np.inf).min(1)
    pmax = np.where(hl, p, -np.inf).max(1)
    x["H"] = np.where(hl, (p - pmin[:, None]) / np.maximum(pmax - pmin, SAN_P)[:, None], 0.0)
    m = x["canh"] >= 0
    e = np.maximum(x["canh"], 0).astype(np.int64)                        # [K,8,L]
    C, Fo = x["C"][e], x["lam"][:, None, None]                           # chỉ luồng k trên tải nền
    de = x["prop"][e] + PKT / C * (1 + h_hang_doi(x["rho0"][e] + Fo / C)) + PROC
    d0 = np.where(hl, np.where(m, de, 0.0).sum(-1), 0.0)
    x["d0"] = d0
    x["dmin"] = np.where(hl, d0, np.inf).min(1)
    x["drange"] = np.maximum(np.where(hl, d0, -np.inf).max(1) - x["dmin"], SAN_D)
    return x


# ---------------------------------------------------------------- điểm và kết quả
def _lk(x, A):
    """A [S,K] → chỉ số cạnh cục bộ [S,K,L] (−1 = trống) và mặt nạ."""
    lk = x["canh"].astype(np.int64)[np.arange(A.shape[1])[None], A]
    return lk, lk >= 0


def tai(x, A, lam=None, chon=None):
    """Tải mỗi cạnh [S,E]; chon [K] bool: chỉ cộng các luồng được chọn."""
    lam = x["lam"] if lam is None else lam
    lam = lam if chon is None else np.where(chon, lam, 0.0)
    A = np.atleast_2d(A)
    lk, m = _lk(x, A)
    S = A.shape[0]
    F = np.zeros((S, len(x["e"])))
    s_i = np.broadcast_to(np.arange(S)[:, None, None], lk.shape)[m]
    np.add.at(F, (s_i, lk[m]), np.broadcast_to(lam[None, :, None], lk.shape)[m])
    return F


def tre_canh(x, F, thuc=False):
    """Trễ cạnh (ms) dưới tải F [S,E]; thuc=True: dùng truyền lan / dung lượng / tải nền thật ở t."""
    p, C, r = (x["prop_thuc"], x["C_thuc"], x["rho_thuc"]) if thuc else (x["prop"], x["C"], x["rho0"])
    return p + PKT / C * (1 + h_hang_doi(r + F / C)) + PROC


def tre_tuyen(x, A, F, thuc=False):
    """[S,K] trễ đầu-cuối của tuyến đã chọn dưới tải F."""
    lk, m = _lk(x, A)
    de = tre_canh(x, F, thuc)[np.arange(len(A))[:, None, None], np.maximum(lk, 0)]
    return np.where(m, de, 0.0).sum(-1)


def diem(x, A, w=W_RR, chon=None):
    """J [S] của S lời giải A [S,K]; chon [K] bool: chỉ tính (tải và điểm) các luồng được chọn."""
    A = np.atleast_2d(A)
    K = A.shape[1]
    F = tai(x, A, chon=chon)
    T = (tre_tuyen(x, A, F) - x["dmin"]) / x["drange"]
    H = x["H"][np.arange(K)[None], A]
    lam = x["lam"] if chon is None else np.where(chon, x["lam"], 0.0)
    return ((w * H + (1 - w) * T) * lam).sum(1) / lam.sum()


def ket_qua(x, a, w=W_RR):
    """Các số đo của một lời giải a [K]."""
    a = np.asarray(a)[None]
    K = a.shape[1]
    lam = x["lam"]
    F = tai(x, a)
    d = tre_tuyen(x, a, F)[0]
    lk, m = _lk(x, a)
    lk, m = lk[0], m[0]
    p = x["p_uv"][np.arange(K), a[0]]
    rho = x["rho0"] + F[0] / x["C"]
    qt = np.where(m, rho[np.maximum(lk, 0)] >= 1, False).any(1)
    dut = np.where(m, ~x["song_thuc"][np.maximum(lk, 0)], False).any(1)
    d_thuc = tre_tuyen(x, a, tai(x, a), thuc=True)[0]
    song = ~dut
    return {"J": float(diem(x, a, w)[0]),
            "H": float((x["H"][np.arange(K), a[0]] * lam).sum() / lam.sum()),
            "T": float((((d - x["dmin"]) / x["drange"]) * lam).sum() / lam.sum()),
            "rui_ro_%": float((p * lam).sum() / lam.sum() * 100),
            "tre_ms": float((d * lam).sum() / lam.sum()),
            "qua_tai_%": float((qt * lam).sum() / lam.sum() * 100),
            "dut_luu_luong_%": float((dut * lam).sum() / lam.sum() * 100),
            "dut_luong_%": float(dut.mean() * 100),
            "tre_thuc_ms": float((d_thuc * lam * song).sum() / (lam * song).sum()) if song.any() else float("nan")}


# ---------------------------------------------------------------- các phương pháp (trên ứng viên)
def toi_uu(x, w=W_RR, lo=8192):
    """Duyệt hết mọi tổ hợp ứng viên. Trả (lời giải, J)."""
    lua_chon = [np.flatnonzero(h) for h in x["hop_le"]]
    tot, gia = None, np.inf
    it = itertools.product(*lua_chon)
    while True:
        A = np.array(list(itertools.islice(it, lo)), np.int64)
        if len(A) == 0:
            break
        J = diem(x, A, w)
        j = int(np.argmin(J))
        if J[j] < gia - 1e-12:
            tot, gia = A[j].copy(), float(J[j])
    return tot, gia


def nhanh_nhat(x):
    return np.zeros(len(x["lam"]), np.int64)


def an_toan_nhat(x):
    return x["i_an_toan"].astype(np.int64)


def diem_doc_lap(x, w=W_RR):
    T0 = (x["d0"] - x["dmin"][:, None]) / x["drange"][:, None]
    return np.where(x["hop_le"], w * x["H"] + (1 - w) * T0, np.inf).argmin(1)


def tuan_tu_tre(x):
    """Luồng lớn trước; mỗi luồng chọn ứng viên có trễ tuyến (tải các luồng trước + chính nó) nhỏ nhất."""
    K = len(x["lam"])
    a = np.zeros(K, np.int64)
    chon = np.zeros(K, bool)
    for k in np.argsort(-x["lam"], kind="stable"):
        chon[k] = True
        c = np.flatnonzero(x["hop_le"][k])
        A = np.repeat(a[None], len(c), 0)
        A[:, k] = c
        a[k] = c[int(np.argmin(tre_tuyen(x, A, tai(x, A, chon=chon))[:, k]))]
    return a


def tham_lam(x, w=W_RR):
    """Luồng lớn trước; mỗi luồng chọn ứng viên làm J (của các luồng đã chọn) nhỏ nhất."""
    K = len(x["lam"])
    a = np.zeros(K, np.int64)
    chon = np.zeros(K, bool)
    for k in np.argsort(-x["lam"], kind="stable"):
        chon[k] = True
        c = np.flatnonzero(x["hop_le"][k])
        A = np.repeat(a[None], len(c), 0)
        A[:, k] = c
        a[k] = c[int(np.argmin(diem(x, A, w, chon)))]
    return a


def tkcb(x, a0, w=W_RR, max_vong=30):
    """Tìm kiếm cục bộ: lần lượt từng luồng đổi sang ứng viên làm J nhỏ nhất (giữ nguyên luồng khác)."""
    a = np.array(a0, np.int64).copy()
    gia = float(diem(x, a, w)[0])
    for _ in range(max_vong):
        doi = False
        for k in range(len(a)):
            c = np.flatnonzero(x["hop_le"][k])
            A = np.repeat(a[None], len(c), 0)
            A[:, k] = c
            J = diem(x, A, w)
            j = int(np.argmin(J))
            if J[j] < gia - 1e-12:
                a, gia, doi = A[j].copy(), float(J[j]), True
        if not doi:
            break
    return a
