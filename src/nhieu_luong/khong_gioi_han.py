"""
khong_gioi_han.py — Bài toán nhiều luồng KHÔNG giới hạn tuyến ứng viên (thí nghiệm 2, results/yagi2/TIEU_CHI_2.md).

Mỗi luồng đi được mọi đường đơn hợp lệ trên các liên kết còn sống có tải nền < 1 (nút người dùng không chuyển tiếp).
Dijkstra dùng scipy (viết bằng C); mỗi nguồn có sẵn một ma trận CSR đã bỏ cạnh đi ra từ nút không được chuyển tiếp,
mỗi lần gọi chỉ thay mảng trọng số → nhanh, dùng chung cho mọi phương pháp (đối chứng và cách dựng tuyến của mô hình).

  MangDay         trạng thái mạng của một tình huống (danh sách cạnh còn sống) + các luồng
  tuan_tu_kgh     Dijkstra tuần tự (giống thí nghiệm 1: trọng số = trễ sau khi cộng luồng, luồng lớn trước)
  dung_tuyen      dựng tuyến tuần tự có hướng dẫn: trọng số = trễ sau khi cộng luồng + γ·(−ln(ŷ + 0,01))
  tkcb_kgh        tìm kiếm cục bộ không giới hạn (đổi sang đường Dijkstra theo chi phí biên)
  tkcb_kgh_10     10 điểm xuất phát, lấy lời giải tốt nhất (mốc)
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra as _dijkstra

from src.ngau_nhien.kich_ban import PKT, PROC
from src.nhieu_luong.bai_toan import h_hang_doi

LN_EPS = 0.01


class MangDay:
    """Cạnh có hướng còn dùng được: eu → ev, kèm truyền lan / dung lượng / tải nền (float64)."""

    def __init__(self, N, can_relay, eu, ev, prop, C, rho0, src, dst, lam):
        self.N, self.can_relay = int(N), np.asarray(can_relay, bool)
        self.eu, self.ev = np.asarray(eu, np.int64), np.asarray(ev, np.int64)
        self.prop, self.C, self.rho0 = (np.asarray(a, float) for a in (prop, C, rho0))
        self.src, self.dst, self.lam = np.asarray(src, np.int64), np.asarray(dst, np.int64), np.asarray(lam, float)
        self.E, self.K = len(self.eu), len(self.lam)
        self.eid = -np.ones((self.N, self.N), np.int16)                      # E < 32768
        self.eid[self.eu, self.ev] = np.arange(self.E)
        self._csr = {}

    @classmethod
    def tu_tinh_huong(cls, th):
        """Từ TinhHuong (bai_toan.py, mạng đầy đủ N×N)."""
        N = th.N
        idx = np.flatnonzero(th.song)
        idx = idx[idx // N != idx % N]
        return cls(N, th.can_relay, idx // N, idx % N, th.prop[idx], th.C[idx], th.rho0[idx], th.src, th.dst, th.lam)

    def luu(self):
        return {"N": self.N, "can_relay": self.can_relay, "eu": self.eu.astype(np.int16), "ev": self.ev.astype(np.int16),
                "prop": self.prop, "C": self.C, "rho0": self.rho0, "src": self.src.astype(np.int16),
                "dst": self.dst.astype(np.int16), "lam": self.lam}

    @classmethod
    def nap(cls, d):
        return cls(d["N"], d["can_relay"], d["eu"], d["ev"], d["prop"], d["C"], d["rho0"], d["src"], d["dst"], d["lam"])

    # ---- trễ, chi phí
    def tre(self, F):
        return self.prop + PKT / self.C * (1 + h_hang_doi(self.rho0 + F / self.C)) + PROC

    def f(self, F):
        return F * self.tre(F)

    def chi_phi_F(self, F):
        return float(self.f(F).sum() / self.lam.sum())

    def tai(self, tuyen):
        """tuyen: list K mảng chỉ số cạnh."""
        F = np.zeros(self.E)
        for k, e in enumerate(tuyen):
            np.add.at(F, e, self.lam[k])
        return F

    def chi_phi(self, tuyen):
        return self.chi_phi_F(self.tai(tuyen))

    def qua_tai(self, F):
        rho = self.rho0 + F / self.C
        return float((F * (rho >= 1)).sum() / max(F.sum(), 1e-12))

    # ---- Dijkstra
    def _cau_truc(self, s):
        """CSR cố định cho nguồn s: bỏ cạnh đi ra từ nút không chuyển tiếp (trừ chính s)."""
        if s not in self._csr:
            giu = np.flatnonzero(self.can_relay[self.eu] | (self.eu == s))
            M = sp.csr_matrix((np.ones(len(giu)), (self.eu[giu], self.ev[giu])), shape=(self.N, self.N))
            M.sort_indices()
            # vị trí trong M.data ↔ chỉ số cạnh
            hang = np.repeat(np.arange(self.N), np.diff(M.indptr))
            thu_tu = self.eid[hang, M.indices]
            self._csr[s] = (M, thu_tu)
        return self._csr[s]

    def duong(self, s, d, w):
        """Đường ngắn nhất s → d theo trọng số cạnh w (>0). Trả mảng chỉ số cạnh (rỗng nếu không tới được)."""
        M, thu_tu = self._cau_truc(int(s))
        M.data = w[thu_tu]
        _, pred = _dijkstra(M, directed=True, indices=int(s), return_predecessors=True)
        if pred[d] < 0 and d != s:
            return np.zeros(0, np.int64)
        nut = [int(d)]
        while nut[-1] != s:
            nut.append(int(pred[nut[-1]]))
        nut = nut[::-1]
        return self.eid[nut[:-1], nut[1:]]

    def khoang_cach(self, s=None, d=None, w=None):
        """Khoảng cách từ s (xuôi) hoặc tới d (ngược, chỉ đi qua nút chuyển tiếp được) theo trọng số w."""
        w = self.tre(np.zeros(self.E)) if w is None else w
        if s is not None:
            M, thu_tu = self._cau_truc(int(s))
            M.data = w[thu_tu]
            return _dijkstra(M, directed=True, indices=int(s))
        giu = np.flatnonzero(self.can_relay[self.eu])
        M = sp.csr_matrix((w[giu], (self.ev[giu], self.eu[giu])), shape=(self.N, self.N))   # đồ thị ngược
        return _dijkstra(M, directed=True, indices=int(d))


# ---------------------------------------------------------------- các phương pháp
def doc_lap_kgh(m: MangDay):
    w = m.tre(np.zeros(m.E))
    return [m.duong(m.src[k], m.dst[k], w) for k in range(m.K)]


def dung_tuyen(m: MangDay, thu_tu=None, yhat=None, cap=None, gamma=0.0):
    """Dựng tuyến tuần tự. yhat: [số cặp, E] bản đồ cạnh (0..1) theo cặp nguồn – đích; cap[k] = chỉ số cặp của luồng k.
    γ = 0 hoặc yhat = None: Dijkstra tuần tự (trọng số = trễ sau khi cộng luồng)."""
    thu_tu = np.argsort(-m.lam, kind="stable") if thu_tu is None else thu_tu
    F = np.zeros(m.E)
    tuyen = [None] * m.K
    phat = None if (yhat is None or gamma == 0) else gamma * -np.log(yhat + LN_EPS)
    for k in thu_tu:
        w = m.tre(F + m.lam[k])
        if phat is not None:
            w = w + phat[cap[k]]
        e = m.duong(m.src[k], m.dst[k], w)
        tuyen[k] = e
        F[e] += m.lam[k]
    return tuyen


def tuan_tu_kgh(m: MangDay):
    return dung_tuyen(m)


def tkcb_kgh(m: MangDay, tuyen0, rng=None, max_vong=30, dem=None):
    """Tìm kiếm cục bộ không giới hạn: lần lượt từng luồng, đổi sang đường Dijkstra theo chi phí biên nếu rẻ hơn."""
    tuyen = [np.asarray(e, np.int64) for e in tuyen0]
    F = m.tai(tuyen)
    luot = 0
    for _ in range(max_vong):
        doi = False
        for k in (rng.permutation(m.K) if rng is not None else range(m.K)):
            lk = m.lam[k]
            F[tuyen[k]] -= lk
            fF = m.f(F)
            w = (m.f(F + lk) - fF) / lk                         # chi phí biên trên một đơn vị lưu lượng
            w = np.maximum(w, 1e-9)
            e = m.duong(m.src[k], m.dst[k], w)
            luot += 1
            if len(e) and w[e].sum() < w[tuyen[k]].sum() * (1 - 1e-9) - 1e-12:
                tuyen[k], doi = e, True
            F[tuyen[k]] += lk
        if not doi:
            break
    if dem is not None:
        dem.append(luot)
    return tuyen


def tkcb_kgh_10(m: MangDay, tuyen_ung_vien=None, seed=0, R=10):
    """Mốc: 10 điểm xuất phát — Dijkstra tuần tự, lời giải TKCB 10 lần trên 8 ứng viên (nếu có), Dijkstra độc lập,
    còn lại Dijkstra tuần tự với thứ tự luồng ngẫu nhiên. Trả (tuyến tốt nhất, chi phí, chi phí của từng lần)."""
    rng = np.random.default_rng(seed)
    dau = [tuan_tu_kgh(m)]
    if tuyen_ung_vien is not None:
        dau.append(tuyen_ung_vien)
    dau.append(doc_lap_kgh(m))
    while len(dau) < R:
        dau.append(dung_tuyen(m, thu_tu=rng.permutation(m.K)))
    tot, gia, cac = None, np.inf, []
    for t0 in dau[:R]:
        t = tkcb_kgh(m, t0, rng)
        c = m.chi_phi(t)
        cac.append(c)
        if c < gia:
            tot, gia = t, c
    return tot, gia, cac


# ---------------------------------------------------------------- cặp nguồn – đích và hành lang
def cap_nguon_dich(m: MangDay):
    """Trả (danh sách cặp (s, d), chỉ số cặp của từng luồng)."""
    cap, chi_so = [], {}
    cua = np.zeros(m.K, np.int64)
    for k in range(m.K):
        key = (int(m.src[k]), int(m.dst[k]))
        if key not in chi_so:
            chi_so[key] = len(cap)
            cap.append(key)
        cua[k] = chi_so[key]
    return cap, cua


def hanh_lang(m: MangDay, cap, he_so=3.0, them_ms=20.0):
    """Cạnh (u, v) thuộc hành lang của cặp (s, d) nếu d0(s→u) + d0(u,v) + d0(v→d) ≤ 3·d0(s→d) + 20 ms.
    Trả (mask [số cặp, E], chi phí rút gọn [số cặp, E] = d0(s→u) + d0(u,v) + d0(v→d) − d0(s→d))."""
    d0 = m.tre(np.zeros(m.E))
    ds = {s: m.khoang_cach(s=s, w=d0) for s in {s for s, _ in cap}}
    dd = {d: m.khoang_cach(d=d, w=d0) for d in {d for _, d in cap}}
    mask = np.zeros((len(cap), m.E), bool)
    rc = np.zeros((len(cap), m.E))
    for c, (s, d) in enumerate(cap):
        qua = ds[s][m.eu] + d0 + dd[d][m.ev]
        ngan = ds[s][d]
        mask[c] = np.isfinite(qua) & (qua <= he_so * ngan + them_ms) & (m.ev != s) & (m.eu != d)
        rc[c] = np.where(np.isfinite(qua), qua - ngan, 1e3)
    return mask, rc


def ban_do_nhan(m: MangDay, tuyen, cap, cua):
    """Nhãn: phần nhu cầu của mỗi cặp đi qua mỗi cạnh [số cặp, E]."""
    y = np.zeros((len(cap), m.E))
    for k, e in enumerate(tuyen):
        y[cua[k], e] += m.lam[k]
    tong = np.bincount(cua, weights=m.lam, minlength=len(cap))
    return y / tong[:, None]
