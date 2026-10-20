"""
doi_chung.py — Các phương pháp đối chứng cho bài toán nhiều luồng (xem results/yagi/TIEU_CHI.md).

  doc_lap        Dijkstra độc lập: mỗi luồng tuyến ngắn nhất trên trễ nền (= ứng viên đầu)
  tuan_tu        Dijkstra tuần tự: luồng lớn trước, Dijkstra trên trễ đã cộng tải các luồng trước (không giới hạn ứng viên)
  tim_cuc_bo     Tìm kiếm cục bộ trên các ứng viên: đổi tuyến từng luồng tới khi không cải thiện
  ls_nhieu_lan   Tìm kiếm cục bộ R lần (điểm xuất phát khác nhau), lấy lời giải tốt nhất → "tốt nhất tìm được"
  frank_wolfe    Cận dưới khi cho phép chia luồng trên mọi tuyến (bài toán lồi)
"""

from __future__ import annotations

import numpy as np

from src.ngau_nhien.khoang_loi import dijkstra


def doc_lap(th):
    return np.zeros(th.K, int)


def tuan_tu(th):
    """Trả (danh sách tuyến, tải F)."""
    F = np.zeros(th.N * th.N)
    tuyen = [None] * th.K
    for k in np.argsort(-th.lam):
        W = np.where(th.song, th.tre(F + th.lam[k]), 0.0).reshape(th.N, th.N)
        r = dijkstra(W, int(th.src[k]), int(th.dst[k]), th.can_relay) or th.ung_vien[k][0]
        tuyen[k] = r
        F[np.array(r[:-1]) * th.N + np.array(r[1:])] += th.lam[k]
    return tuyen, F


def _vec(th):
    """Mảng đệm theo từng luồng [K, 8, L] để tính chi phí biên của cả 8 ứng viên trong một lần (nhanh hơn ~3×)."""
    if getattr(th, "_vec_cache", None) is None:
        m = (th.canh >= 0) & th.hop_le[:, :, None]
        Ei = np.where(m, th.canh, 0)
        th._vec_cache = (Ei, m, th.prop[Ei], th.C[Ei], th.rho0[Ei])
    return th._vec_cache


def _f_dem(F, p, C, r):
    from src.nhieu_luong.bai_toan import h_hang_doi
    from src.ngau_nhien.kich_ban import PKT, PROC
    return F * (p + PKT / C * (1 + h_hang_doi(r + F / C)) + PROC)


def _gia_ung_vien(th, F, k):
    """Chi phí biên Σ_e [f(F_e + λ_k) − f(F_e)] của từng ứng viên của luồng k (∞ nếu ứng viên không tồn tại)."""
    Ei, m, p, C, r = _vec(th)
    Fe, lk = F[Ei[k]], th.lam[k]
    d = np.where(m[k], _f_dem(Fe + lk, p[k], C[k], r[k]) - _f_dem(Fe, p[k], C[k], r[k]), 0.0).sum(1)
    return np.where(th.hop_le[k], d, np.inf)


def tham_lam_ung_vien(th):
    """Như tuan_tu nhưng chỉ chọn trong ứng viên (dùng làm điểm xuất phát cho tìm kiếm cục bộ)."""
    Ei, m = _vec(th)[:2]
    F = np.zeros(th.N * th.N)
    a = np.zeros(th.K, int)
    for k in np.argsort(-th.lam):
        a[k] = int(np.argmin(_gia_ung_vien(th, F, k)))
        F[Ei[k, a[k]][m[k, a[k]]]] += th.lam[k]
    return a


def tim_cuc_bo(th, a0, rng=None, max_vong=30, dem=None):
    """Đổi tuyến từng luồng (best response trên ứng viên) tới khi một vòng không còn cải thiện.
    dem: list (nếu có) — nhận số lượt đánh giá luồng đã dùng."""
    Ei, m = _vec(th)[:2]
    a = np.array(a0, int).copy()
    F = th.tai_tu_chon(a)
    luot = 0
    for _ in range(max_vong):
        doi = False
        for k in (rng.permutation(th.K) if rng is not None else range(th.K)):
            e_cu = Ei[k, a[k]][m[k, a[k]]]
            F[e_cu] -= th.lam[k]
            gia = _gia_ung_vien(th, F, k)
            luot += 1
            c_moi = int(np.argmin(gia))
            if c_moi != a[k] and gia[c_moi] < gia[a[k]] - 1e-9:
                a[k] = c_moi
                doi = True
            F[Ei[k, a[k]][m[k, a[k]]]] += th.lam[k]
        if not doi:
            break
    if dem is not None:
        dem.append(luot)
    return a


def ls_nhieu_lan(th, R=10, seed=0):
    """R lần: từ Dijkstra độc lập, từ tham lam trên ứng viên, còn lại ngẫu nhiên. Trả (lời giải tốt nhất, chi phí)."""
    rng = np.random.default_rng(seed)
    diem_dau = [doc_lap(th), tham_lam_ung_vien(th)]
    while len(diem_dau) < R:
        diem_dau.append(np.array([rng.choice(np.flatnonzero(th.hop_le[k])) for k in range(th.K)]))
    tot, gia = None, np.inf
    for a0 in diem_dau[:R]:
        a = tim_cuc_bo(th, a0, rng)
        c = th.chi_phi(a)
        if c < gia:
            tot, gia = a, c
    return tot, gia


def frank_wolfe(th, so_vong=40):
    """Cận dưới (cho phép chia luồng trên mọi tuyến). Trả (cận dưới, chi phí lời giải chia luồng tốt nhất)."""
    cap = {}
    for s, d, l in zip(th.src, th.dst, th.lam):
        cap[(int(s), int(d))] = cap.get((int(s), int(d)), 0.0) + l
    tong = th.lam.sum()

    def aon(gia):
        W = np.where(th.song, gia, 0.0).reshape(th.N, th.N)
        y = np.zeros(th.N * th.N)
        for (s, d), l in cap.items():
            r = dijkstra(W, s, d, th.can_relay)
            y[np.array(r[:-1]) * th.N + np.array(r[1:])] += l
        return y

    def tong_f(F):
        dung = F > 0
        return th.f(F[dung], np.flatnonzero(dung)).sum()

    x = aon(th.tre(np.zeros(th.N * th.N)))
    can_duoi, tot = -np.inf, tong_f(x)
    for _ in range(so_vong):
        m = th.bien(x)
        y = aon(m)
        fx = tong_f(x)
        gap = float((m * (x - y)).sum())
        can_duoi = max(can_duoi, fx - gap)
        a, b = 0.0, 1.0                                                  # tìm γ theo tỉ lệ vàng
        for _ in range(25):
            g1, g2 = a + 0.382 * (b - a), a + 0.618 * (b - a)
            if tong_f(x + g1 * (y - x)) < tong_f(x + g2 * (y - x)):
                b = g2
            else:
                a = g1
        x = x + (a + b) / 2 * (y - x)
        tot = min(tot, tong_f(x))
        if gap < 1e-4 * fx:
            break
    return can_duoi / tong, tot / tong
