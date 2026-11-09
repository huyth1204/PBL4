"""
the_gioi_rr.py — Thế giới mô phỏng RẼ NHÁNH tại mốc quan sát o (thí nghiệm 4, results/bao_da_tuyen/TIEU_CHI_4.md).

Một thế giới rẽ nhánh giữ nguyên mọi biến ngẫu nhiên tới o của kịch bản quan sát `sinh_kich_ban(seed)`, chỉ bốc lại
phần sau o: nhiễu mưa dưới giờ AR(1) tiếp nối từ trạng thái ở o, và sét ngắt nút mặt đất tiếp nối từ số mốc ngắt còn
lại ở o. UAV hạ cánh do giông / gió giật theo thời tiết giờ nên giống mọi thế giới. Bộ `tao_du_lieu_bao` có sẵn dùng
seed độc lập từ đầu (quá khứ cũng khác) nên không dùng được làm nhãn có điều kiện theo quan sát.

Chỉ tính cho cửa sổ ngắn và các cạnh cần (thay vì 600 snapshot × N×N như sinh_kich_ban), vật lý chép đúng
sinh_kich_ban ở cấu hình xung_dot = False, tai = False (bắt buộc theo bảng rủi ro):
    R = P·exp(z − σ²/2) → suy hao ITU-R P.838 → SNR → dung lượng C2 (đứt nếu SNR < ngưỡng) → ρ = 0,3·C/C2
    sống = có trong lịch quỹ đạo & không đứt do mưa & hai đầu không tắt
"""

from __future__ import annotations

import numpy as np

from src.ngau_nhien.kich_ban import RHO0, hong_tich_luy, r_hong_cua, suy_hao_mua, xac_suat_hong_set
from src.ngau_nhien.thoi_tiet import GIONG


def _kiem(p):
    if p.xung_dot or p.tai or not p.mua or not p.giong:
        raise NotImplementedError("Rẽ nhánh chỉ hỗ trợ mua = giong = True, xung_dot = tai = False")


def phat_lai(g, tt, h0, seed, p):
    """Phát lại phần ngẫu nhiên của sinh_kich_ban(g, tt, h0, seed, p) ĐÚNG thứ tự bốc số.
    Trả dict: P [T,S] mưa nội suy, z [T,S] nhiễu mưa, tat [T,N], con [T,N] số mốc ngắt còn lại sau mốc t (nút mặt
    đất), uav_tat [T,N] (hạ cánh, tất định)."""
    _kiem(p)
    T, N, S, dt_h = g.T, g.N, g.S, p.dt_s / 3600.0
    r_mua, r_giong, _, _ = (np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(4))
    gio = h0 + np.arange(T) * dt_h
    i0 = np.floor(gio).astype(int)
    w = (gio - i0)[:, None]
    P = (1 - w) * tt.precip[i0] + w * tt.precip[np.minimum(i0 + 1, tt.H - 1)]
    giong_diem = np.isin(tt.code[i0], GIONG)
    gust = tt.gust[i0]
    z = _ar1_day(r_mua, T, S, p.phi_mua, p.sigma_mua)
    tat = np.zeros((T, N), bool)
    con_sau = np.zeros((T, N), int)
    uav_tat = np.zeros((T, N), bool)
    nguong = dict(p.uav_gio_nguong)                    # thí nghiệm 8: ngưỡng gió theo loại UAV
    for n in range(N):
        s = g.node_site[n]
        if s < 0:
            continue
        if g.kind[n] == "uav":
            uav_tat[:, n] = giong_diem[:, s] | (gust[:, s] > nguong.get(g.names[n], p.gio_giat_max))
            tat[:, n] |= uav_tat[:, n]
        elif g.dat[n]:
            con = 0
            for t in range(T):
                if con == 0 and giong_diem[t, s] and r_giong.random() < p.p_set:
                    con = r_giong.geometric(1 / p.set_tb)
                if con > 0:
                    tat[t, n], con = True, con - 1
                con_sau[t, n] = con
    hong, ph = np.zeros((T, N), bool), None
    if p.hong_b:                                       # thí nghiệm 8: đúng như sinh_kich_ban (cùng luồng, cùng thứ tự)
        ph = xac_suat_hong_set(g, giong_diem, p, dt_h)
        hong = hong_tich_luy(ph, r_hong_cua(seed))
        tat |= hong
    return {"P": P, "z": z, "tat": tat, "con": con_sau, "uav_tat": uav_tat, "giong_diem": giong_diem,
            "hong": hong, "ph": ph}


def _ar1_day(rng, T, n, phi, sigma):
    """Như kich_ban._ar1 (cùng thứ tự bốc số)."""
    x = np.zeros((T, n))
    x[0] = rng.normal(0, sigma, n)
    e = rng.normal(0, 1, (T, n)) * sigma * np.sqrt(1 - phi ** 2)
    for t in range(1, T):
        x[t] = phi * x[t - 1] + e[t]
    return x


def re_nhanh(g, pl, o, t_cuoi, seeds, p):
    """Bốc lại tương lai (o, t_cuoi] cho từng seed. Trả z [W, T', S] và tat [W, T', N] cho các mốc o+1..t_cuoi."""
    S, N = g.S, g.N
    L = t_cuoi - o
    zs, tats = np.zeros((len(seeds), L, S)), np.zeros((len(seeds), L, N), bool)
    dat = [n for n in range(N) if g.node_site[n] >= 0 and g.kind[n] != "uav" and g.dat[n]]
    for i, sd in enumerate(seeds):
        r_mua, r_giong = (np.random.default_rng(s) for s in np.random.SeedSequence(sd).spawn(2))
        e = r_mua.normal(0, 1, (L, S)) * p.sigma_mua * np.sqrt(1 - p.phi_mua ** 2)
        x = pl["z"][o]
        for j in range(L):
            x = p.phi_mua * x + e[j]
            zs[i, j] = x
        tats[i] = pl["uav_tat"][o + 1:t_cuoi + 1]
        for n in dat:
            s, con = g.node_site[n], int(pl["con"][o, n])
            for j in range(L):
                if con == 0 and pl["giong_diem"][o + 1 + j, s] and r_giong.random() < p.p_set:
                    con = r_giong.geometric(1 / p.set_tb)
                if con > 0:
                    tats[i, j, n], con = True, con - 1
        if p.hong_b:                                   # thí nghiệm 8: nút đã hỏng ở o giữ hỏng; tương lai bốc lại
            r_hong = np.random.default_rng(np.random.SeedSequence(sd).spawn(3)[2])
            tats[i] |= hong_tich_luy(pl["ph"][o + 1:t_cuoi + 1], r_hong, hong0=pl["hong"][o])
    return zs, tats


def trang_thai_canh(g, pl, eu, ev, cac_t, z, tat, p):
    """Trạng thái các cạnh eu → ev ở các mốc cac_t, cho W thế giới.
    z [W, len(cac_t), S], tat [W, len(cac_t), N]. Trả song, C2, rho đều [W, len(cac_t), E]."""
    cac_t = np.asarray(cac_t)
    site = g.site[eu, ev]
    co_site = site >= 0
    P = pl["P"][cac_t][:, np.maximum(site, 0)]                                  # [T', E]
    R = P[None] * np.exp(z[:, :, np.maximum(site, 0)] - p.sigma_mua ** 2 / 2)   # [W, T', E]
    band = g.band[eu, ev][None, None]
    Ls, LG = g.Ls[cac_t][:, eu, ev][None], g.LG[cac_t][:, eu, ev][None]
    A = np.where(co_site[None, None], suy_hao_mua(np.where(co_site[None, None], R, 0.0), band, Ls, LG), 0.0)
    snr = p.snr0_db - A
    dut = snr < p.snr_min_db
    C = g.C[eu, ev][None, None]
    C2 = C * np.log2(1 + 10 ** (snr / 10)) / np.log2(1 + 10 ** (p.snr0_db / 10))
    C2 = np.where(dut, C, np.maximum(C2, 1e-3))
    rho = RHO0 * C / C2
    song = (g.W[cac_t][:, eu, ev] > 0)[None] & ~dut & ~tat[:, :, eu] & ~tat[:, :, ev]
    return song, C2, rho


def the_gioi(g, pl, o, t, delta, eu, ev, seeds, p):
    """W thế giới rẽ nhánh cho cửa sổ [t, t+delta]: song [W, E] (sống suốt cửa sổ), C2 / rho [W, delta+1, E]."""
    zs, tats = re_nhanh(g, pl, o, t + delta, seeds, p)
    k = t - o - 1                                                                # mốc t trong mảng o+1..
    song, C2, rho = trang_thai_canh(g, pl, eu, ev, np.arange(t, t + delta + 1), zs[:, k:], tats[:, k:], p)
    return song.all(1), C2, rho
