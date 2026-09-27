"""
nhieu_mua.py — Mức nhiễu đo trễ do mưa và giông, suy từ vật lý (thay bảng Δ giả định của luật chia đôi).

Chuỗi năm bước:
  1. mưa trong giờ dao động:   R = R̄·exp(s·z − s²/2), z ~ N(0,1), R ≤ 200 mm/h   (E[R] ≈ R̄; s theo ITU-R P.837)
  2. suy hao:                   A = k·R^α·L·r  (+ K_l·M·L khi giông, M = L_mây/H_R)   (ITU-R P.838, P.839, P.840)
  3. tốc độ còn lại (Shannon):  c = log2(1 + SNR0·10^(−A/10)) / log2(1 + SNR0)
  4. hàng đợi M/M/1:            T ~ Exp(μ' − λ), m = E[T] = độ lệch chuẩn T = (12/(cC)) / (1 − ρ0/c)
     tải ρ0/c ≥ 0,9 coi là nghẽn (mất gói): ∫ dx/(1 − x)² phân kỳ tại x = 1 nên phải tách riêng trạng thái này
  5. nhiễu tương đối lúc còn hoạt động:  Δ = sqrt(2·E[m²] − E[m]² − m0²) / W ;  p_mất = P(nghẽn)
Trễ gốc của TV1 đúng bằng W = d/c_ánh_sáng + m0 + 0,5 với ρ0 = 0,3, nên các bước trên dùng lại đúng giả thiết của TV1.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from src.ngau_nhien.kich_ban import H_R, PKT, RHO0, suy_hao_mua

KL840 = {5: 0.0233, 12: 0.1327, 20: 0.3593, 28: 0.6788}   # ITU-R P.840 ở 0 °C, dB/(kg/m²)
L_MAY_GIONG = 3.5      # kg/m²: nước mây vượt 2% thời gian ở VN theo bản đồ P.840 (giờ giông ≈ 2,1%)
S_MUA = 1.4            # dao động mưa trong giờ, khớp mưa 1 phút vượt 0,01% của P.837 (74–104 mm/h)
R_MAX = 200.0          # mm/h: chặn trên, cỡ mưa 1 phút vượt 0,001% thời gian ở VN theo P.837 (213–243 mm/h)
SNR0_DB, NGHEN = 15.0, 0.9
Z = norm.ppf((np.arange(2048) + 0.5) / 2048)              # 2048 điểm chia đều xác suất của N(0,1)


def he_so_mua(s=S_MUA):
    """e^(s·z − s²/2) tại 256 điểm, chuẩn hóa để trung bình đúng bằng 1 (bù sai số của phép chia điểm ở đuôi)."""
    w = np.exp(s * Z - s * s / 2)
    return w / w.mean()


def ti_le_toc_do(A_db, snr0_db=SNR0_DB):
    return np.log2(1 + 10 ** ((snr0_db - A_db) / 10)) / np.log2(1 + 10 ** (snr0_db / 10))


def thoi_gian_o_lai(C, c, rho=RHO0):
    """E[T] = độ lệch chuẩn của T (ms) trong M/M/1; nan khi nghẽn (ρ0/c ≥ NGHEN)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        rc = rho / c
        return np.where(rc < NGHEN, PKT / (c * C) / np.maximum(1 - rc, 1e-9), np.nan)


def he_so_may(band):
    kl = np.zeros(np.shape(band))
    for f, v in KL840.items():
        kl = np.where(band == f, v, kl)
    return kl


def delta(R_gio, giong, band, C, Ls, LG, W, s=S_MUA, rho=RHO0, snr0_db=SNR0_DB, L_may=L_MAY_GIONG):
    """(Δ, p_mất) cho các liên kết (mảng cùng kích thước). R_gio: mưa TB giờ tại điểm của liên kết.
    Δ tính trên những lúc liên kết còn hoạt động; nan nếu nghẽn suốt giờ."""
    x = lambda a: np.asarray(a, float)[..., None]                              # noqa: E731
    R = np.minimum(x(R_gio) * he_so_mua(s), R_MAX)
    A = suy_hao_mua(R, x(band), x(Ls), x(LG)) + x(giong) * x(he_so_may(band)) * (L_may / H_R) * x(Ls)
    c = ti_le_toc_do(A, snr0_db)
    m = thoi_gian_o_lai(x(C), c, rho)
    nghen = ~np.isfinite(m)
    m0 = thoi_gian_o_lai(np.asarray(C, float), 1.0, rho)
    ok = np.isfinite(m)
    n = np.maximum(ok.sum(-1), 1)
    e1, e2 = np.where(ok, m, 0).sum(-1) / n, np.where(ok, m * m, 0).sum(-1) / n
    d = np.sqrt(np.maximum(2 * e2 - e1 ** 2 - m0 ** 2, 0)) / np.asarray(W, float)
    return np.where(ok.any(-1), d, np.nan), nghen.mean(-1)


def nhieu_nen(C, W, rho=RHO0):
    """Nhiễu tương đối của một lần đo khi trời quang: m0 / W."""
    return thoi_gian_o_lai(np.asarray(C, float), 1.0, rho) / np.asarray(W, float)
