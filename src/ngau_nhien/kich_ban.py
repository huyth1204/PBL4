"""
kich_ban.py — Sinh kịch bản ngẫu nhiên cho SAGSIN từ ma trận trễ tất định của TV1.

Trễ gốc của TV1:  W = d_lan_truyen + 12/C + (12/C)·ρ0/(1−ρ0) + 0,5   (ms; C tính bằng Mbps, ρ0 = 0,3)
Ở đây tách lại d_lan_truyen (suy ra khoảng cách, góc ngẩng), rồi dựng lại W theo bốn nguồn ngẫu nhiên:
  mua      mưa thật (Open-Meteo) + dao động dưới giờ → suy hao ITU-R P.838 → giảm dung lượng / đứt liên kết
  giong    giông: UAV phải hạ cánh; sét làm nút mặt đất ngắt tạm thời
  xung_dot sự cố xung đột / an ninh: gây nhiễu vùng, mất nút, tăng tải khẩn cấp (thuần ngẫu nhiên, kéo dài)
  tai      tải theo giờ trong ngày + AR(1) → trễ hàng đợi M/M/1 (ghép với dung lượng đã giảm do mưa, nhiễu)
Tắt cả bốn nguồn thì W dựng lại trùng W gốc.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from src.ngau_nhien.thoi_tiet import GIONG, ThoiTiet

C_KM_MS = 300.0                       # quãng đường ánh sáng đi trong 1 ms
PKT = 12.0                            # gói 1500 byte = 12 kbit → thời gian phục vụ 12/C ms
RHO0, PROC = 0.3, 0.5
R_E, H_R = 6371.0, 5.0                # bán kính Trái Đất; độ cao lớp mưa (ITU-R P.839 cho VN ≈ 5,0–5,2 km)
ALT = {"leo": 550.0, "haps": 20.0, "uav": 2.0, "gateway": 0.0, "user": 0.0}
# ITU-R P.838-3, tính bằng ITU-Rpy (phân cực tròn, góc ngẩng 45°): tần số GHz → (k, α)
ITU838 = {5: (0.000229, 1.6095), 12: (0.02420, 1.1516), 20: (0.09388, 1.0199), 28: (0.2008, 0.9482)}
TRONG_SO = {"Hanoi": 1.3, "HoChiMinh": 1.3, "HAPS-North": 1.2, "HAPS-South": 1.2}


def loai_lien_ket(ka: str, kb: str):
    """(tên, dung lượng Mbps, băng tần GHz hoặc None nếu không chịu mưa)."""
    s = {ka, kb}
    if s == {"leo"}:
        return "ISL", 1000.0, None
    if "leo" in s:
        o = (s - {"leo"}).pop()
        return {"haps": ("ASL", 100.0, None), "uav": ("ASL", 100.0, 12),
                "gateway": ("GSL", 100.0, 20), "user": ("GSL", 100.0, 12)}[o]
    if "haps" in s:
        o = (s - {"haps"}).pop() if len(s) > 1 else "haps"
        return {"haps": ("HAPS-HAPS", 500.0, None), "uav": ("HAPS-UAV", 200.0, 5)}.get(o, ("HAPS-MĐ", 300.0, 28))
    return ("UAV-UAV", 100.0, 5) if s == {"uav"} else ("UAV-MĐ", 100.0, 5)


@dataclass
class ThamSo:
    mua: bool = True
    giong: bool = True
    xung_dot: bool = True
    tai: bool = True
    dt_s: float = 20.0
    # mưa
    phi_mua: float = float(np.exp(-20 / 300))      # tương quan dưới giờ ~5 phút
    sigma_mua: float = 0.8
    snr0_db: float = 15.0
    snr_min_db: float = 0.0
    # giông
    gio_giat_max: float = 15.0                      # m/s, UAV hạ cánh khi vượt
    uav_ha_canh: bool = True                        # tách "hạ cánh an toàn" khỏi "mất nút"
    p_set: float = 0.03                             # xác suất sét gây ngắt / snapshot khi có giông
    set_tb: float = 3.0                             # thời gian ngắt trung bình (snapshot)
    # xung đột
    lam_xd: float = 0.1                             # số sự cố / giờ
    p_loai_xd: tuple = (0.5, 0.25, 0.25)            # gây nhiễu, mất nút, tăng tải
    ban_kinh_km: float = 150.0
    nhieu_db: tuple = (6.0, 20.0)
    tb_nhieu_phut: float = 60.0
    tb_mat_nut_phut: float = 120.0
    tang_tai: tuple = (0.5, 1.5)
    # tải
    rho_tb: float = 0.5
    bien_do_ngay: float = 0.3
    gio_dinh: float = 20.0
    phi_tai: float = float(np.exp(-20 / 600))
    sigma_tai: float = 0.3
    k_dem: float = 250.0                            # bộ đệm (gói) khi nghẽn


def _haversine(la1, lo1, la2, lo2):
    la1, lo1, la2, lo2 = map(np.radians, (la1, lo1, la2, lo2))
    a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * R_E * np.arcsin(np.sqrt(a))


class MangGoc:
    """Tách W tất định của TV1 thành các thành phần cố định theo loại liên kết."""

    def __init__(self, W: np.ndarray, kind, names, tt: ThoiTiet):
        self.W = W.astype(np.float64)
        self.T, self.N = W.shape[0], W.shape[1]
        self.kind, self.names = [str(k) for k in kind], [str(n) for n in names]
        N = self.N
        self.C = np.ones((N, N))
        self.band = np.zeros((N, N), int)
        self.ten = np.full((N, N), "", dtype=object)
        self.site = -np.ones((N, N), int)          # điểm thời tiết quyết định mưa trên liên kết
        self.h1, self.h2 = np.zeros((N, N)), np.zeros((N, N))
        self.node_site = np.array([tt.idx(n) if n in tt.names else -1 for n in self.names])
        for u in range(N):
            for v in range(N):
                if u == v:
                    continue
                ten, C, band = loai_lien_ket(self.kind[u], self.kind[v])
                self.ten[u, v], self.C[u, v], self.band[u, v] = ten, C, band or 0
                a, b = sorted((u, v), key=lambda x: ALT[self.kind[x]])
                self.h1[u, v], self.h2[u, v] = ALT[self.kind[a]], ALT[self.kind[b]]
                if band:
                    ground = [x for x in (u, v) if self.kind[x] in ("gateway", "user")]
                    self.site[u, v] = self.node_site[ground[0] if ground else a]
        co = self.W > 0
        tre_co_dinh = PKT / self.C * (1 + RHO0 / (1 - RHO0)) + PROC
        self.prop = np.where(co, self.W - tre_co_dinh[None], 0.0)
        self.d = self.prop * C_KM_MS
        r1, r2 = R_E + self.h1[None], R_E + self.h2[None]
        with np.errstate(divide="ignore", invalid="ignore"):
            s = (r2 ** 2 - r1 ** 2 - self.d ** 2) / (2 * r1 * self.d)
        self.sin_el = np.clip(np.nan_to_num(s, nan=1.0), np.sin(np.radians(1.0)), 1.0)
        mat_dat = (self.h2 < H_R)[None] & co                     # cả hai đầu dưới lớp mưa: đường gần nằm ngang
        Ls = np.where(mat_dat, self.d, np.maximum(H_R - self.h1[None], 0) / self.sin_el)
        self.Ls = np.where(co, Ls, 0.0)
        self.LG = np.where(mat_dat, self.d, self.Ls * np.sqrt(1 - self.sin_el ** 2))
        S = len(tt.names)
        self.kc_diem = _haversine(tt.lat[:, None], tt.lon[:, None], tt.lat[None], tt.lon[None])   # [S,S] km
        self.trong_so = np.array([TRONG_SO.get(n, 1.0) for n in self.names])
        self.dat = np.array([k in ("gateway", "user") for k in self.kind])
        self.S = S


def suy_hao_mua(R, band, Ls, LG):
    """A (dB) = k R^α · Ls · r,  r = 1/(1 + LG/L0),  L0 = 35·e^{−0,015 R}."""
    k = np.zeros_like(Ls)
    a = np.ones_like(Ls)
    for f, (kf, af) in ITU838.items():
        m = band == f
        k = np.where(m, kf, k)
        a = np.where(m, af, a)
    Rc = np.clip(R, 0, None)
    L0 = 35.0 * np.exp(-0.015 * np.minimum(Rc, 100.0))
    return k * Rc ** a * Ls / (1 + LG / L0)


def _ar1(rng, T, n, phi, sigma):
    x = np.zeros((T, n))
    x[0] = rng.normal(0, sigma, n)
    e = rng.normal(0, 1, (T, n)) * sigma * np.sqrt(1 - phi ** 2)
    for t in range(1, T):
        x[t] = phi * x[t - 1] + e[t]
    return x


def sinh_kich_ban(g: MangGoc, tt: ThoiTiet, h0: int, seed: int, p: ThamSo = ThamSo()):
    """Một kịch bản dài g.T snapshot, bắt đầu ở giờ h0 của kho thời tiết. Trả (W mới, quan sát)."""
    T, N, S, dt_h = g.T, g.N, g.S, p.dt_s / 3600.0
    r_mua, r_giong, r_xd, r_tai = (np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(4))
    gio = h0 + np.arange(T) * dt_h
    i0 = np.floor(gio).astype(int)
    w = (gio - i0)[:, None]
    P = (1 - w) * tt.precip[i0] + w * tt.precip[np.minimum(i0 + 1, tt.H - 1)]                  # [T,S]
    giong_diem = np.isin(tt.code[i0], GIONG)
    gust = tt.gust[i0]
    gio_vn = (tt.gio_utc[i0] + (gio - i0) + 7) % 24

    # 1) mưa → suy hao
    z = _ar1(r_mua, T, S, p.phi_mua, p.sigma_mua)
    R = P * np.exp(z - p.sigma_mua ** 2 / 2) if p.mua else np.zeros((T, S))
    Rl = np.where(g.site[None] >= 0, R[:, np.maximum(g.site, 0)], 0.0)                         # [T,N,N]
    A = np.where(g.site[None] >= 0, suy_hao_mua(Rl, g.band[None], g.Ls, g.LG), 0.0)

    # 2) giông: UAV hạ cánh, sét ngắt nút mặt đất
    #
    # `tat`  = nút KHÔNG dùng được (mất liên kết) — dùng để cắt cạnh trong W.
    # `ha_canh` = UAV HẠ CÁNH AN TOÀN: vẫn tồn tại (không bị "mất"), nhưng không
    #   relay được. Tách riêng để model biết "UAV đang chờ" khác "nút hỏng":
    #   hai tình huống này có cách xử lý khác nhau (chờ gió lặng / bỏ qua nút).
    # Với UAV, cả hai đều đặt để giữ nguyên hành vi cắt cạnh như trước; khác biệt
    # chỉ nằm ở nhãn trong `quan_sat`, không làm đổi W (an toàn cho hồi quy).
    tat = np.zeros((T, N), bool)
    ha_canh = np.zeros((T, N), bool)
    if p.giong:
        for n in range(N):
            s = g.node_site[n]
            if s < 0:
                continue
            if g.kind[n] == "uav":
                if p.uav_ha_canh:
                    # Hạ cánh do giông HOẶC gió giật vượt ngưỡng (2 nguyên nhân tách biệt).
                    do_giong = giong_diem[:, s] | (gust[:, s] > p.gio_giat_max)
                    ha_canh[:, n] |= do_giong
                    tat[:, n] |= do_giong
                else:
                    tat[:, n] |= giong_diem[:, s] | (gust[:, s] > p.gio_giat_max)
            elif g.dat[n]:
                con = 0
                for t in range(T):
                    if con == 0 and giong_diem[t, s] and r_giong.random() < p.p_set:
                        con = r_giong.geometric(1 / p.set_tb)
                    if con > 0:
                        tat[t, n], con = True, con - 1

    # 3) xung đột / an ninh
    J_diem = np.zeros((T, S))
    tang = np.ones((T, N))
    su_co = []
    if p.xung_dot:
        dich_mat_nut = [n for n in range(N) if g.kind[n] in ("gateway", "haps", "uav")]
        for t in range(T):
            if r_xd.random() >= p.lam_xd * dt_h:
                continue
            loai = r_xd.choice(3, p=p.p_loai_xd)
            if loai == 1:
                n = int(r_xd.choice(dich_mat_nut))
                dai = r_xd.geometric(dt_h * 60 / p.tb_mat_nut_phut)
                tat[t:t + dai, n] = True
                su_co.append(("mất nút", t, int(dai), g.names[n]))
                continue
            tam = int(r_xd.choice([g.node_site[n] for n in range(N) if g.dat[n]]))
            vung = g.kc_diem[tam] <= p.ban_kinh_km
            dai = r_xd.geometric(dt_h * 60 / p.tb_nhieu_phut)
            if loai == 0:
                J_diem[t:t + dai, vung] += r_xd.uniform(*p.nhieu_db)
                su_co.append(("gây nhiễu", t, int(dai), tt.names[tam]))
            else:
                trong_vung = np.array([g.node_site[n] >= 0 and vung[g.node_site[n]] for n in range(N)])
                tang[t:t + dai, trong_vung] *= 1 + r_xd.uniform(*p.tang_tai)
                su_co.append(("tăng tải", t, int(dai), tt.names[tam]))
    J = np.where(g.site[None] >= 0, J_diem[:, np.maximum(g.site, 0)], 0.0)

    # 4) dung lượng còn lại
    snr = p.snr0_db - A - J
    dut = snr < p.snr_min_db
    C2 = g.C[None] * np.log2(1 + 10 ** (snr / 10)) / np.log2(1 + 10 ** (p.snr0_db / 10))
    C2 = np.where(dut, g.C[None], np.maximum(C2, 1e-3))

    # 5) tải và hàng đợi
    if p.tai:
        x = _ar1(r_tai, T, N, p.phi_tai, p.sigma_tai)
        nhip = 1 + p.bien_do_ngay * np.cos(2 * np.pi * (gio_vn[:, None] - p.gio_dinh) / 24)
        L = g.trong_so[None] * nhip * np.exp(x - p.sigma_tai ** 2 / 2) * tang
        rho = p.rho_tb * L[:, :, None] * g.C[None] / C2
    else:
        L = tang
        rho = RHO0 * tang[:, :, None] * g.C[None] / C2
    with np.errstate(divide="ignore"):
        hs = np.where(rho < 1, np.minimum(rho / (1 - rho), p.k_dem), p.k_dem)
    Wn = g.prop + PKT / C2 + PKT / C2 * hs + PROC
    song = (g.W > 0) & ~dut & ~tat[:, :, None] & ~tat[:, None, :]
    Wn = np.where(song, Wn, -1.0).astype(np.float32)
    idx = np.arange(N)
    Wn[:, idx, idx] = g.W[:, idx, idx]
    quan_sat = {"R": R.astype(np.float32), "giong": giong_diem, "gust": gust.astype(np.float32),
                "gio_vn": gio_vn.astype(np.float32), "J": J_diem.astype(np.float32),
                "tai": L.astype(np.float32), "tat": tat, "ha_canh": ha_canh,
                "su_co": su_co,
                "ti_le_dut_mua": float(np.mean(dut[g.W > 0])),
                "C2": C2.astype(np.float32), "rho": rho.astype(np.float32), "song": song}   # cho bài toán nhiều luồng
    return Wn, quan_sat


def tham_so_dict(p: ThamSo):
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(p).items()}
