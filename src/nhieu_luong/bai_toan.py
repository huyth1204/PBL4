"""
bai_toan.py — Bài toán định tuyến đồng thời nhiều luồng khẩn cấp trong SAGSIN khi có bão (xem results/yagi/TIEU_CHI.md).

Một tình huống = một snapshot t của một kịch bản thời tiết thật + K luồng khẩn cấp (nguồn: các nút mặt đất bị bão nặng
nhất lúc đó; đích: các gateway ngoài vùng bão). Mỗi luồng có tối đa 8 tuyến ứng viên (Yen, trên trễ nền).
Độ trễ một liên kết phụ thuộc tổng tải đi qua nó (hàng đợi M/M/1, dung lượng đã giảm do mưa):
    d_e(F) = truyền lan + PKT/C·(1 + h(ρ)) + xử lý,   ρ = ρ_nền + F/C,
    h(ρ) = ρ/(1−ρ) khi ρ ≤ 0,99, sau đó kéo dài tuyến tính (giữ tính lồi).
Chi phí của một lời giải = Σ_e F_e·d_e(F_e) / Σ λ  (ms, trễ trung bình theo lưu lượng).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.ai.nhieu_tuyen import yen_near_optimal
from src.ngau_nhien.kich_ban import PKT, PROC

RHO_GAY = 0.99
# Mức tải = phần dung lượng CÒN TRỐNG (C·(1−ρ_nền)) bị các luồng dùng hết tại liên kết nghẽn nhất khi chọn Dijkstra
# độc lập (sửa ngày 18/10/2026, trước khi có kết quả — xem results/yagi/TIEU_CHI.md, mục "Thay đổi")
MUC_TAI = {"nhe": 0.60, "vua": 0.90, "nang": 1.20}
SO_UNG_VIEN = 8


def h_hang_doi(rho):
    return np.where(rho <= RHO_GAY, rho / (1 - np.minimum(rho, RHO_GAY)), RHO_GAY / (1 - RHO_GAY)
                    + (rho - RHO_GAY) / (1 - RHO_GAY) ** 2)


def dh_hang_doi(rho):
    return np.where(rho <= RHO_GAY, 1 / (1 - np.minimum(rho, RHO_GAY)) ** 2, 1 / (1 - RHO_GAY) ** 2)


@dataclass
class TinhHuong:
    N: int
    can_relay: np.ndarray            # [N] nút được chuyển tiếp
    song: np.ndarray                 # [N*N] liên kết có hướng còn sống
    prop: np.ndarray                 # [N*N] ms
    C: np.ndarray                    # [N*N] Mbps (đã giảm do mưa / nhiễu)
    rho0: np.ndarray                 # [N*N] tải nền
    src: np.ndarray                  # [K]
    dst: np.ndarray                  # [K]
    lam: np.ndarray                  # [K] Mbps (đã nhân hệ số mức tải)
    ung_vien: list                   # K danh sách tuyến (danh sách nút)
    canh: np.ndarray                 # [K, 8, Lmax] chỉ số cạnh có hướng, -1 = trống
    hop_le: np.ndarray               # [K, 8] ứng viên có tồn tại
    meta: dict = field(default_factory=dict)

    @property
    def K(self):
        return len(self.lam)

    # ---- trễ và chi phí
    def tre(self, F, e=None):
        p, C, r = (self.prop, self.C, self.rho0) if e is None else (self.prop[e], self.C[e], self.rho0[e])
        return p + PKT / C * (1 + h_hang_doi(r + F / C)) + PROC

    def f(self, F, e=None):
        return F * self.tre(F, e)

    def bien(self, F):
        """Chi phí biên d/dF [F·d(F)] = d + F·d'."""
        return self.tre(F) + F * PKT / self.C ** 2 * dh_hang_doi(self.rho0 + F / self.C)

    def tai_tu_chon(self, a):
        F = np.zeros(self.N * self.N)
        for k, c in enumerate(a):
            e = self.canh[k, c]
            F[e[e >= 0]] += self.lam[k]
        return F

    def tai_tu_tuyen(self, tuyen):
        F = np.zeros(self.N * self.N)
        for k, r in enumerate(tuyen):
            F[np.array(r[:-1]) * self.N + np.array(r[1:])] += self.lam[k]
        return F

    def chi_phi_F(self, F):
        dung = F > 0
        return float(self.f(F[dung], np.flatnonzero(dung)).sum() / self.lam.sum())

    def chi_phi(self, a):
        return self.chi_phi_F(self.tai_tu_chon(a))

    def qua_tai_F(self, F):
        """Tỉ lệ lưu lượng (tính theo λ·số cạnh) đi qua liên kết có ρ ≥ 1."""
        dung = F > 0
        rho = self.rho0[dung] + F[dung] / self.C[dung]
        return float((F[dung] * (rho >= 1)).sum() / F[dung].sum())

    def W_tai(self, F):
        """Ma trận trễ N×N khi đã có tải F (0 = không có liên kết) — cho Dijkstra."""
        return np.where(self.song, self.tre(F), 0.0).reshape(self.N, self.N)


def diem_bao(g, q, t):
    """Mức bão của từng nút mặt đất lúc t: mưa (mm/h) + phần gió giật vượt 15 m/s."""
    d = np.full(g.N, -1.0)
    for n in range(g.N):
        s = g.node_site[n]
        if g.kind[n] in ("gateway", "user") and s >= 0:
            d[n] = q["R"][t, s] + max(0.0, q["gust"][t, s] - 15.0)
    return d


def lien_lac_duoc(W, can_relay):
    """[N,N] True nếu có đường s → d theo luật chuyển tiếp (BFS từ từng nút)."""
    N = len(W)
    ke = [np.flatnonzero(W[u] > 0) for u in range(N)]
    R = np.zeros((N, N), bool)
    for s in range(N):
        thay, hang = {s}, [s]
        while hang:
            u = hang.pop()
            if u != s and not can_relay[u]:
                continue
            for v in ke[u]:
                if v not in thay:
                    thay.add(v)
                    hang.append(v)
        R[s, list(thay)] = True
    return R


def tao_tinh_huong(g, q, t, rng, so_luong=(20, 40), so_nguon=3, cache=None):
    """Sinh một bộ luồng + tuyến ứng viên ở snapshot t; trả {mức tải: TinhHuong} (cùng luồng, khác hệ số)."""
    N = g.N
    prop = g.prop[t].reshape(-1)
    C = q["C2"][t].reshape(-1).astype(float)
    rho0 = q["rho"][t].reshape(-1).astype(float)
    song_vl = q["song"][t].reshape(-1)
    song = song_vl & (rho0 < 1.0)          # tải nền ≥ 1: không còn dung lượng trống → không dùng cho luồng khẩn cấp
    can_relay = np.array([k != "user" for k in g.kind])
    th0 = TinhHuong(N, can_relay, song, prop, C, rho0, np.zeros(0, int), np.zeros(0, int), np.zeros(0),
                    [], np.zeros((0, SO_UNG_VIEN, 1), int), np.zeros((0, SO_UNG_VIEN), bool))
    W0 = th0.W_tai(np.zeros(N * N))
    diem = diem_bao(g, q, t)
    toi_duoc = lien_lac_duoc(W0, can_relay)                         # [N,N] có đường s → d hay không
    gw_all = [n for n in range(N) if g.kind[n] == "gateway"]
    co_lap = [g.names[n] for n in range(N) if g.kind[n] in ("gateway", "user")
              and not any(toi_duoc[n, x] for x in gw_all if x != n)]          # không tới được gateway nào khác
    # Nguồn: nút mặt đất ĐANG chịu bão (mức bão > 0), nặng nhất trước, còn tới được ít nhất một gateway KHÁC chính
    # nó và khác các nguồn đã chọn (sửa 18/10/2026: trước đây "tới được chính nó" cũng tính là liên lạc được, và khi
    # vùng bão bị cắt hẳn thì nút không mưa cũng thành nguồn — xem TIEU_CHI.md)
    nguon, bo_qua = [], []
    for n in np.argsort(-diem):
        if len(nguon) == so_nguon or diem[n] <= 0:
            break
        if any(toi_duoc[n, x] for x in gw_all if x != n and x not in nguon):
            nguon.append(int(n))
        else:
            bo_qua.append(g.names[n])
    gw = [x for x in gw_all if x not in nguon and any(toi_duoc[s, x] for s in nguon)]
    nguon = np.array([s for s in nguon if any(toi_duoc[s, x] for x in gw)], int)   # chỉ bốc nguồn có đích
    if len(nguon) == 0 or not gw:
        return None
    K = int(rng.integers(so_luong[0], so_luong[1] + 1))
    p = (diem[nguon] + 0.5) / (diem[nguon] + 0.5).sum()
    cache = {} if cache is None else cache
    src, dst, lam, uv = [], [], [], []
    for _ in range(20 * K):                       # bốc lại tới khi đủ K luồng có đường
        if len(src) == K:
            break
        s = int(rng.choice(nguon, p=p))
        dich = [x for x in gw if x != s and toi_duoc[s, x]]
        if not dich:
            continue
        d = int(rng.choice(dich))
        key = (t, s, d)
        if key not in cache:
            cache[key] = yen_near_optimal(W0, s, d, can_relay, k=SO_UNG_VIEN, eps=1e9)
        if not cache[key]:
            continue
        src.append(s)
        dst.append(d)
        lam.append(float(rng.lognormal(0.0, 0.5)))
        uv.append(cache[key])
    if len(src) < K:
        return None
    Lmax = max(len(r) - 1 for u in uv for r in u)
    canh = -np.ones((len(src), SO_UNG_VIEN, Lmax), int)
    hop_le = np.zeros((len(src), SO_UNG_VIEN), bool)
    for k, u in enumerate(uv):
        for c, r in enumerate(u):
            e = np.array(r[:-1]) * N + np.array(r[1:])
            canh[k, c, :len(e)] = e
            hop_le[k, c] = True
    lam = np.array(lam)
    base = TinhHuong(N, can_relay, song, prop, C, rho0, np.array(src), np.array(dst), lam, uv, canh, hop_le,
                     {"t": int(t), "nguon": [g.names[n] for n in nguon], "co_lap": co_lap, "nguon_bo_qua": bo_qua,
                      "diem_nguon": [float(diem[n]) for n in nguon], "diem_max": float(diem.max()),
                      "lien_ket_het_dung_luong": int((song_vl & (rho0 >= 1.0)).sum())})
    F0 = base.tai_tu_chon(np.zeros(len(src), int))                    # Dijkstra độc lập = ứng viên đầu
    trong = C * (1 - rho0)                                             # dung lượng còn trống (Mbps)
    dung = (F0 > 0) & (trong > 0)
    if not dung.any():
        return None
    u = np.max(F0[dung] / trong[dung])                                 # phần còn trống bị dùng (λ chưa nhân)
    ra = {}
    for ten, muc in MUC_TAI.items():
        th = TinhHuong(**{**base.__dict__, "lam": lam * muc / u,
                          "meta": {**base.meta, "muc_tai": ten, "rho_nen_max_tuyen_chinh": float(rho0[F0 > 0].max())}})
        ra[ten] = th
    return ra
