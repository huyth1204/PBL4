"""
mo_hinh_rr.py — DM / softmax cho thí nghiệm 3 (điểm 70% rủi ro + 30% tốc độ, results/bao_rr/TIEU_CHI_3.md).

Giữ nguyên kiến trúc MangNL của thí nghiệm 1 (đồ thị hai phía ứng viên – liên kết, khử nhiễu rời rạc kiểu D3PM đều),
chỉ thêm đặc trưng để mạng thấy được hàm mục tiêu mới:
  ứng viên, tĩnh   H (rủi ro đã chuẩn hoá), log p (rủi ro tuyến), T⁰ (độ chậm khi đi một mình), điểm khi đi một mình
  ứng viên, động   T theo tải hiện tại của các luồng khác, điểm 0,7·H + 0,3·T, điểm hơn ứng viên tốt nhất bao nhiêu
  liên kết, tĩnh   √risk, log risk
Lấy mẫu: S chuỗi song song, chấm J thật (bai_toan_rr.diem) từng mẫu, giữ mẫu J nhỏ nhất.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as Fn

from src.nhieu_luong.bai_toan_rr import W_RR, diem
from src.nhieu_luong.mo_hinh import C8, Lo, MangNL, X_deu, alpha_bar, mlp, ngau_nhien_hop_le, tre_t

SO_UV_TINH, SO_UV_DONG, SO_LK_TINH, SO_LK_DONG = 7 + 4, 6 + 3, 6 + 2, 3


class LoRR(Lo):
    def __init__(self, ds, lap=1):
        K = max(len(x["lam"]) for x in ds)
        E = max(len(x["e"]) for x in ds)
        n = len(ds)
        rk = np.zeros((n, E), np.float32)
        H, p, d0 = (np.zeros((n, K, C8), np.float32) for _ in range(3))
        dmin, dr = np.zeros((n, K), np.float32), np.ones((n, K), np.float32)
        for b, x in enumerate(ds):
            k_, e_ = len(x["lam"]), len(x["e"])
            rk[b, :e_] = x["risk"]
            H[b, :k_], p[b, :k_], d0[b, :k_] = x["H"], x["p_uv"], x["d0"]
            dmin[b, :k_], dr[b, :k_] = x["dmin"], x["drange"]
        rep = lambda a: torch.from_numpy(np.repeat(a, lap, axis=0)) if lap > 1 else torch.from_numpy(a)
        self.risk, self.Huv, self.puv, self.d0uv = rep(rk), rep(H), rep(p), rep(d0)
        self.dmin, self.drange = rep(dmin), rep(dr)
        super().__init__(ds, lap)

    def _tinh_tinh(self):
        super()._tinh_tinh()
        hl = self.hop_le.float()[..., None]
        T0 = ((self.d0uv - self.dmin[..., None]) / self.drange[..., None]).clamp(0, 10)
        self.dt_uv_tinh = torch.cat([self.dt_uv_tinh, torch.stack([
            self.Huv, torch.log(self.puv + 1e-4) / 4 + 1, T0 / 2, (W_RR * self.Huv + (1 - W_RR) * T0) / 2], -1) * hl], -1)
        emf = self.em.float()[..., None]
        self.dt_lk_tinh = torch.cat([self.dt_lk_tinh, torch.stack([
            self.risk.sqrt(), torch.log(self.risk + 1e-4) / 4 + 1], -1) * emf], -1)

    def dong(self, X):
        uv, lk, XL, Fl, i_nghen = super().dong(X)
        inc = self.inc
        O = torch.einsum("bkc,bkce->bke", XL, inc)
        FoL = (Fl[:, None] - O).clamp(min=0) + self.lam[..., None]          # tải luồng khác + chính luồng k
        D = torch.einsum("bkce,bke->bkc", inc, tre_t(FoL, self.prop[:, None], self.C[:, None], self.rho0[:, None]))
        T = ((D - self.dmin[..., None]) / self.drange[..., None]).clamp(0, 20)
        d = W_RR * self.Huv + (1 - W_RR) * T
        dv = torch.where(self.hop_le, d, torch.full_like(d, 1e9))
        hon = (d - dv.min(2, keepdim=True).values).clamp(0, 20)
        them = torch.stack([T / 4, d / 4, hon / 4], -1) * self.hop_le[..., None]
        return torch.cat([uv, them], -1), lk, XL, Fl, i_nghen


class MangRR(MangNL):
    def __init__(self, H=64, so_lop=3, t_dim=8):
        super().__init__(H, so_lop, t_dim)
        self.vao_uv = mlp(SO_UV_TINH + SO_UV_DONG + t_dim, H, H)
        self.vao_lk = mlp(SO_LK_TINH + SO_LK_DONG, H, H)


@torch.no_grad()
def lay_mau_dm(net, x, S=16, so_buoc=5, gen=None):
    """S chuỗi khử nhiễu song song cho một tình huống. Trả (lời giải tốt nhất [K], J, mọi J [S])."""
    lo = LoRR([x], lap=S)
    xt = ngau_nhien_hop_le(lo.hop_le, gen)
    for i in range(so_buoc):
        logit = net(lo, Fn.one_hot(xt, C8).float(), torch.full((S,), 1 - i / so_buoc))
        if i == so_buoc - 1:
            x0 = logit.argmax(-1)
        else:
            x0 = torch.multinomial(torch.softmax(logit, -1).view(-1, C8), 1, generator=gen).view(S, -1)
            giu = torch.rand(S, lo.K, generator=gen) < alpha_bar(torch.tensor(1 - (i + 1) / so_buoc))
            xt = torch.where(giu, x0, ngau_nhien_hop_le(lo.hop_le, gen))
    A = x0.numpy().astype(np.int64)
    J = diem(x, A)
    j = int(np.argmin(J))
    return A[j], float(J[j]), J


@torch.no_grad()
def lay_mau_softmax(net, x, S=16, gen=None):
    """Một lần suy luận; mẫu 1 = argmax, S−1 mẫu còn lại bốc độc lập theo từng luồng."""
    lo = LoRR([x])
    p = torch.softmax(net(lo, X_deu(lo), torch.ones(1)), -1)[0]
    A = torch.cat([p.argmax(-1)[None], torch.multinomial(p, S - 1, replacement=True, generator=gen).T], 0)
    A = A.numpy().astype(np.int64)
    J = diem(x, A)
    j = int(np.argmin(J))
    return A[j], float(J[j]), J
