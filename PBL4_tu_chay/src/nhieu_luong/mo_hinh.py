"""
mo_hinh.py — Mô hình học cho bài toán nhiều luồng (xem results/yagi/TIEU_CHI.md).

Đồ thị hai phía "ứng viên – liên kết": mỗi luồng có 8 tuyến ứng viên; mỗi ứng viên nối với các liên kết nó đi qua.
Các luồng chỉ ảnh hưởng nhau qua liên kết dùng chung, nên mạng truyền thông tin ứng viên → liên kết → ứng viên.
Cài đặt dạng mảng đặc [B, K, 8, E] (B tình huống, K luồng, E liên kết dùng chung) cho nhanh trên CPU.

Hai cách dùng CÙNG một mạng (cùng kiến trúc, cùng đặc trưng):
  DM       khử nhiễu ĐỒNG THỜI lựa chọn của mọi luồng. Ở mỗi bước, trạng thái hiện tại x_t (một lựa chọn cho mỗi
           luồng, còn nhiễu) cho biết tải dự kiến trên từng liên kết → mạng thấy luồng nào đang dồn vào đâu và dự
           đoán lại lựa chọn sạch x0 cho mọi luồng. Nhiễu kiểu D3PM đều: giữ x0 với xác suất ᾱ(t), còn lại chọn
           ngẫu nhiên một ứng viên hợp lệ.
  Softmax  đoán một lần: trạng thái = phân bố đều trên các ứng viên (tải dự kiến khi chưa biết gì), t = 1.
Cả hai lấy S mẫu, chấm chi phí thật từng mẫu, giữ mẫu tốt nhất.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

from src.ngau_nhien.kich_ban import PKT, PROC
from src.nhieu_luong.bai_toan import RHO_GAY, h_hang_doi

C8 = 8
AM = -1e9


def h_t(rho):
    r = torch.clamp(rho, max=RHO_GAY)
    return torch.where(rho <= RHO_GAY, r / (1 - r), RHO_GAY / (1 - RHO_GAY) + (rho - RHO_GAY) / (1 - RHO_GAY) ** 2)


def tre_t(F, p, C, r):
    return p + PKT / C * (1 + h_t(r + F / C)) + PROC


class Lo:
    """Gộp tình huống (dạng gọn) thành mảng đặc có đệm; lap = số bản sao mỗi tình huống (S mẫu song song)."""

    def __init__(self, ds, lap=1):
        B = len(ds) * lap
        K = max(len(x["lam"]) for x in ds)
        E = max(len(x["e"]) for x in ds)
        inc = np.zeros((len(ds), K, C8, E), np.float32)
        hl = np.zeros((len(ds), K, C8), bool)
        hl[:, :, 0] = True                                                  # luồng đệm: một ứng viên giả
        fm = np.zeros((len(ds), K), bool)
        lam = np.ones((len(ds), K), np.float32)
        lk = np.zeros((3, len(ds), E), np.float32)
        lk[1] = 1.0                                                         # C đệm = 1
        em = np.zeros((len(ds), E), bool)
        for b, x in enumerate(ds):
            canh = x["canh"].astype(np.int64)
            k_, e_ = canh.shape[0], len(x["e"])
            m = (canh >= 0) & x["hop_le"][:, :, None]
            k_i, c_i, l_i = np.nonzero(m)
            inc[b, k_i, c_i, canh[k_i, c_i, l_i]] = 1.0
            hl[b, :k_], fm[b, :k_], lam[b, :k_] = x["hop_le"], True, x["lam"]
            lk[0, b, :e_], lk[1, b, :e_], lk[2, b, :e_] = x["prop"], x["C"], x["rho0"]
            em[b, :e_] = True
        rep = lambda a: torch.from_numpy(np.repeat(a, lap, axis=0)) if lap > 1 else torch.from_numpy(a)
        self.inc, self.hop_le, self.fm, self.lam = rep(inc), rep(hl), rep(fm), rep(lam)
        self.prop, self.C, self.rho0 = rep(lk[0]), rep(lk[1]), rep(lk[2])
        self.em = rep(em)
        self.B, self.K, self.E = B, K, E
        self._tinh_tinh()

    def _tinh_tinh(self):
        inc, hl = self.inc, self.hop_le.float()
        emf = self.em.float()
        self.trong = (self.C * (1 - self.rho0)).clamp(min=1e-3) * emf + (1 - emf)
        d0 = tre_t(torch.zeros_like(self.C), self.prop, self.C, self.rho0) * emf
        self.so_hop = inc.sum(-1).clamp(min=1)                                        # [B,K,8]
        sum_prop = torch.einsum("bkce,be->bkc", inc, self.prop)
        sum_d0 = torch.einsum("bkce,be->bkc", inc, d0)
        d0_min = torch.where(self.hop_le, sum_d0, torch.full_like(sum_d0, 1e9)).min(2, keepdim=True).values
        d0_rel = (sum_d0 / d0_min.clamp(min=1e-6) - 1).clamp(0, 5)
        min_trong = torch.where(inc > 0, self.trong[:, None, None, :], torch.full_like(inc, 1e9)).amin(-1)
        fmf = self.fm.float()
        lam_tb = (self.lam * fmf).sum(1, keepdim=True) / fmf.sum(1, keepdim=True)
        hang = torch.arange(C8).float()[None, None].expand_as(sum_d0) / C8
        lamk = self.lam[:, :, None].expand_as(sum_d0)
        self.dt_uv_tinh = torch.stack([
            self.so_hop / 10, sum_prop / 20, torch.log1p(sum_d0 / 10), torch.log1p(d0_rel),
            (torch.log(lamk / min_trong.clamp(min=1e-6)).clamp(-12, 4) / 4),
            torch.log(lamk / lam_tb[:, :, None]).clamp(-5, 5), hang], -1) * hl[..., None]
        so_dung = inc.sum((1, 2)) / (C8 * fmf.sum(1, keepdim=True))
        self.dt_lk_tinh = torch.stack([torch.log(self.C / 100), self.rho0, torch.log(self.trong / 10) / 3,
                                       self.prop / 10, torch.log1p(d0 / 10), so_dung], -1) * emf[..., None]

    def dong(self, X):
        """Đặc trưng theo trạng thái X [B,K,8] (xác suất hoặc one-hot)."""
        inc = self.inc
        XL = X * self.lam[..., None] * self.fm[..., None]
        O = torch.einsum("bkc,bkce->bke", XL, inc)                     # tải của chính luồng k trên từng liên kết
        Fl = O.sum(1)                                                  # tải mỗi liên kết [B,E]
        Fo = (Fl[:, None] - O).clamp(min=0)                            # tải của các luồng KHÁC
        FoL = Fo + self.lam[..., None]
        p, C, r = self.prop[:, None], self.C[:, None], self.rho0[:, None]
        dw = tre_t(FoL, p, C, r)
        mg = FoL * dw - Fo * tre_t(Fo, p, C, r)
        rw = r + FoL / C
        D = torch.einsum("bkce,bke->bkc", inc, dw)
        MG = torch.einsum("bkce,bke->bkc", inc, mg) / self.lam[..., None]
        RMe = inc * rw[:, :, None]
        RM, i_nghen = RMe.max(-1)                                      # ρ lớn nhất trên tuyến và liên kết nghẽn nhất
        MGv = torch.where(self.hop_le, MG, torch.full_like(MG, 1e12))
        Dv = torch.where(self.hop_le, D, torch.full_like(D, 1e12))
        mg_rel = (MG / MGv.min(2, keepdim=True).values.clamp(min=1e-6) - 1).clamp(0, 1e4)
        d_rel = (D / Dv.min(2, keepdim=True).values.clamp(min=1e-6) - 1).clamp(0, 1e4)
        uv = torch.stack([torch.log1p(D / 10), torch.log1p(MG / 10), RM.clamp(max=3), torch.log1p(mg_rel),
                          torch.log1p(d_rel), X], -1) * self.hop_le[..., None]
        rho_l = self.rho0 + Fl / self.C
        lk = torch.stack([rho_l.clamp(max=3), torch.log1p(h_t(rho_l)) / 5, (Fl / self.trong).clamp(max=5)], -1) \
            * self.em[..., None]
        return uv, lk, XL, Fl, i_nghen

    @staticmethod
    def chi_phi_gon(x, A):
        """x: tình huống dạng gọn; A [S, K] lựa chọn. Trả [S] chi phí (ms, trễ trung bình theo lưu lượng)."""
        A = np.atleast_2d(A)
        S, K = A.shape
        lk = x["canh"].astype(np.int64)[np.arange(K)[None], A]             # [S, K, L]
        m = lk >= 0
        F = np.zeros((S, len(x["e"])))
        s_i = np.broadcast_to(np.arange(S)[:, None, None], lk.shape)[m]
        np.add.at(F, (s_i, lk[m]), np.broadcast_to(x["lam"][None, :, None], lk.shape)[m])
        p, C, r = x["prop"][None], x["C"][None], x["rho0"][None]
        f = F * (p + PKT / C * (1 + h_hang_doi(r + F / C)) + PROC)
        return f.sum(1) / x["lam"].sum()


def mlp(a, H, b):
    return nn.Sequential(nn.Linear(a, H), nn.LayerNorm(H), nn.SiLU(), nn.Linear(H, b))


class MangNL(nn.Module):
    def __init__(self, H=64, so_lop=3, t_dim=8):
        super().__init__()
        self.H, self.t_dim, self.cau_hinh = H, t_dim, dict(H=H, so_lop=so_lop, t_dim=t_dim)
        self.vao_uv = mlp(7 + 6 + t_dim, H, H)
        self.vao_lk = mlp(6 + 3, H, H)
        self.gui = nn.ModuleList([nn.Linear(H, H) for _ in range(so_lop)])
        self.cap_lk = nn.ModuleList([mlp(2 * H, H, H) for _ in range(so_lop)])
        self.cap_uv = nn.ModuleList([mlp(4 * H, H, H) for _ in range(so_lop)])
        self.ra = nn.Sequential(nn.SiLU(), nn.Linear(H, 1))

    def t_emb(self, t):
        f = torch.exp(torch.arange(self.t_dim // 2).float() * -math.log(100.0) / (self.t_dim // 2))
        a = t[:, None] * 10 * f[None]
        return torch.cat([torch.sin(a), torch.cos(a)], 1)

    def forward(self, lo: Lo, X, t):
        """X [B,K,8], t [B] (mức nhiễu) → logit [B,K,8]."""
        B, K, E, H = lo.B, lo.K, lo.E, self.H
        uv_d, lk_d, XL, Fl, i_nghen = lo.dong(X)
        te = self.t_emb(t)[:, None, None].expand(B, K, C8, self.t_dim)
        hc = self.vao_uv(torch.cat([lo.dt_uv_tinh, uv_d, te], -1))                  # [B,K,8,H]
        he = self.vao_lk(torch.cat([lo.dt_lk_tinh, lk_d], -1))                      # [B,E,H]
        inc = lo.inc
        chia = (Fl + 1e-6)[..., None]
        idx = i_nghen.reshape(B, K * C8, 1).expand(B, K * C8, H)
        for gui, cl, cu in zip(self.gui, self.cap_lk, self.cap_uv):
            m_e = torch.einsum("bkce,bkch->beh", inc, XL[..., None] * gui(hc)) / chia   # trung bình theo tải
            he = he + cl(torch.cat([he, m_e], -1))
            tb = torch.einsum("bkce,beh->bkch", inc, he) / lo.so_hop[..., None]
            nghen = torch.gather(he, 1, idx).view(B, K, C8, H)
            ngu_canh = (X[..., None] * hc).sum(2, keepdim=True).expand_as(hc)
            hc = hc + cu(torch.cat([hc, tb, nghen, ngu_canh], -1))
        return self.ra(hc).squeeze(-1).masked_fill(~lo.hop_le, AM)


# ---------------------------------------------------------------- nhiễu, huấn luyện, lấy mẫu
def alpha_bar(t):
    return torch.cos(t * math.pi / 2) ** 2


def ngau_nhien_hop_le(hop_le, gen=None):
    u = torch.rand(hop_le.shape, generator=gen).masked_fill(~hop_le, -1.0)
    return u.argmax(-1)


def nhan_lo(ds, K):
    y = np.zeros((len(ds), K), np.int64)
    for b, x in enumerate(ds):
        y[b, :len(x["nhan"])] = x["nhan"]
    return torch.from_numpy(y)


def _ce(logit, y, fm):
    l = Fn.cross_entropy(logit.view(-1, C8), y.view(-1), reduction="none")
    return (l * fm.view(-1)).sum() / fm.sum()


def mat_mat_dm(net, lo, y, gen=None):
    t = torch.rand(lo.B, generator=gen)
    giu = torch.rand(lo.B, lo.K, generator=gen) < alpha_bar(t)[:, None]
    xt = torch.where(giu, y, ngau_nhien_hop_le(lo.hop_le, gen))
    return _ce(net(lo, Fn.one_hot(xt, C8).float(), t), y, lo.fm)


def X_deu(lo):
    hl = lo.hop_le.float()
    return hl / hl.sum(-1, keepdim=True)


def mat_mat_softmax(net, lo, y, gen=None):
    return _ce(net(lo, X_deu(lo), torch.ones(lo.B)), y, lo.fm)


@torch.no_grad()
def lay_mau_dm(net, x, S=16, so_buoc=10, gen=None):
    """S chuỗi khử nhiễu song song cho một tình huống. Trả (lời giải tốt nhất [K], chi phí, mọi chi phí [S])."""
    lo = Lo([x], lap=S)
    xt = ngau_nhien_hop_le(lo.hop_le, gen)
    for i in range(so_buoc):
        logit = net(lo, Fn.one_hot(xt, C8).float(), torch.full((S,), 1 - i / so_buoc))
        if i == so_buoc - 1:
            x0 = logit.argmax(-1)
        else:
            x0 = torch.multinomial(torch.softmax(logit, -1).view(-1, C8), 1, generator=gen).view(S, -1)
            giu = torch.rand(S, lo.K, generator=gen) < alpha_bar(torch.tensor(1 - (i + 1) / so_buoc))
            xt = torch.where(giu, x0, ngau_nhien_hop_le(lo.hop_le, gen))
    A = x0.numpy()
    c = Lo.chi_phi_gon(x, A)
    j = int(np.argmin(c))
    return A[j], float(c[j]), c


@torch.no_grad()
def lay_mau_softmax(net, x, S=16, gen=None):
    """Một lần suy luận; mẫu 1 = argmax, S−1 mẫu còn lại bốc độc lập theo từng luồng."""
    lo = Lo([x])
    p = torch.softmax(net(lo, X_deu(lo), torch.ones(1)), -1)[0]                    # [K,8]
    A = torch.cat([p.argmax(-1)[None], torch.multinomial(p, S - 1, replacement=True, generator=gen).T], 0).numpy()
    c = Lo.chi_phi_gon(x, A)
    j = int(np.argmin(c))
    return A[j], float(c[j]), c
