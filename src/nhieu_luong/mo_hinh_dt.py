"""
mo_hinh_dt.py — DM chọn TẬP tuyến cho mọi luồng cùng lúc (thí nghiệm 4, results/bao_da_tuyen/TIEU_CHI_4.md).

Mỗi ứng viên trong kho (≤ 12 mỗi luồng) có 3 trạng thái: 0 không chọn, 1 dự phòng, 2 tuyến chính. Một lựa chọn hợp lệ
của luồng k = đúng 1 tuyến chính + (m_k − 1) dự phòng. Khung giống MangNL của thí nghiệm 1 (đồ thị hai phía ứng viên –
liên kết); đặc trưng động theo trạng thái hiện tại:
  tải tuyến chính F_p, tải dự kiến nếu các luồng chuyển sang dự phòng F_b → ρ lớn nhất trên tuyến khi làm tuyến chính
  và khi làm dự phòng; phần cạnh trùng với các tuyến khác đã chọn của cùng luồng; trùng loại đường lên (LEO / UAV / HAPS).
Nhiễu kiểu D3PM theo luồng: giữ lựa chọn sạch với xác suất ᾱ(t), còn lại một lựa chọn hợp lệ ngẫu nhiên.
Lấy mẫu: S chuỗi song song; giữ chuỗi có log-xác suất (theo chính mô hình) của lời giải cuối cao nhất — không dùng
thông tin tương lai, không dùng J ước lượng bằng bảng.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

from src.nhieu_luong.da_tuyen import SO_KHO, W_RR
from src.nhieu_luong.mo_hinh import alpha_bar, mlp, tre_t

GOC = Path(__file__).resolve().parents[2]
KQ = GOC / "results" / "bao_da_tuyen" / "mo_hinh"
P = SO_KHO
SO_UV_TINH, SO_UV_DONG, SO_LK_TINH, SO_LK_DONG = 15, 9, 18, 3
DAC_TRUNG_6 = ("giong_lan_can_uav", "giong_1h_uav", "cape_uav")      # thí nghiệm 6: giông quanh UAV
VO_CUC = -1e9


class LoDT:
    """Gộp các bài toán gọn (BaiToanGon) thành mảng đặc có đệm; lap = số bản sao mỗi bài toán (S chuỗi).
    them: thêm 3 đặc trưng cạnh của thí nghiệm 6."""

    def __init__(self, ds, lap=1, them=False):
        n, K, E = len(ds), max(x.K for x in ds), max(len(x.U) for x in ds)
        inc = np.zeros((n, K, P, E), np.float32)
        hl = np.zeros((n, K, P), bool)
        hl[:, :, 0] = True                                                     # luồng đệm: một ứng viên giả
        fm = np.zeros((n, K), bool)
        lam, m = np.ones((n, K), np.float32), np.ones((n, K), np.int64)
        dmin, drange = np.zeros((n, K), np.float32), np.ones((n, K), np.float32)
        lk = np.zeros((4, n, E), np.float32)
        lk[1] = 1.0
        em = np.zeros((n, E), bool)
        ftc = np.zeros((n, E, 11 + len(DAC_TRUNG_6) * them), np.float32)
        fuv = np.zeros((n, K, P, 5), np.float32)
        for b, x in enumerate(ds):
            e_ = len(x.U)
            for k in range(x.K):
                pk = x.so_kho[k]
                inc[b, k, :pk, :e_] = x.inc[k]
                hl[b, k, :pk] = True
                fuv[b, k, :pk] = x.dac_trung_uv[k]
            fm[b, :x.K], lam[b, :x.K], m[b, :x.K] = True, x.lam, x.m
            dmin[b, :x.K], drange[b, :x.K] = x.dmin, x.drange
            lk[0, b, :e_], lk[1, b, :e_], lk[2, b, :e_], lk[3, b, :e_] = x.prop, x.C, x.rho0, x.risk
            em[b, :e_] = True
            d = x.dac_trung_canh
            ftc[b, :e_, :5] = np.eye(5)[np.searchsorted([0, 5, 12, 20, 28], d["band"])]
            ftc[b, :e_, 5] = d["R"] / 20
            ftc[b, :e_, 6] = np.clip(d["dR"], -20, 20) / 10
            ftc[b, :e_, 7] = d["gust"] / 15
            ftc[b, :e_, 8], ftc[b, :e_, 9], ftc[b, :e_, 10] = d["giong_dat"], d["giong_uav"], d["mat_dat"]
            if them:
                for j, k in enumerate(DAC_TRUNG_6):
                    ftc[b, :e_, 11 + j] = d[k]
        rep = lambda a: torch.from_numpy(np.repeat(a, lap, axis=0)) if lap > 1 else torch.from_numpy(a)
        self.inc, self.hop_le, self.fm, self.lam, self.m = rep(inc), rep(hl), rep(fm), rep(lam), rep(m)
        self.dmin, self.drange = rep(dmin), rep(drange)
        self.prop, self.C, self.rho0, self.risk = rep(lk[0]), rep(lk[1]), rep(lk[2]), rep(lk[3])
        self.em, self.ftc, self.fuv = rep(em), rep(ftc), rep(fuv)
        self.B, self.K, self.E = n * lap, K, E
        self._tinh_tinh()

    def _tinh_tinh(self):
        inc, hl, emf = self.inc, self.hop_le.float()[..., None], self.em.float()[..., None]
        lamk = self.lam[..., None]
        self.trong = (self.C * (1 - self.rho0)).clamp(min=1e-3) * self.em + ~self.em
        self.so_hop = inc.sum(-1).clamp(min=1)
        d0 = torch.einsum("bkpe,bke->bkp", inc, tre_t(lamk.expand(-1, -1, self.E), self.prop[:, None], self.C[:, None],
                                                      self.rho0[:, None]))
        T0 = ((d0 - self.dmin[..., None]) / self.drange[..., None]).clamp(0, 10)
        lr = torch.einsum("bkpe,be->bkp", inc, torch.log1p(-self.risk.clamp(max=1 - 1e-6)))
        p = -torch.expm1(lr)
        min_trong = torch.where(inc > 0, self.trong[:, None, None], torch.full_like(inc, 1e9)).amin(-1)
        lam_tb = (self.lam * self.fm).sum(1, keepdim=True) / self.fm.sum(1, keepdim=True)
        mm = self.m[..., None].float().expand_as(d0)
        self.dt_uv_tinh = torch.cat([torch.stack([
            self.so_hop / 10, torch.einsum("bkpe,be->bkp", inc, self.prop) / 20, torch.log1p(d0 / 10), T0 / 2,
            (torch.log(lamk / min_trong.clamp(min=1e-6)).clamp(-12, 4) / 4),
            torch.log(lamk / lam_tb[:, :, None]).clamp(-5, 5).expand_as(d0), p, torch.log(p + 1e-4) / 4 + 1,
            (inc * self.risk[:, None, None]).amax(-1), mm / 3], -1), self.fuv], -1) * hl
        self.dt_lk_tinh = torch.cat([torch.stack([
            torch.log(self.C / 100), self.rho0, torch.log(self.trong / 10) / 3, self.prop / 10,
            torch.log1p(tre_t(torch.zeros_like(self.C), self.prop, self.C, self.rho0) / 10), self.risk.sqrt(),
            torch.log(self.risk + 1e-4) / 4 + 1], -1), self.ftc], -1) * emf

    def dong(self, X):
        """X [B,K,P,3] (xác suất / one-hot trạng thái) → đặc trưng ứng viên, liên kết, trọng số tin nhắn."""
        inc, hl = self.inc, self.hop_le.float()
        P2, P1 = X[..., 2] * hl, X[..., 1] * hl
        Ps = P1 + P2
        lamk = self.lam[..., None]
        Op = torch.einsum("bkp,bkpe->bke", P2 * lamk, inc)
        Ob = torch.einsum("bkp,bkpe->bke", P1 * lamk, inc)
        Fp, Fb = Op.sum(1), Ob.sum(1)
        Fp_k, Fb_k = (Fp[:, None] - Op).clamp(min=0), (Fb[:, None] - Ob).clamp(min=0)
        p, C, r = self.prop[:, None], self.C[:, None], self.rho0[:, None]
        D = torch.einsum("bkpe,bke->bkp", inc, tre_t(Fp_k + lamk, p, C, r))
        T = ((D - self.dmin[..., None]) / self.drange[..., None]).clamp(0, 20)
        rp = (inc * (r + (Fp_k + lamk) / C)[:, :, None]).amax(-1)
        rbe = inc * (r + (Fp_k + Fb_k + lamk) / C)[:, :, None]
        rb, i_nghen = rbe.max(-1)
        Osel = torch.einsum("bkp,bkpe->bke", Ps, inc)
        ov = (inc * (Osel[:, :, None] - Ps[..., None] * inc)).sum(-1) / self.so_hop
        typ = self.fuv[..., :3]
        Tsel = torch.einsum("bkp,bkpj->bkj", Ps, typ)
        cung_loai = (typ * (Tsel[:, :, None] - Ps[..., None] * typ)).sum(-1)
        uv = torch.cat([torch.stack([torch.log1p(D / 10), T / 4, rp.clamp(max=3), rb.clamp(max=3), ov.clamp(0, 3),
                                     cung_loai.clamp(0, 3)], -1), X], -1) * hl[..., None]
        lk = torch.stack([(self.rho0 + Fp / self.C).clamp(max=3), (self.rho0 + (Fp + Fb) / self.C).clamp(max=3),
                          (Fb / self.trong).clamp(max=5)], -1) * self.em[..., None]
        return uv, lk, Ps * lamk, Fp + Fb, i_nghen, Ps


class MangDT(nn.Module):
    def __init__(self, H=64, so_lop=3, t_dim=8, them=False):
        super().__init__()
        self.H, self.t_dim, self.them = H, t_dim, bool(them)
        self.cau_hinh = dict(H=H, so_lop=so_lop, t_dim=t_dim, them=self.them)
        self.vao_uv = mlp(SO_UV_TINH + SO_UV_DONG + t_dim, H, H)
        self.vao_lk = mlp(SO_LK_TINH + len(DAC_TRUNG_6) * self.them + SO_LK_DONG, H, H)
        self.gui = nn.ModuleList([nn.Linear(H, H) for _ in range(so_lop)])
        self.cap_lk = nn.ModuleList([mlp(2 * H, H, H) for _ in range(so_lop)])
        self.cap_uv = nn.ModuleList([mlp(4 * H, H, H) for _ in range(so_lop)])
        self.ra = nn.Sequential(nn.SiLU(), nn.Linear(H, 3))

    def t_emb(self, t):
        f = torch.exp(torch.arange(self.t_dim // 2).float() * -math.log(100.0) / (self.t_dim // 2))
        a = t[:, None] * 10 * f[None]
        return torch.cat([torch.sin(a), torch.cos(a)], 1)

    def forward(self, lo: LoDT, X, t):
        """X [B,K,P,3], t [B] → logit [B,K,P,3]."""
        B, K, H = lo.B, lo.K, self.H
        uv_d, lk_d, XL, Fl, i_nghen, Ps = lo.dong(X)
        te = self.t_emb(t)[:, None, None].expand(B, K, P, self.t_dim)
        hc = self.vao_uv(torch.cat([lo.dt_uv_tinh, uv_d, te], -1))
        he = self.vao_lk(torch.cat([lo.dt_lk_tinh, lk_d], -1))
        inc = lo.inc
        chia = (Fl + 1e-6)[..., None]
        idx = i_nghen.reshape(B, K * P, 1).expand(B, K * P, H)
        tong_chon = Ps.sum(2, keepdim=True).clamp(min=1)[..., None]
        for gui, cl, cu in zip(self.gui, self.cap_lk, self.cap_uv):
            m_e = torch.einsum("bkpe,bkph->beh", inc, XL[..., None] * gui(hc)) / chia
            he = he + cl(torch.cat([he, m_e], -1))
            tb = torch.einsum("bkpe,beh->bkph", inc, he) / lo.so_hop[..., None]
            nghen = torch.gather(he, 1, idx).view(B, K, P, H)
            ngu_canh = ((Ps[..., None] * hc).sum(2, keepdim=True) / tong_chon).expand_as(hc)
            hc = hc + cu(torch.cat([hc, tb, nghen, ngu_canh], -1))
        return self.ra(hc)


# ---------------------------------------------------------------- trạng thái, nhiễu, mất mát
def trang_thai_nhan(ds, K):
    """Nhãn (lựa chọn của thầy) → trạng thái [n, K, P]."""
    y = np.zeros((len(ds), K, P), np.int64)
    for b, (x, a) in enumerate(ds):
        for k in range(x.K):
            o = x.opts[k][a[k]]
            y[b, k, o[1:]] = 1
            y[b, k, o[0]] = 2
    return torch.from_numpy(y)


def ngau_nhien(lo, gen=None):
    """Một lựa chọn hợp lệ ngẫu nhiên đều cho mỗi luồng → trạng thái [B,K,P]."""
    hl = lo.hop_le
    u = torch.rand(hl.shape, generator=gen).masked_fill(~hl, -1.0)
    chinh = u.argmax(-1)
    u2 = torch.rand(hl.shape, generator=gen).masked_fill(~hl, -1.0)
    u2.scatter_(-1, chinh[..., None], -2.0)
    hang = u2.argsort(-1, descending=True).argsort(-1)
    x = ((hang < (lo.m - 1)[..., None]) & hl).long()
    x.scatter_(-1, chinh[..., None], 2)
    return x


def nhieu(lo, y, t, gen=None):
    """Giữ lựa chọn của mỗi luồng với xác suất ᾱ(t), còn lại một lựa chọn ngẫu nhiên."""
    giu = torch.rand(lo.B, lo.K, generator=gen) < alpha_bar(t)[:, None]
    return torch.where(giu[..., None], y, ngau_nhien(lo, gen))


def _ce(logit, y, lo):
    m = (lo.hop_le & lo.fm[..., None]).reshape(-1).float()
    l = Fn.cross_entropy(logit.reshape(-1, 3), y.reshape(-1), reduction="none")
    return (l * m).sum() / m.sum()


def mat_mat_dm(net, lo, y, gen=None):
    t = torch.rand(lo.B, generator=gen)
    return _ce(net(lo, Fn.one_hot(nhieu(lo, y, t, gen), 3).float(), t), y, lo)


def X_deu(lo):
    """Trạng thái 'chưa biết gì': mỗi ứng viên hợp lệ là tuyến chính với xác suất 1/n, dự phòng (m − 1)/n."""
    hl = lo.hop_le.float()
    n = hl.sum(-1, keepdim=True)
    p2 = hl / n
    p1 = hl * ((lo.m[..., None] - 1) / n).clamp(max=1)
    return torch.stack([hl - p1 - p2, p1, p2], -1).clamp(min=0)


def mat_mat_softmax(net, lo, y, gen=None):
    return _ce(net(lo, X_deu(lo), torch.ones(lo.B)), y, lo)


# ---------------------------------------------------------------- giải mã, lấy mẫu
def giai_ma(lo, logit, lay_mau=False, gen=None):
    """logit → trạng thái hợp lệ [B,K,P] (1 tuyến chính + m − 1 dự phòng mỗi luồng) và log-xác suất [B]."""
    hl = lo.hop_le
    lp = torch.log_softmax(logit, -1)
    a2 = lp[..., 2].masked_fill(~hl, VO_CUC)
    a1 = (lp[..., 1] - lp[..., 0]).masked_fill(~hl, VO_CUC)
    if lay_mau:
        g2 = -torch.log(-torch.log(torch.rand(a2.shape, generator=gen).clamp(1e-12, 1 - 1e-12)))
        g1 = -torch.log(-torch.log(torch.rand(a1.shape, generator=gen).clamp(1e-12, 1 - 1e-12)))
        a2, a1 = a2 + g2, a1 + g1
    chinh = a2.argmax(-1)
    a1 = a1.scatter(-1, chinh[..., None], 2 * VO_CUC)
    hang = a1.argsort(-1, descending=True).argsort(-1)
    x = ((hang < (lo.m - 1)[..., None]) & hl).long()
    x.scatter_(-1, chinh[..., None], 2)
    lpx = torch.gather(lp, -1, x[..., None]).squeeze(-1)
    return x, ((lpx * (hl & lo.fm[..., None])).sum((1, 2)))


def lua_chon_tu_trang_thai(gon, x):
    """Trạng thái [K,P] của một bài toán → chỉ số lựa chọn mỗi luồng."""
    if not hasattr(gon, "_bang_tra"):
        gon._bang_tra = [{(int(o[0]), frozenset(int(i) for i in o[1:])): j for j, o in enumerate(O)} for O in gon.opts]
    ra = np.zeros(gon.K, np.int64)
    for k in range(gon.K):
        chinh = int(np.flatnonzero(x[k] == 2)[0])
        ra[k] = gon._bang_tra[k][(chinh, frozenset(int(i) for i in np.flatnonzero(x[k] == 1)))]
    return ra


@torch.no_grad()
def chon_dm(net, gon, S=8, so_buoc=5, gen=None):
    lo = LoDT([gon], lap=S, them=net.them)
    x = ngau_nhien(lo, gen)
    for i in range(so_buoc):
        logit = net(lo, Fn.one_hot(x, 3).float(), torch.full((S,), 1 - i / so_buoc))
        cuoi = i == so_buoc - 1
        x0, lp = giai_ma(lo, logit, lay_mau=not cuoi, gen=gen)
        if not cuoi:
            giu = torch.rand(S, lo.K, generator=gen) < alpha_bar(torch.tensor(1 - (i + 1) / so_buoc))
            x = torch.where(giu[..., None], x0, ngau_nhien(lo, gen))
    j = int(lp.argmax())
    return lua_chon_tu_trang_thai(gon, x0[j, :gon.K].numpy())


@torch.no_grad()
def chon_softmax(net, gon):
    lo = LoDT([gon], them=net.them)
    x, _ = giai_ma(lo, net(lo, X_deu(lo), torch.ones(1)))
    return lua_chon_tu_trang_thai(gon, x[0, :gon.K].numpy())


def nap(kieu, thu_muc=None):
    ck = torch.load((thu_muc or KQ) / f"{kieu}.pt", weights_only=False)
    net = MangDT(**ck["cau_hinh"])
    net.load_state_dict(ck["state"])
    return net.eval()


def ham_chon(thu_muc=None, ten_dm="dm", softmax=True):
    """{tên: hàm(gon) → lựa chọn} cho bộ chấm; số bước DM đọc từ <thu_muc>/chon_buoc.json (chọn trên val).
    Mặc định: mô hình của thí nghiệm 4 (results/bao_da_tuyen/mo_hinh)."""
    torch.set_num_threads(1)
    thu_muc = thu_muc or KQ
    tep = thu_muc / f"chon_buoc_{ten_dm}.json"
    cb = json.loads((tep if tep.exists() else thu_muc / "chon_buoc.json").read_text(encoding="utf-8"))
    dm = nap(ten_dm, thu_muc)
    gen = torch.Generator().manual_seed(2026)
    ra = {"dm": lambda gon: chon_dm(dm, gon, cb["S"], cb["so_buoc"], gen)}
    if softmax:
        sm = nap("softmax", thu_muc)
        ra["softmax"] = lambda gon: chon_softmax(sm, gon)
    return ra
