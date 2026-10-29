"""
mo_hinh_kgh.py — DM tự sinh tuyến theo cặp nguồn – đích (thí nghiệm 2, results/yagi2/TIEU_CHI_2.md).

Đầu ra của mạng là "bản đồ cạnh" ŷ[c, e] ∈ [0, 1]: phần nhu cầu của cặp nguồn – đích c nên đi qua cạnh e, chỉ tính
trên hành lang của cặp đó. Mạng truyền tin trên từng cặp (cạnh ↔ nút: cạnh nào nối tiếp cạnh nào) và giữa các cặp
(cùng một cạnh: các cặp đang dồn tải vào đâu). Kích thước mạng phụ thuộc số cặp (5–12), KHÔNG phụ thuộc số luồng.
Tuyến được dựng bằng Dijkstra tuần tự có hướng dẫn (khong_gioi_han.dung_tuyen).

  DM       khử nhiễu Gauss (dự đoán x0, tự điều kiện bằng dự đoán của bước trước), DDIM n bước, S mẫu.
  Một-lần  cùng mạng, đoán bản đồ cạnh một lần từ trạng thái "chưa biết gì"; S lần dựng tuyến với S thứ tự luồng.
"""

from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

from src.ngau_nhien.kich_ban import PKT, PROC
from src.nhieu_luong import khong_gioi_han as kg
from src.nhieu_luong.bai_toan import RHO_GAY

N_DAC_TRUNG_TINH = 12
N_DAC_TRUNG_DONG = 5


def h_t(rho):
    r = torch.clamp(rho, max=RHO_GAY)
    return torch.where(rho <= RHO_GAY, r / (1 - r), RHO_GAY / (1 - RHO_GAY) + (rho - RHO_GAY) / (1 - RHO_GAY) ** 2)


# ---------------------------------------------------------------- một tình huống
class TinhHuong2:
    """Một snapshot + một mức tải ở dạng sẵn cho mạng. hl = (mask, rc) có thể dùng chung giữa các mức tải."""

    def __init__(self, x, muc, hl=None, cap_cua=None):
        mg = dict(x["mang"])
        mg["lam"] = x["lam"][muc]
        self.md = kg.MangDay.nap(mg)
        self.x, self.muc = x, muc
        self.cap, self.cua = cap_cua if cap_cua is not None else kg.cap_nguon_dich(self.md)
        self.mask, self.rc = hl if hl is not None else kg.hanh_lang(self.md, self.cap)
        md = self.md
        C = len(self.cap)
        self.Lam = np.bincount(self.cua, weights=md.lam, minlength=C)
        self.n_c = np.bincount(self.cua, minlength=C)
        self.row_c, self.row_e = np.nonzero(self.mask)
        self.nhan_tuyen = x["nhan"].get(muc) if "nhan" in x else None
        self.chi_phi_nhan = x["chi_phi_nhan"].get(muc) if "chi_phi_nhan" in x else None

    def y_nhan(self):
        if getattr(self, "_y", None) is not None:
            return self._y
        tuyen = [np.asarray(e, np.int64) for e in self.nhan_tuyen]
        y = kg.ban_do_nhan(self.md, tuyen, self.cap, self.cua)
        self._y = y[self.row_c, self.row_e].astype(np.float32)
        return self._y

    def gon(self):
        """Chỉ giữ thứ cần cho huấn luyện (tiết kiệm bộ nhớ): đặc trưng tĩnh, nhãn, cấu trúc hàng."""
        self.tinh()
        self.y_nhan()
        md = self.md
        md.eid, md._csr = None, {}
        self.mask = self.rc = self.x = self.nhan_tuyen = None
        return self

    def tinh(self):
        """Đặc trưng tĩnh của từng hàng (cặp, cạnh) [R, 12]."""
        if getattr(self, "_tinh", None) is not None:
            return self._tinh
        md, c, e = self.md, self.row_c, self.row_e
        d0 = md.tre(np.zeros(md.E))
        trong = np.maximum(md.C * (1 - md.rho0), 1e-3)
        s = np.array([p[0] for p in self.cap])[c]
        d = np.array([p[1] for p in self.cap])[c]
        lam_tb = self.Lam / self.n_c
        tinh = np.stack([
            md.prop[e] / 10, np.log(md.C[e] / 100), md.rho0[e], np.log(trong[e] / 10) / 3, np.log1p(d0[e] / 10),
            np.log1p(self.rc[c, e] / 5), (md.eu[e] == s).astype(float), (md.ev[e] == d).astype(float),
            np.log(self.Lam[c] / md.lam.sum()) / 3, self.n_c[c] / md.K,
            np.clip(np.log(lam_tb[c] / trong[e]), -12, 4) / 4,
            np.clip(np.log(self.Lam[c] / trong[e]), -12, 4) / 4], 1).astype(np.float32)
        self._tinh = tinh
        return tinh


# ---------------------------------------------------------------- gộp thành một đồ thị
class Lo2:
    def __init__(self, ths, lap=1, voi_nhan=False):
        f32 = lambda a: torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))
        cols = {k: [] for k in ("tinh", "Lam", "C", "rho0", "trong", "eg", "tail", "head", "item", "y")}
        off_c = off_e = 0
        self.so_hang, self.ths = [], []
        b = 0
        for th in ths:
            md = th.md
            tinh = th.tinh()
            y = th.y_nhan().astype(np.float32) if voi_nhan else None
            trong = np.maximum(md.C * (1 - md.rho0), 1e-3)
            Cn = len(th.cap)
            for _ in range(lap):
                c, e = th.row_c, th.row_e
                cols["tinh"].append(tinh)
                cols["Lam"].append(th.Lam[c])
                cols["C"].append(md.C[e])
                cols["rho0"].append(md.rho0[e])
                cols["trong"].append(trong[e])
                cols["eg"].append(off_e + e)
                cols["tail"].append((off_c + c) * md.N + md.eu[e])
                cols["head"].append((off_c + c) * md.N + md.ev[e])
                cols["item"].append(np.full(len(c), b))
                if voi_nhan:
                    cols["y"].append(y)
                self.so_hang.append(len(c))
                self.ths.append(th)
                off_c += Cn
                off_e += md.E
                b += 1
        cat = np.concatenate
        self.tinh = f32(cat(cols["tinh"]))
        self.Lam, self.C, self.rho0, self.trong = (f32(cat(cols[k])) for k in ("Lam", "C", "rho0", "trong"))
        self.eg, self.tail, self.head, self.item = (torch.from_numpy(cat(cols[k]).astype(np.int64))
                                                    for k in ("eg", "tail", "head", "item"))
        self.y = f32(cat(cols["y"])) if voi_nhan else None
        self.R, self.nE, self.nB = len(self.eg), off_e, b
        self.nNut = off_c * ths[0].md.N if ths else 0
        dem = lambda idx, n: torch.zeros(n).index_add_(0, idx, torch.ones(len(idx))).clamp(min=1)
        self.dem_head = dem(self.head, self.nNut)
        self.dem_tail = dem(self.tail, self.nNut)
        self.dem_e = dem(self.eg, self.nE)

    def dong(self, y_truoc):
        """Đặc trưng theo bản đồ cạnh hiện tại (dự đoán của bước trước) [R, 5]."""
        F = torch.zeros(self.nE).index_add_(0, self.eg, self.Lam * y_truoc)[self.eg]
        rho = self.rho0 + F / self.C
        khac = (F - self.Lam * y_truoc).clamp(min=0)
        return torch.stack([rho.clamp(max=3), torch.log1p(h_t(rho)) / 5, (F / self.trong).clamp(max=5),
                            (khac / self.trong).clamp(max=5), y_truoc], 1)

    def tach(self, v):
        """Tách vector theo hàng [R] thành danh sách theo từng bản sao."""
        return torch.split(v, self.so_hang)


# ---------------------------------------------------------------- mạng
def mlp(a, H, b):
    return nn.Sequential(nn.Linear(a, H), nn.LayerNorm(H), nn.SiLU(), nn.Linear(H, b))


class MangKGH(nn.Module):
    def __init__(self, H=32, so_lop=3, t_dim=8):
        super().__init__()
        self.H, self.t_dim, self.cau_hinh = H, t_dim, dict(H=H, so_lop=so_lop, t_dim=t_dim)
        self.vao = mlp(N_DAC_TRUNG_TINH + N_DAC_TRUNG_DONG + 1 + t_dim, H, H)
        self.lop = nn.ModuleList([mlp(4 * H, H, H) for _ in range(so_lop)])
        self.ra = nn.Sequential(nn.SiLU(), nn.Linear(H, 1))

    def t_emb(self, t):
        f = torch.exp(torch.arange(self.t_dim // 2).float() * -math.log(100.0) / (self.t_dim // 2))
        a = t[:, None] * 10 * f[None]
        return torch.cat([torch.sin(a), torch.cos(a)], 1)

    def forward(self, lo: Lo2, xt, t, y_truoc):
        """xt [R], t [nB], y_truoc [R] → logit [R]."""
        h = self.vao(torch.cat([lo.tinh, lo.dong(y_truoc), xt[:, None], self.t_emb(t)[lo.item]], 1))
        H = self.H
        for lop in self.lop:
            n_in = torch.zeros(lo.nNut, H).index_add_(0, lo.head, h) / lo.dem_head[:, None]
            n_out = torch.zeros(lo.nNut, H).index_add_(0, lo.tail, h) / lo.dem_tail[:, None]
            e_tb = torch.zeros(lo.nE, H).index_add_(0, lo.eg, h) / lo.dem_e[:, None]
            h = h + lop(torch.cat([h, n_in[lo.tail], n_out[lo.head], e_tb[lo.eg]], 1))
        return self.ra(h).squeeze(1)


# ---------------------------------------------------------------- khuếch tán
def alpha_bar(t):
    return (torch.cos(t * math.pi / 2) ** 2).clamp(1e-4, 0.9999)


def mat_mat_dm(net, lo, gen=None):
    t = torch.rand(lo.nB, generator=gen)
    ab = alpha_bar(t)[lo.item]
    x0 = 2 * lo.y - 1
    xt = ab.sqrt() * x0 + (1 - ab).sqrt() * torch.randn(lo.R, generator=gen)
    y_truoc = torch.zeros(lo.R)
    if torch.rand(1, generator=gen).item() < 0.5:                       # tự điều kiện (Chen et al. 2022)
        with torch.no_grad():
            y_truoc = torch.sigmoid(net(lo, xt, t, y_truoc))
    return Fn.binary_cross_entropy_with_logits(net(lo, xt, t, y_truoc), lo.y)


def mat_mat_mot_lan(net, lo, gen=None):
    return Fn.binary_cross_entropy_with_logits(net(lo, torch.zeros(lo.R), torch.ones(lo.nB), torch.zeros(lo.R)), lo.y)


@torch.no_grad()
def ban_do_dm(net, lo, so_buoc, gen=None):
    x = torch.randn(lo.R, generator=gen)
    y = torch.zeros(lo.R)
    for i in range(so_buoc):
        t = 1 - i / so_buoc
        ab = alpha_bar(torch.tensor(t))
        y = torch.sigmoid(net(lo, x, torch.full((lo.nB,), t), y))
        if i == so_buoc - 1:
            break
        x0 = 2 * y - 1
        eps = (x - ab.sqrt() * x0) / (1 - ab).sqrt()
        ab2 = alpha_bar(torch.tensor(1 - (i + 1) / so_buoc))
        x = ab2.sqrt() * x0 + (1 - ab2).sqrt() * eps
    return y


@torch.no_grad()
def ban_do_mot_lan(net, lo):
    return torch.sigmoid(net(lo, torch.zeros(lo.R), torch.ones(lo.nB), torch.zeros(lo.R)))


def _day_du(th, v):
    yhat = np.zeros((len(th.cap), th.md.E))
    yhat[th.row_c, th.row_e] = v
    return yhat


def giai(net, th: TinhHuong2, kieu="dm", S=1, so_buoc=3, cach="pr", gen=None, rng=None, chi_tiet=False):
    """Giải một tình huống. cach: "pr" = chia luồng theo tỉ lệ các đường phân rã; "gan" = gán theo chi phí biên.
    Trả (tuyến tốt nhất, chi phí, thời gian (s)). Thời gian dựng TinhHuong2 (tính hành lang) do người gọi cộng thêm."""
    t0 = time.perf_counter()
    md = th.md
    if kieu == "dm":
        lo = Lo2([th], lap=S)
        ys = [_day_du(th, v.numpy()) for v in lo.tach(ban_do_dm(net, lo, so_buoc, gen))]
        thu_tu = [None] * S
    else:
        lo = Lo2([th])
        ys = [_day_du(th, ban_do_mot_lan(net, lo).numpy())] * S
        rng = rng or np.random.default_rng(0)
        thu_tu = [None] + [rng.permutation(md.K) for _ in range(S - 1)]
    giai_ma = kg.dung_tuyen_phan_ra if cach == "pr" else kg.dung_tuyen_gan
    tot, gia, cac = None, np.inf, []
    for y, tt in zip(ys, thu_tu):
        r = giai_ma(md, y, th.cap, th.cua, th.mask, thu_tu=tt)
        c = md.chi_phi(r)
        cac.append(c)
        if c < gia:
            tot, gia = r, c
    tg = time.perf_counter() - t0
    return (tot, gia, tg, cac, ys) if chi_tiet else (tot, gia, tg)
