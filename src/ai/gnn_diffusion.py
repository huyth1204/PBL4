"""
gnn_diffusion.py — Nấc B: bộ mã hóa GNN + diffusion chấm điểm từng nút.

Ý tưởng (giống Bellman-Ford / distance-vector):
  * GNN chỉ biết NÚT ĐÍCH -> học cho mỗi nút j một biểu diễn kiểu "còn xa đích bao nhiêu".
  * Bộ khử nhiễu ghép biểu diễn của j với chi phí cạnh cur->j để chấm điểm ứng viên j.
  * Diffusion sinh vector one-hot trên các ứng viên hợp lệ; argmax -> next-hop.

GNN không phụ thuộc số nút N, nên cùng một mô hình chạy được cho mạng lớn/nhỏ khác nhau.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

DELAY_SCALE = 20.0


def sinusoidal_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / max(half - 1, 1))
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


def cosine_alpha_bar(T, s=0.008):
    steps = torch.arange(T + 1, dtype=torch.float32)
    f = torch.cos(((steps / T + s) / (1 + s)) * math.pi / 2) ** 2
    return (f / f[0])[1:]


class GNNLayer(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.msg = nn.Linear(H, H)
        self.upd = nn.Sequential(nn.Linear(3 * H, H), nn.SiLU(), nn.Linear(H, H))
        self.norm = nn.LayerNorm(H)

    def forward(self, h, A_mean, A_delay):
        m = self.msg(h)
        # hai kênh gom tin: trung bình hàng xóm, và trung bình ưu tiên cạnh trễ thấp
        agg = torch.cat([h, A_mean @ m, A_delay @ m], -1)
        return self.norm(h + self.upd(agg))


class GraphEncoder(nn.Module):
    def __init__(self, f_in, H=64, n_layers=4):
        super().__init__()
        self.inp = nn.Linear(f_in, H)
        self.layers = nn.ModuleList([GNNLayer(H) for _ in range(n_layers)])
        self.log_tau = nn.Parameter(torch.tensor(math.log(8.0)))  # thang độ trễ (ms), học được

    def forward(self, X, W):
        A = (W > 0).float()
        A_mean = A / A.sum(-1, keepdim=True).clamp(min=1)
        Wd = torch.where(W > 0, torch.exp(-W / self.log_tau.exp()), torch.zeros_like(W))
        A_delay = Wd / Wd.sum(-1, keepdim=True).clamp(min=1e-6)
        h = self.inp(X)
        for layer in self.layers:
            h = layer(h, A_mean, A_delay)
        return h


class NodeDenoiser(nn.Module):
    """ε_θ cho từng nút i: dựa trên h_i, h_cur, h_dst, bước t, y_t[i], cạnh cur->i."""

    def __init__(self, H=64, t_dim=32):
        super().__init__()
        self.t_dim = t_dim
        self.node_in = nn.Linear(H, H)
        self.ctx_in = nn.Linear(2 * H + t_dim, H)          # h_cur, h_dst, emb(t)
        self.loc_in = nn.Linear(6, H)                      # y_t, có cạnh, trễ cạnh, là đích, mean/max y_t ứng viên
        self.net = nn.Sequential(nn.SiLU(), nn.Linear(H, H), nn.SiLU(), nn.Linear(H, 1))

    def forward(self, y_t, t, h_nodes, cur, dst, rows, mask, cond=True):
        S, N = y_t.shape
        ar = torch.arange(S, device=y_t.device)
        h_cur = h_nodes[ar, cur]
        h_dst = h_nodes[ar, dst] if cond else torch.zeros_like(h_cur)
        is_dst = torch.zeros_like(y_t)
        if cond:
            is_dst[ar, dst] = 1.0
        maskf = mask.float()
        cnt = maskf.sum(-1, keepdim=True).clamp(min=1)
        y_mean = (y_t * maskf).sum(-1, keepdim=True) / cnt
        y_max = torch.where(mask, y_t, torch.full_like(y_t, -5.0)).max(-1, keepdim=True).values
        edge = (rows > 0).float()
        loc = torch.stack([
            y_t, edge, torch.where(rows > 0, rows / DELAY_SCALE, torch.zeros_like(rows)), is_dst,
            y_mean.expand(S, N), y_max.expand(S, N),
        ], -1)
        ctx = self.ctx_in(torch.cat([h_cur, h_dst, sinusoidal_embedding(t, self.t_dim)], -1))
        z = self.node_in(h_nodes) + ctx[:, None, :] + self.loc_in(loc)
        return self.net(z).squeeze(-1)


class GraphDiffusion(nn.Module):
    def __init__(self, f_in, H=64, n_layers=4, T=100):
        super().__init__()
        self.enc = GraphEncoder(f_in, H, n_layers)
        self.den = NodeDenoiser(H)
        self.T = T
        self.register_buffer("alpha_bar", cosine_alpha_bar(T))

    def training_loss(self, b):
        """b: lô tensor từ make_batch (đã chuyển sang torch)."""
        h_paths = self.enc(b["X"], b["W"])                 # (P,N,H) — một lần cho cả đường
        h = h_paths[b["step_p"]]                           # (S,N,H)
        S, N = b["rows"].shape
        y0 = F.one_hot(b["nxt"], N).float()
        t = torch.randint(0, self.T, (S,), device=y0.device)
        noise = torch.randn_like(y0)
        ab = self.alpha_bar[t][:, None]
        y_t = ab.sqrt() * y0 + (1 - ab).sqrt() * noise
        cond = torch.ones(S, dtype=torch.bool)
        if b["drop"] is not None:
            cond = ~b["drop"][b["step_p"]]
        # bước bị bỏ điều kiện (CFG): không cho biết đích
        eps_c = self.den(y_t, t, h, b["cur"], b["dst"][b["step_p"]], b["rows"], b["mask"], cond=True)
        if (~cond).any():
            eps_u = self.den(y_t, t, h, b["cur"], b["dst"][b["step_p"]], b["rows"], b["mask"], cond=False)
            eps = torch.where(cond[:, None], eps_c, eps_u)
        else:
            eps = eps_c
        m = b["mask"].float()
        # chỉ tính lỗi trên các ứng viên hợp lệ — nút bị mặt nạ loại không bao giờ được chọn
        return (((eps - noise) ** 2) * m).sum() / m.sum().clamp(min=1)

    @torch.no_grad()
    def decide(self, h_c, h_u, cur, dst, rows, mask, n_steps=20, guidance=2.0, clamp=1.0):
        """Chọn next-hop cho S quyết định. h_c/h_u: (S,N,H) mã hóa có/không đích."""
        S, N = rows.shape
        y = torch.randn(S, N)
        steps = torch.linspace(self.T - 1, 0, n_steps).long()
        for i, t in enumerate(steps):
            tb = torch.full((S,), int(t), dtype=torch.long)
            eps = self.den(y, tb, h_c, cur, dst, rows, mask, cond=True)
            if guidance != 1.0:
                eps_u = self.den(y, tb, h_u, cur, dst, rows, mask, cond=False)
                eps = eps_u + guidance * (eps - eps_u)
            ab_t = self.alpha_bar[int(t)]
            y0 = ((y - (1 - ab_t).sqrt() * eps) / ab_t.sqrt()).clamp(-clamp, clamp)
            if i < n_steps - 1:
                ab_n = self.alpha_bar[int(steps[i + 1])]
                eps = (y - ab_t.sqrt() * y0) / (1 - ab_t).sqrt()
                y = ab_n.sqrt() * y0 + (1 - ab_n).sqrt() * eps
            else:
                y = y0
        return torch.where(mask, y, torch.full_like(y, -1e9)).argmax(-1)
