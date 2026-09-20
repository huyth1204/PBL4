"""
diffusion.py — Mô hình khuếch tán có điều kiện sinh quyết định next-hop.

condition: [x_chuẩn_hoá(N*N) || one-hot(hiện tại)(N) || one-hot(đích)(N)]
target:    one-hot(next-hop)(N)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / max(half - 1, 1))
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class DenoiseMLP(nn.Module):
    def __init__(self, n_nodes, cond_dim, hidden=128, t_dim=64):
        super().__init__()
        self.t_dim = t_dim
        self.y_proj = nn.Linear(n_nodes, hidden)
        self.cond_proj = nn.Linear(cond_dim, hidden)   # nén điều kiện (sau thay bằng GNN)
        self.t_proj = nn.Sequential(nn.Linear(t_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.net = nn.Sequential(
            nn.Linear(hidden * 3, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, n_nodes),
        )

    def forward(self, y_t, t, cond):
        te = self.t_proj(sinusoidal_embedding(t, self.t_dim))
        h = torch.cat([self.y_proj(y_t), self.cond_proj(cond), te], dim=-1)
        return self.net(h)


def cosine_alpha_bar(T, s=0.008):
    # Lịch cosine: bảo đảm ᾱ_T ≈ 0 với mọi T (lịch tuyến tính chỉ đúng khi T lớn).
    steps = torch.arange(T + 1, dtype=torch.float32)
    f = torch.cos(((steps / T + s) / (1 + s)) * math.pi / 2) ** 2
    return (f / f[0])[1:]


class GaussianDiffusion(nn.Module):
    def __init__(self, net, n_nodes, T=100):
        super().__init__()
        self.net = net
        self.N = n_nodes
        self.T = T
        self.register_buffer("alpha_bar", cosine_alpha_bar(T))

    def q_sample(self, y0, t, noise):
        ab = self.alpha_bar[t][:, None]
        return torch.sqrt(ab) * y0 + torch.sqrt(1 - ab) * noise

    def training_loss(self, cond, y0, drop_prob=0.0):
        B = y0.shape[0]
        if drop_prob > 0:  # CFG: thỉnh thoảng bỏ điều kiện (thay bằng 0) để học cả p(y) lẫn p(y|x)
            keep = (torch.rand(B, device=y0.device) > drop_prob).float()[:, None]
            cond = cond * keep
        t = torch.randint(0, self.T, (B,), device=y0.device)
        noise = torch.randn_like(y0)
        pred = self.net(self.q_sample(y0, t, noise), t, cond)
        return F.mse_loss(pred, noise)

    @torch.no_grad()
    def sample(self, cond, mask, n_steps=30, guidance=1.0, clamp=1.0, neg_inf=-1e9):
        B = cond.shape[0]
        y = torch.randn(B, self.N, device=cond.device)
        steps = torch.linspace(self.T - 1, 0, n_steps).long()
        for i, t in enumerate(steps):
            tb = torch.full((B,), int(t), device=cond.device, dtype=torch.long)
            if guidance != 1.0:  # CFG: đẩy mẫu bám điều kiện mạnh hơn
                eps_c = self.net(y, tb, cond)
                eps_u = self.net(y, tb, torch.zeros_like(cond))
                eps = eps_u + guidance * (eps_c - eps_u)
            else:
                eps = self.net(y, tb, cond)
            ab_t = self.alpha_bar[int(t)]
            # KẸP y0 dự đoán: chặn nổ số khi ᾱ_t nhỏ (√ᾱ ở mẫu ≈ 0).
            y0 = ((y - torch.sqrt(1 - ab_t) * eps) / torch.sqrt(ab_t)).clamp(-clamp, clamp)
            if i < n_steps - 1:
                ab_next = self.alpha_bar[int(steps[i + 1])]
                eps = (y - torch.sqrt(ab_t) * y0) / torch.sqrt(1 - ab_t)
                y = torch.sqrt(ab_next) * y0 + torch.sqrt(1 - ab_next) * eps
            else:
                y = y0
        logits = y + torch.where(mask > 0, torch.zeros_like(y), torch.full_like(y, neg_inf))
        return logits.argmax(dim=-1)


def build_diffusion(n_nodes, cond_dim=None, hidden=128, T=100):
    if cond_dim is None:
        cond_dim = n_nodes * n_nodes + 2 * n_nodes
    return GaussianDiffusion(DenoiseMLP(n_nodes, cond_dim, hidden), n_nodes, T)