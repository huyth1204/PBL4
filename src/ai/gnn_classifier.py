"""
gnn_classifier.py — Đối chứng cho nấc B: cùng bộ mã hóa GNN, nhưng thay diffusion
bằng bộ phân loại softmax trên các ứng viên hợp lệ (cross-entropy).

Nếu biến thể này đạt ngang GraphDiffusion thì độ chính xác đến từ GNN,
còn diffusion chỉ đóng góp những thứ phân loại không có (guidance, lấy mẫu nhiều đường).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.ai.gnn_diffusion import DELAY_SCALE, GraphEncoder
from src.ai.gnn_diffusion_v2 import CandidateAttention


class NodeScorer(nn.Module):
    """Giống NodeDenoiser nhưng không có y_t và bước t."""

    def __init__(self, H=64, attn=False):
        super().__init__()
        self.node_in = nn.Linear(H, H)
        self.ctx_in = nn.Linear(2 * H, H)
        self.loc_in = nn.Linear(3, H)                      # có cạnh, trễ cạnh, là đích
        if attn:                                           # đối chứng cho DM có attention ứng viên
            self.pre = nn.Sequential(nn.SiLU(), nn.Linear(H, H))
            self.cand = CandidateAttention(H)
        self.attn = attn
        self.net = nn.Sequential(nn.SiLU(), nn.Linear(H, H), nn.SiLU(), nn.Linear(H, 1))

    def forward(self, h_nodes, cur, dst, rows, mask=None):
        S, N = rows.shape
        ar = torch.arange(S, device=rows.device)
        is_dst = torch.zeros_like(rows)
        is_dst[ar, dst] = 1.0
        loc = torch.stack([(rows > 0).float(),
                           torch.where(rows > 0, rows / DELAY_SCALE, torch.zeros_like(rows)), is_dst], -1)
        ctx = self.ctx_in(torch.cat([h_nodes[ar, cur], h_nodes[ar, dst]], -1))
        z = self.node_in(h_nodes) + ctx[:, None, :] + self.loc_in(loc)
        if self.attn:
            z = self.cand(self.pre(z), mask)
        return self.net(z).squeeze(-1)


class GraphClassifier(nn.Module):
    def __init__(self, f_in, H=64, n_layers=4, attn=False):
        super().__init__()
        self.enc = GraphEncoder(f_in, H, n_layers)
        self.head = NodeScorer(H, attn)

    def logits(self, h, cur, dst, rows, mask):
        return self.head(h, cur, dst, rows, mask).masked_fill(~mask, -1e9)

    def training_loss(self, b):
        h = self.enc(b["X"], b["W"])[b["step_p"]]
        lg = self.logits(h, b["cur"], b["dst"][b["step_p"]], b["rows"], b["mask"])
        return F.cross_entropy(lg, b["nxt"])

    @torch.no_grad()
    def decide(self, h_c, h_u, cur, dst, rows, mask, n_steps=None, guidance=None, temperature=0.0):
        """Cùng giao diện với GraphDiffusion.decide; h_u, n_steps, guidance bị bỏ qua.
        temperature > 0: lấy mẫu từ softmax thay vì argmax (để so độ đa dạng)."""
        lg = self.logits(h_c, cur, dst, rows, mask)
        if temperature > 0:
            return torch.multinomial(F.softmax(lg / temperature, -1), 1).squeeze(-1)
        return lg.argmax(-1)
