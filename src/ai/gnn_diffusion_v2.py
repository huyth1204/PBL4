"""
gnn_diffusion_v2.py — DM cải tiến cho chọn next-hop (dùng được với GNN 0 lớp).

Ba thay đổi, bật/tắt riêng để đo đóng góp của từng cái:
  param="x0ce"  mạng dự đoán thẳng y0 (logit trên các ứng viên), học bằng cross-entropy
                thay vì dự đoán nhiễu bằng MSE (kiểu CDCD, Dieleman et al. 2022).
  self_cond     đưa ước lượng y0 của bước trước vào làm đầu vào (Chen et al. 2022, Analog Bits).
  attn          một lớp attention giữa các ứng viên hợp lệ: các ứng viên "so kè" với nhau
                qua mỗi bước khử nhiễu, thay vì được chấm điểm độc lập.
  t_power > 1   lấy mẫu bước t nghiêng về mức nhiễu cao, nơi quyết định thật sự được đưa ra
                (ở nhiễu thấp y_t đã lộ đáp án, mạng chỉ học chép lại).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.ai.gnn_diffusion import DELAY_SCALE, GraphEncoder, cosine_alpha_bar, sinusoidal_embedding

NEG = -1e9


class CandidateAttention(nn.Module):
    def __init__(self, H, heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(H, heads, batch_first=True)
        self.norm = nn.LayerNorm(H)
        self.ff = nn.Sequential(nn.Linear(H, 2 * H), nn.SiLU(), nn.Linear(2 * H, H))
        self.norm2 = nn.LayerNorm(H)

    def forward(self, z, mask):
        a, _ = self.attn(z, z, z, key_padding_mask=~mask, need_weights=False)
        z = self.norm(z + a)
        return self.norm2(z + self.ff(z))


class DenoiserV2(nn.Module):
    def __init__(self, H=64, t_dim=32, self_cond=False, attn=False):
        super().__init__()
        self.t_dim, self.self_cond = t_dim, self_cond
        self.node_in = nn.Linear(H, H)
        self.ctx_in = nn.Linear(2 * H + t_dim, H)
        self.loc_in = nn.Linear(6 + int(self_cond), H)
        self.pre = nn.Sequential(nn.SiLU(), nn.Linear(H, H))
        self.cand = CandidateAttention(H) if attn else None
        self.out = nn.Sequential(nn.SiLU(), nn.Linear(H, H), nn.SiLU(), nn.Linear(H, 1))

    def forward(self, y_t, t, h_nodes, cur, dst, rows, mask, cond, x0_prev=None):
        """cond: (S,) bool — hàng nào được biết đích (CFG bỏ điều kiện theo từng hàng)."""
        S, N = y_t.shape
        ar = torch.arange(S, device=y_t.device)
        cf = cond.float()[:, None]
        h_dst = h_nodes[ar, dst] * cf
        is_dst = torch.zeros_like(y_t)
        is_dst[ar, dst] = 1.0
        is_dst = is_dst * cf
        maskf = mask.float()
        y_mean = (y_t * maskf).sum(-1, keepdim=True) / maskf.sum(-1, keepdim=True).clamp(min=1)
        y_max = torch.where(mask, y_t, torch.full_like(y_t, -5.0)).max(-1, keepdim=True).values
        feats = [y_t, (rows > 0).float(), torch.where(rows > 0, rows / DELAY_SCALE, torch.zeros_like(rows)),
                 is_dst, y_mean.expand(S, N), y_max.expand(S, N)]
        if self.self_cond:
            feats.append(x0_prev if x0_prev is not None else torch.zeros_like(y_t))
        ctx = self.ctx_in(torch.cat([h_nodes[ar, cur], h_dst, sinusoidal_embedding(t, self.t_dim)], -1))
        z = self.pre(self.node_in(h_nodes) + ctx[:, None, :] + self.loc_in(torch.stack(feats, -1)))
        if self.cand is not None:
            z = self.cand(z, mask)
        return self.out(z).squeeze(-1)


class GraphDiffusionV2(nn.Module):
    def __init__(self, f_in, H=64, n_layers=0, T=100, param="x0ce", self_cond=False, attn=False, t_power=1.0):
        super().__init__()
        assert param in ("eps", "x0ce")
        self.enc = GraphEncoder(f_in, H, n_layers)
        self.den = DenoiserV2(H, self_cond=self_cond, attn=attn)
        self.T, self.param, self.self_cond, self.t_power = T, param, self_cond, t_power
        self.register_buffer("alpha_bar", cosine_alpha_bar(T))

    def _to_x0(self, out, y_t, ab, mask):
        """Đầu ra mạng -> ước lượng y0 (S,N)."""
        if self.param == "x0ce":
            return F.softmax(out.masked_fill(~mask, NEG), -1)
        y0 = (y_t - (1 - ab).sqrt() * out) / ab.sqrt()
        return y0.clamp(-1, 1) * mask.float()

    def training_loss(self, b):
        h = self.enc(b["X"], b["W"])[b["step_p"]]
        S, N = b["rows"].shape
        dst = b["dst"][b["step_p"]]
        cond = torch.ones(S, dtype=torch.bool) if b["drop"] is None else ~b["drop"][b["step_p"]]
        y0 = F.one_hot(b["nxt"], N).float()
        if self.t_power == 1.0:
            t = torch.randint(0, self.T, (S,))
        else:
            t = (torch.rand(S) ** (1.0 / self.t_power) * self.T).long().clamp(max=self.T - 1)
        noise = torch.randn_like(y0)
        ab = self.alpha_bar[t][:, None]
        y_t = ab.sqrt() * y0 + (1 - ab).sqrt() * noise
        x0_prev = None
        if self.self_cond:
            x0_prev = torch.zeros_like(y_t)
            use = torch.rand(S) < 0.5                      # một nửa số bước được "tự điều kiện"
            if use.any():
                with torch.no_grad():
                    out0 = self.den(y_t, t, h, b["cur"], dst, b["rows"], b["mask"], cond)
                    x0_prev = torch.where(use[:, None], self._to_x0(out0, y_t, ab, b["mask"]), x0_prev)
        out = self.den(y_t, t, h, b["cur"], dst, b["rows"], b["mask"], cond, x0_prev)
        if self.param == "x0ce":
            return F.cross_entropy(out.masked_fill(~b["mask"], NEG), b["nxt"])
        m = b["mask"].float()
        return (((out - noise) ** 2) * m).sum() / m.sum().clamp(min=1)

    @torch.no_grad()
    def decide(self, h_c, h_u, cur, dst, rows, mask, n_steps=20, guidance=2.0, return_x0=False, eta=0.0):
        """eta = 0: DDIM tất định (mặc định). eta = 1: lấy mẫu ngẫu nhiên kiểu DDPM — thêm nhiễu mới ở mỗi bước,
        dùng để sinh nhiều tuyến khác nhau."""
        S, N = rows.shape
        y = torch.randn(S, N)
        x0 = torch.zeros(S, N) if self.self_cond else None
        on, off = torch.ones(S, dtype=torch.bool), torch.zeros(S, dtype=torch.bool)
        steps = torch.linspace(self.T - 1, 0, n_steps).long()
        for i, t in enumerate(steps):
            tb = torch.full((S,), int(t), dtype=torch.long)
            ab = self.alpha_bar[int(t)]
            out = self.den(y, tb, h_c, cur, dst, rows, mask, on, x0)
            if guidance != 1.0:
                out_u = self.den(y, tb, h_u, cur, dst, rows, mask, off, x0)
                out = out_u + guidance * (out - out_u)     # x0ce: CFG trên logit; eps: CFG trên nhiễu
            y0 = self._to_x0(out, y, ab, mask)
            if self.self_cond:
                x0 = y0
            if i < n_steps - 1:
                ab_n = self.alpha_bar[int(steps[i + 1])]
                eps = (y - ab.sqrt() * y0) / (1 - ab).sqrt()
                sigma = eta * ((1 - ab_n) / (1 - ab) * (1 - ab / ab_n)).clamp(min=0).sqrt()
                y = ab_n.sqrt() * y0 + (1 - ab_n - sigma ** 2).clamp(min=0).sqrt() * eps + sigma * torch.randn_like(y)
        pick = torch.where(mask, y0, torch.full_like(y0, NEG)).argmax(-1)
        return (pick, y0) if return_x0 else pick
