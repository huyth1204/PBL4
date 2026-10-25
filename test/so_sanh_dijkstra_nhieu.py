"""
So sánh DM với Dijkstra ở mọi quy mô khi độ trễ đo được có nhiễu: bộ định tuyến chỉ thấy W_đo = W·(1+σz),
tuyến được chấm bằng độ trễ THẬT. Không huấn luyện lại: mô hình học trên số liệu sạch, chạy trên số liệu nhiễu.

    python test/so_sanh_dijkstra_nhieu.py --cau-hinh "73=dataset" "173=data/quy_mo/n173" ... \
        --mo-hinh "DM chung=results/quy_mo/dm_da_quy_mo.pt" "DM học với nhiễu=results/quy_mo/dm_nhieu.pt" \
        --out results/quy_mo/nhieu_do.json
Tệp .npz đi kèm lưu tỉ lệ trễ/tối ưu của từng truy vấn (khóa "N|σ|phương pháp", mỗi seed một hàng) để so sánh cặp.
"""
import argparse
import copy
import heapq
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.ai.data_v2 import SagsinData  # noqa: E402
from test.danh_gia_quy_mo import chay_mo_hinh, load  # noqa: E402


def dijkstra(W, s, d, can_relay, theo_so_chang=False):
    N = len(W)
    dist, prev, pq = np.full(N, np.inf), -np.ones(N, int), [(0.0, s)]
    dist[s] = 0
    while pq:
        du, u = heapq.heappop(pq)
        if u == d:
            break
        if du > dist[u] or (u != s and not can_relay[u]):
            continue
        for v in np.flatnonzero(W[u] > 0):
            w = 1.0 + 1e-6 * W[u, v] if theo_so_chang else W[u, v]     # số chặng, hòa thì theo trễ đo
            if du + w < dist[v]:
                dist[v], prev[v] = du + w, u
                heapq.heappush(pq, (dist[v], v))
    if not np.isfinite(dist[d]):
        return None
    p, v = [], d
    while v != -1:
        p.append(int(v))
        v = prev[v]
    return p[::-1]


def tre(W, r):
    return float(sum(W[a, b] for a, b in zip(r[:-1], r[1:])))


def lam_nhieu(W, sigma, seed):
    """Nhiễu nhân tính, cùng giá trị cho hai chiều của một liên kết, không để trễ âm."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(W.shape).astype(np.float32)
    z = np.triu(z, 1) + np.transpose(np.triu(z, 1), (0, 2, 1))
    return np.where(W > 0, np.maximum(W * (1 + sigma * z), 0.05 * W), W).astype(np.float32)


def thong_ke_loai(tr):
    """Trung bình và phương sai trễ THẬT theo loại liên kết (cặp loại nút), lấy từ tập train."""
    kind = np.array([str(k) for k in tr.kind])
    loai = np.array([[hash(tuple(sorted((a, b)))) for b in kind] for a in kind])
    co = tr.W > 0
    tk = {}
    for L in np.unique(loai):
        x = tr.W[co & (loai[None] == L)]
        if len(x):
            tk[L] = (float(x.mean()), float(x.var()))
    return loai, tk


def co_ve_trung_binh(Wd, sigma, loai, tk, c=1.0):
    """Dijkstra khử nhiễu kiểu Bayes: co số đo về trung bình của loại liên kết, hệ số theo tỉ lệ phương sai.
    Nhiễu nhân tính có phương sai σ²·W², lấy xấp xỉ σ²·(μ² + s²) cho cả loại. c nhân thêm vào λ (chỉnh trên val)."""
    if sigma == 0:
        return Wd
    if not np.isfinite(sigma):                      # bỏ hẳn số đo, chỉ dùng trung bình loại liên kết
        sigma = 1e9
    mu = np.zeros(loai.shape, np.float32)
    lam = np.ones(loai.shape, np.float32)
    for L, (m, v) in tk.items():
        mu[loai == L] = m
        lam[loai == L] = min(1.0, c * v / (v + sigma ** 2 * (m ** 2 + v))) if v > 0 else 0.0
    return np.where(Wd > 0, mu[None] + lam[None] * (Wd - mu[None]), Wd).astype(np.float32)


def chinh_c(va, loai, tk, sigma, seed, n=300, luoi=(0, 0.25, 0.5, 1, 1.5, 2, 3, 5)):
    """Chọn hệ số c cho λ trên val (cùng mô hình nhiễu, nhiễu khác test): c cho trễ thật trung bình nhỏ nhất."""
    idx = np.random.default_rng(1).choice(len(va), min(n, len(va)), replace=False)
    Wd = lam_nhieu(va.W, sigma, seed)
    tot, c_tot = np.inf, 1.0
    for c in luoi:
        Wk = co_ve_trung_binh(Wd, sigma, loai, tk, c)
        r = []
        for i in idx:
            p = dijkstra(Wk[va.sid[i]], va.paths[i][0], va.paths[i][-1], va.can_relay)
            if p is not None:
                W = va.W[va.sid[i]]
                r.append(tre(W, p) / tre(W, va.paths[i]))
        if np.mean(r) < tot:
            tot, c_tot = np.mean(r), c
    return c_tot


def cham(te, idx, routes, oks, luu=None):
    rat, trung, gan, moi = [], [], [], np.full(len(idx), np.nan)
    for k, i in enumerate(idx):
        if not oks[k]:
            continue
        W = te.W[te.sid[i]]
        r = tre(W, routes[k]) / tre(W, te.paths[i])
        rat.append(r)
        moi[k] = r
        gan.append(r <= 1.01)
        trung.append(list(routes[k]) == list(te.paths[i]))
    if luu is not None:
        luu.append(moi)
    return {"toi_dich_%": float(np.mean(oks) * 100), "tre_so_voi_toi_uu": float(np.mean(rat)),
            "trong_1pct_%": float(np.mean(gan) * 100), "trung_toi_uu_%": float(np.mean(trung) * 100)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cau-hinh", nargs="+", required=True, help='"N=thư_mục" hoặc "N=thư_mục:mô_hình_riêng.pt"')
    ap.add_argument("--mo-hinh", nargs="+", default=[], help='"tên=đường_dẫn.pt" dùng cho mọi quy mô')
    ap.add_argument("--sigma", default="0,0.1,0.2,0.4,0.6")
    ap.add_argument("--n-duong", type=int, default=1000)
    ap.add_argument("--seeds", default="1,2")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    sigmas = [float(x) for x in a.sigma.split(",")]
    seeds = [int(x) for x in a.seeds.split(",")]
    chung = [(t.split("=", 1)[0], load(t.split("=", 1)[1])) for t in a.mo_hinh]
    kq, tung_duong, chon_c = {}, {}, {}
    for c in a.cau_hinh:
        n, rest = c.split("=", 1)
        parts = rest.split(":")
        te = SagsinData(Path(parts[0]) / "test.npz")
        loai, tk = thong_ke_loai(SagsinData(Path(parts[0]) / "train.npz"))
        va = SagsinData(Path(parts[0]) / "val.npz")
        idx = np.sort(np.random.default_rng(0).choice(len(te), min(a.n_duong, len(te)), replace=False))
        mo_hinh = chung + ([("DM riêng quy mô", load(parts[1]))] if len(parts) > 1 else [])
        kq[n] = {}
        for sg in sigmas:
            acc = {}
            for sd in (seeds if sg > 0 else seeds[:1]):
                d = copy.copy(te)
                d.W = lam_nhieu(te.W, sg, sd * 1000 + int(sg * 100))
                cs = [("Dijkstra trên trễ đo", d.W, False), ("Dijkstra theo số chặng", d.W, True),
                      ("Dijkstra khử nhiễu (biết σ)", co_ve_trung_binh(d.W, sg, loai, tk), False),
                      ("Dijkstra theo trễ TB loại liên kết", co_ve_trung_binh(d.W, np.inf, loai, tk), False)]
                if sg > 0:
                    c = chinh_c(va, loai, tk, sg, sd * 1000 + int(sg * 100) + 500)
                    chon_c.setdefault(f"{n}|{sg}", []).append(c)
                    cs.append(("Dijkstra khử nhiễu, chỉnh trên val", co_ve_trung_binh(d.W, sg, loai, tk, c), False))
                for ten, Wx, hop in cs:
                    rt = [dijkstra(Wx[te.sid[i]], te.paths[i][0], te.paths[i][-1], te.can_relay, hop) for i in idx]
                    ok = np.array([r is not None for r in rt])
                    acc.setdefault(ten, []).append(cham(te, idx, [r or [] for r in rt], ok,
                                                        tung_duong.setdefault(f"{n}|{sg}|{ten}", [])))
                for ten, m in mo_hinh:
                    rt, ok = chay_mo_hinh(m, d, idx, 1.0, 2, 200)
                    acc.setdefault(ten, []).append(cham(te, idx, rt, ok, tung_duong.setdefault(f"{n}|{sg}|{ten}", [])))
            kq[n][str(sg)] = {ten: {k: float(np.mean([x[k] for x in v])) for k in v[0]} for ten, v in acc.items()}
            print(n, f"σ={sg:.0%}", " | ".join(f"{ten}: {r['tre_so_voi_toi_uu']:.4f}, trùng {r['trung_toi_uu_%']:.1f}%"
                                               for ten, r in kq[n][str(sg)].items()), flush=True)
    kq["c_chon_tren_val"] = chon_c
    json.dump(kq, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    np.savez_compressed(Path(a.out).with_suffix(".npz"), **{k: np.array(v) for k, v in tung_duong.items()})


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"xong sau {time.time() - t0:.0f} s")
