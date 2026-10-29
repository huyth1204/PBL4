"""
thu_trong_so.py — Thí nghiệm nhỏ (không huấn luyện mô hình): chọn trọng số α của score

    S = α · tốc độ + (1 − α) · rủi ro đứt          (hai số hạng chuẩn hoá trong các tuyến ứng viên của cùng truy vấn)

theo HAI giai đoạn thời tiết (bão / nhẹ), giai đoạn cập nhật mỗi 5 hoặc 15 phút. Không dùng dữ liệu Yagi:
  - cửa sổ thuộc NGÀY train (chia_ngay) -> ước lượng xác suất sống của từng loại liên kết (bảng tra);
  - cửa sổ thuộc NGÀY val                -> quét α, đo trễ thực và tỉ lệ đứt của tuyến được chọn.
Thiết lập như E2 của thí nghiệm 5: mức "khắc nghiệt", tuổi 9 (thông tin cũ 3 phút), Δ = 3, đầu vào W dự báo
từ lịch quỹ đạo.

    python -m src.ngau_nhien.thu_trong_so --out results/thu_trong_so
"""

from __future__ import annotations

import argparse
import heapq
import json
import time
from pathlib import Path

import numpy as np

from src.ngau_nhien.du_bao import du_bao_quy_dao
from src.ngau_nhien.khoang_loi import MUC
from src.ngau_nhien.kich_ban import MangGoc, ThamSo, _haversine, sinh_kich_ban
from src.ngau_nhien.tao_du_lieu import chia_ngay, parse_ngay_loai
from src.ngau_nhien.thoi_tiet import ThoiTiet

BANG = [0, 5, 12, 20, 28]                                  # băng tần (0 = không chịu mưa)
MOC_MUA = np.array([0.5, 2, 5, 8, 11, 15, 20, 30, 50])     # mm/h -> 10 khoảng
MOC_GIO = np.array([10.0, 13.0, 15.0])                     # m/s  -> 4 khoảng (+1 cho "không có UAV")
LAMBDA = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.85, 0.999]
ALPHA = [round(x, 1) for x in np.arange(0, 1.01, 0.1)]
D_HONG = 1000.0


# ---------- chọn cửa sổ thời tiết (ngoài Yagi, theo ngày train / val) ----------
def chon_cua_so(tt, so_gio, so, seed):
    """so = {"train": (n_bao, n_nhe), "val": (...)}. Giờ "bão": >= 5% điểm mưa > 11 mm/h hoặc >= 20% điểm gió giật
    > 15 m/s. Cửa sổ bão có >= 4 giờ bão; cửa sổ nhẹ không có giờ bão nào nhưng có mưa (>= 1 mm/h)."""
    T = np.array([str(x) for x in tt.time_utc])
    ngay = np.array([x[:10] for x in T])
    bao_h = ((tt.precip > 11).mean(1) >= 0.05) | ((tt.gust > 15).mean(1) >= 0.20)
    mua_h = tt.precip.max(1) >= 1.0
    ntr, nva = chia_ngay(tt)
    yagi = set(parse_ngay_loai(None))
    ra = {}
    for j, (ten, tap) in enumerate((("train", set(ntr) - yagi), ("val", set(nva) - yagi))):
        dung = np.isin(ngay, list(tap))
        hop = np.zeros(len(ngay) + 1, np.int32)
        np.cumsum(~dung, out=hop[1:])
        h = np.arange(len(T) - so_gio - 1)
        ok = h[hop[h + so_gio + 1] == hop[h]]
        nb = np.array([bao_h[i:i + so_gio + 1].sum() for i in ok])
        nm = np.array([mua_h[i:i + so_gio + 1].sum() for i in ok])
        rng = np.random.default_rng(seed * 10 + j)
        chon = []
        for loai, ung, n in (("bao", ok[nb >= 4], so[ten][0]), ("nhe", ok[(nb == 0) & (nm >= 1)], so[ten][1])):
            for x in rng.permutation(ung):
                if n <= 0:
                    break
                if all(abs(c - x) >= so_gio + 1 for c, _ in chon):
                    chon.append((int(x), loai))
                    n -= 1
        ra[ten] = sorted(chon)
    return ra


# ---------- đặc trưng rủi ro của từng liên kết ----------
class DacTrung:
    def __init__(self, g):
        self.g = g
        self.bang_i = np.searchsorted(BANG, g.band)                       # [N,N] 0..4
        self.uav = np.array([k == "uav" for k in g.kind])
        self.dat = g.dat

    def khoa(self, q, to, lui=45):
        """Khoá bảng tra cho mọi cặp (u, v) tại thời điểm quan sát `to` (chỉ dùng thông tin tới `to`)."""
        g = self.g
        R, R0 = q["R"][to], q["R"][max(to - lui, 0)]
        co = g.site >= 0
        s = np.maximum(g.site, 0)
        mua = np.where(co, R[s], 0.0)
        xu = np.where(co, R[s] - R0[s], 0.0)
        mb = np.where(co, np.searchsorted(MOC_MUA, mua), 0)               # 0..9
        xb = np.where(co, np.digitize(xu, [-1.0, 1.0]), 1)                # 0 giảm, 1 đứng, 2 tăng
        ns = np.maximum(g.node_site, 0)
        gust_n = np.where(self.uav & (g.node_site >= 0), q["gust"][to, ns], -1.0)
        th_n = (g.node_site >= 0) & q["giong"][to, ns]
        gmax = np.maximum(gust_n[:, None], gust_n[None, :])
        gb = np.where(gmax < 0, 0, 1 + np.searchsorted(MOC_GIO, gmax))   # 0 không UAV, 1..4
        thu = (th_n & self.uav)[:, None] | (th_n & self.uav)[None, :]
        thd = (th_n & self.dat)[:, None] | (th_n & self.dat)[None, :]
        k = ((((self.bang_i * 10 + mb) * 3 + xb) * 5 + gb) * 2 + thu) * 2 + thd
        return k.astype(np.int32), (self.bang_i * 10 + mb).astype(np.int32), self.bang_i.astype(np.int32)


SO_KHOA = 5 * 10 * 3 * 5 * 2 * 2


def bang_tra(dem, m=30.0):
    """Xác suất sống làm trơn theo tầng: khoá đầy đủ -> (băng, mưa) -> băng."""
    a, n = dem["song"], dem["n"]
    a2, n2 = dem["song2"], dem["n2"]
    a3, n3 = dem["song3"], dem["n3"]
    p3 = (a3 + 1) / (n3 + 2)
    p2 = (a2 + m * p3[np.arange(50) // 10]) / (n2 + m)
    cha = np.arange(SO_KHOA) // (3 * 5 * 2 * 2)                          # khoá -> (băng, mưa)
    return (a + m * p2[cha]) / (n + m)


# ---------- Dijkstra trên chi phí tuỳ ý ----------
def dijkstra_c(C, s, d, can_relay):
    N = len(C)
    dist, prev, pq = np.full(N, np.inf), -np.ones(N, int), [(0.0, s)]
    dist[s] = 0.0
    while pq:
        du, u = heapq.heappop(pq)
        if u == d:
            break
        if du > dist[u] or (u != s and not can_relay[u]):
            continue
        for v in np.flatnonzero(np.isfinite(C[u])):
            nd = du + C[u, v]
            if nd < dist[v]:
                dist[v], prev[v] = nd, u
                heapq.heappush(pq, (nd, v))
    if not np.isfinite(dist[d]):
        return None
    p, v = [], d
    while v != -1:
        p.append(int(v))
        v = prev[v]
    return p[::-1]


def giai_doan(tt, g, h0, to, k_cn, toa, s, d, ban_kinh=150.0):
    """1 = bão nếu, tại lần cập nhật gần nhất (mỗi k_cn snapshot), một điểm thời tiết trong bán kính 150 km quanh
    nguồn hoặc đích có mưa (ERA5, nội suy theo giờ) > 11 mm/h hoặc gió giật > 15 m/s."""
    t_cn = (to // k_cn) * k_cn
    gio = h0 + t_cn * 20.0 / 3600
    i0 = int(np.floor(gio))
    w = gio - i0
    P = (1 - w) * tt.precip[i0] + w * tt.precip[min(i0 + 1, tt.H - 1)]
    nang = (P > 11) | (tt.gust[i0] > 15)
    for n in (s, d):
        kc = _haversine(toa[n, 0], toa[n, 1], tt.lat, tt.lon)
        if (nang & (kc <= ban_kinh)).any():
            return 1
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="dataset/oracle_dataset_v5.npz")
    ap.add_argument("--split", default="dataset/v5/split_info.json")
    ap.add_argument("--thoi-tiet", default="data/thoi_tiet_vn_mr.npz")
    ap.add_argument("--muc", default="khắc nghiệt")
    ap.add_argument("--tuoi", type=int, default=9)
    ap.add_argument("--delta", type=int, default=3)
    ap.add_argument("--so-cua-so", default="8,4,5,3", help="train bão, train nhẹ, val bão, val nhẹ")
    ap.add_argument("--buoc-fit", type=int, default=3)
    ap.add_argument("--seed", type=int, default=61)
    ap.add_argument("--out", default="results/thu_trong_so")
    a = ap.parse_args()
    out = Path(a.out)
    (out / "tam").mkdir(parents=True, exist_ok=True)
    raw = np.load(a.raw)
    tt = ThoiTiet(a.thoi_tiet)
    g = MangGoc(raw["W_snapshots"], raw["node_kind"], raw["node_order"], tt)
    toa_all = raw["node_latlon_alt"]
    split = json.load(open(a.split, encoding="utf-8"))["snapshot_ids"]
    can_relay = np.array([k != "user" for k in g.kind])
    so_gio = int(np.ceil(g.T * 20 / 3600)) + 2
    n = list(map(int, a.so_cua_so.split(",")))
    cs = chon_cua_so(tt, so_gio, {"train": (n[0], n[1]), "val": (n[2], n[3])}, a.seed)
    json.dump({"cua_so": {k: [(int(h), l, str(tt.time_utc[h])) for h, l in v] for k, v in cs.items()},
               "tham_so": vars(a)}, open(out / "cau_hinh.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print({k: [(str(tt.time_utc[h]), l) for h, l in v] for k, v in cs.items()}, flush=True)
    p = ThamSo(**MUC[a.muc])
    dt = DacTrung(g)
    tu, De = a.tuoi, a.delta

    def W_vao(Wn, q, t):
        # dự báo theo lịch quỹ đạo, chặn nút đã tắt lúc quan sát (bản sửa của nhóm, du_bao.py, 27/10/2026);
        # kết quả thí nghiệm nhỏ trong results/thu_trong_so chạy TRƯỚC bản sửa này
        return du_bao_quy_dao(g.W, Wn[t - tu], t - tu, t, De, q["tat"][t - tu])

    # 1) ước lượng xác suất sống trên cửa sổ ngày train
    tep_dem = out / "tam" / "dem_train.npz"
    if not tep_dem.exists():
        dem = {k: np.zeros(s, np.int64) for k, s in (("song", SO_KHOA), ("n", SO_KHOA), ("song2", 50), ("n2", 50),
                                                       ("song3", 5), ("n3", 5))}
        ids = sorted(split["train"])
        for e, (h0, loai) in enumerate(cs["train"]):
            t0 = time.time()
            Wn, q = sinh_kich_ban(g, tt, h0, a.seed * 1000 + e, p)
            for t in ids[::a.buoc_fit]:
                if t - tu - 45 < 0 or t + De >= g.T:
                    continue
                Wi = W_vao(Wn, q, t)
                thay = np.triu(Wi > 0, 1)
                song = ((Wn[t] > 0) & (Wn[t + 1:t + 1 + De] > 0).all(0))[thay]
                k1, k2, k3 = (x[thay] for x in dt.khoa(q, t - tu))
                for kk, ten, S in ((k1, "", SO_KHOA), (k2, "2", 50), (k3, "3", 5)):
                    dem["n" + ten] += np.bincount(kk, minlength=S)
                    dem["song" + ten] += np.bincount(kk, weights=song, minlength=S).astype(np.int64)
            del Wn, q
            print(f"train {e + 1}/{len(cs['train'])} ({loai}) xong sau {time.time() - t0:.0f}s, "
                  f"đã có {dem['n'].sum():,} mẫu liên kết", flush=True)
        np.savez(tep_dem, **dem)
    dem = dict(np.load(tep_dem))
    p_song = bang_tra(dem)
    p3 = (dem["song3"] + 1) / (dem["n3"] + 2)
    print("xác suất sống theo băng tần", dict(zip(BANG, np.round(p3, 4))), flush=True)

    # 2) quét α trên cửa sổ ngày val
    ids = sorted(split["val"])
    for e, (h0, loai) in enumerate(cs["val"]):
        tep = out / "tam" / f"val_{e}.npz"
        if tep.exists():
            continue
        t0 = time.time()
        Wn, q = sinh_kich_ban(g, tt, h0, a.seed * 1000 + 500 + e, p)
        cap = {}
        for sid, row in zip(raw["snapshot_id"], raw["y_paths_indices"]):
            if int(sid) in ids:
                r = [int(x) for x in row if x >= 0]
                cap.setdefault(int(sid), []).append((r[0], r[-1]))
        rec = {k: [] for k in ("gd5", "gd15", "D_du", "P_du", "chi_phi", "dut_tl", "tre_song", "D_hk", "so_uv", "t")}
        for t in ids:
            if t - tu - 45 < 0 or t + De >= g.T:
                continue
            Wi = W_vao(Wn, q, t)
            win = Wn[t + 1:t + 1 + De]
            k1, _, _ = dt.khoa(q, t - tu)
            ps = np.where(Wi > 0, p_song[k1], 0.0)
            r_dut = np.where(Wi > 0, -np.log(np.clip(ps, 1e-4, 1.0)), np.inf)
            song_cs = (Wn[t] > 0) & (win > 0).all(0)
            Wm = np.where(song_cs & (Wi > 0), win.mean(0), np.inf)
            Wi_c = np.where(Wi > 0, Wi, np.inf)
            toa = toa_all[t - tu]
            for s, d in cap.get(t, []):
                r0 = dijkstra_c(Wi_c, s, d, can_relay)
                if r0 is None:
                    continue
                Ds = sum(Wi[u, v] for u, v in zip(r0[:-1], r0[1:]))
                uv = []
                for lam in LAMBDA:
                    r = dijkstra_c((1 - lam) * Wi_c / Ds + lam * r_dut, s, d, can_relay)
                    if r is not None and r not in uv:
                        uv.append(r)
                rh = dijkstra_c(Wm, s, d, can_relay)
                D_hk = np.nan
                if rh is not None:
                    D_hk = float(np.mean([sum(win[i, u, v] for u, v in zip(rh[:-1], rh[1:])) for i in range(De)]))
                a_D, a_P, a_c, a_dl, a_ts = [], [], [], [], []
                for r in uv:
                    e_ = list(zip(r[:-1], r[1:]))
                    a_D.append(sum(Wi[u, v] for u, v in e_))
                    a_P.append(1 - float(np.prod([ps[u, v] for u, v in e_])))
                    vals = np.array([[win[i, u, v] for u, v in e_] for i in range(De)])
                    sg = (vals > 0).all(1)
                    a_c.append(float(np.where(sg, vals.sum(1), D_HONG).mean()))
                    a_dl.append(float(1 - sg.mean()))
                    a_ts.append(float(vals[sg].sum(1).mean()) if sg.any() else np.nan)
                K = 12
                pad = lambda x: np.pad(np.array(x, float), (0, K - len(x)), constant_values=np.nan)
                rec["gd5"].append(giai_doan(tt, g, h0, t - tu, 15, toa, s, d))
                rec["gd15"].append(giai_doan(tt, g, h0, t - tu, 45, toa, s, d))
                for k, x in (("D_du", a_D), ("P_du", a_P), ("chi_phi", a_c), ("dut_tl", a_dl), ("tre_song", a_ts)):
                    rec[k].append(pad(x))
                rec["D_hk"].append(D_hk)
                rec["so_uv"].append(len(uv))
                rec["t"].append(t)
        del Wn, q
        np.savez(tep, loai=loai, h0=h0, **{k: np.array(v) for k, v in rec.items()})
        print(f"val {e + 1}/{len(cs['val'])} ({loai}) xong sau {time.time() - t0:.0f}s: {len(rec['t'])} truy vấn, "
              f"tỉ lệ giai đoạn bão (15 phút) {np.mean(rec['gd15']) * 100:.0f}%, "
              f"ứng viên TB {np.mean(rec['so_uv']):.1f}", flush=True)
    print("XONG_THU_TRONG_SO", flush=True)


if __name__ == "__main__":
    main()
