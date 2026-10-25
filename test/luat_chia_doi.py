"""
Luật chia đôi: số đo tốt (σ̂ ≤ θ) và quang đãng → Dijkstra; còn lại → DM học với nhiễu.

Nhiễu đo của liên kết (i, j): σ_ij = σ0 + max(Δ_i, Δ_j), trong đó σ0 là chất lượng đo chung của mạng và Δ là phần
thêm do mưa/giông tại nút mặt đất/trên không (thời tiết thật 2024–2025 tại 23 điểm của TV1; vệ tinh Δ = 0).
Mỗi snapshot test nhận một giờ thời tiết ngẫu nhiên. Bộ chuyển ước lượng σ0 bằng hai lần đo trên các liên kết
không dính mưa; mọi bộ định tuyến chạy trên lần đo thứ nhất; tuyến chấm trên trễ thật.

--mua vat_ly: thay bảng Δ giả định bằng Δ suy từ vật lý (src/ngau_nhien/nhieu_mua.py) cho từng liên kết:
σ_ij = sqrt(σ0² + Δ_ij²); mưa chỉ tác động liên kết đi qua lớp mưa tại điểm có mưa (đầu mặt đất hoặc UAV).
Thêm luật "nhiễu dự báo dọc tuyến": Dijkstra nếu mọi liên kết trên tuyến có sqrt(σ̂0² + Δ_ij²) ≤ θ.

    python test/luat_chia_doi.py --cau-hinh "73=dataset" "173=data/quy_mo/n173" ... \
        --thoi-tiet data/thoi_tiet_vn.npz --thoi-gian results/quy_mo/thoi_gian.json \
        --mo-hinh "DM học với nhiễu=results/quy_mo/dm_nhieu.pt" --out results/quy_mo/chia_doi.json
"""
import argparse
import copy
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
from test.so_sanh_dijkstra_nhieu import dijkstra, thong_ke_loai, tre  # noqa: E402
from src.ngau_nhien.kich_ban import MangGoc  # noqa: E402
from src.ngau_nhien.nhieu_mua import delta as delta_vat_ly  # noqa: E402
from src.ngau_nhien.thoi_tiet import ThoiTiet  # noqa: E402

MUA = ((0.1, 0.10), (2.5, 0.25), (10.0, 0.40))      # mưa trên ngưỡng (mm/h) -> σ cộng thêm
GIONG, GIONG_THEM, SIGMA_MAX = (95, 96, 97, 99), 0.20, 0.8


def delta_diem(precip, code):
    d = np.zeros(precip.shape, np.float32)
    for nguong, them in MUA:
        d[precip > nguong + 1e-4] = them
    return d + GIONG_THEM * np.isin(code, GIONG)


def nhieu_lk(W, S, rng):
    """W (N,N) trễ thật, S (N,N) σ từng liên kết."""
    z = np.triu(rng.standard_normal(W.shape).astype(np.float32), 1)
    z = z + z.T
    return np.where(W > 0, np.maximum(W * (1 + S * z), 0.05 * W), W).astype(np.float32)


def uoc_luong_sigma0(A, B, co):
    r = (A[co] - B[co]) / ((A[co] + B[co]) / 2)
    return float(np.median(np.abs(r)) / 0.6745 / np.sqrt(2)) if co.any() else 0.0


def tb_loai(loai, tk):
    mu, v = np.zeros(loai.shape, np.float32), np.zeros(loai.shape, np.float32)
    for L, (m, s2) in tk.items():
        mu[loai == L], v[loai == L] = m, s2
    return mu, v


def khu_nhieu_tung_lk(Wd, S, mu, v):
    """Co số đo về trung bình loại liên kết như so_sanh_dijkstra_nhieu, nhưng với σ riêng từng liên kết."""
    lam = np.where(v > 0, v / np.maximum(v + S ** 2 * (mu ** 2 + v), 1e-12), 0.0)
    return np.where(Wd > 0, mu + lam * (Wd - mu), Wd).astype(np.float32)


def delta_matran(g, te, tt, s, h):
    """Δ vật lý (N,N) và mặt nạ liên kết đi qua mưa, cho snapshot s ở giờ thời tiết h."""
    R, G = tt.precip[h], np.isin(tt.code[h], GIONG)
    ss = np.maximum(g.site, 0)
    mua = (te.W[s] > 0) & (g.site >= 0) & ((R[ss] > 0.1 + 1e-4) | G[ss])
    D = np.zeros(te.W[s].shape, np.float32)
    iu, iv = np.nonzero(mua)
    if len(iu):
        d, _ = delta_vat_ly(R[ss[iu, iv]], G[ss[iu, iv]].astype(float), g.band[iu, iv], g.C[iu, iv],
                            g.Ls[s, iu, iv], g.LG[s, iu, iv], te.W[s, iu, iv])
        D[iu, iv] = np.where(np.isfinite(d), d, 10.0)                        # nghẽn suốt giờ: nhiễu tối đa
    return D, mua


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cau-hinh", nargs="+", required=True, help='"N=thư_mục"')
    ap.add_argument("--mo-hinh", nargs="+", required=True, help='"tên=đường_dẫn.pt"; mô hình đầu tiên dùng trong luật')
    ap.add_argument("--thoi-tiet", required=True)
    ap.add_argument("--thoi-gian", required=True, help="thoi_gian.json: ms mỗi đường của Dijkstra và DM")
    ap.add_argument("--sigma0", default="0,0.05,0.1,0.2,0.4")
    ap.add_argument("--nguong", default="0.05,0.1")
    ap.add_argument("--n-duong", type=int, default=1000)
    ap.add_argument("--seeds", default="1,2")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--mua", choices=["gia_dinh", "vat_ly"], default="gia_dinh")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    tt = np.load(a.thoi_tiet, allow_pickle=True)
    ten_diem = [str(x) for x in tt["site_names"]]
    delta_gio = delta_diem(tt["precip"], tt["code"])                       # (giờ, 23)
    TG = json.load(open(a.thoi_gian, encoding="utf-8"))
    mo_hinh = [(t.split("=", 1)[0], load(t.split("=", 1)[1])) for t in a.mo_hinh]
    DM = mo_hinh[0][0]
    s0s, nguongs, seeds = ([float(x) for x in v.split(",")] for v in (a.sigma0, a.nguong, a.seeds))
    kq, tung = {}, {}
    for c in a.cau_hinh:
        n, thu_muc = c.split("=", 1)
        te = SagsinData(Path(thu_muc) / "test.npz")
        mu, v = tb_loai(*thong_ke_loai(SagsinData(Path(thu_muc) / "train.npz")))
        cot = {x: i for i, x in enumerate(ten_diem)}
        vt = np.array([cot.get(str(x), -1) for x in te.node_order])     # cột thời tiết của từng nút, -1 = vệ tinh
        idx = np.sort(np.random.default_rng(0).choice(len(te), min(a.n_duong, len(te)), replace=False))
        if a.mua == "vat_ly":
            tto = ThoiTiet(a.thoi_tiet)
            g = MangGoc(te.W, te.kind, te.node_order, tto)
        t_d, t_dm = TG[n]["ms_dijkstra_python"], TG[n]["ms_dm_tung_duong"]
        kq[n] = {}
        for s0 in s0s:
            acc = {}
            for sd in seeds:
                rng = np.random.default_rng(int(sd) * 7919 + int(round(s0 * 1000)))
                snaps = np.unique(te.sid[idx])
                gio = dict(zip(snaps.tolist(), rng.integers(0, len(delta_gio), len(snaps)).tolist()))
                A, K = te.W.copy(), te.W.copy()
                s0_hat, mua_vung, mua_lk, D_lk = {}, {}, {}, {}
                for s in snaps:
                    if a.mua == "vat_ly":
                        D_lk[s], mua_lk[s] = delta_matran(g, te, tto, s, gio[s])
                        S = np.minimum(np.sqrt(s0 ** 2 + D_lk[s] ** 2), SIGMA_MAX)
                    else:
                        dn = np.where(vt >= 0, delta_gio[gio[s]][np.maximum(vt, 0)], 0.0).astype(np.float32)
                        S = np.minimum(s0 + np.maximum(dn[:, None], dn[None, :]), SIGMA_MAX)
                        mua_lk[s] = (te.W[s] > 0) & ((dn[:, None] > 0) | (dn[None, :] > 0))
                    A[s] = nhieu_lk(te.W[s], S, rng)
                    K[s] = khu_nhieu_tung_lk(A[s], S, mu, v)
                    B = nhieu_lk(te.W[s], S, rng)
                    s0_hat[s] = uoc_luong_sigma0(A[s], B, np.triu(te.W[s] > 0, 1) & ~mua_lk[s])
                    mua_vung[s] = bool((delta_gio[gio[s]] > 0).any())
                d = copy.copy(te)
                d.W = A
                dij = [dijkstra(A[te.sid[i]], te.paths[i][0], te.paths[i][-1], te.can_relay) for i in idx]
                kn = [dijkstra(K[te.sid[i]], te.paths[i][0], te.paths[i][-1], te.can_relay) for i in idx]
                ra = {"Dijkstra trên số đo": dij, "Dijkstra khử nhiễu (biết σ từng liên kết)": kn}
                for ten, m in mo_hinh:
                    rt, ok = chay_mo_hinh(m, d, idx, 1.0, 2, 200)
                    ra[ten] = [r if o else None for r, o in zip(rt, ok)]
                sang_dm, chi_phi = {}, {}
                for th in nguongs:
                    tot = np.array([s0_hat[te.sid[i]] <= th for i in idx])     # số đo đủ tốt: mới cần chạy Dijkstra
                    vung = np.array([not (s0_hat[te.sid[i]] <= th and not mua_vung[te.sid[i]]) for i in idx])
                    tuyen = np.array([not (s0_hat[te.sid[i]] <= th and p is not None and
                                           not mua_lk[te.sid[i]][p[:-1], p[1:]].any()) for i, p in zip(idx, dij)])
                    luat = [(f"Chia đôi θ={th:g}, quang đãng toàn vùng", vung),
                            (f"Chia đôi θ={th:g}, quang đãng dọc tuyến", tuyen)]
                    if a.mua == "vat_ly":
                        du_bao = np.array([not (s0_hat[te.sid[i]] <= th and p is not None and np.sqrt(
                            s0_hat[te.sid[i]] ** 2 + D_lk[te.sid[i]][p[:-1], p[1:]] ** 2).max() <= th)
                            for i, p in zip(idx, dij)])
                        luat.append((f"Chia đôi θ={th:g}, nhiễu dự báo dọc tuyến", du_bao))
                    for ten_l, sang in luat:
                        ra[ten_l] = [ra[DM][k] if sang[k] else dij[k] for k in range(len(idx))]
                        sang_dm[ten_l] = sang
                        chay_d = tot if "dọc tuyến" in ten_l else ~sang
                        chi_phi[ten_l] = float(np.mean(chay_d * t_d + sang * t_dm))
                for ten, rt in ra.items():
                    rat, trung = np.full(len(idx), np.nan), []
                    for k, i in enumerate(idx):
                        if rt[k] is None:
                            continue
                        W = te.W[te.sid[i]]
                        rat[k] = tre(W, rt[k]) / tre(W, te.paths[i])
                        trung.append(list(rt[k]) == list(te.paths[i]))
                    p_dm = float(sang_dm[ten].mean()) if ten in sang_dm else (1.0 if ten in dict(mo_hinh) else 0.0)
                    ms = chi_phi.get(ten, t_dm if ten in dict(mo_hinh) else t_d)
                    acc.setdefault(ten, []).append({
                        "tre_vuot_%": float(np.nanmean(rat) * 100 - 100), "trung_toi_uu_%": float(np.mean(trung) * 100),
                        "toi_dich_%": float(np.isfinite(rat).mean() * 100), "sang_DM_%": p_dm * 100, "ms_moi_truy_van": ms})
                    tung.setdefault(f"{n}|{s0}|{ten}", []).append(rat)
                sai = [abs(s0_hat[s] - s0) for s in snaps]
                acc.setdefault("_uoc_luong", []).append({"sai_so_sigma0_TB": float(np.mean(sai)),
                                                         "mua_toan_vung_%": float(np.mean([mua_vung[s] for s in snaps]) * 100)})
            kq[n][str(s0)] = {ten: {k: float(np.mean([x[k] for x in v])) for k in v[0]} for ten, v in acc.items()}
            print(n, f"σ0={s0:.0%}", " | ".join(f"{ten}: {r['tre_vuot_%']:.2f}% (DM {r['sang_DM_%']:.0f}%)"
                                                for ten, r in kq[n][str(s0)].items() if ten != "_uoc_luong"), flush=True)
    json.dump(kq, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    np.savez_compressed(Path(a.out).with_suffix(".npz"), **{k: np.array(v) for k, v in tung.items()})


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"xong sau {time.time() - t0:.0f} s")
