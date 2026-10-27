"""
khoang_loi.py — Tính "khoảng lợi" trước khi huấn luyện: trong thế giới ngẫu nhiên, Dijkstra dùng số đo hiện tại
thua tuyến tốt nhất (biết trước tương lai) bao nhiêu, và một bộ dự báo đơn giản lấy lại được bao nhiêu.

Mỗi truy vấn test (t, s, d): chọn tuyến tại t, giữ tuyến trong Δ snapshot kế tiếp (t+1 … t+Δ).
  chi phí   D(P) = trung bình theo τ của [tổng trễ của P tại τ, hoặc D_hong nếu P đứt tại τ]
  ngay_tho  Dijkstra(W_t)                                                   — cách làm hiện nay
  quy_dao   Dijkstra(trung bình W_gốc của cửa sổ × (W_t / W_gốc,t))           — biết lịch quỹ đạo (phần tất định)
            và giả sử phần ngẫu nhiên giữ nguyên như hiện tại: đối chứng cổ điển mạnh nhất
  hau_kiem  Dijkstra(trung bình W của cửa sổ, chỉ liên kết có tại t và sống suốt cửa sổ) — tối ưu khi biết tương lai
Khoảng lợi cho mô hình học = quy_dao − hau_kiem (phần do ngẫu nhiên, lịch quỹ đạo không bù được).

    python -m src.ngau_nhien.khoang_loi --raw dataset/oracle_dataset_v2.npz --split dataset/split_info.json \
        --thoi-tiet data/thoi_tiet_vn.npz --so-kich-ban 30 --out results/khoang_loi.json
"""

from __future__ import annotations

import argparse
import heapq
import json
import time
from collections import defaultdict
from multiprocessing import Pool

import numpy as np

from src.ngau_nhien.kich_ban import MangGoc, ThamSo, sinh_kich_ban, tham_so_dict
from src.ngau_nhien.thoi_tiet import ThoiTiet, doc_thang
from src.ngau_nhien.du_bao import du_bao_quy_dao

CAU_HINH = {
    "tất định (như TV1)": dict(mua=False, giong=False, xung_dot=False, tai=False),
    "chỉ mưa": dict(mua=True, giong=False, xung_dot=False, tai=False),
    "chỉ giông": dict(mua=False, giong=True, xung_dot=False, tai=False),
    "chỉ xung đột": dict(mua=False, giong=False, xung_dot=True, tai=False),
    "chỉ tải": dict(mua=False, giong=False, xung_dot=False, tai=True),
    "tất cả": dict(mua=True, giong=True, xung_dot=True, tai=True),
}
MUC = {"thường": {},
       "khắc nghiệt": dict(sigma_mua=1.2, snr0_db=12.0, p_set=0.1, lam_xd=1.0, rho_tb=0.7)}


def dijkstra(W, s, d, can_relay):
    N = len(W)
    dist, prev, pq = np.full(N, np.inf), -np.ones(N, int), [(0.0, s)]
    dist[s] = 0.0
    while pq:
        du, u = heapq.heappop(pq)
        if u == d:
            break
        if du > dist[u] or (u != s and not can_relay[u]):
            continue
        for v in np.flatnonzero(W[u] > 0):
            nd = du + W[u, v]
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


def chi_phi(win, r, D_hong):
    """(chi phí trung bình trong cửa sổ, có đứt không)."""
    a, b = np.array(r[:-1]), np.array(r[1:])
    vals = win[:, a, b]
    song = (vals > 0).all(1)
    c = np.where(song, vals.sum(1), D_hong)
    return float(c.mean()), bool(not song.all())


def danh_gia(Wn, Wg, truy_van, deltas, tuois, can_relay, D_hong, tat=None):
    """tuoi = độ cũ của thông tin (snapshot): bộ định tuyến ở t chỉ thấy W_{t−tuoi}."""
    T = len(Wn)
    theo_t = defaultdict(list)
    for t, s, d, nhan in truy_van:
        theo_t[t].append((s, d, nhan))
    out = {(a, D): defaultdict(list) for a in tuois for D in deltas}
    for t, qs in theo_t.items():
        for a in tuois:
            if t - a < 0:
                continue
            W0 = Wn[t - a]
            ngay_tho = [dijkstra(W0, s, d, can_relay) for s, d, _ in qs]
            for D in deltas:
                if t + D >= T:
                    continue
                win = Wn[t + 1:t + 1 + D]
                # tuyến lập tại t nên mọi liên kết phải có tại t và sống suốt cửa sổ
                Wm = np.where((Wn[t] > 0) & (win > 0).all(0), win.mean(0), 0.0)
                off = np.zeros(len(W0), bool) if tat is None else tat[t - a]
                Wq = du_bao_quy_dao(Wg, W0, t - a, t, D, off)
                o = out[(a, D)]
                for (s, d, nhan), rn in zip(qs, ngay_tho):
                    rh = dijkstra(Wm, s, d, can_relay)
                    o["co_ngay_tho"].append(rn is not None)
                    o["co_hau_kiem"].append(rh is not None)
                    if rn is None or rh is None:
                        continue
                    rq = dijkstra(Wq, s, d, can_relay) or rn
                    ch, _ = chi_phi(win, rh, D_hong)
                    for ten, r in (("ngay_tho", rn), ("quy_dao", rq)):
                        c, dut = chi_phi(win, r, D_hong)
                        o[f"D_{ten}"].append(c)
                        o[f"dut_{ten}"].append(dut)
                        o[f"tre_them_{ten}"].append(np.nan if dut else c / ch - 1)
                    o["D_hau_kiem"].append(ch)
                    # kiểm tra: hậu kiểm phải là cận dưới (nếu không, D_hong quá nhỏ so với trễ của tuyến còn sống)
                    o["vi_pham"].append(min(o["D_ngay_tho"][-1], o["D_quy_dao"][-1]) < ch - 1e-6)
                    o["D_hk_max"].append(float(Wm[np.array(rh[:-1]), np.array(rh[1:])].sum()))
                    o["khac_quyet_dinh"].append(rq != rh)
                    o["khac_hop_dau"].append(rq[1] != rh[1])
                    o["khac_nhan_goc"].append(rh != nhan)
    return {k: {kk: np.array(v, float) for kk, v in o.items()} for k, o in out.items()}


_G = {}


def _khoi_tao(raw, tt_path):
    d = np.load(raw, allow_pickle=False)
    tt = ThoiTiet(tt_path)
    _G["g"] = MangGoc(d["W_snapshots"], d["node_kind"], d["node_order"], tt)
    _G["tt"] = tt
    _G["can_relay"] = np.array([str(k) != "user" for k in d["node_kind"]])


def _chay(job):
    ten, muc, h0, seed, truy_van, deltas, tuois, D_hong = job
    p = ThamSo(**CAU_HINH[ten], **MUC[muc])
    Wn, qs = sinh_kich_ban(_G["g"], _G["tt"], h0, seed, p)
    kq = danh_gia(Wn, _G["g"].W, truy_van, deltas, tuois, _G["can_relay"], D_hong, tat=qs['tat'])
    return ten, muc, h0, kq, len(qs["su_co"]), qs["ti_le_dut_mua"]


def tong_hop(ds):
    """ds: list kết quả của từng kịch bản (cùng cấu hình, cùng Δ)."""
    gop = {k: np.concatenate([x[k] for x in ds]) for k in ds[0]}
    Dh = gop["D_hau_kiem"].mean()
    r = {"so_cua_so": int(len(gop["D_hau_kiem"])), "tre_hau_kiem_ms": float(Dh),
         "khong_con_duong_%": float((1 - gop["co_hau_kiem"].mean()) * 100)}
    for ten in ("ngay_tho", "quy_dao"):
        tung = [(x[f"D_{ten}"].mean() - x["D_hau_kiem"].mean()) / x["D_hau_kiem"].mean() * 100
                for x in ds if len(x["D_hau_kiem"])]
        r[f"G_{ten}_%"] = float((gop[f"D_{ten}"].mean() - Dh) / Dh * 100)
        r[f"G_{ten}_sai_so_%"] = float(1.96 * np.std(tung) / np.sqrt(len(tung)))
        r[f"dut_{ten}_%"] = float(gop[f"dut_{ten}"].mean() * 100)
        r[f"tre_them_khi_song_{ten}_%"] = float(np.nanmean(gop[f"tre_them_{ten}"]) * 100)
    r["vi_pham_can_duoi_%"] = float(gop["vi_pham"].mean() * 100)
    r["tre_hau_kiem_max_ms"] = float(gop["D_hk_max"].max())
    r["khac_quyet_dinh_%"] = float(gop["khac_quyet_dinh"].mean() * 100)
    r["khac_hop_dau_%"] = float(gop["khac_hop_dau"].mean() * 100)
    r["khac_nhan_goc_%"] = float(gop["khac_nhan_goc"].mean() * 100)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--thoi-tiet", required=True)
    ap.add_argument("--so-kich-ban", type=int, default=30)
    ap.add_argument("--deltas", default="1,3,9,15")
    ap.add_argument("--tuoi", default="0", help="độ cũ thông tin, snapshot, ví dụ 0,3,9")
    ap.add_argument("--D-hong", type=float, default=1000.0, help="phạt mỗi snapshot tuyến bị đứt (ms); phải lớn hơn trễ lớn nhất của một tuyến còn sống")
    ap.add_argument("--cau-hinh", default=",".join(CAU_HINH))
    ap.add_argument("--muc", default="thường", help="mức khắc nghiệt: " + ", ".join(MUC))
    ap.add_argument("--thang", default=None, help="chỉ lấy kịch bản bắt đầu trong các tháng này, ví dụ 5-10")
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/khoang_loi.json")
    a = ap.parse_args()
    deltas = [int(x) for x in a.deltas.split(",")]
    tuois = [int(x) for x in a.tuoi.split(",")]
    d = np.load(a.raw, allow_pickle=False)
    ids = set(json.load(open(a.split, encoding="utf-8"))["snapshot_ids"]["test"])
    T = d["W_snapshots"].shape[0]
    truy_van = []
    for sid, row in zip(d["snapshot_id"], d["y_paths_indices"]):
        p = [int(x) for x in row if x >= 0]
        if int(sid) in ids:
            truy_van.append((int(sid), p[0], p[-1], p))
    tt = ThoiTiet(a.thoi_tiet)
    rng = np.random.default_rng(a.seed)
    so_gio = int(np.ceil(T * 20 / 3600)) + 2
    h0s = rng.choice(tt.gio_bat_dau(so_gio, doc_thang(a.thang)), a.so_kich_ban)
    cfgs = a.cau_hinh.split(",")
    mucs = a.muc.split(",")
    jobs = [(c, m, int(h0), a.seed * 100000 + e, truy_van, deltas, tuois, a.D_hong)
            for e, h0 in enumerate(h0s) for m in mucs for c in cfgs]
    print(f"{len(truy_van)} truy vấn test × {a.so_kich_ban} kịch bản × {len(cfgs)} cấu hình; thời tiết: {tt.nguon}",
          flush=True)
    t0 = time.time()
    kq = defaultdict(list)
    su_co, dut_mua = defaultdict(list), defaultdict(list)
    with Pool(a.procs, initializer=_khoi_tao, initargs=(a.raw, a.thoi_tiet)) as pool:
        for i, (ten, muc, h0, r, n_sc, tl) in enumerate(pool.imap_unordered(_chay, jobs)):
            kq[(muc, ten)].append(r)
            su_co[(muc, ten)].append(n_sc)
            dut_mua[(muc, ten)].append(tl)
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(jobs)} việc, {time.time() - t0:.0f} s", flush=True)
    bang = {}
    for m in mucs:
        for ten in cfgs:
            k = f"{m} | {ten}"
            bang[k] = {"so_su_co_tb": float(np.mean(su_co[(m, ten)])),
                       "ti_le_lien_ket_dut_do_mua_nhieu_%": float(np.mean(dut_mua[(m, ten)]) * 100)}
            for tu in tuois:
                for D in deltas:
                    bang[k][f"tuoi_{tu}|delta_{D}"] = tong_hop([x[(tu, D)] for x in kq[(m, ten)]])
    out = {"thoi_tiet": tt.tom_tat(), "so_truy_van": len(truy_van), "so_kich_ban": a.so_kich_ban,
           "deltas_snapshot": deltas, "tuoi_snapshot": tuois, "thang": a.thang, "D_hong_ms": a.D_hong, "tham_so": tham_so_dict(ThamSo()), "muc": {m: MUC[m] for m in mucs},
           "ket_qua": bang, "thoi_gian_s": time.time() - t0}
    json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\nLưu {a.out} ({time.time() - t0:.0f} s)\n")
    print("G = trễ thêm so với hậu kiểm (có phạt đứt); H = G của đối chứng quy_dao = khoảng lợi cho mô hình học")
    print(f"{'mức | cấu hình':36s} {'tuổi':>4s} {'Δ':>3s} {'G ngây thơ':>11s} {'đứt NT':>7s} {'H':>7s} {'±':>5s} "
          f"{'đứt QĐ':>7s} {'trễ thêm QĐ':>11s} {'khác QĐ':>8s} {'khác hop1':>9s} {'khác nhãn':>9s}")
    for k in bang:
        for tu in tuois:
            for D in deltas:
                r = bang[k][f"tuoi_{tu}|delta_{D}"]
                print(f"{k:36s} {tu:4d} {D:3d} {r['G_ngay_tho_%']:10.2f}% {r['dut_ngay_tho_%']:6.2f}% "
                      f"{r['G_quy_dao_%']:6.2f}% {r['G_quy_dao_sai_so_%']:5.2f} {r['dut_quy_dao_%']:6.2f}% "
                      f"{r['tre_them_khi_song_quy_dao_%']:10.2f}% {r['khac_quyet_dinh_%']:7.2f}% "
                      f"{r['khac_hop_dau_%']:8.2f}% {r['khac_nhan_goc_%']:8.2f}%")


if __name__ == "__main__":
    main()
