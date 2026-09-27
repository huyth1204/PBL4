"""
tao_du_lieu.py — Tạo train/val/test.npz cho bài toán mới: chọn tuyến bền trong cửa sổ Δ ở thế giới ngẫu nhiên.

Đầu vào mô hình (W_snapshots, cùng định dạng cũ nên train_gnn.py dùng lại được) là một trong hai:
  quan_sat  W đo được lúc t − tuổi
  quy_dao   W dự báo = trung bình W_gốc (lịch quỹ đạo) trên cửa sổ × (W đo / W_gốc lúc đo) — cùng thông tin với
            đối chứng cổ điển mạnh nhất
cộng 8 đặc trưng nút (node_extra): tải, mưa, giông, gió giật, gây nhiễu, nút bị tắt, giờ địa phương.
Nhãn: tuyến hậu kiểm = Dijkstra(trung bình W trên t+1..t+Δ, chỉ liên kết có tại t, sống suốt cửa sổ, mô hình thấy được).
Tập test lưu thêm cửa sổ tương lai và chi phí của các đối chứng để chấm điểm mô hình (danh_gia_mo_hinh.py).

    python -m src.ngau_nhien.tao_du_lieu --raw dataset/oracle_dataset_v2.npz --split dataset/split_info.json \
        --thoi-tiet data/thoi_tiet_vn.npz --delta 9 --kich-ban 2,1,2 --out-dir data/ngau_nhien_d9
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.ngau_nhien.khoang_loi import MUC, chi_phi, dijkstra
from src.ngau_nhien.kich_ban import MangGoc, ThamSo, sinh_kich_ban
from src.ngau_nhien.thoi_tiet import ThoiTiet, doc_thang


def dac_trung_nut(g, q, t):
    """8 đặc trưng nút tại t: tải, mưa, giông, gió giật, gây nhiễu, bị tắt, giờ địa phương (sin, cos)."""
    N = g.N
    X = np.zeros((N, 8), np.float32)
    co = g.node_site >= 0
    s = g.node_site[co]
    X[:, 0] = q["tai"][t] / 2
    X[co, 1] = q["R"][t, s] / 20
    X[co, 2] = q["giong"][t, s]
    X[co, 3] = q["gust"][t, s] / 20
    X[co, 4] = q["J"][t, s] / 20
    X[:, 5] = q["tat"][t]
    X[:, 6] = np.sin(2 * np.pi * q["gio_vn"][t] / 24)
    X[:, 7] = np.cos(2 * np.pi * q["gio_vn"][t] / 24)
    return X


def tao_tap(g, tt, raw, ids, n_kb, delta, tuoi, dau_vao, muc, seed0, rng, luu_cua_so, D_hong, thang=None):
    can_relay = np.array([k != "user" for k in g.kind])
    cap = defaultdict(list)
    for sid, row in zip(raw["snapshot_id"], raw["y_paths_indices"]):
        if int(sid) in ids:
            p = [int(x) for x in row if x >= 0]
            cap[int(sid)].append((p[0], p[-1]))
    W_list, X_list, paths, sids, win_list, obs_list, bang = [], [], [], [], [], [], defaultdict(list)
    so_gio = int(np.ceil(g.T * 20 / 3600)) + 2
    for e in range(n_kb):
        h0 = int(rng.choice(tt.gio_bat_dau(so_gio, thang)))
        Wn, q = sinh_kich_ban(g, tt, h0, seed0 + e, ThamSo(**MUC[muc]))
        for t in sorted(ids):
            if t + delta >= g.T or t - tuoi < 0:
                continue
            Wo, Go = Wn[t - tuoi], g.W[t - tuoi]                 # thứ bộ định tuyến thấy
            win = Wn[t + 1:t + 1 + delta]
            song = (Wn[t] > 0) & (win > 0).all(0)
            Wm = np.where(song, win.mean(0), 0.0)                # hậu kiểm (tham chiếu)
            wg = g.W[t + 1:t + 1 + delta]
            with np.errstate(divide="ignore", invalid="ignore"):
                he_so = np.where(Go > 0, np.where(Wo > 0, Wo / Go, 0.0), 1.0)
            Wq = np.where((g.W[t] > 0) & (wg > 0).all(0), wg.mean(0), 0.0) * he_so
            W_in = Wq if dau_vao == "quy_dao" else Wo
            Wl = np.where(W_in > 0, Wm, 0.0)                     # nhãn: hậu kiểm trong các liên kết mô hình thấy
            li = len(W_list)
            dung = False
            for s, d in cap[t]:
                rl = dijkstra(Wl, s, d, can_relay)
                rn = dijkstra(Wo, s, d, can_relay)
                if rl is None or rn is None:
                    continue
                dung = True
                paths.append(rl)
                sids.append(li)
                if luu_cua_so:
                    rh = dijkstra(Wm, s, d, can_relay)
                    rq = dijkstra(Wq, s, d, can_relay) or rn
                    for ten, r in (("hau_kiem", rh), ("ngay_tho", rn), ("quy_dao", rq), ("nhan", rl)):
                        c, dut = chi_phi(win, r, D_hong)
                        bang[f"D_{ten}"].append(c)
                        bang[f"dut_{ten}"].append(dut)
                    bang["kich_ban"].append(e)
            if dung:
                W_list.append(np.where(W_in > 0, W_in, -1.0).astype(np.float32))
                X_list.append(dac_trung_nut(g, q, t - tuoi))
                if luu_cua_so:
                    win_list.append(win.astype(np.float32))
                    obs_list.append(Wo.astype(np.float32))
    L = max(len(p) for p in paths)
    Y = -np.ones((len(paths), max(L, 9)), np.int32)
    for i, p in enumerate(paths):
        Y[i, :len(p)] = p
    out = {"W_snapshots": np.stack(W_list), "node_extra": np.stack(X_list), "y_paths_indices": Y,
           "snapshot_id": np.array(sids, np.int32)}
    if luu_cua_so:
        out["W_cua_so"] = np.stack(win_list)
        out["W_quan_sat"] = np.stack(obs_list)
        out.update({k: np.array(v) for k, v in bang.items()})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--thoi-tiet", required=True)
    ap.add_argument("--delta", type=int, default=9)
    ap.add_argument("--muc", default="thường")
    ap.add_argument("--tuoi", type=int, default=0, help="độ cũ thông tin (snapshot)")
    ap.add_argument("--dau-vao", choices=["quan_sat", "quy_dao"], default="quan_sat",
                    help="quan_sat: W đo được; quy_dao: W dự báo từ lịch quỹ đạo × trạng thái ngẫu nhiên hiện tại")
    ap.add_argument("--kich-ban", default="2,1,2", help="số kịch bản cho train,val,test")
    ap.add_argument("--D-hong", type=float, default=1000.0, help="phạt mỗi snapshot tuyến bị đứt (ms); phải lớn hơn trễ lớn nhất của một tuyến còn sống")
    ap.add_argument("--thang", default=None, help="chỉ lấy kịch bản bắt đầu trong các tháng này, ví dụ 5-10")
    ap.add_argument("--luu-val", action="store_true", help="lưu cả cửa sổ tương lai cho val (để chọn guidance theo chi phí)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    raw = np.load(a.raw, allow_pickle=False)
    tt = ThoiTiet(a.thoi_tiet)
    g = MangGoc(raw["W_snapshots"], raw["node_kind"], raw["node_order"], tt)
    split = json.load(open(a.split, encoding="utf-8"))["snapshot_ids"]
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    meta = {"delta": a.delta, "tuoi": a.tuoi, "dau_vao": a.dau_vao, "muc": a.muc, "thoi_tiet": tt.tom_tat(),
            "D_hong": a.D_hong, "thang": a.thang}
    for i, (ten, k) in enumerate(zip(("train", "val", "test"), map(int, a.kich_ban.split(",")))):
        ds = tao_tap(g, tt, raw, set(split[ten]), k, a.delta, a.tuoi, a.dau_vao, a.muc,
                     a.seed * 1000 + i * 100, rng, ten == "test" or (ten == "val" and a.luu_val), a.D_hong,
                     doc_thang(a.thang))
        ds.update(node_order=raw["node_order"], node_layer=raw["node_layer"], node_kind=raw["node_kind"])
        np.savez_compressed(out / f"{ten}.npz", **ds)
        meta[ten] = {"so_duong": int(len(ds["snapshot_id"])), "so_snapshot": int(len(ds["W_snapshots"]))}
        if ten == "test":
            for x in ("ngay_tho", "quy_dao", "nhan"):
                meta[ten][f"G_{x}_%"] = float((ds[f"D_{x}"].mean() / ds["D_hau_kiem"].mean() - 1) * 100)
                meta[ten][f"dut_{x}_%"] = float(ds[f"dut_{x}"].mean() * 100)
        print(ten, meta[ten], flush=True)
    json.dump(meta, open(out / "meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
