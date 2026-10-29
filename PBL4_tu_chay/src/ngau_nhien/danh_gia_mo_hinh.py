"""
danh_gia_mo_hinh.py — Chấm mô hình trên tập test ngẫu nhiên (tạo bởi tao_du_lieu.py) bằng chi phí THỰC trong cửa sổ Δ.

    python -m src.ngau_nhien.danh_gia_mo_hinh --test data/ngau_nhien_d9/test.npz \
        --mo-hinh "DM mới (nhãn hậu kiểm)=results/dm_hk.pt:2:1" "Softmax (nhãn hậu kiểm)=results/sm_hk.pt:1:1" \
        --out results/danh_gia_ngau_nhien.json
Mỗi mô hình: "tên=đường_dẫn.pt:số_bước_khử_nhiễu:guidance".
"""

from __future__ import annotations

import argparse
import copy
import json
import time

import numpy as np
import torch

from src.ai.data_v2 import SagsinData
from src.ai.train_gnn import build_model, encode_pair, model_chooser, rollout
from src.ngau_nhien.khoang_loi import chi_phi


def tom_tat(D, Dh, dut, toi, trung_hk, kb=None):
    """D, Dh, dut chỉ trên các truy vấn tới đích; toi, trung_hk trên toàn bộ; kb: kịch bản của từng truy vấn tới đích."""
    tt = np.where(dut, np.nan, D / Dh - 1)
    theo_kb = {} if kb is None else {int(e): float((D[kb == e].mean() / Dh[kb == e].mean() - 1) * 100)
                                     for e in np.unique(kb)}
    return {"G_theo_kich_ban_%": theo_kb,
            "toi_dich_%": float(np.mean(toi) * 100),
            "G_%": float((D.mean() / Dh.mean() - 1) * 100),
            "dut_%": float(np.mean(dut) * 100),
            "tre_them_khi_song_%": float(np.nanmean(tt) * 100),
            "trung_hau_kiem_%": float(np.mean(trung_hk) * 100)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True)
    ap.add_argument("--mo-hinh", nargs="+", default=[])
    ap.add_argument("--D-hong", type=float, default=1000.0, help="phạt mỗi snapshot tuyến bị đứt (ms); phải lớn hơn trễ lớn nhất của một tuyến còn sống")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(2)
    te = SagsinData(a.test)
    z = np.load(a.test)
    win, Dh = z["W_cua_so"], z["D_hau_kiem"]
    kb = z["kich_ban"] if "kich_ban" in z.files else None
    kq = {}
    ten_dc = {"ngay_tho": "Dijkstra trên W đo được", "quy_dao": "Dijkstra + lịch quỹ đạo (đối chứng mạnh)",
              "nhan": "Nhãn (hậu kiểm trong phần mô hình thấy)"}
    for ten, hien in ten_dc.items():
        D = z[f"D_{ten}"]
        kq[hien] = tom_tat(D, Dh, z[f"dut_{ten}"], np.ones(len(D), bool), np.full(len(D), np.nan), kb)
    te_cu = copy.copy(te)                      # cho mô hình cũ: chỉ thấy W đo được, không có đặc trưng thêm
    te_cu.W, te_cu.node_extra = z["W_quan_sat"].astype(np.float32), None
    idx = np.arange(len(te))
    for spec in a.mo_hinh:
        ten, rest = spec.split("=", 1)
        path, n_steps, w = rest.rsplit(":", 2)
        ck = torch.load(path, weights_only=False)
        m = build_model(ck["args"], ck["f_in"])
        m.load_state_dict(ck["state"])
        m.eval()
        d = te if ck["f_in"] == te.f_in else te_cu
        t0 = time.time()
        routes, oks = [], []
        torch.manual_seed(0)
        for s in range(0, len(idx), 400):
            part = idx[s:s + 400]
            W = d.W[d.sid[part]]
            dst = np.array([d.paths[i][-1] for i in part])
            extra = None if d.node_extra is None else d.node_extra[d.sid[part]]
            hc, hu = encode_pair(m, W, d.static_feat, dst, extra)
            rt, ok, _ = rollout(d, part, model_chooser(m, hc, hu, float(w), int(n_steps)))
            routes += rt
            oks += list(ok)
        oks = np.array(oks)
        D, dut = [], []
        for i in np.flatnonzero(oks):
            c, d_ = chi_phi(win[te.sid[i]], routes[i], a.D_hong)
            D.append(c)
            dut.append(d_)
        r = tom_tat(np.array(D), Dh[oks], np.array(dut), oks,
                    np.array([o and list(r) == te.paths[i] for i, (r, o) in enumerate(zip(routes, oks))]),
                    None if kb is None else kb[oks])
        r["thoi_gian_s"] = time.time() - t0
        kq[ten] = r
        print(ten, r, flush=True)
    json.dump(kq, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n{'phương pháp':50s} {'tới đích':>9s} {'G':>8s} {'đứt':>7s} {'trễ thêm khi sống':>18s} {'trùng HK':>9s}")
    for ten, r in kq.items():
        print(f"{ten:50s} {r['toi_dich_%']:8.1f}% {r['G_%']:7.2f}% {r['dut_%']:6.2f}% "
              f"{r['tre_them_khi_song_%']:17.2f}% {r['trung_hau_kiem_%']:8.1f}%")


if __name__ == "__main__":
    main()
