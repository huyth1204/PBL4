"""
cham_bao.py — Thí nghiệm 5 (results/bao_v5/TIEU_CHI_5.md): chấm GNN + DM và GNN + softmax trên 16 kịch bản Yagi,
bằng chi phí THỰC trong cửa sổ Δ, kèm khoảng tin cậy bootstrap theo kịch bản.

    python -m src.ngau_nhien.cham_bao --test data/bao_v5_E1/test.npz --out results/bao_v5/cham_E1.json \
        --mo-hinh "DM=results/bao_v5/E1_dm_s0.pt:2:1,results/bao_v5/E1_dm_s1.pt:2:1" \
                  "Softmax=results/bao_v5/E1_sm_s0.pt:1:1,results/bao_v5/E1_sm_s1.pt:1:1"

Mỗi phương pháp: "tên=ck1:bước:guidance,ck2:bước:guidance,..." (các seed). Chi phí mỗi truy vấn của một phương pháp
= trung bình theo seed; truy vấn mà mô hình không tới đích tính chi phí D_hong (đứt suốt cửa sổ).
G = trung bình chi phí / trung bình chi phí hậu kiểm − 1 (cùng định nghĩa với danh_gia_mo_hinh.py).
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from src.ai.data_v2 import SagsinData
from src.ai.train_gnn import build_model, encode_pair, model_chooser, rollout
from src.ngau_nhien.khoang_loi import chi_phi

B_BOOT = 10_000


def chay_mo_hinh(te, win, spec, D_hong):
    path, n_steps, w = spec.rsplit(":", 2)
    ck = torch.load(path, weights_only=False)
    m = build_model(ck["args"], ck["f_in"])
    m.load_state_dict(ck["state"])
    m.eval()
    idx = np.arange(len(te))
    t0 = time.time()
    routes, oks = [], []
    torch.manual_seed(0)
    for s in range(0, len(idx), 400):
        part = idx[s:s + 400]
        W = te.W[te.sid[part]]
        dst = np.array([te.paths[i][-1] for i in part])
        extra = None if te.node_extra is None else te.node_extra[te.sid[part]]
        hc, hu = encode_pair(m, W, te.static_feat, dst, extra)
        rt, ok, _ = rollout(te, part, model_chooser(m, hc, hu, float(w), int(n_steps)))
        routes += rt
        oks += list(ok)
    D = np.full(len(idx), D_hong)
    dut = np.ones(len(idx), bool)
    for i in np.flatnonzero(oks):
        D[i], dut[i] = chi_phi(win[te.sid[i]], routes[i], D_hong)
    return D, dut, np.array(oks), time.time() - t0


def G(D, Dh):
    return float((D.mean() / Dh.mean() - 1) * 100)


def boot_hieu(DA, DB, Dh, kb, rng):
    """Khoảng tin cậy 95% của G_A − G_B (điểm %), bootstrap theo kịch bản (cùng mẫu kịch bản cho cả hai)."""
    cac = np.unique(kb)
    nhom = [np.flatnonzero(kb == e) for e in cac]
    sA = np.array([DA[g].sum() for g in nhom]); sB = np.array([DB[g].sum() for g in nhom]); sh = np.array([Dh[g].sum() for g in nhom])
    chon = rng.integers(0, len(cac), (B_BOOT, len(cac)))
    hieu = (sA[chon].sum(1) - sB[chon].sum(1)) / sh[chon].sum(1) * 100
    return [float(np.percentile(hieu, 2.5)), float(np.percentile(hieu, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True)
    ap.add_argument("--mo-hinh", nargs="+", default=[])
    ap.add_argument("--D-hong", type=float, default=1000.0)
    ap.add_argument("--vung-bac", default="19.5,105.5", help="vĩ độ, kinh độ tối thiểu của vùng bão Bắc Bộ (phân tích phụ)")
    ap.add_argument("--raw", default="dataset/oracle_dataset_v5.npz", help="để lấy toạ độ nút cho phân tích phụ")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(2)
    te = SagsinData(a.test)
    z = np.load(a.test)
    win, Dh, kb = z["W_cua_so"], z["D_hau_kiem"], z["kich_ban"]
    assert len(Dh) == len(te), "số truy vấn của bảng chi phí khác số đường của tập test"
    pp = {"Dijkstra trên W đo được": z["D_ngay_tho"], "Dijkstra + lịch quỹ đạo": z["D_quy_dao"]}
    dut = {"Dijkstra trên W đo được": z["dut_ngay_tho"], "Dijkstra + lịch quỹ đạo": z["dut_quy_dao"]}
    toi = {k: np.ones(len(Dh), bool) for k in pp}
    theo_seed = {}
    for spec in a.mo_hinh:
        ten, ds = spec.split("=", 1)
        Ds, duts, oks = [], [], []
        for s in ds.split(","):
            D, d_, ok, tg = chay_mo_hinh(te, win, s, a.D_hong)
            Ds.append(D); duts.append(d_); oks.append(ok)
            theo_seed.setdefault(ten, []).append({"ck": s, "G_%": G(D, Dh), "toi_dich_%": float(ok.mean() * 100),
                                                 "dut_%": float(d_.mean() * 100), "thoi_gian_s": tg})
            print(ten, theo_seed[ten][-1], flush=True)
        pp[ten], dut[ten], toi[ten] = np.mean(Ds, 0), np.mean(duts, 0), np.mean(oks, 0)
    # vùng bão Bắc Bộ: truy vấn có nguồn hoặc đích ở đó (phân tích phụ, không thuộc tiêu chí)
    la0, lo0 = (float(x) for x in a.vung_bac.split(","))
    raw = np.load(a.raw)
    assert list(raw["node_order"]) == list(z["node_order"]), "tệp --raw không cùng mạng với tập test"
    toa = raw["node_latlon_alt"][0]
    bac = (toa[:, 0] >= la0) & (toa[:, 1] >= lo0)
    vung = np.array([bac[te.paths[i][0]] or bac[te.paths[i][-1]] for i in range(len(te))])
    rng = np.random.default_rng(2026)
    kq = {"so_truy_van": int(len(Dh)), "so_kich_ban": int(len(np.unique(kb))), "phuong_phap": {}, "theo_seed": theo_seed,
          "hieu": {}}
    for ten, D in pp.items():
        kq["phuong_phap"][ten] = {"G_%": G(D, Dh), "dut_%": float(np.mean(dut[ten]) * 100),
                                  "toi_dich_%": float(np.mean(toi[ten]) * 100),
                                  "G_theo_kich_ban_%": {int(e): G(D[kb == e], Dh[kb == e]) for e in np.unique(kb)}}
    kq["vung_bac_bo"] = {"ty_le_truy_van_%": float(vung.mean() * 100),
                         "G_%": {ten: G(D[vung], Dh[vung]) for ten, D in pp.items()}}
    ten_mh = [t for t in pp if t not in ("Dijkstra trên W đo được", "Dijkstra + lịch quỹ đạo")]
    cap = [(x, y) for i, x in enumerate(ten_mh) for y in ten_mh[i + 1:]] + [(x, "Dijkstra + lịch quỹ đạo") for x in ten_mh]
    for x, y in cap:
        d = G(pp[x], Dh) - G(pp[y], Dh)
        kq["hieu"][f"{x} − {y}"] = {"diem_%": d, "ktc95": boot_hieu(pp[x], pp[y], Dh, kb, rng),
                                   "vung_bac_bo_diem_%": G(pp[x][vung], Dh[vung]) - G(pp[y][vung], Dh[vung]),
                                   "vung_bac_bo_ktc95": boot_hieu(pp[x][vung], pp[y][vung], Dh[vung], kb[vung], rng)}
    json.dump(kq, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n{'phương pháp':32s} {'G':>9s} {'đứt':>7s} {'tới đích':>9s}")
    for ten, r in kq["phuong_phap"].items():
        print(f"{ten:32s} {r['G_%']:8.2f}% {r['dut_%']:6.2f}% {r['toi_dich_%']:8.1f}%")
    for k, v in kq["hieu"].items():
        print(f"{k:50s} {v['diem_%']:+8.2f} điểm %  KTC95 [{v['ktc95'][0]:+.2f}; {v['ktc95'][1]:+.2f}]")


if __name__ == "__main__":
    main()
