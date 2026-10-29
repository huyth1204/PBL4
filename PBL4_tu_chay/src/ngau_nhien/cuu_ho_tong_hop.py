"""Tổng hợp TN6+4: tỉ lệ phục vụ theo cách đặt UAV × router, tách loại truy vấn, KTC bootstrap theo kịch bản.

    python -m src.ngau_nhien.cuu_ho_tong_hop --vao results/cuu_ho/yagi
"""

import argparse
import json
from pathlib import Path

import numpy as np

PP = ("khong", "gan_nut_yeu", "vet_can")
RO = ("quy_dao", "score")
TC = {"C1": (("vet_can", "score"), ("khong", "quy_dao")), "C2": (("vet_can", "score"), ("khong", "score")),
      "C3": (("vet_can", "score"), ("vet_can", "quy_dao")), "C4": (("vet_can", "score"), ("gan_nut_yeu", "score"))}


def doc(vao, kind):
    z = [np.load(f) for f in sorted(Path(vao).glob("*.npz"))]
    g = {k: np.concatenate([x[k] for x in z]) for k in z[0].files if k not in ("nhat_ky", "kich_ban")}
    g["kb"] = np.concatenate([np.full(len(x["t"]), i) for i, x in enumerate(z)])
    ks, kd = kind[g["s"]], kind[g["d"]]
    g["loai"] = np.where((ks == "uav") | (kd == "uav"), "co_uav", "mat_dat")
    g["ten_kb"] = [str(x["kich_ban"]) for x in z]
    g["nhat_ky"] = [json.loads(str(x["nhat_ky"])) for x in z]
    return g


def boot(a, b, kb, B=10_000, seed=2026):
    """KTC 95% của hiệu tỉ lệ (a − b, điểm %), lấy mẫu lại theo kịch bản."""
    cac = np.unique(kb)
    sa = np.array([a[kb == e].sum() for e in cac]); sb = np.array([b[kb == e].sum() for e in cac])
    n = np.array([(kb == e).sum() for e in cac])
    ch = np.random.default_rng(seed).integers(0, len(cac), (B, len(cac)))
    h = (sa[ch].sum(1) - sb[ch].sum(1)) / n[ch].sum(1) * 100
    return [float(np.percentile(h, 2.5)), float(np.percentile(h, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vao", required=True)
    ap.add_argument("--raw", default="dataset/oracle_dataset_v5.npz")
    a = ap.parse_args()
    kind = np.array([str(k) for k in np.load(a.raw)["node_kind"]])
    g = doc(a.vao, kind)
    kq = {"so_truy_van": int(len(g["t"])), "so_kich_ban": len(g["ten_kb"]), "ty_le_bao_%": float(g["gd"].mean() * 100),
          "loai_%": {l: float((g["loai"] == l).mean() * 100) for l in ("mat_dat", "co_uav")}, "phuong_phap": {}, "tieu_chi": {}}
    for m in PP:
        kq["phuong_phap"][f"{m}|hau_kiem"] = {"tat_ca": float(g[f"{m}|hau_kiem"].mean() * 100),
                                             **{l: float(g[f"{m}|hau_kiem"][g["loai"] == l].mean() * 100) for l in ("mat_dat", "co_uav")}}
        for r in RO:
            pv = g[f"{m}|{r}|phuc_vu"]
            kq["phuong_phap"][f"{m}|{r}"] = {"tat_ca": float(pv.mean() * 100),
                                            **{l: float(pv[g["loai"] == l].mean() * 100) for l in ("mat_dat", "co_uav")},
                                            "bao": float(pv[g["gd"] == 1].mean() * 100) if (g["gd"] == 1).any() else None,
                                            "tre_ms": float(np.nanmean(g[f"{m}|{r}|tre"]))}
    for k, (x, y) in TC.items():
        A, B_ = g[f"{x[0]}|{x[1]}|phuc_vu"], g[f"{y[0]}|{y[1]}|phuc_vu"]
        md = g["loai"] == "mat_dat"
        kq["tieu_chi"][k] = {"so_sanh": f"{x[0]}+{x[1]} − {y[0]}+{y[1]}", "hieu_diem_%": float((A.mean() - B_.mean()) * 100),
                             "ktc95": boot(A, B_, g["kb"]), "dat": None,
                             "mat_dat_hieu_%": float((A[md].mean() - B_[md].mean()) * 100), "mat_dat_ktc95": boot(A[md], B_[md], g["kb"][md]),
                             "so_truy_van_khac": int((A != B_).sum())}
        kq["tieu_chi"][k]["dat"] = kq["tieu_chi"][k]["ktc95"][0] > 0
    # UAV cứu hộ đi bao xa, đứng ở đâu
    di = {m: 0 for m in ("gan_nut_yeu", "vet_can")}
    for nk in g["nhat_ky"]:
        for m in di:
            pos = [c for tau, mm, c in nk if mm == m]
            di[m] += sum(1 for a_, b_ in zip(pos[:-1], pos[1:]) if a_ != b_)
    kq["so_lan_doi_cho"] = di
    json.dump(kq, open(Path(a.vao) / "tong_hop.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{kq['so_truy_van']} truy vấn, {kq['so_kich_ban']} kịch bản, giai đoạn bão {kq['ty_le_bao_%']:.0f}%, "
          f"truy vấn có đầu UAV {kq['loai_%']['co_uav']:.0f}%")
    print(f"{'':24s} {'tất cả':>8s} {'mặt đất':>8s} {'có UAV':>8s} {'lúc bão':>8s} {'trễ ms':>7s}")
    for k, v in kq["phuong_phap"].items():
        print(f"{k:24s} {v['tat_ca']:7.2f}% {v['mat_dat']:7.2f}% {v['co_uav']:7.2f}% " +
              (f"{v['bao']:7.2f}% {v['tre_ms']:7.2f}" if "bao" in v else ""))
    for k, v in kq["tieu_chi"].items():
        print(f"{k} {v['so_sanh']:34s} {v['hieu_diem_%']:+.3f} điểm KTC [{v['ktc95'][0]:+.3f}; {v['ktc95'][1]:+.3f}] "
              f"{'ĐẠT' if v['dat'] else 'trượt'} | mặt đất {v['mat_dat_hieu_%']:+.3f} [{v['mat_dat_ktc95'][0]:+.3f}; "
              f"{v['mat_dat_ktc95'][1]:+.3f}] | {v['so_truy_van_khac']} truy vấn khác kết quả")
    print("số lần đổi chỗ:", di)


if __name__ == "__main__":
    main()
