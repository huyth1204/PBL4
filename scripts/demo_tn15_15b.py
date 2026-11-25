"""
demo_tn15_15b.py — Dựng trang demo TN15 → TN15b (một tệp HTML chạy offline):
    1. Quá trình học     — đường học của 18 mô hình (x10, 7/3 cùng đường ống, x3, w0,5, w0,9) từ *_nhat_ky.json
    2. Kết quả chấm      — hành trình tinh chỉnh, bản đồ các J, giả thuyết đăng ký trước, kết luận TN15 → TN15b, cơ chế
    3. Mô hình chọn tuyến — DM 7/3 và DM x10 chia đôi màn hình, chọn điểm đầu / điểm cuối (dữ liệu từ demo_tn15_tuyen.py)
    4. Cấu trúc mạng     — sơ đồ MangDT với tham số quan trọng (đếm từ checkpoint)

    python -m scripts.demo_tn15_tuyen     # (một lần, ~5–10 phút) tính trước tuyến của hai mô hình
    python -m scripts.demo_tn15_15b       # ghép → results/bao_da_tuyen/demo_tn15_15b/demo_tn15_15b.html

Nguồn: tn15_may_nhom/ (TN15), tn15b_may_nhom/ (TN15b), tn13_may_nhom/ (DM w0,5 / w0,9), danh_gia_tn15/ (đánh giá TN15).
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

GOC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GOC))
KQ = GOC / "results" / "bao_da_tuyen"
RA = KQ / "demo_tn15_15b"
MAU = GOC / "scripts" / "demo_tn15_15b_mau.html"
T15, T15B, T13 = KQ / "tn15_may_nhom", KQ / "tn15b_may_nhom", KQ / "tn13_may_nhom"
MAT, A_, B_, TRE = 0, 2, 3, 4


def _r(x, n=3):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), n)


# ============================================================================================ 1. đường học
def duong_hoc(tb_seed):
    nguon = [("x10", s, T15 / "mo_hinh13") for s in range(3)] + [("x10", s, T15B / "mo_hinh13") for s in (3, 4)] + \
            [("w07", s, T15B / "mo_hinh13") for s in range(5)] + [("x3", s, T15B / "mo_hinh13") for s in range(3)] + \
            [("w05", s, T13 / "mo_hinh13") for s in range(3)] + [("w09", s, T13 / "mo_hinh13") for s in range(3)]
    ten_tep = {"x10": "dm13_x10_s{}", "w07": "dm13_w0.7_s{}", "x3": "dm13_x3_s{}", "w05": "dm13_w0.5_s{}", "w09": "dm13_w0.9_s{}"}
    ra = []
    for k, s, d in nguon:
        t = d / (ten_tep[k].format(s) + "_nhat_ky.json")
        if not t.exists():
            continue
        nk = json.loads(t.read_text(encoding="utf-8"))["nhat_ky"]
        tot = min(nk, key=lambda r: r["J_val"])
        ra.append({"mh": k, "seed": s, "tn": "TN15" if (k == "x10" and s < 3) else ("TN13" if k in ("w05", "w09") else "TN15b"),
                   "epoch": [r["epoch"] for r in nk], "loss_train": [_r(r["loss_train"], 4) for r in nk],
                   "loss_val": [_r(r["loss_val"], 4) for r in nk], "J_val": [_r(r["J_val"], 4) for r in nk],
                   "J_thay": _r(nk[0]["J_thay_val"], 4), "tot": tot["epoch"], "so_epoch": len(nk),
                   "J_tot": _r(tot["J_val"], 4), "ti_le": _r(tot["loss_val"] / tot["loss_train"], 3),
                   "mat_moi": tb_seed.get(k, {}).get(s, {}).get("mat"), "duoi_moi": tb_seed.get(k, {}).get(s, {}).get("duoi")})
    return ra


# ============================================================================================ 2. kết quả chấm
def nap_cham(nhom):
    T = []
    for f in sorted(glob.glob(str(T15B / "tn15b" / "cham_tep" / f"{nhom}__*.json"))):
        T += json.loads(Path(f).read_text(encoding="utf-8"))
    return T


def theo_seed(T, k):
    ks = [i for i in range(10) if f"{k}_s{i}" in T[0]]
    if not ks:
        X = np.array([x[k] for x in T], float)
        return {"tb": {"mat": _r(X[:, MAT].mean()), "duoi": _r((X[:, TRE] > 100).mean() * 100),
                       "tre": _r(X[:, TRE].mean(), 2), "a": _r(X[:, A_].mean()), "b": _r(X[:, B_].mean())}, "seed": {}}
    sd = {}
    for i in ks:
        X = np.array([x[f"{k}_s{i}"] for x in T], float)
        sd[i] = {"mat": _r(X[:, MAT].mean()), "duoi": _r((X[:, TRE] > 100).mean() * 100), "tre": _r(X[:, TRE].mean(), 2),
                 "a": _r(X[:, A_].mean()), "b": _r(X[:, B_].mean())}
    tb = {m: _r(np.mean([v[m] for v in sd.values()]), 3) for m in ("mat", "duoi", "tre", "a", "b")}
    return {"tb": tb, "seed": sd}


def ket_qua():
    TA, TB = nap_cham("A"), nap_cham("B")
    kq = json.loads((T15B / "tn15b" / "ket_qua_15b.json").read_text(encoding="utf-8"))
    A = {k: theo_seed(TA, k) for k in ("dm", "dm_ktn", "w07", "w07_ktn", "x10", "x10_ktn", "x10_nhanh", "x3", "w05", "w09",
                                       "thay_73", "thay_x10", "dp")}
    B = {k: theo_seed(TB, k) for k in ("dm", "w07", "x10_ktn", "de_xuat")}
    gon = lambda d: {"diem": _r(d["diem"]), "ci": [_r(d["ci"][0]), _r(d["ci"][1])]}
    P = {}
    for k, v in kq["chinh"].items():
        x = {"p_holm": _r(v["p_holm"], 6), "dat": v["dat"]}
        if k in ("P7", "P8"):
            x.update(mat=gon(v["mat"]), duoi=gon(v["duoi"]))
        elif k == "P5":
            x.update(diem=_r(v["he_so_log_X"]), ci=[_r(v["ci"][0]), _r(v["ci"][1])])
        else:
            x.update(gon(v))
        P[k] = x
    S = kq["phu"]
    phu = {"S0_mat": gon(S["S0"]["mat"]), "S0_duoi": gon(S["S0"]["duoi"]), "S1_tre": gon(S["S1"]["tre_tb"]),
           "S1_d50": gon(S["S1"]["duoi_50"]), "S1_d200": gon(S["S1"]["duoi_200"]),
           "S1_trung_vi": S["S1"]["tre_trung_vi"], "S6_tre": gon(S["S6"]["tre_dm_x10_tru_thay_x10"]),
           "S6_mat": gon(S["S6"]["mat_dm_x10_tru_thay_x10"]), "S6_thay_dp": gon(S["S6"]["tre_thay_x10_tru_dp"]),
           "S6_dm_dp": gon(S["S6"]["tre_dm_x10_tru_dp"]), "S7_va": gon(S["S7"]["va"]), "S8": gon(S["S8"]["chon_tru_tb"]),
           "S8_can": _r(S["S8"]["can_tren_biet_truoc"]), "S9": gon(S["S9"]), "S11_duoi": gon(S["S11"]["duoi"]),
           "S11_mat": gon(S["S11"]["mat"]), "S11_ktn": gon(S["S11"]["ktn_x10"]), "S14_mat": gon(S["S14"]["mat"]),
           "S15_mat": gon(S["S15"]["mat_x3_tru_x10"]), "S15_duoi": gon(S["S15"]["duoi_x3_tru_x10"]),
           "S15_tre": gon(S["S15"]["tre_x3_tru_x10"]),
           "S3": {k: {"ti_le": _r(v["ti_le_%"], 1), **gon(v["mat"])} for k, v in S["S3"]["nhom"].items()}}
    co_che_15 = {}
    pt = KQ / "danh_gia_tn15" / "phan_tich.json"
    if pt.exists():
        d = json.loads(pt.read_text(encoding="utf-8"))["co_che_theo_X_eff"]
        co_che_15 = {k: {"ti_le": _r(v["ti_le_%"], 1), "diem": _r(v["mat x10-dm"][0]),
                         "ci": [_r(v["mat x10-dm"][1]), _r(v["mat x10-dm"][2])]} for k, v in d.items()}
    tb15 = json.loads((T15 / "tn15" / "ket_qua_15.json").read_text(encoding="utf-8"))
    return {"A": A, "B": B, "P": P, "P0": {"mat": gon(kq["P0"]["mat"]), "duoi": gon(kq["P0"]["duoi"])}, "phu": phu,
            "co_che_15": co_che_15, "so_tinh_huong": kq["du_lieu"], "dong_bang": kq["dong_bang"],
            "tn15_gop": tb15.get("tong", {})}


# ============================================================================================ 4. cấu trúc mạng
def cau_truc():
    import torch
    d = torch.load(T15B / "mo_hinh13" / "dm13_x10_s3.pt", map_location="cpu", weights_only=False)
    dem = {}
    for k, v in d["state"].items():
        p = k.split(".")
        g = p[0] + ("." + p[1] if p[0] in ("gui", "cap_lk", "cap_uv") else "")
        dem[g] = dem.get(g, 0) + int(v.numel())
    return {"cau_hinh": d["cau_hinh"], "tham_so": dem, "tong": int(sum(dem.values()))}


def main():
    tb_seed = {}
    TA = nap_cham("A")
    for k in ("x10", "w07", "x3", "w05", "w09"):
        tb_seed[k] = {int(i): v for i, v in theo_seed(TA, k)["seed"].items()}
    DL = {"hoc": duong_hoc(tb_seed), "kq": ket_qua(), "mang": cau_truc()}
    tt = RA / "tuyen.json"
    DL["tuyen"] = json.loads(tt.read_text(encoding="utf-8")) if tt.exists() else None
    RA.mkdir(parents=True, exist_ok=True)
    html = MAU.read_text(encoding="utf-8").replace("/*__DU_LIEU__*/", "const DL = " + json.dumps(DL, ensure_ascii=False) + ";")
    ra = RA / "demo_tn15_15b.html"
    ra.write_text(html, encoding="utf-8")
    print(f"đã ghi {ra} ({ra.stat().st_size / 1e6:.2f} MB); tuyến: {'có' if DL['tuyen'] else 'CHƯA có — chạy demo_tn15_tuyen'}")


if __name__ == "__main__":
    main()
