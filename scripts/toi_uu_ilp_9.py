"""
toi_uu_ilp_9.py — Thí nghiệm 9, mục B2 (results/bao_da_tuyen/TIEU_CHI_9.md): mốc tối ưu chính xác cho mất dịch vụ.

    python -m scripts.toi_uu_ilp_9 --tien-trinh 2          # → results/bao_da_tuyen/tn9/ilp_<bộ>.json (mọi tình huống 4–5 luồng)
    python -m scripts.toi_uu_ilp_9 --thu 5                 # thử 5 tình huống, in thời gian và kiểm tra khớp bộ chấm

Với một tình huống đã lưu (bài toán gọn + 32 thế giới CHẤM thu gọn), tìm lựa chọn trong kho (đúng tập lựa chọn hợp lệ
của DM: 1 tuyến chính + m − 1 dự phòng mỗi luồng) cực tiểu mất dịch vụ trung bình có trọng λ trên ĐÚNG 32 thế giới đó.
Đây là cận dưới "biết trước 32 thế giới" — lạc quan, không phương pháp online nào đạt được trong kho này.

Quy hoạch nguyên (scipy.optimize.milp, HiGHS):
  x[k,o] ∈ {0,1}  luồng k chọn lựa chọn o;  Σ_o x[k,o] = 1
  y[k,w,r] = Σ_{o: tuyến dùng của o ở thế giới w là r} x[k,o]          (tuyến dùng = tuyến sống đầu tiên của tập)
  z[w,e] ∈ {0,1}  liên kết e quá tải ở w:  Σ_k λ_k Σ_{r ∋ e} y[k,w,r] ≤ du[w,e] − ε + M·z[w,e]   (quá tải ⇔ F ≥ du)
  l[k,w] ≥ Σ_{o: không tuyến sống} x[k,o];   l[k,w] ≥ Σ_{r ∋ e} y[k,w,r] + z[w,e] − 1   ∀ e
  min Σ_k λ_k/Σλ · Σ_w l[k,w] / W
Kết quả được chấm lại bằng chính TheGioi.cham để kiểm tra.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.optimize import Bounds, LinearConstraint, milp

GOC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GOC))

from scripts.phan_tich_5 import DAI, nap  # noqa: E402
from src.nhieu_luong.huan_luyen_dt import the_gioi_cham  # noqa: E402

RA = GOC / "results" / "bao_da_tuyen" / "tn9"
BO_ILP = ("c6", "mua5", "yagi5", "ngay9", "nam9", "bao9")
GIAY = 60.0                        # giới hạn thời gian mỗi bài (thử: < 0,5 s / bài ở 4–5 luồng → giải MỌI tình huống)


def giai(r, gioi_han=GIAY):
    """→ (lựa chọn [K], mất DV % của ILP, trạng thái, giây)."""
    gon, tg = r["gon"], the_gioi_cham(r)
    du = r["cham"]["du"].astype(float)                                    # [W, U]
    W, U, K = du.shape[0], du.shape[1], gon.K
    lam = np.asarray(gon.lam, float)
    n_x = [len(O) for O in gon.opts]
    n_r = [M.shape[0] for M in gon.inc]
    ox = np.concatenate([[0], np.cumsum(n_x)])                            # chỉ số biến x
    oy = ox[-1] + np.concatenate([[0], np.cumsum([W * n for n in n_r])])  # y[k] khối [W, n_r]
    dung = [sorted(set(np.flatnonzero(M.any(0)))) for M in gon.inc]       # cạnh luồng k có thể dùng
    e_all = sorted(set().union(*map(set, dung)))
    ie = {e: i for i, e in enumerate(e_all)}
    oz = oy[-1]
    ol = oz + W * len(e_all)
    n = ol + K * W
    hang, cot, gt, lo, hi = [], [], [], [], []

    def rang(c, v, a, b):
        i = len(lo)
        hang.extend([i] * len(c))
        cot.extend(c)
        gt.extend(v)
        lo.append(a)
        hi.append(b)

    for k in range(K):
        rang(list(range(ox[k], ox[k + 1])), [1.0] * n_x[k], 1.0, 1.0)
        O = gon.opts[k]
        al = tg.alive[k][:, O]                                            # [W, n_o, m]
        co = al.any(-1)
        dung_r = O[np.arange(len(O))[None], al.argmax(-1)]                # [W, n_o]
        for w in range(W):
            for rr in range(n_r[k]):
                os_ = np.flatnonzero(co[w] & (dung_r[w] == rr))
                rang([oy[k] + w * n_r[k] + rr] + list(ox[k] + os_), [1.0] + [-1.0] * len(os_), 0.0, 0.0)
            os_ = np.flatnonzero(~co[w])                                  # không tuyến sống → mất
            if len(os_):
                rang([ol + k * W + w] + list(ox[k] + os_), [1.0] + [-1.0] * len(os_), 0.0, np.inf)
            for e in dung[k]:
                rs = np.flatnonzero(gon.inc[k][:, e])
                rang([ol + k * W + w, oz + w * len(e_all) + ie[e]] + list(oy[k] + w * n_r[k] + rs),
                     [1.0, -1.0] + [-1.0] * len(rs), -1.0, np.inf)
    co_the = {e: [k for k in range(K) if e in set(dung[k])] for e in e_all}
    for w in range(W):
        for e in e_all:
            eps = 1e-6 * max(du[w, e], 1e-9)
            tran = lam[co_the[e]].sum()
            if tran < du[w, e] - eps:                                     # không thể quá tải
                continue
            c, v = [oz + w * len(e_all) + ie[e]], [-(tran - du[w, e] + eps + 1.0)]
            for k in co_the[e]:
                rs = np.flatnonzero(gon.inc[k][:, e])
                c += list(oy[k] + w * n_r[k] + rs)
                v += [lam[k]] * len(rs)
            rang(c, v, -np.inf, du[w, e] - eps)
    A = sp.csr_matrix((gt, (hang, cot)), shape=(len(lo), n))
    cobj = np.zeros(n)
    for k in range(K):
        cobj[ol + k * W: ol + (k + 1) * W] = lam[k] / lam.sum() / W * 100
    nguyen = np.zeros(n)
    nguyen[:ox[-1]] = 1
    nguyen[oz:ol] = 1
    s = time.perf_counter()
    kq = milp(cobj, constraints=LinearConstraint(A, lo, hi), integrality=nguyen, bounds=Bounds(0, 1),
              options={"time_limit": gioi_han, "disp": False, "mip_rel_gap": 0.0})
    giay = time.perf_counter() - s
    if kq.x is None:
        return None, float("nan"), int(kq.status), giay, float("nan")
    a = np.array([int(np.argmax(kq.x[ox[k]:ox[k + 1]])) for k in range(K)])
    can = float(getattr(kq, "mip_dual_bound", np.nan))
    return a, float(kq.fun), int(kq.status), giay, can


def mau(bo):
    """Mọi tình huống 4–5 luồng của bộ, kèm dải tải."""
    return [(d, r) for r in nap(bo, "k4_5") for d, (lo, hi) in DAI.items() if lo <= r["rho_hat"] < hi]


def _mot(job):
    bo, d, r = job
    a, J, st, giay, can = giai(r)
    tg = the_gioi_cham(r)
    cham = tg.cham(r["gon"].tu_lua_chon(a))["mat_dv_%"] if a is not None else float("nan")
    lai6 = r["ens"]["dm6" if r["rho_hat"] < 1.1 else "dm6_tkcb"]["mat_dv_%"]
    het = np.array([(~al.any(1)).mean() for al in tg.alive])
    lam = np.asarray(r["gon"].lam, float)
    return {"bo": bo, "dai": d, "kb": int(r["kb"]), "t": int(r["t"]), "ilp_%": J, "ilp_cham_%": cham, "trang_thai": st,
            "giay": giay, "can_duoi_%": can, "san_a_%": float((het * lam).sum() / lam.sum() * 100),
            "dphat_%": r["ens"]["dphat"]["mat_dv_%"], "dm6_%": r["ens"]["dm6"]["mat_dv_%"], "lai6_%": lai6,
            "thay_%": r["ens"]["thay"]["mat_dv_%"] if "thay" in r["ens"] else None,
            "online_tkcb_%": r["ens"]["online_tkcb"]["mat_dv_%"]}


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--tien-trinh", type=int, default=2)
    ap.add_argument("--thu", type=int, default=0, help="chỉ thử n tình huống đầu của c6")
    ap.add_argument("--bo", nargs="*", default=list(BO_ILP))
    a = ap.parse_args()
    if a.thu:
        ds = mau("c6")
        for d, r in ds[::max(1, len(ds) // a.thu)][:a.thu]:
            x = _mot(("c6", d, r))
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in x.items()}, flush=True)
        return
    RA.mkdir(parents=True, exist_ok=True)
    for bo in a.bo:                                   # một tệp mỗi bộ, bỏ qua bộ đã có
        tep = RA / f"ilp_{bo}.json"
        if tep.exists():
            print(f"{bo}: đã có", flush=True)
            continue
        jobs = [(bo, d, r) for d, r in mau(bo)]
        print(f"ILP {bo}: {len(jobs)} tình huống", flush=True)
        ra, t0 = [], time.time()
        with Pool(a.tien_trinh) as pool:
            for i, x in enumerate(pool.imap_unordered(_mot, jobs, chunksize=1)):
                ra.append(x)
                if x["giay"] > 30:
                    print(f"  chậm: cửa sổ {x['kb']} mốc {x['t']} {x['dai']} {x['giay']:.0f} s trạng thái {x['trang_thai']}",
                          flush=True)
                if (i + 1) % 200 == 0:
                    print(f"  {i + 1}/{len(jobs)}  ({time.time() - t0:.0f} s)", flush=True)
        tam = str(tep) + ".tam"
        json.dump(ra, open(tam, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
        Path(tam).replace(tep)
        print(f"xong {bo} ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
