"""
tao_du_lieu_bao.py — Sinh `.pkl` (nhãn thầy + phương pháp) cho MỘT cơn bão thật.

Dùng cùng quy trình với `src/nhieu_luong/tao_du_lieu_dt.py`, nhưng thay nguồn
thời tiết bằng tệp bão `data/bao_manh/<tên>.npz`, và lấy cửa sổ theo chính cơn bão
(mỗi cửa sổ là một đoạn giờ trong cơn).

Kết quả: `data/bao_da_tuyen/<tên>_k4_5/*.pkl` — cùng lược đồ với tập TN6, nên
`cham_lai_dt` và `mo_hinh_dt` dùng lại được mà không sửa gì.

    python tao_du_lieu_bao.py --bao soulik
    python tao_du_lieu_bao.py --bao soulik --so-cua-so 5     # thử nhanh
    python tao_du_lieu_bao.py --liet-ke
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

GOC = Path(r"E:\PBL4")
if str(GOC) not in sys.path:
    sys.path.insert(0, str(GOC))

RA = GOC / "data" / "bao_da_tuyen"
BAO = GOC / "data" / "bao_manh"
SO_CUA_SO = 13          # như tập C của TN6 (13 ngày giông)
DAI_CUA_SO = 48         # số mốc giờ mỗi cửa sổ (như tập test)

# --- tham chiếu module sinh dữ liệu của dự án
from src.nhieu_luong import tao_du_lieu_rr as td          # noqa: E402
from src.nhieu_luong import tao_du_lieu_dt as tdt          # noqa: E402
from src.nhieu_luong import da_tuyen as dt                # noqa: E402
from src.nhieu_luong import the_gioi_rr as tg             # noqa: E402
from src.ngau_nhien.ma_tran_rui_ro import ma_tran_online  # noqa: E402
from src.nhieu_luong.tao_du_lieu_dt import (              # noqa: E402
    BETA_GAMMA, MUC_TAI, SO_TG_CHAM, SO_TG_NHAN, SO_DO,
    _seeds, anh_xa, cape_luc, dac_trung_canh, dac_trung_uv, thay,
)
from src.ngau_nhien.kich_ban import sinh_kich_ban         # noqa: E402
from src.ngau_nhien.thoi_tiet import ThoiTiet             # noqa: E402

TUOI, DELTA = td.TUOI, td.DELTA


def danh_muc_bao() -> list[dict]:
    cat = json.loads((BAO / "danh_muc.json").read_text(encoding="utf-8"))
    return cat["bao"]


def tim_bao(ten: str) -> dict:
    for b in danh_muc_bao():
        if b["ten"] == ten or b["ten"].startswith(ten):
            return b
    ten_co = ", ".join(b["ten"] for b in danh_muc_bao())
    raise SystemExit(f"Không có bão '{ten}'. Có: {ten_co}")


def cua_so_bao(b: dict, so: int, dai: int) -> list[tuple[str, int, int]]:
    """[(tên cửa sổ, h0, seed)] — mỗi cửa sổ là một đoạn `dai` giờ trong cơn bão.

    Cơn bão có N giờ; rải `so` cửa sổ đều nhau để phủ toàn bộ cơn (đầu, đỉnh, tan).
    """
    d = np.load(BAO / b["tep"], allow_pickle=False)
    so_gio = len(d["time_utc"])
    dai = min(dai, so_gio)
    if so <= 1:
        moc = [0]
    else:
        moc = np.linspace(0, so_gio - dai, so).round().astype(int).tolist()
    return [(f"{b['ten']}_c{q:02d}", int(h), 800000 + q) for q, h in enumerate(moc)]


def _mot_cua_so(job):
    """Sinh một cửa sổ: giống `tao_du_lieu_dt._mot_cua_so` nhưng dùng thời tiết bão."""
    ten_tep, h0, seed, moc, ten_bao, K_rg, i = job
    if Path(ten_tep).exists():
        return ten_tep, 0, 0.0

    if td._g is None:
        td._khoi_tao()

    g, p = td._g, td._p
    tt = ThoiTiet(str(BAO / tim_bao(ten_bao)["tep"]))
    t_bd = time.time()
    Wn, q = sinh_kich_ban(g, tt, h0, seed, p)
    pl = tg.phat_lai(g, tt, h0, seed, p)
    rng = np.random.default_rng([seed, K_rg[1]])
    ban_ghi = []

    for so, t in enumerate(moc):
        t = int(t)
        o = t - TUOI
        if o < 0:
            continue
        past = {k: q[k][:o + 1] for k in ("R", "gust", "giong", "tat")}
        r = ma_tran_online(td._bang, g, Wn[o], past, t, TUOI, DELTA, tham_so=p)
        th = dt.tao_tinh_huong(g, q, r, t, rng, so_luong=K_rg, so_nguon=tdt.so_nguon(K_rg[1]))
        if th is None:
            continue
        mg = th["mg"]
        Y = dt.yen_m(th)
        S = dt.suurballe_m(th)
        sn, sc = _seeds(seed, t, 0, SO_TG_NHAN), _seeds(seed, t, 1, SO_TG_CHAM)

        for so_muc, muc in enumerate(list(MUC_TAI)):
            lam = th["lam"] * (MUC_TAI[muc] if isinstance(muc, str) else muc) / th["u"]
            DP = dt.dijkstra_phat(th, lam, *BETA_GAMMA)
            bt = dt.BaiToan(th, muc, [Y[k] + DP[k] for k in range(len(lam))])
            gon = dt.BaiToanGon(bt)
            gon.dac_trung_canh = dac_trung_canh(g, q, mg, gon.U, o, cape_luc(h0, o, str(BAO / tim_bao(ten_bao)["tep"])))
            gon.dac_trung_uv = [dac_trung_uv(g, th, k) for k in range(bt.K)]
            En = dt.TheGioi.ensemble(bt, g, pl, p, sn)

            B = dt.TheGioi.tu_bang(bt, np.random.default_rng([seed, t, 2]))
            a_on, _ = dt.tkcb(B, dt.tham_lam(B))
            sol = {"yen": bt.loi_giai(Y), "suurballe": bt.loi_giai(S), "dphat": bt.loi_giai(DP),
                   "online_tkcb": bt.tu_lua_chon(a_on)}
            dau = [a_on] + [a for a in (anh_xa(bt, sol["suurballe"]), anh_xa(bt, sol["dphat"])) if a is not None]
            a_thay, J_thay = thay(bt, En, dau, rng)

            rec = {"gon": gon, "nhan": a_thay.astype(np.int16), "J_nhan": float(J_thay),
                   "kb": i, "bao": ten_bao, "t": t, "muc": muc, "K": bt.K,
                   "nguon": th["nguon"], "tg_ms": {}}

            # Lưu sẵn cặp nút của MỌI tuyến trong kho: gon.U (cột) -> mg.eu/ev (nút).
            # Nhờ vậy bản đồ đọc thẳng, không phải dựng lại Mang sau (nhanh + chính xác).
            # `th["kho"][th["cap"][k]]["kho"]` = danh sách tuyến, mỗi tuyến là
            # DANH SÁCH NÚT [nguồn, ..., đích]. `gon.opts[k][a]` = bộ m chỉ số trong kho.
            cap_nut = []
            for k in range(gon.K):
                kho_k = th["kho"][th["cap"][k]]["kho"]
                ds = []
                for idx in np.asarray(gon.opts[k], int):          # mỗi hàng = m chỉ số
                    ds.append([[int(n) for n in kho_k[j]] for j in idx])
                cap_nut.append(ds)
            rec["cap_nut"] = cap_nut

            # thế giới chấm + quan sát (giống tập test của TN6)
            Ec = dt.TheGioi.ensemble(bt, g, pl, p, sc)
            Q = dt.TheGioi.quan_sat(bt, Wn, q)
            sol["thay"] = bt.tu_lua_chon(a_thay)
            rec["ens"], rec["qs"] = {}, {}
            for ten_pp, x in sol.items():
                for ten_tgw, tgw in (("ens", Ec), ("qs", Q)):
                    c = tgw.cham(x)
                    rec[ten_tgw][ten_pp] = {k: c[k] for k in SO_DO}
            for ten_tgw, tgw in (("cham", Ec), ("cham_qs", Q)):
                du = np.maximum((tgw.Cw * (1 - tgw.rw)).min(1), 1e-9)
                rec[ten_tgw] = {"song": tgw.song[:, gon.cot],
                                "du": du[:, gon.cot].astype(np.float32)}
            ban_ghi.append(rec)

    tam = str(ten_tep) + ".tam"
    with open(tam, "wb") as f:
        pickle.dump(ban_ghi, f, protocol=pickle.HIGHEST_PROTOCOL)
    import os
    os.replace(tam, ten_tep)
    return ten_tep, len(ban_ghi), time.time() - t_bd


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bao", default=None, help="tên cơn bão (vd soulik, bualoi)")
    ap.add_argument("--so-cua-so", type=int, default=SO_CUA_SO)
    ap.add_argument("--dai", type=int, default=DAI_CUA_SO)
    ap.add_argument("--buoc-moc", type=int, default=1, help="lấy mỗi n mốc (thử nhanh: 12)")
    ap.add_argument("--K", default="4-5")
    ap.add_argument("--tien-trinh", type=int, default=3)
    ap.add_argument("--liet-ke", action="store_true")
    a = ap.parse_args()

    if a.liet_ke:
        print(f"{'bão':<18}{'giờ':>5}{'mưa max':>9}{'gió max':>9}")
        for b in danh_muc_bao():
            d = np.load(BAO / b["tep"], allow_pickle=False)
            print(f"{b['ten']:<18}{len(d['time_utc']):>5}{d['precip'].max():>9.1f}"
                  f"{d['gust'].max():>9.1f}")
        return

    if not a.bao:
        raise SystemExit("Cần --bao <tên> (xem --liet-ke)")

    b = tim_bao(a.bao)
    td._khoi_tao_nhe()
    K_rg = tdt.phan_tich_K(a.K)
    moc = td.cac_moc("test")[::a.buoc_moc]
    ds = cua_so_bao(b, a.so_cua_so, a.dai)
    thu_muc = RA / f"{b['ten']}_k{K_rg[0]}" if K_rg[0] == K_rg[1] else RA / f"{b['ten']}_k{K_rg[0]}_{K_rg[1]}"
    thu_muc.mkdir(parents=True, exist_ok=True)

    print(f"bão {b['ten']}: {len(ds)} cửa sổ × {len(moc)} mốc → {thu_muc}")
    jobs = [(str(thu_muc / f"kb_{i:04d}.pkl"), h, sd, moc, b["ten"], K_rg, i)
            for i, (ten, h, sd) in enumerate(ds)]
    with Pool(a.tien_trinh, initializer=td._khoi_tao) as pool:
        for q, (tep, n, giay) in enumerate(pool.imap_unordered(_mot_cua_so, jobs), 1):
            print(f"  {q}/{len(jobs)} {Path(tep).name}: {n} bản ghi ({giay:.0f}s)", flush=True)
    (thu_muc / ".xong").write_text("ok", encoding="utf-8")
    print("xong")


if __name__ == "__main__":
    main()
