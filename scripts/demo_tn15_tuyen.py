"""
demo_tn15_tuyen.py — Dữ liệu cho phần "mô hình chọn tuyến" của demo TN15 / TN15b (results/bao_da_tuyen/demo_tn15_15b/).

Với mỗi kịch bản (chọn theo luật, không theo kết quả mô hình) và MỌI cặp điểm đầu (27 điểm người dùng) – điểm cuối (5 trạm mặt
đất): giữ 4–5 luồng của kịch bản gốc, thay luồng 0 bằng cặp được chọn, dựng kho tuyến, rồi cho HAI mô hình chọn tập tuyến:
    DM 7/3 học lại cùng đường ống (dm13_w0.7_s1)  và  DM x10 (dm13_x10_s1)
Seed 1 = cặp seed có hiệu mất dịch vụ ở GIỮA trong 5 cặp của TN15b (không chọn cặp đẹp nhất). Giải mã không tái nhiễu 5 × 32.
Chấm trên 32 thế giới chấm (bộ sinh riêng, như dữ liệu kiểm tra).

Kịch bản:
    bão  — bao15b, cửa sổ của đợt bão mạnh nhất trong 20 đợt (theo cường độ dùng để chọn đợt), mốc có tổng "điểm bão" cao nhất;
    ngày — ngay15b, cửa sổ đầu tiên theo thời gian, mốc có tổng "điểm bão" cao nhất.

    python -m scripts.demo_tn15_tuyen      # ~5–10 phút, ghi demo_tn15_15b/tuyen.json
"""

from __future__ import annotations

import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np

GOC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GOC))

KQ = GOC / "results" / "bao_da_tuyen"
RA = KQ / "demo_tn15_15b"
MO_HINH = {"7/3": (KQ / "tn15b_may_nhom" / "mo_hinh13", "dm13_w0.7_s1"),
           "x10": (KQ / "tn15_may_nhom" / "mo_hinh13", "dm13_x10_s1")}
MUC = 1.0


def chon_cua_so():
    from src.nhieu_luong import tao_du_lieu_dt as tdt
    tt = tdt.thoi_tiet_9()
    cd = np.maximum((tt.precip > 11).mean(1) / 0.05, (tt.gust > 15).mean(1) / 0.20)
    bao = tdt.cac_cua_so_tap("bao15b")
    T = [str(x) for x in tt.time_utc]
    # cường độ của từng cửa sổ bão = max cd trong 24 giờ quanh h0
    kb_bao = int(np.argmax([cd[max(0, h0 - 12):h0 + 12].max() for _, h0, _ in bao]))
    return [("bão", "bao15b", kb_bao), ("ngày xấu", "ngay15b", 0)], T


def main():
    import torch

    from src.ngau_nhien.kich_ban import sinh_kich_ban
    from src.ngau_nhien.ma_tran_rui_ro import ma_tran_online
    from src.nhieu_luong import da_tuyen as dt
    from src.nhieu_luong import mo_hinh_dt as md
    from src.nhieu_luong import tao_du_lieu_dt as tdt
    from src.nhieu_luong import tao_du_lieu_rr as td
    from src.nhieu_luong import the_gioi_rr as tg
    from src.nhieu_luong.bai_toan_rr import DELTA, TUOI
    torch.set_num_threads(4)
    td._khoi_tao()
    g, p = td._g, td._p
    net = {k: md.nap(ten, d) for k, (d, ten) in MO_HINH.items()}
    z = np.load(td.RAW, allow_pickle=False)
    latlon = z["node_latlon_alt"]
    moc = td.cac_moc("test")
    ds, _ = chon_cua_so()
    nguoi = [n for n in range(g.N) if g.kind[n] == "user"]
    tram = [n for n in range(g.N) if g.kind[n] == "gateway"]
    ra = {"mo_hinh": {k: v[1] for k, v in MO_HINH.items()}, "muc_tai": MUC, "kich_ban": []}
    for nhan, tap, kb in ds:
        s0 = time.time()
        ten, h0, seed = tdt.cac_cua_so_tap(tap)[kb]
        tt, p_sim = tdt.moi_truong(tap, kb)
        Wn, q = sinh_kich_ban(g, tt, h0, seed, p_sim)
        pl = tg.phat_lai(g, tt, h0, seed, p_sim)
        diem = [float(dt.diem_bao(g, q, int(t) - TUOI).sum()) for t in moc]
        t = int(moc[int(np.argmax(diem))])
        o = t - TUOI
        past = {k: q[k][:o + 1] for k in ("R", "gust", "giong", "tat")}
        r = ma_tran_online(td._bang, g, Wn[o], past, t, TUOI, DELTA, tham_so=p)
        th0 = dt.tao_tinh_huong(g, q, r, t, np.random.default_rng([seed, 5]), so_luong=(4, 5), so_nguon=3)
        if th0 is None:
            print(f"  {ten}: không dựng được kịch bản", flush=True)
            continue
        mg = th0["mg"]
        cape = tdt.cape_luc(h0, o, tdt.tep_thoi_tiet(tap))
        sc = tdt._seeds(seed, t, 1, 32)
        kb_ra = {"nhan": nhan, "cua_so": ten, "t": t, "gio_utc": str(tt.time_utc[h0 + int(o * 20 / 3600)])[:16],
                 "luong_goc": [[g.names[int(s)], g.names[int(d)]] for s, d in zip(th0["src"], th0["dst"])],
                 "vi_tri": {g.names[n]: [round(float(latlon[t, n, 0]), 3), round(float(latlon[t, n, 1]), 3)] for n in range(g.N)},
                 "loai": {g.names[n]: g.kind[n] for n in range(g.N)}, "cap": {}}
        for s in nguoi:
            for d in tram:
                kt = dt.kho_tuyen(mg, s, d)
                khoa = f"{g.names[s]}|{g.names[d]}"
                if kt is None:
                    kb_ra["cap"][khoa] = None
                    continue
                th = dict(th0)
                th["kho"] = list(th0["kho"]) + [kt]
                th["src"], th["dst"], th["cap"] = th0["src"].copy(), th0["dst"].copy(), th0["cap"].copy()
                th["src"][0], th["dst"][0], th["cap"][0] = s, d, len(th["kho"]) - 1
                F0 = np.zeros(len(mg.eu))
                for k, c in enumerate(th["cap"]):
                    nh = min(th["kho"][c]["kho"], key=lambda rr: float(mg.d0[mg.dt.canh_cua(rr)].sum()))
                    F0[mg.dt.canh_cua(nh)] += th["lam"][k]
                th["u"] = float(np.max(F0[F0 > 0] / mg.trong[F0 > 0]))
                bt = dt.BaiToan(th, MUC)
                gon = dt.BaiToanGon(bt)
                gon.dac_trung_canh = tdt.dac_trung_canh(g, q, mg, gon.U, o, cape)
                gon.dac_trung_uv = [tdt.dac_trung_uv(g, th, k) for k in range(bt.K)]
                Ec = dt.TheGioi.ensemble(bt, g, pl, p_sim, sc)
                ten_tuyen = lambda rr: [g.names[int(n)] for n in rr]
                x = {"kho": [ten_tuyen(rr) for rr in bt.bang[0][:bt.so_kho[0]]],
                     "tre_kho": [round(float(mg.d0[mg.dt.canh_cua(rr)].sum()), 2) for rr in bt.bang[0][:bt.so_kho[0]]]}
                for k_mh, n_mh in net.items():
                    a = md.chon_sua_dan(n_mh, gon, 32, 5, torch.Generator().manual_seed(zlib.crc32(f"{khoa}|{t}".encode())))
                    sol = bt.tu_lua_chon(a)
                    c = Ec.cham(sol)
                    x[k_mh] = {"tuyen": [int(i) for i in sol[0]],
                               "mat_luong": round(float(c["H"][0]) * 100, 2), "tre_luong": round(float(c["d"][0]), 2),
                               "mat_tong": round(float(c["mat_dv_%"]), 2), "tre_tong": round(float(c["tre_chinh_ms"]), 2),
                               "a_tong": round(float(c["khong_tuyen_song_%"]), 2), "b_tong": round(float(c["du_phong_qua_tai_%"]), 2)}
                kb_ra["cap"][khoa] = x
        ra["kich_ban"].append(kb_ra)
        n_ok = sum(v is not None for v in kb_ra["cap"].values())
        print(f"  {nhan}: {ten}, mốc {t}, {n_ok}/{len(kb_ra['cap'])} cặp tới được ({time.time() - s0:.0f} s)", flush=True)
    RA.mkdir(parents=True, exist_ok=True)
    (RA / "tuyen.json").write_text(json.dumps(ra, ensure_ascii=False), encoding="utf-8")
    print("xong", RA / "tuyen.json", flush=True)


if __name__ == "__main__":
    main()
