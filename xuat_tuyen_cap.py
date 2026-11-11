"""
xuat_tuyen_cap.py — Định tuyến DM / dphat / dijkstra cho cặp NGUỒN–ĐÍCH bất kỳ.

Vì sao cần: `duong_di_nut.json` chỉ có 4–5 luồng mà đề bài chọn trước ở mỗi mốc, nên
chọn cặp khác trên bản đồ 3D thì không có DM/dphat. Script này tính tuyến cho cặp
người dùng muốn, theo đúng quy trình của dự án ở phạm vi MỘT luồng:

  kho tuyến  `da_tuyen.kho_tuyen(mg, s, d)`  (≤12 tuyến: Suurballe, Yen trễ, Yen rủi ro)
  dm         mô hình diffusion (`mo_hinh_dt.ham_chon`) chọn tuyến chính trong kho
  dphat      tuyến trong kho tối thiểu prop + 10·(−ln(1−risk))
  dijkstra   Dijkstra thuần theo trễ trên W của khung

Kết quả ghi `duong_di_cap.json`, khoá `{bao}|{t}|{kb}|{nguồn}|{đích}`:
  {"bao","t","kb","nguon","dich","dm","dphat","dijkstra","so_kho"}
Bản đồ 3D đọc file này và vẽ thêm DM/dphat cho cặp đã chọn (nhãn "1 luồng").

Chạy ở thư mục gốc:

    python xuat_tuyen_cap.py --bao soulik --mau         # 10 cặp mẫu (~1 phút)
    python xuat_tuyen_cap.py --bao soulik               # mọi cặp 32×5 (~15 phút)
    python xuat_tuyen_cap.py --bao soulik --cap CaMau,DaNang --cap PleiKu,Hanoi
    python ban_do_3d.py --bao soulik                    # vẽ lại bản đồ (đọc file ở trên)

Lưu ý: đây là định tuyến 2 node ĐỘC LẬP (tải của riêng luồng đó), nên với cặp đã có
trong kịch bản 4–5 luồng thì tuyến ở đây có thể khác `duong_di_nut.json` (tuyến gốc
tính chung tải nhiều luồng). Bản đồ đánh dấu `*` cho các tuyến loại này.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

GOC = Path(__file__).resolve().parent
if str(GOC) not in sys.path:
    sys.path.insert(0, str(GOC))

BAO = GOC / "data" / "bao_manh"
RA = GOC / "data" / "bao_da_tuyen"
BETA = 10.0
CAP_MAU = [("CaMau", "DaNang"), ("PleiKu", "Hanoi"), ("TruongSa", "HoiAn" if False else "HoChiMinh"),
           ("QuangNinh", "CanTho"), ("Vinh", "DaNang"), ("Hanoi", "DaNang"),
           ("HaiPhong", "HoChiMinh"), ("NhaTrang", "Hanoi"), ("LySon", "DaNang"),
           ("ConDao", "HaiPhong")]


def chon_dphat(mg, kt) -> int:
    """Tuyến trong kho tối thiểu prop + β·(−ln(1−risk)) — như `xuat_tuyen_dm.chon_dphat`."""
    rui = np.clip(mg.risk, 0.0, 1 - 1e-9)
    phi = mg.prop + BETA * (-np.log1p(-rui))
    gia = []
    for r in kt["kho"]:
        e = mg.dt.canh_cua(r)
        gia.append(float(phi[e].sum()) if len(e) else np.inf)
    return int(np.argmin(gia))


def cap_can_tinh(a, names: list[str]) -> list[tuple[int, int]]:
    """Danh sách (s, d) cần tính: cặp chỉ định, hoặc mọi user/gateway × gateway."""
    if a.cap:
        ra = []
        for c in a.cap:
            s_nm, d_nm = [x.strip() for x in c.split(",")]
            if s_nm not in names or d_nm not in names:
                raise SystemExit(f"không có nút {s_nm} / {d_nm}")
            ra.append((names.index(s_nm), names.index(d_nm)))
        return ra
    if a.mau:
        ra = []
        for s_nm, d_nm in CAP_MAU:
            if s_nm in names and d_nm in names:
                ra.append((names.index(s_nm), names.index(d_nm)))
        return ra
    # mặc định: mọi cặp nguồn (user/gateway) → đích (gateway), như bộ sinh nhãn
    kind = a.kind
    nguon = [i for i, k in enumerate(kind) if k in ("user", "gateway")]
    dich = [i for i, k in enumerate(kind) if k == "gateway"]
    return [(s, d) for s in nguon for d in dich if s != d]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bao", default="soulik_2024")
    ap.add_argument("--cap", action="append", default=[], help='"nguồn,đích", lặp lại được')
    ap.add_argument("--mau", action="store_true", help="chỉ 10 cặp mẫu")
    ap.add_argument("--frames", type=int, default=0, help="giới hạn số mốc (mặc định mọi mốc)")
    ap.add_argument("--thu-muc", default="mo_hinh6")
    ap.add_argument("--ten", default="dm6")
    ap.add_argument("--out", default="duong_di_cap.json")
    a = ap.parse_args()

    from src.nhieu_luong import tao_du_lieu_rr as td
    from src.nhieu_luong import da_tuyen as dt
    from src.nhieu_luong import tao_du_lieu_dt as tdt
    from src.nhieu_luong.tao_du_lieu_dt import dac_trung_canh, dac_trung_uv, cape_luc
    from src.nhieu_luong import mo_hinh_dt as md
    from src.ngau_nhien.ma_tran_rui_ro import ma_tran_online
    from src.ngau_nhien.kich_ban import sinh_kich_ban
    from src.ngau_nhien.thoi_tiet import ThoiTiet

    ten = a.bao.split("_")[0]
    # h0 + seed phải khớp tao_du_lieu_bao.py (cửa sổ c00, seed 800000 + q)
    cat = json.loads((BAO / "danh_muc.json").read_text(encoding="utf-8"))["bao"]
    thong_tin = next(b for b in cat if b["ten"] == a.bao or b["ten"].startswith(ten))
    ten_bao = thong_tin["ten"]
    h0 = int(thong_tin["gio_bat_dau_kich_ban"][0])
    seed = 800000

    print("Khởi tạo mạng …", flush=True)
    td._khoi_tao()
    g, p, bang = td._g, td._p, td._bang
    names = [str(x) for x in g.names]
    kind = [str(x) for x in g.kind]
    a.kind = kind

    print(f"Sinh kịch bản {ten_bao} …", flush=True)
    tt = ThoiTiet(str(BAO / f"{ten_bao}.npz"))
    Wn, q = sinh_kich_ban(g, tt, h0, seed, p)

    model = md.ham_chon(GOC / "results" / "bao_da_tuyen" / a.thu_muc, a.ten, softmax=False)["dm"]
    cape_o_ten = str(BAO / f"{ten_bao}.npz")
    muc = list(tdt.MUC_TAI)[0]
    TUOI, DELTA = td.TUOI, td.DELTA

    # các mốc m = mốc của tuyến đã giải mã (duong_di_nut.json), nếu có
    moc = sorted({int(k.split("|")[2]) for k in
                  json.loads((GOC / "duong_di_nut.json").read_text(encoding="utf-8"))
                  if k.startswith(ten_bao + "|")})
    if not moc:
        moc = list(range(TUOI + 5, Wn.shape[0] - 1, 7))
    if a.frames:
        moc = moc[:a.frames]

    cap = cap_can_tinh(a, names)
    print(f"{len(moc)} mốc × {len(cap)} cặp = {len(moc)*len(cap)} phép tính", flush=True)

    ra, t0 = {}, time.time()
    for i, t in enumerate(moc, 1):
        o = t - TUOI
        if o < 0:
            continue
        past = {k: q[k][:o + 1] for k in ("R", "gust", "giong", "tat")}
        r = ma_tran_online(bang, g, Wn[o], past, t, TUOI, DELTA, tham_so=p)
        mg = dt.Mang(g, q, r, t)
        cape_o = cape_luc(h0, o, cape_o_ten)
        for s, d in cap:
            kt = dt.kho_tuyen(mg, s, d)
            if kt is None:
                continue
            # bài toán 1 luồng: tải nền = tuyến nhanh nhất của chính cặp
            th = {"mg": mg, "src": np.array([s]), "dst": np.array([d]),
                  "lam": np.array([1.0]), "kho": [kt], "cap": np.array([0]), "t": mg.t}
            F0 = np.zeros(len(mg.eu))
            nhanh = min(kt["kho"], key=lambda r0: float(mg.d0[mg.dt.canh_cua(r0)].sum()))
            F0[mg.dt.canh_cua(nhanh)] += 1.0
            th["u"] = float(np.max(F0[F0 > 0] / mg.trong[F0 > 0]))
            Y = dt.yen_m(th)
            DP = dt.dijkstra_phat(th, th["lam"] * 1.0 / th["u"], *tdt.BETA_GAMMA)
            bt = dt.BaiToan(th, muc, [Y[0] + DP[0]])
            gon = dt.BaiToanGon(bt)
            gon.dac_trung_canh = dac_trung_canh(g, q, mg, gon.U, o, cape_o)
            gon.dac_trung_uv = [dac_trung_uv(g, th, k) for k in range(bt.K)]
            a_dm = int(model(gon)[0])
            iz = dt.BaiToan.chi_so(bt, 0, bt.bang[0][int(bt.opts[0][a_dm][0])])
            chi = img_dm = int(bt.opts[0][a_dm][0])
            _ = (iz, chi, img_dm)
            tuyen_dm = kt["kho"][int(bt.opts[0][a_dm][0])]
            tuyen_dp = kt["kho"][chon_dphat(mg, kt)]
            dij = mg.dt.duong(s, d, mg.d0)
            ra[f"{ten_bao}|{t}|0|{s}|{d}"] = {
                "bao": ten_bao, "t": int(t), "kb": 0,
                "nguon": names[s], "dich": names[d],
                "dm": [int(x) for x in tuyen_dm],
                "dphat": [int(x) for x in tuyen_dp],
                "dijkstra": [int(x) for x in dij] if dij else None,
                "so_kho": len(kt["kho"]), "mot_luong": True,
            }
        print(f"  mốc {i}/{len(moc)} t={t}: tổng {len(ra)} mục "
              f"({time.time()-t0:.0f}s)", flush=True)

    out = GOC / a.out
    cu = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    cu.update(ra)
    out.write_text(json.dumps(cu, separators=(",", ":")), encoding="utf-8")
    print(f"Đã ghi {out}: {len(ra)} mục mới, tổng {len(cu)}")


if __name__ == "__main__":
    main()
