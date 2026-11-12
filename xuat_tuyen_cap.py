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

CHẠY (chạy ở thư mục gốc):
    python xuat_tuyen_cap.py --bao soulik --mau         # 1 bão, 10 cặp mẫu (~1 phút)
    python xuat_tuyen_cap.py --bao soulik               # 1 bão, mọi cặp 32×5 (~15 phút)
    python xuat_tuyen_cap.py --bao soulik --cap CaMau,DaNang --cap PleiKu,Hanoi
    python xuat_tuyen_cap.py --tat-ca --tien-trinh 6    # CẢ 10 BÃO, song song 6 tiến trình
    python ban_do_3d.py --bao soulik                    # vẽ lại bản đồ (đọc file ở trên)

Mỗi mốc cần 45 bước lịch sử (o_dac_trung), nên mốc t < 54 bị bỏ qua — giống hệt
`tao_du_lieu_bao.py`.

Lưu ý: đây là định tuyến 2 node ĐỘC LẬP (tải của riêng luồng đó), nên với cặp đã có
trong kịch bản 4–5 luồng thì tuyến ở đây có thể khác `duong_di_nut.json` (tuyến gốc
tính chung tải nhiều luồng). Bản đồ đánh dấu `*` cho các tuyến loại này.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

GOC = Path(__file__).resolve().parent
if str(GOC) not in sys.path:
    sys.path.insert(0, str(GOC))

BAO_DIR = GOC / "data" / "bao_manh"
BETA = 10.0
CAP_MAU = [("CaMau", "DaNang"), ("PleiKu", "Hanoi"), ("TruongSa", "HoChiMinh"),
           ("QuangNinh", "CanTho"), ("Vinh", "DaNang"), ("Hanoi", "DaNang"),
           ("HaiPhong", "HoChiMinh"), ("NhaTrang", "Hanoi"), ("LySon", "DaNang"),
           ("ConDao", "HaiPhong")]


# ------------------------------------------------------------------ phần tính toán
def chon_dphat(mg, kt) -> int:
    """Tuyến trong kho tối thiểu prop + β·(−ln(1−risk))."""
    rui = np.clip(mg.risk, 0.0, 1 - 1e-9)
    phi = mg.prop + BETA * (-np.log1p(-rui))
    gia = []
    for r in kt["kho"]:
        e = mg.dt.canh_cua(r)
        gia.append(float(phi[e].sum()) if len(e) else np.inf)
    return int(np.argmin(gia))


def tinh_mot_bao(job):
    """Sinh kịch bản 1 bão rồi tính tuyến cho mọi (mốc, cặp). Chạy trong 1 tiến trình."""
    ten_bao, cap_yeu_cau, gioi_han_moc, ten_mo_hinh, ten_dm, chia, phan = job
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    t0 = time.time()

    from src.nhieu_luong import da_tuyen as dt
    from src.nhieu_luong import mo_hinh_dt as md
    from src.nhieu_luong import tao_du_lieu_dt as tdt
    from src.nhieu_luong import tao_du_lieu_rr as td
    from src.nhieu_luong.tao_du_lieu_dt import dac_trung_canh, dac_trung_uv, cape_luc
    from src.ngau_nhien.kich_ban import sinh_kich_ban
    from src.ngau_nhien.ma_tran_rui_ro import ma_tran_online
    from src.ngau_nhien.thoi_tiet import ThoiTiet

    cat = json.loads((BAO_DIR / "danh_muc.json").read_text(encoding="utf-8"))["bao"]
    ttin = next(b for b in cat if b["ten"] == ten_bao)
    h0 = int(ttin["gio_bat_dau_kich_ban"][0])
    tep = str(BAO_DIR / ttin["tep"])

    td._khoi_tao()
    g, p, bang = td._g, td._p, td._bang
    names = [str(x) for x in g.names]
    kind = [str(x) for x in g.kind]
    tt = ThoiTiet(tep)
    Wn, q = sinh_kich_ban(g, tt, h0, 800000, p)

    model = md.ham_chon(GOC / "results" / "bao_da_tuyen" / ten_mo_hinh, ten_dm,
                        softmax=False)["dm"]
    TUOI, DELTA = td.TUOI, td.DELTA
    muc = list(tdt.MUC_TAI)[0]

    # mốc: lấy từ duong_di_nut.json của đúng bão (nếu có), ngược lại rải đều
    f_dd = GOC / "duong_di_nut.json"
    moc = []
    if f_dd.exists():
        dd = json.loads(f_dd.read_text(encoding="utf-8"))
        moc = sorted({int(k.split("|")[2]) for k in dd if k.startswith(ten_bao + "|")})
    if not moc:
        moc = list(range(0, Wn.shape[0], 11))
    # cần 45 bước lịch sử trước mốc quan sát (o = t - TUOI)
    moc = [t for t in moc if t - TUOI >= 45]
    if gioi_han_moc:
        moc = moc[:gioi_han_moc]

    if cap_yeu_cau:
        cap = [(names.index(s), names.index(d)) for s, d in cap_yeu_cau]
    else:
        nguon = [i for i, k in enumerate(kind) if k in ("user", "gateway")]
        dich = [i for i, k in enumerate(kind) if k == "gateway"]
        cap = [(s, d) for s in nguon for d in dich if s != d]
    # chia việc: tiến trình `phan` chỉ làm các cặp ở chỉ số phan, phan+chia, ...
    if chia > 1:
        cap = cap[phan::chia]

    ra = {}
    for i_moc, t in enumerate(moc, 1):
        o = t - TUOI
        past = {k: q[k][:o + 1] for k in ("R", "gust", "giong", "tat")}
        r = ma_tran_online(bang, g, Wn[o], past, t, TUOI, DELTA, tham_so=p)
        mg = dt.Mang(g, q, r, t)
        cape_o = cape_luc(h0, o, tep)
        for s, d in cap:
            kt = dt.kho_tuyen(mg, s, d)
            if kt is None:
                continue
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
        if i_moc % 8 == 0 or i_moc == len(moc):
            print(f"    {ten_bao}: mốc {i_moc}/{len(moc)} t={t} · {len(ra)} mục "
                  f"({time.time()-t0:.0f}s)", flush=True)
    return ten_bao, ra, round(time.time() - t0)


# ------------------------------------------------------------------ CLI
def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bao", default="soulik_2024", help="tên bão (vd soulik)")
    ap.add_argument("--tat-ca", action="store_true", help="chạy cả 10 bão")
    ap.add_argument("--cap", action="append", default=[], help='"nguồn,đích", lặp lại được')
    ap.add_argument("--mau", action="store_true", help="chỉ 10 cặp mẫu")
    ap.add_argument("--frames", type=int, default=0, help="giới hạn số mốc")
    ap.add_argument("--tien-trinh", type=int, default=6, help="số tiến trình song song")
    ap.add_argument("--chia", type=int, default=1, help="chia danh sách cặp thành N phần")
    ap.add_argument("--phan", type=int, default=0, help="chỉ làm phần thứ K (0..chia-1)")
    ap.add_argument("--thu-muc", default="mo_hinh6")
    ap.add_argument("--ten", default="dm6")
    ap.add_argument("--out", default="duong_di_cap.json")
    a = ap.parse_args()

    cat = json.loads((BAO_DIR / "danh_muc.json").read_text(encoding="utf-8"))["bao"]
    if a.tat_ca:
        ds_bao = [b["ten"] for b in cat]
    else:
        ten = a.bao.split("_")[0]
        ds_bao = [b["ten"] for b in cat if b["ten"] == a.bao or b["ten"].startswith(ten)]
        if not ds_bao:
            raise SystemExit(f"không có bão '{a.bao}'")

    cap_yeu_cau = None
    if a.cap:
        cap_yeu_cau = [tuple(x.strip() for x in c.split(",")) for c in a.cap]
    elif a.mau:
        cap_yeu_cau = CAP_MAU

    jobs = [(b, cap_yeu_cau, a.frames, a.thu_muc, a.ten, a.chia, a.phan) for b in ds_bao]
    print(f"{len(jobs)} bão × {'cặp chỉ định' if cap_yeu_cau else 'mọi cặp'}"
          f" | {a.tien_trinh} tiến trình", flush=True)

    out = GOC / a.out
    t0 = time.time()
    tong = {}

    def ghi(ra, ten):
        """Ghi NGAY sau mỗi bão; nếu chia phần thì ghi file tạm riêng để gộp sau."""
        if a.chia > 1:
            tam = out.with_name(f"{out.stem}.phan{a.phan}{out.suffix}")
            tam.write_text(json.dumps(ra, separators=(",", ":")), encoding="utf-8")
            print(f"  -> {ten} phần {a.phan}: {len(ra)} mục -> {tam.name}", flush=True)
            return
        cu = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
        cu.update(ra)
        out.write_text(json.dumps(cu, separators=(",", ":")), encoding="utf-8")
        print(f"  -> đã ghi {ten}: {len(ra)} mục, file {len(cu)} mục "
              f"({out.stat().st_size/1e6:.1f} MB)", flush=True)

    if len(jobs) == 1 or a.tien_trinh <= 1:
        for i, j in enumerate(jobs, 1):
            ten, ra, giay = tinh_mot_bao(j)
            tong.update(ra)
            print(f"  {i}/{len(jobs)} {ten}: {len(ra)} mục ({giay}s)", flush=True)
            ghi(ra, ten)
    else:
        with get_context("spawn").Pool(min(a.tien_trinh, len(jobs))) as pool:
            for i, (ten, ra, giay) in enumerate(pool.imap_unordered(tinh_mot_bao, jobs), 1):
                tong.update(ra)
                print(f"  {i}/{len(jobs)} {ten}: {len(ra)} mục ({giay}s)", flush=True)
                ghi(ra, ten)

    if a.chia > 1:
        print(f"XONG phần {a.phan}/{a.chia}: {len(tong)} mục trong {time.time()-t0:.0f}s")
        return
    print(f"XONG: {len(tong)} mục mới trong {time.time()-t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
