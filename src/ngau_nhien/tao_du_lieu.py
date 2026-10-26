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

CHỌN GIỜ THỜI TIẾT (mặc định --chon-gio roi)
  Mỗi kịch bản dài ceil(T*20/3600)+2 giờ (~6 giờ với T=600). Bản cũ gọi
  `rng.choice(gio_bat_dau)` riêng cho từng kịch bản: cửa sổ chồng lên nhau và train/val/test
  dùng chung thời tiết. Mặc định mới rải các cửa sổ RỜI NHAU trên trục thời gian và loại
  04–12/9/2024 (bão Yagi, `--ngay-loai khong` để tắt). Sau khi sinh, meta.json ghi `h0`
  của từng tập và script tự kiểm 3 tập không dùng chung giờ nào.
    --chon-gio doc-lap     cách cũ (độc lập, không loại gì) — chỉ để tái lập số đã công bố
    --chon-gio theo-ngay   chia train/val theo NGÀY (`chia_ngay`, cùng luật với
                           tao_du_lieu_nl.chon_gio) rồi mới rải cửa sổ rời nhau
    --ty-le-ngay-rong      chia trục thời gian thành 3 đoạn train → val → test liền nhau
                           (test ở tương lai xa nhất) — dùng để đo ngoại suy theo thời gian
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.ngau_nhien.khoang_loi import MUC, chi_phi, dijkstra
from src.ngau_nhien.kich_ban import MangGoc, ThamSo, sinh_kich_ban
from src.nhieu_luong.tao_du_lieu_nl import NGAY_YAGI     # dùng CHUNG một định nghĩa khoảng ngày bão
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


def chon_h0(tt, n_kb, rng, so_gio, thang=None, kieu="roi", ngay_loai=(), ngay_train=None, ngay_val=None,
            cua_so_cho=None):
    """Chọn `n_kb` giờ bắt đầu cửa sổ thời tiết.

    Bẫy đã sập: gọi `rng.choice(tt.gio_bat_dau(so_gio))` riêng cho từng kịch bản thì các
    cửa sổ (6 giờ với T=600) rải trên 17 544 giờ mà không kiểm tra gì → chồng lên nhau, và
    train/val/test dùng chung khoảng thời tiết dù seed khác nhau. Thêm nữa không loại
    04–12/9/2024 (Yagi) nên train nhiễm chính cơn bão dùng làm test.
      - `roi`      : rải cửa sổ trên trục thời gian rồi xê dịch cho tới khi mọi cửa sổ cách
                     nhau ≥ `so_gio` giờ; bỏ cửa sổ chạm ngày trong `ngay_loai`.
      - `theo-ngay`: như `roi` nhưng chỉ lấy cửa sổ nằm TRỌN trong ngày của một tập
                     (`ngay_train` hoặc `ngay_val`) — luật chia ngày của `chia_ngay`.
      - `doc-lap`  : cách cũ (độc lập, không loại, không ràng buộc) — chỉ để tái lập kết quả
                     đã công bố (truyền `--chon-gio doc-lap`).
    `cua_so_cho` = (lo, hi): giới hạn thêm miền giờ bắt đầu (xem `chia_doan`).
    """
    mien = tt.gio_bat_dau(so_gio, thang)
    if cua_so_cho is not None:
        mien = mien[(mien >= cua_so_cho[0]) & (mien <= cua_so_cho[1])]
        if len(mien) == 0:
            raise SystemExit(f"Đoạn {cua_so_cho} không có giờ bắt đầu nào cho cửa sổ {so_gio} giờ")
    if kieu == "doc-lap":
        return [int(rng.choice(mien)) for _ in range(n_kb)]
    ngay = np.array([str(x)[:10] for x in tt.time_utc])
    if kieu == "theo-ngay":
        tap = ngay_val if ngay_val is not None else ngay_train
        if tap is None:
            raise SystemExit("theo-ngay cần ngay_train hoặc ngay_val (xem chia_ngay)")
        tap = set(x[:10] for x in tap)
    elif kieu != "roi":
        raise SystemExit(f"Kiểu chọn giờ không hợp lệ: {kieu}")
    # Bọc thêm 1 giờ mỗi phía: `sinh_kich_ban` đọc tới `h0 + so_gio + 1`, và cận trên của
    # `gio_bat_dau` + `so_gio` có thể lố 1 giờ sang ngày kế tiếp (2024-09-03T23 -> 09-04).
    if kieu == "theo-ngay":
        dung = np.isin(ngay, list(tap))
    else:
        dung = ~np.isin(ngay, list(ngay_loai)) if len(ngay_loai) else np.ones(len(ngay), bool)
    hop = np.zeros(len(ngay) + 1, np.int32)                          # hop[i] = số giờ không dùng được trong [0, i)
    np.cumsum(~dung, out=hop[1:])
    lo, hi = mien, np.minimum(mien + so_gio + 1, len(ngay))
    mien = mien[hop[hi] == hop[lo]]                                  # cả [h0, h0+so_gio+1) đều dùng được
    if len(mien) == 0:
        raise SystemExit(f"{kieu}: không còn giờ bắt đầu nào hợp lệ "
                         "(sau khi loại ngày cấm / giới hạn trong ngày của tập)")
    n_kb = min(n_kb, len(mien) // so_gio + 1)
    # Rải `n_kb` khối TRỌN Ô đều kín miền rồi xê dịch ngẫu nhiên trong từng ô: các ô vốn đã
    # rời nhau (`so_gio` giờ/ô) nên xê dịch ±(ô - so_gio)/2 vẫn không chạm nhau. Cách cũ rải
    # theo `len(mien)-1` nên ô cuối trùng cận trên -> mọi tập đều chọn đúng cửa sổ cuối cùng
    # (h0 = 17539 xuất hiện ở cả train, val, test — đúng thứ lỗi hàm này sinh ra để chặn).
    o = len(mien) / n_kb
    le = max(0.0, (o - so_gio) / 2.0)
    vitri = np.round(np.arange(n_kb) * o + o / 2 + rng.uniform(-le, le, n_kb)).astype(int)
    chon = np.sort(mien[np.clip(vitri, 0, len(mien) - 1)])
    for i in range(1, n_kb):                                         # ô hẹp -> đẩy sang phải cho chắc
        if chon[i] - chon[i - 1] < so_gio:
            j = np.searchsorted(mien, chon[i - 1] + so_gio)
            if j >= len(mien):
                raise SystemExit(f"Không xếp được {n_kb} cửa sổ {so_gio} giờ rời nhau trong {len(mien)} "
                                 "giờ hợp lệ; giảm --kich-ban, nới --ngay-loai hoặc bỏ --ty-le-ngay-rong")
            chon[i] = mien[j]
    return [int(x) for x in chon]


def parse_ngay_loai(s):
    """'2024-09-04:2024-09-12' -> danh sách ngày (ISO, so sánh chuỗi được).

    None -> dùng NGAY_YAGI (cùng hằng của tao_du_lieu_nl). 'khong' -> không loại ngày nào.
    """
    if s is None:
        a, b = NGAY_YAGI
    elif str(s).strip().lower() in ("", "khong", "không", "none", "0"):
        return ()
    elif ":" in str(s):
        a, b = str(s).split(":", 1)
    else:
        a = b = str(s)
    ngay = np.arange(np.datetime64(a[:10]), np.datetime64(b[:10]) + 1, dtype="datetime64[D]").astype(str)
    return [str(x) for x in ngay]


def chia_ngay(tt, ty_le_train=0.85):
    """Chia theo NGÀY (không chia từng giờ), cùng luật với `tao_du_lieu_nl.chon_gio`:
    15% số ngày cho val; cửa sổ phải nằm trọn trong ngày của đúng tập đó."""
    ngay = np.array(sorted({str(x)[:10] for x in tt.time_utc}))
    r = np.random.default_rng(2026)
    so_val = max(1, int(round((1.0 - ty_le_train) * len(ngay))))
    ngay_val = set(r.choice(ngay, size=so_val, replace=False))
    return [str(d) for d in ngay if d not in ngay_val], sorted(str(d) for d in ngay_val)


def chia_doan(tt, so_gio):
    """Chia trục thời gian thành 3 đoạn liền nhau train → val → test, chừa `so_gio` giờ ở mỗi
    biên để cửa sổ hai bên không chạm nhau (test nằm ở tương lai xa nhất)."""
    H = len(tt.time_utc)
    b1, b2 = int(round(0.70 * H)), int(round(0.85 * H))
    return {"train": (0, b1 - so_gio), "val": (b1, b2 - so_gio), "test": (b2, H - so_gio)}


def kiem_tra_roi_nhau(out: Path, ten_tap, so_gio, kieu):
    """Đọc lại meta.json vừa ghi và khẳng định train/val/test không dùng chung giờ thời tiết."""
    m = json.load(open(out / "meta.json", encoding="utf-8"))
    h = {t: m.get(t, {}).get("h0") for t in ten_tap}
    if any(v is None for v in h.values()):
        print("(meta.json không có h0, bỏ qua kiểm tra cửa sổ rời nhau)")
        return
    if kieu == "doc-lap":
        print("Chế độ doc-lap: cửa sổ KHÔNG được kiểm rời nhau (giữ đúng cách chọn cũ).")
        return
    loi = []
    for x, y in (("train", "val"), ("train", "test"), ("val", "test")):
        chong = [a for a in h[x] if any(a < b + so_gio and b < a + so_gio for b in h[y])]
        if chong:
            loi.append(f"{x} và {y} dùng chung thời tiết ở {len(chong)} cửa sổ (vd h0={chong[0]})")
    if loi:
        raise SystemExit("Cửa sổ thời tiết KHÔNG rời nhau:\n  " + "\n  ".join(loi) +
                         "\nDùng --chon-gio roi (mặc định) hoặc --ty-le-ngay-rong.")
    print(f"Đã kiểm: cửa sổ thời tiết train/val/test rời nhau ({kieu}, {so_gio} giờ mỗi cửa sổ).")


def tao_tap(g, tt, raw, ids, n_kb, delta, tuoi, dau_vao, muc, seed0, rng, luu_cua_so, D_hong, thang=None,
            kieu_gio="roi", ngay_loai=(), ngay_train=None, ngay_val=None, cua_so_cho=None):
    can_relay = np.array([k != "user" for k in g.kind])
    cap = defaultdict(list)
    for sid, row in zip(raw["snapshot_id"], raw["y_paths_indices"]):
        if int(sid) in ids:
            p = [int(x) for x in row if x >= 0]
            cap[int(sid)].append((p[0], p[-1]))
    W_list, X_list, paths, sids, win_list, obs_list, bang = [], [], [], [], [], [], defaultdict(list)
    so_gio = int(np.ceil(g.T * 20 / 3600)) + 2
    h0s = chon_h0(tt, n_kb, rng, so_gio, thang, kieu_gio, ngay_loai, ngay_train, ngay_val, cua_so_cho)
    for e, h0 in enumerate(h0s):
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
    # giờ bắt đầu từng cửa sổ: `kiem_tra_roi_nhau` dùng để khẳng định 3 tập không dùng
    # chung thời tiết, và để tái lập / soát lại một bộ dữ liệu đã sinh.
    out["h0"] = np.array(h0s, np.int32)
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
    ap.add_argument("--chon-gio", choices=["roi", "theo-ngay", "doc-lap"], default="roi",
                    help="roi: cửa sổ thời tiết RỜI NHAU + loại --ngay-loai (mặc định); "
                         "theo-ngay: dùng chon_gio() của tao_du_lieu_nl (chia train/val theo ngày, loại Yagi); "
                         "doc-lap: cách cũ, độc lập hoàn toàn (chỉ để tái lập kết quả đã công bố)")
    ap.add_argument("--ngay-loai", default=None,
                    help="ngày cấm, dạng D0:D1 (mặc định 2024-09-04:2024-09-12 = bão Yagi); 'khong' = không loại")
    ap.add_argument("--ngay-train", type=float, default=0.85,
                    help="với --chon-gio theo-ngay: tỉ lệ ngày dành cho train (val lấy phần còn lại)")
    ap.add_argument("--ty-le-ngay-rong", action="store_true",
                    help="với --chon-gio roi: chia trục thời gian thành 3 đoạn train/val/test rời nhau")
    ap.add_argument("--luu-val", action="store_true", help="lưu cả cửa sổ tương lai cho val (để chọn guidance theo chi phí)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    raw = np.load(a.raw, allow_pickle=False)
    # MangGoc tự kiểm tập điểm thời tiết khớp cấu hình (chốt chặn chính, ở đó mới
    # sinh ra node_site = -1); ThoiTiet cũng kiểm ngay khi đọc tệp.
    tt = ThoiTiet(a.thoi_tiet)
    g = MangGoc(raw["W_snapshots"], raw["node_kind"], raw["node_order"], tt)
    split = json.load(open(a.split, encoding="utf-8"))["snapshot_ids"]
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    ngay_loai = parse_ngay_loai(a.ngay_loai)
    so_gio = int(np.ceil(g.T * 20 / 3600)) + 2
    ngay_tr, ngay_va = chia_ngay(tt, a.ngay_train) if a.chon_gio == "theo-ngay" else (None, None)
    doan = chia_doan(tt, so_gio) if (a.chon_gio == "roi" and a.ty_le_ngay_rong) else None
    meta = {"delta": a.delta, "tuoi": a.tuoi, "dau_vao": a.dau_vao, "muc": a.muc, "thoi_tiet": tt.tom_tat(),
            "D_hong": a.D_hong, "thang": a.thang, "chon_gio": a.chon_gio,
            "ngay_loai": list(ngay_loai), "so_gio_cua_so": so_gio,
            "ty_le_ngay_rong": bool(doan is not None)}
    ten_tap = ("train", "val", "test")
    for i, (ten, k) in enumerate(zip(ten_tap, map(int, a.kich_ban.split(",")))):
        cua_so = doan[ten] if doan is not None else None
        ds = tao_tap(g, tt, raw, set(split[ten]), k, a.delta, a.tuoi, a.dau_vao, a.muc,
                     a.seed * 1000 + i * 100, rng, ten == "test" or (ten == "val" and a.luu_val), a.D_hong,
                     doc_thang(a.thang), a.chon_gio, ngay_loai,
                     ngay_train=ngay_tr if ten != "val" else ngay_va,
                     ngay_val=ngay_va if ten == "val" else ngay_tr,
                     cua_so_cho=cua_so)
        ds.update(node_order=raw["node_order"], node_layer=raw["node_layer"], node_kind=raw["node_kind"])
        np.savez_compressed(out / f"{ten}.npz", **ds)
        meta[ten] = {"so_duong": int(len(ds["snapshot_id"])), "so_snapshot": int(len(ds["W_snapshots"])),
                     "h0": [int(x) for x in ds["h0"]]}
        if ten == "test":
            for x in ("ngay_tho", "quy_dao", "nhan"):
                meta[ten][f"G_{x}_%"] = float((ds[f"D_{x}"].mean() / ds["D_hau_kiem"].mean() - 1) * 100)
                meta[ten][f"dut_{x}_%"] = float(ds[f"dut_{x}"].mean() * 100)
        print(ten, meta[ten], flush=True)
    json.dump(meta, open(out / "meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    kiem_tra_roi_nhau(out, ten_tap, so_gio, a.chon_gio)


if __name__ == "__main__":
    main()
