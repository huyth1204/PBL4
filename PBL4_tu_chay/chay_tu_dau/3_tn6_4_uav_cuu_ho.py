"""Thí nghiệm 3/4 — TN6+4: 2 UAV cứu hộ đổi chỗ mỗi 15 phút + định tuyến theo score, bão Yagi, có xung đột. Chạy từ đầu.

    python chay_tu_dau/3_tn6_4_uav_cuu_ho.py               # đầy đủ 2 phiên bản × (val, Yagi), khoảng 1–1,5 giờ
    python chay_tu_dau/3_tn6_4_uav_cuu_ho.py --chi-v2      # chỉ phiên bản 2 (kết quả chính), khoảng 30–40 phút
    python chay_tu_dau/3_tn6_4_uav_cuu_ho.py --thu         # chạy thử 1 kịch bản mỗi tập, vài phút

Cần bảng tra rủi ro và danh mục bão từ 1_bang_tra_rui_ro.py (ket_qua_tu_chay/bang_tra/). Chưa chạy bước đó thì thêm
--bang-co-san để dùng bản sao nhóm đã dùng (results/cuu_ho/dau_vao/).
Mỗi kịch bản: mô phỏng bão (mức khắc nghiệt, xung đột), 3 cách đặt UAV cứu hộ (khong / gan_nut_yeu / vet_can) × 2 router
(quy_dao / score α 0,8 lúc nhẹ, 0,1 lúc bão), cộng trần hậu kiểm   (src/ngau_nhien/cuu_ho.py)
Tổng hợp + khoảng tin cậy bootstrap theo kịch bản + tiêu chí C1–C4   (src/ngau_nhien/cuu_ho_tong_hop.py)
Phiên bản 1: luật liên kết TV1, mọi truy vấn (tiêu chí results/cuu_ho/TIEU_CHI_6.md).
Phiên bản 2: đường lên 5 GHz tầm xa + chỉ truy vấn mặt đất (tiêu chí results/cuu_ho/TIEU_CHI_6b.md).
"""
import argparse

import numpy as np

from _chung import GOC, bang, buoc, doc, kiem_dau_vao, ktc, so, thu_muc, tieu_de

ap = argparse.ArgumentParser()
ap.add_argument("--thu", action="store_true", help="chạy thử 1 kịch bản mỗi tập")
ap.add_argument("--chi-v2", action="store_true", help="chỉ chạy phiên bản 2")
ap.add_argument("--bang-co-san", action="store_true", help="dùng bảng tra + danh mục bão trong results/cuu_ho/dau_vao/")
a = ap.parse_args()
kiem_dau_vao("data/thoi_tiet_vn_mr.npz", "dataset/oracle_dataset_v5.npz", "dataset/v5/split_info.json")
if a.bang_co_san:
    NGUON = GOC / "results/cuu_ho/dau_vao"
else:
    NGUON = GOC / "ket_qua_tu_chay" / "bang_tra"
    if a.thu and not (NGUON / "bang-tra-bao" / "meta.json").exists():
        NGUON = GOC / "ket_qua_tu_chay" / "bang_tra_thu"      # chạy thử được với bảng của 1_bang_tra_rui_ro.py --thu
    if not (NGUON / "bang-tra-bao" / "meta.json").exists():
        raise SystemExit("Chưa có bảng tra. Chạy trước: python chay_tu_dau/1_bang_tra_rui_ro.py\n"
                         "hoặc dùng bản sao của nhóm: python chay_tu_dau/3_tn6_4_uav_cuu_ho.py --bang-co-san")
BANG, BAO = NGUON / "bang-tra-bao" / "bang_tra.npz", NGUON / "bao-manh"
D = thu_muc("tn6_4", a.thu)
tieu_de("TN6+4 — 2 UAV CỨU HỘ + ĐỊNH TUYẾN THEO SCORE KHI BÃO YAGI",
        "Đặt UAV cứu hộ và định tuyến theo score có tăng tỉ lệ truy vấn được phục vụ không?", a.thu)
print(f"Bảng tra: {BANG.relative_to(GOC)}")

PB = {"v1": [], "v2": ["--duong-len-5ghz", "--chi-mat-dat"]}
if a.chi_v2:
    PB.pop("v1")
for v, co in PB.items():
    for tap in ("val", "yagi"):
        out = D / f"{v}_{tap}"
        buoc(f"Phiên bản {v[1]}, tập {tap}: mô phỏng từng kịch bản",
             ["src.ngau_nhien.cuu_ho", "--tap", tap, "--bang", BANG, "--bao-dir", BAO, *co, "--out", out,
              *(["--chi", "1"] if a.thu else [])], D / "nhat_ky" / f"{v}_{tap}.log")
        buoc(f"Phiên bản {v[1]}, tập {tap}: tổng hợp", ["src.ngau_nhien.cuu_ho_tong_hop", "--vao", out],
             D / "nhat_ky" / f"{v}_{tap}_tong_hop.log")

# ---- in kết quả
PP = (("khong", "không cứu hộ"), ("gan_nut_yeu", "gần nút yếu"), ("vet_can", "vét cạn"))
print("\nKẾT QUẢ trên Yagi (tập val chỉ để sửa lỗi; KTC 95% bootstrap theo kịch bản; 'hơn' = KTC nằm hẳn trên 0)")
for v in PB:
    d = doc(D / f"{v}_yagi" / "tong_hop.json")
    p = d["phuong_phap"]
    print(f"Phiên bản {v[1]}: {d['so_truy_van']} truy vấn, {d['so_kich_ban']} kịch bản")
    bang(["Cách đặt UAV", "router quỹ đạo", "router score", "trần hậu kiểm"],
         [[t, so(p[f"{m}|quy_dao"]["tat_ca"]) + "%", so(p[f"{m}|score"]["tat_ca"]) + "%", so(p[f"{m}|hau_kiem"]["tat_ca"]) + "%"]
          for m, t in PP])
    bang(["Tiêu chí", "So sánh", "Hiệu (điểm %)", "Kết quả"],
         [[k, c["so_sanh"], ktc(c["hieu_diem_%"], c["ktc95"]), "ĐẠT" if c["dat"] else ("ngược chiều" if c["ktc95"][1] < 0 else "trượt")]
          for k, c in d["tieu_chi"].items()])
    print(f"Số lần UAV cứu hộ đổi chỗ: gần nút yếu {d['so_lan_doi_cho']['gan_nut_yeu']}, vét cạn {d['so_lan_doi_cho']['vet_can']}\n")

# ---- so với kết quả gốc, từng truy vấn
GOC_KQ = {"v1": "results/cuu_ho/yagi", "v2": "results/cuu_ho/v2_yagi"}
hang = []
for v in (PB if (not a.thu or a.bang_co_san) else ()):  # chạy thử với bảng tra thu thì không so
    tong = khop = 0
    for f in sorted((D / f"{v}_yagi").glob("yagi_*.npz")):
        cu = GOC / GOC_KQ[v] / f.name
        if not cu.exists():
            continue
        x, y = np.load(f), np.load(cu)
        k = [k for k in y.files if k.endswith(("phuc_vu", "hau_kiem"))]
        tong += 1
        khop += all(np.array_equal(x[kk], y[kk]) for kk in k)
    if tong:
        hang.append([f"Phiên bản {v[1]}", f"{khop}/{tong} kịch bản khớp từng truy vấn"])
if hang:
    print(f"So với kết quả gốc ({', '.join(GOC_KQ.values())}):")
    bang(["", "Kết quả"], hang)
    if a.bang_co_san is False:
        print("(Bảng tra tự học ở bước 1 phải trùng mã băm với bản sao của nhóm thì mới khớp từng truy vấn.)")
