"""
chay_thi_nghiem.py — Chạy lại toàn bộ thí nghiệm 4, 6, 7, 8 hoặc 9 theo đúng thứ tự (tai_lieu/BAO_CAO_THI_NGHIEM_6_7.md,
tai_lieu/HUONG_DAN_TN8.md).

    python -m scripts.chay_thi_nghiem 4                   # thí nghiệm 4 (Yagi, ≈ 1,5 giờ với 2–3 tiến trình)
    python -m scripts.chay_thi_nghiem 6 --chi-in          # chỉ in các lệnh
    python -m scripts.chay_thi_nghiem 7                   # chạy hết
    python -m scripts.chay_thi_nghiem 7 --tu-buoc 9       # chạy tiếp từ bước 9 (bước đã xong tự bỏ qua tệp có sẵn)
    python -m scripts.chay_thi_nghiem hinh                # chỉ vẽ lại hình của tài liệu

Điều kiện trước: dữ liệu của thí nghiệm 4–5 (data/bao_da_tuyen/{train,val}_k4_5, val5_k4_5, mua5_k*, yagi5_k*) và mô hình
results/bao_da_tuyen/mo_hinh/dm.pt (thí nghiệm 4). Mỗi bước sinh dữ liệu / chấm đều bỏ qua tệp đã có, nên chạy lại an
toàn. Thời gian trên máy 8 luồng: thí nghiệm 6 ≈ 6–8 giờ, thí nghiệm 7 ≈ 5–6 giờ.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

GOC = Path(__file__).resolve().parents[1]
M4, M6, M7 = "results/bao_da_tuyen/mo_hinh", "results/bao_da_tuyen/mo_hinh6", "results/bao_da_tuyen/mo_hinh7"
M8 = "results/bao_da_tuyen/mo_hinh8"
CAU_HINH = {"tt": 2, "luong": 1, "song_song": 2, "thiet_bi": "cpu"}      # số tiến trình sinh / chấm, luồng PyTorch, mô hình huấn luyện cùng lúc
K_MOC = {"4-5": [], "10": ["--buoc-moc", "2"], "20": ["--buoc-moc", "4"]}


def sinh(tap, K="4-5", thay=False, tt=2):
    return (f"sinh dữ liệu {tap} {K} luồng", ["-m", "src.nhieu_luong.tao_du_lieu_dt", "--tap", tap, "--K", K, *K_MOC[K],
                                            *([] if thay else ["--khong-thay"]), "--tien-trinh", str(tt)])


def cham(tap, K, mo_hinh, ten, khoa_tkcb10="", tt=1):
    return (f"chấm {ten} trên {tap} {K} luồng", ["-m", "src.nhieu_luong.cham_lai_dt", "--tap", tap, "--K", K, "--mo-hinh",
                                                mo_hinh, "--ten", ten, "--khoa", ten, "--khoa-tkcb10", khoa_tkcb10,
                                                "--tien-trinh", str(tt)])


def buoc_4():
    """Thí nghiệm 4 (results/bao_da_tuyen/BAO_CAO.md): dữ liệu bão có thầy, huấn luyện DM + softmax, chấm trên Yagi."""
    B = [("chạy thử trên val để chỉnh β, γ của Dijkstra phạt", ["-m", "scripts.chay_thu_da_tuyen", "chay", "--tien-trinh", "3"]),
         sinh("val", thay=True, tt=3), sinh("train", thay=True, tt=3),
         ("huấn luyện DM", ["-m", "src.nhieu_luong.huan_luyen_dt", "dm"]),
         ("huấn luyện softmax", ["-m", "src.nhieu_luong.huan_luyen_dt", "softmax"]),
         ("chọn số bước DM trên val", ["-m", "src.nhieu_luong.huan_luyen_dt", "chon-buoc"])]
    B += [sinh("test", K, thay=True, tt=3) for K in K_MOC]
    B += [(f"chấm DM / softmax trên Yagi {K} luồng", ["-m", "src.nhieu_luong.cham_mo_hinh_dt", "--K", K, "--tien-trinh", "3"])
          for K in K_MOC]
    B.append(("phân tích thí nghiệm 4", ["-m", "scripts.phan_tich_dt"]))
    return B


def buoc_6():
    cu = ["train", "train_k4_5", "val", "val_k4_5", "val5", "val5_k4_5"] + \
         [x for b in ("mua5", "yagi5") for K in ("k4_5", "k10", "k20") for x in (b, f"{b}_{K}")]
    B = [sinh("train6", thay=True), sinh("val6", thay=True),
         ("bổ sung 3 đặc trưng giông quanh UAV cho dữ liệu cũ (169 tệp)", ["-m", "scripts.bo_sung_dac_trung", *cu,
                                                                          "--tien-trinh", "2"]),
         ("huấn luyện DM6 (có đặc trưng mới)", ["-m", "src.nhieu_luong.huan_luyen_dt", "dm", "--thi-nghiem", "6", "--ten",
                                               "dm6", "--them"]),
         ("huấn luyện DM6a (không đặc trưng mới)", ["-m", "src.nhieu_luong.huan_luyen_dt", "dm", "--thi-nghiem", "6",
                                                   "--ten", "dm6a"]),
         ("chọn số bước DM6", ["-m", "src.nhieu_luong.huan_luyen_dt", "chon-buoc", "--thi-nghiem", "6", "--ten", "dm6"]),
         ("chọn số bước DM6a", ["-m", "src.nhieu_luong.huan_luyen_dt", "chon-buoc", "--thi-nghiem", "6", "--ten", "dm6a"]),
         cham("val5", "4-5", M6, "dm6"), cham("val6", "4-5", M6, "dm6"),
         ("chọn τ6 theo mất dịch vụ trên val5 + val6", ["-m", "scripts.phan_tich_6", "chon-tau"])]
    B += [sinh("c6", K) for K in K_MOC]
    for tap in ("c6", "mua5", "yagi5"):
        for K in K_MOC:
            B.append(cham(tap, K, M6, "dm6", "online_tkcb10_6"))
        B.append(cham(tap, "4-5", M6, "dm6a"))
    B += [cham("c6", K, M4, "dm") for K in K_MOC]                        # DM của thí nghiệm 4 trên bộ C mới
    B.append(("phân tích thí nghiệm 6", ["-m", "scripts.phan_tich_6", "phan-tich"]))
    return B


def buoc_7():
    B = []
    for t in ("c7wh", "y7wh", "c7w", "c7h", "y7w", "y7h"):                # 7a: thế giới cực đoan giả lập, seed + 800
        B += [sinh(t, "4-5", tt=1), cham(t, "4-5", M6, "dm6", "online_tkcb10_6")]
    for t in ("c7wh", "y7wh"):
        for K in ("10", "20"):
            B += [sinh(t, K, tt=1), cham(t, K, M6, "dm6", "online_tkcb10_6")]
    B += [sinh("train7s", thay=True), sinh("train7x", thay=True)]       # 7b: dữ liệu cho mốc 200%
    for ten, them in (("dm7_25", ["--ti-le", "0.25"]), ("dm7_50", ["--ti-le", "0.5"]), ("dm7_200", ["--gap-doi"])):
        B.append((f"huấn luyện {ten}", ["-m", "src.nhieu_luong.huan_luyen_dt", "dm", "--thi-nghiem", "7", "--ten", ten,
                                       "--them", *them]))
    B.append(("cố định S = 8, 5 bước cho mọi mô hình 7b", "chon_buoc_7"))
    B += [cham(tap, "4-5", M7, ten) for ten in ("dm7_25", "dm7_50", "dm7_200")
          for tap in ("c6", "mua5", "yagi5", "c7wh", "y7wh")]
    B.append(("phân tích thí nghiệm 7", ["-m", "scripts.phan_tich_7"]))
    return B


def buoc_8():
    """Thí nghiệm 8 (results/bao_da_tuyen/TIEU_CHI_8.md): đánh giá chéo 2 × 2 giữa thế giới A (mô hình hỏng cũ, dữ liệu
    của TN6) và thế giới B (UAV 12 / 15 m/s, hỏng theo khoảng cách tới giông). Tập B dùng đúng cửa sổ + seed của tập A."""
    tt, lu = CAU_HINH["tt"], CAU_HINH["luong"]
    # 0) Thế giới A (dữ liệu + DM6 của TN6). git không lưu phần này (data/bao_da_tuyen, *.pt): máy mới clone về thì tự
    #    sinh lại — cùng seed, cùng mã nên trùng với TN6; máy đã có (hoặc đã giải nén TN8_du_lieu.zip) thì tự bỏ qua.
    B = [sinh(t, thay=True, tt=tt) for t in ("train", "train6", "val", "val6")]
    B += [sinh(t, K, tt=tt) for t in ("c6", "yagi5") for K in K_MOC]
    B.append(("huấn luyện DM6 seed 0 (thế giới A) — chỉ khi chưa có", ["-m", "src.nhieu_luong.huan_luyen_dt", "dm",
                                                                      "--thi-nghiem", "6", "--ten", "dm6", "--them",
                                                                      "--seed", "0", "--luong-cpu", str(lu)]))
    B += [sinh(t, thay=True, tt=tt) for t in ("train_b", "train6_b", "val_b", "val6_b")]  # dữ liệu học thế giới B
    B += [sinh(t, K, tt=tt) for t in ("c6_b", "yagi5_b") for K in K_MOC]                  # dữ liệu kiểm tra thế giới B
    for s in (0, 1, 2):                                                                   # DM8b: học trong thế giới B
        B.append((f"huấn luyện DM8b seed {s} (thế giới B)", ["-m", "src.nhieu_luong.huan_luyen_dt", "dm", "--thi-nghiem",
                                                            "8", "--ten", f"dm8b_s{s}", "--them", "--seed", str(s),
                                                            "--luong-cpu", str(lu)]))
    for s in (1, 2):                                                                      # DM6 thêm 2 seed (thế giới A)
        B.append((f"huấn luyện DM6 seed {s} (thế giới A)", ["-m", "src.nhieu_luong.huan_luyen_dt", "dm", "--thi-nghiem",
                                                           "6", "--ten", f"dm6_s{s}", "--them", "--seed", str(s),
                                                           "--luong-cpu", str(lu)]))
    B.append(("cố định S = 8, 5 bước cho các mô hình TN8 (như DM6)", "chon_buoc_8"))
    mo_hinh = [("dm6", M6), ("dm6_s1", M6), ("dm6_s2", M6), ("dm8b_s0", M8), ("dm8b_s1", M8), ("dm8b_s2", M8)]
    for tap in ("c6", "yagi5", "c6_b", "yagi5_b"):                                       # chấm chéo
        for ten, thu_muc in mo_hinh:                                                      # (DM6 trên A có từ TN6 → bỏ qua)
            B.append(cham(tap, "4-5", thu_muc, ten, "online_tkcb10_6" if ten == "dm6" else "", min(tt, 3)))
        for K in ("10", "20"):                                                            # quy mô lớn: chỉ seed 0
            for ten, thu_muc in (("dm6", M6), ("dm8b_s0", M8)):
                B.append(cham(tap, K, thu_muc, ten, "online_tkcb10_6" if ten == "dm6" else "", 1))
    B.append(("phân tích thí nghiệm 8", ["-m", "scripts.phan_tich_8"]))
    return B


def buoc_9():
    """Thí nghiệm 9 (results/bao_da_tuyen/TIEU_CHI_9.md): overfitting (bậc thang độ lạ, đường học, kiểm định đối kháng),
    hiệu chỉnh so sánh bội, mốc tối ưu ILP, ablation DM. Cần dữ liệu + DM6 của TN6."""
    tt = CAU_HINH["tt"]
    B = []
    for t in ("hoc9", "seed9", "hinh9", "ngay9", "nam9", "bao9"):                    # 6 bậc, có thầy
        B += [sinh(t, thay=True, tt=tt), cham(t, "4-5", M6, "dm6", "online_tkcb10_6", 1)]
    B += [("huấn luyện lại DM6, lưu mọi epoch (dm6r)", ["-m", "src.nhieu_luong.huan_luyen_dt", "dm", "--thi-nghiem", "9",
                                                         "--ten", "dm6r", "--them", "--moi-epoch"]),
          ("huấn luyện softmax cùng dữ liệu DM6 (softmax9)", ["-m", "src.nhieu_luong.huan_luyen_dt", "softmax",
                                                              "--thi-nghiem", "9", "--ten", "softmax9", "--them"]),
          ("B1: hiệu chỉnh so sánh bội TN4 / TN6 / TN7a", ["-m", "scripts.kiem_dinh_boi_9"]),
          ("B2: tối ưu ILP mất dịch vụ", ["-m", "scripts.toi_uu_ilp_9", "--tien-trinh", str(min(tt, 3))]),
          ("B3: ablation DM", ["-m", "scripts.ablation_9", "--tien-trinh", str(min(tt, 3))]),
          ("B3: ablation softmax9", ["-m", "scripts.ablation_9", "--bien-the", "softmax9", "--tien-trinh", str(min(tt, 3))]),
          ("A2: đường học", ["-m", "scripts.duong_hoc_9", "--tien-trinh", str(min(tt, 3))]),
          ("A3: kiểm định đối kháng", ["-m", "scripts.doi_khang_9"]),
          ("phân tích thí nghiệm 9", ["-m", "scripts.phan_tich_9"])]
    return B


def buoc_10():
    """Thí nghiệm 10 (results/bao_da_tuyen/TIEU_CHI_10.md): "học tới nơi" — 4 bước chọn bằng kiểm thử chéo 4 phần trên tập
    học (năng lực, số chuỗi S, thầy, dữ liệu A + B), rồi chấm bộ kiểm tra một lần. Dùng lại val và bộ kiểm tra của TN8 /
    TN9 nếu có, thiếu thì tự sinh. Không cần mô hình cũ: đối chứng DM6r học lại trên cùng tập học."""
    tt, lu, ss = CAU_HINH["tt"], CAU_HINH["luong"], CAU_HINH["song_song"]
    tn10 = lambda *x: ["-m", "scripts.tn10", *x, "--tien-trinh", str(min(tt, 4)), "--song-song", str(ss),
                       "--luong-cpu", str(lu), "--thiet-bi", CAU_HINH["thiet_bi"]]
    # 0) val (chọn checkpoint) + bộ kiểm tra — có rồi thì bỏ qua
    B = [sinh(t, thay=True, tt=tt) for t in ("val", "val6", "val_b", "val6_b")]
    B += [sinh(t, K, tt=tt) for t in ("c6", "yagi5", "c6_b", "yagi5_b") for K in K_MOC]
    B += [sinh(t, thay=True, tt=tt) for t in ("ngay9", "nam9", "bao9")]                # bộ kiểm tra có thầy (TN9)
    # 1) tập học A có thế giới chấm (kiểm thử chéo) → bước 1, 2
    B += [sinh(t, thay=True, tt=tt) for t in ("train10", "train6_10")]
    B += [("bước 1: huấn luyện 4 cấu hình × 4 phần", tn10("huan-luyen", "--buoc", "1")),
          ("bước 1: chọn cấu hình (kiểm thử chéo)", tn10("chon", "--buoc", "1")),
          ("bước 2: chọn số chuỗi S (kiểm thử chéo)", tn10("chon", "--buoc", "2"))]
    # 3) thầy 64 thế giới nhãn
    B += [sinh("val10_t32", thay=True, tt=tt), sinh("val10_t64", thay=True, tt=tt),
          ("bước 3: so thầy 64 với 32 thế giới nhãn", tn10("chon", "--buoc", "3a")),
          ("bước 3: nhãn thầy 64 cho tập học A (nếu qua)", ["-m", "scripts.tn10", "sinh", "--buoc", "3",
                                                            "--tien-trinh", str(tt)]),
          ("bước 3: huấn luyện 4 phần trên nhãn thầy 64 (nếu qua)", tn10("huan-luyen", "--buoc", "3")),
          ("bước 3: chọn (kiểm thử chéo)", tn10("chon", "--buoc", "3b"))]
    # 4) dữ liệu A + B
    B += [sinh(t, thay=True, tt=tt) for t in ("train_b10", "train6_b10")]
    B += [("bước 4: nhãn thầy 64 cho tập học B (nếu bước 3 chọn)", ["-m", "scripts.tn10", "sinh", "--buoc", "4",
                                                                     "--tien-trinh", str(tt)]),
          ("bước 4: huấn luyện 4 phần trên A + B", tn10("huan-luyen", "--buoc", "4")),
          ("bước 4: chọn (kiểm thử chéo A + B)", tn10("chon", "--buoc", "4")),
          ("huấn luyện DM10 và DM6r trên toàn bộ tập học (3 seed)", tn10("huan-luyen", "--buoc", "day")),
          ("chốt DM10, DM6r", tn10("chon", "--buoc", "chot")),
          ("chấm kiểm tra Lai(DM6r), Lai(DM10)", tn10("cham")),
          ("phân tích thí nghiệm 10", ["-m", "scripts.phan_tich_10"])]
    return B


CHON_BUOC_8 = (f"{M8}/chon_buoc.json", f"{M6}/chon_buoc_dm6.json", f"{M6}/chon_buoc_dm6_s1.json",
               f"{M6}/chon_buoc_dm6_s2.json")


def chon_buoc_8():
    for t in CHON_BUOC_8:
        tep = GOC / t
        if tep.exists():                     # giữ tệp có sẵn (chon_buoc_dm6.json của TN6 đã chọn 5 bước trên val)
            continue
        tep.parent.mkdir(parents=True, exist_ok=True)
        tep.write_text(json.dumps({"S": 8, "so_buoc": 5, "quy_tac": "thí nghiệm 8: cố định 5 bước, S = 8 như DM6"},
                                  ensure_ascii=False, indent=1), encoding="utf-8")


def buoc_hinh():
    return [("hình dữ liệu", ["-m", "scripts.truc_quan_du_lieu"]), ("hình kết quả", ["-m", "scripts.hinh_bao_cao_6_7"])]


def chon_buoc_7():
    tep = GOC / M7 / "chon_buoc.json"
    tep.parent.mkdir(parents=True, exist_ok=True)
    tep.write_text(json.dumps({"S": 8, "so_buoc": 5, "quy_tac": "thí nghiệm 7b: cố định 5 bước, S = 8 (không chọn lại)"},
                              ensure_ascii=False, indent=1), encoding="utf-8")


DL = GOC / "data" / "bao_da_tuyen"
SO_CUA_SO = {"train": 59, "val": 16, "test": 13, "train6": 60, "val6": 16, "val5": 16, "train7s": 59, "train7x": 60,
             "train_b": 59, "train6_b": 60, "val_b": 16, "val6_b": 16, "val10_t32": 16, "val10_t64": 16,
             "train10": 59, "train6_10": 60, "train_b10": 59, "train6_b10": 60}
M9 = "results/bao_da_tuyen/mo_hinh9"
THU_MUC_MH = {"4": M4, "6": M6, "7": M7, "8": M8, "9": M9}


def _doi(lenh, ten, mac_dinh=None):
    return lenh[lenh.index(ten) + 1] if ten in lenh else mac_dinh


def _kb(tap, K):
    import glob
    return sorted(glob.glob(str(DL / f"{tap}_k{K.replace('-', '_')}" / "kb_*.pkl")))


def _ban_ghi_dau(tep):
    import pickle
    with open(tep, "rb") as f:
        recs = pickle.load(f)
    return recs[0] if recs else None


def da_xong(lenh):
    """Bước đã có sản phẩm (dữ liệu / mô hình / kết quả chấm) thì bỏ qua — không sinh lại, không huấn luyện lại.
    Phân tích và vẽ hình luôn chạy (nhanh, đọc từ kết quả có sẵn)."""
    if lenh == "chon_buoc_7":
        return (GOC / M7 / "chon_buoc.json").exists()
    if lenh == "chon_buoc_8":
        return all((GOC / t).exists() for t in CHON_BUOC_8)
    mo = lenh[1]
    if mo == "src.nhieu_luong.tao_du_lieu_dt":
        tap = _doi(lenh, "--tap")
        return len(_kb(tap, _doi(lenh, "--K"))) >= SO_CUA_SO.get(tap, 13)
    if mo == "scripts.chay_thu_da_tuyen":
        return (DL / "chay_thu" / ".xong").exists()
    if mo == "src.nhieu_luong.huan_luyen_dt":
        tn, viec = _doi(lenh, "--thi-nghiem", "4"), lenh[2]
        thu_muc = GOC / THU_MUC_MH[tn]
        if viec == "chon-buoc":
            ten = _doi(lenh, "--ten", "dm")
            return (thu_muc / ("chon_buoc.json" if ten == "dm" else f"chon_buoc_{ten}.json")).exists()
        return (thu_muc / f"{_doi(lenh, '--ten', viec)}.xong").exists()
    if mo == "scripts.bo_sung_dac_trung":
        cap = [x for x in lenh[2:lenh.index("--tien-trinh")]][1::2]
        for thu_muc in cap:
            import glob
            teps = sorted(glob.glob(str(DL / thu_muc / "kb_*.pkl")))
            r = _ban_ghi_dau(teps[-1]) if teps else None
            if r is None or "cape_uav" not in r["gon"].dac_trung_canh:
                return False
        return True
    if mo == "scripts.phan_tich_6" and lenh[2] == "chon-tau":
        return (GOC / "results" / "bao_da_tuyen" / "chon_tau_6.json").exists()
    if mo == "scripts.duong_hoc_9":
        return (GOC / "results" / "bao_da_tuyen" / "tn9" / "duong_hoc.json").exists()
    if mo in ("src.nhieu_luong.cham_lai_dt", "src.nhieu_luong.cham_mo_hinh_dt"):
        if mo == "src.nhieu_luong.cham_lai_dt":
            teps, can = _kb(_doi(lenh, "--tap"), _doi(lenh, "--K")), [_doi(lenh, "--khoa") + "_tkcb"]
        else:
            teps, can = _kb("test", _doi(lenh, "--K", "4-5")), ["dm", "softmax"]
        if not teps:
            return False
        for tep in (teps[0], teps[-1]):
            r = _ban_ghi_dau(tep)
            if r is None or any(k not in r.get("ens", {}) for k in can):
                return False
        return True
    return False


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("thi_nghiem", choices=["4", "6", "7", "8", "9", "10", "hinh"])
    ap.add_argument("--chi-in", action="store_true", help="chỉ in lệnh, không chạy")
    ap.add_argument("--tu-buoc", type=int, default=1)
    ap.add_argument("--lam-lai", action="store_true",
                    help="chạy lại cả bước đã có sản phẩm (mặc định bỏ qua: không sinh / huấn luyện / chấm lại)")
    ap.add_argument("--tien-trinh", type=int, default=None,
                    help="số tiến trình sinh dữ liệu (mặc định: nửa số luồng CPU, tối đa 6; mỗi tiến trình ~2,5 GB RAM)")
    ap.add_argument("--luong-cpu", type=int, default=4, help="số luồng PyTorch khi huấn luyện (thí nghiệm 8, 10)")
    ap.add_argument("--song-song", type=int, default=2, help="thí nghiệm 10: số mô hình huấn luyện cùng lúc")
    ap.add_argument("--thiet-bi", default="cpu", choices=["cpu", "auto", "cuda"],
                    help="thí nghiệm 10: huấn luyện trên GPU nếu có (Colab: auto)")
    a = ap.parse_args()
    import os
    CAU_HINH["tt"] = a.tien_trinh or max(1, min(6, (os.cpu_count() or 2) // 2))
    CAU_HINH["luong"] = a.luong_cpu
    CAU_HINH["song_song"] = a.song_song
    CAU_HINH["thiet_bi"] = a.thiet_bi
    B = {"4": buoc_4, "6": buoc_6, "7": buoc_7, "8": buoc_8, "9": buoc_9, "10": buoc_10, "hinh": buoc_hinh}[a.thi_nghiem]()
    bo_qua = 0
    for i, (mo_ta, lenh) in enumerate(B, 1):
        if i < a.tu_buoc:
            continue
        hien = lenh if isinstance(lenh, str) else "python " + " ".join(f'"{x}"' if x == "" else x for x in lenh)
        if not a.lam_lai and da_xong(lenh):
            print(f"[{i}/{len(B)}] {mo_ta} — đã có, bỏ qua", flush=True)
            bo_qua += 1
            continue
        print(f"[{i}/{len(B)}] {mo_ta}\n    {hien}", flush=True)
        if a.chi_in:
            continue
        if lenh == "chon_buoc_7":
            chon_buoc_7()
        elif lenh == "chon_buoc_8":
            chon_buoc_8()
        else:
            subprocess.run([sys.executable, "-W", "ignore", *lenh], cwd=GOC, check=True)
    print(f"xong — {bo_qua} bước đã có sẵn được bỏ qua", flush=True)
    if not a.chi_in and a.thi_nghiem in ("4", "6"):
        in_ket_qua(a.thi_nghiem)


TEP_KQ = {"4": ("ket_qua.json", "KET_QUA.md", "hinh_ket_qua.png", "BAO_CAO.md"),
          "6": ("ket_qua_6.json", "KET_QUA_6.md", "hinh_6.png", "BAO_CAO_6.md")}


def in_ket_qua(tn):
    """In bảng tóm tắt kết quả ra màn hình và chỉ chỗ các tệp kết quả."""
    kq = GOC / "results" / "bao_da_tuyen"
    r = json.loads((kq / TEP_KQ[tn][0]).read_text(encoding="utf-8"))
    o = lambda tc, k4: f"{'THẮNG' if tc.get('thang') else 'không':>5} {tc[k4][0]:+5.2f}"
    print("\n" + "=" * 78)
    if tn == "4":
        print("KẾT QUẢ THÍ NGHIỆM 4 — DM so với Dijkstra phạt trên bão Yagi (ô: thắng? | Δ mất dịch vụ, điểm %)")
        print(f"{'':10}" + "".join(f"{m:>16}" for m in ("tải nhẹ", "tải vừa", "tải nặng")))
        tong = 0
        for K, nhan in (("k4_5", "4–5 luồng"), ("k10", "10 luồng"), ("k20", "20 luồng")):
            hang = [r[K][m]["tieu_chi"] for m in ("nhe", "vua", "nang")]
            tong += sum(bool(tc.get("thang")) for tc in hang)
            print(f"{nhan:10}" + "".join(f"{o(tc, 'C4_mat_dv_dm_tru_dphat_diem_%'):>16}" for tc in hang))
        print(f"Tổng: DM thắng {tong}/9 ô")
    else:
        print("KẾT QUẢ THÍ NGHIỆM 6 — Lai6 so với Dijkstra phạt (ô: thắng? | Δ mất dịch vụ, điểm %)")
        print(f"{'':26}" + "".join(f"{d:>16}" for d in ("dải thấp", "dải giữa", "dải cao")))
        for bo, ten in (("c6", "C ngày giông"), ("mua5", "B mưa lớn"), ("yagi5", "A Yagi")):
            tong = 0
            for K, nhan in (("k4_5", "4–5"), ("k10", "10"), ("k20", "20")):
                hang = [r[f"{bo}_{K}"][d]["tieu_chi"] for d in ("thap", "giua", "cao")]
                tong += sum(bool(tc.get("thang")) for tc in hang)
                print(f"{ten + ' · ' + nhan + ' luồng':26}" + "".join(f"{o(tc, 'C4_mat_dv'):>16}" for tc in hang))
            print(f"{'':26}→ {ten}: Lai6 thắng {tong}/9 ô")
    print("Tệp kết quả:")
    for t in TEP_KQ[tn]:
        print(f"  {kq / t}")
    print("=" * 78)


if __name__ == "__main__":
    main()
