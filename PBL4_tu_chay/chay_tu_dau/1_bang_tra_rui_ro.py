"""Thí nghiệm 1/4 — bảng tra rủi ro bão 3.000 ô (storm-lookup-v1 của nhóm), chạy từ đầu.

    python chay_tu_dau/1_bang_tra_rui_ro.py          # đầy đủ, khoảng 15–25 phút
    python chay_tu_dau/1_bang_tra_rui_ro.py --thu    # chạy thử (3 cơn, lấy mẫu thưa), vài phút

Các bước:
  1) Cắt 10 cơn bão 2024–2025 ngoài Yagi từ thời tiết 74 điểm   (src/ngau_nhien/bao_manh.py)
  2) Mô phỏng từng cơn, đếm liên kết sống/đứt vào 3.000 ô, học bảng; val = Prapiroon, Trami, Kalmaegi
                                                              (src/ngau_nhien/ma_tran_rui_ro.py)
  3) Chấm bảng đã đóng băng trên một cửa sổ Yagi               (src/ngau_nhien/danh_gia_bang_rui_ro.py)
Đầu vào: data/thoi_tiet_vn_mr.npz, dataset/oracle_dataset_v5.npz, dataset/v5/split_info.json.
Kết quả: ket_qua_tu_chay/bang_tra/ (bảng dùng tiếp cho 3_tn6_4_uav_cuu_ho.py). Tài liệu: dataset/BANG_TRA_RUI_RO.md.
"""
import argparse
import json

from _chung import GOC, bang, buoc, doc, kiem_dau_vao, so, thu_muc, tieu_de

ap = argparse.ArgumentParser()
ap.add_argument("--thu", action="store_true", help="chạy thử nhanh để kiểm tra máy")
a = ap.parse_args()
kiem_dau_vao("data/thoi_tiet_vn_mr.npz", "dataset/oracle_dataset_v5.npz", "dataset/v5/split_info.json")
D = thu_muc("bang_tra", a.thu)
NK = D / "nhat_ky"
tieu_de("BẢNG TRA RỦI RO BÃO 3.000 Ô",
        "Bảng tra theo băng tần × mưa × xu hướng × gió × giông có dự báo xác suất đứt liên kết tốt hơn hằng số không?", a.thu)

buoc("1) Cắt 10 cơn bão", ["src.ngau_nhien.bao_manh", "--out", D / "bao-manh"], NK / "1_bao_manh.log",
     xong=D / "bao-manh" / "danh_muc.json", don_neu_do_dang=D / "bao-manh")
DM = D / "bao-manh" / "danh_muc.json"
if a.thu:  # chạy thử: chỉ 2 cơn train nhỏ + 1 cơn val, lấy mẫu thưa
    dm = doc(DM)
    dm["bao"] = [b for b in dm["bao"] if b["ten"] in ("matmo_2025", "nongfa_2025", "kalmaegi_2025")]
    DM = D / "bao-manh" / "danh_muc_thu.json"
    json.dump(dm, open(DM, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
buoc("2) Đếm và học bảng tra", ["src.ngau_nhien.ma_tran_rui_ro", "--catalog", DM, "--out", D / "bang-tra-bao",
                                *(["--stride", "30", "--val", "kalmaegi_2025"] if a.thu else [])],
     NK / "2_ma_tran_rui_ro.log", xong=D / "bang-tra-bao" / "meta.json", don_neu_do_dang=D / "bang-tra-bao")
buoc("3) Chấm trên Yagi", ["src.ngau_nhien.danh_gia_bang_rui_ro", "--table-dir", D / "bang-tra-bao",
                           "--out", D / "danh-gia-yagi"], NK / "3_danh_gia_yagi.log",
     xong=D / "danh-gia-yagi" / "evaluation.json", don_neu_do_dang=D / "danh-gia-yagi")

# ---- in kết quả
V = doc(D / "bang-tra-bao" / "validation.json")
Y = doc(D / "danh-gia-yagi" / "evaluation.json")
M = doc(D / "bang-tra-bao" / "meta.json")
print("\nKẾT QUẢ (Brier và log-loss: nhỏ là tốt; 'hằng số' = một xác suất sống chung học từ train)")
hang = [[ten, f"{v['lookup']['samples']:,}".replace(",", "."), so(v["lookup"]["brier"] * 1000, 3), so(v["constant_train_baseline"]["brier"] * 1000, 3),
         so(v["lookup"]["log_loss"] * 1000, 2), so(v["constant_train_baseline"]["log_loss"] * 1000, 2)]
        for ten, v in [*V.items(), ("Yagi (giữ riêng)", Y)]]
bang(["Tập", "Số mẫu", "Brier bảng ×1000", "Brier hằng ×1000", "Log-loss bảng ×1000", "Log-loss hằng ×1000"], hang)
print("Theo băng tần trên Yagi (Brier ×1000):",
      ", ".join(f"{k} {so(v['brier'] * 1000, 2)}" for k, v in Y["by_band"].items()))
tot = all(v["lookup"]["brier"] < v["constant_train_baseline"]["brier"] for v in [*V.values(), Y])
print("Kết luận:", "bảng tra tốt hơn hằng số ở mọi tập." if tot else "bảng tra KHÔNG tốt hơn hằng số ở mọi tập.")
print("Lưu ý: 90% mẫu là cạnh không chịu mưa nên điểm gộp dễ đẹp; tín hiệu chính là băng tần × mưa.\n")

# ---- so với bản gốc dùng trong TN6+4
goc = GOC / "results/cuu_ho/dau_vao/bang-tra-bao/meta.json"
if not a.thu and goc.exists():
    khop = doc(goc)["table_sha256"] == M["table_sha256"]
    print("So với bảng gốc (results/cuu_ho/dau_vao/bang-tra-bao):",
          "KHỚP tuyệt đối (cùng mã băm)." if khop else "KHÁC mã băm — kiểm tra phiên bản thư viện/dữ liệu.")
    print("Số liệu TV2 tái lập 27/10: val gộp Brier 1,552 (hằng 1,947); Yagi 1,296 (hằng 1,564).")
print(f"Bảng tra để dùng cho TN6+4: {(D / 'bang-tra-bao' / 'bang_tra.npz').relative_to(GOC)}")
