"""Thí nghiệm 4/4 — TN5: GNN + DM so với GNN + softmax, định tuyến đơn luồng khi bão Yagi, mạng v5. Chạy từ đầu.

    python chay_tu_dau/4_tn5_dm_vs_softmax.py                # đầy đủ: 8 mô hình, khoảng 9–10 giờ trên 2 lõi CPU
    python chay_tu_dau/4_tn5_dm_vs_softmax.py --threads 4    # dùng 4 luồng CPU cho huấn luyện (nhanh hơn)
    python chay_tu_dau/4_tn5_dm_vs_softmax.py --thu          # chạy thử: dữ liệu nhỏ, 1 epoch, 1 seed — khoảng 15–25 phút

Dừng giữa chừng rồi chạy lại thì tiếp tục từ mô hình chưa xong (mô hình xong khi có tệp *_results.json).
Các bước, cho mỗi thiết lập E1 (mức "thường", tuổi 0, giữ tuyến Δ 9) và E2 ("khắc nghiệt", tuổi 9, Δ 3):
  1) Sinh dữ liệu: 12 cửa sổ bão 6 giờ cho train, 3 cho val, test = 16 kịch bản Yagi   (src/ngau_nhien/tao_du_lieu.py)
  2) Huấn luyện GNN + DM (diffusion2, x0ce, 2 bước DDIM) và GNN + softmax, mỗi loại 2 seed, 12 epoch   (src/ai/train_gnn.py)
  3) Chọn guidance của DM trên val trong {1, 2, 3}; chấm trên Yagi + KTC bootstrap   (src/ngau_nhien/cham_bao.py)
Tiêu chí ghi trước: results/bao_v5/TIEU_CHI_5.md. Kết quả: ket_qua_tu_chay/tn5/.
"""
import argparse

from _chung import GOC, bang, buoc, doc, kiem_dau_vao, ktc, so, thu_muc, tieu_de

ap = argparse.ArgumentParser()
ap.add_argument("--thu", action="store_true", help="chạy thử nhỏ để kiểm tra máy")
ap.add_argument("--threads", type=int, default=2, help="số luồng CPU khi huấn luyện (kết quả gốc dùng 2)")
a = ap.parse_args()
RAW, SPLIT, TT = "dataset/oracle_dataset_v5.npz", "dataset/v5/split_info.json", "data/thoi_tiet_vn_mr.npz"
kiem_dau_vao(RAW, SPLIT, TT)
D = thu_muc("tn5", a.thu)
NK = D / "nhat_ky"
tieu_de("TN5 — GNN + DM và GNN + softmax khi bão Yagi (đơn luồng, mạng v5)",
        "DM có định tuyến tốt hơn softmax, và cả hai có tốt hơn Dijkstra + lịch quỹ đạo không? (G nhỏ là tốt)", a.thu)

E = {"E1": ["--muc", "thường", "--tuoi", "0", "--delta", "9", "--buoc", "2,2,2", "--seed", "51"],
     "E2": ["--muc", "khắc nghiệt", "--tuoi", "9", "--delta", "3", "--buoc", "2,2,1", "--seed", "52"]}
SEED = (0,) if a.thu else (0, 1)
EPOCH = "1" if a.thu else "12"
KB = "2,1,2" if a.thu else "12,3,16"

for e, ts in E.items():
    buoc(f"{e}: sinh dữ liệu", ["src.ngau_nhien.tao_du_lieu", "--raw", RAW, "--split", SPLIT, "--thoi-tiet", TT,
                                 "--dau-vao", "quy_dao", "--chon-gio", "bao", "--kich-ban", KB, "--luu-val", *ts,
                                 "--out-dir", D / f"du_lieu_{e}"], NK / f"{e}_sinh.log", xong=D / f"du_lieu_{e}" / "test.npz")
MH = D / "mo_hinh"
MH.mkdir(exist_ok=True)
MHR = MH.relative_to(GOC).as_posix()  # đường dẫn tương đối: tránh dấu ':' của ổ đĩa Windows trong chuỗi mô hình
for e in E:
    for s in SEED:
        for loai, dau in (("dm", ["--head", "diffusion2", "--param", "x0ce", "--ddim", "2"]), ("sm", ["--head", "classifier"])):
            tag = f"{e}_{loai}_s{s}"
            buoc(f"Huấn luyện {tag}", ["src.ai.train_gnn", "--data-dir", D / f"du_lieu_{e}", *dau, "--layers", "4",
                                        "--epochs", EPOCH, "--seed", s, "--tag", tag, "--out-dir", MH, "--threads", a.threads],
                 NK / f"hl_{tag}.log", xong=MH / f"{tag}_results.json")

KQ = {}
for e in E:
    dm = lambda w: ",".join(f"{MHR}/{e}_dm_s{s}.pt:2:{w}" for s in SEED)
    buoc(f"{e}: chọn guidance trên val", ["src.ngau_nhien.cham_bao", "--test", D / f"du_lieu_{e}" / "val.npz",
                                          "--out", D / f"chon_w_{e}.json", "--mo-hinh", *(f"w{w}={dm(w)}" for w in (1, 2, 3))],
         NK / f"{e}_chon_w.log", xong=D / f"chon_w_{e}.json")
    pp = doc(D / f"chon_w_{e}.json")["phuong_phap"]
    w = min((1, 2, 3), key=lambda k: pp[f"w{k}"]["G_%"])
    print(f"{e}: guidance chọn trên val = {w}", flush=True)
    sm = ",".join(f"{MHR}/{e}_sm_s{s}.pt:1:1" for s in SEED)
    buoc(f"{e}: chấm trên Yagi", ["src.ngau_nhien.cham_bao", "--test", D / f"du_lieu_{e}" / "test.npz", "--out", D / f"cham_{e}.json",
                                  "--mo-hinh", f"GNN + DM={dm(w)}", f"GNN + softmax={sm}"], NK / f"{e}_cham.log", xong=D / f"cham_{e}.json")
    KQ[e] = (w, doc(D / f"cham_{e}.json"))

# ---- in kết quả
PP = ("Dijkstra trên W đo được", "Dijkstra + lịch quỹ đạo", "GNN + softmax", "GNN + DM")
TC = (("K1", "GNN + DM − GNN + softmax"), ("K2", "GNN + DM − Dijkstra + lịch quỹ đạo"), ("K3", "GNN + softmax − Dijkstra + lịch quỹ đạo"))
TEN = {"E1": "E1: thông tin mới (tuổi 0, giữ tuyến 3 phút)", "E2": "E2: thông tin cũ 3 phút (tuổi 9, giữ tuyến 1 phút)"}
print(f"\nKẾT QUẢ trên {KB.split(',')[2]} kịch bản Yagi (G = % chi phí vượt hậu kiểm, nhỏ là tốt)")
for e, (w, d) in KQ.items():
    print(f"{TEN[e]} — {d['so_truy_van']} truy vấn, guidance DM = {w}")
    bang(["Phương pháp", "G (%)", "Đứt (%)", "Tới đích (%)"],
         [[k, so(d["phuong_phap"][k]["G_%"], 1), so(d["phuong_phap"][k]["dut_%"], 2), so(d["phuong_phap"][k]["toi_dich_%"], 2)] for k in PP])
    hang = []
    for ma, k in TC:
        h = d["hieu"][k]
        lo, hi = h["ktc95"]
        hang.append([ma, k, ktc(h["diem_%"], h["ktc95"], 1), "ĐẠT" if hi < 0 else ("ngược chiều" if lo > 0 else "trượt")])
    bang(["Tiêu chí", "So sánh", "Hiệu G, điểm % [KTC 95%]", "Kết quả"], hang)

if not a.thu:
    print("So với kết quả gốc results/bao_v5/sua_du_bao/ (mô hình gốc huấn luyện trên dữ liệu TRƯỚC khi nhóm sửa dự báo quỹ đạo,")
    print("chấm trên dữ liệu đã sửa). Mã hiện tại sinh dữ liệu đã sửa ngay từ đầu, nên E1 (tuổi 0, dữ liệu không đổi) nên rất gần;")
    print("E2 được huấn luyện lại trên dữ liệu khác chút ít, số liệu sẽ lệch — xem kết luận có đổi không.")
    hang = []
    for e in KQ:
        cu = GOC / "results/bao_v5/sua_du_bao" / f"cham_{e}.json"
        if cu.exists():
            c = doc(cu)["phuong_phap"]
            hang += [[f"{e} {k}", so(KQ[e][1]["phuong_phap"][k]["G_%"], 1), so(c[k]["G_%"], 1)] for k in PP[1:]]
    bang(["", "G mới (%)", "G gốc (%)"], hang)
