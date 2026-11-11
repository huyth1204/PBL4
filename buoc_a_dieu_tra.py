"""BƯỚC A: điều tra cấu trúc tuyến trong cap_nut.

3 câu hỏi:
 (a) mỗi phương án có gồm đoạn nguồn->nút truy cập->đích không, hay chỉ lõi vệ tinh?
 (b) luồng k đi từ nút nào đến nút nào?
 (c) vì sao `nguon` có 3 tên mà K = 4-5?
"""
import pickle
import sys
from pathlib import Path

import numpy as np

GOC = Path(r"E:\PBL4")
sys.path.insert(0, str(GOC))

raw = np.load(GOC / "dataset" / "oracle_dataset_v5.npz", allow_pickle=False)
names = [str(x) for x in raw["node_order"]]
kind = np.asarray(raw["node_kind"]).astype(str)
layer = np.asarray(raw["node_layer"]).astype(str)

p = GOC / "data" / "bao_da_tuyen" / "soulik_2024_k4_5" / "kb_0000.pkl"
recs = pickle.load(open(p, "rb"))
r = recs[0]
K = r["K"]
cap = r["cap_nut"]
nguon = r["nguon"]

print("=" * 78)
print("THÔNG TIN CƠ BẢN")
print("=" * 78)
print(f"  K = {K}")
print(f"  nguon ({len(nguon)} tên): {nguon}")
print(f"  cap_nut: {len(cap)} luồng, mỗi luồng {len(cap[0])} phương án")
print(f"  mỗi phương án có {len(cap[0][0])} tuyến (chính + dự phòng)")
print(f"  số nút mỗi tuyến: {[len(t) for t in cap[0][0]]}")

print("\n" + "=" * 78)
print("(b) LUỒNG k ĐI TỪ ĐÂU ĐẾN ĐÂU?")
print("=" * 78)
for k in range(K):
    tuyen = cap[k][0][0]                    # phương án 0, tuyến chính
    print(f"  luồng {k}: {len(tuyen)} nút")
    print(f"    {' -> '.join(f'{names[n]}({kind[n]})' for n in tuyen)}")
    print(f"    nút: {tuyen}")

print("\n" + "=" * 78)
print("(a) CÓ GỒM ĐOẠN NGUỒN -> ĐÍCH KHÔNG?")
print("=" * 78)
idx_nguon = [names.index(n) for n in nguon]
print(f"  index nguồn: {dict(zip(nguon, idx_nguon))}")

tong_tuyen = 0
co_nguon = 0
co_dich_ground = 0
for k in range(K):
    for a in range(min(30, len(cap[k]))):
        for tuyen in cap[k][a]:
            tong_tuyen += 1
            if tuyen[0] in idx_nguon or tuyen[-1] in idx_nguon:
                co_nguon += 1
            if kind[tuyen[0]] in ("gateway", "user", "uav") or \
               kind[tuyen[-1]] in ("gateway", "user", "uav"):
                co_dich_ground += 1
print(f"  mẫu {tong_tuyen} tuyến (30 phương án đầu mỗi luồng):")
print(f"    có nút nguồn ở đầu/cuối: {co_nguon}")
print(f"    có nút ground/uav ở đầu/cuối: {co_dich_ground}")

print("\n" + "=" * 78)
print("(c) VÌ SAO nguon CÓ 3 TÊN MÀ K=4-5?")
print("=" * 78)
print(f"  'nguon' trong .pkl = {nguon}")
print(f"  Đây là TẬP nguồn (không phải mỗi luồng 1 nguồn)")

# Kiem tra: co trung khong
from collections import Counter
dem = Counter()
for k in range(K):
    tuyen = cap[k][0][0]
    dem[tuyen[0]] += 1
print(f"\n  nút đầu của từng luồng (phương án 0):")
for n, c in dem.items():
    print(f"    {names[n]} ({kind[n]}): {c} luồng")
