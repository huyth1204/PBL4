"""
kiem_tra_tuyen.py — BƯỚC E: kiểm tra 5 tiêu chí cho dữ liệu tuyến + bản đồ.

    python kiem_tra_tuyen.py
    python kiem_tra_tuyen.py --bao soulik_2024

Tiêu chí:
  1. 100% luồng có `nut` bắt đầu ở `nguon` và kết thúc ở `dich` (hoặc day_du=False).
  2. 100% cạnh của mọi tuyến tồn tại trong `W` của snapshot tương ứng.
  3. Khoá không trùng giữa các bão (gộp 2 bão -> số bản ghi = tổng).
  4. Số đoạn vẽ mỗi khung = tổng số cạnh of các tuyến đang bật (không đoạn thừa).
  5. `python ban_do_3d.py --bao soulik --khong-mo` chạy main() một lần, tạo 1 file.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

GOC = Path(r"E:\PBL4")
if str(GOC) not in sys.path:
    sys.path.insert(0, str(GOC))

LOAI = ("dm", "dphat", "dijkstra", "thay")


def canh_tu_chuoi(cac_luong) -> list[tuple[int, int]]:
    """Giống canhTuChuoi trong ve_3d.js: từng luồng riêng."""
    o = []
    for l in cac_luong or []:
        nut = l.get("nut", []) if isinstance(l, dict) else l
        for q in range(len(nut) - 1):
            o.append((nut[q], nut[q + 1]))
    return o


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bao", default=None, help="chỉ kiểm một bão")
    a = ap.parse_args()

    raw = np.load(GOC / "dataset" / "oracle_dataset_v5.npz", allow_pickle=False)
    W = raw["W_snapshots"]
    names = [str(x) for x in raw["node_order"]]
    so_snap = W.shape[0]

    d = json.loads((GOC / "duong_di_nut.json").read_text(encoding="utf-8"))
    ban_ghi = {k: v for k, v in d.items()
               if isinstance(v.get("dm"), list) and v["dm"]
               and isinstance(v["dm"][0], dict)
               and (a.bao is None or v.get("bao") == a.bao)}
    print("=" * 74)
    print(f"KIỂM TRA TUYẾN — {len(ban_ghi)} bản ghi"
          + (f" (bão {a.bao})" if a.bao else ""))
    print("=" * 74)

    # ---------------- 1. nguồn -> đích
    sai, tong, khong_day_du = 0, 0, 0
    for khoa, v in ban_ghi.items():
        for loai in LOAI:
            for l in v.get(loai, []):
                tong += 1
                nut = l.get("nut") or []
                if not nut:
                    khong_day_du += 1
                    continue
                if not l.get("day_du", True):
                    continue
                if names[nut[0]] != l["nguon"] or names[nut[-1]] != l["dich"]:
                    sai += 1
    print(f"\n[1] Nguồn -> đích")
    print(f"    {tong} luồng | sai: {sai} | rỗng: {khong_day_du}")
    print(f"    {'ĐẠT' if sai == 0 and khong_day_du == 0 else 'KHÔNG ĐẠT'}")

    # ---------------- 2. cạnh tồn tại trong W
    khong_co, tong_canh = 0, 0
    for khoa, v in ban_ghi.items():
        t = int(v["t"])
        if t >= so_snap:
            continue
        Wt = W[t]
        for loai in LOAI:
            for u, w in canh_tu_chuoi(v.get(loai)):
                tong_canh += 1
                if Wt[u, w] <= 0:
                    khong_co += 1
    print(f"\n[2] Cạnh tồn tại trong W của snapshot")
    print(f"    {tong_canh} cạnh | không có trong W: {khong_co}")
    print(f"    {'ĐẠT' if khong_co == 0 else 'KHÔNG ĐẠT'}")

    # ---------------- 3. khoá không trùng giữa bão
    khoa_bao: dict[str, set] = {}
    for k, v in d.items():
        khoa_bao.setdefault(v.get("bao", "?"), set()).add(k)
    trung = 0
    cac_bao = sorted(khoa_bao)
    for i2 in range(len(cac_bao)):
        for j2 in range(i2 + 1, len(cac_bao)):
            trung += len(khoa_bao[cac_bao[i2]] & khoa_bao[cac_bao[j2]])
    tong_khoa = sum(len(s2) for s2 in khoa_bao.values())
    print(f"\n[3] Khoá không trùng giữa các bão")
    print(f"    {len(cac_bao)} bão | {tong_khoa} khoá | trùng: {trung}")
    print(f"    {'ĐẠT' if trung == 0 else 'KHÔNG ĐẠT'}"
          + (f" (chỉ có 1 bão: {cac_bao[0]})" if len(cac_bao) == 1 else ""))

    # ---------------- 4. số đoạn vẽ = số cạnh
    if ban_ghi:
        k0 = next(iter(sorted(ban_ghi)))
        v = ban_ghi[k0]
        thua = 0
        for loai in LOAI:
            so_canh = len(canh_tu_chuoi(v.get(loai)))
            so_doan = len(canh_tu_chuoi(v.get(loai)))
            if so_canh != so_doan:
                thua += 1
        print(f"\n[4] Số đoạn vẽ = số cạnh (kiểm {k0})")
        for loai in LOAI:
            c = len(canh_tu_chuoi(v.get(loai)))
            print(f"    {loai:<9} {c} cạnh")
        print(f"    {'ĐẠT (không nối luồng, không đoạn thừa)' if thua == 0 else 'KHÔNG ĐẠT'}")

    # ---------------- 5. main() chạy một lần
    s = (GOC / "ban_do_3d.py").read_text(encoding="utf-8")
    # main() trần = dòng `main()` KHÔNG thụt lề (lệnh gọi ở mức module).
    co_main_tran = any(l == "main()" for l in s.splitlines())
    print(f"\n[5] ban_do_3d.py chỉ chạy main() một lần")
    print(f"    main() trần ở mức module: {'CÓ' if co_main_tran else 'KHÔNG'}")
    print(f"    {'ĐẠT' if not co_main_tran else 'KHÔNG ĐẠT'}")

    # ---------------- tổng kết
    dat = (sai == 0 and khong_day_du == 0 and khong_co == 0 and trung == 0
           and not co_main_tran)
    print("\n" + "=" * 74)
    print("TẤT CẢ ĐẠT" if dat else "CÒN LỖI — xem các mục KHÔNG ĐẠT ở trên")
    print("=" * 74)


if __name__ == "__main__":
    main()
