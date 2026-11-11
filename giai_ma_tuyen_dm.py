"""
giai_ma_tuyen_dm.py — BƯỚC 2/4: đổi chỉ số cạnh (U) thành cặp nút thật.

CHỈ ĐỌC: `duong_di_dm.json` (do xuat_tuyen_dm.py tạo) + `dataset/oracle_dataset_v5.npz`.
Ghi ra `duong_di_nut.json` — dạng cặp nút để `ban_do_3d.py` vẽ được.

Cách giải mã: `gon.U` là chỉ số vào `np.flatnonzero(W[t] > 0)` của mạng gốc
`MangGoc.W` (600 snapshot), nên cạnh e ứng với cặp (eu[e], ev[e]).

BƯỚC 2/4 của quy trình — xem docstring `xuat_tuyen_dm.py` để biết đủ 4 bước.

    python giai_ma_tuyen_dm.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

GOC = Path(r"E:\PBL4")
if str(GOC) not in sys.path:
    sys.path.insert(0, str(GOC))


def main() -> None:
    from src.ngau_nhien.kich_ban import MangGoc
    from src.ngau_nhien.thoi_tiet import ThoiTiet

    raw = np.load(GOC / "dataset" / "oracle_dataset_v5.npz", allow_pickle=False)
    g = MangGoc(raw["W_snapshots"], raw["node_kind"], raw["node_order"],
                ThoiTiet(str(GOC / "data" / "thoi_tiet_vn_mr.npz")))

    dm = json.loads((GOC / "duong_di_dm.json").read_text(encoding="utf-8"))
    print(f"đọc {len(dm)} bản ghi tuyến DM")

    # cache flatnonzero theo t
    cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def cap_nut(t: int, canh: list[int]) -> list[tuple[int, int]]:
        if t not in cache:
            cache[t] = np.nonzero(g.W[t] > 0)
        eu, ev = cache[t]
        return [(int(eu[e]), int(ev[e])) for e in canh if 0 <= e < len(eu)]

    ra: dict[str, dict] = {}
    so_loi = 0
    for khoa, v in dm.items():
        t = int(v["t"])
        try:
            ra[khoa] = {
                "t": t,
                "K": v["K"],
                "nguon": v["nguon"],
                "bao": str(v.get("bao", "")),
                # mỗi luồng: danh sách cặp nút (u,v)
                "dm": [[list(p) for p in cap_nut(t, luong)] for luong in v["dm"]],
                "thay": [[list(p) for p in cap_nut(t, luong)] for luong in v["thay"]],
            }
        except Exception:
            so_loi += 1

    out = GOC / "duong_di_nut.json"
    out.write_text(json.dumps(ra, separators=(",", ":")), encoding="utf-8")
    print(f"đã ghi: {out} ({out.stat().st_size/1e6:.2f} MB, {len(ra)} bản ghi, lỗi {so_loi})")

    # kiểm tra 1 mẫu
    k0 = next(iter(ra))
    print(f"\nví dụ {k0}:")
    print(f"  DM  luồng 0: {ra[k0]['dm'][0]}")
    print(f"  thầy luồng 0: {ra[k0]['thay'][0]}")


if __name__ == "__main__":
    main()
