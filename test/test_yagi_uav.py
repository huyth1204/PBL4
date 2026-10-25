"""
test_yagi_uav.py — Kiểm chứng luật gió giật UAV khi bão Yagi.

BỐI CẢNH
Trong bão Yagi (9/2024) gió giật 10 m vượt 15 m/s ở gần như MỌI điểm trong
nhiều giờ liền. Luật cũ gộp "gió mạnh" vào cùng một cờ với "mất nút":
    tat[:, uav] |= gust[:, s] > 15
nên cả tầng UAV biến mất suốt cơn bão. Thực tế UAV chỉ HẠ CÁNH an toàn rồi
cất cánh lại — đây là trạng thái chờ, khác hẳn nút hỏng.

THAY ĐỔI
Thêm cờ `ha_canh` (hạ cánh an toàn) là trạng thái RIÊNG, song song với `tat`.
`tat` vẫn đặt để giữ nguyên hành vi cắt cạnh (hồi quy an toàn). Nhờ vậy model
phân biệt được "UAV chờ gió lặng" với "nút hỏng".

Chạy:  venv\\Scripts\\python.exe test\\test_yagi_uav.py  [đường_dẫn oracle_dataset_v2.npz]
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

GOC = Path(__file__).resolve().parents[1]
if str(GOC) not in sys.path:
    sys.path.insert(0, str(GOC))

from src.ngau_nhien import thoi_tiet  # noqa: E402
from src.ngau_nhien.kich_ban import MangGoc, ThamSo, sinh_kich_ban  # noqa: E402
from src.ngau_nhien.thoi_tiet import ThoiTiet  # noqa: E402

RAW = os.environ.get("PBL4_RAW") or str(GOC / "dataset" / "oracle_dataset_v2.npz")
TMP = Path(tempfile.mkdtemp())
TAT = dict(mua=False, xung_dot=False, tai=False)     # chỉ bật nguồn đang xét
GIO_GIAT_BAO = 32.0                                   # m/s — mức điển hình của Yagi


def _thoi_tiet_bat_thuong(code=None, gust=None, H: int = 48):
    """Thời tiết tổng hợp: mọi giờ, mọi điểm cùng một mã WMO và mức gió giật."""
    diem = thoi_tiet.cac_diem()
    S = len(diem)
    f = TMP / f"tt_yagi_{np.random.randint(1e9)}.npz"
    np.savez(f, site_names=np.array([d[0] for d in diem]), lat=np.array([d[1] for d in diem]),
             lon=np.array([d[2] for d in diem]),
             time_utc=np.array([f"2024-09-07T{h % 24:02d}:00" for h in range(H)]),
             precip=np.zeros((H, S), np.float32),
             code=np.full((H, S), 0 if code is None else code, np.int16),
             gust=np.full((H, S), 3.0 if gust is None else gust, np.float32),
             nguon=np.array("test-yagi"))
    return ThoiTiet(str(f))


_RAW = np.load(RAW)
W0 = _RAW["W_snapshots"][:48]


def _mang(tt: ThoiTiet) -> MangGoc:
    return MangGoc(W0, _RAW["node_kind"], _RAW["node_order"], tt)


def _uav(g: MangGoc) -> list[int]:
    return [i for i, k in enumerate(g.kind) if k == "uav"]


def kiem_tra() -> list[tuple[str, bool, str]]:
    """Hai tình huống: giông (gió yếu) và bão Yagi (gió giật mạnh), rồi gió lặng lại."""
    kq: list[tuple[str, bool, str]] = []

    # 1) Giông, gió yếu: UAV hạ cánh — cả tat lẫn ha_canh đều bật.
    tt = _thoi_tiet_bat_thuong(code=95, gust=3.0)
    g = _mang(tt)
    uav = _uav(g)
    _, q = sinh_kich_ban(g, tt, 0, 1, ThamSo(**TAT, giong=True))
    kq.append(("Giông: UAV vào 'tat' (hợp đồng cũ giữ nguyên)",
               bool(q["tat"][:, uav].all()), f"{len(uav)} UAV"))
    kq.append(("Giông: UAV vào 'ha_canh' (trạng thái mới có mặt)",
               bool(q["ha_canh"][:, uav].all()), f"{len(uav)} UAV"))

    # 2) Bão Yagi: gió giật 32 m/s, không có giông -> UAV hạ cánh vì gió.
    tt_b = _thoi_tiet_bat_thuong(code=0, gust=GIO_GIAT_BAO)
    g_b = _mang(tt_b)
    uav_b = _uav(g_b)
    _, q_b = sinh_kich_ban(g_b, tt_b, 0, 2, ThamSo(**TAT, giong=True))
    kq.append((f"Yagi {GIO_GIAT_BAO:.0f} m/s: UAV hạ cánh toàn bộ",
               bool(q_b["ha_canh"][:, uav_b].all()), f"{len(uav_b)} UAV"))
    kq.append(("Yagi: 'ha_canh' do gió giật, không cần giông",
               bool(q_b["giong"].sum() == 0), "không có giông trong quan sát"))

    # 3) Gió lặng lại: cả hai cờ hết -> UAV cất cánh trở lại.
    tt_l = _thoi_tiet_bat_thuong(code=0, gust=3.0)
    g_l = _mang(tt_l)
    uav_l = _uav(g_l)
    _, q_l = sinh_kich_ban(g_l, tt_l, 0, 2, ThamSo(**TAT, giong=True))
    kq.append(("Gió lặng: UAV không còn trong 'ha_canh'",
               bool(not q_l["ha_canh"][:, uav_l].any()), "đã cất cánh"))
    kq.append(("Gió lặng: UAV không còn trong 'tat'",
               bool(not q_l["tat"][:, uav_l].any()), "nút hoạt động"))
    return kq


def _run() -> int:
    kq = kiem_tra()
    print()
    rong = max(len(t) for t, _, _ in kq) + 2
    for ten, dat, ct in kq:
        print(f"  {'✔' if dat else '✘'} {ten:<{rong}} {ct}")
    so_dat = sum(d for _, d, _ in kq)
    print(f"\nKết quả: {so_dat}/{len(kq)} đạt")
    return 0 if so_dat == len(kq) else 1


def test_yagi_uav_ha_canh():
    """Điểm vào cho pytest."""
    for ten, dat, ct in kiem_tra():
        assert dat, f"{ten}: {ct}"


if __name__ == "__main__":
    sys.exit(_run())
