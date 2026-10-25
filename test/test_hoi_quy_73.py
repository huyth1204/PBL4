"""
test_hoi_quy_73.py — Chốt hồi quy: cấu hình MẶC ĐỊNH phải tái lập đúng mạng 73 nút.

Mốc vàng là `dataset/oracle_dataset_v2.npz` (bản gốc, không bao giờ bị ghi đè).
Test so bốn thứ, theo thứ tự chặt dần:
  1. node_order  — phải GIỐNG HỆT từng ký tự (đây là hợp đồng chỉ số nút).
  2. reach_stats — phải GIỐNG HỆT (số cặp reachable mỗi snapshot).
  3. Số cạnh tồn tại (W != -1 ngoài đường chéo) — phải GIỐNG HỆT.
  4. W            — khớp trong sai số float32 khi lưu file (1e-4 ms).

Chạy:  venv\\Scripts\\python.exe test\\test_hoi_quy_73.py
hoặc  pytest test/test_hoi_quy_73.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")  # console Windows mặc định cp1252

GOC = Path(__file__).resolve().parents[1]
if str(GOC) not in sys.path:
    sys.path.insert(0, str(GOC))

MOC_VANG = GOC / "dataset" / "oracle_dataset_v2.npz"
# Bản v2 là mốc bất biến: số nút phải đúng 73 và sai số W chỉ do float32 khi lưu.
SO_NUT_73 = 73
SAI_SO_W = 1e-4


def _tai_w(npz_path: Path) -> np.ndarray:
    with np.load(npz_path, allow_pickle=False) as d:
        return d["W_snapshots"]


def _so_canh(W: np.ndarray) -> np.ndarray:
    """Số cạnh tồn tại mỗi snapshot: đếm ô != -1 và ngoài đường chéo."""
    n = W.shape[1]
    ngoai_cheo = ~np.eye(n, dtype=bool)[None, :, :]
    return ((W != -1.0) & ngoai_cheo).sum(axis=(1, 2))


def _so_sanh_voi_moc(W_moi: np.ndarray, W_cu: np.ndarray, ten: str, ket_qua: list):
    """So W mới với mốc vàng, ghi kết quả vào danh sách."""
    if W_moi.shape != W_cu.shape:
        ket_qua.append((f"{ten}: kích thước W", False,
                        f"mới {W_moi.shape} vs mốc {W_cu.shape}"))
        return
    khac_ton_tai = int(((_so_canh(W_moi) > 0) != (_so_canh(W_cu) > 0)).sum())
    lech = float(np.max(np.abs(W_moi - W_cu))) if W_moi.size else 0.0
    ket_qua.append((f"{ten}: kích thước W khớp mốc", W_moi.shape == W_cu.shape, str(W_moi.shape)))
    ket_qua.append((f"{ten}: W lệch tối đa ≤ {SAI_SO_W}", lech <= SAI_SO_W, f"{lech:.3e}"))
    ket_qua.append((f"{ten}: 0 cạnh khác về tồn tại", khac_ton_tai == 0, f"{khac_ton_tai} snapshot lệch"))


def kiem_tra(moc: Path = MOC_VANG) -> list[tuple[str, bool, str]]:
    """Sinh lại mạng mặc định trong bộ nhớ và so với mốc vàng. Trả danh sách (tên, đạt, chi_tiết)."""
    from dataset.oracle_labeler_sagsin import generate_dataset_snapshots

    moc = Path(moc)
    if not moc.exists():
        return [("Tìm thấy mốc vàng", False, f"thiếu {moc}")]

    print(f"→ Sinh lại mạng mặc định và so với mốc {moc.name} ...")
    ds = generate_dataset_snapshots(
        tle_path=GOC / "data" / "tle" / "starlink.txt",
        num_snapshots=600, time_step_seconds=20, num_pairs_per_snapshot=50,
        sample_sat_count=50, min_elevation_deg=15.0, seed=42, start_time="tle",
        sat_select="random", infra=("space", "haps", "uav"), mo_rong=False,
    )
    with np.load(moc, allow_pickle=False) as d_cu:
        order_cu = np.asarray(d_cu["node_order"])
        reach_cu = np.asarray(d_cu["reach_stats"])
        W_cu = np.asarray(d_cu["W_snapshots"])

    order_moi = np.asarray(ds["node_order"])
    reach_moi = np.asarray(ds["reach_stats"])
    W_moi = np.asarray(ds["W_snapshots"])

    kq: list[tuple[str, bool, str]] = []
    # 1. Hợp đồng chỉ số nút: danh sách tên phải giống hệt, đúng thứ tự.
    giong_order = order_moi.shape == order_cu.shape and bool(np.all(order_moi == order_cu))
    kq.append((f"node_order giống hệt mốc ({len(order_cu)} nút)", giong_order,
               "khớp" if giong_order else _mota_lech_order(order_moi, order_cu)))
    # 2. Số nút phải đúng 73.
    kq.append((f"Đúng {SO_NUT_73} nút", order_moi.shape[0] == SO_NUT_73, f"{order_moi.shape[0]} nút"))
    # 3. reach_stats giống hệt.
    giong_reach = reach_moi.shape == reach_cu.shape and bool(np.all(reach_moi == reach_cu))
    kq.append(("reach_stats giống hệt mốc", giong_reach, str(reach_moi.shape)))
    # 4. Số cạnh tồn tại từng snapshot giống hệt.
    canh_moi, canh_cu = _so_canh(W_moi), _so_canh(W_cu)
    giong_canh = canh_moi.shape == canh_cu.shape and bool(np.all(canh_moi == canh_cu))
    kq.append(("Số cạnh mỗi snapshot giống hệt mốc", giong_canh,
               f"tổng mới {int(canh_moi.sum())} vs mốc {int(canh_cu.sum())}"))
    # 5. W khớp trong sai số float32.
    _so_sanh_voi_moc(W_moi, W_cu, "W", kq)
    return kq


def _mota_lech_order(moi: np.ndarray, cu: np.ndarray) -> str:
    """Mô tả ngắn vị trí lệch đầu tiên để dễ sửa."""
    n = min(len(moi), len(cu))
    for i in range(n):
        if moi[i] != cu[i]:
            return f"lệch ở vị trí {i}: mới='{moi[i]}' vs mốc='{cu[i]}'"
    return f"độ dài khác: {len(moi)} vs {len(cu)}"


def _run() -> int:
    kq = kiem_tra()
    print()
    rong = max(len(t) for t, _, _ in kq) + 2
    for ten, dat, ct in kq:
        print(f"  {'✔' if dat else '✘'} {ten:<{rong}} {ct}")
    so_dat = sum(d for _, d, _ in kq)
    print(f"\nKết quả: {so_dat}/{len(kq)} đạt")
    return 0 if so_dat == len(kq) else 1


def test_hoi_quy_73():
    """Điểm vào cho pytest: mọi mục kiểm tra phải đạt."""
    kq = kiem_tra()
    for ten, dat, ct in kq:
        assert dat, f"{ten}: {ct}"


if __name__ == "__main__":
    sys.exit(_run())
