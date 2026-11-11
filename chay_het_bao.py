"""Chạy nối tiếp: chờ sinh đủ .pkl 10 bão -> chấm DM6 -> xuất tuyến.

Chạy nền một lần; tự bỏ qua phần đã xong nên có thể chạy lại an toàn.
    python chay_het_bao.py
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

GOC = Path(r"E:\PBL4")
BAOS = ["prapiroon_2024", "soulik_2024", "trami_2024", "wutip_2025", "wipha_2025",
        "kajiki_2025", "nongfa_2025", "bualoi_2025", "matmo_2025", "kalmaegi_2025"]
SO_CUA_SO = 13
RUN = GOC / "data" / "bao_da_tuyen"
MO_HINH = "results/bao_da_tuyen/mo_hinh6"


def du_pkl(ten: str) -> bool:
    d = RUN / f"{ten}_k4_5"
    return d.exists() and len(list(d.glob("kb_*.pkl"))) >= SO_CUA_SO


def chay(args: list[str], ten: str) -> bool:
    print(f"\n{'='*70}\n>>> {ten}\n{'='*70}", flush=True)
    t0 = time.time()
    r = subprocess.run([sys.executable, "-W", "ignore", *args], cwd=GOC)
    print(f"<<< {ten}: exit={r.returncode} ({time.time()-t0:.0f}s)", flush=True)
    return r.returncode == 0


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    # 1. Sinh .pkl cho bão nào còn thiếu
    for b in BAOS:
        if du_pkl(b):
            print(f"[bỏ qua] {b}: đã đủ .pkl", flush=True)
            continue
        chay(["tao_du_lieu_bao.py", "--bao", b, "--so-cua-so", str(SO_CUA_SO),
              "--tien-trinh", "4"], f"sinh .pkl {b}")

    # 2. Chấm DM6 trên từng bão
    for b in BAOS:
        if not du_pkl(b):
            print(f"[bỏ qua chấm] {b}: thiếu .pkl", flush=True)
            continue
        chay(["-m", "src.nhieu_luong.cham_lai_dt", "--tap", b, "--K", "4-5",
              "--mo-hinh", MO_HINH, "--ten", "dm6", "--khoa", "dm6",
              "--khoa-tkcb10", "x"], f"chấm DM6 {b}")

    print("\n" + "=" * 70)
    print("XONG TOÀN BỘ: sinh .pkl + chấm DM6 cho 10 bão")
    print("=" * 70)


if __name__ == "__main__":
    main()
