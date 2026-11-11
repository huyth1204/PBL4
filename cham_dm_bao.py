"""Chờ process sinh .pkl xong (không còn python khác) rồi chấm DM6 cho 10 bão.

Chạy nền một lần. Tự bỏ qua bão đã chấm (kiểm tra khoá 'dm6' trong .pkl).
"""
from __future__ import annotations

import pickle
import subprocess
import sys
import time
from pathlib import Path

GOC = Path(r"E:\PBL4")
BAOS = ["prapiroon_2024", "soulik_2024", "trami_2024", "wutip_2025", "wipha_2025",
        "kajiki_2025", "nongfa_2025", "bualoi_2025", "matmo_2025", "kalmaegi_2025"]
RUN = GOC / "data" / "bao_da_tuyen"
MO_HINH = "results/bao_da_tuyen/mo_hinh6"


def da_cham(ten: str) -> bool:
    """Bão đã chấm DM6 chưa (kiểm tra khoá dm6 trong .pkl đầu tiên)."""
    d = RUN / f"{ten}_k4_5"
    ps = sorted(d.glob("kb_*.pkl")) if d.exists() else []
    if not ps:
        return False
    try:
        r = pickle.load(open(ps[0], "rb"))[0]
        return "dm6" in r.get("ens", {})
    except Exception:
        return False


def dem_python() -> int:
    """Số tiến trình python đang chạy (trừ chính mình)."""
    out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq python.exe", "/NH"],
                         capture_output=True, text=True, errors="replace").stdout
    return sum(1 for l in out.splitlines() if l.strip().lower().startswith("python.exe"))


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("chờ process sinh .pkl xong…", flush=True)
    for _ in range(720):                      # tối đa 6 giờ chờ
        if dem_python() <= 1:
            break
        time.sleep(30)
    print("không còn process sinh khác — bắt đầu chấm", flush=True)

    for b in BAOS:
        d = RUN / f"{b}_k4_5"
        n = len(list(d.glob("kb_*.pkl"))) if d.exists() else 0
        if n < 13:
            print(f"[thiếu] {b}: chỉ {n}/13 .pkl", flush=True)
            continue
        if da_cham(b):
            print(f"[bỏ qua] {b}: đã chấm DM6", flush=True)
            continue
        print(f"\n>>> chấm DM6 {b}", flush=True)
        t0 = time.time()
        r = subprocess.run([sys.executable, "-W", "ignore", "-m", "src.nhieu_luong.cham_lai_dt",
                            "--tap", b, "--K", "4-5", "--mo-hinh", MO_HINH,
                            "--ten", "dm6", "--khoa", "dm6", "--khoa-tkcb10", "x"], cwd=GOC)
        print(f"<<< {b}: exit={r.returncode} ({time.time()-t0:.0f}s)", flush=True)

    print("\n" + "=" * 70)
    print("XONG: chấm DM6 cho mọi bão có đủ .pkl")
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
