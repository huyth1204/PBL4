"""Sinh lại .pkl cho 10 bão với cap_nut (lưu sẵn cặp nút).

Giữ nguyên `.pkl` cũ (không có cap_nut) vào `_cu_khong_capnut/`.
Chạy nền: ~3-4 giờ cho 10 bão.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import time
from pathlib import Path

GOC = Path(r"E:\PBL4")
BAOS = ["prapiroon_2024", "soulik_2024", "trami_2024", "wutip_2025", "wipha_2025",
        "kajiki_2025", "nongfa_2025", "bualoi_2025", "matmo_2025", "kalmaegi_2025"]
RUN = GOC / "data" / "bao_da_tuyen"
CU = RUN / "_cu_khong_capnut"


def co_capnut(ten: str) -> bool:
    import pickle
    d = RUN / f"{ten}_k4_5"
    ps = sorted(d.glob("kb_*.pkl"))
    if len(ps) < 13:
        return False
    try:
        return "cap_nut" in pickle.load(open(ps[0], "rb"))[0]
    except Exception:
        return False


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    for b in BAOS:
        if co_capnut(b):
            print(f"[bỏ qua] {b}: đã có cap_nut", flush=True)
            continue
        d = RUN / f"{b}_k4_5"
        if d.exists():
            CU.mkdir(parents=True, exist_ok=True)
            dich = CU / d.name
            if dich.exists():
                shutil.rmtree(dich)
            shutil.move(str(d), str(dich))
            print(f"[lưu cũ] {b} -> _cu_khong_capnut/", flush=True)
        print(f">>> sinh {b}", flush=True)
        t0 = time.time()
        r = subprocess.run([sys.executable, "-W", "ignore", "tao_du_lieu_bao.py",
                            "--bao", b, "--so-cua-so", "13", "--tien-trinh", "4"], cwd=GOC)
        print(f"<<< {b}: exit={r.returncode} ({time.time()-t0:.0f}s)", flush=True)

    print("\nXONG sinh cap_nut cho 10 bão", flush=True)


if __name__ == "__main__":
    main()
