"""Xuất tuyến cho MỌI bão đã có cap_nut -> duong_di_nut.json (gộp nhiều bão)."""
import pickle
import subprocess
import sys
from pathlib import Path

GOC = Path(r"E:\PBL4")
BAOS = ["prapiroon_2024", "soulik_2024", "trami_2024", "wutip_2025", "wipha_2025",
        "kajiki_2025", "nongfa_2025", "bualoi_2025", "matmo_2025", "kalmaegi_2025"]


def co_capnut(b: str) -> bool:
    d = GOC / "data" / "bao_da_tuyen" / f"{b}_k4_5"
    ps = sorted(d.glob("kb_*.pkl"))
    if len(ps) < 13:
        return False
    try:
        return "cap_nut" in pickle.load(open(ps[0], "rb"))[0]
    except Exception:
        return False


for b in BAOS:
    if not co_capnut(b):
        print(f"[bo qua] {b}: chua co cap_nut", flush=True)
        continue
    print(f">>> xuat tuyen {b}", flush=True)
    r = subprocess.run([sys.executable, "xuat_tuyen_dm.py", "--tap", f"{b}_k4_5",
                        "--ten", "dm6", "--out", "duong_di_nut.json"], cwd=GOC)
    if r.returncode != 0:
        print(f"<<< {b} LOI", flush=True)

print("\n>>> them dijkstra", flush=True)
subprocess.run([sys.executable, "them_dijkstra.py"], cwd=GOC)
print("XONG")
