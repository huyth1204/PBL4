"""Chay 6 tien trinh song song (moi cai 1 phan cap) cho 1 bao, roi gop ket qua."""

import json
import subprocess
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
G = Path(r"E:\PBL4")
BAO = sys.argv[1] if len(sys.argv) > 1 else "trami"
N = 6
OUT = G / "duong_di_cap.json"

for f in G.glob("duong_di_cap.phan*.json"):
    f.unlink(missing_ok=True)
OUT.unlink(missing_ok=True)

t0 = time.time()
procs = []
for phan in range(N):
    log = open(G / f"_phan{phan}.log", "w", encoding="utf-8")
    p = subprocess.Popen([sys.executable, str(G / "xuat_tuyen_cap.py"),
                          "--bao", BAO, "--tien-trinh", "1",
                          "--chia", str(N), "--phan", str(phan)],
                         cwd=str(G), stdout=log, stderr=subprocess.STDOUT)
    procs.append((phan, p, log))
    print(f"  khởi động phần {phan} (pid {p.pid})", flush=True)

for phan, p, log in procs:
    p.wait()
    log.close()
    print(f"  phần {phan} xong: exit {p.returncode} "
          f"({time.time()-t0:.0f}s)", flush=True)

tong = {}
for phan in range(N):
    f = G / f"duong_di_cap.phan{phan}.json"
    if f.exists():
        d = json.loads(f.read_text(encoding="utf-8"))
        tong.update(d)
        print(f"  gộp phần {phan}: {len(d)} mục")
OUT.write_text(json.dumps(tong, separators=(",", ":")), encoding="utf-8")
print(f"XONG: {len(tong)} mục -> {OUT} ({OUT.stat().st_size/1e6:.1f} MB, "
      f"{time.time()-t0:.0f}s)")
