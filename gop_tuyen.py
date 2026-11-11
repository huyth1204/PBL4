"""
gop_tuyen.py — BƯỚC 2/3: thêm tuyến dphat vào `duong_di_nut.json`.

Dùng `cap_nut` đã lưu trong .pkl: chọn lựa chọn `a_dp` tối thiểu
`prop + β·(−ln(1−risk))` (β = 10, BETA_GAMMA[0]), rồi tra thẳng danh sách nút.
Không dựng lại mạng.

    python gop_tuyen.py --tap soulik_2024_k4_5
    python ban_do_3d.py --bao soulik
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

GOC = Path(r"E:\PBL4")
if str(GOC) not in sys.path:
    sys.path.insert(0, str(GOC))

RA = GOC / "data" / "bao_da_tuyen"
BETA = 10.0


def chon_dphat(gon) -> list[int]:
    """Lựa chọn trong kho có tổng `prop + β·(−ln(1−risk))` nhỏ nhất, mỗi luồng một bộ."""
    rui = np.clip(gon.risk, 0.0, 1.0 - 1e-9)
    phi = gon.prop + BETA * (-np.log(1.0 - rui))
    ra = []
    for k in range(gon.K):
        chi_phi = [float(phi[np.asarray(opt, int)].sum()) if len(opt) else np.inf
                   for opt in gon.opts[k]]
        ra.append(int(np.argmin(chi_phi)))
    return ra


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tap", default="soulik_2024_k4_5")
    ap.add_argument("--out", default="duong_di_nut.json")
    a = ap.parse_args()

    tb = a.tap.split("_k")[0]
    ps = sorted((RA / a.tap).glob("kb_*.pkl"))
    nut = json.loads((GOC / a.out).read_text(encoding="utf-8"))
    t0 = time.time()
    moi, thieu = 0, 0

    for pk in ps:
        for r in pickle.load(open(pk, "rb")):
            cap = r.get("cap_nut")
            if not cap:
                thieu += 1
                continue
            a_dp = chon_dphat(r["gon"])
            khoa = f"{pk.stem}|{int(r['t'])}|{int(r['kb'])}"
            nut.setdefault(khoa, {"t": int(r["t"]), "K": int(r["K"]),
                                  "nguon": list(r["nguon"]), "bao": tb})
            nut[khoa]["dphat"] = [cap[k][a_dp[k]][0] for k in range(len(cap))]
            moi += 1
        print(f"  {pk.name}", flush=True)

    out = GOC / a.out
    out.write_text(json.dumps(nut, separators=(",", ":")), encoding="utf-8")
    print(f"\nthiếu cap_nut: {thieu} | cập nhật dphat: {moi} ({time.time()-t0:.1f}s) → {out}")
    k0 = next(iter(nut))
    print(f"ví dụ {k0}:")
    print(f"  dm    luồng 0: {nut[k0].get('dm', ['?'])[0]}")
    print(f"  dphat luồng 0: {nut[k0]['dphat'][0]}")


if __name__ == "__main__":
    main()
