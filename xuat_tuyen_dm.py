"""
xuat_tuyen_dm.py — BƯỚC B: xuất tuyến DM / thầy / dphat / dijkstra sang `duong_di_nut.json`.

Dữ liệu lấy từ `cap_nut` đã lưu sẵn trong `.pkl`:
    cap_nut[k][a] = [tuyến_chính, dự_phòng_1, ...], mỗi tuyến là DANH SÁCH NÚT
                    [nguồn, ..., đích] (đã kiểm ở Bước A: 360/360 mẫu có nguồn->đích).

Mỗi luồng xuất dạng:
    {"luong": k, "nguon": tên, "dich": tên, "nut": [nút...], "day_du": true}

Khoá bản ghi: `{bao}|{ten_pkl}|{t}|{kb}` (Bước B mục 3 — tránh ghi đè khi gộp bão).

Bốn loại tuyến:
  dm       — mô hình (chỉ số lựa chọn từ `model(gon)`)
  thay     — nhãn thầy (`r["nhan"]`)
  dphat    — Dijkstra phạt rủi ro (chọn trong kho theo `prop + 10·(−ln(1−risk))`)
  dijkstra — Dijkstra thuần theo tổng trễ trên `W` của đúng snapshot (bước B mục 51)

    python xuat_tuyen_dm.py --tap soulik_2024_k4_5 --ten dm6 --out duong_di_nut.json
    python gop_tuyen.py --tap soulik_2024_k4_5        # thêm dphat + dijkstra
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
BETA = 10.0          # BETA_GAMMA[0] của dijkstra_phat


# ------------------------------------------------------------------ tiện ích


def nap_model(thu_muc: str, ten: str):
    from src.nhieu_luong import mo_hinh_dt as md
    return md.ham_chon(GOC / "results" / "bao_da_tuyen" / thu_muc, ten, softmax=False)["dm"]


def chon_dphat(gon) -> list[int]:
    """Lựa chọn trong kho tối thiểu `prop + β·(−ln(1−risk))`, mỗi luồng một bộ."""
    rui = np.clip(gon.risk, 0.0, 1.0 - 1e-9)
    phi = gon.prop + BETA * (-np.log(1.0 - rui))
    ra = []
    for k in range(gon.K):
        chi_phi = [float(phi[np.asarray(opt, int)].sum()) if len(opt) else np.inf
                   for opt in gon.opts[k]]
        ra.append(int(np.argmin(chi_phi)))
    return ra


def luong_day_du(k: int, tuyen: list[int], names: list[str]) -> dict:
    """Gói một tuyến thành bản ghi luồng có nguồn/đích rõ ràng."""
    return {
        "luong": k,
        "nguon": names[tuyen[0]],
        "dich": names[tuyen[-1]],
        "nut": [int(n) for n in tuyen],
        "day_du": True,      # kho tuyến của dự án luôn là đường đi đầy đủ (Bước A)
    }


def sap_xep_thanh_duong(canh) -> tuple[list[int] | None, bool]:
    """Dự phòng cho dữ liệu cũ dạng CẠNH: sắp xếp thành chuỗi nút liên tục."""
    ke = {}
    for u, v in canh:
        ke.setdefault(u, []).append(v)
        ke.setdefault(v, []).append(u)
    dau = [n for n, l in ke.items() if len(l) == 1]
    if len(dau) != 2 or any(len(l) > 2 for l in ke.values()):
        return None, False
    chuoi, truoc, nut = [dau[0]], None, dau[0]
    while True:
        tiep = [x for x in ke[nut] if x != truoc]
        if not tiep:
            break
        truoc, nut = nut, tiep[0]
        chuoi.append(nut)
    return chuoi, len(chuoi) == len(ke)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tap", default="soulik_2024_k4_5")
    ap.add_argument("--thu-muc", default="mo_hinh6")
    ap.add_argument("--ten", default="dm6")
    ap.add_argument("--out", default="duong_di_nut.json")
    a = ap.parse_args()

    tb = a.tap.split("_k")[0]
    ps = sorted((RA / a.tap).glob("kb_*.pkl"))
    if not ps:
        raise SystemExit(f"không có .pkl trong {RA / a.tap}")
    print(f"bão: {tb} | {len(ps)} tệp | model: {a.thu_muc}/{a.ten}.pt")

    names = [str(x) for x in np.load(GOC / "dataset" / "oracle_dataset_v5.npz",
                                     allow_pickle=False)["node_order"]]
    model = nap_model(a.thu_muc, a.ten)

    ra, bo_qua, khong_day_du = {}, 0, 0
    t0 = time.time()
    for q, pk in enumerate(ps, 1):
        for r in pickle.load(open(pk, "rb")):
            cap = r.get("cap_nut")
            if not cap:
                bo_qua += 1
                continue
            khoa = f"{tb}|{pk.stem}|{int(r['t'])}|{int(r['kb'])}"
            a_dm = model(r["gon"])
            a_dp = chon_dphat(r["gon"])

            def theo(a_sel, luong=0):
                """Lấy tuyến CHÍNH của mỗi luồng theo bộ chỉ số `a_sel`."""
                return [luong_day_du(k, cap[k][int(a_sel[k])][0], names)
                        for k in range(len(cap))]

            item = {"t": int(r["t"]), "K": int(r["K"]), "bao": tb,
                    "tap": a.tap, "kb": int(r["kb"]),
                    "nguon": list(r["nguon"]),
                    "dm": theo(a_dm),
                    "dphat": theo(a_dp)}
            if r.get("nhan") is not None:
                item["thay"] = theo(r["nhan"])
            # dự phòng của DM (tuyến thứ 2 trong cùng phương án)
            phu = []
            for k in range(len(cap)):
                ds = cap[k][int(a_dm[k])]
                phu.append([int(n) for n in ds[1]] if len(ds) > 1 else None)
            item["dm_du_phong"] = phu

            # kiểm tra đủ nguồn -> đích
            for loai in ("dm", "thay", "dphat"):
                for l in item.get(loai, []):
                    if not l["nut"]:
                        khong_day_du += 1
            ra[khoa] = item
        print(f"  {q}/{len(ps)} {pk.name}", flush=True)

    out = GOC / a.out
    cu = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    cu.update(ra)
    out.write_text(json.dumps(cu, separators=(",", ":")), encoding="utf-8")
    print(f"\nbỏ qua (thiếu cap_nut): {bo_qua} | luồng rỗng: {khong_day_du}")
    print(f"đã ghi: {out} ({len(ra)} bản ghi mới, tổng {len(cu)}, {time.time()-t0:.1f}s)")

    k0 = next(iter(ra))
    v = ra[k0]
    print(f"\nví dụ {k0}:")
    for l in v["dm"][:2]:
        print(f"  DM luồng {l['luong']}: {l['nguon']} -> {l['dich']} "
              f"({len(l['nut'])} nút): {l['nut']}")
    for l in v["dphat"][:2]:
        print(f"  dphat luồng {l['luong']}: {l['nguon']} -> {l['dich']} "
              f"({len(l['nut'])} nút): {l['nut']}")


if __name__ == "__main__":
    main()
