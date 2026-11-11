"""
ban_do_3d.py — Bản đồ 3D mạng SAGSIN (vệ tinh – HAPS – UAV – trạm).

Chạy ở E:/PBL4:
    python ban_do_3d.py --bao soulik             # bão soulik -> ban_do_3d.html
    python ban_do_3d.py --bao soulik --frames 30 # đổi số khung
    python ban_do_3d.py                          # không --bao: thời tiết cả năm, ngày mưa nhất
    python ban_do_3d.py --in dataset/v5/val.npz  # đổi tập dữ liệu nền
    python ban_do_3d.py --khong-mo-phong         # bỏ mô phỏng đứt gãy mưa/gió (~4 s thay vì 24 s)
    python ban_do_3d.py --out ban_do_soulik.html # tên file ra khác (mỗi bão 1 file)
    python ban_do_3d.py --liet-ke                # liệt kê tệp dữ liệu + 10 cơn bão

Cần có: assets_ban_do/{khung_dau.txt, khung_than.txt, ve_3d.js} và duong_di_nut.json.

Dữ liệu mỗi khung đưa xuống ve_3d.js (biến F):
  p, isl, acc          vị trí nút; cạnh vệ tinh–vệ tinh; cạnh nối mặt đất
  brk, brk_mua, brk_nut cạnh đứt thật (W <= 0) và đứt do mưa / do nút bị tắt
  nuts                 thông số từng nút (khoảng cách, góc ngẩng, băng thông…)
  dmTuyen              tuyến DM/dphat/dijkstra/thầy từng luồng + trễ từng luồng
  ac, cap, nb, R, G    trễ tuyến, (tổng cặp, cặp tới được), số cạnh, mưa, gió

Thông số cạnh tính từ toạ độ CỦA ĐÚNG KHUNG đang vẽ (thong_so_canh), không dùng
link_stats của v5 vì bảng đó chỉ có 54 snapshot, lệch mốc bão (75..588).

Làm lại dữ liệu cho một bão (chạy ở thư mục gốc):
    python tao_du_lieu_bao.py --bao soulik
    python xuat_tuyen_dm.py --tap soulik_2024_k4_5 --ten dm6
    python them_dijkstra.py                 # tuỳ chọn
    python ban_do_3d.py --bao soulik

Định tuyến cho cặp NGUỒN–ĐÍCH bất kỳ (DM/dphat cho cặp ngoài 4–5 luồng kịch bản):
    python xuat_tuyen_cap.py --bao soulik --mau          # 10 cặp mẫu (~1 phút)
    python xuat_tuyen_cap.py --bao soulik                # mọi cặp 32×5 (~15 phút)
    python xuat_tuyen_cap.py --bao soulik --cap CaMau,DaNang --cap PleiKu,Hanoi
    python ban_do_3d.py --bao soulik        # đọc duong_di_cap.json; cặp đó có DM/dphat, nhãn *

Tuyến trên bản đồ:
    DM tím · dphat xanh lá · dijkstra xanh dương · thầy xám (bật/tắt ở chân trang)
"""


from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

for _luong in (sys.stdout, sys.stderr):          # console Windows hay mặc định cp1258
    try:
        _luong.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

GOC = Path(__file__).resolve().parent
ASSET = GOC / "assets_ban_do"
BAO_DIR = GOC / "data" / "bao_manh"
BAO_DANH_MUC = BAO_DIR / "danh_muc.json"
THOI_TIET = GOC / "data" / "thoi_tiet_vn_mr.npz"

# Thứ tự loại nút — PHẢI khớp KSTYLE trong ve_3d.js
KIND = {"leo": 0, "haps": 1, "uav": 2, "gateway": 3, "user": 4}
SAT, AIR, GROUND = "space", "air", "ground"
R_TRAI_DAT_KM = 6378.137          # bán kính cầu dùng để dựng xyz của nút (km)

# Tên các tệp bão có trong data/bao_manh — dùng để chạy lại bộ mô phỏng lấy đứt gãy.
BANG_XAC_NHAN = {
    # 10 cơn bão mạnh (data/bao_manh/*.npz)
    "prapiroon_2024", "soulik_2024", "trami_2024", "wutip_2025", "wipha_2025",
    "kajiki_2025", "nongfa_2025", "bualoi_2025", "matmo_2025", "kalmaegi_2025",
    # thời tiết cả năm (data/thoi_tiet_vn_mr.npz) — cũng chạy được cùng bộ mô phỏng
    "thoi_tiet_vn_mr",
}


# Hai nút quần đảo (toạ độ lấy từ node_latlon_alt, trùng src/physics/sites.py).
DAO = ("HoangSa", "TruongSa")

# ---------------------------------------------------------------------------
# Rà soát dữ liệu (không đọc phần nặng nếu không cần)
# ---------------------------------------------------------------------------

UNG_VIEN = [
    ("v5/test  — 224 nút, 150 Starlink phủ VN (bản mới nhất)", "dataset/v5/test.npz"),
    ("v5/val   — 224 nút", "dataset/v5/val.npz"),
    ("v5/train — 224 nút, 294 snapshot (nặng hơn)", "dataset/v5/train.npz"),
    ("v4/test  — 330 nút, chòm Walker mô phỏng", "dataset/v4/test.npz"),
    ("v3/test  — 174 nút", "dataset/v3/test.npz"),
    ("demo n373 — 373 nút (không có node_latlon_alt)", "huong_dan/du_lieu_demo/n373.npz"),
    ("demo n273 — 273 nút (không có node_latlon_alt)", "huong_dan/du_lieu_demo/n273.npz"),
    ("demo n173 — 173 nút (không có node_latlon_alt)", "huong_dan/du_lieu_demo/n173.npz"),
    ("demo n73  — 73 nút (không có node_latlon_alt)", "huong_dan/du_lieu_demo/n73.npz"),
]

# Tệp còn trong dự án nhưng KHÔNG vẽ được (giải thích trong --liet-ke)
KHONG_VE = [
    ("dataset/test.npz", "bộ v2 73 nút: không có node_latlon_alt"),
    ("dataset/train.npz", "bộ v2: không có node_latlon_alt"),
    ("dataset/val.npz", "bộ v2: không có node_latlon_alt"),
    ("dataset/oracle_dataset_v2.npz", "bản gốc v2: không có node_latlon_alt"),
    ("dataset/oracle_dataset_v3.npz", "bản gốc v3 — dùng dataset/v3/{test,val}.npz"),
    ("dataset/oracle_dataset_v4.npz", "bản gốc v4 — dùng dataset/v4/{test,val}.npz"),
]


def doc_khoa(path: Path, *khoa: str) -> dict:
    """Chỉ nạp các khoá cần (tránh link_stats của oracle gốc nặng hàng trăm MB)."""
    out = {}
    with np.load(path, allow_pickle=False) as raw:
        for k in khoa:
            if k in raw.files:
                out[k] = raw[k]
    return out


def kiem_tra(path: Path) -> tuple[bool, str]:
    """Tệp có vẽ được không + lý do."""
    if not path.exists():
        return False, "không có tệp"
    try:
        d = doc_khoa(path, "W_snapshots", "node_order", "node_latlon_alt",
                     "snapshot_id", "y_paths_indices")
    except Exception as e:
        return False, f"không đọc được ({e})"
    if "W_snapshots" not in d:
        return False, "thiếu W_snapshots (chỉ là đặc trưng/mẫu, không phải mạng)"
    W = d["W_snapshots"]
    if W.ndim != 3 or W.shape[1] != W.shape[2]:
        return False, f"W_snapshots shape {W.shape} không phải [S, N, N]"
    n = W.shape[1]
    if "node_order" not in d or len(d["node_order"]) != n:
        return False, "node_order không khớp N"
    if "node_latlon_alt" not in d:
        return False, "thiếu node_latlon_alt (không dựng được toạ độ)"
    la = np.asarray(d["node_latlon_alt"])
    if la.ndim != 3 or la.shape[1:] != (n, 3):
        return False, f"node_latlon_alt shape {la.shape} không khớp N={n}"
    for k in ("snapshot_id", "y_paths_indices"):
        if k not in d:
            return False, f"thiếu {k}"
    if int((W > 0).sum()) == 0:
        return False, "W_snapshots không có liên kết nào"
    return True, f"S={W.shape[0]}, N={n}, {len(d['y_paths_indices'])} mẫu"


def in_liet_ke() -> None:
    print(f"{'TỆP':<44}{'TRẠNG THÁI':<13}GHI CHÚ")
    for ten, rel in UNG_VIEN:
        ok, ly_do = kiem_tra(GOC / rel)
        print(f"{rel:<44}{('DÙNG ĐƯỢC' if ok else 'BỎ QUA'):<13}{ly_do}")
    for rel, vi_sao in KHONG_VE:
        if (GOC / rel).exists():
            print(f"{rel:<44}{'BỎ QUA':<13}{vi_sao}")
    for p in (THOI_TIET,):
        if p.exists():
            d = doc_khoa(p, "site_names", "time_utc", "precip", "gust", "cape")
            co = [k for k in ("precip", "gust", "cape") if k in d]
            print(f"{str(p.relative_to(GOC)):<44}{'THỜI TIẾT':<13}"
                  f"{len(d['site_names'])} điểm, {len(d['time_utc'])} giờ (cột: {', '.join(co)})")
    bao = doc_bao()
    if bao:
        print(f"\n{'BÃO (data/bao_manh)':<44}{'KHẮC NGHIỆT':<13}giờ  mưa max  gió max")
        for b in bao:
            print(f"  {b['ten']:<42}{len(b['time']):>4}{float(b['precip'].max()):>9.1f}"
                  f"{float(b['gust'].max()):>9.1f}")
    else:
        print("\nKhông đọc được data/bao_manh/danh_muc.json")
    print("\nToạ độ: node_latlon_alt có sẵn trong v5/v3/v4 nên KHÔNG cần dựng lại từ TLE.")


def tim_nguon(ten: str | None) -> Path:
    if ten:
        p = Path(ten)
        if not p.is_absolute():
            p = GOC / p
        ok, ly_do = kiem_tra(p)
        if not ok:
            raise SystemExit(f"'{ten}' không vẽ được: {ly_do}\nChạy --liet-ke để xem tệp dùng được.")
        return p
    for _ten, rel in UNG_VIEN:
        p = GOC / rel
        if kiem_tra(p)[0]:
            return p
    raise SystemExit("Không có tệp nào vẽ được. Chạy --liet-ke để xem chi tiết.")


# ---------------------------------------------------------------------------
# Thời tiết
# ---------------------------------------------------------------------------


def doc_bao() -> list[dict]:
    """Danh mục 10 cơn bão mạnh (data/bao_manh) — mỗi cơn là một tệp thời tiết đã cắt.

    Đây mới là dữ liệu thời tiết KHẮC NGHIỆT: mưa tới 43 mm/h, gió giật 38.7 m/s,
    CAPE 5060 — so với thời tiết bình thường trong thoi_tiet_vn_mr.npz (mưa tối đa 15 mm/h).
    """
    if not BAO_DANH_MUC.exists():
        return []
    try:
        cat = json.loads(BAO_DANH_MUC.read_text(encoding="utf-8"))
    except Exception:
        return []
    ra = []
    for b in cat.get("bao", []):
        p = BAO_DIR / str(b.get("tep", ""))
        if not p.exists():
            continue
        d = doc_khoa(p, "site_names", "time_utc", "precip", "gust", "cape")
        if not all(k in d for k in ("site_names", "time_utc", "precip", "gust")):
            continue
        ra.append({
            "ten": str(b.get("ten", p.stem)),
            "ten_tep": p.stem,
            "tep": f"data/bao_manh/{p.name}",
            "names": [str(x) for x in d["site_names"]],
            "time": [str(x) for x in d["time_utc"]],
            "precip": np.nan_to_num(np.asarray(d["precip"], np.float32)),
            "gust": np.nan_to_num(np.asarray(d["gust"], np.float32)),
            "cape": np.nan_to_num(np.asarray(d.get("cape", np.zeros_like(d["precip"])), np.float32)),
            "giong": True,
            "gio_bat_dau": list(b.get("gio_bat_dau_kich_ban", [])) or None,
        })
    return ra


def doc_thoi_tiet() -> dict:
    """Thời tiết bình thường (cả năm) — dùng khi muốn xem lúc trời không bão."""
    if THOI_TIET.exists():
        d = doc_khoa(THOI_TIET, "site_names", "time_utc", "precip", "gust", "cape")
        return {
            "ten": "thời tiết cả năm (bình thường)",
            "tep": str(THOI_TIET.relative_to(GOC)),
            "ten_tep": THOI_TIET.stem,   # để chạy dut_gay() như với bão
            "names": [str(x) for x in d["site_names"]],
            "time": [str(x) for x in d["time_utc"]],
            "precip": np.nan_to_num(np.asarray(d["precip"], np.float32)),
            "gust": np.nan_to_num(np.asarray(d["gust"], np.float32)),
            "cape": np.nan_to_num(np.asarray(d.get("cape", np.zeros_like(d["precip"])), np.float32)),
            "gio_bat_dau": None,
            "la_tn6": True,   # bộ thời tiết của TN6
        }
    print("  ! Không thấy tệp thời tiết — mưa/gió để 0")
    z = np.zeros((1, 0), np.float32)
    return {"ten": "không có", "tep": None, "names": [], "time": [], "precip": z, "gust": z,
            "cape": z, "gio_bat_dau": None}


def chon_nguon_thoi_tiet(ten_bao: str | None, uu_tien_bao: bool,
                         n_snap: int = 1, diem_ground: list[str] | None = None) -> dict:
    """--bao <tên> | --bao manh-nhat | (mặc định) cơn nặng nhất
    | --thoi-tiet-binh-thuong."""
    danh_sach = doc_bao()
    for b in danh_sach:
        b["_ground"] = set(diem_ground or [])
    # --thoi-tiet-binh-thuong ưu tiên hơn mặc định của --bao
    if not uu_tien_bao:
        return doc_thoi_tiet()
    if ten_bao is None:
        return doc_thoi_tiet()
    if not danh_sach:
        print("  ! Không đọc được data/bao_manh — dùng thời tiết bình thường")
        return doc_thoi_tiet()
    if ten_bao and ten_bao not in ("manh-nhat", "auto"):
        for b in danh_sach:
            if b["ten"] == ten_bao or b["ten"].startswith(ten_bao):
                return b
        raise SystemExit(f"Không có bão '{ten_bao}'. Có: "
                         f"{', '.join(b['ten'] for b in danh_sach)}, tat-ca")
    # cơn "nặng nhất" = tổng mưa lớn nhất trên các điểm MẶT ĐẤT
    return max(danh_sach, key=lambda b: float(
        b["precip"][:, [k for k, nm in enumerate(b["names"]) if nm in b["_ground"]]].sum()
        if b["_ground"] else b["precip"].sum()))


def gio_mua_nhat(tt: dict, n_frame: int = 1, khoang: int = 1) -> int:
    """Giờ bắt đầu khung hình.

    Bão dài 25–101 giờ, nên chỉ tiến 1 giờ mỗi khung thì cả 10 khung vẫn nằm trong vài
    giờ đầu và không thấy bão lớn lên. Ở chế độ bão, `khoang` được tính lại cho phủ trọn
    cơn: mỗi khung cách nhau `len(time)/n_frame` giờ, bắt đầu từ giờ 0 của cơn. Nhờ vậy
    chuỗi khung đi hết: hình thành → đỉnh → tan.

    Không có --bao: lấy ngày mưa nhiều nhất trong chuỗi cả năm.
    """
    if tt["tep"] is None or not tt["time"]:
        return -1
    if tt.get("gio_bat_dau") is not None:      # là một cơn bão: phủ trọn cơn
        return 0
    g = np.arange(0, len(tt["time"]), 24)      # lấy mẫu mỗi 24 giờ (1 ngày)
    dem = (tt["precip"][g] > 2.5).sum(1)
    return int(g[int(np.argmax(dem))])


def thoi_diem(tt: dict, gio: int, sid: int) -> str:
    """Giờ VN (UTC+7) của cột thời tiết đang dùng."""
    if 0 <= gio < len(tt["time"]):
        raw = tt["time"][gio]
        for fmt in ("%Y-%m-%dT%H:%M", "%Y-%m-%dT%H:%M:%S"):
            try:
                dt = datetime.strptime(raw, fmt).replace(tzinfo=timezone.utc)
                return (dt + timedelta(hours=7)).strftime("%d/%m %H:%M")
            except ValueError:
                pass
        return raw[5:16]
    return f"snapshot {sid}"


# ---------------------------------------------------------------------------
# Liên kết
# ---------------------------------------------------------------------------


def loai_cap(layer: np.ndarray) -> np.ndarray:
    """(N,N) int8: 0 = ISL (vệ tinh–vệ tinh), 1 = GSL, 2 = LOCAL."""
    space = layer == SAT
    t = np.full((layer.size, layer.size), 2, np.int8)
    t[space[:, None] & space[None, :]] = 0
    t[space[:, None] & ~space[None, :]] = 1
    t[~space[:, None] & space[None, :]] = 1
    return t


def bang_canh(W: np.ndarray, loai: np.ndarray) -> dict:
    """Cạnh có W > 0 của khung: u, v, trọng số (ms) và loại (0 ISL / 1 GSL / 2 LOCAL)."""
    iu, iv = np.nonzero(np.triu(W > 0, 1))
    return {"u": iu, "v": iv, "w": W[iu, iv].astype(np.float64), "loai": loai[iu, iv]}


def cap_phang(u: np.ndarray, v: np.ndarray) -> list[int]:
    """[u0, v0, u1, v1, ...] — dạng mảng phẳng mà ve_3d.js đọc."""
    out = np.empty(u.size * 2, np.int32)
    out[0::2] = u
    out[1::2] = v
    return out.tolist()


# ---------------------------------------------------------------------------
# Dựng các khung hình
# ---------------------------------------------------------------------------



def loc_canh(cap: dict, loai_can: int, dut: set) -> list[int]:
    """Cặp [u,v,...] của cạnh thuộc `loai_can`, bỏ các cạnh đã đứt.

    Cạnh đứt vẽ riêng màu đỏ; nếu vẫn nằm trong lớp sống thì bị vẽ chồng hai màu.
    """
    u, v, t = cap["u"], cap["v"], cap["loai"]
    giu = (t == loai_can) & np.array([(int(a), int(b)) not in dut for a, b in zip(u, v)])
    return cap_phang(u[giu], v[giu])


def thanh_phan(W0: np.ndarray) -> tuple[np.ndarray, int]:
    """Nhãn thành phần liên thông của đồ thị liên kết thật (W0 > 0).

    Trả (nhãn từng nút, số thành phần). Nút đứng một mình là thành phần riêng — đó
    chính là "nút không nối được với ai" theo dữ liệu, không phải suy diễn.
    """
    n = W0.shape[0]
    nhan = np.full(n, -1, np.int32)
    k = 0
    for st in range(n):
        if nhan[st] >= 0:
            continue
        nhan[st] = k
        q = [st]
        while q:
            u = q.pop()
            for v in np.where(W0[u] > 0)[0]:
                if nhan[v] < 0:
                    nhan[v] = k
                    q.append(int(v))
        k += 1
    return nhan, k


def thong_so_canh(latlon_alt: np.ndarray, kind: np.ndarray, W0: np.ndarray,
                   iu: np.ndarray, iv: np.ndarray) -> dict:
    """Khoảng cách / góc ngẩng / băng thông / mất gói của TỪNG cặp nút (u, v) tại khung.

    Bảng `link_stats` của v5 chỉ có 54 snapshot lệch pha với khung đang vẽ (bão dùng
    chuỗi oracle 600), nên thông số cạnh được tính lại từ toạ độ của CHÍNH khung đó,
    dùng đúng luật trong src/physics/air_ground.py và src/graph/*.py:
      GSL  (sóng–mặt đất)            100 Mbps / 1e-2
      ASL  (vệ tinh–HAPS/UAV)        100 Mbps / 1e-2
      ISL  (vệ tinh–vệ tinh)        1000 Mbps / 1e-3
      LOCAL (HAPS/UAV/trạm, Air–Air)  theo LOCAL_RULES
    """
    from src.physics.air_ground import LOCAL_RULES
    from src.graph.graph_builder import (BANDWIDTH_GSL_MBPS, BANDWIDTH_ISL_MBPS,
                                         PACKET_LOSS_GSL, PACKET_LOSS_ISL)
    from src.graph.multilayer_builder import BANDWIDTH_ASL_MBPS, PACKET_LOSS_ASL

    la = np.radians(latlon_alt.astype(np.float64))
    xyz = np.stack([(R_TRAI_DAT_KM + latlon_alt[:, 2]) * np.cos(la[:, 0]) * np.cos(la[:, 1]),
                    (R_TRAI_DAT_KM + latlon_alt[:, 2]) * np.cos(la[:, 0]) * np.sin(la[:, 1]),
                    (R_TRAI_DAT_KM + latlon_alt[:, 2]) * np.sin(la[:, 0])], 1)
    up = np.stack([np.cos(la[:, 0]) * np.cos(la[:, 1]),
                   np.cos(la[:, 0]) * np.sin(la[:, 1]),
                   np.sin(la[:, 0])], 1)

    vec = xyz[iv] - xyz[iu]
    dist = np.linalg.norm(vec, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        elev = np.degrees(np.arcsin(np.clip((vec * up[iu]).sum(1) / np.maximum(dist, 1e-9), -1, 1)))
    elev[np.isnan(elev)] = 90.0

    sat = kind == "leo"
    a_sat, b_sat = sat[iu], sat[iv]
    bw = np.where(a_sat | b_sat,
                  np.where(a_sat & b_sat, BANDWIDTH_ISL_MBPS,
                           np.where((kind[iu] == "leo") | (kind[iv] == "leo"),
                                    BANDWIDTH_ASL_MBPS, BANDWIDTH_GSL_MBPS)),
                  0.0)
    loss = np.where(a_sat & b_sat, PACKET_LOSS_ISL,
                    np.where(a_sat | b_sat, np.where(kind[iu] == "leo", PACKET_LOSS_ASL,
                                                    PACKET_LOSS_GSL), 0.0))
    local = np.where(~a_sat & ~b_sat)[0]
    for q in local:
        rule = local_rule(LOCAL_RULES, kind[iu[q]], kind[iv[q]])
        if rule is not None:
            bw[q], loss[q] = rule.bandwidth_mbps, rule.packet_loss
    return {"dist": dist, "elev": elev, "bw": bw, "loss": loss,
            "co_so": (dist > 0) & (bw > 0)}


def local_rule(rules: dict, ka: str, kb: str):
    """Luật liên kết cục bộ cho cặp loại nút — giống cách oracle gán băng thông."""
    for cap in (frozenset((ka, kb)), frozenset((ka, "uav")), frozenset((kb, "uav"))):
        if cap in rules:
            return rules[cap]
    return None


def dem_toi_duoc(comp: np.ndarray, ten_nut: list[str], sides: np.ndarray) -> tuple[int, int]:
    """[tổng cặp nguồn–đích, số cặp cùng thành phần liên thông] của khung hiện tại.

    Chỉ đếm trên các nút mà bộ sinh nhãn coi là endpoint (cột `endpoint` của
    oracle_dataset_v5); nếu không có dữ liệu đó thì lấy mọi nút không phải vệ tinh.
    """
    if sides.dtype.kind == "O" or sides.ndim != 1 or sides.size != len(ten_nut):
        sides = np.array([bool(x) and not str(n).upper().startswith("STARLINK")
                          for x, n in zip(sides, ten_nut)])
    idx = np.where(sides.astype(bool))[0]
    if idx.size < 2:
        return 0, 0
    tong = idx.size * (idx.size - 1) // 2
    c = comp[idx]
    _, dem = np.unique(c, return_counts=True)
    toi = int((dem * (dem - 1) // 2).sum())
    return int(tong), toi


def chon_snapshot(n_frame: int, snapshot: int | None, n_snap: int) -> list[int]:
    """Danh sách snapshot cho các khung, canh giữa quanh `snapshot` (mặc định giữa tệp)."""
    tam = n_snap // 2 if snapshot is None else snapshot
    if not 0 <= tam < n_snap:
        raise SystemExit(f"--snapshot {tam} ngoài [0, {n_snap - 1}]")
    dau = min(max(0, tam - n_frame // 2), max(0, n_snap - n_frame))
    return list(range(dau, min(dau + n_frame, n_snap)))


def tuyen_theo_snapshot(data: dict, n: int) -> dict[int, np.ndarray]:
    """Tuyến oracle đầu tiên của mỗi snapshot (chỉ số nút, đã lọc -1)."""
    y = np.asarray(data["y_paths_indices"])
    sid = np.asarray(data["snapshot_id"])
    out: dict[int, np.ndarray] = {}
    for m in range(y.shape[0]):
        s = int(sid[m])
        if s in out:
            continue
        pr = y[m][(y[m] >= 0) & (y[m] < n)]
        if pr.size >= 2:
            out[s] = pr.astype(int)
    return out


def dung_luoi(data: dict, tt: dict, n_frame: int, snapshot: int | None,
              khoang: int, ten_bao: str | None = None,
              kich_ban: int = 0, khong_mo_phong: bool = False) -> dict:
    """Mỗi khung ứng với một snapshot của dataset; giờ thời tiết đi kèm.

    Chế độ bình thường: các snapshot liên tiếp, giờ tiến `khoang` giờ mỗi khung.
    Chế độ nhiều bão: `<tt>["moc"]` đã là danh sách (snapshot, giờ) dựng sẵn — dùng nguyên.
    """
    names = [str(x) for x in data["node_order"]]
    n = len(names)
    layer = np.asarray(data["node_layer"]).astype(str) if "node_layer" in data else \
        np.where(np.char.startswith(np.char.upper(np.asarray(names)), "STARLINK"), SAT, GROUND)
    kind_s = np.asarray(data["node_kind"]).astype(str) if "node_kind" in data else \
        np.where(layer == SAT, "leo", "user")
    kinds = [KIND.get(k, 4) for k in kind_s]
    loai = loai_cap(layer)
    W = np.asarray(data["W_snapshots"], np.float32)
    la = np.asarray(data["node_latlon_alt"], np.float32)
    if la.shape[0] < W.shape[0]:
        # W đã được thay bằng oracle 600 (bão có tuyến DM) -> toạ độ cũng phải đủ dài
        la = doc_khoa(GOC / "dataset" / "oracle_dataset_v5.npz",
                      "node_latlon_alt")["node_latlon_alt"].astype(np.float32)
    # Endpoint của bộ sinh nhãn (oracle_dataset_v5); thiếu thì suy từ node_kind.
    endpoint = (np.asarray(data["endpoint"], dtype=bool) if "endpoint" in data else
                np.asarray(data["node_kind"]).astype(str) != "leo")
    h0_bao = 0
    if ten_bao:
        try:
            _cat = json.loads((GOC / 'data' / 'bao_manh' / 'danh_muc.json')
                              .read_text(encoding='utf-8'))
            _b = next((x for x in _cat['bao'] if x['ten'] == ten_bao), None)
            if _b and _b.get('gio_bat_dau_kich_ban'):
                h0_bao = int(_b['gio_bat_dau_kich_ban'][0])
        except Exception:
            h0_bao = 0

    # Không dùng `link_stats` nữa: bảng đó chỉ có 54 snapshot của v5, lệch pha với
    # khung đang vẽ (bão chạy trên chuỗi 600). Thông số mọi cạnh được tính lại từ
    # toạ độ của chính khung bằng thong_so_canh().

    temps = tuyen_theo_snapshot(data, n)
    # Tuyến DM (nếu có) cho các mốc oracle có trong dữ liệu TN6.
    # Tuyến DM/dphat/dijkstra/thầy cho ĐÚNG bão + kịch bản đang vẽ (Bước C).
    dm_theo_t = nap_tuyen_dm(ten_bao, kich_ban)
    # Tuyến tính riêng cho cặp bất kỳ (xuat_tuyen_cap.py) — nhúng kèm vào dmTuyen
    # dưới khoá `cap` để giao diện tra khi người xem chọn 2 nút ngoài 4–5 luồng.
    cap_theo_t = nap_tuyen_cap(ten_bao)
    dung_tn6 = bool(dm_theo_t)
    if ten_bao and not dm_theo_t:
        print(f"  ! Chưa có tuyến cho bão {ten_bao} kịch bản {kich_ban} trong "
              f"duong_di_nut.json — bản đồ chỉ vẽ mưa/gió + tuyến mặc định.")
    # Chuỗi thời tiết cả năm để tra (ngày, t) của cửa sổ TN6.
    _tt_nam = tt

    ground = np.where(layer == GROUND)[0]
    site_of = np.full(ground.size, -1)
    for k, i in enumerate(ground):
        if names[i] in tt["names"]:
            site_of[k] = tt["names"].index(names[i])
    co_tt = np.where(site_of >= 0)[0]
    air_ground = [i for i in range(n) if layer[i] != SAT]        # UAV + HAPS + ground

    # Danh sách (snapshot, giờ thời tiết, nhãn) cho từng khung
    moc = tt.get("moc")
    if moc:
        ke_hoach = [(int(s), int(g), lab) for s, g, lab in moc]
    else:
        # DỰNG KHUNG từ danh sách mốc có thật của dữ liệu đang nạp:
        #   - có tuyến DM -> mốc `t` của tuyến (75..588 trong chuỗi oracle 600);
        #   - không có DM  -> snapshot CỦA CHÍNH tệp .npz (vd v5/test: 0..53).
        # Luôn kẹp mọi mốc theo số snapshot của W để không bao giờ IndexError
        # (trước đây chọn theo lịch thời tiết nên vượt 54 snapshot của v5).
        n_snap_w = int(W.shape[0])
        if dm_theo_t:
            ung_vien = [int(t) for t in sorted(dm_theo_t) if 0 <= int(t) < n_snap_w]
        else:
            ung_vien = list(range(n_snap_w))
        if not ung_vien:
            ung_vien = list(range(n_snap_w))
        if snapshot is not None:
            tam = min(range(len(ung_vien)), key=lambda q: abs(ung_vien[q] - snapshot))
            dau = max(0, tam - n_frame // 2)
        else:
            dau = max(0, (len(ung_vien) - n_frame) // 2)
        frames_orig = ung_vien[dau:dau + n_frame] or ung_vien[:1]

        so_gio = len(tt["time"])
        # Kế hoạch giờ: nếu CÓ tuyến DM thì rải đều khắp cơn như cũ; nếu KHÔNG có
        # tuyến DM (thời tiết cả năm / tập val) thì lấy từ giờ mưa lớn nhất và đi
        # `khoang` giờ mỗi khung, luôn kẹp trong [0, so_gio-1].
        bo = gio_mua_nhat(tt, n_frame, khoang)
        # Ở chế độ bão: rải khung đều khắp cơn. Không dùng phép % vì sẽ quay vòng
        # về đầu cơn khi số khung × bước vượt quá độ dài cơn.
        if dm_theo_t and bo == 0 and tt.get("gio_bat_dau") is not None and n_frame > 1:
            buoc = max(1, int(round(so_gio / n_frame)))
        else:
            buoc = khoang
        cuoi = so_gio - 1
        trai_deu = buoc != khoang      # rải đều khắp cơn -> chia đúng khoảng mốc

        def _gio(fi: int) -> int:
            if bo < 0:
                return -1
            if trai_deu:
                return 0 if n_frame <= 1 else round(fi * cuoi / (n_frame - 1))
            return min(bo + fi * buoc, cuoi)

        ke_hoach = [(s, _gio(fi), thoi_diem(tt, _gio(fi), s))
                    for fi, s in enumerate(frames_orig)]

    # Nguồn/đích để tô nổi bật: lấy từ tuyến của CHÍNH khung này (tuyến DM nếu có,
    # không thì mẫu đầu tiên của y_paths_indices cùng snapshot).
    y = np.asarray(data["y_paths_indices"])
    sid = np.asarray(data["snapshot_id"])
    kind_s_tat_ca = np.asarray(data["node_kind"]).astype(str)
    src = dst = 0
    for s, _g, _l in ke_hoach:
        _dmk = dm_theo_t.get(int(s))
        if _dmk:
            _luong = next((l for l in _dmk.get("dm") or [] if l.get("nut")), None)
            if _luong:
                src, dst = int(_luong["nut"][0]), int(_luong["nut"][-1])
                break
        m = np.where(sid == s)[0]
        if m.size:
            pr = y[m[0]][y[m[0]] >= 0]
            if pr.size >= 2:
                src, dst = int(pr[0]), int(pr[-1])
                break

    frames, chuoi_ac, chuoi_nb = [], [], []
    for s, gio, nhan in ke_hoach:
        W0 = W[s]
        cap = bang_canh(W0, loai)

        # Mưa / gió theo MẶT ĐẤT — dữ liệu thật, khớp tên điểm.
        # Khung có tuyến DM thì lấy ĐÚNG giờ của cửa sổ TN6 (bao + t) để mưa/gió
        # khớp tuyến; khung khác dùng giờ theo kế hoạch.
        gio_dung = gio
        _dmk = dm_theo_t.get(int(s))
        if _dmk and _dmk.get("bao"):
            _g2 = gio_that_tu_bao(_tt_nam, _dmk["bao"], _dmk["t"])
            if _g2 >= 0:
                gio_dung = _g2
        R = np.zeros(ground.size, np.float32)
        G = np.zeros(ground.size, np.float32)
        if gio_dung >= 0 and co_tt.size:
            nguon_tt = _tt_nam if gio_dung != gio else tt
            R[co_tt] = nguon_tt["precip"][gio_dung, site_of[co_tt]]
            G[co_tt] = nguon_tt["gust"][gio_dung, site_of[co_tt]]

        tuyen = temps.get(s)
        route = tuyen.tolist() if tuyen is not None else []
        adapt = [x for q in range(len(route) - 1) for x in (route[q], route[q + 1])]

        # Chi phí tuyến = tổng trễ của các chặng, lấy thẳng từ W của snapshot này.
        # Khi vẽ bão, tuyến hiển thị là tuyến DM (không phải y_paths_indices), nên
        # ưu tiên tính trễ của tuyến DM; nếu không có mới dùng tuyến mặc định.
        ac = None
        _dm = dm_theo_t.get(int(s)) if dm_theo_t else None
        _tuyen_tinh = None
        if _dm and _dm.get("dm"):
            luong = _dm["dm"]
            for l in luong:
                if int(l.get("luong", 0)) == 0 and l.get("nut"):
                    _tuyen_tinh = [int(x) for x in l["nut"]]
                    break
        if _tuyen_tinh is None and len(route) >= 2:
            _tuyen_tinh = route
        if _tuyen_tinh and len(_tuyen_tinh) >= 2:
            # W0[u,v] là TRỄ CẢ CHẶNG (không phải độ trễ tích luỹ từ nguồn), nên
            # tổng trễ tuyến = tổng các chặng. Chỉ báo `ac` khi mọi chặng còn sống.
            w_seg = [float(W0[a, b]) for a, b in zip(_tuyen_tinh[:-1], _tuyen_tinh[1:])]
            if all(w > 0 for w in w_seg):
                ac = round(sum(w_seg), 2)

        # Thông số hình học của MỌI cạnh tồn tại (W > 0) tại CHÍNH khung này.
        cs_canh = thong_so_canh(la[s], kind_s_tat_ca, W0, cap["u"], cap["v"])
        khoa_canh = (cap["u"].astype(np.int64) * n + cap["v"]).tolist()
        ts_theo_canh = dict(zip(khoa_canh, zip(cs_canh["dist"].tolist(),
                                               cs_canh["elev"].tolist(),
                                               cs_canh["bw"].tolist(),
                                               cs_canh["loss"].tolist())))

        # Thông số từng chặng của TUYẾN ĐANG VẼ; cạnh đứt (W <= 0) ghi 0.
        thong_so_tuyen: list[float] = []
        _tuyen_ls = _tuyen_tinh if _tuyen_tinh else route
        for a, b in zip(_tuyen_ls[:-1], _tuyen_ls[1:]):
            ts = ts_theo_canh.get(min(a, b) * n + max(a, b))
            if ts is None:
                thong_so_tuyen += [0.0, 0.0, 0.0, 0.0]
            else:
                thong_so_tuyen += [round(ts[0], 1), round(ts[1], 1),
                                   round(ts[2], 0), round(ts[3], 4)]
        # Bảng thông số riêng cho TỪNG LUỒNG của tuyến (dm/dphat/dijkstra/thầy) để
        # giao diện đọc đúng luồng đang chọn thay vì dùng chung số của luồng 0.
        _dm_khung = dm_theo_t.get(int(s))
        if _dm_khung:
            _dm_khung = dict(_dm_khung)
            for _loai in ("dm", "dphat", "dijkstra", "thay", "dm_du_phong"):
                _ds = _dm_khung.get(_loai)
                if not isinstance(_ds, list):
                    continue
                for _l in _ds:
                    if not isinstance(_l, dict) or not _l.get("nut"):
                        continue
                    _chuoi = [int(x) for x in _l["nut"]]
                    _ts = []
                    for a, b in zip(_chuoi[:-1], _chuoi[1:]):
                        r = ts_theo_canh.get(min(a, b) * n + max(a, b))
                        _ts += [0.0, 0.0, 0.0, 0.0] if r is None else \
                            [round(r[0], 1), round(r[1], 1), round(r[2], 0), round(r[3], 4)]
                    _l["link"] = _ts
            _dm_khung["ac_theo_luong"] = [
                round(sum(float(W0[a, b]) for a, b in
                          zip([int(x) for x in _l["nut"]][:-1], [int(x) for x in _l["nut"]][1:])), 2)
                if _l.get("nut") and all(W0[a, b] > 0 for a, b in
                                         zip([int(x) for x in _l["nut"]][:-1],
                                             [int(x) for x in _l["nut"]][1:])) else None
                for _l in _dm_khung.get("dm") or [] if isinstance(_l, dict) and _l.get("nut")]
        else:
            _dm_khung = None

        # Tuyến DM/dphat của cặp bất kỳ (tính riêng, 1 luồng) cho ĐÚNG mốc này.
        _cap = cap_theo_t.get(int(s))
        if _cap:
            if _dm_khung is None:
                _dm_khung = {"K": 0, "dm": [], "dphat": [], "dijkstra": [], "thay": [],
                             "nguon": [], "bao": ten_bao, "t": int(s), "kb": kich_ban}
            _ds_cap = []
            for _sn, _dn, _ts_cap in _cap["cap"]:
                _item = {"nguon": _sn, "dich": _dn, "mot_luong": True,
                         "nut": [int(x) for x in _ts_cap["dm"]],
                         "dphat": [int(x) for x in (_ts_cap.get("dphat") or [])],
                         "dijkstra": [int(x) for x in (_ts_cap.get("dijkstra") or [])]}
                _ds_cap.append(_item)
            _dm_khung["cap"] = _ds_cap

        chuoi_ac.append(ac)
        chuoi_nb.append(int(cs_canh["co_so"].sum()))

        # Chạy lại bộ mô phỏng để tách cạnh đứt theo nguyên nhân:
        # mưa (SNR thấp) hay một đầu bị tắt.
        # `--khong-mo-phong`: hiện phút cuối, không chạy kich_ban (nhanh gấp ~15 lần).
        dut_mua, dut_nut, nut_tat, dut_giong, nut_giong = [], [], [], [], []
        if ten_bao and ten_bao in BANG_XAC_NHAN and not khong_mo_phong:
            try:
                dut_mua, dut_nut, nut_tat, dut_giong, nut_giong = dut_gay(
                    W0, data, ten_bao, s, h0_bao, ground)
            except Exception as e:      # thiếu thư viện / tệp: bỏ qua, không chặn bản đồ
                print(f"  ! không tính được đứt gãy cho snapshot {s}: {e}")

        # --- "Không có đường đi" tính trên CHÍNH W của khung này ---
        # reach_stats chỉ có 54 snapshot của v5 nên không dùng: đếm lại trên đồ thị
        # W0 đang vẽ -> số cặp tới được luôn khớp với cạnh đứt hiển thị.
        nhan_tp, so_tp = thanh_phan(W0)
        dem_tp = np.bincount(nhan_tp)
        co_lap = [int(i) for i in range(n) if dem_tp[nhan_tp[i]] == 1]
        tong_cap, toi_duoc = dem_toi_duoc(nhan_tp, names, endpoint)

        # Tập cạnh đứt của kịch bản (mưa / một đầu bị tắt) để loại khỏi lớp sống.
        dut_set = {(a, b) for a, b in zip(dut_mua[0::2] + dut_nut[0::2],
                                          dut_mua[1::2] + dut_nut[1::2])}
        # Cạnh W <= 0 là cạnh đứt THẬT của khung: vẽ màu đỏ và cũng phải biến khỏi
        # lớp sống, nếu không 3D có vạch đứt mà bảng thông số lại báo còn sống.
        dut_that = [int(x) for x, co in zip(np.stack([cap["u"], cap["v"]], 1).reshape(-1),
                                            np.repeat(~cs_canh["co_so"], 2)) if co]
        dut_ve = dut_set.union(zip(dut_that[0::2], dut_that[1::2]))

        frames.append({
            "t": thoi_diem(_tt_nam, gio_dung, s) if gio_dung != gio else nhan,
            # Thông số từng nút tại ĐÚNG khung; tách cạnh sống/đứt theo cạnh đứt thật.
            "nuts": thong_so_nut(data, s, dut_set, set(nut_tat),
                                 W0=W0, latlon_alt=la[s], khoa_canh=khoa_canh,
                                 ts_canh=ts_theo_canh),
            # Tuyến tra theo cặp nút (từ y_paths_indices của snapshot này).
            "routeMap": ban_do_tuyen(data, s, dut_set),
            # Tuyến DM cho khung này (nếu trùng mốc oracle có tuyến).
            "dmTuyen": _dm_khung,
            "p": np.round(la[s].reshape(-1), 4).tolist(),
            "isl": loc_canh(cap, 0, dut_ve),
            "acc": loc_canh(cap, 1, dut_ve) + loc_canh(cap, 2, dut_ve),
            # Đường đi tách theo nút bị tắt: chặng còn dùng được / chặng đứt.
            "adapt": [x for q in range(0, len(adapt), 2)
                      if adapt[q] not in set(nut_tat) and adapt[q + 1] not in set(nut_tat)
                      for x in (adapt[q], adapt[q + 1])],
            "adapt_dut": [x for q in range(0, len(adapt), 2)
                          if adapt[q] in set(nut_tat) or adapt[q + 1] in set(nut_tat)
                          for x in (adapt[q], adapt[q + 1])],
            "R": np.round(R, 2).tolist(),
            "G": np.round(G, 2).tolist(),
            "ac": ac,
            "link": thong_so_tuyen,
            # [tổng cặp endpoint, cặp cùng thành phần liên thông]; hiệu = số cặp mất đường.
            "cap": [tong_cap, toi_duoc],
            # Số liên kết của khung: tồn tại / đã đứt (W <= 0) — dùng cho HUD.
            "nb": [int(cs_canh["co_so"].sum()), int((~cs_canh["co_so"]).sum())],
            "co_lap": co_lap,
            "so_tp": int(so_tp),
            # Cạnh đứt THẬT (W <= 0) của khung, tách thêm theo nguyên nhân mô phỏng:
            "brk": dut_that + dut_mua + dut_nut + dut_giong,   # tất cả cạnh đứt
            "brk_giong": dut_giong,      # đứt khi một đầu nằm vùng giông
            "brk_mua": dut_mua,            # đứt vì mưa
            "brk_nut": dut_nut,            # đứt vì nút bị tắt
            "down": nut_tat,               # nút đang bị tắt (UAV/trạm/HAPS)
            # Ground đang có giông (vẽ tia sét nhấp nháy).
            "storm": nut_giong,
        })

    island = [1 if nm in DAO else 0 for nm in names]

    return {
        "meta": {
            "names": names,
            "kinds": kinds,
            "src": src,
            "dst": dst,
            "ground": [int(i) for i in ground],
            "island": island,
            "stepSec": int(np.asarray(data.get("step_sec", [20])).ravel()[0])
            if "step_sec" in data else 20,
            "muc": "thời tiết thật",
            "nhan": 1,
            "event": "",
            "dungTN6": bool(dung_tn6),
            "series": {"ac": chuoi_ac, "bc": chuoi_ac, "nb": chuoi_nb},
        },
        "frames": frames,
    }



def nap_tuyen_dm(bao: str | None = None, kich_ban: int = 0) -> dict[int, dict]:
    """Đọc `duong_di_nut.json` và trả {mốc t: bản ghi} cho MỘT bão + một kịch bản.

    Dạng bản ghi (do xuat_tuyen_dm.py xuất, Bước B):
      {t, K, kb, nguon, dm, thay, dphat, dijkstra, dm_du_phong}
      trong đó mỗi loại tuyến là list K luồng:
      {luong, nguon, dich, nut: [nút...], day_du}

    `bao`: tên bão cần lọc (vd "soulik_2024"). None = giữ mọi bão.
    `kich_ban`: chỉ số `kb` muốn xem (mặc định 0) — mỗi mốc t có nhiều kịch bản.
    """
    f = GOC / "duong_di_nut.json"
    if not f.exists():
        return {}
    import json as _j
    tat_ca = _j.loads(f.read_text(encoding="utf-8"))
    ra: dict[int, dict] = {}
    for khoa, v in tat_ca.items():
        if not isinstance(v.get("dm"), list) or not v["dm"] or \
                not isinstance(v["dm"][0], dict):
            continue                      # bỏ qua bản ghi dạng cũ
        if bao and v.get("bao") != bao:
            continue
        if int(v.get("kb", 0)) != int(kich_ban):
            continue
        t = int(v["t"])
        if t not in ra:
            ra[t] = v
    return ra


def nap_tuyen_cap(bao: str | None = None) -> dict[int, dict]:
    """Đọc `duong_di_cap.json` (do xuat_tuyen_cap.py sinh) — tuyến DM/dphat cho cặp
    NGUỒN–ĐÍCH bất kỳ, tính ở phạm vi MỘT luồng.

    Trả {mốc t: {cap: [(nguon, dich, {dm, dphat, dijkstra})]}}. Không có file -> {}.
    """
    f = GOC / "duong_di_cap.json"
    if not f.exists():
        return {}
    import json as _j
    tat_ca = _j.loads(f.read_text(encoding="utf-8"))
    ra: dict[int, dict] = {}
    for khoa, v in tat_ca.items():
        if not isinstance(v.get("dm"), list) or not v["dm"]:
            continue
        if bao and v.get("bao") != bao:
            continue
        t = int(v["t"])
        ds = ra.setdefault(t, {"cap": []})
        ds["cap"].append((v["nguon"], v["dich"],
                          {"dm": v["dm"], "dphat": v.get("dphat"),
                           "dijkstra": v.get("dijkstra"), "mot_luong": True}))
    return ra


def anh_xa_khung(oracle_path: Path, test_path: Path) -> dict[int, int]:
    """Ánh xạ khung của bản đồ (test) sang mốc thời gian oracle 600 snapshot.

    Bản đồ dùng 54 snapshot trích từ chuỗi 600; tuyến DM dùng chính chuỗi 600 nên
    phải biết khung nào ứng với mốc nào.
    """
    if not oracle_path.exists() or not test_path.exists():
        return {}
    O = np.load(oracle_path, allow_pickle=False)["W_snapshots"]
    T = np.load(test_path, allow_pickle=False)["W_snapshots"]
    ra = {}
    for k in range(T.shape[0]):
        d = np.abs(O - T[k]).reshape(O.shape[0], -1).max(1)
        j = int(np.argmin(d))
        if d[j] < 1e-6:
            ra[k] = j
    return ra



def gio_that_tu_bao(tt: dict, bao: str, t: int) -> int:
    """Chỉ số giờ thật trong chuỗi thời tiết từ (ngày cửa sổ, offset t).

    `.pkl` ghi `bao` = ngày giông gốc (vd "xau_2025-08-17") và `t` = offset trong
    cửa sổ. Giờ thật = chỉ số của ngày đó lúc 00:00 + t.
    """
    import re as _re
    m = _re.search(r"(\d{4})-(\d{2})-(\d{2})", bao or "")
    if not m or not tt["time"]:
        return -1
    khoa = f"{m.group(1)}-{m.group(2)}-{m.group(3)}T00:00"
    try:
        i0 = tt["time"].index(khoa)
    except ValueError:
        return -1
    i = i0 + int(t)
    return i if i < len(tt["time"]) else -1



# ---------------------------------------------------------------------------
# Đứt gãy thật: chạy lại bộ mô phỏng của dự án (src/ngau_nhien/kich_ban.py)
# ---------------------------------------------------------------------------


def tham_so_khac_nghiet() -> "ThamSo":
    """Tham số mức 'khắc nghiệt' lấy nguyên từ data/bao_manh/bang_tra/meta.json."""
    from src.ngau_nhien.kich_ban import ThamSo

    meta = json.loads((GOC / "data" / "bao_manh" / "bang_tra" / "meta.json")
                      .read_text(encoding="utf-8"))
    t = meta["parameters"]
    return ThamSo(
        mua=t["mua"], giong=t["giong"], xung_dot=t["xung_dot"], tai=t["tai"],
        dt_s=t["dt_s"], phi_mua=t["phi_mua"], sigma_mua=t["sigma_mua"],
        snr0_db=t["snr0_db"], snr_min_db=t["snr_min_db"],
        gio_giat_max=t["gio_giat_max"], uav_ha_canh=t["uav_ha_canh"],
        p_set=t["p_set"], set_tb=t["set_tb"],
    )


_DUT_CACHE: dict = {}


def dut_gay(W0: np.ndarray, data: dict, ten_bao: str, snapshot: int,
            h0: int, ground: np.ndarray, buoc: int = 40):
    """Chạy sinh_kich_ban() trên một snapshot, trả về 2 danh sách cặp đứt.

    Dùng lặp snapshot thành `buoc` bước thời gian để mô phỏng dao động mưa/gió;
    lấy bước giữa làm đại diện.

    Trả (dut_mua, dut_nut, nut_tat, dut_giong) — mỗi danh sách cặp dạng [u,v,u,v,...]:
      dut_mua   SNR thấp do mưa (suy hao), không nút nào bị tắt
      dut_giong như trên nhưng một đầu nằm vùng có giông (sét/nhiễu đi kèm)
      dut_nut   một đầu bị tắt: UAV hạ cánh, trạm gặp sét, hoặc xung đột
      nut_tat   các nút đang bị tắt ở bước đó
    """
    # Cùng (bão, snapshot, số bước) cho cùng kết quả -> cache để không dựng lại MangGoc
    # (224x224 vòng lặp) cho mỗi khung.
    khoa = (ten_bao, int(snapshot), int(buoc))
    if khoa in _DUT_CACHE:
        return _DUT_CACHE[khoa]

    from src.ngau_nhien.kich_ban import MangGoc, sinh_kich_ban
    from src.ngau_nhien.thoi_tiet import ThoiTiet

    names = [str(x) for x in data["node_order"]]
    kind = [str(x) for x in data["node_kind"]]
    tep = GOC / "data" / "bao_manh" / f"{ten_bao}.npz"
    if not tep.exists():                      # thời tiết cả năm
        tep = GOC / "data" / f"{ten_bao}.npz"
    tt = ThoiTiet(tep)
    W3 = np.repeat(W0[None], buoc, axis=0)          # cùng hình học, thời tiết dao động
    g = MangGoc(W3, kind, names, tt)
    Wn, qs = sinh_kich_ban(g, tt, int(h0), 12345, tham_so_khac_nghiet())

    t = buoc // 2
    tat, song, giong = qs["tat"][t], qs["song"][t], qs["giong"][t]
    co_goc = g.W[t] > 0
    hai_dau_tat = tat[:, None] | tat[None, :]
    dut = co_goc & ~song

    def cap(mask):
        iu, iv = np.nonzero(np.triu(mask, 1))
        out = np.empty(iu.size * 2, np.int32)
        out[0::2], out[1::2] = iu, iv
        return out.tolist()

    nut = dut & hai_dau_tat            # một đầu bị tắt
    khong_nut = dut & ~hai_dau_tat      # không nút nào tắt -> suy hao

    # Cạnh nằm trong vùng giông: ít nhất một đầu ở điểm thời tiết đang có giông.
    site = np.maximum(g.node_site, 0)
    nut_giong_mask = (g.node_site >= 0) & giong[site]
    co_giong = nut_giong_mask[:, None] | nut_giong_mask[None, :]

    do_giong = khong_nut & co_giong
    do_mua = khong_nut & ~co_giong

    # Ground đang có giông, theo CHỈ SỐ TRONG M.ground (cùng hệ với R/G) để renderer
    # đối chiếu được với mảng thời tiết.
    idx_ground = {int(i): k for k, i in enumerate(ground)}
    nut_giong = [idx_ground[int(i)] for i in np.where(g.dat & nut_giong_mask)[0]
                 if int(i) in idx_ground]

    ket_qua = (cap(do_mua), cap(nut), [int(i) for i in np.where(tat)[0]],
               cap(do_giong), nut_giong)
    _DUT_CACHE[khoa] = ket_qua
    return ket_qua


def thong_so_nut(data: dict, snapshot: int, canh_dut: set = frozenset(),
                 nut_ngat: set = frozenset(), W0: np.ndarray | None = None,
                 latlon_alt: np.ndarray | None = None, khoa_canh: list[int] | None = None,
                 ts_canh: dict | None = None) -> list[dict]:
    """Thông số từng nút tại ĐÚNG khung đang vẽ (W + toạ độ của chính khung đó).

    3 nhóm theo yêu cầu báo cáo:
      mat_dat  <- gateway + user (mặt đất)
      uav      <- UAV
      ve_tinh  <- vệ tinh LEO

    "Liên kết" là mọi cặp có luật (cột `exists` của oracle); "còn dùng được" là
    cặp có W > 0 và không nằm trong `canh_dut` (kịch bản mưa / nút bị tắt). Nhờ vậy
    3D vẽ vạch đứt và bảng thông số luôn nói cùng một chuyện.
    """
    names = [str(x) for x in data["node_order"]]
    kind = np.asarray(data["node_kind"]).astype(str)
    n = len(names)
    W_all = np.asarray(data["W_snapshots"], np.float32)
    if W0 is None:
        W0 = W_all[snapshot]
    if latlon_alt is None:
        la_all = np.asarray(data["node_latlon_alt"], np.float32)
        if snapshot >= la_all.shape[0]:
            # snapshot vượt số khung của tệp (bão dùng chuỗi 600 của oracle)
            la_all = doc_khoa(GOC / "dataset" / "oracle_dataset_v5.npz",
                              "node_latlon_alt")["node_latlon_alt"].astype(np.float32)
        latlon_alt = la_all[snapshot]
    if khoa_canh is None or ts_canh is None:
        iu, iv = np.nonzero(np.triu(W_all[snapshot] > 0, 1))
        cs = thong_so_canh(latlon_alt, kind, W0, iu, iv)
        khoa_canh = (iu.astype(np.int64) * n + iv).tolist()
        ts_canh = dict(zip(khoa_canh, zip(cs["dist"].tolist(), cs["elev"].tolist(),
                                          cs["bw"].tolist(), cs["loss"].tolist())))

    # Cạnh (i, j) có luật liên kết: đọc từ khoá cạnh (chỉ chứa cạnh W > 0) cộng mọi
    # cặp có trong kịch bản đứt; cặp không tồn tại và không đứt thì không có luật.
    theo_nut: dict[int, list[tuple[int, float, float, float, float]]] = {i: [] for i in range(n)}
    for khoa, (dist, elev, bw, loss) in ts_canh.items():
        a, b = divmod(khoa, n)
        theo_nut[a].append((b, dist, elev, bw, loss))
        theo_nut[b].append((a, dist, elev, bw, loss))
    for a, b in canh_dut:
        if min(a, b) * n + max(a, b) in ts_canh:
            continue
        theo_nut[min(a, b)].append((max(a, b), 0.0, 0.0, 0.0, 0.0))
        theo_nut[max(a, b)].append((min(a, b), 0.0, 0.0, 0.0, 0.0))

    nhom_cua = {"gateway": "mat_dat", "user": "mat_dat", "uav": "uav", "leo": "ve_tinh"}
    ra = []
    for i, nm in enumerate(names):
        nhom = nhom_cua.get(kind[i])
        if nhom is None:
            continue
        lk = theo_nut[i]
        song = [(j, d, e, bw, ls) for (j, d, e, bw, ls) in lk
                if W0[i, j] > 0 and (min(i, j), max(i, j)) not in canh_dut
                and i not in nut_ngat and j not in nut_ngat]
        dist = [d for (_j, d, e, bw, ls) in song]
        bw_lk = [bw for (_j, d, e, bw, ls) in song]
        loss_lk = [ls for (_j, d, e, bw, ls) in song]
        elev_lk = [e for (_j, d, e, bw, ls) in song]
        ra.append({
            "nhom": nhom, "ten": nm, "loai_goc": kind[i],
            "lat": round(float(latlon_alt[i, 0]), 4),
            "lon": round(float(latlon_alt[i, 1]), 4),
            "cao_km": round(float(latlon_alt[i, 2]), 2),
            "so_link": len(lk),
            "so_song": len(song),
            "so_dut": len(lk) - len(song),
            "km_tb": round(float(np.mean(dist)), 1) if dist else None,
            "km_max": round(float(np.max(dist)), 1) if dist else None,
            "elev_tb": round(float(np.mean(elev_lk)), 1) if elev_lk else None,
            "bw_tb": round(float(np.mean(bw_lk)), 0) if bw_lk else None,
            "loss_max": round(float(np.max(loss_lk)), 4) if loss_lk else None,
        })
    return ra


def ban_do_tuyen(data: dict, snapshot: int, dut_set: set) -> dict:
    """Bảng tra cặp nút -> tuyến, dựng TỪ y_paths_indices của chính snapshot đó.

    Khoá "min-max" của cặp (u, v); giá trị {p: [u,v,...], d: [u,v,...], c: tổng trễ ms}.
    Chỉ chứa các cặp CÓ mẫu trong dataset — không tự bịa tuyến.
    """
    y = np.asarray(data["y_paths_indices"])
    sid = np.asarray(data["snapshot_id"])
    W = np.asarray(data["W_snapshots"])
    n = W.shape[1]
    ra = {}
    for m in np.where(sid == snapshot)[0]:
        route = [int(x) for x in y[m] if 0 <= x < n]
        if len(route) < 2:
            continue
        a, b = route[0], route[-1]
        khoa = f"{min(a, b)}-{max(a, b)}"
        if khoa in ra:
            continue
        pairs = [x for q in range(len(route) - 1) for x in (route[q], route[q + 1])]
        dut = [x for q in range(0, len(pairs), 2)
               if (pairs[q], pairs[q + 1]) in dut_set for x in (pairs[q], pairs[q + 1])]
        W0 = W[snapshot]
        tong = 0.0
        for u, v in zip(route[:-1], route[1:]):
            if W0[u, v] > 0:
                tong += float(W0[u, v])
            else:
                tong = None
                break
        ra[khoa] = {"p": pairs, "d": dut,
                    "c": None if tong is None else round(tong, 2)}
    return ra



# ---------------------------------------------------------------------------
# Xuất HTML
# ---------------------------------------------------------------------------


def duong_ngan_nhat(W0: np.ndarray, kind: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Dijkstra mọi cặp trên W của MỘT khung (chi phí = trễ ms).

    Luật chuyển tiếp giống dữ liệu train (bai_toan_rr.DoThi._giu):
        can_relay[u] or u == nguồn
    tức nút `user` chỉ không được làm TRUNG GIAN; nó vẫn được là điểm xuất phát.
    Trả (dist [N,N], pred [N,N]) — pred[a, b] = nút trước b trên đường a -> b.

    Mẹo: bỏ cạnh đi ra khỏi `user` một lần, rồi bù riêng cho từng nguồn là user
    (chạy Dijkstra thêm một lượt trên đồ thị đầy đủ) — tránh phải nhân bản đồ thị N lần.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra

    n = W0.shape[0]
    w = W0.astype(np.float64)
    eu, ev = np.nonzero(np.triu(w > 0, 1))
    relay = kind != "user"
    k1, k2 = relay[eu], relay[ev]
    rows = np.concatenate([eu[k1], ev[k2]])
    cols = np.concatenate([ev[k1], eu[k2]])
    data = np.concatenate([w[eu, ev][k1], w[ev, eu][k2]])
    G = coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    dist, pred = dijkstra(G, directed=True, return_predecessors=True)

    # Nguồn là user: cho nó đi ra bằng đồ thị đầy đủ (cả hai chiều W > 0).
    nguon_user = np.where(~relay)[0]
    if nguon_user.size:
        rows2 = np.concatenate([eu, ev])
        cols2 = np.concatenate([ev, eu])
        data2 = np.concatenate([w[eu, ev], w[ev, eu]])
        G2 = coo_matrix((data2, (rows2, cols2)), shape=(n, n)).tocsr()
        d2, p2 = dijkstra(G2, directed=True, return_predecessors=True)
        dist[nguon_user] = d2[nguon_user]
        pred[nguon_user] = p2[nguon_user]
    return dist, pred


def dong_goi_pred(pred: np.ndarray) -> str:
    """Nén pred thành chuỗi base64 (float32) để nhúng gọn vào HTML."""
    import base64
    raw = pred.astype(np.float32).tobytes()
    return base64.b64encode(np.frombuffer(raw, np.uint8)).decode("ascii")


def doc_bo_cuc() -> tuple[str, str, str]:
    """Đọc 3 mảnh giao diện trong assets_ban_do/ và kiểm tra chúng còn đúng mốc."""
    dau, than, ve = (ASSET / "khung_dau.txt"), (ASSET / "khung_than.txt"), (ASSET / "ve_3d.js")
    if not all(p.exists() for p in (dau, than, ve)):
        raise SystemExit(f"Thiếu mẫu giao diện trong {ASSET} (khung_dau.txt, khung_than.txt, ve_3d.js)")
    d = dau.read_text(encoding="utf-8")
    t = than.read_text(encoding="utf-8")
    v = ve.read_text(encoding="utf-8")
    if 'id="gl"' not in t or "const COUNTRIES" not in t:
        raise SystemExit("khung_than.txt hỏng")
    if "const M = D.meta" not in v:
        raise SystemExit("ve_3d.js hỏng")
    return d, t, v


def xuat_html(D: dict, ra: Path) -> None:
    """Ghép khung + dữ liệu JSON + ve_3d.js thành một file HTML tự chứa.

    Trễ và thông số từng luồng đã nằm trong D["frames"][i]["dmTuyen"]
    (ac_theo_luong, *.link) nên không nhúng thêm khối DO_TRE_TUYEN nữa.
    """
    d, t, v = doc_bo_cuc()

    # Bảng pred Dijkstra cho TỪNG khung: nhờ vậy giao diện dựng được đường đi cho
    # bất kỳ cặp nút nào, không chỉ 4-5 luồng có sẵn trong dữ liệu.
    oracle = GOC / "dataset" / "oracle_dataset_v5.npz"
    P = {}
    with np.load(oracle, allow_pickle=False) as raw:
        W = raw["W_snapshots"]
        kind = np.asarray(raw["node_kind"]).astype(str)
        for f in D["frames"]:
            seg = f.get("dmTuyen") or {}
            s = int(seg.get("t", -1))
            if 0 <= s < W.shape[0] and str(s) not in P:
                _, pred = duong_ngan_nhat(W[s], kind)
                P[str(s)] = dong_goi_pred(pred)

    js = json.dumps(D, ensure_ascii=False, separators=(",", ":"))
    pj = json.dumps(P, separators=(",", ":"))
    html = d + t + "const D = " + js + ";\nconst PRED = " + pj + ";\n" + v
    ra.parent.mkdir(parents=True, exist_ok=True)
    ra.write_text(html, encoding="utf-8", newline="")
    print(f"Đã tạo: {ra}  ({ra.stat().st_size / 1e6:.1f} MB)")


def main() -> Path:
    """Điểm vào CLI: chọn dữ liệu/thời tiết, dựng lưới khung, xuất HTML."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="vao", default=None, help="tệp .npz (mặc định dataset/v5/test.npz)")
    ap.add_argument("--out", default="ban_do_3d.html", help="tệp HTML xuất ra")
    ap.add_argument("--snapshot", type=int, default=None, help="snapshot giữa khung hình")
    ap.add_argument("--khoang-gio", type=int, default=1, help="mỗi khung tiến bao nhiêu giờ")
    ap.add_argument("--bao", default=None,
                    help="tên bão (soulik, bualoi, ...) hoặc bỏ trống. Để có tuyến DM+dphat "
                         "phải chạy tao_du_lieu_bao -> xuat_tuyen_dm (-> them_dijkstra) "
                         "(xem docstring đầu tệp)")
    ap.add_argument("--kich-ban", type=int, default=0,
                    help="chỉ số kịch bản kb muốn xem (mỗi mốc t có nhiều kb; mặc định 0)")
    ap.add_argument("--frames", type=int, default=0,
                    help="số khung; ở chế độ bão rải đều khắp cơn (mặc định 40)")
    ap.add_argument("--thoi-tiet-binh-thuong", action="store_true",
                    help="dùng thời tiết cả năm thay vì bão")
    ap.add_argument("--liet-ke", action="store_true", help="liệt kê tệp dữ liệu và trạng thái")
    ap.add_argument("--khong-mo", action="store_true", help="không tự mở trình duyệt")
    ap.add_argument("--khong-mo-phong", action="store_true",
                    help="bỏ bước chạy lại bộ mô phỏng đứt gãy theo mưa/gió (nhanh ~15x, "
                         "vẫn vẽ đủ liên kết và cạnh đứt thật W<=0)")
    a = ap.parse_args()

    if a.liet_ke:
        in_liet_ke()
        return Path()

    src = tim_nguon(a.vao)
    print(f"Đọc: {src.relative_to(GOC) if src.is_relative_to(GOC) else src}")
    data = doc_khoa(src, "W_snapshots", "node_order", "node_layer", "node_kind",
                    "node_latlon_alt", "snapshot_id", "y_paths_indices",
                    "link_stats", "link_stat_cols", "reach_stats")
    n_snap = np.asarray(data["W_snapshots"]).shape[0]
    layer = np.asarray(data["node_layer"]).astype(str)
    gnd = [str(nm) for nm, ly in zip(data["node_order"], layer) if ly == GROUND]

    # --bao <ten>  : thời tiết của đúng cơn bão (có đứt do mưa/gió)
    # (không --bao): thời tiết cả năm, lấy ngày mưa nhiều nhất (không có đứt do bão)
    tt = chon_nguon_thoi_tiet(a.bao, uu_tien_bao=not a.thoi_tiet_binh_thuong,
                              n_snap=n_snap, diem_ground=gnd)
    print(f"Thời tiết: {tt['ten']} ({tt['tep'] or 'không có'})")
    n_khung = a.frames or min(n_snap, len(tt["time"]))
    print(f"Số khung: {n_khung}")
    # `ten_tep` của nguồn thời tiết bão trùng `bao` trong duong_di_nut.json
    # (vd "soulik_2024"); khi dùng thời tiết bình thường thì không có tên bão -> None.
    _ten_bao = tt.get("ten_tep") if a.bao else None
    # Nếu bão này có tuyến DM: `t` là mốc trong chuỗi 600 snapshot của oracle, nên
    # thay `W_snapshots` (54 của v5/test) bằng bản đầy đủ để không lỗi chỉ số.
    if _ten_bao and nap_tuyen_dm(_ten_bao, a.kich_ban):
        _oracle = GOC / "dataset" / "oracle_dataset_v5.npz"
        if _oracle.exists():
            data["W_snapshots"] = doc_khoa(_oracle, "W_snapshots")["W_snapshots"]
            print(f"  W: dùng oracle_dataset_v5 ({data['W_snapshots'].shape[0]} snapshot)")
    D = dung_luoi(data, tt, n_khung, a.snapshot, a.khoang_gio,
                  ten_bao=_ten_bao, kich_ban=a.kich_ban,
                  khong_mo_phong=a.khong_mo_phong)
    meta = D["meta"]
    _co_dm = bool(D["meta"].get("dungTN6"))
    _che_do = (f"{tt['ten']} · dphat vs DM" if _co_dm
               else f"{tt['ten']} · Dijkstra thuần (không có DM)")
    meta["event"] = f"{_che_do} · {len(meta['names'])} nút · {len(D['frames'])} khung"
    ra = Path(a.out)
    if not ra.is_absolute():
        ra = GOC / ra
    xuat_html(D, ra)
    k = meta["kinds"]
    print("  " + ", ".join(f"{ten} {k.count(v)}" for ten, v in
                           [("vệ tinh", 0), ("HAPS", 1), ("UAV", 2), ("gateway", 3), ("user", 4)]))
    f0 = D["frames"][0]
    rmax = max(f0["R"]) if f0["R"] else 0.0
    print(f"  liên kết: {(len(f0['acc']) + len(f0['isl'])) // 2} | "
          f"mưa lớn nhất {rmax:.1f} mm/h | trễ tuyến {f0['ac']} ms")
    if not a.khong_mo:
        webbrowser.open(ra.resolve().as_uri())
    return ra


if __name__ == "__main__":
    main()
