"""
tn15.py — Thí nghiệm 15 (results/bao_da_tuyen/TIEU_CHI_15.md): DM học theo J VẬT LÝ (X = 10 ms cho 1 điểm % mất DV) thay cho
J chuẩn hoá 7/3. Cơ sở: ở TN13, thầy theo x10 đạt mất DV 26,65% / trễ 18,6 ms trên val, so với 26,73% / 24,8 ms của 7/3 (chưa
CI, chưa có DM). TN15 kiểm phát hiện đó với CI trên thầy, rồi học DM theo x10 và chấm trên 11 bộ kiểm tra; đồng thời xác
nhận trước (đăng ký) phát hiện phụ của TN12b: giải mã KHÔNG tái nhiễu tốt hơn.

    python -m scripts.tn15 toi-da          # chạy hết — trên máy đã chạy TN13 (dùng lại data/bao_da_tuyen/tn13/)

Từng bước (chạy lại an toàn — phần đã xong tự bỏ qua):
    chuan-bi    kiểm / sinh dữ liệu TN13 cần dùng: val giai đoạn A (thầy theo mọi mục tiêu), 11 bộ kiểm tra có lưới Dijkstra
    sinh        gán nhãn lại tập học (train10, train6_10, x11) theo x10 → data/bao_da_tuyen/tn13/x10/
    kiem        tình huống sinh lại có trùng bản gốc (cảnh báo nếu < 99%)
    huan-luyen  dm13_x10_s0..2 (cấu hình DM11, checkpoint theo J x10 trên val) → results/bao_da_tuyen/mo_hinh13/
    cham        11 bộ × 3 seed: DM11 / DM x10, mỗi mô hình giải mã đầy đủ và không tái nhiễu → tn15/cham_tep/
    phan-tich   → tn15/KET_QUA_15.md, ket_qua_15.json, hinh_15_*.png; gói results/bao_da_tuyen/TN15_ket_qua.zip
Log: logs/tn15.log; mỗi mô hình logs/tn15_<tên>.log.
"""

from __future__ import annotations

import argparse
import glob
import json
import pickle
import shutil
import sys
import time
from pathlib import Path

import numpy as np

GOC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GOC))
from scripts import tn11 as t11  # noqa: E402
from scripts import tn12 as t12  # noqa: E402
from scripts import tn13 as t13  # noqa: E402

KQ, DL, LOG, M11 = t11.KQ, t11.DL, t11.LOG, t11.M11
RA, M13, DL13 = KQ / "tn15", t13.M13, t13.DL13
BO_45, CUC_DOAN, TEN_BO, DO, SEEDS = t11.BO_45, t11.CUC_DOAN, t11.TEN_BO, t11.DO, t11.SEEDS
MT = "x10"                                               # mục tiêu J vật lý của TN15
CFG = t12.CFG
KHONG_KEM = 0.10                                         # điểm % — cận trên CI của hiệu mất DV
NGUONG_DUOI = 100.0                                      # ms — "tình huống nghẽn nặng"
J_MAT, J_A, J_B, J_TRE = 0, 2, 3, 4
BIEN = [("DM11 (7/3)", "dm"), ("DM11 (7/3), không tái nhiễu", "dm_ktn"),
        ("DM x10", "x10"), ("DM x10, không tái nhiễu", "x10_ktn")]


def dp_chon():
    """Bộ β, γ của Dijkstra phạt chọn cho x10 trên val (TN13, chon_b.json); thiếu → dp_50_50 (TN13 chọn bộ này cho x10)."""
    t = t13.RA / "chon_b.json"
    if t.exists():
        return json.loads(t.read_text(encoding="utf-8")).get("dp_chon", {}).get(MT, "dp_50_50")
    return "dp_50_50"


# ============================================================================================ dữ liệu
def lenh_chuan_bi(a):
    thieu = [f"{h}_s{s}" for h in ("dm11",) for s in SEEDS if not (M11 / f"{h}_s{s}.pt").exists()]
    if thieu:
        raise SystemExit(f"thiếu mô hình TN11 ở {M11}: {thieu} (máy khác máy TN11: thêm --tn11 như TN12)")
    if not all((DL13 / d / ".xong").exists() for d in t13.VAL.values()):
        print("  chưa có val giai đoạn A của TN13 → sinh (thầy theo mọi mục tiêu của TN13)", flush=True)
        t13.lenh_sinh_a(a)
    for b in BO_45:
        t13.sinh(b, DL13 / "kiem_tra", a, "--khong-thay", "--luoi-dphat")


def lenh_sinh(a):
    for tap in t13.HOC:
        t13.sinh(tap, DL13 / MT, a, "--muc-tieu", MT, "--chi-nhan")


def lenh_kiem(a):
    nhan = lambda x, y: ("nhan_goc" not in y) or np.array_equal(x["nhan"], y["nhan_goc"])
    dp = lambda x, y: abs(x["ens"]["dphat"]["mat_dv_%"] - y["ens"]["dphat"]["mat_dv_%"]) < 1e-9
    cap = [(DL / d, DL13 / MT / d, nhan) for d in t13.HOC.values()]
    cap += [(DL / t11.thu_muc_bo(b, "4-5"), DL13 / "kiem_tra" / t11.thu_muc_bo(b, "4-5"), dp) for b in BO_45]
    t13.kiem_tai_lap(cap, "tn15")
    RA.mkdir(parents=True, exist_ok=True)
    shutil.copy(t13.RA / "kiem_tai_lap_tn15.json", RA / "kiem_tai_lap.json")


# ============================================================================================ huấn luyện
def lenh_huan_luyen(a):
    thieu = [d for d in t13.HOC.values() if not (DL13 / MT / d / ".xong").exists()]
    if thieu:
        raise SystemExit(f"chưa gán nhãn lại theo {MT}: {thieu} — chạy bước sinh")
    viec = [(f"dm13_{MT}_s{s}", t12.lenh_hoc("dm", 18, f"dm13_{MT}_s{s}", s, a, ["--muc-tieu", MT])) for s in SEEDS]
    t12.hang_doi(viec, a.song_song, M13, "tn15")


# ============================================================================================ chấm
def _cham_tep(job):
    import torch

    from src.nhieu_luong import mo_hinh_dt as md
    from src.nhieu_luong.cham_lai_dt import giong_ban_ghi
    from src.nhieu_luong.cham_mo_hinh_dt import _tg
    tep_goc, tep_moi = job
    torch.set_num_threads(1)
    goc = pickle.load(open(tep_goc, "rb"))
    moi = pickle.load(open(tep_moi, "rb")) if Path(tep_moi).exists() else [None] * len(goc)
    G = lambda sd: torch.Generator().manual_seed(sd)
    ra = []
    for r, m in zip(goc, moi):
        gon = r["gon"]
        tgw, sd, x = _tg(gon, r["cham"]), giong_ban_ghi(r), t11._co_ban(r)
        x["khop"] = bool(m is not None and m["t"] == r["t"] and m["muc"] == r["muc"])
        if x["khop"]:
            x["luoi"] = {g: [float(v[k]) for k in DO] for g, v in m["ens"].items() if g.startswith("dp_")}
        for s in SEEDS:
            for k, net in (("dm", t11._net(str(M11), f"dm11_s{s}")), ("x10", t11._net(str(M13), f"dm13_{MT}_s{s}"))):
                x[f"{k}_s{s}"] = t11._do(gon, tgw, md.chon_dm(net, gon, CFG["S"], 5, G(sd)))
                x[f"{k}_ktn_s{s}"] = t11._do(gon, tgw, md.chon_sua_dan(net, gon, CFG["S"], 5, G(sd)))
        ra.append(x)
    return tep_goc, ra


def lenh_cham(a):
    thieu = [f"dm13_{MT}_s{s}" for s in SEEDS if not (M13 / f"dm13_{MT}_s{s}.xong").exists()]
    if thieu:
        raise SystemExit(f"chưa xong mô hình TN15: {thieu}")
    viec = []
    for b in BO_45:
        d = t11.thu_muc_bo(b, "4-5")
        viec += [(f"{d}__{Path(t).stem}.json", (t, str(DL13 / "kiem_tra" / d / Path(t).name)))
                 for t in sorted(glob.glob(str(DL / d / "kb_*.pkl")))]
    if not viec:
        raise SystemExit(f"không thấy bộ kiểm tra TN11 ở {DL}")
    print(f"  chấm bộ kiểm tra: {len(viec)} tệp", flush=True)
    t11._chay_tep(_cham_tep, viec, a.tien_trinh, RA / "cham_tep")


# ============================================================================================ phân tích
def _ho(rs, k, j):
    return t11._ho(rs, k, SEEDS, j)


def _tom(mat, tre, a_, b_):
    return {"mat_dv": float(mat.mean()), "a": float(a_.mean()), "b": float(b_.mean()), "tre": float(tre.mean()),
            "tre_trung_vi": float(np.median(tre)), "tre_duoi": float(np.mean(tre > NGUONG_DUOI) * 100)}


def phan_tich_thay():
    """H15.0 — thầy theo x10 so với thầy 7/3 trên val giai đoạn A của TN13 (cùng tình huống, cùng thế giới nhãn), CI theo cửa
    sổ. Chẩn đoán xác nhận phát hiện của TN13 (TN13 chỉ có trung bình)."""
    rs, cum = [], []
    for i, d in enumerate(t13.VAL.values()):
        for f in sorted(glob.glob(str(DL13 / d / "kb_*.pkl"))):
            for r in pickle.load(open(f, "rb")):
                if f"thay_{MT}" in r.get("ens", {}) and "thay_w0.7" in r["ens"]:
                    rs.append(r)
                    cum.append(i * 1000 + int(r["kb"]))
    if not rs:
        return None
    cum = np.array(cum)
    lay = lambda ten, m: np.array([r["ens"][ten][m] for r in rs], float)
    m73, t73, mx, tx = lay("thay_w0.7", "mat_dv_%"), lay("thay_w0.7", "tre_chinh_ms"), lay(f"thay_{MT}", "mat_dv_%"), \
        lay(f"thay_{MT}", "tre_chinh_ms")
    return {"n": len(rs), "thay_73": {"mat_dv": float(m73.mean()), "tre": float(t73.mean()), "tre_trung_vi": float(np.median(t73)),
                                      "tre_duoi": float(np.mean(t73 > NGUONG_DUOI) * 100)},
            "thay_x10": {"mat_dv": float(mx.mean()), "tre": float(tx.mean()), "tre_trung_vi": float(np.median(tx)),
                         "tre_duoi": float(np.mean(tx > NGUONG_DUOI) * 100)},
            "hieu_mat": t11._bs(mx - m73, cum), "hieu_tre": t11._bs(tx - t73, cum),
            "hieu_duoi": t11._bs(((tx > NGUONG_DUOI).astype(float) - (t73 > NGUONG_DUOI)) * 100, cum)}


def phan_tich_bo(T, g_dp):
    bo, gop = {}, {}
    for b, rs in T.items():
        rs = [r for r in rs if r.get("khop")]
        kb = np.array([r["kb"] for r in rs])
        v = {k: {j: _ho(rs, k, j) for j in (J_MAT, J_A, J_B, J_TRE)} for _, k in BIEN}
        v["dp"] = {j: np.array([r["luoi"][g_dp][j] for r in rs], float) for j in (J_MAT, J_A, J_B, J_TRE)}
        duoi = lambda k: (v[k][J_TRE] > NGUONG_DUOI).astype(float) * 100
        d = lambda x, y, j=J_MAT: t11._bs(v[x][j] - v[y][j], kb)
        bo[b] = {"n": len(rs), "trung_binh": {ten: _tom(*(v[k][j] for j in (J_MAT, J_TRE, J_A, J_B)))
                                              for ten, k in BIEN + [(f"Dijkstra phạt ({g_dp})", "dp")]},
                 "H15.1_mat": d("x10", "dm"), "H15.1_tre": d("x10", "dm", J_TRE),
                 "H15.2_duoi": t11._bs(duoi("x10") - duoi("dm"), kb),
                 "H15.3_mat": d("x10", "dp"), "x10_tre_vs_dp": d("x10", "dp", J_TRE),
                 "x10_duoi_vs_dp": t11._bs(duoi("x10") - duoi("dp"), kb),
                 "H15.4_mat": d("dm_ktn", "dm"), "ktn_x10_mat": d("x10_ktn", "x10"),
                 "x10ktn_vs_dm_mat": d("x10_ktn", "dm"), "x10ktn_vs_dm_tre": d("x10_ktn", "dm", J_TRE),
                 "x10_a": d("x10", "dm", J_A), "x10_b": d("x10", "dm", J_B)}
        for k in [k for _, k in BIEN] + ["dp"]:
            for j in (J_MAT, J_TRE):
                gop.setdefault((k, j), []).append(v[k][j])
    tong = {k: {"mat_dv": float(np.concatenate(gop[(k, J_MAT)]).mean()), "tre": float(np.concatenate(gop[(k, J_TRE)]).mean()),
                "tre_trung_vi": float(np.median(np.concatenate(gop[(k, J_TRE)]))),
                "tre_duoi": float(np.mean(np.concatenate(gop[(k, J_TRE)]) > NGUONG_DUOI) * 100)}
            for k in [k for _, k in BIEN] + ["dp"]}
    luoi_gop = {}
    for rs in T.values():
        for r in rs:
            if r.get("khop"):
                for g, x in r["luoi"].items():
                    luoi_gop.setdefault(g, []).append((x[J_MAT], x[J_TRE]))
    tong["_luoi"] = {g: [float(np.mean([p[0] for p in x])), float(np.mean([p[1] for p in x]))] for g, x in luoi_gop.items()}
    return bo, tong


def doc_hoc():
    ra = {}
    for f in sorted(M13.glob(f"dm13_{MT}_s*_nhat_ky.json")):
        j = json.loads(f.read_text(encoding="utf-8"))
        nk = j["nhat_ky"]
        b = min(nk, key=lambda r: r["J_val"])
        ra[f.name.replace("_nhat_ky.json", "")] = {
            "so_epoch": len(nk), "epoch_tot": b["epoch"], "ti_le_val_train": b["loss_val"] / b["loss_train"],
            "J_val_tot": b["J_val"], "J_thay": b["J_thay_val"],
            "chua_toi": len(nk) >= CFG["epochs"] and b["epoch"] >= 0.9 * CFG["epochs"],
            "nk": [[r["epoch"], r["loss_train"], r["loss_val"], r["J_val"]] for r in nk]}
    return ra


def gia_thuyet(bo):
    n = len(bo)
    can = max(1, int(np.ceil(8 / 11 * n)))
    hi = lambda ci: ci[1][1]
    dem = lambda f: [b for b, x in bo.items() if f(x)]
    h1 = dem(lambda x: hi(x["H15.1_mat"]) < KHONG_KEM and hi(x["H15.1_tre"]) < 0)
    h2 = dem(lambda x: hi(x["H15.2_duoi"]) < 0)
    h3 = dem(lambda x: hi(x["H15.3_mat"]) < 0)
    h4 = dem(lambda x: hi(x["H15.4_mat"]) < 0)
    return {"so_bo_can": can, "so_bo": n,
            "H15.1": {"bo_dat": h1, "dat": len(h1) >= can}, "H15.2": {"bo_dat": h2, "dat": len(h2) >= can},
            "H15.3": {"bo_dat": h3, "dat": len(h3) >= can}, "H15.4": {"bo_dat": h4, "dat": len(h4) >= can}}


# -------------------------------------------------------------------------------------------- biểu đồ
MAU = {"dm": "#2a78d6", "dm_ktn": "#6250d6", "x10": "#eb6834", "x10_ktn": "#e34948", "dp": "#888888"}


def ve(tong, thay):
    plt = t11._plt()
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    for g, (m, t) in tong["_luoi"].items():
        ax[0].plot([t], [m], "s", color=MAU["dp"], ms=6, alpha=0.6)
    ax[0].plot([], [], "s", color=MAU["dp"], label="Dijkstra phạt (16 bộ β, γ)")
    for (ten, k), dau in zip(BIEN, ("o", "^", "o", "^")):
        ax[0].plot([tong[k]["tre"]], [tong[k]["mat_dv"]], dau, color=MAU[k], ms=9, mec="white", mew=1.5, label=ten)
    ax[0].set_xlabel("trễ tuyến chính trung bình (ms)")
    ax[0].set_ylabel("mất dịch vụ (%), 11 bộ gộp")
    ax[0].set_title("TN15 — DM theo J vật lý x10 so với 7/3", loc="left", fontsize=10)
    ax[0].legend(fontsize=7)
    ten = [t for t, _ in BIEN] + ["Dijkstra phạt"]
    khoa = [k for _, k in BIEN] + ["dp"]
    x = np.arange(len(khoa))
    ax[1].bar(x - 0.2, [tong[k]["tre_trung_vi"] for k in khoa], 0.38, color="#bbbbbb", label="trễ trung vị (ms)")
    ax[1].bar(x + 0.2, [tong[k]["tre_duoi"] for k in khoa], 0.38, color=[MAU[k] for k in khoa],
              label=f"% tình huống trễ > {NGUONG_DUOI:.0f} ms")
    ax[1].set_xticks(x, ten, rotation=20, ha="right", fontsize=7)
    ax[1].set_title("trễ ở giữa (trung vị) và phần đuôi nghẽn", loc="left", fontsize=10)
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(RA / "hinh_15.png", bbox_inches="tight")
    plt.close(fig)


# -------------------------------------------------------------------------------------------- báo cáo
def viet_md(ra):
    ci = t11._ci
    bo, tong, gt, th = ra["bo"], ra["tong"], ra["gia_thuyet"], ra["thay"]
    L = ["# Thí nghiệm 15 — kết quả (sinh tự động bởi `scripts/tn15.py`)", "",
         f"Tiêu chí: TIEU_CHI_15.md. Mất DV %, trễ ms; hiệu bằng điểm % / ms, CI 95% bootstrap theo cửa sổ; 4–5 luồng, trung "
         f"bình 3 seed. Đuôi = % tình huống trễ > {NGUONG_DUOI:.0f} ms. Dijkstra phạt: {ra['dp_chon']} (chọn cho x10 trên val).", ""]
    if th:
        L += ["## 0. Thầy trên val (xác nhận TN13, H15.0 — chẩn đoán)", "",
              f"{th['n']} tình huống. Thầy 7/3: mất {th['thay_73']['mat_dv']:.2f}%, trễ {th['thay_73']['tre']:.1f} ms (trung vị "
              f"{th['thay_73']['tre_trung_vi']:.1f}, đuôi {th['thay_73']['tre_duoi']:.1f}%). Thầy x10: mất {th['thay_x10']['mat_dv']:.2f}%, "
              f"trễ {th['thay_x10']['tre']:.1f} ms (trung vị {th['thay_x10']['tre_trung_vi']:.1f}, đuôi {th['thay_x10']['tre_duoi']:.1f}%).",
              f"Hiệu x10 − 7/3: mất {ci(th['hieu_mat'])}, trễ {ci(th['hieu_tre'], 1)}, đuôi {ci(th['hieu_duoi'], 1)} điểm %.", ""]
    L += ["## 1. Trung bình theo bộ — mất DV % / trễ tb ms / trễ trung vị ms / đuôi %", "",
          "| Phương pháp | " + " | ".join(TEN_BO[b] for b in bo) + " |", "|---|" + "---|" * len(bo)]
    for ten in next(iter(bo.values()))["trung_binh"]:
        L.append(f"| {ten} | " + " | ".join(
            f"{x['trung_binh'][ten]['mat_dv']:.2f} / {x['trung_binh'][ten]['tre']:.1f} / {x['trung_binh'][ten]['tre_trung_vi']:.1f} / "
            f"{x['trung_binh'][ten]['tre_duoi']:.1f}" for x in bo.values()) + " |")
    L += ["", "Gộp 11 bộ: " + "; ".join(f"{ten}: mất {tong[k]['mat_dv']:.2f}%, trễ {tong[k]['tre']:.1f} ms (trung vị "
                                         f"{tong[k]['tre_trung_vi']:.1f}, đuôi {tong[k]['tre_duoi']:.1f}%)"
                                         for ten, k in BIEN + [("Dijkstra phạt", "dp")]) + ". Hình: `hinh_15.png`.", "",
          "## 2. Giả thuyết theo bộ (CI 95%)", "",
          "| Bộ | H15.1 mất: x10 − 7/3 | H15.1 trễ tb | H15.2 đuôi (điểm %) | H15.3 mất: x10 − Dijkstra | trễ: x10 − Dijkstra | "
          "đuôi: x10 − Dijkstra | H15.4 mất: DM11 không tái nhiễu − đầy đủ | x10 không tái nhiễu − 7/3 đầy đủ: mất / trễ |",
          "|---|" + "---|" * 8]
    for b, x in bo.items():
        L.append(f"| {TEN_BO[b]} | {ci(x['H15.1_mat'])} | {ci(x['H15.1_tre'], 1)} | {ci(x['H15.2_duoi'], 1)} | "
                 f"{ci(x['H15.3_mat'])} | {ci(x['x10_tre_vs_dp'], 1)} | {ci(x['x10_duoi_vs_dp'], 1)} | {ci(x['H15.4_mat'])} | "
                 f"{ci(x['x10ktn_vs_dm_mat'])} / {ci(x['x10ktn_vs_dm_tre'], 1)} |")
    L += ["", f"Đạt = đúng chiều ở ≥ {gt['so_bo_can']}/{gt['so_bo']} bộ; không kém = cận trên CI < +{KHONG_KEM} điểm.", ""]
    for k in ("H15.1", "H15.2", "H15.3", "H15.4"):
        L.append(f"- **{k}: {'ĐẠT' if gt[k]['dat'] else 'KHÔNG ĐẠT'}** — bộ đạt: {gt[k]['bo_dat']}")
    L += ["", "## 3. Thành phần (a) / (b): x10 − 7/3 theo bộ", "", "| Bộ | (a) không còn tuyến sống | (b) quá tải |",
          "|---|---|---|"]
    for b, x in bo.items():
        L.append(f"| {TEN_BO[b]} | {ci(x['x10_a'])} | {ci(x['x10_b'])} |")
    L += ["", "## 4. Học (DM x10)", "", "| Mô hình | Epoch chạy | Epoch tốt nhất | val / train | J x10 val (thầy) | Cờ |",
          "|---|---:|---:|---:|---|---|"]
    for k, x in ra["hoc"].items():
        co = ("CHƯA TỚI " if x["chua_toi"] else "") + ("QUÁ KHỚP? " if x["ti_le_val_train"] > 1.25 else "")
        L.append(f"| {k} | {x['so_epoch']} | {x['epoch_tot']} | {x['ti_le_val_train']:.2f} | {x['J_val_tot']:.3f} "
                 f"({x['J_thay']:.3f}) | {co or 'ổn'} |")
    L += ["", f"Tái lập dữ liệu: {ra['tai_lap']}. Bản ghi bộ kiểm tra ghép được với bản sinh lại: {ra['ti_le_khop']:.1%}."]
    (RA / "KET_QUA_15.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def lenh_phan_tich(a):
    T = {b: rs for b in BO_45 if (rs := t11._gop(RA / "cham_tep", t11.thu_muc_bo(b, "4-5")))}
    if not T:
        raise SystemExit("chưa có kết quả chấm — chạy bước cham")
    g = dp_chon()
    bo, tong = phan_tich_bo(T, g)
    hoc = doc_hoc()
    tl = RA / "kiem_tai_lap.json"
    ra = {"dp_chon": g, "bo": bo, "tong": tong, "thay": phan_tich_thay(), "gia_thuyet": gia_thuyet(bo),
          "hoc": {k: {kk: vv for kk, vv in x.items() if kk != "nk"} for k, x in hoc.items()},
          "tai_lap": json.loads(tl.read_text(encoding="utf-8"))["ti_le_khop"] if tl.exists() else None,
          "ti_le_khop": float(np.mean([r.get("khop", False) for rs in T.values() for r in rs]))}
    RA.mkdir(parents=True, exist_ok=True)
    (RA / "ket_qua_15.json").write_text(json.dumps(ra, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    ve(tong, ra["thay"])
    viet_md(ra)
    print("  " + "; ".join(f"{k}: {'ĐẠT' if v['dat'] else 'KHÔNG ĐẠT'}" for k, v in ra["gia_thuyet"].items()
                           if k.startswith("H")), flush=True)
    dong_goi()


def dong_goi():
    """Một tệp để gửi lại: kết quả + bản ghi chấm + đường học và trọng số DM x10 + log."""
    import zipfile
    tep = KQ / "TN15_ket_qua.zip"
    gom = [(f, Path(RA.name) / f.relative_to(RA)) for f in sorted(RA.rglob("*")) if f.is_file()]
    gom += [(f, Path(M13.name) / f.name) for f in sorted(M13.glob(f"dm13_{MT}_s*")) if f.suffix in (".json", ".pt", ".xong")]
    gom += [(f, Path("logs") / f.name) for f in sorted(LOG.glob("tn15*.log"))]
    with zipfile.ZipFile(tep, "w", zipfile.ZIP_DEFLATED) as z:
        for f, t in gom:
            z.write(f, str(t))
    print(f"  đã gói {len(gom)} tệp → {tep} ({tep.stat().st_size / 1e6:.1f} MB) — gửi tệp này", flush=True)


# ============================================================================================ chạy
BUOC = (("chuan-bi", lenh_chuan_bi, "sinh"), ("sinh", lenh_sinh, "sinh"), ("kiem", lenh_kiem, None),
        ("huan-luyen", lenh_huan_luyen, None), ("cham", lenh_cham, "cham"), ("phan-tich", lenh_phan_tich, None))


def lenh_toi_da(a):
    p = t11.tai_nguyen(a.ram, a.vram)
    print(f"máy: {p['cpu']} luồng CPU, ngân sách RAM {p['ram']} GB, GPU: {p['gpu']}", flush=True)
    print(f"→ sinh {p['sinh']} tiến trình; học {p['hoc']} mô hình cùng lúc × {p['luong']} luồng ({p['thiet_bi']}); "
          f"chấm {p['cham']} tiến trình", flush=True)
    a.song_song, a.luong_cpu, a.thiet_bi = min(p["hoc"], len(SEEDS)), p["luong"], p["thiet_bi"]
    for ten, f, tt in BUOC:
        if tt:
            a.tien_trinh = p[tt]
        print(f"=== {ten} {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
        f(a)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("viec", choices=[b for b, _, _ in BUOC] + ["toi-da"])
    ap.add_argument("--tien-trinh", type=int, default=4)
    ap.add_argument("--song-song", type=int, default=3)
    ap.add_argument("--luong-cpu", type=int, default=4)
    ap.add_argument("--thiet-bi", default="auto", choices=["cpu", "auto", "cuda"])
    ap.add_argument("--ram", type=float, default=20.0, help="ngân sách RAM (GB) cho toi-da")
    ap.add_argument("--vram", type=float, default=8.0, help="ngân sách VRAM (GB) cho toi-da")
    ap.add_argument("--tn11", default=None, help="thư mục repo đã chạy TN11 hoặc tệp TN11_cho_TN12_TN13.zip")
    a = ap.parse_args()
    t12.mo_log("tn15", sys.argv[1:])
    if a.tn11:
        t12.lay_tn11(a.tn11)
    if a.viec == "toi-da":
        lenh_toi_da(a)
    else:
        next(f for b, f, _ in BUOC if b == a.viec)(a)
    print("xong", flush=True)


if __name__ == "__main__":
    main()
