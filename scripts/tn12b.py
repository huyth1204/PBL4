"""
tn12b.py — Thí nghiệm 12b (results/bao_da_tuyen/TIEU_CHI_12B.md): đối chứng "sửa dần KHÔNG khuếch tán". TN12 cho thấy độ chính
xác đến từ cơ chế sửa dần có phối hợp + nhiều chuỗi; TN12b hỏi: cần đúng KHUẾCH TÁN (học trên nhãn bị làm nhiễu theo lịch ᾱ)
hay chỉ cần một mạng tự sửa nhiều vòng?

    SD12 = cùng mạng, cùng dữ liệu, cùng đặc trưng phối hợp với DM11, nhưng học tự sửa lời giải của CHÍNH nó (không nhiễu)
    so sánh trên 11 bộ kiểm tra 4–5 luồng, 3 seed: SD12 5 × 32 / SD12 tất định / DM11 giải mã không tái nhiễu / DM11 / SM11

    python -m scripts.tn12b toi-da              # chạy hết (cần mô hình + dữ liệu TN11 như TN12; --tn11 <thư mục | zip>)

Từng bước (chạy lại an toàn — phần đã xong tự bỏ qua):
    huan-luyen   sd12_s0..s2 → results/bao_da_tuyen/mo_hinh12b/
    cham         11 bộ kiểm tra → tn12b/cham_tep/
    phan-tich    → tn12b/KET_QUA_12B.md, ket_qua_12b.json, hinh_12b_*.png; gói results/bao_da_tuyen/TN12B_ket_qua.zip
Log: logs/tn12b.log; mỗi mô hình logs/tn12b_<tên>.log.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np

GOC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GOC))
from scripts import tn11 as t11  # noqa: E402
from scripts import tn12 as t12  # noqa: E402

KQ, DL, LOG, M11 = t11.KQ, t11.DL, t11.LOG, t11.M11
RA, M12B = KQ / "tn12b", KQ / "mo_hinh12b"
BO_45, TEN_BO, DO, SEEDS = t11.BO_45, t11.TEN_BO, t11.DO, t11.SEEDS
CFG = t12.CFG                                            # 120 epoch, H 128, dừng sớm 20, S 32 — như DM11
TN_SO, SO_BUOC = 19, 5
NGUONG_TUONG_DUONG = 0.3                                 # điểm % — như H12.5
J_MAT, J_A, J_B, J_TRE = 0, 2, 3, 4
BAC = [("Dijkstra phạt", "dphat", False), ("SM11 · chọn trong 32 mẫu", "sm32", True),
       ("SD12 · tất định (5 bước, 1 chuỗi)", "sd_td", True), ("SD12 · sửa dần 5 × 32", "sd", True),
       ("DM11 · giải mã không tái nhiễu", "dm_ktn", True), ("DM11 · đầy đủ 5 × 32", "dm", True)]


# ============================================================================================ huấn luyện
def lenh_huan_luyen(a):
    thieu = [f"{h}_s{s}" for h in ("dm11", "sm11") for s in SEEDS if not (M11 / f"{h}_s{s}.pt").exists()]
    if thieu:
        raise SystemExit(f"thiếu mô hình TN11 ở {M11}: {thieu} (máy khác máy TN11: thêm --tn11 như TN12)")
    viec = [(f"sd12_s{s}", t12.lenh_hoc("sua_dan", TN_SO, f"sd12_s{s}", s, a)) for s in SEEDS]
    t12.hang_doi(viec, a.song_song, M12B, "tn12b")


# ============================================================================================ chấm
def _cham_tep(job):
    import torch

    from src.nhieu_luong import mo_hinh_dt as md
    from src.nhieu_luong.cham_lai_dt import giong_ban_ghi
    from src.nhieu_luong.cham_mo_hinh_dt import _tg
    (tep,) = job
    torch.set_num_threads(1)
    import pickle
    G = lambda sd: torch.Generator().manual_seed(sd)
    ra = []
    for r in pickle.load(open(tep, "rb")):
        gon = r["gon"]
        tgw, sd, x = _tg(gon, r["cham"]), giong_ban_ghi(r), t11._co_ban(r)
        luu = lambda k, a, ms: x.update({k: t11._do(gon, tgw, a), f"ms_{k}": ms})
        for s in SEEDS:
            dm = t11._net(str(M11), f"dm11_s{s}")
            sd12 = t11._net(str(M12B), f"sd12_s{s}")
            sm = t11._net(str(M11), f"sm11_s{s}")
            luu(f"dm_s{s}", *t12._gio(lambda: md.chon_dm(dm, gon, CFG["S"], SO_BUOC, G(sd))))
            luu(f"dm_ktn_s{s}", *t12._gio(lambda: md.chon_sua_dan(dm, gon, CFG["S"], SO_BUOC, G(sd))))
            luu(f"sd_s{s}", *t12._gio(lambda: md.chon_sua_dan(sd12, gon, CFG["S"], SO_BUOC, G(sd))))
            luu(f"sd_td_s{s}", *t12._gio(lambda: md.chon_sua_dan(sd12, gon, 1, SO_BUOC, G(sd), lay_mau=False)))
            luu(f"sm32_s{s}", *t12._gio(lambda: md.chon_softmax_mau(sm, gon, CFG["S"], G(sd))))
        ra.append(x)
    return tep, ra


def lenh_cham(a):
    thieu = [f"sd12_s{s}" for s in SEEDS if not (M12B / f"sd12_s{s}.xong").exists()]
    if thieu:
        raise SystemExit(f"chưa xong mô hình TN12b: {thieu}")
    viec = [(f"{t11.thu_muc_bo(b, '4-5')}__{Path(t).stem}.json", (t,))
            for b in BO_45 for t in sorted(glob.glob(str(DL / t11.thu_muc_bo(b, "4-5") / "kb_*.pkl")))]
    if not viec:
        raise SystemExit(f"không thấy bộ kiểm tra TN11 ở {DL}")
    print(f"  chấm bộ kiểm tra: {len(viec)} tệp", flush=True)
    t11._chay_tep(_cham_tep, viec, a.tien_trinh, RA / "cham_tep")


# ============================================================================================ phân tích
def _v(rs, k, co_seed, j=J_MAT):
    return t11._ho(rs, k, SEEDS, j) if co_seed else t11._cot(rs, k, j)


def _ci_dat(ci, chieu):
    """chieu = -1: CI dưới 0; +1: CI trên 0; 0: CI nằm trong ±NGUONG_TUONG_DUONG."""
    lo, hi = ci[1]
    return hi < 0 if chieu < 0 else lo > 0 if chieu > 0 else (-NGUONG_TUONG_DUONG < lo and hi < NGUONG_TUONG_DUONG)


def phan_tich_bo(T):
    bo, gop = {}, {}
    for b, rs in T.items():
        kb = np.array([r["kb"] for r in rs])
        v = {k: {j: _v(rs, k, cs, j) for j in (J_MAT, J_A, J_B, J_TRE)} for _, k, cs in BAC}
        d = lambda x, y, j=J_MAT: t11._bs(v[x][j] - v[y][j], kb)
        bo[b] = {"n": len(rs),
                 "trung_binh": {ten: {"mat_dv": float(v[k][J_MAT].mean()), "a": float(v[k][J_A].mean()),
                                      "b": float(v[k][J_B].mean()), "tre": float(v[k][J_TRE].mean()),
                                      "tre_trung_vi": float(np.median(v[k][J_TRE])),
                                      "tre_tren_100": float(np.mean(v[k][J_TRE] > 100) * 100)} for ten, k, cs in BAC},
                 "H12b.1": d("dm", "sd"), "H12b.2": d("sd", "sm32"), "sd_vs_dphat": d("sd", "dphat"),
                 "tai_nhieu": d("dm", "dm_ktn"), "lay_mau_chuoi": d("sd", "sd_td"), "dm_vs_dphat": d("dm", "dphat")}
        for _, k, cs in BAC:
            for j in (J_MAT, J_B):
                gop.setdefault((k, j), []).append(v[k][j])
        gop.setdefault("cum", []).append(BO_45.index(b) * 1000 + kb)
    cum = np.concatenate(gop.pop("cum"))
    tong = {ten: {"mat_dv": t11._bs(np.concatenate(gop[(k, J_MAT)]), cum),
                  "qua_tai": float(np.concatenate(gop[(k, J_B)]).mean())} for ten, k, _ in BAC}
    l_dm, l_sd = np.concatenate(gop[("dm", J_MAT)]), np.concatenate(gop[("sd", J_MAT)])
    b_dm, b_sd = np.concatenate(gop[("dm", J_B)]), np.concatenate(gop[("sd", J_B)])
    tong["_hieu_gop_dm_tru_sd"] = t11._bs(l_dm - l_sd, cum)
    tong["_ti_le_qua_tai"] = float((b_dm - b_sd).mean() / (l_dm - l_sd).mean()) if abs((l_dm - l_sd).mean()) > 1e-9 \
        else float("nan")
    return bo, tong


def doc_hoc():
    ra = {}
    for f in sorted(M12B.glob("sd12_s*_nhat_ky.json")):
        j = json.loads(f.read_text(encoding="utf-8"))
        nk = j["nhat_ky"]
        b = min(nk, key=lambda r: r["J_val"])
        ra[f.name.replace("_nhat_ky.json", "")] = {
            "so_epoch": len(nk), "epoch_tot": b["epoch"], "dung_som_o": j.get("dung_som_o"),
            "ti_le_val_train": b["loss_val"] / b["loss_train"], "J_val_tot": b["J_val"], "J_thay": b["J_thay_val"],
            "chua_toi": len(nk) >= CFG["epochs"] and b["epoch"] >= 0.9 * CFG["epochs"],
            "nk": [[r["epoch"], r["loss_train"], r["loss_val"], r["J_val"]] for r in nk]}
    return ra


def kiem_lap_lai(T):
    """DM11 5 × 32 và SM11 32 mẫu của TN12b phải trùng TN12 (cùng mô hình, cùng hạt giống) nếu có tn12/cham_tep."""
    c12 = KQ / "tn12" / "cham_tep"
    if not c12.exists():
        return None
    lech, n = 0.0, 0
    for b, rs in T.items():
        r12 = t11._gop(c12, t11.thu_muc_bo(b, "4-5"))
        if len(r12) != len(rs):
            return {"loi": f"{b}: số bản ghi khác ({len(rs)} vs {len(r12)})"}
        for x, y in zip(rs, r12):
            for s in SEEDS:
                lech = max(lech, abs(x[f"dm_s{s}"][0] - y[f"dm_5x32_s{s}"][0]), abs(x[f"sm32_s{s}"][0] - y[f"sm_32_s{s}"][0]))
                n += 2
    return {"so_so_sanh": n, "lech_toi_da": lech}


def gia_thuyet(bo, hoc):
    dem = lambda k, chieu: [b for b, x in bo.items() if _ci_dat(x[k], chieu)]
    n = len(bo)
    can = max(1, int(np.ceil(8 / 11 * n)))
    td, dm_tot, sd_tot = dem("H12b.1", 0), dem("H12b.1", -1), dem("H12b.1", +1)
    kl1 = ("tương đương — sửa dần có phối hợp là đủ, khuếch tán chỉ là một cách huấn luyện nó" if len(td) >= can else
           "khuếch tán tạo thêm độ chính xác ngoài sửa dần" if len(dm_tot) >= can else
           "sửa dần không nhiễu tốt hơn khuếch tán" if len(sd_tot) >= can else "không kết luận")
    h2 = dem("H12b.2", -1)
    qua_khop = [k for k, x in hoc.items() if x["ti_le_val_train"] > 1.25]
    return {"so_bo_can": can, "so_bo": n,
            "H12b.1": {"tuong_duong": td, "dm_tot_hon": dm_tot, "sd_tot_hon": sd_tot, "ket_luan": kl1,
                       "dat": kl1 != "không kết luận"},
            "H12b.2": {"bo_dat": h2, "dat": len(h2) >= can},
            "mo_ta": {"sd_hon_dphat": dem("sd_vs_dphat", -1), "tai_nhieu_giup": dem("tai_nhieu", -1),
                      "lay_mau_chuoi_giup": dem("lay_mau_chuoi", -1), "sd_co_co_qua_khop": qua_khop}}


# -------------------------------------------------------------------------------------------- biểu đồ
MAU = ["#888888", "#1baf7a", "#e87ba4", "#eda100", "#6250d6", "#2a78d6"]


def ve_thang(tong):
    plt = t11._plt()
    ten = [t for t, _, _ in BAC]
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    for i, t in enumerate(ten):
        m, (lo, hi) = tong[t]["mat_dv"]
        ax.plot([lo, hi], [i, i], color=MAU[i], lw=2.2)
        ax.plot([m], [i], "o", color=MAU[i], ms=7, mec="white", mew=1.5)
        ax.text(hi + 0.08, i, f"{m:.2f}", va="center", fontsize=8, color="#333333")
    ax.set_yticks(range(len(ten)), ten)
    ax.invert_yaxis()
    ax.set_xlabel("mất dịch vụ (%), 11 bộ gộp, CI 95% theo cửa sổ — sang trái là tốt hơn")
    ax.set_title("TN12b — sửa dần không khuếch tán so với khuếch tán", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(RA / "hinh_12b_thang.png", bbox_inches="tight")
    plt.close(fig)


def ve_hoc(hoc):
    if not hoc:
        return
    plt = t11._plt()
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
    for j, (k, x) in enumerate(sorted(hoc.items())):
        e, mau = np.array(x["nk"]), t11.MAU_PHAN[j % 4]
        ax[0].plot(e[:, 0], e[:, 1], color=mau, lw=1.5, label=f"{k} · train")
        ax[0].plot(e[:, 0], e[:, 2], color=mau, lw=1.5, ls="--", label=f"{k} · val")
        ax[1].plot(e[:, 0], e[:, 3], color=mau, lw=1.5, label=k)
        ax[1].plot([x["epoch_tot"]], [x["J_val_tot"]], "o", color=mau, ms=6, mec="white", mew=1.5)
    ax[0].set_title("SD12 · loss (liền = train, gạch = val)", loc="left", fontsize=9)
    ax[1].set_title("SD12 · J val (chấm = checkpoint giữ lại)", loc="left", fontsize=9)
    ax[0].legend(fontsize=6, ncol=2)
    ax[1].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(RA / "hinh_12b_hoc.png", bbox_inches="tight")
    plt.close(fig)


# -------------------------------------------------------------------------------------------- báo cáo
def viet_md(ra):
    ci = t11._ci
    bo, tong, gt = ra["bo"], ra["tong"], ra["gia_thuyet"]
    L = ["# Thí nghiệm 12b — kết quả (sinh tự động bởi `scripts/tn12b.py`)", "",
         "Tiêu chí: TIEU_CHI_12B.md. Mất DV %, hiệu bằng điểm %, CI 95% bootstrap theo cửa sổ; 4–5 luồng, trung bình 3 seed. "
         "SD12 = cùng mạng / dữ liệu / đặc trưng phối hợp với DM11, học tự sửa lời giải của chính nó, không có quá trình nhiễu.",
         "", "## 1. Mất dịch vụ (%) theo bộ", "",
         "| Phương pháp | " + " | ".join(TEN_BO[b] for b in bo) + " | Gộp 11 bộ | Quá tải (b) gộp |",
         "|---|" + "---:|" * (len(bo) + 2)]
    for ten, _, _ in BAC:
        m, (lo, hi) = tong[ten]["mat_dv"]
        L.append(f"| {ten} | " + " | ".join(f"{x['trung_binh'][ten]['mat_dv']:.2f}" for x in bo.values())
                 + f" | {m:.2f} [{lo:.2f}; {hi:.2f}] | {tong[ten]['qua_tai']:.2f} |")
    L += ["", f"Gộp 11 bộ, DM11 − SD12: {ci(tong['_hieu_gop_dm_tru_sd'])}; phần quá tải trong phần chênh: "
          f"{tong['_ti_le_qua_tai']:.0%}. Hình: `hinh_12b_thang.png`.", "", "## 2. Hiệu theo bộ (điểm %, CI 95%)", "",
          "| Bộ | H12b.1: DM11 − SD12 | H12b.2: SD12 − SM11 32 mẫu | SD12 − Dijkstra phạt | DM11 − DM11 không tái nhiễu | "
          "SD12 5 × 32 − SD12 tất định |", "|---|---|---|---|---|---|"]
    for b, x in bo.items():
        L.append(f"| {TEN_BO[b]} | {ci(x['H12b.1'])} | {ci(x['H12b.2'])} | {ci(x['sd_vs_dphat'])} | {ci(x['tai_nhieu'])} | "
                 f"{ci(x['lay_mau_chuoi'])} |")
    L += ["", f"Đạt = đúng chiều ở ≥ {gt['so_bo_can']}/{gt['so_bo']} bộ; tương đương = CI trong ±{NGUONG_TUONG_DUONG} điểm.", ""]
    for k in ("H12b.1", "H12b.2"):
        v = gt[k]
        L.append(f"- **{k}: {'ĐẠT' if v['dat'] else 'KHÔNG ĐẠT'}** — " + ", ".join(f"{kk} = {vv}" for kk, vv in v.items()
                                                                                  if kk != "dat"))
    L.append(f"- Mô tả: {gt['mo_ta']}")
    L += ["", "## 3. Thành phần và trễ (gộp trung bình các bộ)", "",
          "| Phương pháp | (a) không còn tuyến sống | (b) quá tải | trễ tb ms | trễ trung vị ms | % tình huống trễ > 100 ms |",
          "|---|---:|---:|---:|---:|---:|"]
    for ten, _, _ in BAC:
        tb = lambda k: float(np.mean([x["trung_binh"][ten][k] for x in bo.values()]))
        L.append(f"| {ten} | {tb('a'):.2f} | {tb('b'):.2f} | {tb('tre'):.1f} | {tb('tre_trung_vi'):.2f} | {tb('tre_tren_100'):.1f} |")
    L += ["", "## 4. Học (SD12)", "", "| Mô hình | Epoch chạy | Epoch tốt nhất | val / train | J val (thầy) | Cờ |",
          "|---|---:|---:|---:|---|---|"]
    for k, x in ra["hoc"].items():
        co = ("CHƯA TỚI " if x["chua_toi"] else "") + ("QUÁ KHỚP? " if x["ti_le_val_train"] > 1.25 else "")
        L.append(f"| {k} | {x['so_epoch']} | {x['epoch_tot']} | {x['ti_le_val_train']:.2f} | {x['J_val_tot']:.3f} "
                 f"({x['J_thay']:.3f}) | {co or 'ổn'} |")
    L += ["", f"Hình: `hinh_12b_hoc.png`. Kiểm lặp lại DM11 / SM11 so với TN12: {ra['lap_lai']}."]
    (RA / "KET_QUA_12B.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def lenh_phan_tich(a):
    T = {b: rs for b in BO_45 if (rs := t11._gop(RA / "cham_tep", t11.thu_muc_bo(b, "4-5")))}
    if not T:
        raise SystemExit("chưa có kết quả chấm — chạy bước cham")
    bo, tong = phan_tich_bo(T)
    hoc = doc_hoc()
    ra = {"cau_hinh": CFG, "bo": bo, "tong": tong, "hoc": {k: {kk: vv for kk, vv in x.items() if kk != "nk"}
                                                           for k, x in hoc.items()}, "lap_lai": kiem_lap_lai(T)}
    ra["gia_thuyet"] = gia_thuyet(bo, hoc)
    RA.mkdir(parents=True, exist_ok=True)
    (RA / "ket_qua_12b.json").write_text(json.dumps(ra, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    ve_thang(tong)
    ve_hoc(hoc)
    viet_md(ra)
    gt = ra["gia_thuyet"]
    print(f"  H12b.1: {gt['H12b.1']['ket_luan']}; H12b.2: {'ĐẠT' if gt['H12b.2']['dat'] else 'KHÔNG ĐẠT'}", flush=True)
    t12.dong_goi("TN12B", RA, M12B, "tn12b")


# ============================================================================================ chạy
def lenh_toi_da(a):
    p = t11.tai_nguyen(a.ram, a.vram)
    print(f"máy: {p['cpu']} luồng CPU, ngân sách RAM {p['ram']} GB, GPU: {p['gpu']}", flush=True)
    print(f"→ học {p['hoc']} mô hình cùng lúc × {p['luong']} luồng ({p['thiet_bi']}); chấm {p['cham']} tiến trình", flush=True)
    a.song_song, a.luong_cpu, a.thiet_bi, a.tien_trinh = min(p["hoc"], len(SEEDS)), p["luong"], p["thiet_bi"], p["cham"]
    for ten, f in (("huan-luyen", lenh_huan_luyen), ("cham", lenh_cham), ("phan-tich", lenh_phan_tich)):
        print(f"=== {ten} {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
        f(a)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("viec", choices=["huan-luyen", "cham", "phan-tich", "toi-da"])
    ap.add_argument("--tien-trinh", type=int, default=4)
    ap.add_argument("--song-song", type=int, default=3)
    ap.add_argument("--luong-cpu", type=int, default=4)
    ap.add_argument("--thiet-bi", default="auto", choices=["cpu", "auto", "cuda"])
    ap.add_argument("--ram", type=float, default=20.0, help="ngân sách RAM (GB) cho toi-da")
    ap.add_argument("--vram", type=float, default=8.0, help="ngân sách VRAM (GB) cho toi-da")
    ap.add_argument("--tn11", default=None, help="thư mục repo đã chạy TN11 hoặc tệp TN11_cho_TN12_TN13.zip")
    a = ap.parse_args()
    t12.mo_log("tn12b", sys.argv[1:])
    if a.tn11:
        t12.lay_tn11(a.tn11)
    {"huan-luyen": lenh_huan_luyen, "cham": lenh_cham, "phan-tich": lenh_phan_tich, "toi-da": lenh_toi_da}[a.viec](a)
    print("xong", flush=True)


if __name__ == "__main__":
    main()
