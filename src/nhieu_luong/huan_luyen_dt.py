"""
huan_luyen_dt.py — Huấn luyện DM / softmax chọn tập tuyến (thí nghiệm 4 và 6, results/bao_da_tuyen/TIEU_CHI_4.md, _6.md).

    python -m src.nhieu_luong.huan_luyen_dt dm                           # thí nghiệm 4
    python -m src.nhieu_luong.huan_luyen_dt softmax
    python -m src.nhieu_luong.huan_luyen_dt chon-buoc                    # số bước DM ∈ {3, 5, 10} trên val, S = 8
    python -m src.nhieu_luong.huan_luyen_dt dm --thi-nghiem 6 --ten dm6 --them      # thí nghiệm 6, có đặc trưng mới
    python -m src.nhieu_luong.huan_luyen_dt dm --thi-nghiem 6 --ten dm6a            # thí nghiệm 6, không đặc trưng mới
    python -m src.nhieu_luong.huan_luyen_dt chon-buoc --thi-nghiem 6 --ten dm6

Nhãn = lời giải của thầy (TKCB trên J của 32 thế giới NHÃN), chỉ ở K = 4–5. Chọn checkpoint theo J trung bình trên 300
tình huống val cố định (thí nghiệm 6: 150 val bão + 150 val6), chấm bằng 32 thế giới CHẤM (seed độc lập với bộ nhãn).
Không dùng dữ liệu test.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

from src.nhieu_luong import da_tuyen as dt
from src.nhieu_luong import mo_hinh_dt as md

GOC = Path(__file__).resolve().parents[2]
DL = GOC / "data" / "bao_da_tuyen"
BUOC, S = (3, 5, 10), 8
THI_NGHIEM = {4: {"train": ("train_k4_5",), "val": ("val_k4_5",), "thu_muc": md.KQ},
              6: {"train": ("train_k4_5", "train6_k4_5"), "val": ("val_k4_5", "val6_k4_5"),
                  "thu_muc": GOC / "results" / "bao_da_tuyen" / "mo_hinh6"},
              7: {"train": ("train_k4_5", "train6_k4_5"), "val": ("val_k4_5", "val6_k4_5"),
                  "thu_muc": GOC / "results" / "bao_da_tuyen" / "mo_hinh7"},
              # thí nghiệm 8: DM8b học trong thế giới B (cùng cửa sổ / seed với dữ liệu của DM6 = thế giới A)
              8: {"train": ("train_b_k4_5", "train6_b_k4_5"), "val": ("val_b_k4_5", "val6_b_k4_5"),
                  "thu_muc": GOC / "results" / "bao_da_tuyen" / "mo_hinh8"},
              # thí nghiệm 9: đúng dữ liệu của DM6, lưu mọi epoch (đường học, chọn checkpoint) + softmax cùng dữ liệu
              9: {"train": ("train_k4_5", "train6_k4_5"), "val": ("val_k4_5", "val6_k4_5"),
                  "thu_muc": GOC / "results" / "bao_da_tuyen" / "mo_hinh9"}}
# thí nghiệm 10 (TIEU_CHI_10.md): 10 = dữ liệu A, thầy 32 thế giới; 11 = A, thầy 64; 12 = A + B, thầy 32; 13 = A + B,
# thầy 64. Tập học có thế giới chấm (train10…) → kiểm thử chéo 4 phần theo cửa sổ (--phan). Checkpoint chọn trên val
# của (các) thế giới học, KHÔNG dùng phần giữ lại.
_M10 = GOC / "results" / "bao_da_tuyen" / "mo_hinh10"
_VA, _VB = ("val_k4_5", "val6_k4_5"), ("val_b_k4_5", "val6_b_k4_5")
_HA, _HB = ("train10_k4_5", "train6_10_k4_5"), ("train_b10_k4_5", "train6_b10_k4_5")
_t64 = lambda ds: tuple(d.replace("_k4_5", "_t64_k4_5") for d in ds)
THI_NGHIEM.update({10: {"train": _HA, "val": _VA, "thu_muc": _M10}, 11: {"train": _t64(_HA), "val": _VA, "thu_muc": _M10},
                   12: {"train": _HA + _HB, "val": _VA + _VB, "thu_muc": _M10},
                   13: {"train": _t64(_HA + _HB), "val": _VA + _VB, "thu_muc": _M10}})
# thí nghiệm 11a (TIEU_CHI_11A.md): cùng dữ liệu / phần với 10, mô hình "nhìn trước" (--nhin-truoc) lưu riêng
THI_NGHIEM[14] = {"train": _HA, "val": _VA, "thu_muc": GOC / "results" / "bao_da_tuyen" / "mo_hinh11a"}
SO_PHAN, HAT_PHAN = 4, 10
_SO_CUA_SO_GOC = {"train": 59, "train6": 60}


def phan_cua(thu_muc, kb):
    """Phần (0..3) của cửa sổ kb trong tập học TN10. Chia theo CỬA SỔ GỐC (train / train6): cùng cửa sổ thời tiết thì cùng
    phần ở thế giới A, B và ở nhãn thầy 32 / 64 → không rò rỉ khi học A + B."""
    goc = "train6" if thu_muc.startswith(("train6", "train6_b")) else "train"
    n = _SO_CUA_SO_GOC[goc]
    hv = np.random.default_rng(HAT_PHAN + (goc == "train6")).permutation(n)
    return int(hv[int(kb)] % SO_PHAN)
GAP_DOI = ("train7s_k4_5", "train7x_k4_5")      # thí nghiệm 7b: thêm vào để có mốc 200%


def nap(tap, K="k4_5"):
    return nap_thu_muc(f"{tap}_{K}")


def nap_thu_muc(ten, ti_le=1.0):
    """Mọi bản ghi của thư mục; ti_le < 1 (thí nghiệm 7b): chỉ lấy ngẫu nhiên round(ti_le · số cửa sổ) cửa sổ (tệp),
    default_rng(7)."""
    teps = sorted(glob.glob(str(DL / ten / "kb_*.pkl")))
    if ti_le < 1:
        teps = [teps[i] for i in sorted(np.random.default_rng(7).choice(len(teps), round(ti_le * len(teps)), replace=False))]
    return [r for f in teps for r in pickle.load(open(f, "rb"))]


def the_gioi_cham(r):
    """Thế giới chấm đã lưu (sống, dung lượng còn trống nhỏ nhất) → TheGioi trên kho."""
    if "_tg" not in r:
        du = r["cham"]["du"].astype(float)
        r["_tg"] = dt.TheGioi(r["gon"], r["cham"]["song"], du[:, None], np.zeros_like(du)[:, None])
    return r["_tg"]


def val_co_dinh(va, n=300):
    return [va[i] for i in np.random.default_rng(99).choice(len(va), min(n, len(va)), replace=False)]


def val_tron(cac_val, n=300):
    """n tình huống val cố định, chia đều cho các thư mục val."""
    return [r for va in cac_val for r in val_co_dinh(va, n // len(cac_val))]


def danh_gia(net, kieu, ds, so_buoc=5, seed=0):
    """J trung bình (thế giới chấm) của lời giải mô hình, và của thầy trên cùng thế giới."""
    net.eval()
    gen = torch.Generator().manual_seed(seed)
    J, Jt = [], []
    for r in ds:
        a = md.chon_dm(net, r["gon"], S, so_buoc, gen) if kieu == "dm" else md.chon_softmax(net, r["gon"])
        J.append(the_gioi_cham(r).cham(r["gon"].tu_lua_chon(a))["J"])
        Jt.append(r["ens"]["thay"]["J"])
    net.train()
    return float(np.mean(J)), float(np.mean(Jt))


def chia_lo(ds, B, rng):
    idx = rng.permutation(len(ds))
    lo = []
    for a in range(0, len(idx), 20 * B):
        g = sorted(idx[a:a + 20 * B], key=lambda i: (ds[i][0].K, len(ds[i][0].U)))
        lo += [g[j:j + B] for j in range(0, len(g), B)]
    rng.shuffle(lo)
    return lo


def huan_luyen(kieu, epochs=20, B=32, H=64, lr=1e-3, seed=0, tn=4, ten=None, them=False, ti_le=1.0, gap_doi=False,
               luong=1, moi_epoch=False, bo_phan=None, thiet_bi="cpu"):
    torch.set_num_threads(luong)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    cfg = THI_NGHIEM[tn]
    ra, ten = cfg["thu_muc"], ten or kieu
    ra.mkdir(parents=True, exist_ok=True)
    cac_train = cfg["train"] + (GAP_DOI if gap_doi else ())
    tr = [(r["gon"], r["nhan"]) for d in cac_train for r in nap_thu_muc(d, ti_le)
          if bo_phan is None or phan_cua(d, r["kb"]) != bo_phan]           # TN10: bỏ phần giữ lại của kiểm thử chéo
    cac_val = [nap_thu_muc(d) for d in cfg["val"]]
    va_nho = val_tron(cac_val)
    vl = [(r["gon"], r["nhan"]) for va in cac_val for r in va]
    print(f"{ten} ({kieu}, đặc trưng mới: {them}, tỉ lệ {ti_le}, gấp đôi {gap_doi}): train {len(tr)}, val {len(vl)}",
          flush=True)
    net = md.MangDT(H=H, them=them)
    dev = chon_thiet_bi(thiet_bi, net, tr, B, them, md.mat_mat_dm if kieu == "dm" else md.mat_mat_softmax)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    so_lo = math.ceil(len(tr) / B) * epochs
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(s / so_lo, 1))))
    mat = md.mat_mat_dm if kieu == "dm" else md.mat_mat_softmax
    nk, tot, t0 = [], np.inf, time.time()
    for ep in range(epochs):
        L = []
        for b in chia_lo(tr, B, rng):
            d = [tr[i] for i in b]
            lo = md.LoDT([x for x, _ in d], them=them, dev=dev)
            l = mat(net, lo, md._den(md.trang_thai_nhan(d, lo.K), dev))
            opt.zero_grad()
            l.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sch.step()
            L.append(l.item())
        with torch.no_grad():
            g = torch.Generator().manual_seed(1)
            Lv = []
            for j in range(0, len(vl), 64):
                d = vl[j:j + 64]
                lo = md.LoDT([x for x, _ in d], them=them, dev=dev)
                Lv.append(mat(net.eval(), lo, md._den(md.trang_thai_nhan(d, lo.K), dev), g).item())
            net.train()
        J, Jt = danh_gia(net, kieu, va_nho)
        nk.append({"epoch": ep + 1, "loss_train": float(np.mean(L)), "loss_val": float(np.mean(Lv)), "J_val": J,
                   "J_thay_val": Jt, "giay": time.time() - t0})
        print(f"  ep {ep + 1:2d}  loss {np.mean(L):.4f}  val {np.mean(Lv):.4f}  J val {J:.4f} (thầy {Jt:.4f})  "
              f"({time.time() - t0:.0f} s)", flush=True)
        ck = {"state": {k: v.cpu() for k, v in net.state_dict().items()}, "cau_hinh": net.cau_hinh, "epoch": ep + 1,
              "J_val": J}                          # luôn lưu trọng số CPU → nạp được trên máy không có GPU
        if moi_epoch:                                   # thí nghiệm 9: đường học trên các bậc độ lạ
            torch.save(ck, ra / f"{ten}_ep{ep + 1:02d}.pt")
        if J < tot:
            tot = J
            torch.save(ck, ra / f"{ten}.pt")
        json.dump({"nhat_ky": nk, "tot_nhat_J_val": tot}, open(ra / f"{ten}_nhat_ky.json", "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
    (ra / f"{ten}.xong").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))


def chon_thiet_bi(thiet_bi, net, tr, B, them, mat):
    """Thí nghiệm 10 (Colab): "auto" / "cuda" → thử một lô trên GPU; lỗi hoặc không có GPU → CPU. Trả thiết bị hoặc None
    (CPU, đường cũ không đổi). Không làm lệch bộ sinh ngẫu nhiên toàn cục."""
    if thiet_bi == "cpu" or not torch.cuda.is_available():
        if thiet_bi == "cuda":
            print("  không có GPU → dùng CPU", flush=True)
        return None
    dev, st = torch.device("cuda"), torch.get_rng_state()
    try:
        net.to(dev)
        d = tr[:B]
        lo = md.LoDT([x for x, _ in d], them=them, dev=dev)
        mat(net, lo, md._den(md.trang_thai_nhan(d, lo.K), dev)).backward()
        net.zero_grad(set_to_none=True)
        print(f"  huấn luyện trên GPU: {torch.cuda.get_device_name(0)}", flush=True)
        return dev
    except Exception as e:                                               # noqa: BLE001 — mọi lỗi GPU → CPU
        print(f"  GPU lỗi ({type(e).__name__}: {e}) → dùng CPU", flush=True)
        net.cpu()
        net.zero_grad(set_to_none=True)
        return None
    finally:
        torch.set_rng_state(st)


def chon_buoc(tn=4, ten="dm"):
    torch.set_num_threads(1)
    cfg = THI_NGHIEM[tn]
    va = val_tron([nap_thu_muc(d) for d in cfg["val"]])
    net = md.nap(ten, cfg["thu_muc"])
    bang = {}
    for n in BUOC:
        s = time.perf_counter()
        J, Jt = danh_gia(net, "dm", va, n)
        bang[n] = {"J_val": J, "J_thay_val": Jt, "giay": time.perf_counter() - s}
        print(n, bang[n], flush=True)
    tot = min(BUOC, key=lambda n: (round(bang[n]["J_val"], 9), n))
    tep = cfg["thu_muc"] / ("chon_buoc.json" if ten == "dm" else f"chon_buoc_{ten}.json")
    json.dump({"S": S, "so_buoc": tot, "bang": bang, "quy_tac": "J val nhỏ nhất (thế giới chấm); hoà → ít bước"},
              open(tep, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("chọn", tot)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("viec", choices=["dm", "softmax", "chon-buoc"])
    ap.add_argument("--thi-nghiem", type=int, default=4, choices=sorted(THI_NGHIEM))
    ap.add_argument("--ten", default=None)
    ap.add_argument("--them", action="store_true", help="thêm 3 đặc trưng giông quanh UAV (thí nghiệm 6)")
    ap.add_argument("--ti-le", type=float, default=1.0, help="thí nghiệm 7b: phần cửa sổ train dùng")
    ap.add_argument("--gap-doi", action="store_true", help="thí nghiệm 7b: thêm dữ liệu train7s + train7x")
    ap.add_argument("--seed", type=int, default=0, help="hạt giống huấn luyện (thí nghiệm 8: 0, 1, 2)")
    ap.add_argument("--luong-cpu", type=int, default=1, help="số luồng CPU cho PyTorch khi huấn luyện")
    ap.add_argument("--moi-epoch", action="store_true", help="thí nghiệm 9: lưu checkpoint mọi epoch (<tên>_epNN.pt)")
    ap.add_argument("--epochs", type=int, default=20, help="thí nghiệm 10: số epoch")
    ap.add_argument("--H", type=int, default=64, help="thí nghiệm 10: độ rộng mạng")
    ap.add_argument("--bo-phan", type=int, default=None, help="thí nghiệm 10: phần (0..3) giữ lại của kiểm thử chéo")
    ap.add_argument("--thiet-bi", default="cpu", choices=["cpu", "auto", "cuda"], help="thí nghiệm 10: auto = GPU nếu có")
    ap.add_argument("--nhin-truoc", action="store_true",
                    help="thí nghiệm 11a: thêm 3 đặc trưng tương lai thật (chẩn đoán; cần scripts.tn11a bo-sung trước)")
    a = ap.parse_args()
    if a.nhin_truoc:
        a.them = 2
    if a.viec == "chon-buoc":
        chon_buoc(a.thi_nghiem, a.ten or "dm")
    else:
        huan_luyen(a.viec, epochs=a.epochs, H=a.H, tn=a.thi_nghiem, ten=a.ten, them=a.them, ti_le=a.ti_le,
                   gap_doi=a.gap_doi, seed=a.seed, luong=a.luong_cpu, moi_epoch=a.moi_epoch, bo_phan=a.bo_phan,
                   thiet_bi=a.thiet_bi)
