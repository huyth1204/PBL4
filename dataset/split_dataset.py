"""
Chia oracle_dataset.npz thành train / val / test.

Chạy (từ thư mục gốc dự án):
    python split_dataset.py                          # đọc dataset/oracle_dataset.npz
    python split_dataset.py --compare                # in bảng so sánh các cách chia
    python split_dataset.py --mode time              # chia theo khối thời gian thường

Đầu ra, cạnh file vào:
    train.npz  val.npz  test.npz  split_info.json

VÌ SAO KHÔNG XÁO NGẪU NHIÊN TỪNG SNAPSHOT
    Hai snapshot cách nhau 20 s có ~97% cạnh giống nhau. Nếu snapshot này vào train
    và snapshot kề bên vào test thì test đã "thấy" gần hết đáp án.

VÌ SAO CHIA THEO THỜI GIAN CŨNG CHƯA ĐỦ
    Topo mạng lặp lại theo chu kỳ (đo được ~47 phút, xem estimate_period): snapshot
    tại thời điểm t gần như trùng snapshot tại t ± 47 phút. Một khối test nằm giữa
    dữ liệu vẫn có "bản sao" nằm trong train.

CHẾ ĐỘ MẶC ĐỊNH `phase`
    Gộp các snapshot theo PHA trong chu kỳ (pha = id mod P). Snapshot cùng pha, tức
    các bản sao của nhau, luôn vào CÙNG một tập. Test khi đó là các cấu hình topo mà
    mô hình chưa từng thấy lần nào trong train.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

TRAIN, VAL, TEST, DROP = 0, 1, 2, -1
NAMES = {TRAIN: "train", VAL: "val", TEST: "test"}


# ----------------------------------------------------------------------------
# Đọc dữ liệu và rút ra biểu diễn theo SNAPSHOT
# ----------------------------------------------------------------------------
def load_npz(path: Path) -> dict[str, np.ndarray]:
    d = np.load(path, allow_pickle=False)
    return {k: d[k] for k in d.files}


def snapshot_edges(data: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Trả về (ids, first_row, E):
      ids       : các snapshot_id khác nhau, đã sắp xếp
      first_row : chỉ số dòng đầu tiên của mỗi snapshot trong file
      E         : (S, N*N) bool, cạnh nào tồn tại (x > 0, bỏ đường chéo)
    Kiểm tra luôn: mọi mẫu cùng snapshot phải có cùng x_weights.
    """
    sid = data["snapshot_id"]
    n = len(data["node_order"])
    ids, first = np.unique(sid, return_index=True)

    row_of = np.full(int(ids.max()) + 1, -1, dtype=np.int64)
    row_of[ids] = first
    if not np.array_equal(data["x_weights"], data["x_weights"][row_of[sid]]):
        raise ValueError("Có mẫu cùng snapshot_id nhưng khác x_weights.")

    W = data["x_weights"][first].reshape(len(ids), n, n)
    E = (W > 0) & ~np.eye(n, dtype=bool)
    return ids, first, E.reshape(len(ids), -1)


def jaccard_matrix(E: np.ndarray) -> np.ndarray:
    F = E.astype(np.float32)
    inter = F @ F.T
    size = F.sum(1)
    return inter / np.maximum(size[:, None] + size[None, :] - inter, 1.0)


# ----------------------------------------------------------------------------
# Ước lượng chu kỳ lặp của topo
# ----------------------------------------------------------------------------
def similarity_profile(E: np.ndarray, ids: np.ndarray, max_lag: int) -> np.ndarray:
    """prof[l] = độ tương đồng Jaccard trung bình giữa snapshot k và k + l."""
    idx = np.full(int(ids.max()) + 1, -1, dtype=np.int64)
    idx[ids] = np.arange(len(ids))
    F = E.astype(np.float32)
    size = F.sum(1)
    prof = np.full(max_lag + 1, np.nan)
    for lag in range(1, max_lag + 1):
        a_ids = ids[ids + lag <= ids.max()]
        b_ids = a_ids + lag
        ok = idx[b_ids] >= 0
        a, b = idx[a_ids[ok]], idx[b_ids[ok]]
        if len(a) == 0:
            continue
        inter = (F[a] * F[b]).sum(1)
        prof[lag] = np.mean(inter / np.maximum(size[a] + size[b] - inter, 1.0))
    return prof


def find_period(prof: np.ndarray, rise: float = 0.3) -> int | None:
    """
    Đỉnh đầu tiên của đường tương đồng sau khi nó đã giảm xuống đáy rồi hồi lên.
    Không phụ thuộc bước thời gian (20 s hay 60 s đều được). None = không thấy chu kỳ.
    """
    p = np.nan_to_num(prof, nan=0.0)
    lo, lag = np.inf, 1
    while lag < len(p):
        lo = min(lo, p[lag])
        if p[lag] > lo + rise:
            break
        lag += 1
    else:
        return None
    while lag + 1 < len(p) and p[lag + 1] >= p[lag]:
        lag += 1
    return lag


# ----------------------------------------------------------------------------
# Gán snapshot vào train / val / test
# ----------------------------------------------------------------------------
def _block_counts(n_blocks: int, ratios) -> tuple[int, int, int]:
    n_va = max(1, round(n_blocks * ratios[1])) if ratios[1] > 0 else 0
    n_te = max(1, round(n_blocks * ratios[2])) if ratios[2] > 0 else 0
    n_tr = n_blocks - n_va - n_te
    if n_tr < 1:
        raise ValueError(f"Chỉ có {n_blocks} khối, quá ít để chia 3 tập. Giảm block_size.")
    return n_tr, n_va, n_te


def assign_random(ids, ratios, seed):
    """Chỉ để SO SÁNH: xáo ngẫu nhiên từng snapshot (cách sai, hay gặp)."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(ids))
    n_va, n_te = round(len(ids) * ratios[1]), round(len(ids) * ratios[2])
    code = np.full(len(ids), TRAIN, dtype=np.int8)
    code[order[:n_va]] = VAL
    code[order[n_va:n_va + n_te]] = TEST
    return code


def assign_blocks(ids, period, block_size, gap, ratios, seed):
    """
    period=None : khối trên trục THỜI GIAN (id), không vòng.
    period=P    : khối trên trục PHA (id mod P), vòng: pha P-1 kề pha 0.
    Ở biên giữa hai khối thuộc hai tập khác nhau, bỏ `gap` snapshot mỗi phía.
    Trả về (code, info); code[i] = TRAIN/VAL/TEST/DROP cho ids[i].
    """
    circular = period is not None
    coord = ids % period if circular else ids
    length = period if circular else int(ids.max()) + 1
    n_all = -(-length // block_size)
    blk = coord // block_size
    present = sorted(set(blk.tolist()))

    n_tr, n_va, n_te = _block_counts(len(present), ratios)
    shuffled = list(present)
    np.random.default_rng(seed).shuffle(shuffled)
    split_of = {b: (TRAIN if i < n_tr else VAL if i < n_tr + n_va else TEST)
                for i, b in enumerate(shuffled)}

    def neighbor(b, step):
        nb = (b + step) % n_all if circular else b + step
        return split_of.get(nb)  # None nếu không có khối đó

    code = np.empty(len(ids), dtype=np.int8)
    for i, (c, b) in enumerate(zip(coord.tolist(), blk.tolist())):
        sp = split_of[b]
        lo, hi = b * block_size, min((b + 1) * block_size, length) - 1
        prev_sp, next_sp = neighbor(b, -1), neighbor(b, +1)
        near_lo = c - lo < gap and prev_sp is not None and prev_sp != sp
        near_hi = hi - c < gap and next_sp is not None and next_sp != sp
        code[i] = DROP if (near_lo or near_hi) else sp

    info = {"n_blocks": len(present), "block_size": block_size, "gap": gap,
            "period": period,
            "blocks": {NAMES[s]: sorted(b for b, v in split_of.items() if v == s)
                       for s in (TRAIN, VAL, TEST)}}
    return code, info


# ----------------------------------------------------------------------------
# Cân bằng thành phần giữa các tập
# ----------------------------------------------------------------------------
def per_snapshot_stats(data, ids, gs_names):
    """Với mỗi snapshot: số mẫu, số mẫu có nguồn là trạm mặt đất, tổng số hop."""
    order = [str(x) for x in data["node_order"]]
    gs_idx = [i for i, nm in enumerate(order) if nm in gs_names]
    src = data["x_sources"].argmax(1)
    src_is_gs = np.isin(src, gs_idx).astype(float)
    hops = ((data["y_paths_indices"] >= 0).sum(1) - 1).astype(float)
    pos = np.searchsorted(ids, data["snapshot_id"])
    n = len(ids)
    return (np.bincount(pos, minlength=n).astype(float),
            np.bincount(pos, weights=src_is_gs, minlength=n),
            np.bincount(pos, weights=hops, minlength=n))


def imbalance(code, stats, min_share):
    """
    Độ lệch thành phần của val/test so với train, lấy mức xấu nhất trong:
    tỉ lệ mẫu có nguồn là trạm mặt đất (điểm %) và số hop trung bình (tương đối).
    inf nếu val hoặc test chiếm < min_share số mẫu còn lại.
    """
    cnt, gs, hp = stats
    total = cnt[code != DROP].sum()

    def rates(c):
        m = code == c
        n = cnt[m].sum()
        return (gs[m].sum() / n, hp[m].sum() / n, n / total) if n > 0 else None

    tr, va, te = rates(TRAIN), rates(VAL), rates(TEST)
    if tr is None or va is None or te is None or va[2] < min_share or te[2] < min_share:
        return float("inf")
    return max(max(abs(x[0] - tr[0]), abs(x[1] - tr[1]) / tr[1]) for x in (va, te))


# ----------------------------------------------------------------------------
# Báo cáo chất lượng cách chia
# ----------------------------------------------------------------------------
def nearest_similarity(J, code, a, b):
    """Với mỗi snapshot thuộc tập a: độ tương đồng lớn nhất với MỘT snapshot thuộc tập b."""
    rows, cols = np.where(code == a)[0], np.where(code == b)[0]
    if len(rows) == 0 or len(cols) == 0:
        return None
    m = J[np.ix_(rows, cols)].max(1)
    return {"median": float(np.median(m)), "p90": float(np.percentile(m, 90)), "max": float(m.max())}


def summarize(J, code):
    kept = (code != DROP).sum()
    return {
        "kept": float(kept / len(code)),
        "val_vs_train": nearest_similarity(J, code, VAL, TRAIN),
        "test_vs_train": nearest_similarity(J, code, TEST, TRAIN),
        "test_vs_val": nearest_similarity(J, code, TEST, VAL),
    }


def print_leak(name, s):
    def f(x):
        return "  -  " if x is None else f"{x['median']:.2f} / {x['p90']:.2f} / {x['max']:.2f}"
    print(f"  {name:<26} giữ {s['kept']:>5.0%} | test↔train {f(s['test_vs_train'])} "
          f"| val↔train {f(s['val_vs_train'])}")


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="Chia oracle_dataset.npz thành train/val/test.")
    ap.add_argument("--in", dest="inp", default="dataset/oracle_dataset.npz")
    ap.add_argument("--outdir", default=None, help="mặc định: thư mục của file vào")
    ap.add_argument("--mode", choices=["phase", "time"], default="phase")
    ap.add_argument("--ratios", type=float, nargs=3, default=(0.7, 0.15, 0.15))
    ap.add_argument("--block-size", type=int, default=None,
                    help="số snapshot mỗi khối (mặc định: 1/12 chu kỳ, hoặc 1/12 độ dài dữ liệu ở --mode time)")
    ap.add_argument("--gap", type=int, default=None, help="snapshot bỏ mỗi phía biên giữa hai tập (mặc định: block/4)")
    ap.add_argument("--period", type=int, default=None, help="chu kỳ (bước); mặc định tự ước lượng")
    ap.add_argument("--seed", type=int, default=42, help="dùng khi --balance-seeds 0")
    ap.add_argument("--balance-seeds", type=int, default=3000,
                    help="thử seed 0..N-1, chọn cách gán khối cho val/test giống train nhất "
                         "về tỉ lệ nguồn=trạm và số hop (0 = tắt, dùng --seed)")
    ap.add_argument("--min-share", type=float, default=0.10,
                    help="val và test mỗi tập phải chiếm ít nhất tỉ lệ này số mẫu còn lại")
    ap.add_argument("--gs-names", nargs="+", default=["Hanoi", "DaNang", "HoChiMinh"])
    ap.add_argument("-v", "--verbose", action="store_true", help="in thêm phần kiểm tra chi tiết")
    ap.add_argument("--compare", action="store_true", help="in bảng so sánh, không ghi file")
    a = ap.parse_args(argv)
    if abs(sum(a.ratios) - 1.0) > 1e-6:
        raise SystemExit(f"--ratios phải cộng bằng 1, nhận {a.ratios}")
    def vprint(*args, **kw):  # chỉ in khi có -v
        if a.verbose:
            print(*args, **kw)

    inp = Path(a.inp)
    if not inp.exists():
        raise SystemExit(f"Không thấy {inp}. Dùng --in để chỉ đường dẫn.")
    outdir = Path(a.outdir) if a.outdir else inp.parent

    data = load_npz(inp)
    n_samples = len(data["snapshot_id"])
    n_nodes = len(data["node_order"])
    ids, first, E = snapshot_edges(data)
    J = jaccard_matrix(E)
    vprint(f"Dữ liệu gốc: {n_samples:,} mẫu, {len(ids)} snapshot, {n_nodes} nút")

    # --- chu kỳ ---
    period = a.period
    if period is None:
        prof = similarity_profile(E, ids, max_lag=len(ids) // 2)
        period = find_period(prof)
    if period is not None:
        vprint(f"Chu kỳ lặp  : {period} bước (topo gần như y hệt sau chừng đó bước, "
               f"độ giống {similarity_profile(E, ids, period)[period]:.2f})")
    else:
        vprint("Chu kỳ lặp  : không thấy trong dữ liệu")
    if a.mode == "phase" and (period is None or ids.max() + 1 < 1.5 * period):
        print("!! Dữ liệu ngắn hơn 1,5 chu kỳ nên không chia theo pha được, chuyển sang --mode time.")
        a.mode = "time"

    def params(span):  # mặc định: ~12 khối, đệm = 1/4 khối
        b = a.block_size or max(2, round(span / 12))
        return b, (a.gap if a.gap is not None else max(1, b // 4))

    span = period if a.mode == "phase" else int(ids.max()) + 1
    block, gap = params(span)
    use_period = period if a.mode == "phase" else None

    if a.compare:
        print("\nĐộ giống (median / p90 / max) của snapshot val/test với snapshot GẦN NHẤT trong train.")
        print(f"Càng thấp càng ít rò rỉ (seed={a.seed}):\n")
        print_leak("random từng snapshot", summarize(J, assign_random(ids, a.ratios, a.seed)))
        bt, gt = params(int(ids.max()) + 1)
        c, _ = assign_blocks(ids, None, bt, gt, a.ratios, a.seed)
        print_leak(f"khối thời gian ({bt}/{gt})", summarize(J, c))
        if period is not None:
            bp, gp = params(period)
            c, _ = assign_blocks(ids, period, bp, gp, a.ratios, a.seed)
            print_leak(f"khối pha ({bp}/{gp}) mặc định", summarize(J, c))
        print("(trong ngoặc: block/gap)")
        return

    stats = per_snapshot_stats(data, ids, set(a.gs_names))
    seed, score = a.seed, None
    if a.balance_seeds > 0:
        scored = [(imbalance(assign_blocks(ids, use_period, block, gap, a.ratios, sd_)[0], stats, a.min_share), sd_)
                  for sd_ in range(a.balance_seeds)]
        best_score, best_seed = min(scored)
        if np.isfinite(best_score):
            seed, score = best_seed, best_score
        else:
            print(f"!! Không seed nào thoả val/test >= {a.min_share:.0%} số mẫu; dùng --seed {a.seed}.")
    code, info = assign_blocks(ids, use_period, block, gap, a.ratios, seed)

    # --- ghi file ---
    code_of_row = np.full(int(ids.max()) + 1, DROP, dtype=np.int8)
    code_of_row[ids] = code
    sample_code = code_of_row[data["snapshot_id"]]
    per_sample = [k for k, v in data.items() if k != "node_order" and v.shape[:1] == (n_samples,)]
    kept_total = int((sample_code != DROP).sum())

    n_drop = int((code == DROP).sum())
    gs_idx = [i for i, nm in enumerate(data["node_order"]) if str(nm) in set(a.gs_names)]
    hops = (data["y_paths_indices"] >= 0).sum(1) - 1

    print(f"Chia xong {n_samples:,} mẫu ({len(ids)} snapshot):\n")
    print(f"{'':<8}{'Snapshot':>10}{'Mẫu':>9}{'Tỉ lệ':>8}")
    ids_by_split, comp = {}, {}
    for s in (TRAIN, VAL, TEST):
        rows = np.where(sample_code == s)[0]
        ids_by_split[NAMES[s]] = ids[code == s].tolist()
        gs_src = np.isin(data["x_sources"][rows].argmax(1), gs_idx).mean() if len(rows) else 0
        comp[s] = (float(hops[rows].mean()) if len(rows) else 0.0, float(gs_src))
        print(f"{NAMES[s].capitalize():<8}{int((code == s).sum()):>10}{len(rows):>9,}{len(rows) / kept_total:>8.1%}")
        extra = {k: data[k] for k in ("node_positions_km", "node_latlon_alt", "snapshot_times") if k in data}
        np.savez_compressed(
            outdir / f"{NAMES[s]}.npz",
            **{k: data[k][rows] for k in per_sample},
            **extra,
            node_order=data["node_order"],
        )
    print(f"\nBỏ {n_drop}/{len(ids)} snapshot ở biên để 3 tập không dính sát nhau (còn {kept_total:,} mẫu).")

    if a.verbose:
        how = "theo pha của chu kỳ" if a.mode == "phase" else "theo khối thời gian liền nhau"
        print("\n--- Chi tiết kiểm tra ---")
        print(f"Cách chia: {how}; {info['n_blocks']} khối x {block} snapshot, chừa {gap} ở mỗi biên.")
        print(f"\n{'':<8}{'Hop TB':>9}{'Nguồn là trạm':>15}")
        for s in (TRAIN, VAL, TEST):
            print(f"{NAMES[s].capitalize():<8}{comp[s][0]:>9.2f}{comp[s][1]:>15.1%}")
        d_gs = max(abs(comp[v][1] - comp[TRAIN][1]) for v in (VAL, TEST)) * 100
        d_hop = max(abs(comp[v][0] - comp[TRAIN][0]) / comp[TRAIN][0] for v in (VAL, TEST)) * 100
        ok = d_gs <= 5 and d_hop <= 10
        print(f"-> Ba tập chênh nhau tối đa {d_gs:.1f} điểm % / {d_hop:.1f}%: "
              f"{'cân bằng tốt' if ok else 'HƠI LỆCH, thử đổi --block-size hoặc --seed'}.")
        summ = summarize(J, code)
        base = summarize(J, assign_random(ids, a.ratios, a.seed))["test_vs_train"]
        t = summ["test_vs_train"]
        if t is not None and base is not None:
            print(f"\nTest giống Train: trung vị {t['median']:.2f}, cao nhất {t['max']:.2f} "
                  f"(thang 0-1, càng thấp càng tốt; chia ngẫu nhiên sẽ là {base['median']:.2f}).")

    # kiểm tra cứng: không snapshot nào ở hai tập
    for x, y in ((TRAIN, VAL), (TRAIN, TEST), (VAL, TEST)):
        assert not set(ids[code == x]) & set(ids[code == y]), f"{NAMES[x]} và {NAMES[y]} trùng snapshot"

    with open(outdir / "split_info.json", "w", encoding="utf-8") as f:
        json.dump({"input": str(inp), "mode": a.mode, "ratios": list(a.ratios), "seed": seed, "balance_score": score,
                   "block_size": block, "gap": gap, "period": period,
                   "blocks": info["blocks"], "snapshot_ids": ids_by_split,
                   "dropped_snapshot_ids": ids[code == DROP].tolist()},
                  f, ensure_ascii=False, indent=1)
    print(f"\nĐã ghi train.npz, val.npz, test.npz, split_info.json vào {outdir}")
    if not a.verbose:
        print("(thêm -v để xem phần kiểm tra chi tiết)")


if __name__ == "__main__":
    main()
