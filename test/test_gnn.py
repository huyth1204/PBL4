"""Test cho nấc B (data_v2 + gnn_diffusion). Chạy: python test/test_gnn.py"""
import heapq
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.ai.data_v2 import SagsinData, make_batch, node_features, valid_mask  # noqa: E402
from src.ai.gnn_classifier import GraphClassifier  # noqa: E402
from src.ai.gnn_diffusion import GraphDiffusion, GraphEncoder  # noqa: E402
from src.ai.gnn_diffusion_v2 import GraphDiffusionV2  # noqa: E402
from src.ai.train_gnn import encode_pair, to_t  # noqa: E402

torch.set_num_threads(2)
KINDS8 = ["user", "user", "gateway", "gateway", "haps", "haps", "leo", "leo"]
LAYER = {"user": "ground", "gateway": "ground", "haps": "air", "uav": "air", "leo": "space"}


def relay_dijkstra(W, src, can_relay):
    """Dijkstra có quy tắc chuyển tiếp: không đi RA từ nút không relay (trừ nguồn)."""
    N = len(W)
    dist, prev, pq = np.full(N, np.inf), -np.ones(N, int), [(0.0, src)]
    dist[src] = 0
    while pq:
        dcur, u = heapq.heappop(pq)
        if dcur > dist[u] or (u != src and not can_relay[u]):
            continue
        for v in np.flatnonzero(W[u] > 0):
            if dcur + W[u, v] < dist[v]:
                dist[v], prev[v] = dcur + W[u, v], u
                heapq.heappush(pq, (dist[v], v))
    return dist, prev


def make_toy_npz(path, n_snap=20, pairs=10, seed=0):
    rng = np.random.default_rng(seed)
    N = len(KINDS8)
    can_relay = np.array([k != "user" for k in KINDS8])
    endpoints = [i for i, k in enumerate(KINDS8) if k in ("user", "gateway")]
    Ws, paths, sids = [], [], []
    for s in range(n_snap):
        W = -np.ones((N, N), np.float32)
        order = rng.permutation(N)
        for a, b in zip(order[:-1], order[1:]):            # cây khung -> luôn liên thông
            W[a, b] = W[b, a] = rng.uniform(1, 15)
        for _ in range(6):
            a, b = rng.choice(N, 2, replace=False)
            W[a, b] = W[b, a] = rng.uniform(1, 15)
        np.fill_diagonal(W, -1)
        Ws.append(W)
        for _ in range(pairs):
            src, dst = rng.choice(endpoints, 2, replace=False)
            dist, prev = relay_dijkstra(W, src, can_relay)
            if not np.isfinite(dist[dst]):
                continue
            p, v = [], dst
            while v != -1:
                p.append(int(v)); v = prev[v]
            paths.append(p[::-1]); sids.append(s)
    L = max(len(p) for p in paths)
    Y = np.array([p + [-1] * (L - len(p)) for p in paths], np.int32)
    np.savez(path, W_snapshots=np.stack(Ws), y_paths_indices=Y, snapshot_id=np.array(sids, np.int32),
             node_order=np.array([f"n{i}" for i in range(N)]),
             node_kind=np.array(KINDS8), node_layer=np.array([LAYER[k] for k in KINDS8]))


_TMP = tempfile.mkdtemp()
TOY = Path(_TMP) / "toy.npz"
make_toy_npz(TOY)


def test_mask_loai_user_tru_khi_la_dich():
    can_relay = np.array([True, False, True])
    rows = np.array([[0.0, 5.0, 3.0], [0.0, 5.0, 3.0]])
    m = valid_mask(rows, np.array([2, 1]), can_relay)
    assert m[0].tolist() == [False, False, True]    # đích=2: user 1 bị loại
    assert m[1].tolist() == [False, True, True]     # đích=1: user 1 chính là đích -> hợp lệ


def test_mask_loai_nut_da_di_qua():
    rows = np.array([[0.0, 5.0, 3.0]])
    visited = np.array([[False, False, True]])
    m = valid_mask(rows, np.array([1]), np.array([True, True, True]), visited)
    assert m[0].tolist() == [False, True, False]


def test_co_dich_va_bo_dieu_kien():
    d = SagsinData(TOY)
    W = d.W[:3]
    X = node_features(W, d.static_feat, np.array([0, 2, 3]))
    assert X.shape == (3, d.N, d.f_in)
    assert X[1, 2, -1] == 1.0 and X[1, :, -1].sum() == 1.0
    Xu = node_features(W, d.static_feat, np.array([0, 2, 3]), np.array([False, True, False]))
    assert Xu[1, :, -1].sum() == 0.0                # CFG: bỏ cờ đích


def test_nhan_luon_nam_trong_mat_na():
    d = SagsinData(TOY)
    b = make_batch(d, np.arange(len(d)))
    assert b["mask"][np.arange(len(b["nxt"])), b["nxt"]].all()


def test_loss_va_gradient():
    d = SagsinData(TOY)
    m = GraphDiffusion(d.f_in, 32, 2)
    loss = m.training_loss(to_t(make_batch(d, np.arange(8), 0.3, np.random.default_rng(0))))
    assert loss.dim() == 0 and loss.item() > 0
    loss.backward()
    assert m.enc.inp.weight.grad is not None


def test_khong_bao_gio_chon_nut_bi_cam():
    d = SagsinData(TOY)
    m = GraphDiffusion(d.f_in, 32, 2).eval()
    b = make_batch(d, np.arange(len(d)))
    hc, hu = encode_pair(m, b["W"], d.static_feat, b["dst"])
    sp = torch.from_numpy(b["step_p"])
    pred = m.decide(hc[sp], hu[sp], torch.from_numpy(b["cur"]), torch.from_numpy(b["dst"][b["step_p"]]),
                    torch.from_numpy(b["rows"]), torch.from_numpy(b["mask"]), n_steps=5, guidance=3.0)
    assert b["mask"][np.arange(len(pred)), pred.numpy()].all()


def test_gnn_bat_bien_thu_tu_nut():
    # Đổi thứ tự nút -> đầu ra đổi theo đúng hoán vị (không học thuộc chỉ số nút)
    torch.manual_seed(0)
    enc = GraphEncoder(12, 16, 3).eval()
    N = 7
    W = torch.rand(1, N, N) * 10
    W = torch.where(W > 5, W, torch.full_like(W, -1.0))
    W = torch.maximum(W, W.transpose(1, 2))
    X = torch.rand(1, N, 12)
    perm = torch.randperm(N)
    h = enc(X, W)
    hp = enc(X[:, perm], W[:, perm][:, :, perm])
    assert torch.allclose(h[:, perm], hp, atol=1e-5)


def test_chay_duoc_voi_so_nut_khac_nhau():
    enc = GraphEncoder(12, 16, 2).eval()
    for N in (5, 40, 173):
        W = torch.full((2, N, N), -1.0)
        W[:, 0, 1] = W[:, 1, 0] = 3.0
        assert enc(torch.rand(2, N, 12), W).shape == (2, N, 16)


def test_hoc_duoc_dinh_tuyen_tren_mang_nho():
    # END-TO-END: trên mạng 8 nút đủ 3 tầng, mô hình phải bắt chước được Dijkstra
    torch.manual_seed(0)
    d = SagsinData(TOY)
    m = GraphDiffusion(d.f_in, 48, 3)
    opt = torch.optim.Adam(m.parameters(), 3e-3)
    rng = np.random.default_rng(0)
    for _ in range(400):
        b = to_t(make_batch(d, rng.choice(len(d), 32, replace=False), 0.15, rng))
        loss = m.training_loss(b)
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    b = make_batch(d, np.arange(len(d)))
    hc, hu = encode_pair(m, b["W"], d.static_feat, b["dst"])
    sp = torch.from_numpy(b["step_p"])
    torch.manual_seed(0)
    pred = m.decide(hc[sp], hu[sp], torch.from_numpy(b["cur"]), torch.from_numpy(b["dst"][b["step_p"]]),
                    torch.from_numpy(b["rows"]), torch.from_numpy(b["mask"]), n_steps=20, guidance=2.0)
    acc = (pred.numpy() == b["nxt"]).mean()
    rand = np.mean(1.0 / b["mask"].sum(1))
    print(f"    (next-hop {acc * 100:.0f}% so với ngẫu nhiên {rand * 100:.0f}%)")
    assert acc >= 0.8, f"chỉ đạt {acc:.2f}"


def test_phan_loai_doi_chung_hoc_duoc_va_ton_trong_mat_na():
    # biến thể bỏ diffusion: cùng giao diện decide, không chọn nút bị cấm, học được trên mạng nhỏ
    torch.manual_seed(0)
    d = SagsinData(TOY)
    m = GraphClassifier(d.f_in, 48, 3)
    opt = torch.optim.Adam(m.parameters(), 3e-3)
    rng = np.random.default_rng(0)
    for _ in range(300):
        loss = m.training_loss(to_t(make_batch(d, rng.choice(len(d), 32, replace=False), 0.0, rng)))
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    b = make_batch(d, np.arange(len(d)))
    hc, hu = encode_pair(m, b["W"], d.static_feat, b["dst"])
    sp = torch.from_numpy(b["step_p"])
    args = (torch.from_numpy(b["cur"]), torch.from_numpy(b["dst"][b["step_p"]]),
            torch.from_numpy(b["rows"]), torch.from_numpy(b["mask"]))
    pred = m.decide(hc[sp], hu[sp], *args).numpy()
    samp = m.decide(hc[sp], hu[sp], *args, temperature=1.0).numpy()
    assert b["mask"][np.arange(len(pred)), pred].all() and b["mask"][np.arange(len(samp)), samp].all()
    acc = (pred == b["nxt"]).mean()
    print(f"    (phân loại: next-hop {acc * 100:.0f}%)")
    assert acc >= 0.8


def test_dm_v2_moi_cau_hinh_chay_duoc_va_ton_trong_mat_na():
    d = SagsinData(TOY)
    b = make_batch(d, np.arange(len(d)), 0.3, np.random.default_rng(1))
    for param, sc, at in [("eps", False, False), ("x0ce", False, False), ("x0ce", True, False), ("x0ce", True, True)]:
        torch.manual_seed(0)
        m = GraphDiffusionV2(d.f_in, 32, 0, param=param, self_cond=sc, attn=at)
        loss = m.training_loss(to_t(b))
        loss.backward()
        assert torch.isfinite(loss) and m.den.loc_in.weight.grad is not None
        m.eval()
        hc, hu = encode_pair(m, b["W"], d.static_feat, b["dst"])
        sp = torch.from_numpy(b["step_p"])
        pred = m.decide(hc[sp], hu[sp], torch.from_numpy(b["cur"]), torch.from_numpy(b["dst"][b["step_p"]]),
                        torch.from_numpy(b["rows"]), torch.from_numpy(b["mask"]), n_steps=4, guidance=3.0)
        assert b["mask"][np.arange(len(pred)), pred.numpy()].all(), (param, sc, at)


def test_dm_v2_x0ce_hoc_duoc_tren_mang_nho():
    torch.manual_seed(0)
    d = SagsinData(TOY)
    m = GraphDiffusionV2(d.f_in, 48, 3, param="x0ce", self_cond=True)
    opt = torch.optim.Adam(m.parameters(), 3e-3)
    rng = np.random.default_rng(0)
    for _ in range(300):
        loss = m.training_loss(to_t(make_batch(d, rng.choice(len(d), 32, replace=False), 0.15, rng)))
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    b = make_batch(d, np.arange(len(d)))
    hc, hu = encode_pair(m, b["W"], d.static_feat, b["dst"])
    sp = torch.from_numpy(b["step_p"])
    pred = m.decide(hc[sp], hu[sp], torch.from_numpy(b["cur"]), torch.from_numpy(b["dst"][b["step_p"]]),
                    torch.from_numpy(b["rows"]), torch.from_numpy(b["mask"]), n_steps=10, guidance=2.0)
    acc = (pred.numpy() == b["nxt"]).mean()
    print(f"    (DM x0+CE: next-hop {acc * 100:.0f}%)")
    assert acc >= 0.8


def test_lookahead_dung_cot_dich_va_bi_xoa_khi_bo_dieu_kien():
    d = SagsinData(TOY)
    W, dst = d.W[:2], np.array([2, 3])
    X = node_features(W, d.static_feat, dst, None, True)
    assert X.shape[-1] == d.f_in + 2
    assert np.array_equal(X[0, :, -2], (W[0, :, 2] > 0).astype(np.float32))
    Xu = node_features(W, d.static_feat, dst, np.array([True, False]), True)
    assert Xu[0, :, -2:].sum() == 0 and np.array_equal(Xu[1], X[1])


def test_phan_loai_co_attention():
    d = SagsinData(TOY)
    m = GraphClassifier(d.f_in, 32, 0, attn=True)
    loss = m.training_loss(to_t(make_batch(d, np.arange(8))))
    loss.backward()
    assert m.head.cand.attn.in_proj_weight.grad is not None


def test_yen_khop_vet_can_va_dung_quy_tac_chuyen_tiep():
    from src.ai.nhieu_tuyen import route_delay, yen_near_optimal
    d = SagsinData(TOY)
    can = d.can_relay
    for s in range(0, 20, 4):
        W = d.W[s]
        for u, v in [(0, 1), (2, 3), (0, 3)]:
            allp = []

            def dfs(x, path):
                if x == v:
                    allp.append(list(path))
                    return
                if x != u and not can[x]:
                    return
                for y in np.flatnonzero(W[x] > 0):
                    if y not in path:
                        path.append(int(y)); dfs(y, path); path.pop()
            dfs(u, [u])
            if not allp:
                continue
            best = min(route_delay(W, p) for p in allp)
            ref = sorted(route_delay(W, p) for p in allp if route_delay(W, p) <= best * 1.2 + 1e-9)[:3]
            got = yen_near_optimal(W, u, v, can, k=3, eps=0.2)
            assert [round(route_delay(W, p), 5) for p in got] == [round(x, 5) for x in ref]
            for p in got:
                assert all(can[x] for x in p[1:-1]) and len(set(p)) == len(p)


def test_du_phong_tranh_lien_ket_hong():
    from src.ai.nhieu_tuyen import evaluate_samples
    d = SagsinData(TOY)
    i = next(k for k in range(len(d)) if len(d.paths[k]) >= 3)
    prim = tuple(d.paths[i])
    r = evaluate_samples(d, [i], [prim], [[prim, prim]])
    assert r["so_tuyen_khac_nhau_TB"] == 1 and (np.isnan(r["co_san_du_phong"]) or r["co_san_du_phong"] == 0)


def test_dm_lay_mau_eta_ton_trong_mat_na():
    d = SagsinData(TOY)
    m = GraphDiffusionV2(d.f_in, 32, 2, param="x0ce").eval()
    b = make_batch(d, np.arange(len(d)))
    hc, hu = encode_pair(m, b["W"], d.static_feat, b["dst"])
    sp = torch.from_numpy(b["step_p"])
    for eta in (0.0, 1.0):
        pred = m.decide(hc[sp], hu[sp], torch.from_numpy(b["cur"]), torch.from_numpy(b["dst"][b["step_p"]]),
                        torch.from_numpy(b["rows"]), torch.from_numpy(b["mask"]), n_steps=6, guidance=0.5, eta=eta)
        assert b["mask"][np.arange(len(pred)), pred.numpy()].all()


def test_du_lieu_that_neu_co():
    dd = ROOT / "dataset"
    if not (dd / "train.npz").exists():
        print("    (bỏ qua: chưa có dataset/train.npz)")
        return
    if "W_snapshots" not in np.load(dd / "train.npz").files:
        print("    (bỏ qua: dataset/train.npz vẫn là định dạng 2 tầng cũ)")
        return
    d = SagsinData(dd / "train.npz")
    b = make_batch(d, np.arange(min(500, len(d))))
    assert b["mask"][np.arange(len(b["nxt"])), b["nxt"]].all()


def _run():
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    ok = 0
    for n, f in tests:
        try:
            f()
            print(f"  [PASS] {n}")
            ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"  [FAIL] {n}: {type(e).__name__}: {e}")
    print(f"\n{ok}/{len(tests)} đạt.")
    return ok == len(tests)


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
