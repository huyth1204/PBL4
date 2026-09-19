"""Test cho src/data/dataset.py. Chạy: python tests/test_dataset.py"""
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
from dataset.dataset import (  # noqa: E402
    cut_nexthop, mask_row, normalize_state, one_hot,
    snapshot_id, split_by_snapshot,
)

# Mạng 4 nút có hướng: A->B=10, B->C=20, C->D=5. Định tuyến A->D.
def good_sample():
    W = [0, 10, -1, -1,
         -1, 0, 20, -1,
         -1, -1, 0, 5,
         -1, -1, -1, 0]
    return {"x": [float(v) for v in W], "source": "A", "target": "D",
            "y_star": ["A", "B", "C", "D"], "node_order": ["A", "B", "C", "D"]}


def test_normalize():
    x = normalize_state([0, 10, -1, -1, -1, 0, 20, -1, -1, -1, 0, 5, -1, -1, -1, 0], 4)
    assert x.min() >= 0 and x.max() <= 1
    assert abs(x[1] - 0.4) < 1e-6 and abs(x[6] - 0.8) < 1e-6  # 10/25, 20/25


def test_normalize_bat_N_sai():
    try:
        normalize_state([1, 2, 3], 4)  # 3 != 16
        assert False
    except ValueError:
        pass


def test_mask_dung_hang():
    W = np.array([[0, 10, -1, -1], [-1, 0, 20, -1], [-1, -1, 0, 5], [-1, -1, -1, 0]], float)
    assert mask_row(W, 0).tolist() == [0, 1, 0, 0]
    assert mask_row(W, 2).tolist() == [0, 0, 0, 1]


def test_cut_ra_3_nexthop():
    items = cut_nexthop(good_sample())
    assert len(items) == 3
    assert [it["next"] for it in items] == [1, 2, 3]
    assert [it["cur"] for it in items] == [0, 1, 2]
    assert all(it["dst"] == 3 for it in items)


def test_cut_co_mask_dung():
    items = cut_nexthop(good_sample())
    assert items[0]["mask"].tolist() == [0, 1, 0, 0]
    assert all(it["mask"][it["next"]] == 1 for it in items)  # nhãn luôn nằm trong mask


def test_thieu_node_order_bao_loi():
    s = good_sample()
    del s["node_order"]
    try:
        cut_nexthop(s)
        assert False
    except KeyError:
        pass


def test_nhan_mau_thuan_bao_loi():
    s = good_sample()
    s["y_star"] = ["A", "C", "D"]  # A->C không có cạnh
    try:
        cut_nexthop(s)
        assert False
    except ValueError:
        pass


def test_split_theo_snapshot_khong_ro_ri():
    s1, s2 = good_sample(), good_sample()
    s2["x"] = [v + 1 for v in s2["x"]]  # snapshot khác
    b = split_by_snapshot([s1, s2], seed=1)
    total = len(b["train"]) + len(b["val"]) + len(b["test"])
    assert total == 2
    # cùng snapshot_id không bao giờ ở hai tập
    ids = {k: {snapshot_id(x) for x in v} for k, v in b.items()}
    assert not (ids["train"] & ids["test"])


def test_torch_dataset_neu_co():
    try:
        import torch  # noqa: F401
    except ImportError:
        print("  (bỏ qua test torch — chưa cài)")
        return
    from dataset.dataset import RoutingDataset, apply_validity_mask
    ds = RoutingDataset([good_sample()])
    assert len(ds) == 3 and ds.n_nodes == 4
    item = ds[0]
    assert tuple(item["condition"].shape) == (4 * 4 + 4 + 4,)
    assert tuple(item["target"].shape) == (4,)
    # mặt nạ: nút cấm có logit cao vẫn không được chọn
    logits = torch.tensor([[0.1, 9.9, 0.2, 0.3]])
    mask = torch.tensor([[1.0, 0, 1, 0]])
    assert apply_validity_mask(logits, mask).argmax().item() in (0, 2)


def _run():
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    ok = 0
    for n, f in tests:
        try:
            f(); print(f"  [PASS] {n}"); ok += 1
        except Exception as e:
            print(f"  [FAIL] {n}: {e}")
    print(f"\n{ok}/{len(tests)} đạt.")
    return ok == len(tests)


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)