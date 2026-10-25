"""Test cho src/data/dataset.py. Chạy: python tests/test_dataset.py"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset.dataset import (  # noqa: E402
    cut_nexthop, mask_row, normalize_state, one_hot,
    snapshot_id, split_by_snapshot,
)
from test.tien_ich_test import BoQua, chay_tests, dung_utf8  # noqa: E402

dung_utf8()
ROOT = Path(__file__).resolve().parents[1]

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
    assert abs(x[1] - 0.5) < 1e-6 and abs(x[6] - 1.0) < 1e-6  # 10/20, 20/20


def test_normalize_bat_N_sai():
    try:
        normalize_state([1, 2, 3], 4)  # 3 != 16
        assert False
    except ValueError:
        pass


def test_mask_dung_hang():
    W = np.array([[0, 10, -1, -1], [-1, 0, 20, -1], [-1, -1, 0, 5], [-1, -1, -1, 0]], float)
    # Dùng `== 1` (mặt nạ chỉ cần đúng/sai), không khẳng định 0 nghĩa là "không nối":
    # trọng số 0 là liên kết 0 ms và phải hợp lệ.
    assert (mask_row(W, 0) == [1, 1, 0, 0]).all()
    assert (mask_row(W, 2) == [0, 0, 1, 1]).all()


def test_lien_ket_0ms_van_hop_le():
    # Cột 1 của hàng 0 = 0.0 (liên kết 0 ms) phải được coi là HỢP LỆ,
    # còn -1 (không nối) thì không.
    W = np.array([[0, 0.0, -1.0], [-1, 0, 7.0], [-1, -1, 0]], float)
    assert mask_row(W, 0).tolist() == [1, 1, 0]
    assert mask_row(W, 1).tolist() == [0, 1, 1]


def test_cut_ra_3_nexthop():
    items = cut_nexthop(good_sample())
    assert len(items) == 3
    assert [it["next"] for it in items] == [1, 2, 3]
    assert [it["cur"] for it in items] == [0, 1, 2]
    assert all(it["dst"] == 3 for it in items)


def test_cut_co_mask_dung():
    items = cut_nexthop(good_sample())
    assert items[0]["mask"].tolist() == [1, 1, 0, 0]
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
    # Cần >= 3 snapshot để chia đủ 3 tập; thêm s3 cho hợp lệ.
    s1, s2, s3 = good_sample(), good_sample(), good_sample()
    s2["x"] = [v + 1 for v in s2["x"]]  # snapshot khác
    s3["x"] = [v + 2 for v in s3["x"]]
    b = split_by_snapshot([s1, s2, s3], seed=1)
    total = len(b["train"]) + len(b["val"]) + len(b["test"])
    assert total == 3
    # Không tập nào rỗng, và cùng snapshot_id không bao giờ ở hai tập
    assert all(len(v) >= 1 for v in b.values()), {k: len(v) for k, v in b.items()}
    ids = {k: {snapshot_id(x) for x in v} for k, v in b.items()}
    assert not (ids["train"] & ids["test"])
    assert not (ids["train"] & ids["val"])
    assert not (ids["val"] & ids["test"])


def test_split_qua_it_snapshot_bao_loi():
    # 2 snapshot trước đây cho val RỖNG mà không báo gì; nay phải báo lỗi rõ ràng.
    s1, s2 = good_sample(), good_sample()
    s2["x"] = [v + 1 for v in s2["x"]]
    try:
        split_by_snapshot([s1, s2], seed=1)
        assert False, "phải ném ValueError"
    except ValueError:
        pass


def test_split_nhieu_snapshot_khong_rong_tap_nao():
    base = [good_sample() for _ in range(9)]
    for k, s in enumerate(base):
        s["x"] = [v + k for v in s["x"]]
    b = split_by_snapshot(base, seed=3)
    assert all(len(v) >= 1 for v in b.values()), {k: len(v) for k, v in b.items()}


def test_torch_dataset_neu_co():
    try:
        import torch  # noqa: F401
    except ImportError:
        raise BoQua("chưa cài torch")
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


def test_doc_npz_that_neu_co():
    """Cầu nối dữ liệu thật: RoutingDataset phải đọc được npz của oracle_labeler."""
    f = ROOT / "dataset" / "oracle_dataset_v2.npz"
    if not f.exists():
        raise BoQua("chưa có dataset/oracle_dataset_v2.npz")
    from dataset.dataset import RoutingDataset, samples_from_npz
    raw = samples_from_npz(f)
    assert raw, "không dựng được mẫu nào từ npz"
    s = raw[0]
    assert len(s["x"]) == len(s["node_order"]) ** 2
    assert s["y_star"][0] == s["source"] and s["y_star"][-1] == s["target"], s["y_star"]
    ds = RoutingDataset(f)
    assert ds.n_nodes == len(s["node_order"]) and len(ds) > 0


def _run():
    return chay_tests(globals(), "test_dataset")


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)