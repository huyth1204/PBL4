"""
Giai đoạn 4 (phần dữ liệu): PyTorch Dataset đọc nhãn Dijkstra
===============================================================

Module này định nghĩa `RoutingDataset`, lớp kế thừa `torch.utils.data.Dataset`,
dùng để nạp dữ liệu (x, source, target, y*) do `oracle_labeler.py` sinh ra
(mặc định: `dataset/sample_labels.json`) và chuẩn bị sẵn cho huấn luyện
GNN + Diffusion:

1. Đọc toàn bộ file .json (danh sách mẫu {x, source, target, y_star, n_hops}).
2. Xây "từ điển" ánh xạ tên nút (vd: "STARLINK-1080") -> chỉ số nguyên,
   dùng chung cho toàn bộ dataset (không phải riêng từng snapshot), để
   nhóm AI mã hoá one-hot / embedding cho nguồn - đích.
3. Với mỗi mẫu:
      - x       -> tensor trạng thái mạng (vector trọng số đã flatten)
      - src/dst -> chỉ số nguyên (int) + vector one-hot, dùng làm điều kiện
      - cond    -> điều kiện đầy đủ = ghép [x || one-hot(src) || one-hot(dst)]
      - path    -> danh sách chỉ số nút trên đường đi tối ưu y* (đã ánh xạ
                   qua từ điển ở bước 2), đệm (pad) về cùng độ dài để gom
                   batch được (giá trị đệm = PAD_IDX = -1).
4. Cung cấp thêm `collate_fn` để DataLoader gộp các mẫu có đường đi dài
   ngắn khác nhau thành 1 batch.

Chạy thử:
    python src/data/dataset.py
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

PAD_IDX = -1  # chỉ số đệm (padding) cho các bước ngoài độ dài đường đi thực


# ----------------------------------------------------------------------------
# Bước 1: xây từ điển tên nút -> chỉ số (dùng chung toàn dataset)
# ----------------------------------------------------------------------------
def build_node_vocab(samples: list[dict]) -> dict[str, int]:
    """
    Duyệt toàn bộ mẫu, gom tất cả tên nút xuất hiện ở 'source', 'target'
    và trong từng 'y_star', trả về từ điển {tên_nút: chỉ_số} theo thứ tự
    bảng chữ cái (ổn định giữa các lần chạy, để train/val/test khớp nhau
    nếu build lại từ cùng 1 tập dữ liệu).
    """
    nodes = set()
    for s in samples:
        nodes.add(s["source"])
        nodes.add(s["target"])
        nodes.update(s["y_star"])
    return {name: idx for idx, name in enumerate(sorted(nodes))}


# ----------------------------------------------------------------------------
# Bước 2: lớp Dataset chính
# ----------------------------------------------------------------------------
class RoutingDataset(Dataset):
    """
    Đọc file nhãn (mặc định dataset/sample_labels.json) sinh bởi
    oracle_labeler.py và trả về từng mẫu (x, source, target, y*) đã được
    mã hoá sẵn cho huấn luyện.

    Tham số
    -------
    json_path : str | Path
        Đường dẫn tới file .json nhãn.
    node_vocab : dict[str, int] | None
        Từ điển tên nút -> chỉ số dùng chung. Nếu None, tự xây từ chính
        file nhãn này (build_node_vocab). Nên truyền vào khi có nhiều
        Dataset (train/val/test) cần dùng chung 1 từ điển.
    max_hops : int | None
        Độ dài tối đa (số nút) của đường đi dùng để đệm (pad). Nếu None,
        tự lấy độ dài đường đi dài nhất có trong dữ liệu. Nên truyền vào
        khi val/test có thể có đường đi dài hơn tập train đã thấy.
    """

    def __init__(
        self,
        json_path: str | Path = "dataset/sample_labels.json",
        node_vocab: dict[str, int] | None = None,
        max_hops: int | None = None,
    ) -> None:
        super().__init__()
        self.json_path = Path(json_path)
        with open(self.json_path, "r", encoding="utf-8") as f:
            self.samples: list[dict] = json.load(f)

        if len(self.samples) == 0:
            raise ValueError(f"File nhãn rỗng: {self.json_path}")

        # --- từ điển ánh xạ nguồn/đích/đường đi -> chỉ số ---
        self.node_vocab = node_vocab if node_vocab is not None else build_node_vocab(self.samples)
        self.vocab_size = len(self.node_vocab)

        # --- độ dài đường đi tối đa để pad ---
        longest = max(len(s["y_star"]) for s in self.samples)
        self.max_hops = max_hops if max_hops is not None else longest

        # --- kích thước vector trạng thái mạng x (phải đều nhau trong file) ---
        self.state_dim = len(self.samples[0]["x"])

    def __len__(self) -> int:
        return len(self.samples)

    def _one_hot(self, node_name: str) -> torch.Tensor:
        vec = torch.zeros(self.vocab_size, dtype=torch.float32)
        idx = self.node_vocab.get(node_name)
        if idx is not None:
            vec[idx] = 1.0
        return vec

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        s = self.samples[i]

        x = torch.tensor(s["x"], dtype=torch.float32)  # (state_dim,)

        src_idx = self.node_vocab[s["source"]]
        dst_idx = self.node_vocab[s["target"]]
        src_onehot = self._one_hot(s["source"])
        dst_onehot = self._one_hot(s["target"])

        # ghép điều kiện: trạng thái mạng (x) + one-hot(nguồn) + one-hot(đích)
        condition = torch.cat([x, src_onehot, dst_onehot], dim=0)

        # ánh xạ đường đi y* (tên nút) -> chỉ số, rồi đệm về max_hops
        path_idx = [self.node_vocab[n] for n in s["y_star"]]
        n_hops = len(path_idx)
        padded_path = path_idx + [PAD_IDX] * (self.max_hops - n_hops)

        return {
            "x": x,
            "source_idx": torch.tensor(src_idx, dtype=torch.long),
            "target_idx": torch.tensor(dst_idx, dtype=torch.long),
            "source_onehot": src_onehot,
            "target_onehot": dst_onehot,
            "condition": condition,
            "path": torch.tensor(padded_path, dtype=torch.long),
            "n_hops": torch.tensor(n_hops, dtype=torch.long),
        }


# ----------------------------------------------------------------------------
# Bước 3: collate_fn (mọi tensor đã cùng shape sau khi pad ở __getitem__,
# nhưng khai báo tường minh để dễ chỉnh sửa cách gộp batch sau này)
# ----------------------------------------------------------------------------
def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
    return {key: torch.stack([b[key] for b in batch], dim=0) for key in batch[0]}


# ----------------------------------------------------------------------------
# Demo / smoke test
# ----------------------------------------------------------------------------
def main():
    json_path = Path("dataset/sample_labels.json")
    if not json_path.exists():
        raise FileNotFoundError(f"Không tìm thấy {json_path}.")

    dataset = RoutingDataset(json_path)
    print(f"[1/3] Nạp {len(dataset)} mẫu từ {json_path}")
    print(f"      -> Số nút trong từ điển (vocab_size): {dataset.vocab_size}")
    print(f"      -> Kích thước trạng thái mạng (state_dim): {dataset.state_dim}")
    print(f"      -> Độ dài đường đi tối đa (max_hops): {dataset.max_hops}")

    loader = DataLoader(dataset, batch_size=min(4, len(dataset)), shuffle=True, collate_fn=collate_fn)
    batch = next(iter(loader))

    print("[2/3] Ví dụ 1 batch:")
    for k, v in batch.items():
        print(f"      -> {k:<15}: shape {tuple(v.shape)}")

    print("[3/3] Mẫu đầu tiên trong batch:")
    print(
        f"      source_idx={batch['source_idx'][0].item()}, "
        f"target_idx={batch['target_idx'][0].item()}, "
        f"n_hops={batch['n_hops'][0].item()}"
    )
    print(f"      path (đã pad, -1 = padding)={batch['path'][0].tolist()}")

    print("\nHoàn tất smoke test dataset.py.")


if __name__ == "__main__":
    main()
