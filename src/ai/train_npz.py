import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.ai.dataset_npz import load_items, split_by_snapshot  # noqa: E402
from src.ai.diffusion import build_diffusion  # noqa: E402


def eval_acc(model, ds, device, n=3000, n_steps=20):
    idx = np.random.default_rng(0).choice(len(ds), min(n, len(ds)), replace=False)
    loader = DataLoader(torch.utils.data.Subset(ds, idx), batch_size=512)
    correct = total = 0
    rand_correct = 0  # baseline: chọn ngẫu nhiên trong nút hợp lệ
    for b in loader:
        cond, tgt, mask = b["condition"].to(device), b["target"], b["mask"]
        pred = model.sample(cond.to(device), mask.to(device), n_steps=n_steps).cpu()
        gold = tgt.argmax(-1)
        correct += (pred == gold).sum().item()
        total += len(gold)
        deg = mask.sum(-1).clamp(min=1)
        rand_correct += (1.0 / deg).sum().item()  # kỳ vọng trúng nếu đoán ngẫu nhiên hợp lệ
    return correct / total, rand_correct / total


def main():
    npz = Path("dataset/oracle_dataset.npz")
    if not npz.exists():
        npz = Path("oracle_dataset.npz")
    print(f"Đọc {npz} ...")
    x_norm, items, N, snaps, bad = load_items(npz, strict=True)
    print(f"N={N}, {len(items)} mẫu next-hop, {len(snaps)} snapshot, nhãn lỗi={bad}")

    buckets = split_by_snapshot(x_norm, items, N)
    tr, va = buckets["train"], buckets["val"]
    print(f"train={len(tr)}, val={len(va)}, test={len(buckets['test'])}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(max(1, torch.get_num_threads()))
    model = build_diffusion(N, hidden=256, T=100).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = DataLoader(tr, batch_size=512, shuffle=True, num_workers=0)

    print(f"device={device}, tham số={sum(p.numel() for p in model.parameters()):,}")
    EPOCHS = 25
    for ep in range(EPOCHS):
        model.train()
        t0 = time.time()
        tot = 0.0
        for b in loader:
            cond, target = b["condition"].to(device), b["target"].to(device)
            opt.zero_grad()
            loss = model.training_loss(cond, target)
            loss.backward()
            opt.step()
            tot += loss.item() * len(cond)
        if ep % 5 == 0 or ep == EPOCHS - 1:
            model.eval()
            acc, rand = eval_acc(model, va, device)
            print(f"epoch {ep:2d} | loss {tot/len(tr):.4f} | "
                  f"val next-hop acc {acc*100:5.1f}% | baseline ngẫu nhiên {rand*100:4.1f}% | "
                  f"{time.time()-t0:.1f}s")

    # kết quả cuối trên test
    acc, rand = eval_acc(model, buckets["test"], device)
    print(f"\n[TEST] next-hop acc {acc*100:.1f}%  (baseline ngẫu nhiên {rand*100:.1f}%)")
    torch.save(model.state_dict(), "checkpoints_diffusion.pt")
    print("Đã lưu checkpoints_diffusion.pt")


if __name__ == "__main__":
    main()