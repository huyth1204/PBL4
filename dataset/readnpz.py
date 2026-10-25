import numpy as np

d = np.load("dataset/oracle_dataset.npz", allow_pickle=False)

print("Các trường có trong file:", d.files)
print()
for key in d.files:
    arr = d[key]
    print(f"{key:20s} shape={arr.shape}  dtype={arr.dtype}")

print()
print("Số mẫu (samples):", len(d["x_weights"]))
print("Số snapshot khác nhau:", len(np.unique(d["snapshot_id"])))
print("Số node (N):", len(d["node_order"]))

# Kiểm tra 1 mẫu cụ thể (mẫu đầu tiên)
print("\n--- Mẫu #0 ---")
print("Node order (5 đầu):", d["node_order"][:5])
print("y_paths_text[0]:", d["y_paths_text"][0])
print("x_weights[0] shape:", d["x_weights"][0].shape)
print("x_adjacency[0] shape:", d["x_adjacency"][0].shape)
print("x_positions[0] shape:", d["x_positions"][0].shape)
print("snapshot_id[0]:", d["snapshot_id"][0])