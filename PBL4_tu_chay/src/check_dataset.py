import numpy as np

for name in ["oracle_dataset", "train", "val", "test"]:
    d = np.load(f"dataset/{name}.npz")
    ll = d["node_latlon_alt"]
    print(f"\n== {name} ==")
    print("shape      :", ll.shape)                 # oracle: (600, 53, 3)
    print("có NaN     :", np.isnan(ll).any())       # phải là False
    print("lat trong [-90, 90]  :", ll[..., 0].min().round(2), ll[..., 0].max().round(2))
    print("lon trong [-180, 180]:", ll[..., 1].min().round(2), ll[..., 1].max().round(2))
    print("alt (km)   :", ll[..., 2].min().round(1), ll[..., 2].max().round(1))

# Kiểm tra 3 trạm mặt đất trong file gốc
d = np.load("dataset/oracle_dataset.npz")
order = list(d["node_order"])
for gs in ["Hanoi", "DaNang", "HoChiMinh"]:
    print(gs, d["node_latlon_alt"][0, order.index(gs)])