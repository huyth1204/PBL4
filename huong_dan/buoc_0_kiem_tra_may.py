"""
Bước 0 — Kiểm tra máy
Kiểm tra phiên bản Python, các thư viện cần có và các tệp dữ liệu, mô hình của dự án.
Chạy trong PyCharm: bấm ▶ xanh cạnh dòng `if __name__ == "__main__":` (hoặc chuột phải vào tệp → Run).
Sẽ thấy: sơ đồ 10 bước của hướng dẫn và bảng kiểm tra xanh / đỏ.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from huong_dan.tien_ich import *  # noqa: E402,F403

TEP_CAN = ["dataset/test.npz", "dataset/oracle_dataset_v2.npz", "data/tle/starlink.txt", "data/thoi_tiet_vn.npz",
           "results/dmx0_4lop.pt", "results/quy_mo/dm_da_quy_mo.pt", "results/quy_mo/dm_nhieu.pt",
           "huong_dan/du_lieu_demo/n373.npz"]
THU_VIEN = [("numpy", "mọi bước"), ("scipy", "mọi bước"), ("torch", "mô hình DM"), ("matplotlib", "vẽ hình"),
            ("plotly", "bản đồ, bước 9"), ("skyfield", "vị trí vệ tinh, bước 9"), ("requests", "một test thời tiết"),
            ("pytest", "chạy test trong PyCharm")]


def chay():
    import importlib
    import platform
    fig, ax = plt.subplots(figsize=(15, 1.9))
    ve_chuoi(["0. Kiểm tra\nmáy", "1. Dữ liệu\nTV1", "2. Chạy\ncác test", "3. DM định\ntuyến", "4. Bốn quy\nmô mạng",
              "5. Hỏng\nliên kết", "6. Số đo\ncó nhiễu", "7. Luật\nchia đôi", "8. Nhiễu\ndo mưa", "9. Bản đồ\ntuyến"],
             ax=ax, mau={0: "#e8f0fb"}, w=1.25)
    ax.set_title("Các bước của hướng dẫn: mỗi bước là một tệp huong_dan/buoc_*.py", loc="left", fontsize=11)
    kq = [kiem_tra("Python ≥ 3.10", lambda: (sys.version_info >= (3, 10), platform.python_version()))]
    for ten, dung in THU_VIEN:
        kq.append(kiem_tra(f"Thư viện {ten}", lambda ten=ten, dung=dung: (
            True, f"phiên bản {importlib.import_module(ten).__version__} — dùng cho {dung}")))
    for f in TEP_CAN:
        kq.append(kiem_tra(f"Có tệp {f}", lambda f=f: ((GOC / f).exists(), "" if (GOC / f).exists()
                                                        else "thiếu: xem mục Lỗi thường gặp trong hướng dẫn")))
    return kq


if __name__ == "__main__":
    hien_thi(chay, "Bước 0 — Kiểm tra máy")
