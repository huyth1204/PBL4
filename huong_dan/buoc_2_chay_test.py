"""
Bước 2 — Chạy các file test của dự án
File test: test/test_dataset.py (TV1), test/test_gnn.py (18 test, mô hình GNN + DM), test/test_mo_rong.py (12 test,
phần mở rộng: nhiễu đo, Dijkstra khử nhiễu, luật chia đôi, nhiễu mưa), test/test_ngau_nhien.py (10 test, thời tiết thật,
kịch bản ngẫu nhiên). Mất khoảng 30–60 giây.
Trong PyCharm còn một cách đồ họa khác: chuột phải thư mục test → Run 'pytest in test' (cây test xanh / đỏ).
Sẽ thấy: biểu đồ số test đạt theo từng file và bảng chi tiết từng test.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from huong_dan.tien_ich import *  # noqa: E402,F403

CAC_FILE = ["test.test_dataset", "test.test_gnn", "test.test_mo_rong", "test.test_ngau_nhien"]


def chay():
    tat_ca, dem = [], {}
    for mod in CAC_FILE:
        try:
            kq = chay_file_test(mod)
        except Exception as e:  # noqa: BLE001
            kq = [{"ten": f"nạp {mod}", "dat": False, "chi_tiet": f"{type(e).__name__}: {e}", "giay": 0}]
        dem[mod.replace("test.", "") + ".py"] = (sum(k["dat"] for k in kq), len(kq))
        tat_ca += [dict(k, ten=f"{mod.replace('test.', '')}: {k['ten']}") for k in kq]
    fig, ax = plt.subplots(figsize=(10, 2.8))
    ten = list(dem)
    dat = [dem[t][0] for t in ten]
    loi = [dem[t][1] - dem[t][0] for t in ten]
    ax.barh(ten, dat, color=MAU["luc"], label="đạt")
    ax.barh(ten, loi, left=dat, color="#d62728", label="lỗi")
    for y, (a, b) in enumerate(dem.values()):
        ax.text(b + 0.3, y, f"{a}/{b}", va="center", fontsize=10)
    ax.set_xlabel("Số test")
    ax.invert_yaxis()
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("Kết quả các file test", loc="left", fontsize=11)
    fig.tight_layout()
    return tat_ca


if __name__ == "__main__":
    hien_thi(chay, "Bước 2 — Các file test")
