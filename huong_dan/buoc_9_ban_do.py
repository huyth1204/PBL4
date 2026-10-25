"""
Bước 9 — Bản đồ tuyến trên lãnh thổ Việt Nam
File dự án: visualization.py (bản đồ mạng của nhóm) và truc_quan_dm.py (viết tiếp: thêm tuyến DM, số đo nhiễu, vùng mưa).
Vị trí vệ tinh tính lại từ TLE (data/tle/starlink.txt), vị trí HAPS/UAV/trạm từ src/physics — đúng thời điểm của snapshot.
Sẽ thấy: ba bản đồ mở trong trình duyệt (kéo để di chuyển, cuộn để phóng to, rê chuột để xem tên nút):
  1. dữ liệu sạch: tuyến DM trùng tuyến tối ưu;
  2. số đo nhiễu σ = 40%: Dijkstra bị đánh lừa, DM vẫn đúng;
  3. một giờ mưa thật: liên kết có mưa, luật chia đôi chuyển sang DM.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from huong_dan.tien_ich import *  # noqa: E402,F403

CHE_DO = [("sach", []), ("nhieu", ["--sigma", "0.4"]), ("mua", [])]


def chay(mo_trinh_duyet=True):
    import json

    import truc_quan_dm
    kq = []
    for ten, them in CHE_DO:
        def mot(ten=ten, them=them):
            cu = sys.argv
            sys.argv = ["truc_quan_dm.py", "--che-do", ten] + them + ([] if mo_trinh_duyet else ["--khong-mo"])
            try:
                ra = truc_quan_dm.main()
            finally:
                sys.argv = cu
            return ra.exists(), json.load(open(ra.with_suffix(".json"), encoding="utf-8"))["tieu_de"]
        kq.append(kiem_tra(f"Bản đồ chế độ '{ten}'", mot))
    return kq


if __name__ == "__main__":
    hien_thi(chay, "Bước 9 — Bản đồ tuyến")
