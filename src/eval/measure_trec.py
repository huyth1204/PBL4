#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
Giai đoạn 4: Đo thời gian phục hồi hệ thống T_rec bằng StarryNet
====================================================================

Dựa trên đúng cấu trúc API thật của StarryNet (xem example.py gốc).

Kịch bản đo:
    1. Khởi tạo emulation 5x5 vệ tinh + 2 trạm mặt đất (dùng config.json mặc định).
    2. Chạy routing daemon (OSPF) để mạng có đường đi ổn định.
    3. Lấy baseline: đường đi/routing table TRƯỚC khi có sự cố.
    4. Gây damage (ngắt ngẫu nhiên một tỷ lệ liên kết) tại thời điểm DAMAGE_TIME.
    5. Đặt lịch ping liên tục QUA khoảng thời gian trước/sau damage, để log
       ping tự nhiên ghi lại: lúc nào bắt đầu mất gói (do damage), và lúc
       nào bắt đầu nhận gói trở lại (do routing đã hội tụ xong).
    6. Sau khi emulation chạy xong (sn.stop_emulation()), phân tích file log
       ping để tính T_rec = (thời điểm ping thành công trở lại) - (thời điểm damage).

LƯU Ý QUAN TRỌNG:
    - Đây là code cho baseline OSPF/Dijkstra-như-định-tuyến-động (hệ thống tự
      hội tụ lại theo giao thức mạng thật, KHÔNG phải chạy lại thuật toán
      Dijkstra của bạn ở oracle_labeler.py). T_rec đo ở đây là thời gian
      routing protocol thật sự cần để tự phục hồi trên mạng ảo hóa.
    - Để so sánh "Dijkstra baseline" (chạy lại thuật toán) với "model AI"
      (suy luận hằng số), bạn cần đo T_compute riêng biệt: chạy hàm
      find_optimal_path() từ oracle_labeler.py và đo bằng time.time(),
      SONG SONG với việc đo T_rec vật lý ở đây. Hai phép đo này bổ sung
      cho nhau, không thay thế nhau.
    - GIỚI HẠN ĐÃ BIẾT (môi trường WSL2): forwarding gói tin đa chặng
      (multi-hop, qua ≥1 node trung gian) giữa các container hiện KHÔNG
      hoạt động, dù routing table (BIRD/OSPF) hoàn toàn đúng và ip_forward
      đã bật. Đã điều tra: `ovs-vsctl show` trả về rỗng, cho thấy Open
      vSwitch không thực sự quản lý forwarding giữa container dù tên
      container có tiền tố "ovs_container_*" — khả năng cao là vấn đề
      tương thích WSL2/Docker networking, chưa xác định được cách khắc
      phục triệt để. Vì vậy script này dùng cặp node LIỀN KỀ (1-hop),
      nơi đã xác nhận forwarding hoạt động ổn định, làm phương án đo đạc
      khả thi trong lúc chờ điều tra thêm vấn đề multi-hop.

Cách chạy (BẮT BUỘC chạy trong thư mục ~/StarryNet trên Ubuntu, vì cần
import package `starrynet` nằm cùng thư mục):

    cd ~/StarryNet
    cp /mnt/d/PBL4/src/eval/measure_trec.py .
    python3 measure_trec.py
"""

import os
import threading
import time as walltime
from pathlib import Path

from starrynet.sn_observer import *
from starrynet.sn_orchestrater import *
from starrynet.sn_synchronizer import *

# ----------------------------------------------------------------------------
# Cấu hình kịch bản đo (dùng đúng cấu trúc 5x5 vệ tinh + 2 GS mặc định)
# ----------------------------------------------------------------------------
AS = [[1, 27]]  # Node #1 đến #27 cùng 1 AS (25 vệ tinh + 2 trạm mặt đất)
GS_LAT_LONG = [[50.110924, 8.682127], [46.635700, 14.311817]]  # Frankfurt, Austria
CONFIG_PATH = "./config.json"
HELLO_INTERVAL = 1  # giây, khoảng OSPF hello packet

DAMAGE_RATIO = 0.95     # tỷ lệ liên kết bị ngắt ngẫu nhiên — RẤT CAO (95%)
                         # để tối đa hóa xác suất đúng liên kết đang theo dõi
                         # bị ngắt. Do giới hạn hạ tầng hiện tại (multi-hop
                         # forwarding giữa container qua WSL2/OVS chưa hoạt
                         # động — xem ghi chú ở đầu file), dùng cặp node
                         # LIỀN KỀ (1-hop) làm phương án chắc chắn có kết quả
                         # trước deadline, thay vì multi-hop chưa sửa xong.
DAMAGE_TIME = 10        # giây (time_index) khi damage xảy ra
PING_NODE_A = 1
PING_NODE_B = 2          # 2 vệ tinh liền kề (1-hop) — đã xác nhận hoạt động ổn định 100%
PING_START = DAMAGE_TIME - 5   # bắt đầu ping trước damage 5 giây để có baseline
PING_END = DAMAGE_TIME + 40    # ping tới 40 giây sau damage để chắc chắn bắt được lúc hồi phục

# stop_emulation() đôi khi treo vô hạn ở bước dọn dẹp cuối dù công việc dọn
# dẹp thực tế đã xong (đã xác nhận nhiều lần: docker ps -a trống sau vài
# phút, nhưng hàm không tự trả về). Dùng timeout để không phải Ctrl+C thủ
# công mỗi lần chạy.
STOP_EMULATION_TIMEOUT_S = 60


def call_stop_emulation_with_timeout(sn, timeout: int = STOP_EMULATION_TIMEOUT_S):
    """
    Gọi sn.stop_emulation() trong 1 thread riêng với giới hạn thời gian.
    Nếu quá timeout mà vẫn chưa xong, in cảnh báo và cho phép chương trình
    tiếp tục/kết thúc bình thường (dữ liệu container thực tế thường đã
    dọn xong từ trước, chỉ là hàm không tự trả về).
    """
    t = threading.Thread(target=sn.stop_emulation, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        print(f"      !! stop_emulation() vẫn chưa trả về sau {timeout}s.")
        print("         Dữ liệu log (ping/route) đã được ghi ra file trước bước này,")
        print("         nên KHÔNG bị mất. Bỏ qua và kết thúc chương trình luôn.")
        print("         (Kiểm tra thủ công bằng 'docker ps -a' nếu muốn chắc chắn.)")
        return False
    return True


def main():
    print("=" * 70)
    print("Giai đoạn 4: Đo T_rec bằng StarryNet")
    print("=" * 70)

    sn = StarryNet(CONFIG_PATH, GS_LAT_LONG, HELLO_INTERVAL, AS)

    print("\n[1/6] Tạo node...")
    t0 = walltime.time()
    sn.create_nodes()
    print(f"      -> xong sau {walltime.time() - t0:.1f}s (thời gian thật)")

    print("[2/6] Tạo liên kết...")
    t0 = walltime.time()
    sn.create_links()
    print(f"      -> xong sau {walltime.time() - t0:.1f}s (thời gian thật)")

    print("[3/6] Khởi động routing daemon (OSPF)...")
    t0 = walltime.time()
    sn.run_routing_deamon()
    print(f"      -> xong sau {walltime.time() - t0:.1f}s (thời gian thật)")

    print(f"[4/6] Đặt lịch: damage tỷ lệ {DAMAGE_RATIO} tại giây {DAMAGE_TIME}...")
    sn.set_damage(DAMAGE_RATIO, DAMAGE_TIME)

    print(f"      Đặt lịch ping liên tục {PING_NODE_A}<->{PING_NODE_B} "
          f"từ giây {PING_START} đến {PING_END} để bắt trọn quá trình mất/hồi gói...")
    for t in range(PING_START, PING_END):
        sn.set_ping(PING_NODE_A, PING_NODE_B, t)

    print(f"      Ghi routing table của node #{PING_NODE_B} tại vài mốc sau damage "
          f"để đối chiếu thời điểm hội tụ...")
    for t in [DAMAGE_TIME, DAMAGE_TIME + 5, DAMAGE_TIME + 10, DAMAGE_TIME + 20]:
        sn.check_routing_table(PING_NODE_B, t)

    print("\n[5/6] Bắt đầu emulation (sẽ chạy đúng Duration(s) trong config.json)...")
    print("      Đây là thời gian THẬT sẽ trôi qua — không phải mô phỏng nhanh.")
    t_emulation_start = walltime.time()
    sn.start_emulation()
    stopped_cleanly = call_stop_emulation_with_timeout(sn)
    t_emulation_end = walltime.time()
    print(f"      -> Emulation + cleanup mất {t_emulation_end - t_emulation_start:.1f}s (thời gian thật)"
          f"{'' if stopped_cleanly else ' (cleanup bị bỏ qua do timeout, xem cảnh báo ở trên)'}")

    print("\n[6/6] Hoàn tất thu thập dữ liệu.")
    print("      Các file log ping/routing table được lưu tại thư mục làm việc")
    print("      (thường có dạng StarryNet-.../  — hoặc theo tên constellation).")
    print(f"      Tìm file ping log của node #{PING_NODE_A}<->#{PING_NODE_B} "
          f"(dạng ping-{PING_NODE_A}-{PING_NODE_B}_<giây>.txt)")
    print(f"      để xác định thủ công thời điểm bắt đầu mất gói (≈ giây {DAMAGE_TIME}) "
          "và thời điểm ping thành công trở lại (T_rec = hiệu số 2 mốc này).")
    print("\nGợi ý phân tích thủ công:")
    print("  1. Mở file ping log (đường dẫn in trong output của set_ping khi emulation")
    print("     chạy, hoặc tìm bằng: find . -newer config.json -name '*ping*')")
    print("  2. Tìm dòng cuối cùng có 'Destination unreachable' hoặc timeout")
    print("     ngay SAU giây", DAMAGE_TIME, "— đây là lúc damage bắt đầu ảnh hưởng.")
    print("  3. Tìm dòng ĐẦU TIÊN ping thành công trở lại SAU dòng đó.")
    print("  4. T_rec (giây) = (thời điểm ping thành công trở lại) - "
          f"{DAMAGE_TIME} (thời điểm damage).")

    # Force kết thúc tiến trình ngay, tránh bị treo bởi thread nền
    # (stop_emulation chạy daemon=True) nếu nó vẫn chưa trả về.
    print("\nKết thúc chương trình.")
    os._exit(0)


if __name__ == "__main__":
    main()
