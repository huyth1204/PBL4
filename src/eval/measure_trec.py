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
    - GIỚI HẠN ĐÃ BIẾT (môi trường WSL2, ĐÃ XÁC ĐỊNH NGUYÊN NHÂN GỐC):
      forwarding gói tin đa chặng (multi-hop) từng KHÔNG hoạt động dù
      routing table (BIRD/OSPF) "trông" đúng và ip_forward đã bật.

      Nguyên nhân THẬT SỰ (đã xác nhận qua debug thủ công, không phải OVS/
      ip_forward/rp_filter/iptables như nghi ngờ ban đầu):

          sn.run_routing_deamon() sinh ĐÚNG file cấu hình OSPF cho từng
          node tại "/B{i}.conf" bên trong container, nhưng KHÔNG copy nó
          vào "/etc/bird/bird.conf" — nơi bird thực sự đọc khi khởi động.
          Kết quả: bird nạp nhầm file mặc định rỗng (stock Debian, cú
          pháp BIRD 1.x: "import none;" ngay trong "protocol kernel"),
          bird báo lỗi cú pháp và CRASH (defunct/zombie) ngay từ đầu.
          Vì bird chết, kernel routing table chỉ có các subnet kết nối
          trực tiếp — không có route multi-hop nào cả => ping 1-hop vẫn
          sống (không cần forward), multi-hop luôn "Network is unreachable"
          hoặc mất gói.

      Hàm fix_bird_routing() dưới đây khắc phục tận gốc: copy đúng
      "/B{i}.conf" -> "/etc/bird/bird.conf" và khởi động lại bird cho
      TỪNG container theo THỨ TỰ TUẦN TỰ (có nghỉ giữa các node) để
      tránh hiệu ứng "thundering herd" (27 con bird cùng gửi Hello/DBD
      một lúc làm nghẽn CPU của Docker Desktop/WSL2, khiến một số cặp
      OSPF neighbor kẹt mãi ở ExStart/Exchange/Loading do gói bị trễ/rớt
      đúng lúc handshake). Restart tuần tự + đợi hội tụ đủ lâu đã xác
      nhận đưa toàn bộ 27 node về trạng thái Full 100%.

    - Cách chạy phiên bản này BẮT BUỘC vẫn phải chạy trong thư mục
      ~/StarryNet trên Ubuntu (WSL2), vì cần import package `starrynet`
      nằm cùng thư mục:

          cd ~/StarryNet
          cp /mnt/d/PBL4/src/eval/measure_trec.py .
          python3 measure_trec.py
"""

import os
import subprocess
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
PING_NODE_B = 2         # 2 vệ tinh liền kề (1-hop) — đã xác nhận hoạt động ổn định 100%
PING_START = DAMAGE_TIME - 5   # bắt đầu ping trước damage 5 giây để có baseline
PING_END = DAMAGE_TIME + 40    # ping tới 40 giây sau damage để chắc chắn bắt được lúc hồi phục

# stop_emulation() đôi khi treo vô hạn ở bước dọn dẹp cuối dù công việc dọn
# dẹp thực tế đã xong (đã xác nhận nhiều lần: docker ps -a trống sau vài
# phút, nhưng hàm không tự trả về). Dùng timeout để không phải Ctrl+C thủ
# công mỗi lần chạy.
STOP_EMULATION_TIMEOUT_S = 60

# ----------------------------------------------------------------------------
# Cấu hình bản vá bird routing (Giai đoạn 4 - fix)
# ----------------------------------------------------------------------------
N_NODES = 27                    # tổng số container (25 vệ tinh + 2 trạm mặt đất)
BIRD_RESTART_STAGGER_S = 3       # nghỉ giữa mỗi node khi restart tuần tự
BIRD_CONVERGENCE_WAIT_S = 90     # đợi OSPF hội tụ sau khi restart toàn bộ


def fix_bird_routing(n_nodes: int = N_NODES,
                      stagger_s: int = BIRD_RESTART_STAGGER_S,
                      wait_after_s: int = BIRD_CONVERGENCE_WAIT_S) -> None:
    """
    Khắc phục lỗi gốc: bird trong mỗi container ovs_container_{i} đọc nhầm
    file cấu hình mặc định (/etc/bird/bird.conf rỗng, cú pháp BIRD 1.x) thay
    vì file OSPF thật mà StarryNet đã sinh ra tại /B{i}.conf, khiến bird
    crash và routing table chỉ có route kết nối trực tiếp (không multi-hop).

    Với MỖI container:
        1. Dọn route "rác" mà bird cũ (nếu từng chạy) đã đẩy vào kernel.
        2. Kill sạch tiến trình bird cũ (kể cả zombie/defunct).
        3. Xoá control socket cũ (nếu còn) để tránh "Connection refused"
           hoặc "address already in use" khi bird mới khởi động.
        4. Copy ĐÚNG file cấu hình /B{i}.conf -> /etc/bird/bird.conf.
        5. Khởi động lại bird với đúng config.

    QUAN TRỌNG: các bước trên chạy TUẦN TỰ với độ trễ `stagger_s` giữa mỗi
    node, KHÔNG chạy đồng loạt. Restart 27 con bird cùng lúc ("thundering
    herd") đã được xác nhận gây nghẽn CPU tạm thời trên Docker Desktop/WSL2,
    khiến một số cặp OSPF neighbor bị lệch nhịp handshake và kẹt mãi ở
    ExStart/Exchange/Loading dù cấu hình hoàn toàn đúng.

    Sau khi restart xong toàn bộ, hàm đợi thêm `wait_after_s` giây để OSPF
    có đủ thời gian bầu DR/BDR, trao đổi LSA và hội tụ SPF trước khi bất kỳ
    bước nào khác (damage, ping, đo T_rec) được thực hiện.
    """
    print("=" * 70)
    print("[Fix] Khắc phục lỗi bird đọc sai file cấu hình OSPF")
    print("=" * 70)

    for i in range(1, n_nodes + 1):
        container = f"ovs_container_{i}"
        cmd = (
            f"docker exec {container} sh -c "
            f"'ip route flush proto bird 2>/dev/null; "
            f"pkill -9 bird 2>/dev/null; "
            f"rm -f /usr/local/var/run/bird.ctl; "
            f"cp /B{i}.conf /etc/bird/bird.conf; "
            f"bird -c /etc/bird/bird.conf'"
        )
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        status = "OK" if result.returncode == 0 else f"LỖI (code={result.returncode})"
        print(f"      [{i:2d}/{n_nodes}] {container}: {status}")
        if result.returncode != 0 and result.stderr.strip():
            print(f"           stderr: {result.stderr.strip()}")

        walltime.sleep(stagger_s)  # tránh thundering herd

    print(f"\n      Đợi {wait_after_s}s để OSPF hội tụ (bầu DR/BDR, flood LSA, tính SPF) ...")
    walltime.sleep(wait_after_s)

    # Xác minh nhanh: đếm trạng thái neighbor trên toàn bộ node.
    print("      Kiểm tra nhanh trạng thái OSPF neighbor toàn mạng ...")
    full_count = 0
    other_count = 0
    for i in range(1, n_nodes + 1):
        container = f"ovs_container_{i}"
        cmd = f"docker exec {container} birdc show ospf neighbors 2>/dev/null"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        for line in result.stdout.splitlines()[2:]:  # bỏ 2 dòng header
            parts = line.split()
            if len(parts) < 3:
                continue
            state = parts[2].split("/")[0]
            if state == "Full":
                full_count += 1
            else:
                other_count += 1

    print(f"      -> {full_count} adjacency Full, {other_count} chưa hội tụ hết.")
    if other_count > 0:
        print("      !! Vẫn còn adjacency chưa Full. Có thể cần đợi thêm hoặc")
        print("         chạy lại fix_bird_routing() một lần nữa trước khi tiếp tục.")
    print("[Fix] Hoàn tất.\n")


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

    print("\n[1/7] Tạo node...")
    t0 = walltime.time()
    sn.create_nodes()
    print(f"      -> xong sau {walltime.time() - t0:.1f}s (thời gian thật)")

    print("[2/7] Tạo liên kết...")
    t0 = walltime.time()
    sn.create_links()
    print(f"      -> xong sau {walltime.time() - t0:.1f}s (thời gian thật)")

    print("[3/7] Khởi động routing daemon (OSPF)...")
    t0 = walltime.time()
    sn.run_routing_deamon()
    print(f"      -> xong sau {walltime.time() - t0:.1f}s (thời gian thật)")

    print("[4/7] Áp bản vá bird routing (xem ghi chú đầu file để biết nguyên nhân) ...")
    t0 = walltime.time()
    fix_bird_routing()
    print(f"      -> xong sau {walltime.time() - t0:.1f}s (thời gian thật, bao gồm thời gian đợi hội tụ)")

    print(f"[5/7] Đặt lịch: damage tỷ lệ {DAMAGE_RATIO} tại giây {DAMAGE_TIME}...")
    sn.set_damage(DAMAGE_RATIO, DAMAGE_TIME)

    print(f"      Đặt lịch ping liên tục {PING_NODE_A}<->{PING_NODE_B} "
          f"từ giây {PING_START} đến {PING_END} để bắt trọn quá trình mất/hồi gói...")
    for t in range(PING_START, PING_END):
        sn.set_ping(PING_NODE_A, PING_NODE_B, t)

    print(f"      Ghi routing table của node #{PING_NODE_B} tại vài mốc sau damage "
          f"để đối chiếu thời điểm hội tụ...")
    for t in [DAMAGE_TIME, DAMAGE_TIME + 5, DAMAGE_TIME + 10, DAMAGE_TIME + 20]:
        sn.check_routing_table(PING_NODE_B, t)

    print("\n[6/7] Bắt đầu emulation (sẽ chạy đúng Duration(s) trong config.json)...")
    print("      Đây là thời gian THẬT sẽ trôi qua — không phải mô phỏng nhanh.")
    t_emulation_start = walltime.time()
    sn.start_emulation()
    stopped_cleanly = call_stop_emulation_with_timeout(sn)
    t_emulation_end = walltime.time()
    print(f"      -> Emulation + cleanup mất {t_emulation_end - t_emulation_start:.1f}s (thời gian thật)"
          f"{'' if stopped_cleanly else ' (cleanup bị bỏ qua do timeout, xem cảnh báo ở trên)'}")

    print("\n[7/7] Hoàn tất thu thập dữ liệu.")
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