#!/bin/bash
# Chạy toàn bộ hướng "mưa + giông + xung đột + tải" từ đầu tới cuối.
#   bash src/ngau_nhien/chay_tat_ca.sh            # thời tiết thật (cần Internet để tải Open-Meteo lần đầu)
#   bash src/ngau_nhien/chay_tat_ca.sh gia-lap    # thời tiết giả lập (không cần Internet)
set -e
cd "$(dirname "$0")/../.."
RAW=dataset/oracle_dataset_v2.npz
SPLIT=dataset/split_info.json
OUT=results/ngau_nhien
mkdir -p data "$OUT"

if [ "$1" == "gia-lap" ]; then
  TT=data/thoi_tiet_gia_lap.npz; THANG=""
  [ -f $TT ] || python -m src.ngau_nhien.thoi_tiet gia-lap --out $TT
else
  TT=data/thoi_tiet_vn.npz; THANG="--thang 5-10"
  if [ ! -f $TT ]; then
    python -m src.ngau_nhien.thoi_tiet tai --start 2024-01-01 --end 2025-12-31 --out $TT
    python -m src.ngau_nhien.thoi_tiet them-giong --vao $TT          # ERA5 không có giông: lấy CAPE
  fi
fi
python -m src.ngau_nhien.thoi_tiet xem $TT

# 1) kiểm thử
python test/test_ngau_nhien.py $RAW

# 2) khoảng lợi trước khi huấn luyện (~13 phút mỗi lệnh với 2 nhân)
python -m src.ngau_nhien.khoang_loi --raw $RAW --split $SPLIT --thoi-tiet $TT --so-kich-ban 24 \
  --muc "thường,khắc nghiệt" --deltas 1,3,9,45,90 --tuoi 0,9 $THANG --out $OUT/khoang_loi.json
[ -n "$THANG" ] && python -m src.ngau_nhien.khoang_loi --raw $RAW --split $SPLIT --thoi-tiet $TT --so-kich-ban 24 \
  --muc "thường,khắc nghiệt" --deltas 1,3,9,45,90 --tuoi 0,9 --out $OUT/khoang_loi_ca_nam.json

# 3) hai bộ dữ liệu (train, val có cửa sổ tương lai, test 10 kịch bản)
#   E1: mức thường, thông tin mới, giữ tuyến 3 phút, mô hình chỉ thấy W đo được
#   E2: mức khắc nghiệt, thông tin cũ 3 phút, giữ tuyến 1 phút, mô hình thấy W dự báo từ lịch quỹ đạo
python -m src.ngau_nhien.tao_du_lieu --raw $RAW --split $SPLIT --thoi-tiet $TT --delta 9 --tuoi 0 \
  --dau-vao quan_sat --muc "thường" --kich-ban 2,1,10 --seed 11 $THANG --luu-val --out-dir data/ngau_nhien_E1
python -m src.ngau_nhien.tao_du_lieu --raw $RAW --split $SPLIT --thoi-tiet $TT --delta 3 --tuoi 9 \
  --dau-vao quy_dao --muc "khắc nghiệt" --kich-ban 3,1,10 --seed 12 $THANG --luu-val --out-dir data/ngau_nhien_E2

# 4) huấn luyện DM (y0 + cross-entropy) và softmax đối chứng trên nhãn hậu kiểm
for E in E1 E2; do
  python -m src.ai.train_gnn --data-dir data/ngau_nhien_$E --head diffusion2 --param x0ce --layers 4 --ddim 2 \
    --epochs 12 --tag ${E}_dm --out-dir $OUT
  python -m src.ai.train_gnn --data-dir data/ngau_nhien_$E --head classifier --layers 4 --epochs 12 \
    --tag ${E}_sm --out-dir $OUT
done

# 5) chọn guidance của DM theo chi phí thực trên val, rồi chấm trên test (kèm mô hình cũ nếu có)
CU=""
[ -f results/dmx0_4lop.pt ] && CU="$CU DM-cũ-nhãn-tất-định=results/dmx0_4lop.pt:2:1"
[ -f results/clsB_73.pt ] && CU="$CU Softmax-cũ-nhãn-tất-định=results/clsB_73.pt:1:1"
for E in E1 E2; do
  python -m src.ngau_nhien.danh_gia_mo_hinh --test data/ngau_nhien_$E/val.npz --out $OUT/chon_w_$E.json \
    --mo-hinh "w1=$OUT/${E}_dm.pt:2:1" "w2=$OUT/${E}_dm.pt:2:2" "w3=$OUT/${E}_dm.pt:2:3"
  W=$(python -c "import json; d=json.load(open('$OUT/chon_w_$E.json')); print(min(('w1','w2','w3'), key=lambda k: d[k]['G_%'])[1])")
  echo "$E: guidance chọn trên val = $W"
  python -m src.ngau_nhien.danh_gia_mo_hinh --test data/ngau_nhien_$E/test.npz --out $OUT/danh_gia_$E.json \
    --mo-hinh "DM-mới-nhãn-hậu-kiểm=$OUT/${E}_dm.pt:2:$W" "Softmax-mới-nhãn-hậu-kiểm=$OUT/${E}_sm.pt:1:1" $CU
done
