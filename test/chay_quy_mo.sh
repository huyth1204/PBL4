#!/bin/bash
# DM theo quy mô mạng: sinh dữ liệu 173/273/373 nút bằng bộ gán nhãn của TV1, huấn luyện, đánh giá.
#   bash test/chay_quy_mo.sh          (khoảng 6–7 giờ trên CPU 2 nhân; 373 nút tốn nhất)
set -e
cd "$(dirname "$0")/.."
Q=data/quy_mo; R=results/quy_mo
mkdir -p $Q $R
for S in 150 250 350; do
  N=$((S + 23))
  [ -f $Q/n$N.npz ] || python dataset/oracle_labeler_sagsin.py --out $Q/n$N.npz --sats $S
  [ -f $Q/n$N/test.npz ] || python dataset/split_dataset.py --in $Q/n$N.npz --outdir $Q/n$N
done
python -m src.ai.train_gnn --data-dir $Q/n173 --head diffusion2 --param x0ce --layers 4 --ddim 2 --epochs 12 --tag dm_173 --out-dir $R
python -m src.ai.train_gnn --data-dir $Q/n273 --head diffusion2 --param x0ce --layers 4 --ddim 2 --epochs 12 --tag dm_273 --out-dir $R
python -m src.ai.train_gnn --data-dir $Q/n373 --head diffusion2 --param x0ce --layers 4 --ddim 2 --epochs 8 --eval-every 2 --tag dm_373 --out-dir $R
python -m src.ai.train_gnn --data-dir $Q/n173 --head classifier --layers 4 --epochs 8 --eval-every 2 --tag sm_173 --out-dir $R

# đánh giá (bỏ phần đo thời gian; đo riêng khi máy rảnh ở bước sau)
python test/danh_gia_quy_mo.py --raw dataset/oracle_dataset_v2.npz --data-dir dataset --khong-do-thoi-gian \
  --mo-hinh "DM huấn luyện tại N=results/dmx0_4lop.pt:2:1" "Softmax huấn luyện tại N=results/clsB_73.pt:1:1" --out $R/danh_gia_73.json
python test/danh_gia_quy_mo.py --raw $Q/n173.npz --data-dir $Q/n173 --khong-do-thoi-gian \
  --mo-hinh "DM huấn luyện tại N=$R/dm_173.pt:2:1" "DM 73 nút (chuyển giao)=results/dmx0_4lop.pt:2:1" \
            "Softmax huấn luyện tại N=$R/sm_173.pt:1:1" --out $R/danh_gia_173.json
for N in 273 373; do
  python test/danh_gia_quy_mo.py --raw $Q/n$N.npz --data-dir $Q/n$N --khong-do-thoi-gian --chunk 100 \
    --mo-hinh "DM huấn luyện tại N=$R/dm_$N.pt:2:1" "DM 73 nút (chuyển giao)=results/dmx0_4lop.pt:2:1" --out $R/danh_gia_$N.json
done
python test/do_thoi_gian_quy_mo.py --out $R/thoi_gian.json --cau-hinh \
  "73=dataset:results/dmx0_4lop.pt:1:results/clsB_73.pt" "173=$Q/n173:$R/dm_173.pt:1:$R/sm_173.pt" \
  "273=$Q/n273:$R/dm_273.pt:1" "373=$Q/n373:$R/dm_373.pt:1"
python test/do_huan_luyen_quy_mo.py --dirs dataset $Q/n173 $Q/n273 $Q/n373 --epoch 15 12 12 8 --out $R/huan_luyen.json

# Một mô hình cho mọi quy mô (khởi tạo từ mô hình 73 nút): đủ 4 quy mô, và bản bỏ 373 nút để thử ngoại suy
python -m src.ai.train_da_quy_mo --data-dirs dataset $Q/n173 $Q/n273 $Q/n373 --init results/dmx0_4lop.pt \
  --epochs 12 --lo-moi-quy-mo 100 --tag dm_da_quy_mo --out-dir $R
python -m src.ai.train_da_quy_mo --data-dirs dataset $Q/n173 $Q/n273 --init results/dmx0_4lop.pt \
  --epochs 12 --lo-moi-quy-mo 100 --tag dm_bo_373 --out-dir $R
for M in da_quy_mo bo_373; do
  python test/danh_gia_quy_mo.py --raw dataset/oracle_dataset_v2.npz --data-dir dataset --khong-do-thoi-gian \
    --mo-hinh "DM $M=$R/dm_$M.pt:2:1" --out $R/${M}_73.json
  for N in 173 273 373; do
    python test/danh_gia_quy_mo.py --raw $Q/n$N.npz --data-dir $Q/n$N --khong-do-thoi-gian --chunk 100 \
      --mo-hinh "DM $M=$R/dm_$M.pt:2:1" --out $R/${M}_$N.json
  done
done

# So với Dijkstra khi số đo trễ có nhiễu: học tiếp với nhiễu đo (σ ~ U[0; 0,6]) rồi so với 5 bản Dijkstra (~35 phút)
python -m src.ai.train_da_quy_mo --data-dirs dataset $Q/n173 $Q/n273 $Q/n373 --init $R/dm_da_quy_mo.pt \
  --nhieu 0.6 --lr 3e-4 --epochs 6 --lo-moi-quy-mo 60 --eval-every 1 --threads 2 --tag dm_nhieu --out-dir $R
python test/so_sanh_dijkstra_nhieu.py --cau-hinh "73=dataset:results/dmx0_4lop.pt" "173=$Q/n173:$R/dm_173.pt" \
  "273=$Q/n273:$R/dm_273.pt" "373=$Q/n373:$R/dm_373.pt" \
  --mo-hinh "DM chung nhiều quy mô=$R/dm_da_quy_mo.pt" "DM học với nhiễu=$R/dm_nhieu.pt" --out $R/nhieu_do.json

# Luật chia đôi: DM chỉ học nhiễu cao (để so), quét σ mịn để chọn ngưỡng, rồi thử luật với thời tiết thật (~40 phút)
python -m src.ai.train_da_quy_mo --data-dirs dataset $Q/n173 $Q/n273 $Q/n373 --init $R/dm_da_quy_mo.pt \
  --nhieu 0.6 --nhieu-min 0.15 --lr 3e-4 --epochs 6 --lo-moi-quy-mo 60 --eval-every 1 --threads 2 --tag dm_nhieu_cao --out-dir $R
python test/so_sanh_dijkstra_nhieu.py --cau-hinh "73=dataset" "173=$Q/n173" "273=$Q/n273" "373=$Q/n373" \
  --mo-hinh "DM chung nhiều quy mô=$R/dm_da_quy_mo.pt" "DM học với nhiễu=$R/dm_nhieu.pt" "DM chỉ học nhiễu cao=$R/dm_nhieu_cao.pt" \
  --sigma 0,0.025,0.05,0.075,0.1,0.15,0.2,0.3,0.4,0.6 --out $R/nhieu_do_min.json
python test/luat_chia_doi.py --cau-hinh "73=dataset" "173=$Q/n173" "273=$Q/n273" "373=$Q/n373" \
  --thoi-tiet data/thoi_tiet_vn.npz --thoi-gian $R/thoi_gian.json --mo-hinh "DM học với nhiễu=$R/dm_nhieu.pt" \
  --sigma0 0,0.025,0.05,0.075,0.1,0.15,0.2,0.4 --nguong 0.05,0.075,0.1 --out $R/chia_doi.json

# Mức nhiễu do mưa/giông suy từ vật lý (src/ngau_nhien/nhieu_mua.py), rồi chạy lại luật chia đôi với mức đó (~8 phút)
python test/phan_tich_nhieu_mua.py --data-dir dataset --thoi-tiet data/thoi_tiet_vn.npz --out $R/nhieu_mua.json
python test/luat_chia_doi.py --cau-hinh "73=dataset" "173=$Q/n173" "273=$Q/n273" "373=$Q/n373" \
  --thoi-tiet data/thoi_tiet_vn.npz --thoi-gian $R/thoi_gian.json --mo-hinh "DM học với nhiễu=$R/dm_nhieu.pt" \
  --sigma0 0,0.025,0.05,0.075,0.1,0.15,0.2,0.4 --nguong 0.075,0.1 --mua vat_ly --out $R/chia_doi_vl.json
