# CUDA_VISIBLE_DEVICES=0 python -u main.py \
#     --mode train \
#     --debug 0 \
#     --dataset rams \
#     --conf conf/rams/config.json \
#     --use_cpu 0 \
#     --pretrain 1 \
#     --epochs 20 \
#     --accumulate_step 1 \
#     --batch_size 16

CUDA_VISIBLE_DEVICES=0 python -u main.py \
    --mode train \
    --debug 0 \
    --dataset rams \
    --conf conf/rams/config.json \
    --use_cpu 0 \
    --pretrain 0 \
    --epochs 20 \
    --accumulate_step 8 \
    --batch_size 1
