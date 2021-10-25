CUDA_VISIBLE_DEVICES=0,1 python -u main.py \
    --mode train \
    --debug 0 \
    --dataset rams \
    --conf conf/rams/config.json \
    --use_cpu 0 \
    --epochs 50 \
    --train_bert 0 \
    --lr 1e-3

# CUDA_VISIBLE_DEVICES=0,1 python -u main.py \
#     --mode train \
#     --debug 0 \
#     --dataset rams \
#     --conf conf/rams/config.json \
#     --use_cpu 0 \
#     --checkpoint cache/rams/best.json \
#     --epochs 50 \
#     --lr 1e-5 \
#     --train_bert 1