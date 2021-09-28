CUDA_VISIBLE_DEVICES=0 python -u main.py \
    --mode train \
    --debug 1 \
    --dataset ace05 \
    --conf conf/ace05/config.json \
    --use_cpu 1