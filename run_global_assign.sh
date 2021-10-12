CUDA_VISIBLE_DEVICES=0,1 python -u main.py \
    --mode train \
    --debug 0 \
    --dataset ace05 \
    --conf conf/ace05/config.json \
    --use_cpu 0