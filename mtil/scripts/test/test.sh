##!/bin/bash

set -v
set -e
set -x
GPU=0
dataset=(Aircraft Caltech101 CIFAR100 DTD EuroSAT Flowers Food MNIST OxfordPet StanfordCars SUN397)
exp_no=your_exp
for ((i = 0; i < ${#dataset[@]}; i++)); do
    lora_paths_str=""
    for ((k = 0; k <= i; k++)); do
        if [ -n "$lora_paths_str" ]; then
            lora_paths_str="${lora_paths_str},"
        fi
        lora_paths_str="${lora_paths_str}ckpt/fewshotlora/${exp_no}/${dataset[k]}.pt"
    done
    eval_datasets=$(IFS=,; echo "${dataset[*]}")
    CUDA_VISIBLE_DEVICES=${GPU} python -m src.main --eval-only \
        --train-mode="lora" \
        --resfile=/your_result/result/${exp_no}.csv \
        --eval-datasets=${eval_datasets} \
        --load ${lora_paths_str}\
        --data-location your_location \
        --params q k v o \
        --at 1 \
        --r 12 \
        --linear 1 \
        --adaw 1 \
        --lora-paths "$lora_paths_str"
    done
done
