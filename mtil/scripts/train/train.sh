##!/bin/bash

set -v
set -e
set -x
exp_no=yourexp
GPU=1
dataset=(Aircraft Caltech101 CIFAR100 DTD EuroSAT Flowers Food MNIST OxfordPet StanfordCars SUN397)
lr=(2e-3 2e-3 2e-3 2e-3 2e-3 2e-3 2e-3 2e-3 2e-3 2e-3 2e-3)

#dataset=(StanfordCars Food MNIST OxfordPet Flowers SUN397 Aircraft Caltech101 DTD EuroSAT CIFAR100)
#lr=(2e-3 2e-3 1e-3 2e-3 2e-3 2e-3 3e-3 2e-3 2e-3 1e-3 3e-3)


data_loc=/your_loc



j=0
CUDA_VISIBLE_DEVICES=${GPU} python -m src.main \
    --train-mode=lora \
    --train-dataset=${dataset[j]} \
    --lr=${lr[j]} \
    --iterations 500 \
    --method finetune \
    --params q k v o\
    --at 1 \
    --r 12 \
    --linear 1 \
    --adaw 1 \
    --save ckpt/fewshotlora/exp_${exp_no} \
    --data-location ${data_loc} \
    --batch-size 32 \
    --few_shot=5 



for ((i = 1; i < 11; i++)); do
#for ((i = 2; i < 10; i++)); do
    dataset_cur=${dataset[i]}
        lora_paths_str=""
        for ((k = 0; k < i; k++)); do
            if [ -n "$lora_paths_str" ]; then
                lora_paths_str="${lora_paths_str},"
            fi
            lora_paths_str="${lora_paths_str}ckpt/fewshotlora/exp_${exp_no}/${dataset[k]}.pt"
        done
    CUDA_VISIBLE_DEVICES=${GPU} python -m src.main \
        --batch-size 32 \
        --train-mode=lora \
        --train-dataset=${dataset_cur} \
        --lr=${lr[i]} \
        --r 12 \
        --method finetune \
        --params q k v o \
        --at 1 \
        --linear 1 \
        --adaw 1 \
        --iterations 500 \
        --save ckpt/fewshotlora/exp_${exp_no} \
        --load ckpt/fewshotlora/exp_${exp_no}/${dataset_pre}.pt \
        --data-location ${data_loc} \
        --lora-paths "$lora_paths_str" \
        --few_shot=5 
done

model_ckpt_path=your_weights.pt
for ((i = 0; i < ${#dataset[@]}; i++)); do
    lora_paths_str=""
    for ((k = 0; k <= i; k++)); do
        if [ -n "$lora_paths_str" ]; then
            lora_paths_str="${lora_paths_str},"
        fi
        lora_paths_str="${lora_paths_str}ckpt/fewshotlora/exp_${exp_no}/${dataset[k]}.pt"
    done
    eval_datasets=$(IFS=,; echo "${dataset[*]}")
    CUDA_VISIBLE_DEVICES=${GPU} python -m src.main --eval-only \
        --train-mode="lora" \
        --resfile=/your_resfile/${exp_no}.csv \
        --eval-datasets=${eval_datasets} \
        --load ${model_ckpt_path}\
        --data-location ${data_loc}  \
        --params q k v o \
        --at 1 \
        --r 12 \
        --linear 1 \
        --adaw 1 \
        --lora-paths "$lora_paths_str"
        
    done
done
