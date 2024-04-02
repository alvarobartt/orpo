#!/bin/bash
ACCELERATE_LOG_LEVEL=INFO accelerate launch --config_file ./src/accelerate/fsdp.yaml main.py \
    --lr 5e-6 \
    --torch_compile False \
    --alpha 0.05 \
    --lr_scheduler_type inverse_sqrt \
    --warmup_steps 100 \
    --model_name Qwen/Qwen1.5-0.5B \
    --data_name argilla/distilabel-capybara-dpo-7k-binarized \
    --num_train_epochs 3 \
    --optim adamw_bnb_8bit \
    --gradient_accumulation_steps 1 \
    --prompt_max_length 1792 \
    --response_max_length 2048 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --num_proc 12 \
    --seed 42 \
    --flash_attention_2