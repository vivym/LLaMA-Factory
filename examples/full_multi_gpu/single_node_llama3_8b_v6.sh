#!/bin/bash

# https://wandb.ai/viv/huggingface/runs/54og1z8a

deepspeed --num_gpus 8 ../../src/train.py \
    --deepspeed ../deepspeed/ds_z3_config.json \
    --stage sft \
    --do_train \
    --model_name_or_path meta-llama/Meta-Llama-3-8B \
    --dataset roleplay-sft-v1 \
    --dataset_dir ../../data \
    --template llama3 \
    --finetuning_type full \
    --output_dir ../../saves/RoleLlama3-8B/full_sft/v6 \
    --overwrite_output_dir \
    --cutoff_len 2048 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --warmup_steps 100 \
    --save_steps 200 \
    --eval_steps 200 \
    --evaluation_strategy steps \
    --learning_rate 1e-5 \
    --num_train_epochs 2.2 \
    --max_samples 990160 \
    --val_size 0.05 \
    --ddp_timeout 180000000 \
    --plot_loss \
    --bf16 \
    --flash_attn "fa2" \
    --report_to wandb \
    --packing True
