#!/bin/bash

export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES=0

mkdir -p /root/autodl-tmp/openvla/output_m6
mkdir -p /root/autodl-tmp/openvla/adapter-tmp_m6

torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path "/root/autodl-tmp/openvla/openvla-7b" \
  --data_root_dir "/root/autodl-tmp/tensorflow_datasets" \
  --dataset_name metaworld_m6_50e \
  --run_root_dir "/root/autodl-tmp/openvla/output_m6" \
  --adapter_tmp_dir "/root/autodl-tmp/openvla/adapter-tmp_m6" \
  --lora_rank 32 \
  --batch_size 16 \
  --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --image_aug True \
  --max_steps 20000 \
  --save_steps 2000 \
  --save_latest_checkpoint_only False \
  --wandb_project "m6-finetune" \
  --wandb_entity "1469512941-" \
  2>&1 | tee /root/autodl-tmp/openvla/output_m6/train_log.txt
