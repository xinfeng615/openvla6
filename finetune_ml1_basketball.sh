#!/bin/bash
# ML1 Basketball 单任务微调脚本
# 使用 OpenVLA-7B 预训练模型，在 Basketball 数据上进行 LoRA 微调

export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES=0

# 创建输出目录
mkdir -p /root/autodl-tmp/openvla/output_ml1
mkdir -p /root/autodl-tmp/openvla/adapter-tmp_ml1

echo "============================================================"
echo "🚀 ML1 Basketball 单任务微调"
echo "============================================================"

# 运行微调命令
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
    --vla_path "/root/autodl-tmp/openvla/openvla-7b" \
    --data_root_dir "/root/autodl-tmp/tensorflow_datasets" \
    --dataset_name metaworld_ml1_50e \
    --run_root_dir "/root/autodl-tmp/openvla/output_ml1" \
    --adapter_tmp_dir "/root/autodl-tmp/openvla/adapter-tmp_ml1" \
    --lora_rank 32 \
    --batch_size 16 \
    --grad_accumulation_steps 1 \
    --learning_rate 5e-4 \
    --image_aug True \
    --max_steps 5000 \
    --save_steps 5000 \
    --save_latest_checkpoint_only True \
    --wandb_project "ml1-basketball" \
    --wandb_entity "1469512941-" \
    2>&1 | tee /root/autodl-tmp/openvla/output_ml1/train_log.txt

echo "============================================================"
echo "🎉 微调完成！"
echo "============================================================"
