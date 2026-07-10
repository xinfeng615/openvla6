#!/bin/bash
# ML1 Basketball 单任务评估脚本

export CUDA_VISIBLE_DEVICES=0
export MUJOCO_GL=egl

echo "============================================================"
echo "🎯 ML1 Basketball 单任务评估"
echo "============================================================"

python experiments/robot/metaworld/run_ml1_basketball_eval.py \
  --model_family openvla \
  --pretrained_checkpoint /root/autodl-tmp/openvla/output_ml1/openvla-7b+metaworld_ml1_50e+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug \
  --task_name basketball-v3 \
  --task_suite_name metaworld_ml1_50e \
  --center_crop True \
  --num_trials_per_task 50 \
  --use_wandb True \
  --wandb_project "ml1-basketball-eval" \
  --wandb_entity "1469512941-"

echo "============================================================"
echo "🎉 评估完成！"
echo "============================================================"
