#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export MUJOCO_GL=egl

python experiments/robot/metaworld/run_m6_zeroshot_eval.py \
  --model_family openvla \
  --pretrained_checkpoint ./openvla-7b \
  --task_suite_name metaworld_m6_50e \
  --num_trials_per_task 10 \
  --max_steps 500 \
  --action_scale 1.0 \
  --save_gifs True \
  --gif_episodes_per_task 1
