"""
run_ml1_basketball_eval.py

在 Meta-World Basketball 仿真环境中运行模型评估。

用法:
    python experiments/robot/metaworld/run_ml1_basketball_eval.py \
        --model_family openvla \
        --pretrained_checkpoint <CHECKPOINT_PATH> \
        --task_name basketball-v3 \
        --center_crop [ True | False ] \
        --run_id_note <OPTIONAL TAG TO INSERT INTO RUN ID FOR LOGGING> \
        --use_wandb [ True | False ] \
        --wandb_project <PROJECT> \
        --wandb_entity <ENTITY>
"""

import os
import sys
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union
from tqdm import tqdm
import draccus
import numpy as np
import wandb

# 追加当前目录到系统路径，以便解释器能找到 experiments.robot 模块
sys.path.append("../..")
from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
from experiments.robot.metaworld.metaworld_utils import resize_image
from experiments.robot.metaworld.metaworld_env import MetaworldEnv


@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # 模型特定参数
    #################################################################################################################
    model_family: str = "openvla"                    # 模型系列
    pretrained_checkpoint: Union[str, Path] = ""     # 预训练模型权重路径
    load_in_8bit: bool = False                       # (仅限 OpenVLA) 使用 8-bit 量化加载
    load_in_4bit: bool = False                       # (仅限 OpenVLA) 使用 4-bit 量化加载

    center_crop: bool = True                         # 是否中心裁剪？(如果训练时使用了随机裁剪图像增强，需设为 True)

    #################################################################################################################
    # 仿真环境特定参数
    #################################################################################################################
    task_name: str = "basketball-v3"                 # ML1 单任务名称
    task_suite_name: str = "metaworld_ml1_50e"      # 任务套件名称
    num_steps_wait: int = 10                         # 等待物体在仿真器中物理稳定的步数
    num_trials_per_task: int = 50                    # 每个任务测试的回合数 (Rollouts)

    #################################################################################################################
    # 实用工具参数
    #################################################################################################################
    run_id_note: Optional[str] = None                # 添加到运行 ID 中的额外注释，方便日志区分
    local_log_dir: str = "./experiments/logs"        # 评估日志的本地保存目录

    use_wandb: bool = True                          # 是否将测试结果同步记录到 Weights & Biases (W&B)
    wandb_project: str = "ml1-basketball-eval"      # 记录到的 W&B 项目名称
    wandb_entity: str = "1469512941-"                # 记录所属的 W&B 实体/团队名称

    seed: int = 7                                   # 随机种子（用于保证测试结果的可复现性）

    # fmt: on


@draccus.wrap()
def eval_ml1_basketball(cfg: GenerateConfig) -> None:
    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint (模型路径) 不能为空!"
    if "image_aug" in cfg.pretrained_checkpoint:
        assert cfg.center_crop, "期望 `center_crop==True`，因为该模型是在开启图像增强的情况下训练的!"
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "不能同时使用 8-bit 和 4-bit 量化加载!"

    # 设置全局随机种子
    set_seed_everywhere(cfg.seed)

    # [OpenVLA] 设置动作反归一化的键名
    cfg.unnorm_key = cfg.task_suite_name

    # 加载模型
    model = get_model(cfg)

    # [OpenVLA] 检查模型内部的 `norm_stats` 是否包含该动作反归一化键名
    if cfg.model_family == "openvla":
        if cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
            cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
        assert cfg.unnorm_key in model.norm_stats, f"在 VLA 模型的 `norm_stats` 中未找到动作反归一化键名 {cfg.unnorm_key}!"

    # [OpenVLA] 获取 Hugging Face 图像与文本处理器 (Processor)
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)

    # 初始化本地日志系统
    run_id = f"EVAL-{cfg.task_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    print(f"日志将记录到本地文件: {local_log_filepath}")

    # 同时初始化 Weights & Biases 在线日志记录
    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
            mode="online"
        )

    # 获取模型期望的图像输入尺寸
    resize_size = get_image_resize_size(cfg)
    
    print(f"{'='*60}")
    print(f"🎯 开始评估 ML1 Basketball 单任务")
    print(f"   任务名称: {cfg.task_name}")
    print(f"   测试回合数: {cfg.num_trials_per_task}")
    print(f"{'='*60}")
    log_file.write(f"评估任务: {cfg.task_name}\n")
    
    # 创建 Basketball 环境
    env = MetaworldEnv(cfg.task_name)

    # 运行测试回合 (rollouts)
    total_return = 0
    total_accuracy = 0
    episode_lengths = []
    
    for i in tqdm(range(cfg.num_trials_per_task), desc="评估进度"):
        obs, info = env.reset()
        images = []
        episode_return = 0.0
        
        for j in range(500):  # 每回合最多执行 500 步
            # 构造符合 OpenVLA 要求的观测字典
            observation = {
                "full_image": resize_image(
                    copy.deepcopy(obs["image_primary"]), resize_size=(resize_size, resize_size),
                ),
                "state": np.concatenate(
                    (obs["proprio"][:3], np.zeros(shape=(3,)), obs["proprio"][3:4])
                ),
            }
            # 提取自然语言任务指令
            task_description = env.get_task()["language_instruction"][0]
            images.append(observation["full_image"])
            
            # 获取模型输出的动作指令
            action = get_action(
                cfg,
                model,
                observation,
                task_description,
                processor=processor,
            )
            # 将夹爪动作从 [0,1] 归一化为 [-1,+1]
            action = normalize_gripper_action(action, binarize=True)

            # [OpenVLA] 反转夹爪动作
            if cfg.model_family == "openvla":
                action = invert_gripper_action(action)
            
            action = np.concatenate([action[:3], action[-1:]])
            
            # 环境步进，执行动作
            obs, reward, done, trunc, info = env.step(action)

            episode_return += reward
            if done or trunc:
                break
        
        episode_lengths.append(j + 1)
        total_return += episode_return
        total_accuracy += int(done)
            
        # 每 5 个回合上传一次测试视频到 WandB
        if i % 5 == 0:
            wandb.log({"rollout_video": wandb.Video(np.array(images).transpose(0, 3, 1, 2)[::10])})

    # 计算统计数据
    avg_return = total_return / cfg.num_trials_per_task
    avg_accuracy = total_accuracy / cfg.num_trials_per_task
    avg_episode_length = np.mean(episode_lengths)
    std_episode_length = np.std(episode_lengths)
    
    # 打印并记录评估结果
    result_msg = (
        f"\n{'='*60}\n"
        f"📊 ML1 Basketball 评估结果\n"
        f"{'='*60}\n"
        f"   平均回报 (Average Return): {avg_return:.4f}\n"
        f"   平均成功率 (Average Accuracy): {avg_accuracy:.2%}\n"
        f"   平均回合长度: {avg_episode_length:.1f} ± {std_episode_length:.1f} 步\n"
        f"   成功回合数: {total_accuracy}/{cfg.num_trials_per_task}\n"
        f"{'='*60}"
    )
    
    print(result_msg)
    log_file.write(result_msg)
    log_file.close()
    
    # 上传汇总结果到 WandB
    wandb.log({
        "average_return": avg_return,
        "average_accuracy": avg_accuracy,
        "average_episode_length": avg_episode_length,
        "std_episode_length": std_episode_length,
        "successful_episodes": total_accuracy,
        "total_episodes": cfg.num_trials_per_task,
    })
    
    wandb.finish()


if __name__ == "__main__":
    eval_ml1_basketball()
