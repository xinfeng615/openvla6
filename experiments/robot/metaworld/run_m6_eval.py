"""Evaluate a fine-tuned OpenVLA checkpoint on the six-task MetaWorld M6 suite."""

import copy
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import draccus
import numpy as np
import wandb
from tqdm import tqdm

sys.path.append("../..")

from experiments.robot.metaworld.m6_tasks import M6_DATASET_NAME, M6_TASK_NAMES, task_instruction
from experiments.robot.metaworld.metaworld_env import MetaworldEnv
from experiments.robot.metaworld.metaworld_utils import resize_image
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


@dataclass
class GenerateConfig:
    model_family: str = "openvla"
    pretrained_checkpoint: Union[str, Path] = ""
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    center_crop: bool = True

    task_suite_name: str = M6_DATASET_NAME
    num_steps_wait: int = 10
    num_trials_per_task: int = 50
    max_steps: int = 500

    run_id_note: Optional[str] = None
    local_log_dir: str = "./experiments/logs"
    use_wandb: bool = True
    wandb_project: str = "m6-eval"
    wandb_entity: str = "1469512941-"
    seed: int = 7


@draccus.wrap()
def eval_m6(cfg: GenerateConfig) -> None:
    if not cfg.pretrained_checkpoint:
        raise ValueError("cfg.pretrained_checkpoint cannot be empty.")
    if "image_aug" in str(cfg.pretrained_checkpoint) and not cfg.center_crop:
        raise ValueError("Use center_crop=True when evaluating a checkpoint trained with image augmentation.")
    if cfg.load_in_8bit and cfg.load_in_4bit:
        raise ValueError("Cannot enable both 8-bit and 4-bit loading.")

    set_seed_everywhere(cfg.seed)
    cfg.unnorm_key = cfg.task_suite_name

    model = get_model(cfg)
    if cfg.model_family == "openvla":
        if cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
            cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
        if cfg.unnorm_key not in model.norm_stats:
            raise KeyError(f"Missing action normalization stats for {cfg.unnorm_key!r}.")

    processor = get_processor(cfg) if cfg.model_family == "openvla" else None
    resize_size = get_image_resize_size(cfg)

    run_id = f"EVAL-M6-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")

    log_file = open(local_log_filepath, "w", encoding="utf-8")
    print(f"Logging M6 evaluation to: {local_log_filepath}")
    log_file.write(f"Task suite: {cfg.task_suite_name}\n")
    log_file.write(f"Tasks: {', '.join(M6_TASK_NAMES)}\n")

    if cfg.use_wandb:
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=run_id, mode="online")

    overall_successes = 0
    overall_episodes = 0

    for task_name in M6_TASK_NAMES:
        env = MetaworldEnv(task_name)
        total_return = 0.0
        total_success = 0

        for episode_idx in tqdm(range(cfg.num_trials_per_task), desc=task_name):
            obs, _ = env.reset()
            for _ in range(cfg.num_steps_wait):
                obs, _, _, _, _ = env.step(np.zeros(4, dtype=np.float32))

            images = []
            episode_return = 0.0
            done = False

            for _ in range(cfg.max_steps):
                observation = {
                    "full_image": resize_image(copy.deepcopy(obs["image_primary"]), resize_size=(resize_size, resize_size)),
                    "state": np.concatenate((obs["proprio"][:3], np.zeros(shape=(3,)), obs["proprio"][3:4])),
                }
                task_description = task_instruction(task_name)
                images.append(observation["full_image"])

                action = get_action(cfg, model, observation, task_description, processor=processor)
                action = normalize_gripper_action(action, binarize=True)
                if cfg.model_family == "openvla":
                    action = invert_gripper_action(action)
                action = np.concatenate([action[:3], action[-1:]])

                obs, reward, done, trunc, _ = env.step(action)
                episode_return += float(reward)
                if done or trunc:
                    break

            total_return += episode_return
            total_success += int(done)
            overall_successes += int(done)
            overall_episodes += 1

            if cfg.use_wandb and episode_idx % 5 == 0 and images:
                wandb.log({f"{task_name}/rollout_video": wandb.Video(np.array(images).transpose(0, 3, 1, 2)[::10])})

        avg_return = total_return / cfg.num_trials_per_task
        success_rate = total_success / cfg.num_trials_per_task
        msg = f"{task_name}: average_return={avg_return:.4f}, success_rate={success_rate:.2%}"
        print(msg)
        log_file.write(msg + "\n")

        if cfg.use_wandb:
            wandb.log({task_name: {"average_return": avg_return, "success_rate": success_rate}})

    overall_success_rate = overall_successes / overall_episodes
    summary = f"Overall M6 success_rate={overall_success_rate:.2%} ({overall_successes}/{overall_episodes})"
    print(summary)
    log_file.write(summary + "\n")
    log_file.close()

    if cfg.use_wandb:
        wandb.log({"m6/overall_success_rate": overall_success_rate})
        wandb.finish()


if __name__ == "__main__":
    eval_m6()
