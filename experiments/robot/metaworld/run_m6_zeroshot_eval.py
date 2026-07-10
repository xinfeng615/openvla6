"""Strict zero-shot evaluation of the original OpenVLA-7B on MetaWorld M6.

This script intentionally does not use MetaWorld action normalization
statistics. It decodes OpenVLA action tokens directly to normalized values and
maps the first three action dimensions plus gripper to MetaWorld's 4-DoF action
space. The reported accuracy is environment success rate.
"""

import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Union

import draccus
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.append("../..")

from experiments.robot.metaworld.m6_tasks import (
    M6_DATASET_NAME,
    M6_TASK_NAMES,
    get_m6_env_classes,
    get_m6_task_variants,
    task_instruction,
)
from experiments.robot.openvla_utils import get_processor, get_vla_normalized_action
from experiments.robot.robot_utils import DATE_TIME, get_model, set_seed_everywhere


@dataclass
class GenerateConfig:
    model_family: str = "openvla"
    pretrained_checkpoint: Union[str, Path] = "openvla-7b"
    load_in_8bit: bool = False
    load_in_4bit: bool = False

    task_suite_name: str = M6_DATASET_NAME
    task_name: Optional[str] = None
    num_trials_per_task: int = 10
    max_steps: int = 500
    num_steps_wait: int = 0
    action_scale: float = 1.0
    render_size: int = 256
    camera_name: str = "corner2"
    save_gifs: bool = True
    gif_episodes_per_task: int = 1
    gif_frame_stride: int = 5
    gif_frame_duration_ms: int = 100

    local_log_dir: str = "./experiments/logs"
    local_video_dir: str = "./experiments/logs/videos"
    seed: int = 7


def _step_env(env, action):
    result = env.step(action)
    if len(result) == 5:
        return result
    obs, reward, done, info = result
    return obs, reward, done, False, info


def _metaworld_action(normalized_action: np.ndarray, action_scale: float) -> np.ndarray:
    """Map OpenVLA's decoded 7-DoF normalized action to MetaWorld's 4-DoF action."""
    xyz = np.clip(normalized_action[:3] * action_scale, -1.0, 1.0)

    # get_vla_normalized_action decodes token IDs to bin centers in [-1, 1].
    # MetaWorld uses -1 for open and +1 for closed.
    gripper = -1.0 if normalized_action[-1] > 0.0 else 1.0
    return np.asarray([xyz[0], xyz[1], xyz[2], gripper], dtype=np.float32)


def _save_gif(frames, output_path: Path, frame_duration_ms: int) -> None:
    if not frames:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pil_frames = [Image.fromarray(frame) for frame in frames]
    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=frame_duration_ms,
        loop=0,
        optimize=False,
    )


@draccus.wrap()
def eval_m6_zeroshot(cfg: GenerateConfig) -> None:
    if cfg.model_family != "openvla":
        raise ValueError("This evaluator only supports model_family='openvla'.")
    if cfg.load_in_8bit and cfg.load_in_4bit:
        raise ValueError("Cannot enable both 8-bit and 4-bit loading.")

    checkpoint = Path(cfg.pretrained_checkpoint)
    if checkpoint.exists() and (checkpoint / "dataset_statistics.json").exists():
        print(
            "Warning: dataset_statistics.json exists in this checkpoint. "
            "M6 zero-shot mode will intentionally ignore it."
        )

    set_seed_everywhere(cfg.seed)
    model = get_model(cfg)
    processor = get_processor(cfg)

    env_classes = get_m6_env_classes(cfg.seed)
    tasks_by_env = get_m6_task_variants(cfg.seed)

    task_names = list(M6_TASK_NAMES)
    if cfg.task_name is not None:
        if cfg.task_name not in env_classes:
            raise ValueError(f"Task {cfg.task_name!r} is not in M6. Available tasks: {task_names}")
        task_names = [cfg.task_name]

    run_id = f"EVAL-ZEROSHOT-M6-{cfg.task_suite_name}-{DATE_TIME}"
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    log_path = Path(cfg.local_log_dir) / f"{run_id}.json"
    video_dir = Path(cfg.local_video_dir) / run_id

    all_results = []
    total_successes = 0
    total_episodes = 0

    for task_name in task_names:
        try:
            env = env_classes[task_name](
                render_mode="rgb_array",
                camera_name=cfg.camera_name,
                width=cfg.render_size,
                height=cfg.render_size,
            )
        except TypeError:
            env = env_classes[task_name](render_mode="rgb_array")

        env.model.vis.global_.offwidth = cfg.render_size
        env.model.vis.global_.offheight = cfg.render_size

        task_variants = tasks_by_env[task_name]
        if not task_variants:
            raise RuntimeError(f"No benchmark task variants found for {task_name}.")

        num_episodes = cfg.num_trials_per_task if cfg.num_trials_per_task > 0 else len(task_variants)
        instruction = task_instruction(task_name)
        task_successes = 0
        task_returns = []
        task_lengths = []
        task_gifs = []
        normalized_action_samples = []
        metaworld_action_samples = []

        for episode_idx in tqdm(range(num_episodes), desc=task_name, position=0, leave=True):
            task = task_variants[episode_idx % len(task_variants)]
            env.set_task(task)
            env.reset()

            for _ in range(cfg.num_steps_wait):
                _step_env(env, np.zeros(4, dtype=np.float32))

            episode_return = 0.0
            success = False
            record_gif = cfg.save_gifs and episode_idx < cfg.gif_episodes_per_task
            gif_frames = []

            for step_idx in range(cfg.max_steps):
                image = np.asarray(env.render(), dtype=np.uint8)
                if record_gif and step_idx % cfg.gif_frame_stride == 0:
                    gif_frames.append(image.copy())

                normalized_action = get_vla_normalized_action(
                    model,
                    processor,
                    str(cfg.pretrained_checkpoint),
                    {"full_image": image},
                    instruction,
                    action_dim=7,
                )
                action = _metaworld_action(normalized_action, cfg.action_scale)

                if episode_idx == 0 and step_idx < 5:
                    normalized_action_samples.append(normalized_action.tolist())
                    metaworld_action_samples.append(action.tolist())

                _, reward, terminated, truncated, info = _step_env(env, action)
                episode_return += float(reward)
                success = bool(info.get("success", False))
                if success or terminated or truncated:
                    break

            if record_gif:
                gif_frames.append(np.asarray(env.render(), dtype=np.uint8).copy())
                outcome = "success" if success else "failure"
                gif_path = video_dir / f"{task_name}-episode-{episode_idx + 1:03d}-{outcome}.gif"
                _save_gif(gif_frames, gif_path, cfg.gif_frame_duration_ms)
                task_gifs.append(str(gif_path))

            task_successes += int(success)
            total_successes += int(success)
            total_episodes += 1
            task_returns.append(episode_return)
            task_lengths.append(step_idx + 1)

        env.close()
        task_result = {
            "task": task_name,
            "instruction": instruction,
            "episodes": num_episodes,
            "successes": task_successes,
            "success_rate": task_successes / num_episodes,
            "average_return": float(np.mean(task_returns)),
            "return_std": float(np.std(task_returns)),
            "average_episode_length": float(np.mean(task_lengths)),
            "normalized_action_samples": normalized_action_samples,
            "metaworld_action_samples": metaworld_action_samples,
            "gifs": task_gifs,
        }
        all_results.append(task_result)
        print(
            f"{task_name}: success={task_result['success_rate']:.1%}, "
            f"return={task_result['average_return']:.3f}, "
            f"length={task_result['average_episode_length']:.1f}"
        )

    summary = {
        "protocol": "strict_zero_shot_m6_normalized_action_decode",
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(cfg).items()},
        "tasks": all_results,
        "total_episodes": total_episodes,
        "total_successes": total_successes,
        "overall_success_rate": total_successes / total_episodes if total_episodes else 0.0,
    }
    log_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Overall M6 zero-shot success rate: {summary['overall_success_rate']:.1%}")
    print(f"Results saved to: {log_path}")


if __name__ == "__main__":
    eval_m6_zeroshot()
