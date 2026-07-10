"""Collect expert demonstrations for the six-task MetaWorld M6 suite."""

from __future__ import annotations

import os
from pathlib import Path

import h5py
import metaworld.policies as policies
import numpy as np

from experiments.robot.metaworld.m6_tasks import M6_POLICY_CLASS_NAMES, M6_TASK_NAMES, task_instruction
from experiments.robot.metaworld.metaworld_env import MetaworldEnv


DEFAULT_SAVE_DIR = Path("/root/autodl-tmp/metaworld_m6_hdf5")


def collect_m6_data(
    episodes_per_task: int = 50,
    max_steps: int = 500,
    save_dir: Path = DEFAULT_SAVE_DIR,
    max_attempts_per_task: int | None = None,
) -> None:
    """Collect successful expert rollouts for each M6 task."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    save_dir.mkdir(parents=True, exist_ok=True)
    max_attempts_per_task = max_attempts_per_task or episodes_per_task * 20

    for task_name in M6_TASK_NAMES:
        print(f"\nCollecting M6 task: {task_name}")
        env = MetaworldEnv(task_name)

        policy_class_name = M6_POLICY_CLASS_NAMES[task_name]
        policy_cls = getattr(policies, policy_class_name)
        policy = policy_cls()
        instruction = task_instruction(task_name)

        h5_path = save_dir / f"{task_name}.hdf5"
        with h5py.File(h5_path, "w") as f:
            data_grp = f.create_group("data")
            success_count = 0
            total_attempts = 0

            while success_count < episodes_per_task and total_attempts < max_attempts_per_task:
                total_attempts += 1
                obs_dict, info = env.reset()
                images, proprios, actions = [], [], []
                episode_success = False

                for _ in range(max_steps):
                    images.append(obs_dict["image_primary"])
                    proprios.append(obs_dict["proprio"])

                    raw_state = info["state"]
                    action = policy.get_action(raw_state)
                    actions.append(action)

                    obs_dict, _, done, trunc, info = env.step(action)
                    episode_success = episode_success or bool(info.get("success", False))
                    if done or trunc:
                        break

                if not episode_success:
                    if total_attempts % 10 == 0:
                        print(
                            f"  retrying {task_name}: attempts={total_attempts}, "
                            f"successes={success_count}/{episodes_per_task}"
                        )
                    continue

                ep_grp = data_grp.create_group(f"demo_{success_count}")
                ep_grp.create_dataset("image_primary", data=np.asarray(images, dtype=np.uint8))
                ep_grp.create_dataset("proprio", data=np.asarray(proprios, dtype=np.float32))
                ep_grp.create_dataset("action", data=np.asarray(actions, dtype=np.float32))
                ep_grp.attrs["language_instruction"] = instruction

                success_count += 1
                print(
                    f"  saved success {success_count}/{episodes_per_task} "
                    f"for {task_name} ({len(actions)} steps)"
                )

        print(f"Saved {task_name} demonstrations to {h5_path}")


if __name__ == "__main__":
    collect_m6_data()
