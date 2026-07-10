"""Shared task definitions for the six-task MetaWorld M6 workflow."""

from __future__ import annotations

from typing import Dict, Iterable, List

import metaworld


M6_DATASET_NAME = "metaworld_m6_50e"

M6_TASK_NAMES = (
    "pick-place-wall-v3",
    "coffee-pull-v3",
    "pick-out-of-hole-v3",
    "box-close-v3",
    "peg-insert-side-v3",
    "basketball-v3",
)

M6_POLICY_CLASS_NAMES: Dict[str, str] = {
    "peg-insert-side-v3": "SawyerPegInsertionSideV3Policy",
    "basketball-v3": "SawyerBasketballV3Policy",
    "coffee-pull-v3": "SawyerCoffeePullV3Policy",
    "pick-place-wall-v3": "SawyerPickPlaceWallV3Policy",
    "pick-out-of-hole-v3": "SawyerPickOutOfHoleV3Policy",
    "box-close-v3": "SawyerBoxCloseV3Policy",
}

M6_TASK_INSTRUCTIONS: Dict[str, str] = {
    "peg-insert-side-v3": "insert a peg sideways",
    "basketball-v3": "dunk the basketball into the basket",
    "coffee-pull-v3": "pull a mug from a coffee machine",
    "pick-place-wall-v3": "pick a puck, bypass a wall, and place the puck",
    "pick-out-of-hole-v3": "pick up a puck from a hole",
    "box-close-v3": "grasp the cover and close the box with it",
}


def make_mt50(seed: int = 7):
    """Create MT50 with compatibility for MetaWorld versions that ignore seeds."""
    try:
        return metaworld.MT50(seed=seed)
    except TypeError:
        return metaworld.MT50()


def get_m6_env_classes(seed: int = 7):
    """Return env classes for the six M6 tasks without train/test splitting."""
    benchmark = make_mt50(seed)
    env_classes = {}
    for task_name in M6_TASK_NAMES:
        if task_name not in benchmark.train_classes:
            raise KeyError(f"{task_name!r} is not available in MetaWorld MT50.")
        env_classes[task_name] = benchmark.train_classes[task_name]
    return env_classes


def get_m6_task_variants(seed: int = 7) -> Dict[str, List]:
    """Return benchmark task variants for each M6 task."""
    benchmark = make_mt50(seed)
    all_tasks: Iterable = getattr(benchmark, "train_tasks", [])
    task_variants = {
        task_name: [task for task in all_tasks if task.env_name == task_name]
        for task_name in M6_TASK_NAMES
    }
    missing = [task_name for task_name, variants in task_variants.items() if not variants]
    if missing:
        raise RuntimeError(f"No MetaWorld task variants found for: {missing}")
    return task_variants


def task_instruction(task_name: str) -> str:
    """Return the detailed natural-language instruction used for M6 training and evaluation."""
    if task_name not in M6_TASK_INSTRUCTIONS:
        raise KeyError(f"{task_name!r} is not part of the M6 task suite.")
    return M6_TASK_INSTRUCTIONS[task_name]
