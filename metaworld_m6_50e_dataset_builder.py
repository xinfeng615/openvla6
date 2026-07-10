"""Build a TFDS/RLDS dataset for the six-task MetaWorld M6 suite."""

from __future__ import annotations

import os
from pathlib import Path

import h5py
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from experiments.robot.metaworld.m6_tasks import task_instruction


tfds.core.utils.gcs_utils._is_gcs_disabled = True


class MetaworldM6_50e(tfds.core.GeneratorBasedBuilder):
    """Convert M6 HDF5 expert rollouts into RLDS-compatible TFDS episodes."""

    VERSION = tfds.core.Version("1.0.0")

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            features=tfds.features.FeaturesDict(
                {
                    "steps": tfds.features.Dataset(
                        {
                            "observation": tfds.features.FeaturesDict(
                                {
                                    "image_primary": tfds.features.Image(shape=(256, 256, 3), dtype=tf.uint8),
                                    "state": tfds.features.Tensor(shape=(4,), dtype=tf.float32),
                                }
                            ),
                            "action": tfds.features.Tensor(shape=(4,), dtype=tf.float32),
                            "discount": tfds.features.Scalar(dtype=tf.float32),
                            "reward": tfds.features.Scalar(dtype=tf.float32),
                            "is_first": tfds.features.Scalar(dtype=tf.bool),
                            "is_last": tfds.features.Scalar(dtype=tf.bool),
                            "is_terminal": tfds.features.Scalar(dtype=tf.bool),
                            "language_instruction": tfds.features.Text(),
                        }
                    ),
                    "episode_metadata": tfds.features.FeaturesDict(
                        {
                            "file_path": tfds.features.Text(),
                            "task_name": tfds.features.Text(),
                        }
                    ),
                }
            ),
        )

    def _split_generators(self, dl_manager):
        return {"train": self._generate_examples(path="/root/autodl-tmp/metaworld_m6_hdf5")}

    def _generate_examples(self, path):
        episode_id = 0
        path = Path(path)

        for h5_file in sorted(os.listdir(path)):
            if not h5_file.endswith(".hdf5"):
                continue

            file_path = path / h5_file
            task_name = file_path.stem
            print(f"Processing M6 file: {file_path}")

            with h5py.File(file_path, "r") as f:
                data_grp = f["data"]

                for ep_key in sorted(data_grp.keys()):
                    ep_grp = data_grp[ep_key]
                    images = ep_grp["image_primary"][:]
                    proprios = ep_grp["proprio"][:]
                    actions = ep_grp["action"][:]
                    instruction = task_instruction(task_name)

                    episode_length = len(actions)
                    steps = []
                    for i in range(episode_length):
                        steps.append(
                            {
                                "observation": {
                                    "image_primary": images[i],
                                    "state": proprios[i][:4].astype(np.float32),
                                },
                                "action": actions[i][:4].astype(np.float32),
                                "discount": 1.0,
                                "reward": float(i == episode_length - 1),
                                "is_first": i == 0,
                                "is_last": i == episode_length - 1,
                                "is_terminal": i == episode_length - 1,
                                "language_instruction": instruction,
                            }
                        )

                    yield str(episode_id), {
                        "steps": steps,
                        "episode_metadata": {
                            "file_path": str(file_path),
                            "task_name": task_name,
                        },
                    }
                    episode_id += 1

        print(f"Processed {episode_id} M6 trajectories.")


if __name__ == "__main__":
    builder = MetaworldM6_50e(data_dir="/root/autodl-tmp/tensorflow_datasets")
    print("Building MetaWorld M6 RLDS dataset...")
    builder.download_and_prepare()
    print("Saved dataset to /root/autodl-tmp/tensorflow_datasets/metaworld_m6_50e")
