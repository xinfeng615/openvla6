"""
merge_lora_checkpoint.py

Merge a saved LoRA adapter checkpoint into the base OpenVLA model for evaluation.

Example:
    python vla-scripts/merge_lora_checkpoint.py \
        --base_model /root/autodl-tmp/openvla/openvla-7b \
        --adapter_checkpoint /root/autodl-tmp/openvla/output_m6/<run-id>/checkpoints/step-20000 \
        --output_dir /root/autodl-tmp/openvla/output_m6/<run-id>/merged-step-20000
"""

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import draccus
import torch
from peft import PeftModel
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor


@dataclass
class MergeConfig:
    base_model: str
    adapter_checkpoint: Path
    output_dir: Path
    processor_path: Optional[Path] = None
    dataset_statistics_path: Optional[Path] = None


def register_openvla_auto_classes() -> None:
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)


def has_processor_files(path: Path) -> bool:
    return any(
        (path / filename).exists()
        for filename in (
            "preprocessor_config.json",
            "processor_config.json",
            "tokenizer_config.json",
            "tokenizer.model",
        )
    )


def resolve_processor_source(cfg: MergeConfig) -> str:
    if cfg.processor_path is not None:
        return str(cfg.processor_path)
    if has_processor_files(cfg.adapter_checkpoint):
        return str(cfg.adapter_checkpoint)

    run_dir = cfg.adapter_checkpoint.parent.parent
    if has_processor_files(run_dir):
        return str(run_dir)

    return cfg.base_model


def resolve_dataset_statistics(cfg: MergeConfig) -> Optional[Path]:
    if cfg.dataset_statistics_path is not None:
        return cfg.dataset_statistics_path

    candidates = [
        cfg.adapter_checkpoint / "dataset_statistics.json",
        cfg.adapter_checkpoint.parent.parent / "dataset_statistics.json",
        cfg.adapter_checkpoint.parent / "dataset_statistics.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


@draccus.wrap()
def merge_lora_checkpoint(cfg: MergeConfig) -> None:
    if not cfg.adapter_checkpoint.exists():
        raise FileNotFoundError(f"Missing adapter checkpoint: {cfg.adapter_checkpoint}")
    if not (cfg.adapter_checkpoint / "adapter_config.json").exists():
        raise FileNotFoundError(f"Missing adapter_config.json in: {cfg.adapter_checkpoint}")

    register_openvla_auto_classes()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading base model from: {cfg.base_model}")
    base_vla = AutoModelForVision2Seq.from_pretrained(
        cfg.base_model,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    print(f"Loading LoRA adapter from: {cfg.adapter_checkpoint}")
    merged_vla = PeftModel.from_pretrained(base_vla, cfg.adapter_checkpoint)
    merged_vla = merged_vla.merge_and_unload()

    print(f"Saving merged model to: {cfg.output_dir}")
    merged_vla.save_pretrained(cfg.output_dir)

    processor_source = resolve_processor_source(cfg)
    print(f"Saving processor files from: {processor_source}")
    processor = AutoProcessor.from_pretrained(processor_source, trust_remote_code=True)
    processor.save_pretrained(cfg.output_dir)

    dataset_statistics = resolve_dataset_statistics(cfg)
    if dataset_statistics is None:
        print("WARNING: dataset_statistics.json was not found; evaluation may fail for fine-tuned checkpoints.")
    else:
        output_statistics = cfg.output_dir / "dataset_statistics.json"
        if dataset_statistics.resolve() != output_statistics.resolve():
            shutil.copyfile(dataset_statistics, output_statistics)
        print(f"Copied dataset statistics from: {dataset_statistics}")

    print(f"Done. Use this for evaluation: --pretrained_checkpoint {cfg.output_dir}")


if __name__ == "__main__":
    merge_lora_checkpoint()
