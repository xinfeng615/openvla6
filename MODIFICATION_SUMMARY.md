# OpenVLA-Zero 修改总结

本文档总结本轮对话中围绕微调复现性、LoRA checkpoint 保存/合并、评估顺序、工具脚本和评估报错修复所做的代码修改。

## 1. 微调结果不一致的原因分析与修复

最初的问题是：相同代码重复微调两次，评估准确率不一样。评估代码已经设置了随机种子，因此重点检查训练侧。

排查后发现训练侧存在多个未固定的随机源：

- LoRA 初始化使用 `init_lora_weights="gaussian"`，如果训练前没有固定 PyTorch seed，LoRA 初始权重会不同。
- RLDS 数据管线使用 `dataset.shuffle(...)`，原先没有传入 seed。
- 图像增强中使用 `tf.random.uniform(...)` 生成增强随机种子，原先没有固定。
- Python、NumPy、PyTorch、TensorFlow 的随机种子没有在微调脚本中统一设置。
- CUDA/cuDNN/TF32 算子存在非确定性路径，可能导致微小数值差异累积。

### 修改文件

- `vla-scripts/finetune.py`
- `prismatic/vla/datasets/datasets.py`
- `prismatic/vla/datasets/rlds/dataset.py`

### 主要修改

在 `vla-scripts/finetune.py` 中新增：

```python
def set_training_seed(seed: int, deterministic: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)
```

新增配置项：

```python
seed: int = 7
deterministic_training: bool = True
```

并在模型、LoRA、数据集创建前调用：

```python
set_training_seed(cfg.seed, cfg.deterministic_training)
```

同时将 seed 传入：

- `RLDSDataset(...)`
- `DataLoader(generator=torch.Generator().manual_seed(...))`
- RLDS 的 `sample_from_datasets`
- RLDS 的 `dataset.shuffle`
- 图像增强中的 `tf.random.uniform`

### 关于 `deterministic_training`

`deterministic_training=True` 会让训练更可复现，但可能变慢，因为它会：

- 关闭 `cudnn.benchmark`
- 禁用 TF32
- 尽量使用确定性 CUDA 算子

如果设置为：

```bash
--deterministic_training False
```

训练速度可能更快，但不再保证 CUDA 算子层面的严格确定性。由于 seed、shuffle、LoRA 初始化和图像增强已经固定，两次训练通常仍会比原来稳定很多，但仍可能存在小幅差异。

## 2. 训练中只保存 LoRA Adapter

原始 `finetune.py` 的 checkpoint 逻辑在每次保存时会：

1. 保存 LoRA adapter。
2. 重新加载一份 base OpenVLA-7B。
3. 用 `PeftModel.from_pretrained(...)` 加载 adapter。
4. 执行 `merge_and_unload()`。
5. 保存合并后的完整模型。

这会在训练进程中额外加载一份 7B 模型，导致 checkpoint 保存时内存/显存峰值过高。

### 修改文件

- `vla-scripts/finetune.py`
- `finetune_m6.sh`

### 主要修改

将训练中的保存逻辑改为：只保存 adapter、processor 和 `dataset_statistics.json`，不再训练中合并完整模型。

新的 checkpoint 目录形式：

```text
<run_dir>/
  dataset_statistics.json
  checkpoints/
    step-1000/
      adapter_config.json
      adapter_model.safetensors
      tokenizer / processor files
      dataset_statistics.json
    step-2000/
      ...
```

默认配置改为：

```python
save_latest_checkpoint_only: bool = False
```

这样默认保存所有 checkpoint，便于后续选择任意 step 进行合并和评估。

训练保存逻辑核心变为：

```python
if cfg.save_latest_checkpoint_only:
    checkpoint_dir = run_dir / "checkpoints" / "latest"
else:
    checkpoint_dir = run_dir / "checkpoints" / f"step-{optimizer_step_idx}"

processor.save_pretrained(checkpoint_dir)
save_dataset_statistics(vla_dataset.dataset_statistics, checkpoint_dir)
vla.module.save_pretrained(checkpoint_dir)
```

同时修正了保存 step 计数，使 `step-2000` 对应第 2000 次 optimizer update 后的 adapter，而不是旧代码里基于 `batch_idx // grad_accumulation_steps` 的偏移计数。

### M6 脚本修改

`finetune_m6.sh` 修改为：

```bash
--save_steps 2000 \
--save_latest_checkpoint_only False \
```

如果想每 1000 步保存，直接改为：

```bash
--save_steps 1000
```

## 3. 新增独立 LoRA 合并脚本

为了保持评估代码不变，同时避免训练中合并导致内存不足，新增独立合并脚本。

### 新增文件

- `vla-scripts/merge_lora_checkpoint.py`

### 用法

```bash
python vla-scripts/merge_lora_checkpoint.py \
  --base_model /root/autodl-tmp/openvla/openvla-7b \
  --adapter_checkpoint /root/autodl-tmp/openvla/output_m6/<run-id>/checkpoints/step-20000 \
  --output_dir /root/autodl-tmp/openvla/output_m6/<run-id>/merged-step-20000
```

评估时将 `--pretrained_checkpoint` 指向 `merged-step-20000` 即可。

### 合并逻辑

该脚本最终被调整为完全按原始 `finetune.py` 的合并逻辑执行：

```python
base_vla = AutoModelForVision2Seq.from_pretrained(
    cfg.base_model,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)
merged_vla = PeftModel.from_pretrained(base_vla, cfg.adapter_checkpoint)
merged_vla = merged_vla.merge_and_unload()
merged_vla.save_pretrained(cfg.output_dir)
```

也就是说：

- 不再传 `max_shard_size`
- 不再支持额外可配置的 `torch_dtype`
- `save_pretrained()` 使用 HuggingFace 默认保存逻辑

脚本仍会在合并后额外保存：

- processor/tokenizer 文件
- `dataset_statistics.json`

这样合并后的目录包含现有评估代码需要的文件。

### 关于 safetensors shard 数量

曾经出现合并后为 3 个 `.safetensors`，而原保存逻辑为 4 个的问题。原因是新脚本一开始显式设置过：

```python
max_shard_size="7GB"
```

这会改变 HuggingFace 切分权重文件的策略。现在已移除该参数，回到原始保存逻辑。

## 4. 修复 Draccus Dataclass 报错

运行：

```bash
bash eval_m6_zeroshot.sh
```

时出现：

```text
TypeError: must be called with a dataclass type or instance
```

根因是评估脚本顶部使用了：

```python
from __future__ import annotations
```

这会让函数注解在运行时变为字符串。在当前环境的 `draccus` 版本中，`@draccus.wrap()` 没有正确把字符串注解解析回 dataclass 类型。

### 修改文件

- `experiments/robot/metaworld/run_m6_zeroshot_eval.py`
- `experiments/robot/metaworld/run_m6_eval.py`

### 修改内容

移除：

```python
from __future__ import annotations
```

这样：

```python
def eval_m6_zeroshot(cfg: GenerateConfig) -> None:
```

中的 `GenerateConfig` 在运行时就是实际 dataclass 类型，`draccus.wrap()` 可以正常解析。

## 5. 修改 M6 评估顺序

用户要求评估顺序改为：

1. pick-place-wall
2. coffee-pull
3. pick-out-of-hole
4. box-close
5. peg insert side
6. basketball

### 修改文件

- `experiments/robot/metaworld/m6_tasks.py`

### 修改内容

将：

```python
M6_TASK_NAMES = (
    "peg-insert-side-v3",
    "basketball-v3",
    "coffee-pull-v3",
    "pick-place-wall-v3",
    "pick-out-of-hole-v3",
    "box-close-v3",
)
```

改为：

```python
M6_TASK_NAMES = (
    "pick-place-wall-v3",
    "coffee-pull-v3",
    "pick-out-of-hole-v3",
    "box-close-v3",
    "peg-insert-side-v3",
    "basketball-v3",
)
```

`run_m6_eval.py` 和 `run_m6_zeroshot_eval.py` 都从 `M6_TASK_NAMES` 读取任务顺序，因此两种评估都会按该顺序执行。

## 6. 新增 HDF5 转 GIF 工具

为了检查采集数据质量，新增一个工具目录和 HDF5 转 GIF 脚本。

### 新增文件

- `tool/hdf5_to_gif.py`
- `tool/README.md`

### 默认支持的数据结构

采集脚本保存的 HDF5 结构为：

```text
data/
  demo_0/
    image_primary
    proprio
    action
    attrs["language_instruction"]
  demo_1/
    ...
```

工具默认读取：

```text
data/demo_x/image_primary
```

### 用法

转换单个 HDF5 文件中的所有 demo：

```bash
python tool/hdf5_to_gif.py \
  --input /root/autodl-tmp/metaworld_m6_hdf5/button-press-v3.hdf5 \
  --output-dir /root/autodl-tmp/metaworld_gifs
```

只转换一个 episode：

```bash
python tool/hdf5_to_gif.py \
  --input /root/autodl-tmp/metaworld_m6_hdf5/button-press-v3.hdf5 \
  --episode demo_0
```

批量转换一个目录下所有 `.hdf5`：

```bash
python tool/hdf5_to_gif.py \
  --input /root/autodl-tmp/metaworld_m6_hdf5 \
  --output-dir /root/autodl-tmp/metaworld_gifs
```

可选参数包括：

- `--fps`
- `--stride`
- `--max-frames`
- `--resize WIDTH HEIGHT`
- `--recursive`
- `--image-key`

## 7. 关于单任务和六任务微调步数的建议

对话中还讨论了单任务与六任务联合微调的步数换算。

粗略换算：

```text
单任务 2000 步
≈ 六任务联合训练 12000 总步时，每个任务平均被采到约 2000 步
```

不是 24000 步，因为 M6 是 6 个任务混合，平均每个任务获得：

```text
每个任务平均曝光量 ≈ 总训练步数 / 6
```

建议正式 M6 联合微调先使用：

```bash
--max_steps 20000
--save_steps 1000
```

然后评估中间 checkpoint，而不是只看最后一步。重点关注：

```text
step-5000 到 step-15000
```

因为最优 rollout 成功率可能出现在中间。

## 8. 当前检查结果

已对以下文件做 Python 语法级编译检查，均通过：

- `vla-scripts/finetune.py`
- `vla-scripts/merge_lora_checkpoint.py`
- `experiments/robot/metaworld/m6_tasks.py`
- `experiments/robot/metaworld/run_m6_eval.py`
- `experiments/robot/metaworld/run_m6_zeroshot_eval.py`
- `prismatic/vla/datasets/datasets.py`
- `prismatic/vla/datasets/rlds/dataset.py`
- `tool/hdf5_to_gif.py`

注意事项：

- 本地环境缺少部分训练/评估依赖，例如 `draccus`、`tensorflow`、`dlimp`，因此没有在本机运行完整训练或完整评估。
- 本地 `h5py/numpy` 曾出现二进制版本不匹配，因此 HDF5 转 GIF 工具只做了语法检查和 `--help` 检查；在服务器 `openvla` 环境中应使用正常的 `h5py` 运行。
- `finetune_ml1_basketball.sh` 仍然显式设置：

```bash
--save_latest_checkpoint_only True
```

如果希望 ML1 也保存所有 step checkpoint，需要手动改成：

```bash
--save_latest_checkpoint_only False
```

M6 脚本 `finetune_m6.sh` 已经设置为保存所有 checkpoint。
