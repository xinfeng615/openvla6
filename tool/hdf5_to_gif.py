"""
Convert collected MetaWorld HDF5 rollouts to GIFs for quick data-quality inspection.

Expected default HDF5 layout:
    data/demo_0/image_primary
    data/demo_1/image_primary
    ...

Examples:
    # Convert all demos in one file.
    python tool/hdf5_to_gif.py --input /root/autodl-tmp/metaworld_m6_hdf5/button-press-v3.hdf5

    # Convert only demo_0.
    python tool/hdf5_to_gif.py --input /root/autodl-tmp/metaworld_m6_hdf5/button-press-v3.hdf5 --episode demo_0

    # Convert every .hdf5 file in a directory.
    python tool/hdf5_to_gif.py --input /root/autodl-tmp/metaworld_m6_hdf5 --output-dir /root/autodl-tmp/metaworld_gifs
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert HDF5 rollout images to GIFs.")
    parser.add_argument("--input", type=Path, required=True, help="Input .hdf5 file or directory containing .hdf5 files.")
    parser.add_argument("--output-dir", type=Path, default=Path("tool/gifs"), help="Directory for generated GIFs.")
    parser.add_argument("--episode", type=str, default=None, help="Episode key to export, e.g. demo_0. Defaults to all.")
    parser.add_argument("--image-key", type=str, default="image_primary", help="Image dataset key inside each episode.")
    parser.add_argument("--fps", type=float, default=20.0, help="Output GIF frame rate.")
    parser.add_argument("--stride", type=int, default=1, help="Keep one frame every N frames.")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional maximum number of frames per GIF.")
    parser.add_argument("--resize", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"), default=None)
    parser.add_argument("--recursive", action="store_true", help="Recursively search for .hdf5 files in input dir.")
    return parser.parse_args()


def iter_hdf5_files(input_path: Path, recursive: bool) -> Iterable[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in {".hdf5", ".h5"}:
            raise ValueError(f"Input file is not .hdf5/.h5: {input_path}")
        yield input_path
        return

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    pattern = "**/*" if recursive else "*"
    for path in sorted(input_path.glob(pattern)):
        if path.is_file() and path.suffix.lower() in {".hdf5", ".h5"}:
            yield path


def natural_demo_key(key: str) -> tuple[int, str]:
    if key.startswith("demo_"):
        suffix = key.removeprefix("demo_")
        if suffix.isdigit():
            return int(suffix), key
    return 10**12, key


def normalize_frame(frame: np.ndarray) -> np.ndarray:
    frame = np.asarray(frame)

    if frame.ndim == 2:
        frame = np.repeat(frame[..., None], 3, axis=-1)
    elif frame.ndim == 3 and frame.shape[0] in {1, 3, 4} and frame.shape[-1] not in {1, 3, 4}:
        frame = np.moveaxis(frame, 0, -1)

    if frame.dtype != np.uint8:
        frame = frame.astype(np.float32)
        if frame.size and frame.max() <= 1.0:
            frame = frame * 255.0
        frame = np.clip(frame, 0, 255).astype(np.uint8)

    if frame.ndim != 3 or frame.shape[-1] not in {1, 3, 4}:
        raise ValueError(f"Expected image frame shape HxWxC, got {frame.shape}")

    if frame.shape[-1] == 1:
        frame = np.repeat(frame, 3, axis=-1)
    return frame


def frames_to_gif(frames: np.ndarray, output_path: Path, fps: float, resize: Optional[tuple[int, int]]) -> None:
    if len(frames) == 0:
        raise ValueError("Cannot write GIF with zero frames.")

    pil_frames = []
    for frame in frames:
        image = Image.fromarray(normalize_frame(frame))
        if resize is not None:
            image = image.resize(resize, Image.Resampling.BILINEAR)
        pil_frames.append(image)

    duration_ms = max(1, int(round(1000.0 / fps)))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )


def export_file(
    hdf5_path: Path,
    output_dir: Path,
    episode: Optional[str],
    image_key: str,
    fps: float,
    stride: int,
    max_frames: Optional[int],
    resize: Optional[tuple[int, int]],
) -> int:
    import h5py

    exported = 0
    with h5py.File(hdf5_path, "r") as h5_file:
        if "data" not in h5_file:
            raise KeyError(f"{hdf5_path} does not contain a top-level 'data' group.")

        data_group = h5_file["data"]
        episode_keys = [episode] if episode else sorted(data_group.keys(), key=natural_demo_key)

        for episode_key in episode_keys:
            if episode_key not in data_group:
                raise KeyError(f"{hdf5_path} does not contain episode '{episode_key}'.")

            episode_group = data_group[episode_key]
            if image_key not in episode_group:
                raise KeyError(f"{hdf5_path}:{episode_key} does not contain image key '{image_key}'.")

            frames = episode_group[image_key][:]
            frames = frames[::stride]
            if max_frames is not None:
                frames = frames[:max_frames]

            task_name = hdf5_path.stem
            output_path = output_dir / task_name / f"{episode_key}.gif"
            frames_to_gif(frames, output_path, fps=fps, resize=resize)

            instruction = episode_group.attrs.get("language_instruction", "")
            if isinstance(instruction, bytes):
                instruction = instruction.decode("utf-8", errors="replace")
            print(f"Saved {output_path} | frames={len(frames)} | instruction={instruction}")
            exported += 1

    return exported


def main() -> None:
    args = parse_args()
    if args.fps <= 0:
        raise ValueError("--fps must be positive.")
    if args.stride <= 0:
        raise ValueError("--stride must be positive.")
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("--max-frames must be positive when provided.")

    resize = tuple(args.resize) if args.resize is not None else None
    total = 0
    for hdf5_path in iter_hdf5_files(args.input, args.recursive):
        total += export_file(
            hdf5_path=hdf5_path,
            output_dir=args.output_dir,
            episode=args.episode,
            image_key=args.image_key,
            fps=args.fps,
            stride=args.stride,
            max_frames=args.max_frames,
            resize=resize,
        )

    print(f"Done. Exported {total} GIF(s) to {args.output_dir}")


if __name__ == "__main__":
    main()
