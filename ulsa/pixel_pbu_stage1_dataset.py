from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import torch
from torch.utils.data import Dataset

from ulsa.pixel_pbu_stage1 import (
    Stage1PBUBatch,
    VariableMaskBudgetSampler,
    make_stage1_pbu_batch,
)


def map_range_tensor(
    value: torch.Tensor,
    *,
    source_range: tuple[float, float] = (-60.0, 0.0),
    target_range: tuple[float, float] = (-1.0, 1.0),
) -> torch.Tensor:
    src0, src1 = (float(source_range[0]), float(source_range[1]))
    dst0, dst1 = (float(target_range[0]), float(target_range[1]))
    if abs(src1 - src0) <= 1.0e-12:
        raise ValueError(f"source_range has zero width: {source_range}")
    out = (value.to(dtype=torch.float32) - src0) / (src1 - src0)
    return out * (dst1 - dst0) + dst0


def list_stage1_hdf5_files(data_root: str | Path, *, split: str) -> list[Path]:
    split_dir = Path(data_root) / str(split)
    if not split_dir.exists():
        raise FileNotFoundError(f"Stage 1 data split directory does not exist: {split_dir}")
    files = sorted(split_dir.glob("*.hdf5")) + sorted(split_dir.glob("*.h5"))
    if not files:
        raise FileNotFoundError(f"No HDF5 files found in Stage 1 split directory: {split_dir}")
    return files


@dataclass(frozen=True)
class Stage1FrameIndex:
    path: Path
    frame_idx: int
    n_frames: int


class Stage1EchoNetFrameDataset(Dataset):
    """Frame-level EchoNet dataset for Stage 1 PBU real-data plumbing.

    The dataset reads `data/image_sc` frames from `processed_echonet/<split>`.
    It returns target and simple temporal-history tensors only; mask sampling
    and PBU batch construction stay in the Stage 1 contract layer.
    """

    def __init__(
        self,
        data_root: str | Path = "processed_echonet",
        *,
        split: str = "train",
        key: str = "data/image_sc",
        image_range: tuple[float, float] = (-60.0, 0.0),
        output_range: tuple[float, float] = (-1.0, 1.0),
        min_history: int = 2,
        frame_stride: int = 1,
        max_files: int | None = None,
        max_frames_per_file: int | None = None,
        max_items: int | None = None,
    ) -> None:
        self.data_root = Path(data_root)
        self.split = str(split)
        self.key = str(key)
        self.image_range = (float(image_range[0]), float(image_range[1]))
        self.output_range = (float(output_range[0]), float(output_range[1]))
        self.min_history = max(0, int(min_history))
        self.frame_stride = max(1, int(frame_stride))
        self.files = list_stage1_hdf5_files(self.data_root, split=self.split)
        if max_files is not None:
            self.files = self.files[: max(0, int(max_files))]
        if not self.files:
            raise ValueError("Stage 1 dataset has no files after max_files filtering")
        self.index = self._build_index(
            max_frames_per_file=max_frames_per_file,
            max_items=max_items,
        )
        if not self.index:
            raise ValueError("Stage 1 dataset has no frame items after filtering")

    def _build_index(
        self,
        *,
        max_frames_per_file: int | None,
        max_items: int | None,
    ) -> list[Stage1FrameIndex]:
        index: list[Stage1FrameIndex] = []
        for path in self.files:
            with h5py.File(path, "r") as f:
                if self.key not in f:
                    raise KeyError(f"Missing key {self.key!r} in {path}")
                n_frames = int(f[self.key].shape[0])
            stop = n_frames
            if max_frames_per_file is not None:
                stop = min(stop, self.min_history + max(0, int(max_frames_per_file)))
            for frame_idx in range(self.min_history, stop, self.frame_stride):
                index.append(Stage1FrameIndex(path=path, frame_idx=int(frame_idx), n_frames=int(n_frames)))
                if max_items is not None and len(index) >= int(max_items):
                    return index
        return index

    def __len__(self) -> int:
        return len(self.index)

    def _read_frame(self, path: Path, frame_idx: int) -> torch.Tensor:
        with h5py.File(path, "r") as f:
            frame = torch.as_tensor(f[self.key][int(frame_idx)], dtype=torch.float32)
        frame = map_range_tensor(frame, source_range=self.image_range, target_range=self.output_range)
        return frame.unsqueeze(0)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.index[int(idx)]
        target = self._read_frame(item.path, item.frame_idx)
        prev = self._read_frame(item.path, max(0, item.frame_idx - 1))
        prev_prev = self._read_frame(item.path, max(0, item.frame_idx - 2))
        return {
            "target": target,
            "prev_x_final": prev,
            "prev_prev_x_final": prev_prev,
            "file_path": str(item.path),
            "frame_idx": int(item.frame_idx),
            "n_frames": int(item.n_frames),
            "split": self.split,
        }


def collate_stage1_frame_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("Cannot collate an empty Stage 1 sample list")
    return {
        "target": torch.stack([sample["target"] for sample in samples], dim=0),
        "prev_x_final": torch.stack([sample["prev_x_final"] for sample in samples], dim=0),
        "prev_prev_x_final": torch.stack([sample["prev_prev_x_final"] for sample in samples], dim=0),
        "file_path": [str(sample["file_path"]) for sample in samples],
        "frame_idx": [int(sample["frame_idx"]) for sample in samples],
        "n_frames": [int(sample["n_frames"]) for sample in samples],
        "split": [str(sample["split"]) for sample in samples],
    }


def make_stage1_pbu_batch_from_frame_batch(
    frame_batch: dict[str, Any],
    *,
    sampler: VariableMaskBudgetSampler,
    budget: int,
    mask_family: str,
    prior_kind: str,
    roll: int = 0,
) -> Stage1PBUBatch:
    target = frame_batch["target"].to(device=sampler.generator.device, dtype=sampler.generator.dtype)
    prev = frame_batch["prev_x_final"].to(device=sampler.generator.device, dtype=sampler.generator.dtype)
    prev_prev = frame_batch["prev_prev_x_final"].to(device=sampler.generator.device, dtype=sampler.generator.dtype)
    heuristic_map = torch.abs(prev)
    return make_stage1_pbu_batch(
        target=target,
        sampler=sampler,
        budget=int(budget),
        mask_family=mask_family,
        prior_kind=prior_kind,
        prev_x_final=prev,
        prev_prev_x_final=prev_prev,
        heuristic_map=heuristic_map,
        roll=int(roll),
    )
