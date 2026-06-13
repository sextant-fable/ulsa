import tempfile
from pathlib import Path
import json
import os
import subprocess
import sys

import h5py
import torch
from torch.utils.data import DataLoader

from ulsa.pixel_pbu_stage1 import MASK_RANDOM, PRIOR_COPY_LAST, Stage1PBUWrapper, VariableMaskBudgetSampler
from ulsa.pixel_pbu_stage1_dataset import (
    Stage1EchoNetFrameDataset,
    collate_stage1_frame_samples,
    make_stage1_pbu_batch_from_frame_batch,
    map_range_tensor,
)


def _make_tiny_hdf5(root: Path) -> None:
    split_dir = root / "train"
    split_dir.mkdir(parents=True)
    path = split_dir / "tiny.hdf5"
    values = torch.linspace(-60.0, 0.0, steps=5 * 8 * 8, dtype=torch.float32).reshape(5, 8, 8).numpy()
    with h5py.File(path, "w") as f:
        group = f.create_group("data")
        group.create_dataset("image_sc", data=values)


def test_stage1_dataset_reads_and_normalizes_frame_triplets():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _make_tiny_hdf5(root)
        dataset = Stage1EchoNetFrameDataset(
            data_root=root,
            split="train",
            max_files=1,
            max_items=2,
            max_frames_per_file=2,
        )
        assert len(dataset) == 2
        sample = dataset[0]
        assert tuple(sample["target"].shape) == (1, 8, 8)
        assert tuple(sample["prev_x_final"].shape) == (1, 8, 8)
        assert tuple(sample["prev_prev_x_final"].shape) == (1, 8, 8)
        assert sample["frame_idx"] == 2
        assert torch.min(sample["target"]) >= -1.0
        assert torch.max(sample["target"]) <= 1.0
        assert torch.allclose(map_range_tensor(torch.tensor([-60.0, 0.0])), torch.tensor([-1.0, 1.0]))


def test_stage1_dataset_collates_and_builds_pbu_batch():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _make_tiny_hdf5(root)
        dataset = Stage1EchoNetFrameDataset(data_root=root, split="train", max_items=2)
        loader = DataLoader(dataset, batch_size=2, collate_fn=collate_stage1_frame_samples)
        frame_batch = next(iter(loader))
        sampler = VariableMaskBudgetSampler(image_shape=(8, 8), n_lines=8, seed=91)
        batch = make_stage1_pbu_batch_from_frame_batch(
            frame_batch,
            sampler=sampler,
            budget=2,
            mask_family=MASK_RANDOM,
            prior_kind=PRIOR_COPY_LAST,
        )
        model = Stage1PBUWrapper(
            n_lines=8,
            base_channels=8,
            channel_mult=(1, 2),
            num_res_blocks=1,
            groupnorm_groups=4,
        )
        output = model(batch)
        assert tuple(batch.x_gt.shape) == (2, 1, 8, 8)
        assert batch.selected_lines.sum(dim=1).tolist() == [2, 2]
        assert output.observed_consistency_l1.item() <= 1.0e-6
        assert torch.isfinite(output.x_final).all()


def test_stage1_train_dry_run_cli_reads_tiny_data_without_writing_outputs():
    repo_root = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _make_tiny_hdf5(root)
        before = sorted(str(path.relative_to(root)) for path in root.rglob("*"))
        env = dict(os.environ)
        env["PYTHONPATH"] = str(repo_root / "ulsa")
        cmd = [
            sys.executable,
            str(repo_root / "ulsa/train_pixel_pbu_stage1.py"),
            "--dry_run",
            "--data_root",
            str(root),
            "--split",
            "train",
            "--max_files",
            "1",
            "--max_items",
            "2",
            "--max_frames_per_file",
            "2",
            "--batch_size",
            "2",
            "--max_steps",
            "1",
            "--budget",
            "2",
            "--mask_family",
            MASK_RANDOM,
            "--prior_kind",
            PRIOR_COPY_LAST,
            "--n_lines",
            "8",
            "--device",
            "cpu",
        ]
        result = subprocess.run(cmd, cwd=repo_root, env=env, text=True, capture_output=True, check=True)
        payload = json.loads(result.stdout)
        assert payload["mode"] == "real_data_dry_run_no_checkpoint"
        assert payload["steps"] == 1
        assert payload["summary"]["mean_metrics"]["observed_consistency_l1"] <= 1.0e-6
        after = sorted(str(path.relative_to(root)) for path in root.rglob("*"))
        assert after == before


def test_stage1_train_cli_refuses_non_dry_run():
    repo_root = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _make_tiny_hdf5(root)
        env = dict(os.environ)
        env["PYTHONPATH"] = str(repo_root / "ulsa")
        result = subprocess.run(
            [
                sys.executable,
                str(repo_root / "ulsa/train_pixel_pbu_stage1.py"),
                "--data_root",
                str(root),
                "--split",
                "train",
                "--max_files",
                "1",
                "--max_items",
                "1",
                "--max_steps",
                "1",
                "--device",
                "cpu",
            ],
            cwd=repo_root,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode != 0
        assert "formal training is intentionally disabled" in result.stderr
