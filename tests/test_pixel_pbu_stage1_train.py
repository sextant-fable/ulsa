import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import h5py
import torch
from torch.utils.data import DataLoader

from ulsa.pixel_pbu_stage1 import (
    MASK_RANDOM,
    MASK_ROLLED_EQUISPACED,
    PRIOR_COPY_LAST,
    PRIOR_EMA,
    Stage1PBUWrapper,
    VariableMaskBudgetSampler,
)
from ulsa.pixel_pbu_stage1_dataset import collate_stage1_frame_samples, make_stage1_pbu_batch_from_frame_batch, Stage1EchoNetFrameDataset
from ulsa.pixel_pbu_stage1_train import Stage1PBULossWeights, run_stage1_pbu_train_step
from research_harness.parsers import summarize_output


OUTPUT_SENTINELS = ("output", "research_runs", "checkpoints", "logs")


def _make_tiny_hdf5(root: Path) -> None:
    split_dir = root / "train"
    split_dir.mkdir(parents=True)
    path = split_dir / "tiny.hdf5"
    values = torch.linspace(-60.0, 0.0, steps=5 * 8 * 8, dtype=torch.float32).reshape(5, 8, 8).numpy()
    with h5py.File(path, "w") as f:
        group = f.create_group("data")
        group.create_dataset("image_sc", data=values)


def _tiny_batch(root: Path):
    dataset = Stage1EchoNetFrameDataset(data_root=root, split="train", max_items=2)
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_stage1_frame_samples)
    frame_batch = next(iter(loader))
    sampler = VariableMaskBudgetSampler(image_shape=(8, 8), n_lines=8, seed=107)
    return make_stage1_pbu_batch_from_frame_batch(
        frame_batch,
        sampler=sampler,
        budget=2,
        mask_family=MASK_RANDOM,
        prior_kind=PRIOR_COPY_LAST,
    )


def test_stage1_train_step_backward_smoke_has_finite_loss_and_gradients():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _make_tiny_hdf5(root)
        batch = _tiny_batch(root)
        model = Stage1PBUWrapper(
            n_lines=8,
            base_channels=8,
            channel_mult=(1, 2),
            num_res_blocks=1,
            groupnorm_groups=4,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4)
        result = run_stage1_pbu_train_step(
            batch=batch,
            model=model,
            optimizer=optimizer,
            weights=Stage1PBULossWeights(),
            backward=True,
        )
        assert torch.isfinite(result.total_loss).all()
        assert result.grad_norm is not None
        assert result.grad_norm > 0.0
        assert result.metrics["observed_consistency_l1"] <= 1.0e-6
        assert result.metrics["loss_observed_raw_l1"] >= 0.0


def test_stage1_train_cli_backward_smoke_no_checkpoint_or_output():
    repo_root = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _make_tiny_hdf5(root)
        before = sorted(str(path.relative_to(root)) for path in root.rglob("*"))
        repo_before = {
            name: (repo_root / name).stat().st_mtime_ns if (repo_root / name).exists() else None
            for name in OUTPUT_SENTINELS
        }
        env = dict(os.environ)
        env["PYTHONPATH"] = str(repo_root / "ulsa")
        result = subprocess.run(
            [
                sys.executable,
                str(repo_root / "ulsa/train_pixel_pbu_stage1.py"),
                "--dry_run",
                "--backward_smoke",
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
                "99",
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
            ],
            cwd=repo_root,
            env=env,
            text=True,
            capture_output=True,
            check=True,
        )
        payload = json.loads(result.stdout)
        assert payload["mode"] == "real_data_backward_smoke_no_checkpoint"
        assert payload["backward_smoke"] is True
        assert payload["steps"] == 1
        assert payload["summary"]["mean_metrics"]["observed_consistency_l1"] <= 1.0e-6
        assert payload["summary"]["mean_metrics"]["grad_norm"] > 0.0
        after = sorted(str(path.relative_to(root)) for path in root.rglob("*"))
        assert after == before
        repo_after = {
            name: (repo_root / name).stat().st_mtime_ns if (repo_root / name).exists() else None
            for name in OUTPUT_SENTINELS
        }
        assert repo_after == repo_before


def test_stage1_train_cli_refuses_no_args_formal_training_guard():
    repo_root = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root / "ulsa")
    result = subprocess.run(
        [sys.executable, str(repo_root / "ulsa/train_pixel_pbu_stage1.py")],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "formal training is intentionally disabled" in result.stderr


def test_stage1_train_cli_train_smoke_writes_only_declared_output_and_checkpoint():
    repo_root = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        root = base / "data"
        out = base / "stage1_train_smoke"
        _make_tiny_hdf5(root)
        before = sorted(str(path.relative_to(root)) for path in root.rglob("*"))
        repo_before = {
            name: (repo_root / name).stat().st_mtime_ns if (repo_root / name).exists() else None
            for name in OUTPUT_SENTINELS
        }
        env = dict(os.environ)
        env["PYTHONPATH"] = str(repo_root / "ulsa")
        result = subprocess.run(
            [
                sys.executable,
                str(repo_root / "ulsa/train_pixel_pbu_stage1.py"),
                "--train_smoke",
                "--output_dir",
                str(out),
                "--save_checkpoint",
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
                "2",
                "--smoke_max_steps_cap",
                "2",
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
            ],
            cwd=repo_root,
            env=env,
            text=True,
            capture_output=True,
            check=True,
        )
        payload = json.loads(result.stdout)
        assert payload["mode"] == "real_data_train_smoke"
        assert payload["train_smoke"] is True
        assert payload["steps"] == 2
        assert payload["requested_steps"] == 2
        assert payload["output_dir"] == str(out)
        assert payload["checkpoint"] == str(out / "stage1_pbu_smoke_last.pt")
        assert payload["basic_metrics_path"] == str(out / "basic_metrics.json")
        assert payload["timing_path"] == str(out / "timing.json")
        assert payload["summary"]["mean_metrics"]["observed_consistency_l1"] <= 1.0e-6
        assert payload["summary"]["mean_metrics"]["grad_norm"] > 0.0

        assert sorted(path.name for path in out.iterdir()) == [
            "args.json",
            "basic_metrics.json",
            "stage1_pbu_smoke_last.pt",
            "summary.json",
            "timing.json",
            "train_metrics.jsonl",
        ]
        summary = json.loads((out / "summary.json").read_text())
        assert summary["mode"] == "real_data_train_smoke"
        assert summary["steps"] == 2
        assert len((out / "train_metrics.jsonl").read_text().strip().splitlines()) == 2
        basic_metrics = json.loads((out / "basic_metrics.json").read_text())
        timing = json.loads((out / "timing.json").read_text())
        assert basic_metrics["psnr_mean"] == payload["summary"]["mean_metrics"]["psnr"]
        assert basic_metrics["observed_consistency_l1"] <= 1.0e-6
        assert timing["n_frames"] == 2
        assert timing["total_elapsed_s"] >= 0.0
        parsed = summarize_output(out)
        assert parsed["primary"]["psnr_mean"] == basic_metrics["psnr_mean"]
        assert parsed["primary"]["n_frames"] == 2

        after = sorted(str(path.relative_to(root)) for path in root.rglob("*"))
        assert after == before
        repo_after = {
            name: (repo_root / name).stat().st_mtime_ns if (repo_root / name).exists() else None
            for name in OUTPUT_SENTINELS
        }
        assert repo_after == repo_before


def test_stage1_train_cli_train_smoke_mixed_schedule_records_step_metadata():
    repo_root = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        root = base / "data"
        out = base / "stage1_train_smoke_mixed"
        _make_tiny_hdf5(root)
        env = dict(os.environ)
        env["PYTHONPATH"] = str(repo_root / "ulsa")
        result = subprocess.run(
            [
                sys.executable,
                str(repo_root / "ulsa/train_pixel_pbu_stage1.py"),
                "--train_smoke",
                "--output_dir",
                str(out),
                "--save_checkpoint",
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
                "5",
                "--smoke_max_steps_cap",
                "5",
                "--schedule_mode",
                "mixed",
                "--budgets",
                "0,2,4",
                "--mask_families",
                f"{MASK_RANDOM},{MASK_ROLLED_EQUISPACED}",
                "--prior_kinds",
                f"{PRIOR_COPY_LAST},{PRIOR_EMA}",
                "--heavy_metric_interval",
                "2",
                "--n_lines",
                "8",
                "--device",
                "cpu",
            ],
            cwd=repo_root,
            env=env,
            text=True,
            capture_output=True,
            check=True,
        )
        payload = json.loads(result.stdout)
        assert payload["schedule"]["mode"] == "mixed"
        assert payload["schedule"]["budgets"] == [0, 2, 4]
        assert payload["schedule"]["prior_kinds"] == [PRIOR_COPY_LAST, PRIOR_EMA]
        assert payload["budget"] is None
        assert payload["mask_family"] is None
        assert payload["prior_kind"] is None
        assert payload["heavy_metric_steps"] == [0, 1, 3, 4]
        train_rows = [json.loads(line) for line in (out / "train_metrics.jsonl").read_text().splitlines()]
        assert [row["budget"] for row in train_rows] == [0, 2, 4, 0, 2]
        assert {row["mask_family"] for row in train_rows}.issubset({MASK_RANDOM, MASK_ROLLED_EQUISPACED})
        assert train_rows[2]["heavy_metrics"] is False
        assert train_rows[2]["metrics"]["ssim"] is None
        assert train_rows[1]["heavy_metrics"] is True
        assert train_rows[1]["metrics"]["ssim"] is not None


def test_stage1_train_cli_train_smoke_refuses_over_cap_before_writing_output():
    repo_root = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        root = base / "data"
        out = base / "stage1_train_smoke"
        _make_tiny_hdf5(root)
        env = dict(os.environ)
        env["PYTHONPATH"] = str(repo_root / "ulsa")
        result = subprocess.run(
            [
                sys.executable,
                str(repo_root / "ulsa/train_pixel_pbu_stage1.py"),
                "--train_smoke",
                "--output_dir",
                str(out),
                "--data_root",
                str(root),
                "--split",
                "train",
                "--max_files",
                "1",
                "--max_items",
                "2",
                "--max_steps",
                "3",
                "--smoke_max_steps_cap",
                "2",
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
        assert "exceeds smoke cap" in result.stderr
        assert not out.exists()


def test_stage1_train_cli_train_smoke_rejects_checkpoint_name_escape_before_output():
    repo_root = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        root = base / "data"
        out = base / "stage1_train_smoke"
        escaped = base / "escaped.pt"
        _make_tiny_hdf5(root)
        env = dict(os.environ)
        env["PYTHONPATH"] = str(repo_root / "ulsa")
        for bad_name in ["../escaped.pt", str(escaped)]:
            result = subprocess.run(
                [
                    sys.executable,
                    str(repo_root / "ulsa/train_pixel_pbu_stage1.py"),
                    "--train_smoke",
                    "--output_dir",
                    str(out),
                    "--save_checkpoint",
                    "--checkpoint_name",
                    bad_name,
                    "--data_root",
                    str(root),
                    "--split",
                    "train",
                    "--max_files",
                    "1",
                    "--max_items",
                    "2",
                    "--max_steps",
                    "1",
                    "--smoke_max_steps_cap",
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
            assert "checkpoint_name must be a simple filename" in result.stderr
            assert not out.exists()
            assert not escaped.exists()
