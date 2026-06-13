"""Train/evaluate the Stage 2 deterministic pixel temporal prior smoke path."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time
from typing import Any


def _set_backend() -> None:
    os.environ["KERAS_BACKEND"] = "torch"
    os.environ.setdefault("MPLBACKEND", "Agg")


_set_backend()

THIS_DIR = Path(__file__).resolve().parent
ULSA_ROOT = THIS_DIR
REPO_ROOT = ULSA_ROOT.parent
ZEA_TOOL_ROOT = REPO_ROOT / "zea_tool"
for p in [ULSA_ROOT, ZEA_TOOL_ROOT]:
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

import h5py
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from ulsa.pixel_measurement import MeasurementOperator
from ulsa.pixel_belief_state import BudgetState
from ulsa.pixel_pbu_stage1 import (
    MASK_RANDOM,
    PRIOR_COPY_LAST,
    Stage1PBUBatch,
    Stage1PBUWrapper,
    VariableMaskBudgetSampler,
    make_stage1_pbu_batch,
)
from ulsa.pixel_pbu_stage1_dataset import list_stage1_hdf5_files, map_range_tensor
from ulsa.pixel_temporal_prior import (
    ResidualPixelTemporalPrior,
    mean_or_none,
    prior_prediction_metrics,
)


def _parse_csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in str(value).split(",") if part.strip())


def _parse_csv_strings(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in str(value).split(",") if part.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 2 deterministic pixel temporal prior smoke.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--train_smoke", action="store_true")
    mode.add_argument("--eval", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None, help="Stage 2 prior checkpoint for --eval.")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--allow_overwrite", action="store_true")
    parser.add_argument("--save_checkpoint", action="store_true")
    parser.add_argument("--checkpoint_name", type=str, default="stage2_prior_smoke_last.pt")
    parser.add_argument("--data_root", type=str, default="processed_echonet")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--key", type=str, default="data/image_sc")
    parser.add_argument("--max_files", type=int, default=4)
    parser.add_argument("--max_items", type=int, default=256)
    parser.add_argument("--max_frames_per_file", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--preload_to_memory", action="store_true")
    parser.add_argument("--history_size", type=int, default=3)
    parser.add_argument("--history_mode", type=str, default="mixed", choices=("gt", "corrupt", "pbu", "mixed"))
    parser.add_argument("--rollout_train_length", type=int, default=1)
    parser.add_argument("--rollout_history_update", type=str, default="prior", choices=("prior", "pbu"))
    parser.add_argument("--rollout_loss_weight", type=float, default=1.0)
    parser.add_argument(
        "--pbu_conditioned_loss_weight",
        type=float,
        default=0.0,
        help="Additional frozen-PBU output reconstruction loss weight during rollout training.",
    )
    parser.add_argument("--pbu_checkpoint", type=str, default=None)
    parser.add_argument("--pbu_base_channels", type=int, default=8)
    parser.add_argument("--budgets", type=_parse_csv_ints, default=(7, 14, 21))
    parser.add_argument("--mask_families", type=_parse_csv_strings, default=(MASK_RANDOM, "rolled_equispaced", "fixed104"))
    parser.add_argument("--n_lines", type=int, default=112)
    parser.add_argument("--seed", type=int, default=211)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--smoke_max_steps_cap", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=2.0e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=0)
    parser.add_argument(
        "--train_spearman_every",
        type=int,
        default=0,
        help="Compute expensive train-time uncertainty Spearman every N steps; 0 disables it during training.",
    )
    parser.add_argument("--base_channels", type=int, default=16)
    parser.add_argument("--channel_mult", type=_parse_csv_ints, default=(1, 2))
    parser.add_argument("--num_res_blocks", type=int, default=1)
    parser.add_argument("--groupnorm_groups", type=int, default=4)
    parser.add_argument("--delta_clip_scale", type=float, default=0.25)
    parser.add_argument("--prediction_mode", type=str, default="residual", choices=("residual", "direct"))
    parser.add_argument("--eval_rollout_frames", type=int, default=50)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


class Stage2SequenceDataset(Dataset):
    def __init__(
        self,
        data_root: str | Path,
        *,
        split: str,
        key: str = "data/image_sc",
        history_size: int = 3,
        image_range: tuple[float, float] = (-60.0, 0.0),
        output_range: tuple[float, float] = (-1.0, 1.0),
        max_files: int | None = None,
        max_frames_per_file: int | None = None,
        max_items: int | None = None,
        preload_to_memory: bool = False,
        target_horizon: int = 1,
    ) -> None:
        self.data_root = Path(data_root)
        self.split = str(split)
        self.key = str(key)
        self.history_size = max(1, int(history_size))
        self.target_horizon = max(1, int(target_horizon))
        self.image_range = image_range
        self.output_range = output_range
        self.files = list_stage1_hdf5_files(self.data_root, split=self.split)
        if max_files is not None:
            self.files = self.files[: max(0, int(max_files))]
        self.index: list[tuple[Path, int, int]] = []
        for path in self.files:
            with h5py.File(path, "r") as f:
                n_frames = int(f[self.key].shape[0])
            stop = n_frames
            if max_frames_per_file is not None:
                stop = min(stop, self.history_size + max(0, int(max_frames_per_file)))
            target_stop = max(self.history_size, stop - self.target_horizon + 1)
            for frame_idx in range(self.history_size, target_stop):
                self.index.append((path, int(frame_idx), int(n_frames)))
                if max_items is not None and len(self.index) >= int(max_items):
                    break
            if max_items is not None and len(self.index) >= int(max_items):
                break
        if not self.index:
            raise RuntimeError("Stage 2 sequence dataset has no items after filtering")
        self._frame_cache: dict[tuple[str, int], torch.Tensor] = {}
        if bool(preload_to_memory):
            needed: dict[Path, set[int]] = {}
            for path, frame_idx, _n_frames in self.index:
                frames = needed.setdefault(path, set())
                start = int(frame_idx) - self.history_size
                stop = int(frame_idx) + self.target_horizon
                for local_idx in range(start, stop):
                    frames.add(int(local_idx))
            for path, frame_indices in needed.items():
                with h5py.File(path, "r") as f:
                    ds = f[self.key]
                    for frame_idx in sorted(frame_indices):
                        frame = torch.as_tensor(ds[int(frame_idx)], dtype=torch.float32)
                        mapped = map_range_tensor(
                            frame,
                            source_range=self.image_range,
                            target_range=self.output_range,
                        ).unsqueeze(0)
                        self._frame_cache[(str(path), int(frame_idx))] = mapped

    def __len__(self) -> int:
        return len(self.index)

    def _read_frame(self, path: Path, frame_idx: int) -> torch.Tensor:
        cached = self._frame_cache.get((str(path), int(frame_idx)))
        if cached is not None:
            return cached.clone()
        with h5py.File(path, "r") as f:
            frame = torch.as_tensor(f[self.key][int(frame_idx)], dtype=torch.float32)
        return map_range_tensor(frame, source_range=self.image_range, target_range=self.output_range).unsqueeze(0)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        path, frame_idx, n_frames = self.index[int(idx)]
        history = [self._read_frame(path, frame_idx - self.history_size + offset) for offset in range(self.history_size)]
        target = self._read_frame(path, frame_idx)
        target_sequence = [self._read_frame(path, frame_idx + offset) for offset in range(self.target_horizon)]
        return {
            "history": torch.stack(history, dim=0),
            "target": target,
            "target_sequence": torch.stack(target_sequence, dim=0),
            "file_path": str(path),
            "frame_idx": int(frame_idx),
            "n_frames": int(n_frames),
            "split": self.split,
        }


def collate_stage2_sequences(samples: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "history": torch.stack([sample["history"] for sample in samples], dim=0),
        "target": torch.stack([sample["target"] for sample in samples], dim=0),
        "target_sequence": torch.stack([sample["target_sequence"] for sample in samples], dim=0),
        "file_path": [str(sample["file_path"]) for sample in samples],
        "frame_idx": [int(sample["frame_idx"]) for sample in samples],
        "n_frames": [int(sample["n_frames"]) for sample in samples],
        "split": [str(sample["split"]) for sample in samples],
    }


def _resolve_data_root(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    repo_path = REPO_ROOT / path
    return repo_path if repo_path.exists() else path


def _prepare_output_dir(value: str, *, allow_overwrite: bool) -> Path:
    path = Path(value)
    if path.exists() and any(path.iterdir()) and not bool(allow_overwrite):
        raise RuntimeError(f"Refusing to write Stage 2 outputs into non-empty directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, allow_nan=False)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _make_prior_model(args: argparse.Namespace, *, device: torch.device) -> ResidualPixelTemporalPrior:
    return ResidualPixelTemporalPrior(
        history_size=int(args.history_size),
        base_channels=int(args.base_channels),
        channel_mult=tuple(int(x) for x in args.channel_mult),
        num_res_blocks=int(args.num_res_blocks),
        groupnorm_groups=int(args.groupnorm_groups),
        delta_clip_scale=float(args.delta_clip_scale),
        prediction_mode=str(getattr(args, "prediction_mode", "residual")),
    ).to(device=device)


def _load_prior_checkpoint(model: ResidualPixelTemporalPrior, checkpoint: str) -> dict[str, Any]:
    ckpt = torch.load(checkpoint, map_location="cpu")
    if isinstance(ckpt, dict) and "stage2_prior_state_dict" in ckpt:
        model.load_state_dict(ckpt["stage2_prior_state_dict"], strict=True)
        return ckpt
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        return ckpt
    model.load_state_dict(ckpt, strict=True)
    return {"checkpoint": str(checkpoint)}


def _load_pbu(args: argparse.Namespace, *, device: torch.device) -> Stage1PBUWrapper | None:
    if args.pbu_checkpoint is None:
        return None
    model = Stage1PBUWrapper(
        n_lines=int(args.n_lines),
        base_channels=int(args.pbu_base_channels),
        channel_mult=(1, 2),
        num_res_blocks=1,
        groupnorm_groups=4,
        delta_clip_scale=0.05,
    ).to(device=device)
    ckpt = torch.load(args.pbu_checkpoint, map_location="cpu")
    if isinstance(ckpt, dict) and "stage1_pbu_wrapper_state_dict" in ckpt:
        model.load_state_dict(ckpt["stage1_pbu_wrapper_state_dict"], strict=True)
    elif isinstance(ckpt, dict) and "pixel_belief_updater_state_dict" in ckpt:
        model.model.load_state_dict(ckpt["pixel_belief_updater_state_dict"], strict=True)
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.model.load_state_dict(ckpt["model_state_dict"], strict=True)
    else:
        model.load_state_dict(ckpt, strict=True)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def _schedule_budget_family(args: argparse.Namespace, step_idx: int, sampler: VariableMaskBudgetSampler) -> tuple[int, str]:
    budgets = [int(x) for x in args.budgets]
    families = [str(x) for x in args.mask_families]
    budget = budgets[int(step_idx) % len(budgets)]
    compatible = [family for family in families if family in sampler.compatible_families_for_budget(budget)]
    if not compatible:
        compatible = [MASK_RANDOM]
    family = compatible[(int(step_idx) // max(1, len(budgets))) % len(compatible)]
    return int(budget), str(family)


def _weak_fill(target: torch.Tensor, mask: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
    zero_fill = torch.where(mask > 0, obs, torch.zeros_like(obs))
    blurred = F.avg_pool2d(zero_fill, kernel_size=5, stride=1, padding=2)
    return torch.where(mask > 0, obs, blurred)


def _stage1_batch_from_prior(
    *,
    x_gt: torch.Tensor,
    x_prior: torch.Tensor,
    u_prior: torch.Tensor,
    prev_x_final: torch.Tensor,
    prev_u_post: torch.Tensor,
    measurement,
    budget: int,
    mask_family: str,
) -> Stage1PBUBatch:
    return Stage1PBUBatch(
        x_gt=x_gt,
        x_prior=x_prior,
        u_prior=u_prior,
        obs=measurement.obs,
        mask=measurement.mask,
        selected_lines=measurement.action.selected_lines,
        line_history=measurement.action.selected_lines.to(dtype=x_gt.dtype),
        budget=BudgetState(total=int(budget), used=int(budget), phase=0, max_updates=1),
        prev_x_final=prev_x_final,
        prev_u_post=prev_u_post,
        change_map=torch.abs(x_prior - prev_x_final),
        mask_family=str(mask_family),
        prior_kind="stage2_prior_rollout_train",
    )


def make_history_context(
    *,
    raw_history: torch.Tensor,
    sampler: VariableMaskBudgetSampler,
    args: argparse.Namespace,
    pbu_model: Stage1PBUWrapper | None,
    step_idx: int,
    mode: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    raw_history = raw_history.to(device=sampler.generator.device, dtype=torch.float32)
    b, k, _c, h, w = raw_history.shape
    if mode == "mixed":
        choices = ("gt", "corrupt", "pbu" if pbu_model is not None else "corrupt")
        mode = choices[int(step_idx) % len(choices)]
    history_x = []
    history_u = []
    history_mask = []
    history_selected = []
    prev_x = raw_history[:, 0]
    for hist_idx in range(k):
        target = raw_history[:, hist_idx]
        if mode == "gt":
            x = target
            u = torch.zeros_like(target)
            mask = torch.ones_like(target)
        else:
            budget, family = _schedule_budget_family(args, step_idx + hist_idx, sampler)
            action = sampler.sample_action(
                budget=budget,
                family=family,
                batch_size=b,
                heuristic_map=torch.abs(target - prev_x),
                roll=int(step_idx + hist_idx),
            )
            measurement = MeasurementOperator().measure(target, action)
            if mode == "pbu" and pbu_model is not None:
                batch = make_stage1_pbu_batch(
                    target=target,
                    sampler=sampler,
                    budget=budget,
                    mask_family=family,
                    prior_kind=PRIOR_COPY_LAST,
                    prev_x_final=prev_x,
                    prev_prev_x_final=prev_x,
                    heuristic_map=torch.abs(target - prev_x),
                    roll=int(step_idx + hist_idx),
                )
                with torch.no_grad():
                    out = pbu_model(batch)
                x = out.x_final.detach()
                u = out.u_post.detach()
                mask = batch.mask
            else:
                x = _weak_fill(target, measurement.mask, measurement.obs)
                u = (1.0 - measurement.mask).detach()
                mask = measurement.mask
        history_x.append(x)
        history_u.append(u)
        history_mask.append(mask)
        history_selected.append(mask)
        prev_x = x.detach()
    return (
        torch.stack(history_x, dim=1),
        torch.stack(history_u, dim=1),
        torch.stack(history_mask, dim=1),
        torch.stack(history_selected, dim=1),
    )


def _training_loss(output, target: torch.Tensor) -> torch.Tensor:
    err = output.mu_prior - target
    l1 = torch.mean(torch.abs(err))
    mse = torch.mean(err * err)
    inv_var = torch.exp(-output.logvar).clamp_max(100.0)
    nll = torch.mean(inv_var * (err.detach() ** 2) + output.logvar)
    return l1 + 0.25 * mse + 0.02 * nll


def _image_reconstruction_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    err = prediction - target
    l1 = torch.mean(torch.abs(err))
    mse = torch.mean(err * err)
    return l1 + 0.25 * mse


def _append_history(
    history_x: torch.Tensor,
    history_u: torch.Tensor,
    history_mask: torch.Tensor,
    history_selected: torch.Tensor,
    *,
    next_x: torch.Tensor,
    next_u: torch.Tensor,
    next_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.cat([history_x[:, 1:], next_x[:, None]], dim=1),
        torch.cat([history_u[:, 1:], next_u[:, None]], dim=1),
        torch.cat([history_mask[:, 1:], next_mask[:, None]], dim=1),
        torch.cat([history_selected[:, 1:], next_mask[:, None]], dim=1),
    )


def _rollout_training_forward(
    *,
    model: ResidualPixelTemporalPrior,
    pbu_model: Stage1PBUWrapper | None,
    sampler: VariableMaskBudgetSampler,
    args: argparse.Namespace,
    history_x: torch.Tensor,
    history_u: torch.Tensor,
    history_mask: torch.Tensor,
    history_selected: torch.Tensor,
    target_sequence: torch.Tensor,
    step_idx: int,
) -> tuple[torch.Tensor, dict[str, float | None]]:
    rollout_len = max(1, int(args.rollout_train_length))
    if int(target_sequence.shape[1]) < rollout_len:
        raise ValueError(
            f"target_sequence length {int(target_sequence.shape[1])} is shorter than rollout_train_length={rollout_len}"
        )
    if rollout_len > 1 and str(args.rollout_history_update) == "pbu" and pbu_model is None:
        raise RuntimeError("rollout_history_update=pbu requires --pbu_checkpoint")
    measurement_op = MeasurementOperator()
    loss_weight = float(args.rollout_loss_weight)
    pbu_loss_weight = max(0.0, float(args.pbu_conditioned_loss_weight))
    if pbu_loss_weight > 0.0 and pbu_model is None:
        raise RuntimeError("pbu_conditioned_loss_weight > 0 requires --pbu_checkpoint")
    weighted_terms: list[torch.Tensor] = []
    weight_sum = 0.0
    metric_rows: list[dict[str, float | None]] = []
    pbu_metric_rows: list[dict[str, float | None]] = []
    for rollout_idx in range(rollout_len):
        target_t = target_sequence[:, rollout_idx].to(device=history_x.device, dtype=history_x.dtype)
        output = model(
            history_x=history_x,
            history_u=history_u,
            history_mask=history_mask,
            history_selected=history_selected,
        )
        weight = 1.0 if rollout_idx == 0 else loss_weight
        weighted_terms.append(float(weight) * _training_loss(output, target_t))
        weight_sum += float(weight)
        metric_rows.append(
            prior_prediction_metrics(
                prediction=output.mu_prior.detach(),
                target=target_t,
                uncertainty=None,
            )
        )
        needs_pbu = pbu_loss_weight > 0.0 or (
            rollout_idx + 1 < rollout_len and str(args.rollout_history_update) == "pbu"
        )
        pbu_out = None
        measurement = None
        if needs_pbu:
            budget, family = _schedule_budget_family(args, step_idx + rollout_idx, sampler)
            action = sampler.sample_action(
                budget=budget,
                family=family,
                batch_size=int(target_t.shape[0]),
                heuristic_map=output.u_prior.detach() + torch.abs(output.mu_prior.detach() - history_x[:, -1]),
                roll=int(step_idx + rollout_idx),
            )
            measurement = measurement_op.measure(target_t, action)
            pbu_batch = _stage1_batch_from_prior(
                x_gt=target_t,
                x_prior=output.mu_prior if pbu_loss_weight > 0.0 else output.mu_prior.detach(),
                u_prior=output.u_prior if pbu_loss_weight > 0.0 else output.u_prior.detach(),
                prev_x_final=history_x[:, -1].detach(),
                prev_u_post=history_u[:, -1].detach(),
                measurement=measurement,
                budget=budget,
                mask_family=family,
            )
            if pbu_loss_weight > 0.0:
                pbu_out = pbu_model(pbu_batch) if pbu_model is not None else None
                if pbu_out is not None:
                    weighted_terms.append(
                        float(weight) * pbu_loss_weight * _image_reconstruction_loss(pbu_out.x_final, target_t)
                    )
                    weight_sum += float(weight) * pbu_loss_weight
                    pbu_metric_rows.append(
                        prior_prediction_metrics(
                            prediction=pbu_out.x_final.detach(),
                            target=target_t,
                            uncertainty=None,
                        )
                    )
            else:
                with torch.no_grad():
                    pbu_out = pbu_model(pbu_batch) if pbu_model is not None else None
        if rollout_idx + 1 >= rollout_len:
            continue
        if str(args.rollout_history_update) == "pbu":
            if pbu_out is None:
                raise RuntimeError("PBU rollout update failed to produce output")
            next_x = pbu_out.x_final.detach()
            next_u = pbu_out.u_post.detach()
            if measurement is None:
                raise RuntimeError("PBU rollout update has no measurement")
            next_mask = measurement.mask.detach()
        else:
            next_x = output.mu_prior.detach()
            next_u = output.u_prior.detach()
            next_mask = torch.ones_like(next_x)
        history_x, history_u, history_mask, history_selected = _append_history(
            history_x,
            history_u,
            history_mask,
            history_selected,
            next_x=next_x,
            next_u=next_u,
            next_mask=next_mask,
        )
    loss = torch.stack(weighted_terms).sum() / max(1.0e-12, weight_sum)
    summary = {
        "psnr": _mean_metric(metric_rows, "psnr"),
        "ssim": _mean_metric(metric_rows, "ssim"),
        "mae": _mean_metric(metric_rows, "mae"),
        "mse": _mean_metric(metric_rows, "mse"),
        "pbu_psnr": _mean_metric(pbu_metric_rows, "psnr"),
        "pbu_ssim": _mean_metric(pbu_metric_rows, "ssim"),
        "uncertainty_error_spearman": None,
    }
    return loss, summary


def _mean_metric(rows: list[dict[str, Any]], key: str) -> float | None:
    return mean_or_none([row.get(key) for row in rows])


def run_train(args: argparse.Namespace, *, device: torch.device, output_dir: Path) -> dict[str, Any]:
    if int(args.max_steps) > int(args.smoke_max_steps_cap):
        raise RuntimeError(f"max_steps={args.max_steps} exceeds smoke cap {args.smoke_max_steps_cap}")
    dataset = Stage2SequenceDataset(
        _resolve_data_root(args.data_root),
        split=str(args.split),
        key=str(args.key),
        history_size=int(args.history_size),
        max_files=int(args.max_files),
        max_items=int(args.max_items),
        max_frames_per_file=int(args.max_frames_per_file),
        preload_to_memory=bool(args.preload_to_memory),
        target_horizon=max(1, int(args.rollout_train_length)),
    )
    num_workers = max(0, int(args.num_workers))
    loader = DataLoader(
        dataset,
        batch_size=max(1, int(args.batch_size)),
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_stage2_sequences,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )
    first = dataset[0]["target"]
    sampler = VariableMaskBudgetSampler(
        image_shape=(int(first.shape[-2]), int(first.shape[-1])),
        n_lines=int(args.n_lines),
        device=device,
        dtype=torch.float32,
        seed=int(args.seed),
    )
    pbu_model = _load_pbu(args, device=device)
    model = _make_prior_model(args, device=device)
    init_ckpt = None
    if args.checkpoint is not None:
        init_ckpt = _load_prior_checkpoint(model, args.checkpoint)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr))
    rows: list[dict[str, Any]] = []
    first_step_s = None
    start_s = time.perf_counter()
    loader_iter = iter(loader)
    progress_path = output_dir / "progress.json"
    log_every = max(0, int(args.log_every))
    with open(output_dir / "train_metrics.jsonl", "w", encoding="utf-8") as metrics_f:
        for step_idx in range(int(args.max_steps)):
            step_start = time.perf_counter()
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                batch = next(loader_iter)
            target = batch["target"].to(device=device, dtype=torch.float32)
            target_sequence = batch["target_sequence"].to(device=device, dtype=torch.float32)
            history_x, history_u, history_mask, history_selected = make_history_context(
                raw_history=batch["history"],
                sampler=sampler,
                args=args,
                pbu_model=pbu_model,
                step_idx=step_idx,
                mode=str(args.history_mode),
            )
            if int(args.rollout_train_length) > 1:
                loss, metrics = _rollout_training_forward(
                    model=model,
                    pbu_model=pbu_model,
                    sampler=sampler,
                    args=args,
                    history_x=history_x,
                    history_u=history_u,
                    history_mask=history_mask,
                    history_selected=history_selected,
                    target_sequence=target_sequence,
                    step_idx=step_idx,
                )
            else:
                output = model(
                    history_x=history_x,
                    history_u=history_u,
                    history_mask=history_mask,
                    history_selected=history_selected,
                )
                loss = _training_loss(output, target)
                spearman_every = max(0, int(args.train_spearman_every))
                compute_train_spearman = spearman_every > 0 and (
                    step_idx == 0
                    or step_idx + 1 == int(args.max_steps)
                    or (step_idx + 1) % spearman_every == 0
                )
                metrics = prior_prediction_metrics(
                    prediction=output.mu_prior.detach(),
                    target=target,
                    uncertainty=(output.u_prior.detach() if compute_train_spearman else None),
                )
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite Stage 2 training loss at step {step_idx}: {loss}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.grad_clip))
            optimizer.step()
            row = {
                "step": int(step_idx),
                "loss": float(loss.detach().cpu().item()),
                "grad_norm": float(grad_norm.detach().cpu().item()) if torch.is_tensor(grad_norm) else float(grad_norm),
                "rollout_train_length": int(args.rollout_train_length),
                "rollout_history_update": str(args.rollout_history_update),
                "pbu_conditioned_loss_weight": float(args.pbu_conditioned_loss_weight),
                "prediction_mode": str(args.prediction_mode),
                **{f"prior_{k}": v for k, v in metrics.items()},
            }
            rows.append(row)
            metrics_f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            should_flush = (
                step_idx == 0
                or step_idx + 1 == int(args.max_steps)
                or (log_every > 0 and (step_idx + 1) % log_every == 0)
            )
            if should_flush:
                metrics_f.flush()
                progress = {
                    "step": int(step_idx + 1),
                    "max_steps": int(args.max_steps),
                    "elapsed_s": float(time.perf_counter() - start_s),
                    "loss": row["loss"],
                    "prior_psnr": row.get("prior_psnr"),
                    "history_mode": str(args.history_mode),
                    "dataset_items": int(len(dataset)),
                    "batch_size": int(args.batch_size),
                    "preload_to_memory": bool(args.preload_to_memory),
                    "num_workers": int(args.num_workers),
                    "train_spearman_every": int(args.train_spearman_every),
                    "rollout_train_length": int(args.rollout_train_length),
                    "rollout_history_update": str(args.rollout_history_update),
                    "rollout_loss_weight": float(args.rollout_loss_weight),
                    "pbu_conditioned_loss_weight": float(args.pbu_conditioned_loss_weight),
                    "prediction_mode": str(args.prediction_mode),
                }
                _write_json(progress_path, progress)
            if first_step_s is None:
                first_step_s = float(time.perf_counter() - step_start)
    total_elapsed_s = time.perf_counter() - start_s
    checkpoint_path = None
    if bool(args.save_checkpoint):
        checkpoint_path = output_dir / Path(args.checkpoint_name).name
        torch.save(
            {
                "mode": "stage2_prior_train_smoke",
                "args": {key: value for key, value in sorted(vars(args).items())},
                "stage2_prior_state_dict": model.state_dict(),
                "model_state_dict": model.state_dict(),
            },
            checkpoint_path,
        )
    n_steps = len(rows)
    steady_elapsed = total_elapsed_s - (first_step_s or 0.0)
    timing = {
        "steady_fps_excl_first": (float((n_steps - 1) / steady_elapsed) if n_steps > 1 and steady_elapsed > 0 else None),
        "total_fps": float(n_steps / total_elapsed_s) if total_elapsed_s > 0 else None,
        "first_frame_s": first_step_s,
        "n_frames": int(n_steps),
        "total_elapsed_s": float(total_elapsed_s),
        "note": "Stage 2 train-smoke step timing, not inference-only prior timing.",
    }
    basic = {
        "psnr_mean": _mean_metric(rows, "prior_psnr"),
        "ssim_mean": _mean_metric(rows, "prior_ssim"),
        "mae_mean": _mean_metric(rows, "prior_mae"),
        "loss_mean": _mean_metric(rows, "loss"),
        "grad_norm_mean": _mean_metric(rows, "grad_norm"),
        "uncertainty_error_spearman": _mean_metric(rows, "prior_uncertainty_error_spearman"),
        "observed_consistency_l1": None,
        "note": "Stage 2 prior train-smoke diagnostics only; Stage 2 velocity decision uses eval/rollout records.",
    }
    payload = {
        "mode": "stage2_prior_train_smoke",
        "data_root": str(args.data_root),
        "split": str(args.split),
        "dataset_items": int(len(dataset)),
        "steps": int(n_steps),
        "history_mode": str(args.history_mode),
        "preload_to_memory": bool(args.preload_to_memory),
        "num_workers": int(args.num_workers),
        "train_spearman_every": int(args.train_spearman_every),
        "rollout_train_length": int(args.rollout_train_length),
        "rollout_history_update": str(args.rollout_history_update),
        "rollout_loss_weight": float(args.rollout_loss_weight),
        "pbu_conditioned_loss_weight": float(args.pbu_conditioned_loss_weight),
        "prediction_mode": str(args.prediction_mode),
        "pbu_checkpoint": str(args.pbu_checkpoint) if args.pbu_checkpoint else None,
        "initialized_from_checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "initialized_checkpoint_keys": sorted(init_ckpt.keys()) if isinstance(init_ckpt, dict) else None,
        "checkpoint": str(checkpoint_path) if checkpoint_path else None,
        "basic_metrics_path": str(output_dir / "basic_metrics.json"),
        "timing_path": str(output_dir / "timing.json"),
        "no_nan": True,
    }
    _write_json(output_dir / "basic_metrics.json", basic)
    _write_json(output_dir / "timing.json", timing)
    _write_json(output_dir / "summary.json", payload)
    return payload


def _eval_one_step(
    *,
    model: ResidualPixelTemporalPrior,
    args: argparse.Namespace,
    device: torch.device,
    output_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    dataset = Stage2SequenceDataset(
        _resolve_data_root(args.data_root),
        split=str(args.split),
        key=str(args.key),
        history_size=int(args.history_size),
        max_files=int(args.max_files),
        max_items=int(args.max_items),
        max_frames_per_file=int(args.max_frames_per_file),
    )
    loader = DataLoader(dataset, batch_size=max(1, int(args.batch_size)), shuffle=False, num_workers=0, collate_fn=collate_stage2_sequences)
    first = dataset[0]["target"]
    sampler = VariableMaskBudgetSampler(
        image_shape=(int(first.shape[-2]), int(first.shape[-1])),
        n_lines=int(args.n_lines),
        device=device,
        dtype=torch.float32,
        seed=int(args.seed),
    )
    pbu_model = _load_pbu(args, device=device)
    rows: list[dict[str, Any]] = []
    first_forward_s = None
    forward_elapsed_s = 0.0
    for batch_idx, batch in enumerate(loader):
        target = batch["target"].to(device=device, dtype=torch.float32)
        history_x, history_u, history_mask, history_selected = make_history_context(
            raw_history=batch["history"],
            sampler=sampler,
            args=args,
            pbu_model=pbu_model,
            step_idx=batch_idx,
            mode=str(args.history_mode),
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start_s = time.perf_counter()
        with torch.no_grad():
            output = model(
                history_x=history_x,
                history_u=history_u,
                history_mask=history_mask,
                history_selected=history_selected,
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start_s
        forward_elapsed_s += elapsed
        if first_forward_s is None:
            first_forward_s = float(elapsed)
        copy_last = history_x[:, -1]
        ema = 0.8 * history_x[:, -1] + 0.2 * history_x[:, -2]
        prior_metrics = prior_prediction_metrics(prediction=output.mu_prior, target=target, uncertainty=output.u_prior)
        copy_metrics = prior_prediction_metrics(prediction=copy_last, target=target)
        ema_metrics = prior_prediction_metrics(prediction=ema, target=target)
        rows.append(
            {
                "batch_idx": int(batch_idx),
                "n_frames": int(target.shape[0]),
                **{f"prior_{k}": v for k, v in prior_metrics.items()},
                **{f"copy_last_{k}": v for k, v in copy_metrics.items()},
                **{f"ema_{k}": v for k, v in ema_metrics.items()},
            }
        )
    n_frames = sum(int(row["n_frames"]) for row in rows)
    steady_elapsed = forward_elapsed_s - (first_forward_s or 0.0)
    timing = {
        "steady_fps_excl_first": (
            float((n_frames - int(rows[0]["n_frames"])) / steady_elapsed)
            if len(rows) > 1 and steady_elapsed > 0.0
            else None
        ),
        "total_fps": float(n_frames / forward_elapsed_s) if forward_elapsed_s > 0 else None,
        "first_frame_s": (float(first_forward_s / max(1, int(rows[0]["n_frames"]))) if rows else None),
        "n_frames": int(n_frames),
        "total_elapsed_s": float(forward_elapsed_s),
        "note": "Stage 2 prior model forward-only timing on eval batches; excludes data loading/history synthesis/PBU history synthesis.",
    }
    summary = {
        "dataset_items": int(len(dataset)),
        "num_batches": int(len(rows)),
        "history_mode": str(args.history_mode),
        "prior_psnr_mean": _mean_metric(rows, "prior_psnr"),
        "prior_ssim_mean": _mean_metric(rows, "prior_ssim"),
        "prior_mae_mean": _mean_metric(rows, "prior_mae"),
        "copy_last_psnr_mean": _mean_metric(rows, "copy_last_psnr"),
        "copy_last_ssim_mean": _mean_metric(rows, "copy_last_ssim"),
        "ema_psnr_mean": _mean_metric(rows, "ema_psnr"),
        "ema_ssim_mean": _mean_metric(rows, "ema_ssim"),
        "uncertainty_error_spearman": _mean_metric(rows, "prior_uncertainty_error_spearman"),
        "no_nan": True,
        "timing": timing,
    }
    _write_csv(output_dir / "one_step_eval.csv", rows)
    return summary, rows


def _eval_recursive_rollout(
    *,
    model: ResidualPixelTemporalPrior,
    args: argparse.Namespace,
    device: torch.device,
    output_dir: Path,
) -> dict[str, Any]:
    files = list_stage1_hdf5_files(_resolve_data_root(args.data_root), split=str(args.split))[:1]
    if not files:
        return {"enabled": False, "reason": "no files"}
    path = files[0]
    with h5py.File(path, "r") as f:
        data = torch.as_tensor(f[str(args.key)][:], dtype=torch.float32)
    data = map_range_tensor(data, source_range=(-60.0, 0.0), target_range=(-1.0, 1.0)).unsqueeze(1)
    h = int(args.history_size)
    max_targets = min(int(args.eval_rollout_frames), int(data.shape[0]) - h)
    if max_targets <= 0:
        return {"enabled": False, "reason": "not enough frames"}
    history = data[:h].to(device=device)
    recursive_history = history.clone()
    rows: list[dict[str, Any]] = []
    for offset in range(max_targets):
        target = data[h + offset : h + offset + 1].to(device=device)
        teacher_history = data[offset : offset + h].to(device=device).unsqueeze(0)
        recursive_input = recursive_history[-h:].unsqueeze(0)
        zeros = torch.zeros_like(teacher_history)
        ones = torch.ones_like(teacher_history)
        with torch.no_grad():
            teacher = model(history_x=teacher_history, history_u=zeros, history_mask=ones, history_selected=ones)
            recursive = model(
                history_x=recursive_input,
                history_u=torch.zeros_like(recursive_input),
                history_mask=torch.ones_like(recursive_input),
                history_selected=torch.ones_like(recursive_input),
            )
        teacher_metrics = prior_prediction_metrics(prediction=teacher.mu_prior, target=target, uncertainty=teacher.u_prior)
        recursive_metrics = prior_prediction_metrics(prediction=recursive.mu_prior, target=target, uncertainty=recursive.u_prior)
        rows.append(
            {
                "frame_offset": int(offset + 1),
                "teacher_psnr": teacher_metrics["psnr"],
                "recursive_psnr": recursive_metrics["psnr"],
                "teacher_ssim": teacher_metrics["ssim"],
                "recursive_ssim": recursive_metrics["ssim"],
                "psnr_drop_vs_teacher": float(teacher_metrics["psnr"] - recursive_metrics["psnr"]),
            }
        )
        recursive_history = torch.cat([recursive_history, recursive.mu_prior.detach()], dim=0)
    first_bad = None
    for row in rows:
        if float(row["psnr_drop_vs_teacher"]) > 1.0:
            first_bad = int(row["frame_offset"])
            break
    _write_csv(output_dir / "recursive_prior_rollout.csv", rows)
    return {
        "enabled": True,
        "file_path": str(path),
        "frames": int(len(rows)),
        "teacher_psnr_mean": _mean_metric(rows, "teacher_psnr"),
        "recursive_psnr_mean": _mean_metric(rows, "recursive_psnr"),
        "mean_psnr_drop_vs_teacher": _mean_metric(rows, "psnr_drop_vs_teacher"),
        "first_bad_frame_drop_gt_1db": first_bad,
        "first_10_recursive_psnr": _mean_metric(rows[:10], "recursive_psnr"),
        "last_10_recursive_psnr": _mean_metric(rows[-10:], "recursive_psnr"),
    }


def run_eval(args: argparse.Namespace, *, device: torch.device, output_dir: Path) -> dict[str, Any]:
    if args.checkpoint is None:
        raise RuntimeError("--eval requires --checkpoint")
    model = _make_prior_model(args, device=device)
    ckpt = _load_prior_checkpoint(model, args.checkpoint)
    model.eval()
    one_step_summary, _rows = _eval_one_step(model=model, args=args, device=device, output_dir=output_dir)
    rollout_summary = _eval_recursive_rollout(model=model, args=args, device=device, output_dir=output_dir)
    basic = {
        "psnr_mean": one_step_summary.get("prior_psnr_mean"),
        "ssim_mean": one_step_summary.get("prior_ssim_mean"),
        "copy_last_psnr_mean": one_step_summary.get("copy_last_psnr_mean"),
        "ema_psnr_mean": one_step_summary.get("ema_psnr_mean"),
        "uncertainty_error_spearman": one_step_summary.get("uncertainty_error_spearman"),
        "recursive_prior_psnr_mean": rollout_summary.get("recursive_psnr_mean"),
        "recursive_prior_mean_drop_vs_teacher": rollout_summary.get("mean_psnr_drop_vs_teacher"),
        "observed_consistency_l1": None,
        "note": "Stage 2 prior eval metrics; observed consistency is checked in Stage 2.5/3 PBU projection runs.",
    }
    payload = {
        "mode": "stage2_prior_eval",
        "checkpoint": str(args.checkpoint),
        "checkpoint_reloaded": True,
        "checkpoint_keys": sorted(ckpt.keys()) if isinstance(ckpt, dict) else None,
        "data_root": str(args.data_root),
        "split": str(args.split),
        "history_mode": str(args.history_mode),
        "pbu_checkpoint": str(args.pbu_checkpoint) if args.pbu_checkpoint else None,
        "one_step": one_step_summary,
        "recursive_rollout": rollout_summary,
        "no_nan": True,
        "basic_metrics_path": str(output_dir / "basic_metrics.json"),
        "timing_path": str(output_dir / "timing.json"),
    }
    _write_json(output_dir / "basic_metrics.json", basic)
    _write_json(output_dir / "timing.json", one_step_summary["timing"])
    _write_json(output_dir / "summary.json", payload)
    return payload


def main() -> None:
    args = parse_args()
    device = torch.device("cuda:0" if args.device == "auto" and torch.cuda.is_available() else "cpu")
    if args.device != "auto":
        device = torch.device(args.device)
    torch.manual_seed(int(args.seed))
    output_dir = _prepare_output_dir(args.output_dir, allow_overwrite=bool(args.allow_overwrite))
    if bool(args.train_smoke):
        payload = run_train(args, device=device, output_dir=output_dir)
    else:
        payload = run_eval(args, device=device, output_dir=output_dir)
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
