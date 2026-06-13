"""Train a standalone Stage2 pixel rectified-flow temporal prior."""

from __future__ import annotations

import argparse
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

import torch
from torch.utils.data import DataLoader

from train_pixel_temporal_prior_stage2 import (  # noqa: E402
    Stage2SequenceDataset,
    _image_reconstruction_loss,
    _load_pbu,
    _mean_metric,
    _prepare_output_dir,
    _resolve_data_root,
    _schedule_budget_family,
    _stage1_batch_from_prior,
    _write_json,
    collate_stage2_sequences,
    make_history_context,
)
from ulsa.pixel_measurement import MeasurementOperator  # noqa: E402
from ulsa.pixel_pbu_stage1 import MASK_RANDOM, VariableMaskBudgetSampler  # noqa: E402
from ulsa.pixel_rectified_flow_prior import StandalonePixelRectifiedFlowPrior  # noqa: E402
from ulsa.pixel_temporal_prior import prior_prediction_metrics  # noqa: E402


def _parse_csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in str(value).split(",") if part.strip())


def _parse_csv_strings(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in str(value).split(",") if part.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train standalone Stage2 pixel rectified-flow prior.")
    parser.add_argument("--pbu_checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--allow_overwrite", action="store_true")
    parser.add_argument("--save_checkpoint", action="store_true")
    parser.add_argument("--checkpoint_name", type=str, default="stage2_pixel_rectified_flow_last.pt")
    parser.add_argument("--data_root", type=str, default="processed_echonet")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--key", type=str, default="data/image_sc")
    parser.add_argument("--max_files", type=int, default=4)
    parser.add_argument("--max_items", type=int, default=1024)
    parser.add_argument("--max_frames_per_file", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--preload_to_memory", action="store_true")
    parser.add_argument("--history_size", type=int, default=3)
    parser.add_argument("--history_mode", type=str, default="mixed", choices=("gt", "corrupt", "pbu", "mixed"))
    parser.add_argument("--rollout_train_length", type=int, default=1)
    parser.add_argument("--rollout_loss_weight", type=float, default=0.75)
    parser.add_argument("--rollout_history_source", type=str, default="pbu", choices=("pbu", "flow_prior", "blend"))
    parser.add_argument("--history_blend_alpha", type=float, default=0.5)
    parser.add_argument("--budgets", type=_parse_csv_ints, default=(14,))
    parser.add_argument("--mask_families", type=_parse_csv_strings, default=("fixed104", MASK_RANDOM))
    parser.add_argument("--n_lines", type=int, default=112)
    parser.add_argument("--seed", type=int, default=913)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--smoke_max_steps_cap", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=2.0e-4)
    parser.add_argument("--weight_decay", type=float, default=1.0e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--flow_steps", type=int, default=2)
    parser.add_argument("--train_noise_scale", type=float, default=0.05)
    parser.add_argument("--start_mode", type=str, default="noisy_last", choices=("last", "noisy_last", "gaussian"))
    parser.add_argument("--flow_loss_weight", type=float, default=1.0)
    parser.add_argument("--image_loss_weight", type=float, default=0.75)
    parser.add_argument("--integrated_loss_weight", type=float, default=0.25)
    parser.add_argument("--pbu_loss_weight", type=float, default=0.5)
    parser.add_argument("--start_anchor_weight", type=float, default=0.005)
    parser.add_argument("--flow_base_channels", type=int, default=32)
    parser.add_argument("--flow_channel_mult", type=_parse_csv_ints, default=(1, 2))
    parser.add_argument("--flow_num_res_blocks", type=int, default=1)
    parser.add_argument("--flow_groupnorm_groups", type=int, default=4)
    parser.add_argument("--velocity_clip_scale", type=float, default=1.25)
    parser.add_argument("--pbu_base_channels", type=int, default=8)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def _make_flow(args: argparse.Namespace, *, device: torch.device) -> StandalonePixelRectifiedFlowPrior:
    return StandalonePixelRectifiedFlowPrior(
        history_size=int(args.history_size),
        base_channels=int(args.flow_base_channels),
        channel_mult=tuple(int(x) for x in args.flow_channel_mult),
        num_res_blocks=int(args.flow_num_res_blocks),
        groupnorm_groups=int(args.flow_groupnorm_groups),
        velocity_clip_scale=float(args.velocity_clip_scale),
    ).to(device=device)


def _validate_schedule_args(args: argparse.Namespace) -> None:
    if not tuple(args.budgets):
        raise ValueError("--budgets must contain at least one budget")
    if not tuple(args.mask_families):
        raise ValueError("--mask_families must contain at least one mask family")


def _flow_matching_loss(pred_v: torch.Tensor, target_v: torch.Tensor) -> torch.Tensor:
    err = pred_v - target_v
    return torch.mean(torch.abs(err)) + 0.25 * torch.mean(err * err)


def _make_start(args: argparse.Namespace, history_x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    last_x = history_x[:, -1]
    mode = str(args.start_mode)
    if mode == "gaussian":
        x0 = torch.randn_like(target).clamp(-1.25, 1.25)
    else:
        x0 = last_x
        if mode == "noisy_last" and float(args.train_noise_scale) > 0.0:
            x0 = x0 + float(args.train_noise_scale) * torch.randn_like(x0)
    return x0.clamp(-1.25, 1.25)


def _one_rollout_step(
    *,
    flow_model: StandalonePixelRectifiedFlowPrior,
    pbu_model,
    sampler: VariableMaskBudgetSampler,
    args: argparse.Namespace,
    history_x: torch.Tensor,
    history_u: torch.Tensor,
    history_mask: torch.Tensor,
    history_selected: torch.Tensor,
    target: torch.Tensor,
    step_idx: int,
    rollout_idx: int,
) -> tuple[
    torch.Tensor,
    dict[str, float | str | None],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    x0 = _make_start(args, history_x, target)
    t = torch.rand((int(target.shape[0]), 1, 1, 1), device=target.device, dtype=target.dtype)
    x_t = ((1.0 - t) * x0 + t * target).clamp(-1.25, 1.25)
    target_v = target - x0
    flow_out = flow_model(
        x_t=x_t,
        t=t,
        x_start=x0,
        history_x=history_x,
        history_u=history_u,
        history_mask=history_mask,
        history_selected=history_selected,
    )
    refined = flow_model.integrate(
        x_start=x0,
        steps=int(args.flow_steps),
        history_x=history_x,
        history_u=history_u,
        history_mask=history_mask,
        history_selected=history_selected,
    )
    budget, family = _schedule_budget_family(args, step_idx + rollout_idx, sampler)
    heuristic = refined.u_prior.detach() + torch.abs(refined.mu_prior.detach() - history_x[:, -1].detach())
    action = sampler.sample_action(
        budget=budget,
        family=family,
        batch_size=int(target.shape[0]),
        heuristic_map=heuristic,
        roll=int(step_idx + rollout_idx),
    )
    measurement = MeasurementOperator().measure(target, action)
    pbu_batch = _stage1_batch_from_prior(
        x_gt=target,
        x_prior=refined.mu_prior,
        u_prior=refined.u_prior,
        prev_x_final=history_x[:, -1].detach(),
        prev_u_post=history_u[:, -1].detach(),
        measurement=measurement,
        budget=budget,
        mask_family=family,
    )
    pbu_out = pbu_model(pbu_batch)
    flow_loss = _flow_matching_loss(flow_out.velocity, target_v)
    image_loss = _image_reconstruction_loss(flow_out.mu_prior, target)
    integrated_loss = _image_reconstruction_loss(refined.mu_prior, target)
    pbu_loss = _image_reconstruction_loss(pbu_out.x_final, target)
    anchor_loss = torch.mean(torch.abs(refined.mu_prior - x0))
    loss = (
        float(args.flow_loss_weight) * flow_loss
        + float(args.image_loss_weight) * image_loss
        + float(args.integrated_loss_weight) * integrated_loss
        + float(args.pbu_loss_weight) * pbu_loss
        + float(args.start_anchor_weight) * anchor_loss
    )
    flow_metrics = prior_prediction_metrics(prediction=refined.mu_prior.detach(), target=target, uncertainty=None)
    pbu_metrics = prior_prediction_metrics(prediction=pbu_out.x_final.detach(), target=target, uncertainty=None)
    start_metrics = prior_prediction_metrics(prediction=x0.detach(), target=target, uncertainty=None)
    metrics: dict[str, float | str | None] = {
        "budget": float(budget),
        "mask_family": str(family),
        "start_psnr": start_metrics["psnr"],
        "flow_psnr": flow_metrics["psnr"],
        "pbu_psnr": pbu_metrics["psnr"],
        "pbu_ssim": pbu_metrics["ssim"],
        "observed_consistency_l1": float(pbu_out.observed_consistency_l1.detach().cpu().item()),
        "flow_loss": float(flow_loss.detach().cpu().item()),
        "image_loss": float(image_loss.detach().cpu().item()),
        "integrated_loss": float(integrated_loss.detach().cpu().item()),
        "pbu_loss": float(pbu_loss.detach().cpu().item()),
        "anchor_loss": float(anchor_loss.detach().cpu().item()),
    }
    return (
        loss,
        metrics,
        pbu_out.x_final.detach(),
        pbu_out.u_post.detach(),
        refined.mu_prior.detach(),
        refined.u_prior.detach(),
        measurement.mask.detach(),
    )


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


def _train_batch(
    *,
    flow_model: StandalonePixelRectifiedFlowPrior,
    pbu_model,
    sampler: VariableMaskBudgetSampler,
    args: argparse.Namespace,
    batch: dict[str, Any],
    step_idx: int,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float | str | None]]:
    history_x, history_u, history_mask, history_selected = make_history_context(
        raw_history=batch["history"],
        sampler=sampler,
        args=args,
        pbu_model=pbu_model,
        step_idx=step_idx,
        mode=str(args.history_mode),
    )
    target_sequence = batch["target_sequence"].to(device=device, dtype=torch.float32)
    rollout_len = max(1, int(args.rollout_train_length))
    if int(target_sequence.shape[1]) < rollout_len:
        raise RuntimeError("target_sequence is shorter than rollout_train_length")
    losses: list[torch.Tensor] = []
    weights: list[float] = []
    metric_rows: list[dict[str, float | str | None]] = []
    for rollout_idx in range(rollout_len):
        target = target_sequence[:, rollout_idx]
        loss, metrics, pbu_x, pbu_u, flow_x, flow_u, next_mask = _one_rollout_step(
            flow_model=flow_model,
            pbu_model=pbu_model,
            sampler=sampler,
            args=args,
            history_x=history_x,
            history_u=history_u,
            history_mask=history_mask,
            history_selected=history_selected,
            target=target,
            step_idx=step_idx,
            rollout_idx=rollout_idx,
        )
        weight = 1.0 if rollout_idx == 0 else float(args.rollout_loss_weight)
        losses.append(float(weight) * loss)
        weights.append(float(weight))
        metric_rows.append(metrics)
        if rollout_idx + 1 < rollout_len:
            if str(args.rollout_history_source) == "flow_prior":
                next_x = flow_x
                next_u = flow_u
            elif str(args.rollout_history_source) == "blend":
                alpha = min(1.0, max(0.0, float(args.history_blend_alpha)))
                next_x = (alpha * pbu_x + (1.0 - alpha) * flow_x).detach()
                next_u = (alpha * pbu_u + (1.0 - alpha) * flow_u).detach()
            else:
                next_x = pbu_x
                next_u = pbu_u
            history_x, history_u, history_mask, history_selected = _append_history(
                history_x,
                history_u,
                history_mask,
                history_selected,
                next_x=next_x,
                next_u=next_u,
                next_mask=next_mask,
            )
    loss = torch.stack(losses).sum() / max(1.0e-12, sum(weights))
    summary: dict[str, float | str | None] = {
        "start_psnr": _mean_metric(metric_rows, "start_psnr"),
        "flow_psnr": _mean_metric(metric_rows, "flow_psnr"),
        "pbu_psnr": _mean_metric(metric_rows, "pbu_psnr"),
        "pbu_ssim": _mean_metric(metric_rows, "pbu_ssim"),
        "observed_consistency_l1": max(float(row["observed_consistency_l1"]) for row in metric_rows),
        "flow_loss": _mean_metric(metric_rows, "flow_loss"),
        "image_loss": _mean_metric(metric_rows, "image_loss"),
        "integrated_loss": _mean_metric(metric_rows, "integrated_loss"),
        "pbu_loss": _mean_metric(metric_rows, "pbu_loss"),
        "anchor_loss": _mean_metric(metric_rows, "anchor_loss"),
    }
    return loss, summary


def run_train(args: argparse.Namespace, *, device: torch.device, output_dir: Path) -> dict[str, Any]:
    if int(args.max_steps) > int(args.smoke_max_steps_cap):
        raise RuntimeError(f"max_steps={args.max_steps} exceeds smoke cap {args.smoke_max_steps_cap}")
    _validate_schedule_args(args)
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
        shuffle=True,
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
    if pbu_model is None:
        raise RuntimeError("standalone pixel-flow training requires --pbu_checkpoint")
    flow_model = _make_flow(args, device=device)
    flow_model.train()
    optimizer = torch.optim.AdamW(
        flow_model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    loader_iter = iter(loader)
    rows: list[dict[str, Any]] = []
    first_step_s = None
    start_s = time.perf_counter()
    log_every = max(0, int(args.log_every))
    with open(output_dir / "train_metrics.jsonl", "w", encoding="utf-8") as metrics_f:
        for step_idx in range(int(args.max_steps)):
            step_start = time.perf_counter()
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                batch = next(loader_iter)
            loss, metrics = _train_batch(
                flow_model=flow_model,
                pbu_model=pbu_model,
                sampler=sampler,
                args=args,
                batch=batch,
                step_idx=step_idx,
                device=device,
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite standalone pixel-flow loss at step {step_idx}: {loss}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(flow_model.parameters(), max_norm=float(args.grad_clip))
            optimizer.step()
            row = {
                "step": int(step_idx),
                "loss": float(loss.detach().cpu().item()),
                "grad_norm": float(grad_norm.detach().cpu().item()) if torch.is_tensor(grad_norm) else float(grad_norm),
                "history_mode": str(args.history_mode),
                "rollout_train_length": int(args.rollout_train_length),
                "rollout_history_source": str(args.rollout_history_source),
                "flow_steps": int(args.flow_steps),
                "start_mode": str(args.start_mode),
                **metrics,
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
                _write_json(
                    output_dir / "progress.json",
                    {
                        "step": int(step_idx + 1),
                        "max_steps": int(args.max_steps),
                        "elapsed_s": float(time.perf_counter() - start_s),
                        "loss": row["loss"],
                        "flow_psnr": row.get("flow_psnr"),
                        "pbu_psnr": row.get("pbu_psnr"),
                        "observed_consistency_l1": row.get("observed_consistency_l1"),
                    },
                )
            if first_step_s is None:
                first_step_s = float(time.perf_counter() - step_start)
    total_elapsed_s = time.perf_counter() - start_s
    checkpoint_path = None
    if bool(args.save_checkpoint):
        checkpoint_path = output_dir / Path(args.checkpoint_name).name
        torch.save(
            {
                "mode": "stage2_standalone_pixel_rectified_flow_train",
                "args": {key: value for key, value in sorted(vars(args).items())},
                "pbu_checkpoint": str(args.pbu_checkpoint),
                "pixel_rectified_flow_prior_state_dict": flow_model.state_dict(),
                "model_state_dict": flow_model.state_dict(),
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
        "note": "Standalone pixel rectified-flow train step timing; includes flow rollout and frozen PBU loss.",
    }
    basic = {
        "psnr_mean": _mean_metric(rows, "pbu_psnr"),
        "flow_prior_psnr_mean": _mean_metric(rows, "flow_psnr"),
        "start_psnr_mean": _mean_metric(rows, "start_psnr"),
        "ssim_mean": _mean_metric(rows, "pbu_ssim"),
        "loss_mean": _mean_metric(rows, "loss"),
        "observed_consistency_l1": max(float(row["observed_consistency_l1"]) for row in rows),
        "note": "Train diagnostics only; full-loop eval is required for method claims.",
    }
    payload = {
        "mode": "stage2_standalone_pixel_rectified_flow_train",
        "data_root": str(args.data_root),
        "split": str(args.split),
        "dataset_items": int(len(dataset)),
        "steps": int(n_steps),
        "history_mode": str(args.history_mode),
        "rollout_train_length": int(args.rollout_train_length),
        "rollout_history_source": str(args.rollout_history_source),
        "flow_steps": int(args.flow_steps),
        "start_mode": str(args.start_mode),
        "pbu_checkpoint": str(args.pbu_checkpoint),
        "checkpoint": str(checkpoint_path) if checkpoint_path else None,
        "basic_metrics_path": str(output_dir / "basic_metrics.json"),
        "timing_path": str(output_dir / "timing.json"),
        "no_base_prior_checkpoint": True,
        "no_nan": True,
    }
    _write_json(output_dir / "basic_metrics.json", basic)
    _write_json(output_dir / "timing.json", timing)
    _write_json(output_dir / "summary.json", payload)
    return payload


def main() -> None:
    args = parse_args()
    device = torch.device("cuda:0" if args.device == "auto" and torch.cuda.is_available() else "cpu")
    if args.device != "auto":
        device = torch.device(args.device)
    torch.manual_seed(int(args.seed))
    output_dir = _prepare_output_dir(args.output_dir, allow_overwrite=bool(args.allow_overwrite))
    payload = run_train(args, device=device, output_dir=output_dir)
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
