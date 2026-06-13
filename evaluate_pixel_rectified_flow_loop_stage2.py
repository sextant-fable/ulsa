"""Full-loop evaluator for standalone Stage2 pixel rectified-flow prior."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
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

from train_pixel_rectified_flow_prior_stage2 import _make_flow  # noqa: E402
from train_pixel_temporal_prior_stage2 import (  # noqa: E402
    _load_pbu,
    _prepare_output_dir,
    _resolve_data_root,
    _stage1_batch_from_prior,
    _write_json,
)
from ulsa.pixel_measurement import MeasurementOperator  # noqa: E402
from ulsa.pixel_acquisition_stage3 import (  # noqa: E402
    action_from_heuristic_scores,
    edge_magnitude,
    heuristic_line_scores,
    normalize_line_scores,
    rbf_greedy_topk,
    selected_indices_from_mask,
)
from ulsa.pixel_pbu_stage1 import MASK_HEURISTIC, VariableMaskBudgetSampler  # noqa: E402
from ulsa.pixel_pbu_stage1_dataset import list_stage1_hdf5_files, map_range_tensor  # noqa: E402
from ulsa.pixel_rectified_flow_prior import StandalonePixelRectifiedFlowPrior  # noqa: E402
from ulsa.pixel_temporal_prior import mean_or_none, prior_prediction_metrics  # noqa: E402
from ulsa.pixel_timing import TimingProfiler  # noqa: E402


def _parse_csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in str(value).split(",") if part.strip())


def _parse_csv_strings(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in str(value).split(",") if part.strip())


def _parse_float_weights(value: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for part in str(value).split(","):
        part = part.strip()
        if not part:
            continue
        key, raw = part.split("=", 1)
        out[str(key).strip()] = float(raw)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate standalone Stage2 pixel rectified-flow full-loop rollout.")
    parser.add_argument("--flow_checkpoint", type=str, required=True)
    parser.add_argument("--pbu_checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--allow_overwrite", action="store_true")
    parser.add_argument("--data_root", type=str, default="processed_echonet")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--key", type=str, default="data/image_sc")
    parser.add_argument("--max_files", type=int, default=1)
    parser.add_argument("--history_size", type=int, default=3)
    parser.add_argument("--rollout_frames", type=int, default=50)
    parser.add_argument("--rollout_history_source", type=str, default="pbu", choices=("pbu", "flow_prior", "blend"))
    parser.add_argument("--history_blend_alpha", type=float, default=0.5)
    parser.add_argument("--short_lengths", type=_parse_csv_ints, default=(2, 3, 5, 10, 20, 50))
    parser.add_argument("--budget", type=int, default=14)
    parser.add_argument("--mask_families", type=_parse_csv_strings, default=("fixed104",))
    parser.add_argument("--n_lines", type=int, default=112)
    parser.add_argument("--seed", type=int, default=914)
    parser.add_argument("--roll_stride", type=int, default=1)
    parser.add_argument("--roll_offset", type=int, default=0)
    parser.add_argument("--heuristic_weights", type=_parse_float_weights, default={})
    parser.add_argument("--heuristic_rbf_sigma", type=float, default=3.0)
    parser.add_argument("--heuristic_rbf_suppression", type=float, default=0.45)
    parser.add_argument("--hybrid_base_count", type=int, default=10)
    parser.add_argument("--flow_steps", type=int, default=2)
    parser.add_argument("--eval_noise_scale", type=float, default=0.0)
    parser.add_argument("--flow_base_channels", type=int, default=32)
    parser.add_argument("--flow_channel_mult", type=_parse_csv_ints, default=(1, 2))
    parser.add_argument("--flow_num_res_blocks", type=int, default=1)
    parser.add_argument("--flow_groupnorm_groups", type=int, default=4)
    parser.add_argument("--velocity_clip_scale", type=float, default=1.25)
    parser.add_argument("--pbu_base_channels", type=int, default=8)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


MANUAL_SELECTOR_FAMILIES = {
    "heuristic_rbf",
    "heuristic_plain",
    "uncertainty_rbf",
    "change_rbf",
    "edge_rbf",
    "hybrid_rolled_heuristic",
}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _load_sequence(args: argparse.Namespace) -> tuple[Path, torch.Tensor]:
    files = list_stage1_hdf5_files(_resolve_data_root(args.data_root), split=str(args.split))[: int(args.max_files)]
    if not files:
        raise RuntimeError("No standalone pixel-flow input files")
    path = files[0]
    with h5py.File(path, "r") as f:
        data = torch.as_tensor(f[str(args.key)][:], dtype=torch.float32)
    data = map_range_tensor(data, source_range=(-60.0, 0.0), target_range=(-1.0, 1.0)).unsqueeze(1)
    return path, data


def _load_flow(args: argparse.Namespace, *, device: torch.device) -> tuple[StandalonePixelRectifiedFlowPrior, dict[str, Any]]:
    ckpt = torch.load(args.flow_checkpoint, map_location="cpu")
    if not isinstance(ckpt, dict):
        ckpt = {"model_state_dict": ckpt, "checkpoint": str(args.flow_checkpoint)}
    flow_args = argparse.Namespace(**vars(args))
    saved_args = ckpt.get("args")
    if isinstance(saved_args, dict):
        for key in (
            "history_size",
            "flow_base_channels",
            "flow_channel_mult",
            "flow_num_res_blocks",
            "flow_groupnorm_groups",
            "velocity_clip_scale",
        ):
            if key in saved_args:
                setattr(flow_args, key, saved_args[key])
    model = _make_flow(flow_args, device=device)
    if "pixel_rectified_flow_prior_state_dict" in ckpt:
        state_dict = ckpt["pixel_rectified_flow_prior_state_dict"]
    elif "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model, ckpt


def _sample_action(
    *,
    sampler: VariableMaskBudgetSampler,
    family: str,
    budget: int,
    target: torch.Tensor,
    heuristic_map: torch.Tensor,
    step_idx: int,
    roll: int,
):
    return sampler.sample_action(
        budget=int(budget),
        family=str(family),
        batch_size=int(target.shape[0]),
        heuristic_map=heuristic_map if str(family) == MASK_HEURISTIC else None,
        roll=int(roll),
    )


def _manual_selector_action(
    *,
    args: argparse.Namespace,
    sampler: VariableMaskBudgetSampler,
    family: str,
    budget: int,
    prior_mu: torch.Tensor,
    prior_u: torch.Tensor,
    prev_x: torch.Tensor,
    prev_u: torch.Tensor,
    prev_mask: torch.Tensor,
    roll: int,
):
    generator = sampler.generator
    family = str(family)
    change_map = torch.abs(prior_mu - prev_x)
    line_history = generator.linewise_mean(prev_mask)
    if family in {"heuristic_rbf", "heuristic_plain", "hybrid_rolled_heuristic"}:
        scores = heuristic_line_scores(
            generator=generator,
            u_prior=prior_u,
            change_map=change_map,
            prior_image=prior_mu,
            line_history=line_history,
            previous_residual_ema=prev_u,
            weights=dict(args.heuristic_weights),
        )
    elif family == "uncertainty_rbf":
        scores = normalize_line_scores(generator.linewise_mean(torch.abs(prior_u)))
    elif family == "change_rbf":
        scores = normalize_line_scores(generator.linewise_mean(change_map))
    elif family == "edge_rbf":
        scores = normalize_line_scores(generator.linewise_mean(edge_magnitude(prior_mu)))
    else:
        raise ValueError(f"Unsupported manual selector family {family!r}")

    if family == "heuristic_plain":
        return generator.action_from_scores(scores, k=int(budget))
    if family == "hybrid_rolled_heuristic":
        base_count = min(max(0, int(args.hybrid_base_count)), int(budget))
        base = sampler.sample_indices(
            budget=base_count,
            family="rolled_equispaced",
            batch_size=int(prior_mu.shape[0]),
            roll=int(roll),
        )
        base_selected = generator.selected_lines_from_indices(base, batch_size=int(prior_mu.shape[0]))
        remaining = int(budget) - base_count
        if remaining <= 0:
            return generator.action_from_indices(base, batch_size=int(prior_mu.shape[0]))
        extra = rbf_greedy_topk(
            scores,
            k=remaining,
            available=torch.logical_not(base_selected),
            sigma=float(args.heuristic_rbf_sigma),
            suppression=float(args.heuristic_rbf_suppression),
        )
        selected = base_selected | extra
        indices = selected_indices_from_mask(selected)
        return generator.action_from_indices(indices, batch_size=int(prior_mu.shape[0]))
    return action_from_heuristic_scores(
        generator=generator,
        scores=scores,
        budget=int(budget),
        sigma=float(args.heuristic_rbf_sigma),
        suppression=float(args.heuristic_rbf_suppression),
    )


def _flow_start(history_x: torch.Tensor, noise_scale: float) -> torch.Tensor:
    start = history_x[:, -1]
    if float(noise_scale) > 0.0:
        start = start + float(noise_scale) * torch.randn_like(start)
    return start.clamp(-1.25, 1.25)


def _predict_prior(
    *,
    flow_model: StandalonePixelRectifiedFlowPrior,
    history_x: torch.Tensor,
    history_u: torch.Tensor,
    history_mask: torch.Tensor,
    steps: int,
    noise_scale: float,
):
    return flow_model.integrate(
        x_start=_flow_start(history_x, noise_scale),
        steps=int(steps),
        history_x=history_x,
        history_u=history_u,
        history_mask=history_mask,
        history_selected=history_mask,
    )


def _run_one_family(
    *,
    args: argparse.Namespace,
    family: str,
    data: torch.Tensor,
    flow_model: StandalonePixelRectifiedFlowPrior,
    pbu_model,
    sampler: VariableMaskBudgetSampler,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    h = int(args.history_size)
    max_targets = min(int(args.rollout_frames), int(data.shape[0]) - h)
    if max_targets <= 0:
        raise RuntimeError("Not enough frames for standalone pixel-flow rollout")
    measurement_op = MeasurementOperator()
    profiler = TimingProfiler(sync_cuda=(device.type == "cuda"), device=device)
    rollout_x = data[:h].to(device=device)
    rollout_u = torch.zeros_like(rollout_x)
    rollout_mask = torch.ones_like(rollout_x)
    rows: list[dict[str, Any]] = []
    for offset in range(max_targets):
        target = data[h + offset : h + offset + 1].to(device=device)
        rollout_history = rollout_x[-h:].unsqueeze(0)
        rollout_history_u = rollout_u[-h:].unsqueeze(0)
        rollout_history_mask = rollout_mask[-h:].unsqueeze(0)
        teacher_history = data[offset : offset + h].to(device=device).unsqueeze(0)
        teacher_zero = torch.zeros_like(teacher_history)
        teacher_one = torch.ones_like(teacher_history)
        with torch.no_grad():
            profiler.start_frame(offset)
            with profiler.section("flow_prior"):
                rollout_prior = _predict_prior(
                    flow_model=flow_model,
                    history_x=rollout_history,
                    history_u=rollout_history_u,
                    history_mask=rollout_history_mask,
                    steps=int(args.flow_steps),
                    noise_scale=float(args.eval_noise_scale),
                )
            with profiler.section("selection"):
                heuristic = rollout_prior.u_prior + torch.abs(rollout_prior.mu_prior - rollout_history[:, -1])
                roll = int(args.roll_offset) + int(args.roll_stride) * int(offset)
                if str(family) in MANUAL_SELECTOR_FAMILIES:
                    action = _manual_selector_action(
                        args=args,
                        sampler=sampler,
                        family=str(family),
                        budget=int(args.budget),
                        prior_mu=rollout_prior.mu_prior,
                        prior_u=rollout_prior.u_prior,
                        prev_x=rollout_history[:, -1],
                        prev_u=rollout_history_u[:, -1],
                        prev_mask=rollout_history_mask[:, -1],
                        roll=roll,
                    )
                else:
                    action = _sample_action(
                        sampler=sampler,
                        family=family,
                        budget=int(args.budget),
                        target=target,
                        heuristic_map=heuristic,
                        step_idx=offset,
                        roll=roll,
                    )
            with profiler.section("mask_projection"):
                measurement = measurement_op.measure(target, action)
            with profiler.section("pbu"):
                rollout_batch = _stage1_batch_from_prior(
                    x_gt=target,
                    x_prior=rollout_prior.mu_prior,
                    u_prior=rollout_prior.u_prior,
                    prev_x_final=rollout_history[:, -1],
                    prev_u_post=rollout_history_u[:, -1],
                    measurement=measurement,
                    budget=int(args.budget),
                    mask_family=family,
                )
                rollout_out = pbu_model(rollout_batch)
            profiler.end_frame()
            teacher_prior = _predict_prior(
                flow_model=flow_model,
                history_x=teacher_history,
                history_u=teacher_zero,
                history_mask=teacher_one,
                steps=int(args.flow_steps),
                noise_scale=0.0,
            )
            teacher_batch = _stage1_batch_from_prior(
                x_gt=target,
                x_prior=teacher_prior.mu_prior,
                u_prior=teacher_prior.u_prior,
                prev_x_final=teacher_history[:, -1],
                prev_u_post=teacher_zero[:, -1],
                measurement=measurement,
                budget=int(args.budget),
                mask_family=family,
            )
            teacher_out = pbu_model(teacher_batch)
        teacher_metrics = prior_prediction_metrics(prediction=teacher_out.x_final, target=target, uncertainty=None)
        rollout_metrics = prior_prediction_metrics(prediction=rollout_out.x_final, target=target, uncertainty=None)
        prior_metrics = prior_prediction_metrics(prediction=rollout_prior.mu_prior, target=target, uncertainty=None)
        observed = float(rollout_out.observed_consistency_l1.detach().cpu().item())
        row = {
            "mask_family": str(family),
            "frame_offset": int(offset + 1),
            "budget": int(args.budget),
            "selected_count": int(torch.sum(action.selected_lines).detach().cpu().item()),
            "flow_prior_psnr": prior_metrics["psnr"],
            "rollout_pbu_psnr": rollout_metrics["psnr"],
            "teacher_forced_pbu_psnr": teacher_metrics["psnr"],
            "teacher_forced_replay_mask_pbu_psnr": teacher_metrics["psnr"],
            "rollout_pbu_ssim": rollout_metrics["ssim"],
            "teacher_forced_pbu_ssim": teacher_metrics["ssim"],
            "teacher_forced_replay_mask_pbu_ssim": teacher_metrics["ssim"],
            "pbu_psnr_drop_vs_teacher_forced": float(teacher_metrics["psnr"] - rollout_metrics["psnr"]),
            "mask_source": "rollout_selection",
            "observed_consistency_l1": observed,
        }
        rows.append(row)
        if not all(torch.isfinite(t).all().item() for t in (rollout_prior.mu_prior, rollout_out.x_final, rollout_out.u_post)):
            raise RuntimeError(f"NaN/Inf detected in standalone pixel-flow rollout for family={family} frame={offset + 1}")
        if observed > 1.0e-6:
            raise RuntimeError(f"observed_consistency_l1={observed} exceeds 1e-6 for family={family}")
        if str(args.rollout_history_source) == "flow_prior":
            next_x = rollout_prior.mu_prior.detach()
            next_u = rollout_prior.u_prior.detach()
        elif str(args.rollout_history_source) == "blend":
            alpha = min(1.0, max(0.0, float(args.history_blend_alpha)))
            next_x = (alpha * rollout_out.x_final.detach() + (1.0 - alpha) * rollout_prior.mu_prior.detach()).detach()
            next_u = (alpha * rollout_out.u_post.detach() + (1.0 - alpha) * rollout_prior.u_prior.detach()).detach()
        else:
            next_x = rollout_out.x_final.detach()
            next_u = rollout_out.u_post.detach()
        rollout_x = torch.cat([rollout_x, next_x], dim=0)
        rollout_u = torch.cat([rollout_u, next_u], dim=0)
        rollout_mask = torch.cat([rollout_mask, measurement.mask.detach()], dim=0)

    short = {}
    for length in args.short_lengths:
        length = min(int(length), len(rows))
        short[f"T{int(length)}_rollout_psnr"] = mean_or_none([row["rollout_pbu_psnr"] for row in rows[:length]])
        short[f"T{int(length)}_teacher_forced_psnr"] = mean_or_none(
            [row["teacher_forced_pbu_psnr"] for row in rows[:length]]
        )
        short[f"T{int(length)}_drop_vs_teacher"] = mean_or_none(
            [row["pbu_psnr_drop_vs_teacher_forced"] for row in rows[:length]]
        )
    first_bad = None
    for row in rows:
        if float(row["pbu_psnr_drop_vs_teacher_forced"]) > 1.0:
            first_bad = int(row["frame_offset"])
            break
    summary = {
        "mask_family": str(family),
        "frames": int(len(rows)),
        "budget": int(args.budget),
        "flow_prior_psnr_mean": mean_or_none([row["flow_prior_psnr"] for row in rows]),
        "rollout_pbu_psnr_mean": mean_or_none([row["rollout_pbu_psnr"] for row in rows]),
        "teacher_forced_pbu_psnr_mean": mean_or_none([row["teacher_forced_pbu_psnr"] for row in rows]),
        "mean_drop_vs_teacher_forced": mean_or_none([row["pbu_psnr_drop_vs_teacher_forced"] for row in rows]),
        "frames_1_5_psnr": mean_or_none([row["rollout_pbu_psnr"] for row in rows[:5]]),
        "frames_16_20_psnr": mean_or_none([row["rollout_pbu_psnr"] for row in rows[15:20]]),
        "last_10_psnr": mean_or_none([row["rollout_pbu_psnr"] for row in rows[-10:]]),
        "first_10_psnr": mean_or_none([row["rollout_pbu_psnr"] for row in rows[:10]]),
        "first_bad_frame_drop_gt_1db": first_bad,
        "max_observed_consistency_l1": max(float(row["observed_consistency_l1"]) for row in rows),
        "selected_count_all_match_budget": all(int(row["selected_count"]) == int(args.budget) for row in rows),
        "timing": profiler.summary(),
        **short,
    }
    if summary["frames_1_5_psnr"] is not None and summary["frames_16_20_psnr"] is not None:
        summary["drop_frames_1_5_to_16_20"] = float(summary["frames_1_5_psnr"] - summary["frames_16_20_psnr"])
    else:
        summary["drop_frames_1_5_to_16_20"] = None
    if summary["first_10_psnr"] is not None and summary["last_10_psnr"] is not None:
        summary["drop_first10_to_last10"] = float(summary["first_10_psnr"] - summary["last_10_psnr"])
    else:
        summary["drop_first10_to_last10"] = None
    return rows, summary


def main() -> None:
    args = parse_args()
    device = torch.device("cuda:0" if args.device == "auto" and torch.cuda.is_available() else "cpu")
    if args.device != "auto":
        device = torch.device(args.device)
    torch.manual_seed(int(args.seed))
    output_dir = _prepare_output_dir(args.output_dir, allow_overwrite=bool(args.allow_overwrite))
    flow_model, ckpt = _load_flow(args, device=device)
    pbu_model = _load_pbu(args, device=device)
    if pbu_model is None:
        raise RuntimeError("standalone pixel-flow eval requires --pbu_checkpoint")
    path, data = _load_sequence(args)
    sampler = VariableMaskBudgetSampler(
        image_shape=(int(data.shape[-2]), int(data.shape[-1])),
        n_lines=int(args.n_lines),
        device=device,
        dtype=torch.float32,
        seed=int(args.seed),
    )
    all_rows: list[dict[str, Any]] = []
    summaries = []
    for family in args.mask_families:
        rows, summary = _run_one_family(
            args=args,
            family=str(family),
            data=data,
            flow_model=flow_model,
            pbu_model=pbu_model,
            sampler=sampler,
            device=device,
        )
        all_rows.extend(rows)
        summaries.append(summary)
    max_observed = max(float(item["max_observed_consistency_l1"]) for item in summaries)
    timing_values = [
        item["timing"].get("steady_fps_excl_first")
        for item in summaries
        if item["timing"].get("steady_fps_excl_first") is not None
    ]
    timing = {
        "steady_fps_excl_first": mean_or_none(timing_values),
        "note": "Mean of per-mask-family standalone pixel-flow online timing; excludes teacher-forced branch.",
        "family_timing": {item["mask_family"]: item["timing"] for item in summaries},
    }
    basic = {
        "psnr_mean": mean_or_none([row["rollout_pbu_psnr"] for row in all_rows]),
        "ssim_mean": mean_or_none([row["rollout_pbu_ssim"] for row in all_rows]),
        "flow_prior_psnr_mean": mean_or_none([row["flow_prior_psnr"] for row in all_rows]),
        "teacher_forced_psnr_mean": mean_or_none([row["teacher_forced_pbu_psnr"] for row in all_rows]),
        "rollout_drop_psnr_mean": mean_or_none([row["pbu_psnr_drop_vs_teacher_forced"] for row in all_rows]),
        "observed_consistency_l1": max_observed,
        "first_bad_frame": min(
            [int(item["first_bad_frame_drop_gt_1db"]) for item in summaries if item["first_bad_frame_drop_gt_1db"] is not None],
            default=None,
        ),
    }
    payload = {
        "mode": "stage2_standalone_pixel_rectified_flow_full_loop",
        "flow_checkpoint": str(args.flow_checkpoint),
        "pbu_checkpoint": str(args.pbu_checkpoint),
        "checkpoint_reloaded": True,
        "flow_checkpoint_keys": sorted(ckpt.keys()) if isinstance(ckpt, dict) else None,
        "rollout_history_source": str(args.rollout_history_source),
        "history_blend_alpha": float(args.history_blend_alpha),
        "data_root": str(args.data_root),
        "split": str(args.split),
        "file_path": str(path),
        "mask_families": [str(x) for x in args.mask_families],
        "flow_steps": int(args.flow_steps),
        "eval_noise_scale": float(args.eval_noise_scale),
        "roll_stride": int(args.roll_stride),
        "roll_offset": int(args.roll_offset),
        "heuristic_weights": dict(args.heuristic_weights),
        "heuristic_rbf_sigma": float(args.heuristic_rbf_sigma),
        "heuristic_rbf_suppression": float(args.heuristic_rbf_suppression),
        "hybrid_base_count": int(args.hybrid_base_count),
        "summaries": summaries,
        "no_base_prior_checkpoint": True,
        "no_nan": True,
        "max_observed_consistency_l1": max_observed,
    }
    _write_csv(output_dir / "rectified_flow_loop_frame_metrics.csv", all_rows)
    _write_json(output_dir / "basic_metrics.json", basic)
    _write_json(output_dir / "timing.json", timing)
    _write_json(output_dir / "summary.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
