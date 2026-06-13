"""Offline oracle selector diagnostic for standalone Stage2 pixel flow."""

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

import torch

from evaluate_pixel_rectified_flow_loop_stage2 import (  # noqa: E402
    _load_flow,
    _load_sequence,
    _predict_prior,
    _sample_action,
    _write_csv,
)
from train_pixel_temporal_prior_stage2 import (  # noqa: E402
    _load_pbu,
    _prepare_output_dir,
    _stage1_batch_from_prior,
    _write_json,
)
from ulsa.pixel_measurement import MeasurementOperator  # noqa: E402
from ulsa.pixel_pbu_stage1 import VariableMaskBudgetSampler  # noqa: E402
from ulsa.pixel_line_policy import build_stage2j_line_features  # noqa: E402
from ulsa.pixel_temporal_prior import mean_or_none, prior_prediction_metrics, psnr_from_mse  # noqa: E402


def _parse_csv_strings(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in str(value).split(",") if part.strip())


def _parse_csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in str(value).split(",") if part.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline oracle line selector for standalone pixel-flow Stage2.")
    parser.add_argument("--flow_checkpoint", type=str, required=True)
    parser.add_argument("--pbu_checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--allow_overwrite", action="store_true")
    parser.add_argument("--data_root", type=str, default="processed_echonet")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--key", type=str, default="data/image_sc")
    parser.add_argument("--max_files", type=int, default=1)
    parser.add_argument("--history_size", type=int, default=3)
    parser.add_argument("--rollout_frames", type=int, default=20)
    parser.add_argument("--budget", type=int, default=14)
    parser.add_argument("--n_lines", type=int, default=112)
    parser.add_argument("--seed", type=int, default=914)
    parser.add_argument("--flow_steps", type=int, default=2)
    parser.add_argument("--eval_noise_scale", type=float, default=0.0)
    parser.add_argument("--oracle_batch_size", type=int, default=112)
    parser.add_argument("--compare_families", type=_parse_csv_strings, default=("fixed104", "fixed14", "rolled_equispaced", "heuristic_generated", "random"))
    parser.add_argument("--short_lengths", type=_parse_csv_ints, default=(2, 3, 5, 10, 20))
    parser.add_argument("--write_candidate_gains", action="store_true")
    parser.add_argument("--write_feature_labels", action="store_true")
    parser.add_argument("--flow_base_channels", type=int, default=32)
    parser.add_argument("--flow_channel_mult", type=_parse_csv_ints, default=(1, 2))
    parser.add_argument("--flow_num_res_blocks", type=int, default=1)
    parser.add_argument("--flow_groupnorm_groups", type=int, default=4)
    parser.add_argument("--velocity_clip_scale", type=float, default=1.25)
    parser.add_argument("--pbu_base_channels", type=int, default=8)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def _batch_psnr(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target = target.to(device=prediction.device, dtype=prediction.dtype).expand_as(prediction)
    mse = torch.mean((prediction - target) ** 2, dim=(1, 2, 3))
    return psnr_from_mse(mse, data_range=2.0)


def _masked_l1(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = torch.sum(mask).clamp_min(1.0)
    return torch.sum(torch.abs(value) * mask) / denom


def _slice_pbu_output(out, idx: int, measurement):
    idx = int(idx)
    x_raw = out.x_raw[idx : idx + 1].detach()
    x_final = out.x_final[idx : idx + 1].detach()
    u_post = out.u_post[idx : idx + 1].detach()
    obs = measurement.obs[idx : idx + 1].detach()
    mask = measurement.mask[idx : idx + 1].detach()
    return type(out)(
        x_raw=x_raw,
        x_final=x_final,
        u_post=u_post,
        delta_x=out.delta_x[idx : idx + 1].detach(),
        logvar=out.logvar[idx : idx + 1].detach(),
        delta_raw=out.delta_raw[idx : idx + 1].detach(),
        observed_consistency_l1=_masked_l1(x_final - obs, mask).detach(),
        observed_raw_l1=_masked_l1(x_raw - obs, mask).detach(),
    )


def _run_pbu_for_indices(
    *,
    pbu_model,
    measurement_op: MeasurementOperator,
    sampler: VariableMaskBudgetSampler,
    target: torch.Tensor,
    prior_mu: torch.Tensor,
    prior_u: torch.Tensor,
    prev_x: torch.Tensor,
    prev_u: torch.Tensor,
    line_indices: torch.Tensor,
    budget: int,
    mask_family: str,
):
    if line_indices.ndim == 1:
        line_indices = line_indices.view(1, -1)
    batch_size = int(line_indices.shape[0])
    action = sampler.generator.action_from_indices(line_indices, batch_size=batch_size)
    target_b = target.expand(batch_size, -1, -1, -1)
    measurement = measurement_op.measure(target_b, action)
    batch = _stage1_batch_from_prior(
        x_gt=target_b,
        x_prior=prior_mu.expand(batch_size, -1, -1, -1),
        u_prior=prior_u.expand(batch_size, -1, -1, -1),
        prev_x_final=prev_x.expand(batch_size, -1, -1, -1),
        prev_u_post=prev_u.expand(batch_size, -1, -1, -1),
        measurement=measurement,
        budget=int(budget),
        mask_family=str(mask_family),
    )
    out = pbu_model(batch)
    return out, measurement, action, _batch_psnr(out.x_final, target)


def _oracle_greedy_select(
    *,
    args: argparse.Namespace,
    pbu_model,
    measurement_op: MeasurementOperator,
    sampler: VariableMaskBudgetSampler,
    target: torch.Tensor,
    prior_mu: torch.Tensor,
    prior_u: torch.Tensor,
    prev_x: torch.Tensor,
    prev_u: torch.Tensor,
    prior_psnr: float,
) -> tuple[Any, Any, list[int], list[dict[str, Any]]]:
    selected: list[int] = []
    available = set(range(int(args.n_lines)))
    candidate_rows: list[dict[str, Any]] = []
    current_psnr = float(prior_psnr)
    current_out = None
    current_measurement = None
    batch_size = max(1, int(args.oracle_batch_size))
    for step in range(int(args.budget)):
        candidates = sorted(available)
        step_records: list[dict[str, Any]] = []
        best_psnr = -float("inf")
        best_line = None
        best_out = None
        best_measurement = None
        for start in range(0, len(candidates), batch_size):
            chunk = candidates[start : start + batch_size]
            indices = torch.as_tensor(
                [selected + [int(line)] for line in chunk],
                dtype=torch.long,
                device=target.device,
            )
            out, measurement, _action, psnrs = _run_pbu_for_indices(
                pbu_model=pbu_model,
                measurement_op=measurement_op,
                sampler=sampler,
                target=target,
                prior_mu=prior_mu,
                prior_u=prior_u,
                prev_x=prev_x,
                prev_u=prev_u,
                line_indices=indices,
                budget=int(args.budget),
                mask_family="oracle_greedy",
            )
            psnrs_cpu = psnrs.detach().cpu()
            local_best_idx = int(torch.argmax(psnrs).detach().cpu().item())
            local_best_psnr = float(psnrs_cpu[local_best_idx].item())
            if local_best_psnr > best_psnr:
                best_psnr = local_best_psnr
                best_line = int(chunk[local_best_idx])
                best_out = _slice_pbu_output(out, local_best_idx, measurement)
                best_measurement = type(measurement)(
                    obs=measurement.obs[local_best_idx : local_best_idx + 1].detach(),
                    mask=measurement.mask[local_best_idx : local_best_idx + 1].detach(),
                    action=sampler.generator.action_from_indices(selected + [best_line], batch_size=1),
                )
            if bool(args.write_candidate_gains) or bool(args.write_feature_labels):
                for line, value in zip(chunk, psnrs_cpu.tolist(), strict=True):
                    step_records.append(
                        {
                            "greedy_step": int(step + 1),
                            "line_idx": int(line),
                            "candidate_psnr": float(value),
                            "candidate_gain_vs_current": float(value - current_psnr),
                        }
                    )
        if best_line is None or best_out is None or best_measurement is None:
            raise RuntimeError("Oracle greedy failed to select a line")
        if bool(args.write_candidate_gains) or bool(args.write_feature_labels):
            step_records.sort(key=lambda row: float(row["candidate_psnr"]), reverse=True)
            for rank, row in enumerate(step_records, start=1):
                row["candidate_rank"] = int(rank)
                row["selected_by_oracle"] = bool(int(row["line_idx"]) == int(best_line))
            candidate_rows.extend(step_records)
        selected.append(int(best_line))
        available.remove(int(best_line))
        current_psnr = float(best_psnr)
        current_out = best_out
        current_measurement = best_measurement
    if current_out is None or current_measurement is None:
        raise RuntimeError("Oracle greedy produced no output")
    return current_out, current_measurement, selected, candidate_rows


def _selector_output(
    *,
    args: argparse.Namespace,
    family: str,
    pbu_model,
    measurement_op: MeasurementOperator,
    sampler: VariableMaskBudgetSampler,
    target: torch.Tensor,
    prior_mu: torch.Tensor,
    prior_u: torch.Tensor,
    prev_x: torch.Tensor,
    prev_u: torch.Tensor,
    heuristic_map: torch.Tensor,
    step_idx: int,
):
    action = _sample_action(
        sampler=sampler,
        family=str(family),
        budget=int(args.budget),
        target=target,
        heuristic_map=heuristic_map,
        step_idx=int(step_idx),
    )
    measurement = measurement_op.measure(target, action)
    batch = _stage1_batch_from_prior(
        x_gt=target,
        x_prior=prior_mu,
        u_prior=prior_u,
        prev_x_final=prev_x,
        prev_u_post=prev_u,
        measurement=measurement,
        budget=int(args.budget),
        mask_family=str(family),
    )
    out = pbu_model(batch)
    return out, measurement, action


def _selected_indices(action) -> list[int]:
    return [int(x) for x in torch.nonzero(action.selected_lines[0], as_tuple=False).flatten().detach().cpu().tolist()]


def _line_set_summary(rows: list[dict[str, Any]], *, prefix: str) -> dict[str, Any]:
    counts: dict[int, int] = {}
    for row in rows:
        for item in str(row[f"{prefix}_lines"]).split(" "):
            if item == "":
                continue
            line = int(item)
            counts[line] = counts.get(line, 0) + 1
    top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:20]
    return {
        f"{prefix}_line_top20": [{"line": int(line), "count": int(count)} for line, count in top],
        f"{prefix}_unique_lines": int(len(counts)),
    }


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
        raise RuntimeError("oracle selector requires --pbu_checkpoint")
    path, data = _load_sequence(args)
    sampler = VariableMaskBudgetSampler(
        image_shape=(int(data.shape[-2]), int(data.shape[-1])),
        n_lines=int(args.n_lines),
        device=device,
        dtype=torch.float32,
        seed=int(args.seed),
    )
    measurement_op = MeasurementOperator()
    h = int(args.history_size)
    max_targets = min(int(args.rollout_frames), int(data.shape[0]) - h)
    rollout_x = data[:h].to(device=device)
    rollout_u = torch.zeros_like(rollout_x)
    rollout_mask = torch.ones_like(rollout_x)
    rows: list[dict[str, Any]] = []
    selector_rows: list[dict[str, Any]] = []
    candidate_rows_all: list[dict[str, Any]] = []
    feature_tensors: list[torch.Tensor] = []
    gain_tensors: list[torch.Tensor] = []
    selected_tensors: list[torch.Tensor] = []
    feature_names: tuple[str, ...] | None = None
    with torch.no_grad():
        for offset in range(max_targets):
            target = data[h + offset : h + offset + 1].to(device=device)
            history = rollout_x[-h:].unsqueeze(0)
            history_u = rollout_u[-h:].unsqueeze(0)
            history_mask = rollout_mask[-h:].unsqueeze(0)
            prior = _predict_prior(
                flow_model=flow_model,
                history_x=history,
                history_u=history_u,
                history_mask=history_mask,
                steps=int(args.flow_steps),
                noise_scale=float(args.eval_noise_scale),
            )
            prev_x = history[:, -1]
            prev_u = history_u[:, -1]
            prior_metrics = prior_prediction_metrics(prediction=prior.mu_prior, target=target, uncertainty=None)
            oracle_out, oracle_measurement, oracle_lines, candidate_rows = _oracle_greedy_select(
                args=args,
                pbu_model=pbu_model,
                measurement_op=measurement_op,
                sampler=sampler,
                target=target,
                prior_mu=prior.mu_prior,
                prior_u=prior.u_prior,
                prev_x=prev_x,
                prev_u=prev_u,
                prior_psnr=float(prior_metrics["psnr"]),
            )
            oracle_metrics = prior_prediction_metrics(prediction=oracle_out.x_final, target=target, uncertainty=None)
            oracle_set = set(int(x) for x in oracle_lines)
            if bool(args.write_feature_labels):
                features, feature_spec = build_stage2j_line_features(
                    generator=sampler.generator,
                    mu_prior=prior.mu_prior,
                    u_prior=prior.u_prior,
                    prev_x=prev_x,
                    prev_u=prev_u,
                    prev_mask=history_mask[:, -1],
                )
                feature_names = tuple(feature_spec.names)
                gain = torch.full((int(args.n_lines),), -float("inf"), device=device, dtype=torch.float32)
                for item in candidate_rows:
                    if int(item["greedy_step"]) == 1:
                        gain[int(item["line_idx"])] = float(item["candidate_gain_vs_current"])
                if not bool(torch.all(torch.isfinite(gain)).item()):
                    raise RuntimeError("Feature-label export requires candidate gains for all first-step lines")
                selected_mask = torch.zeros((int(args.n_lines),), device=device, dtype=torch.float32)
                selected_mask[torch.as_tensor(oracle_lines, device=device, dtype=torch.long)] = 1.0
                feature_tensors.append(features.detach().cpu()[0])
                gain_tensors.append(gain.detach().cpu())
                selected_tensors.append(selected_mask.detach().cpu())
            row: dict[str, Any] = {
                "frame_offset": int(offset + 1),
                "budget": int(args.budget),
                "flow_prior_psnr": prior_metrics["psnr"],
                "oracle_pbu_psnr": oracle_metrics["psnr"],
                "oracle_pbu_ssim": oracle_metrics["ssim"],
                "oracle_gain_over_prior_psnr": float(oracle_metrics["psnr"] - prior_metrics["psnr"]),
                "oracle_lines": " ".join(str(int(x)) for x in oracle_lines),
                "observed_consistency_l1": float(oracle_out.observed_consistency_l1.detach().cpu().reshape(-1)[0].item()),
            }
            heuristic = prior.u_prior + torch.abs(prior.mu_prior - prev_x)
            for family in args.compare_families:
                comp_out, _comp_measurement, comp_action = _selector_output(
                    args=args,
                    family=str(family),
                    pbu_model=pbu_model,
                    measurement_op=measurement_op,
                    sampler=sampler,
                    target=target,
                    prior_mu=prior.mu_prior,
                    prior_u=prior.u_prior,
                    prev_x=prev_x,
                    prev_u=prev_u,
                    heuristic_map=heuristic,
                    step_idx=offset,
                )
                comp_metrics = prior_prediction_metrics(prediction=comp_out.x_final, target=target, uncertainty=None)
                comp_lines = _selected_indices(comp_action)
                comp_set = set(comp_lines)
                overlap = len(oracle_set & comp_set)
                selector_rows.append(
                    {
                        "frame_offset": int(offset + 1),
                        "family": str(family),
                        "psnr": comp_metrics["psnr"],
                        "ssim": comp_metrics["ssim"],
                        "regret_vs_oracle_psnr": float(oracle_metrics["psnr"] - comp_metrics["psnr"]),
                        "overlap_with_oracle": int(overlap),
                        "jaccard_with_oracle": float(overlap / max(1, len(oracle_set | comp_set))),
                        "lines": " ".join(str(int(x)) for x in comp_lines),
                    }
                )
                row[f"{family}_psnr"] = comp_metrics["psnr"]
                row[f"{family}_overlap_with_oracle"] = int(overlap)
            rows.append(row)
            for item in candidate_rows:
                item["frame_offset"] = int(offset + 1)
                candidate_rows_all.append(item)
            if row["observed_consistency_l1"] > 1.0e-6:
                raise RuntimeError(f"Oracle observed consistency failed at frame {offset + 1}: {row['observed_consistency_l1']}")
            rollout_x = torch.cat([rollout_x, oracle_out.x_final.detach()], dim=0)
            rollout_u = torch.cat([rollout_u, oracle_out.u_post.detach()], dim=0)
            rollout_mask = torch.cat([rollout_mask, oracle_measurement.mask.detach()], dim=0)

    selector_summary: dict[str, Any] = {}
    for family in args.compare_families:
        fam_rows = [row for row in selector_rows if row["family"] == str(family)]
        selector_summary[str(family)] = {
            "psnr_mean": mean_or_none([row["psnr"] for row in fam_rows]),
            "ssim_mean": mean_or_none([row["ssim"] for row in fam_rows]),
            "regret_vs_oracle_psnr_mean": mean_or_none([row["regret_vs_oracle_psnr"] for row in fam_rows]),
            "overlap_with_oracle_mean": mean_or_none([row["overlap_with_oracle"] for row in fam_rows]),
            "jaccard_with_oracle_mean": mean_or_none([row["jaccard_with_oracle"] for row in fam_rows]),
        }
    short: dict[str, Any] = {}
    for length in args.short_lengths:
        length = min(int(length), len(rows))
        short[f"T{length}_oracle_psnr"] = mean_or_none([row["oracle_pbu_psnr"] for row in rows[:length]])
    summary = {
        "mode": "stage2j_standalone_pixel_flow_oracle_selector",
        "flow_checkpoint": str(args.flow_checkpoint),
        "pbu_checkpoint": str(args.pbu_checkpoint),
        "checkpoint_reloaded": True,
        "flow_checkpoint_keys": sorted(ckpt.keys()) if isinstance(ckpt, dict) else None,
        "data_root": str(args.data_root),
        "split": str(args.split),
        "file_path": str(path),
        "frames": int(len(rows)),
        "budget": int(args.budget),
        "oracle_psnr_mean": mean_or_none([row["oracle_pbu_psnr"] for row in rows]),
        "oracle_ssim_mean": mean_or_none([row["oracle_pbu_ssim"] for row in rows]),
        "flow_prior_psnr_mean": mean_or_none([row["flow_prior_psnr"] for row in rows]),
        "oracle_gain_over_prior_psnr_mean": mean_or_none([row["oracle_gain_over_prior_psnr"] for row in rows]),
        "max_observed_consistency_l1": max(float(row["observed_consistency_l1"]) for row in rows),
        "selector_comparison_on_oracle_history": selector_summary,
        "no_base_prior_checkpoint": True,
        "no_nan": True,
        **short,
        **_line_set_summary(rows, prefix="oracle"),
    }
    _write_csv(output_dir / "oracle_selector_frame_metrics.csv", rows)
    _write_csv(output_dir / "oracle_selector_comparison.csv", selector_rows)
    if bool(args.write_candidate_gains):
        _write_csv(output_dir / "oracle_candidate_gains.csv", candidate_rows_all)
    if bool(args.write_feature_labels):
        torch.save(
            {
                "features": torch.stack(feature_tensors, dim=0),
                "oracle_gain": torch.stack(gain_tensors, dim=0),
                "oracle_selected": torch.stack(selected_tensors, dim=0),
                "feature_names": list(feature_names or ()),
                "meta": {
                    "mode": "stage2j_oracle_first_step_line_labels",
                    "flow_checkpoint": str(args.flow_checkpoint),
                    "pbu_checkpoint": str(args.pbu_checkpoint),
                    "data_root": str(args.data_root),
                    "split": str(args.split),
                    "file_path": str(path),
                    "frames": int(len(feature_tensors)),
                    "budget": int(args.budget),
                    "label": "candidate_gain_vs_current at greedy step 1 plus final oracle selected mask",
                },
            },
            output_dir / "oracle_line_feature_labels.pt",
        )
    _write_json(
        output_dir / "basic_metrics.json",
        {
            "psnr_mean": summary["oracle_psnr_mean"],
            "ssim_mean": summary["oracle_ssim_mean"],
            "flow_prior_psnr_mean": summary["flow_prior_psnr_mean"],
            "observed_consistency_l1": summary["max_observed_consistency_l1"],
        },
    )
    _write_json(output_dir / "summary.json", summary)
    _write_json(
        output_dir / "timing.json",
        {
            "steady_fps_excl_first": None,
            "note": "Offline oracle selector uses GT to evaluate candidate PBU outputs and is not an online FPS benchmark.",
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
