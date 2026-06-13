"""Stage 1 variable-mask PBU real-data dry-run and bounded train smoke.

This is not yet a formal trainer. Dry-run modes validate that the accepted
Stage 1 PBU contract can read `processed_echonet` frames and run forward/eval
plumbing on real data without creating checkpoints or output directories.
`--train_smoke` is a bounded, explicit-output engineering smoke path.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time


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

from ulsa.pixel_pbu_stage1 import (
    MASK_ADVERSARIAL_GAP,
    MASK_CLUSTERED_LOCAL,
    MASK_FIXED10,
    MASK_FIXED104,
    MASK_RANDOM,
    MASK_ROLLED_EQUISPACED,
    PRIOR_COPY_LAST,
    PRIOR_EMA,
    PRIOR_WEAK_INTERPOLATION,
    PRIOR_ZERO_FILL,
    STAGE1_BUDGETS,
    SUPPORTED_STAGE1_MASK_FAMILIES,
    SUPPORTED_STAGE1_PRIORS,
    Stage1PBUWrapper,
    VariableMaskBudgetSampler,
)
from ulsa.pixel_pbu_stage1_dataset import (
    Stage1EchoNetFrameDataset,
    collate_stage1_frame_samples,
    make_stage1_pbu_batch_from_frame_batch,
)
from ulsa.pixel_pbu_stage1_eval import compute_stage1_eval_metrics, summarize_stage1_records, Stage1EvalRecord
from ulsa.pixel_pbu_stage1_train import run_stage1_pbu_train_step, stage1_train_metrics_to_jsonable


DEFAULT_MIXED_MASK_FAMILIES = (
    MASK_RANDOM,
    MASK_ROLLED_EQUISPACED,
    MASK_CLUSTERED_LOCAL,
    MASK_ADVERSARIAL_GAP,
    MASK_FIXED10,
    MASK_FIXED104,
)
DEFAULT_MIXED_PRIORS = (
    PRIOR_COPY_LAST,
    PRIOR_EMA,
    PRIOR_WEAK_INTERPOLATION,
    PRIOR_ZERO_FILL,
)


def _parse_csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in str(value).split(",") if part.strip())


def _parse_csv_strings(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in str(value).split(",") if part.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dry-run Stage 1 variable-mask PBU on real EchoNet frames.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry_run", action="store_true", help="Run forward/eval plumbing without persistent outputs.")
    mode.add_argument(
        "--train_smoke",
        action="store_true",
        help="Run bounded real training smoke. Requires --output_dir and is not formal gate evidence.",
    )
    parser.add_argument(
        "--backward_smoke",
        action="store_true",
        help="Run one in-memory optimizer step during dry-run. No checkpoints or outputs are written.",
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Required for --train_smoke.")
    parser.add_argument(
        "--save_checkpoint",
        action="store_true",
        help="For --train_smoke only, save a clearly named final smoke checkpoint in --output_dir.",
    )
    parser.add_argument("--checkpoint_name", type=str, default="stage1_pbu_smoke_last.pt")
    parser.add_argument("--allow_overwrite", action="store_true", help="Allow writing into a non-empty output_dir.")
    parser.add_argument("--smoke_max_steps_cap", type=int, default=100)
    parser.add_argument("--data_root", type=str, default="processed_echonet")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--key", type=str, default="data/image_sc")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_files", type=int, default=1)
    parser.add_argument("--max_items", type=int, default=2)
    parser.add_argument("--max_frames_per_file", type=int, default=2)
    parser.add_argument("--budget", type=int, default=14)
    parser.add_argument("--mask_family", type=str, default=MASK_RANDOM)
    parser.add_argument("--prior_kind", type=str, default=PRIOR_COPY_LAST)
    parser.add_argument("--n_lines", type=int, default=112)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--max_steps", type=int, default=1)
    parser.add_argument(
        "--schedule_mode",
        type=str,
        default="fixed",
        choices=("fixed", "mixed"),
        help="fixed uses --budget/--mask_family/--prior_kind; mixed cycles through --budgets/--mask_families/--prior_kinds.",
    )
    parser.add_argument(
        "--budgets",
        type=_parse_csv_ints,
        default=None,
        help="Comma-separated budgets for --schedule_mode mixed. Defaults to Stage 1 required budgets.",
    )
    parser.add_argument(
        "--mask_families",
        type=_parse_csv_strings,
        default=None,
        help="Comma-separated mask families for --schedule_mode mixed.",
    )
    parser.add_argument(
        "--prior_kinds",
        type=_parse_csv_strings,
        default=None,
        help="Comma-separated prior kinds for --schedule_mode mixed.",
    )
    parser.add_argument(
        "--heavy_metric_interval",
        type=int,
        default=0,
        help="Compute train-time SSIM/Spearman every N steps. 0 disables heavy train metrics; eval still computes them.",
    )
    parser.add_argument("--base_channels", type=int, default=8)
    parser.add_argument("--channel_mult", type=str, default="1,2")
    parser.add_argument("--num_res_blocks", type=int, default=1)
    parser.add_argument("--groupnorm_groups", type=int, default=4)
    parser.add_argument("--delta_clip_scale", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def _parse_channel_mult(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in str(value).split(",") if part.strip())


def _resolve_data_root(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    repo_path = REPO_ROOT / path
    if repo_path.exists():
        return repo_path
    return path


def _prepare_output_dir(value: str | None, *, allow_overwrite: bool) -> Path:
    if value is None:
        raise RuntimeError("--train_smoke requires an explicit --output_dir")
    path = Path(value)
    if path.exists() and any(path.iterdir()) and not bool(allow_overwrite):
        raise RuntimeError(f"Refusing to write train smoke outputs into non-empty directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _local_checkpoint_name(value: str) -> str:
    path = Path(value)
    if path.is_absolute() or len(path.parts) != 1 or path.name in {"", ".", ".."}:
        raise RuntimeError(f"--checkpoint_name must be a simple filename inside --output_dir, got {value!r}")
    return path.name


def _write_json(path: Path, payload: dict[str, object]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, allow_nan=False)


def _jsonable_args(args: argparse.Namespace) -> dict[str, object]:
    return {key: value for key, value in sorted(vars(args).items())}


def _validate_members(name: str, values: tuple[str, ...], supported: tuple[str, ...]) -> None:
    unsupported = [value for value in values if value not in supported]
    if unsupported:
        raise RuntimeError(f"Unsupported {name}: {unsupported}; supported={list(supported)}")


def _resolve_stage1_train_schedule(args: argparse.Namespace) -> dict[str, object]:
    if str(args.schedule_mode) == "fixed":
        budgets = (int(args.budget),)
        mask_families = (str(args.mask_family),)
        prior_kinds = (str(args.prior_kind),)
    else:
        budgets = tuple(int(x) for x in (args.budgets if args.budgets is not None else STAGE1_BUDGETS))
        mask_families = tuple(
            str(x) for x in (args.mask_families if args.mask_families is not None else DEFAULT_MIXED_MASK_FAMILIES)
        )
        prior_kinds = tuple(str(x) for x in (args.prior_kinds if args.prior_kinds is not None else DEFAULT_MIXED_PRIORS))
    if not budgets:
        raise RuntimeError("Stage 1 train schedule must include at least one budget")
    if not mask_families:
        raise RuntimeError("Stage 1 train schedule must include at least one mask family")
    if not prior_kinds:
        raise RuntimeError("Stage 1 train schedule must include at least one prior kind")
    if any(int(budget) < 0 for budget in budgets):
        raise RuntimeError(f"Stage 1 budgets must be non-negative, got {budgets}")
    _validate_members("mask_families", mask_families, SUPPORTED_STAGE1_MASK_FAMILIES)
    _validate_members("prior_kinds", prior_kinds, SUPPORTED_STAGE1_PRIORS)
    return {
        "mode": str(args.schedule_mode),
        "budgets": [int(x) for x in budgets],
        "mask_families": list(mask_families),
        "prior_kinds": list(prior_kinds),
    }


def _schedule_item_for_step(
    *,
    step_idx: int,
    schedule: dict[str, object],
    sampler: VariableMaskBudgetSampler,
) -> tuple[int, str, str]:
    budgets = [int(x) for x in schedule["budgets"]]
    mask_families = [str(x) for x in schedule["mask_families"]]
    prior_kinds = [str(x) for x in schedule["prior_kinds"]]
    budget = int(budgets[int(step_idx) % len(budgets)])
    if budget > int(sampler.n_lines):
        raise RuntimeError(f"Stage 1 budget must be <= n_lines, got {budget}>{int(sampler.n_lines)}")
    compatible = [family for family in mask_families if family in sampler.compatible_families_for_budget(budget)]
    if not compatible:
        raise RuntimeError(
            f"No compatible mask family for budget={budget}; requested={mask_families}; "
            f"compatible={list(sampler.compatible_families_for_budget(budget))}"
        )
    schedule_slot = int(step_idx) // max(1, len(budgets))
    mask_slot = schedule_slot % len(compatible)
    prior_slot = schedule_slot % len(prior_kinds)
    return budget, str(compatible[mask_slot]), str(prior_kinds[prior_slot])


def _include_heavy_metrics_for_step(step_idx: int, *, interval: int, max_steps: int) -> bool:
    interval = int(interval)
    if interval <= 0:
        return False
    return int(step_idx) == 0 or (int(step_idx) + 1) % interval == 0 or int(step_idx) == int(max_steps) - 1


def _basic_metrics_from_summary(summary: dict[str, object]) -> dict[str, object]:
    mean_metrics = summary.get("mean_metrics", {})
    if not isinstance(mean_metrics, dict):
        mean_metrics = {}
    return {
        "psnr_mean": mean_metrics.get("psnr"),
        "psnr_std": None,
        "ssim_mean": mean_metrics.get("ssim"),
        "ssim_std": None,
        "mse_mean": None,
        "mae_mean": mean_metrics.get("mean_abs_error"),
        "observed_consistency_l1": mean_metrics.get("observed_consistency_l1"),
        "observed_raw_l1": mean_metrics.get("observed_raw_l1"),
        "grad_norm": mean_metrics.get("grad_norm"),
        "total_loss": mean_metrics.get("total_loss"),
        "note": "Stage 1 train-smoke diagnostic metrics only; not formal quality evidence.",
    }


def _timing_payload(*, first_step_s: float | None, total_elapsed_s: float, n_steps: int) -> dict[str, object]:
    n_steps = int(n_steps)
    total_elapsed_s = float(max(0.0, total_elapsed_s))
    first = None if first_step_s is None else float(max(0.0, first_step_s))
    total_fps = float(n_steps / total_elapsed_s) if total_elapsed_s > 0.0 else None
    steady_fps = None
    if n_steps > 1 and first is not None:
        steady_elapsed = max(0.0, total_elapsed_s - first)
        steady_fps = float((n_steps - 1) / steady_elapsed) if steady_elapsed > 0.0 else None
    return {
        "steady_fps_excl_first": steady_fps,
        "total_fps": total_fps,
        "first_frame_s": first,
        "n_frames": n_steps,
        "total_elapsed_s": total_elapsed_s,
        "note": "Stage 1 train-smoke step timing; not inference-only PBU forward timing.",
    }


def main() -> None:
    args = parse_args()
    if not args.dry_run and not args.train_smoke:
        raise RuntimeError(
            "Stage 1 formal training is intentionally disabled in this scaffold. "
            "Use --dry_run for no-checkpoint plumbing smoke or --train_smoke with an explicit output directory; "
            "formal training still needs an approved output/checkpoint plan."
        )
    if bool(args.backward_smoke) and not bool(args.dry_run):
        raise RuntimeError("--backward_smoke is only valid with --dry_run")
    if (args.output_dir is not None or bool(args.save_checkpoint)) and not bool(args.train_smoke):
        raise RuntimeError("--output_dir and --save_checkpoint are only valid with --train_smoke")
    if bool(args.train_smoke) and int(args.max_steps) > int(args.smoke_max_steps_cap):
        raise RuntimeError(
            f"--train_smoke max_steps={int(args.max_steps)} exceeds smoke cap {int(args.smoke_max_steps_cap)}"
        )
    if int(args.heavy_metric_interval) < 0:
        raise RuntimeError("--heavy_metric_interval must be non-negative")
    schedule = _resolve_stage1_train_schedule(args)
    checkpoint_name = _local_checkpoint_name(args.checkpoint_name) if bool(args.save_checkpoint) else None
    output_dir = None
    if bool(args.train_smoke):
        output_dir = _prepare_output_dir(args.output_dir, allow_overwrite=bool(args.allow_overwrite))

    device = torch.device("cuda:0" if args.device == "auto" and torch.cuda.is_available() else "cpu")
    if args.device != "auto":
        device = torch.device(args.device)
    torch.manual_seed(int(args.seed))

    dataset = Stage1EchoNetFrameDataset(
        data_root=_resolve_data_root(args.data_root),
        split=args.split,
        key=args.key,
        max_files=int(args.max_files),
        max_items=int(args.max_items),
        max_frames_per_file=int(args.max_frames_per_file),
    )
    loader = DataLoader(
        dataset,
        batch_size=max(1, int(args.batch_size)),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_stage1_frame_samples,
    )
    first = dataset[0]["target"]
    image_shape = (int(first.shape[-2]), int(first.shape[-1]))
    sampler = VariableMaskBudgetSampler(
        image_shape=image_shape,
        n_lines=int(args.n_lines),
        device=device,
        dtype=torch.float32,
        seed=int(args.seed),
    )
    model = Stage1PBUWrapper(
        n_lines=int(args.n_lines),
        base_channels=int(args.base_channels),
        channel_mult=_parse_channel_mult(args.channel_mult),
        num_res_blocks=int(args.num_res_blocks),
        groupnorm_groups=int(args.groupnorm_groups),
        delta_clip_scale=float(args.delta_clip_scale),
    ).to(device=device)
    optimizer = None
    should_train = bool(args.backward_smoke) or bool(args.train_smoke)
    if should_train:
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr))
        model.train()
    else:
        model.eval()

    records: list[Stage1EvalRecord] = []
    train_metrics: list[dict[str, object]] = []
    max_steps = max(1, int(args.max_steps))
    if bool(args.backward_smoke):
        max_steps = min(max_steps, 1)
    first_step_s = None
    loop_start_s = time.perf_counter()
    loader_iter = iter(loader)
    for step_idx in range(max_steps):
        step_start_s = time.perf_counter()
        try:
            frame_batch = next(loader_iter)
        except StopIteration:
            if not bool(args.train_smoke):
                break
            loader_iter = iter(loader)
            frame_batch = next(loader_iter)
        step_budget, step_mask_family, step_prior_kind = _schedule_item_for_step(
            step_idx=step_idx,
            schedule=schedule,
            sampler=sampler,
        )
        batch = make_stage1_pbu_batch_from_frame_batch(
            frame_batch,
            sampler=sampler,
            budget=int(step_budget),
            mask_family=str(step_mask_family),
            prior_kind=str(step_prior_kind),
            roll=step_idx,
        )
        include_heavy_metrics = _include_heavy_metrics_for_step(
            step_idx,
            interval=int(args.heavy_metric_interval),
            max_steps=max_steps,
        )
        if should_train:
            result = run_stage1_pbu_train_step(
                batch=batch,
                model=model,
                optimizer=optimizer,
                backward=True,
                grad_clip=float(args.grad_clip),
                include_heavy_metrics=include_heavy_metrics,
            )
            output = result.output
            step_metrics = stage1_train_metrics_to_jsonable(result.metrics)
            train_metrics.append(
                {
                    "step": int(step_idx),
                    "budget": int(step_budget),
                    "mask_family": str(step_mask_family),
                    "prior_kind": str(step_prior_kind),
                    "heavy_metrics": bool(include_heavy_metrics),
                    "metrics": step_metrics,
                }
            )
        else:
            with torch.no_grad():
                output = model(batch)
            step_metrics = compute_stage1_eval_metrics(
                batch=batch,
                x_raw=output.x_raw,
                x_final=output.x_final,
                u_post=output.u_post,
                observed_raw_l1=output.observed_raw_l1,
                observed_consistency_l1=output.observed_consistency_l1,
                include_ssim=include_heavy_metrics,
                include_spearman=include_heavy_metrics,
            )
        records.append(
            Stage1EvalRecord(
                budget=int(step_budget),
                mask_family=str(step_mask_family),
                prior_kind=str(step_prior_kind),
                variant=(
                    "real_data_train_smoke"
                    if bool(args.train_smoke)
                    else "real_data_backward_smoke"
                    if bool(args.backward_smoke)
                    else "real_data_dry_run"
                ),
                model_ablation="full",
                metrics=step_metrics,
                metadata={
                    "step": int(step_idx),
                    "schedule_mode": str(schedule["mode"]),
                    "heavy_metrics": bool(include_heavy_metrics),
                },
            )
        )
        step_elapsed_s = time.perf_counter() - step_start_s
        if first_step_s is None:
            first_step_s = float(step_elapsed_s)
    total_elapsed_s = time.perf_counter() - loop_start_s

    checkpoint_path = None
    if bool(args.train_smoke):
        _write_json(output_dir / "args.json", _jsonable_args(args))
        with open(output_dir / "train_metrics.jsonl", "w", encoding="utf-8") as f:
            for item in train_metrics:
                f.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n")
        if bool(args.save_checkpoint):
            checkpoint_path = output_dir / str(checkpoint_name)
            torch.save(
                {
                    "mode": "real_data_train_smoke",
                    "steps": int(len(records)),
                    "args": _jsonable_args(args),
                    "stage1_pbu_wrapper_state_dict": model.state_dict(),
                    "model_state_dict": model.model.state_dict(),
                },
                checkpoint_path,
            )

    payload = {
        "mode": (
            "real_data_train_smoke"
            if bool(args.train_smoke)
            else "real_data_backward_smoke_no_checkpoint"
            if bool(args.backward_smoke)
            else "real_data_dry_run_no_checkpoint"
        ),
        "data_root": str(args.data_root),
        "split": str(args.split),
        "key": str(args.key),
        "dataset_items": int(len(dataset)),
        "steps": int(len(records)),
        "requested_steps": int(max_steps),
        "budget": int(args.budget) if str(schedule["mode"]) == "fixed" else None,
        "mask_family": str(args.mask_family) if str(schedule["mode"]) == "fixed" else None,
        "prior_kind": str(args.prior_kind) if str(schedule["mode"]) == "fixed" else None,
        "schedule": schedule,
        "heavy_metric_interval": int(args.heavy_metric_interval),
        "heavy_metric_steps": [
            int(record.metadata["step"])
            for record in records
            if record.metadata is not None and bool(record.metadata.get("heavy_metrics"))
        ],
        "summary": summarize_stage1_records(records),
        "backward_smoke": bool(args.backward_smoke),
        "train_smoke": bool(args.train_smoke),
        "output_dir": str(output_dir) if output_dir is not None else None,
        "checkpoint": str(checkpoint_path) if checkpoint_path is not None else None,
        "formal_gate_note": (
            "This run validates real-data Stage 1 PBU plumbing only. "
            "Dry-run/backward-smoke modes save no checkpoints or outputs. "
            "Train-smoke mode is bounded and writes only to the declared output_dir. "
            "It is not formal Stage 1 gate evidence."
        ),
    }
    if output_dir is not None:
        basic_metrics = _basic_metrics_from_summary(payload["summary"])
        timing = _timing_payload(
            first_step_s=first_step_s,
            total_elapsed_s=total_elapsed_s,
            n_steps=len(records),
        )
        payload["basic_metrics_path"] = str(output_dir / "basic_metrics.json")
        payload["timing_path"] = str(output_dir / "timing.json")
        _write_json(output_dir / "basic_metrics.json", basic_metrics)
        _write_json(output_dir / "timing.json", timing)
        _write_json(output_dir / "summary.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
