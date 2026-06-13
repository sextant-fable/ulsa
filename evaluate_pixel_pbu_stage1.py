"""Dry-run and evaluate the Stage 1 variable-mask PBU contract.

This script is intentionally separate from the older Stage 3A replay-cache
PixelBeliefUpdater evaluator. Synthetic mode validates contract plumbing.
Real-data mode evaluates an explicit Stage 1 checkpoint on EchoNet frames and
can write harness-visible smoke metrics.
"""

from __future__ import annotations

import argparse
import csv
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
    PRIOR_COPY_LAST,
    STAGE1_BUDGETS,
    SUPPORTED_STAGE1_MASK_FAMILIES,
    SUPPORTED_STAGE1_MODEL_ABLATIONS,
    SUPPORTED_STAGE1_PRIORS,
    Stage1PBUWrapper,
    VariableMaskBudgetSampler,
)
from ulsa.pixel_pbu_stage1_dataset import (
    Stage1EchoNetFrameDataset,
    collate_stage1_frame_samples,
)
from ulsa.pixel_pbu_stage1_eval import (
    STAGE1_EVAL_VARIANT_NORMAL,
    SUPPORTED_STAGE1_EVAL_VARIANTS,
    run_stage1_contract_eval,
    summarize_stage1_gate_diagnostics,
    summarize_stage1_records,
)


def _parse_csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def _parse_csv_strings(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Stage 1 variable-mask PBU contract.")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--allow_overwrite", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--synthetic", action="store_true", help="Use deterministic synthetic tensors.")
    mode.add_argument("--real_data", action="store_true", help="Evaluate a checkpoint on real EchoNet frame tensors.")
    parser.add_argument("--data_root", type=str, default="processed_echonet")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--key", type=str, default="data/image_sc")
    parser.add_argument("--max_files", type=int, default=1)
    parser.add_argument("--max_items", type=int, default=2)
    parser.add_argument("--max_frames_per_file", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--n_lines", type=int, default=32)
    parser.add_argument("--seed", type=int, default=61)
    parser.add_argument("--budgets", type=_parse_csv_ints, default=STAGE1_BUDGETS)
    parser.add_argument(
        "--mask_families",
        type=_parse_csv_strings,
        default=SUPPORTED_STAGE1_MASK_FAMILIES,
    )
    parser.add_argument("--prior_kind", type=str, default=PRIOR_COPY_LAST, choices=SUPPORTED_STAGE1_PRIORS)
    parser.add_argument(
        "--variants",
        type=_parse_csv_strings,
        default=(STAGE1_EVAL_VARIANT_NORMAL,),
        help="Comma-separated variants: normal,shuffled_observation,wrong_mask,no_observation.",
    )
    parser.add_argument(
        "--model_ablations",
        type=_parse_csv_strings,
        default=("full",),
        help="Comma-separated model ablations: full,no_residual,no_budget_token,no_hard_projection.",
    )
    parser.add_argument("--base_channels", type=int, default=8)
    parser.add_argument("--channel_mult", type=_parse_csv_ints, default=(1, 2))
    parser.add_argument("--num_res_blocks", type=int, default=1)
    parser.add_argument("--groupnorm_groups", type=int, default=4)
    parser.add_argument("--delta_clip_scale", type=float, default=0.05)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def _make_synthetic_target(*, batch_size: int, height: int, width: int, seed: int, device: torch.device) -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    noise = torch.randn((int(batch_size), 1, int(height), int(width)), generator=gen, dtype=torch.float32)
    x = torch.linspace(-1.0, 1.0, steps=int(width), dtype=torch.float32).view(1, 1, 1, int(width))
    y = torch.linspace(-1.0, 1.0, steps=int(height), dtype=torch.float32).view(1, 1, int(height), 1)
    pattern = torch.sin(3.0 * x) * torch.cos(2.0 * y)
    target = 0.65 * pattern + 0.15 * noise
    return target.clamp(-1.0, 1.0).to(device=device)


def _load_checkpoint_if_requested(model: Stage1PBUWrapper, checkpoint: str | None) -> str | None:
    if checkpoint is None:
        return None
    path = Path(checkpoint)
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "stage1_pbu_wrapper_state_dict" in ckpt:
        model.load_state_dict(ckpt["stage1_pbu_wrapper_state_dict"], strict=True)
    elif isinstance(ckpt, dict) and "pixel_belief_updater_state_dict" in ckpt:
        state_dict = ckpt["pixel_belief_updater_state_dict"]
        model.model.load_state_dict(state_dict, strict=True)
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.model.load_state_dict(ckpt["model_state_dict"], strict=True)
    else:
        try:
            model.load_state_dict(ckpt, strict=True)
        except RuntimeError:
            model.model.load_state_dict(ckpt, strict=True)
    return str(path)


def _resolve_data_root(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    repo_path = REPO_ROOT / path
    if repo_path.exists():
        return repo_path
    return path


def _prepare_output_dir(value: str | None, *, allow_overwrite: bool) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if path.exists() and any(path.iterdir()) and not bool(allow_overwrite):
        raise RuntimeError(f"Refusing to write Stage 1 eval outputs into non-empty directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_json(path: Path, payload: dict[str, object]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, allow_nan=False)


def _flatten_row(row: dict[str, object]) -> dict[str, object]:
    out: dict[str, object] = {}
    for key, value in row.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                out[f"{key}_{sub_key}"] = sub_value
        else:
            out[key] = value
    return out


def _write_csv_rows(path: Path, rows: list[dict[str, object]]) -> None:
    flat_rows = [_flatten_row(row) for row in rows]
    fieldnames = sorted({key for row in flat_rows for key in row.keys()})
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in flat_rows:
            writer.writerow(row)


def _mean_metrics(summary: dict[str, object]) -> dict[str, object]:
    metrics = summary.get("mean_metrics", {})
    return metrics if isinstance(metrics, dict) else {}


def _primary_eval_records(records: list) -> tuple[list, bool]:
    primary = [record for record in records if record.variant == STAGE1_EVAL_VARIANT_NORMAL and record.model_ablation == "full"]
    if primary:
        return primary, True
    return list(records), False


def _basic_metrics_from_summary(summary: dict[str, object], *, note: str) -> dict[str, object]:
    metrics = _mean_metrics(summary)
    return {
        "psnr_mean": metrics.get("psnr"),
        "psnr_std": None,
        "ssim_mean": metrics.get("ssim"),
        "ssim_std": None,
        "mse_mean": None,
        "mae_mean": metrics.get("mean_abs_error"),
        "observed_consistency_l1": metrics.get("observed_consistency_l1"),
        "observed_raw_l1": metrics.get("observed_raw_l1"),
        "uncertainty_error_spearman": metrics.get("uncertainty_error_spearman"),
        "note": note,
    }


def _timing_payload(
    *,
    first_batch_s: float | None,
    first_batch_n_frames: int,
    total_elapsed_s: float,
    n_frames: int,
    n_batches: int,
    n_records: int,
) -> dict[str, object]:
    n_frames = int(n_frames)
    first_batch_n_frames = max(0, int(first_batch_n_frames))
    elapsed = float(max(0.0, total_elapsed_s))
    first = None if first_batch_s is None else float(max(0.0, first_batch_s))
    first_frame = None
    if first is not None and first_batch_n_frames > 0:
        first_frame = float(first / first_batch_n_frames)
    total_fps = float(n_frames / elapsed) if elapsed > 0.0 else None
    steady_fps = None
    steady_n_frames = n_frames - first_batch_n_frames
    if steady_n_frames > 0 and first is not None:
        steady_elapsed = max(0.0, elapsed - first)
        steady_fps = float(steady_n_frames / steady_elapsed) if steady_elapsed > 0.0 else None
    return {
        "steady_fps_excl_first": steady_fps,
        "total_fps": total_fps,
        "first_frame_s": first_frame,
        "first_batch_s": first,
        "first_batch_n_frames": first_batch_n_frames,
        "n_frames": n_frames,
        "n_batches": int(n_batches),
        "n_records": int(n_records),
        "total_elapsed_s": elapsed,
        "note": (
            "Stage 1 eval-loop timing over input frames, including data loading, mask construction, "
            "model forward sweeps, and metrics. n_records counts budget/mask/variant/ablation eval records."
        ),
    }


def _make_model(args: argparse.Namespace, *, n_lines: int, device: torch.device) -> Stage1PBUWrapper:
    return Stage1PBUWrapper(
        n_lines=int(n_lines),
        base_channels=int(args.base_channels),
        channel_mult=tuple(int(x) for x in args.channel_mult),
        num_res_blocks=int(args.num_res_blocks),
        groupnorm_groups=int(args.groupnorm_groups),
        delta_clip_scale=float(args.delta_clip_scale),
    ).to(device=device)


def _run_synthetic_eval(args: argparse.Namespace, *, device: torch.device) -> tuple[Stage1PBUWrapper, list, str | None, dict]:
    target = _make_synthetic_target(
        batch_size=int(args.batch_size),
        height=int(args.height),
        width=int(args.width),
        seed=int(args.seed),
        device=device,
    )
    sampler = VariableMaskBudgetSampler(
        image_shape=(int(args.height), int(args.width)),
        n_lines=int(args.n_lines),
        device=device,
        dtype=torch.float32,
        seed=int(args.seed),
    )
    model = _make_model(args, n_lines=int(args.n_lines), device=device)
    checkpoint = _load_checkpoint_if_requested(model, args.checkpoint)
    model.eval()
    eval_start_s = time.perf_counter()
    records = run_stage1_contract_eval(
        model=model,
        target=target,
        sampler=sampler,
        budgets=tuple(int(x) for x in args.budgets),
        mask_families=tuple(str(x) for x in args.mask_families),
        prior_kind=str(args.prior_kind),
        variants=tuple(str(x) for x in args.variants),
        model_ablations=tuple(str(x) for x in args.model_ablations),
    )
    eval_elapsed_s = time.perf_counter() - eval_start_s
    return model, records, checkpoint, {
        "n_frames": int(args.batch_size),
        "n_batches": 1,
        "first_batch_s": float(eval_elapsed_s),
        "first_batch_n_frames": int(args.batch_size),
    }


def _run_real_data_eval(args: argparse.Namespace, *, device: torch.device) -> tuple[Stage1PBUWrapper, list, str, dict]:
    if args.checkpoint is None:
        raise RuntimeError("--real_data Stage 1 eval requires an explicit --checkpoint")
    dataset = Stage1EchoNetFrameDataset(
        data_root=_resolve_data_root(args.data_root),
        split=str(args.split),
        key=str(args.key),
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
    model = _make_model(args, n_lines=int(args.n_lines), device=device)
    checkpoint = _load_checkpoint_if_requested(model, args.checkpoint)
    model.eval()

    records = []
    n_batches = 0
    first_batch_s = None
    first_batch_n_frames = 0
    for frame_batch in loader:
        batch_start_s = time.perf_counter()
        record_metadata = {
            "batch_index": int(n_batches),
            "file_path": [str(path) for path in frame_batch["file_path"]],
            "frame_idx": [int(idx) for idx in frame_batch["frame_idx"]],
            "n_frames": [int(value) for value in frame_batch["n_frames"]],
            "split": [str(value) for value in frame_batch["split"]],
        }
        records.extend(
            run_stage1_contract_eval(
                model=model,
                target=frame_batch["target"],
                sampler=sampler,
                budgets=tuple(int(x) for x in args.budgets),
                mask_families=tuple(str(x) for x in args.mask_families),
                prior_kind=str(args.prior_kind),
                variants=tuple(str(x) for x in args.variants),
                model_ablations=tuple(str(x) for x in args.model_ablations),
                prev_x_final=frame_batch["prev_x_final"],
                prev_prev_x_final=frame_batch["prev_prev_x_final"],
                record_metadata=record_metadata,
            )
        )
        batch_elapsed_s = time.perf_counter() - batch_start_s
        n_batches += 1
        if first_batch_s is None:
            first_batch_s = float(batch_elapsed_s)
            first_batch_n_frames = int(frame_batch["target"].shape[0])
    return model, records, checkpoint, {
        "n_frames": int(len(dataset)),
        "n_batches": int(n_batches),
        "first_batch_s": first_batch_s,
        "first_batch_n_frames": int(first_batch_n_frames) if first_batch_s is not None else 0,
    }


def main() -> None:
    args = parse_args()
    for family in args.mask_families:
        if family not in SUPPORTED_STAGE1_MASK_FAMILIES:
            raise ValueError(f"Unsupported mask family {family!r}")
    for variant in args.variants:
        if variant not in SUPPORTED_STAGE1_EVAL_VARIANTS:
            raise ValueError(f"Unsupported eval variant {variant!r}")
    for model_ablation in args.model_ablations:
        if model_ablation not in SUPPORTED_STAGE1_MODEL_ABLATIONS:
            raise ValueError(f"Unsupported Stage 1 model ablation {model_ablation!r}")
    if not args.synthetic and not args.real_data:
        raise ValueError("Choose either --synthetic or --real_data for Stage 1 eval.")
    if bool(args.real_data) and args.checkpoint is None:
        raise RuntimeError("--real_data Stage 1 eval requires an explicit --checkpoint")

    device = torch.device("cuda:0" if args.device == "auto" and torch.cuda.is_available() else "cpu")
    if args.device != "auto":
        device = torch.device(args.device)
    torch.manual_seed(int(args.seed))

    start_s = time.perf_counter()
    if bool(args.real_data):
        _model, records, checkpoint, timing_info = _run_real_data_eval(args, device=device)
        mode = "real_data_checkpoint_eval"
    else:
        _model, records, checkpoint, timing_info = _run_synthetic_eval(args, device=device)
        mode = "synthetic_contract_smoke" if checkpoint is None else "synthetic_checkpoint_contract"
    total_elapsed_s = time.perf_counter() - start_s
    summary = summarize_stage1_records(records)
    primary_records, has_primary_subset = _primary_eval_records(records)
    primary_summary = summarize_stage1_records(primary_records)
    diagnostics = summarize_stage1_gate_diagnostics(records)
    output_dir = _prepare_output_dir(args.output_dir, allow_overwrite=bool(args.allow_overwrite))
    payload = {
        "mode": mode,
        "checkpoint": checkpoint,
        "synthetic": bool(args.synthetic),
        "real_data": bool(args.real_data),
        "data_root": str(args.data_root) if bool(args.real_data) else None,
        "split": str(args.split) if bool(args.real_data) else None,
        "key": str(args.key) if bool(args.real_data) else None,
        "max_files": int(args.max_files) if bool(args.real_data) else None,
        "max_items": int(args.max_items) if bool(args.real_data) else None,
        "max_frames_per_file": int(args.max_frames_per_file) if bool(args.real_data) else None,
        "budgets": [int(x) for x in args.budgets],
        "mask_families": [str(x) for x in args.mask_families],
        "variants": [str(x) for x in args.variants],
        "model_ablations": [str(x) for x in args.model_ablations],
        "prior_kind": str(args.prior_kind),
        "n_lines": int(args.n_lines),
        "image_shape": [int(args.height), int(args.width)] if bool(args.synthetic) else None,
        "summary": summary,
        "primary_record_filter": {"variant": STAGE1_EVAL_VARIANT_NORMAL, "model_ablation": "full"},
        "primary_summary": primary_summary,
        "diagnostics": diagnostics,
        "records": [record.to_dict() for record in records],
        "formal_gate_note": (
            "This Stage 1 eval validates evaluator plumbing and can support gate evidence only when "
            "run with a trained checkpoint, real held-out split, harness record, ablations, timing profile, "
            "review, and evidence audit."
        ),
    }
    if args.output_json is not None:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        _write_json(out, payload)
    if output_dir is not None:
        basic_note = (
            "Stage 1 eval diagnostic metrics from the primary normal/full subset. "
            "Formal gate evidence requires trained checkpoint, held-out split, review, and audit."
        )
        if not has_primary_subset:
            basic_note = (
                "Stage 1 eval diagnostic metrics. No normal/full subset was available, so these are aggregate metrics. "
                "Formal gate evidence requires trained checkpoint, held-out split, review, and audit."
            )
        basic_metrics = _basic_metrics_from_summary(primary_summary, note=basic_note)
        timing = _timing_payload(
            first_batch_s=timing_info.get("first_batch_s"),
            first_batch_n_frames=int(timing_info.get("first_batch_n_frames", 0)),
            total_elapsed_s=total_elapsed_s,
            n_frames=int(timing_info.get("n_frames", 0)),
            n_batches=int(timing_info.get("n_batches", 0)),
            n_records=len(records),
        )
        payload["basic_metrics_path"] = str(output_dir / "basic_metrics.json")
        payload["timing_path"] = str(output_dir / "timing.json")
        payload["diagnostics_path"] = str(output_dir / "diagnostics.json")
        _write_json(output_dir / "basic_metrics.json", basic_metrics)
        _write_json(output_dir / "timing.json", timing)
        _write_json(output_dir / "diagnostics.json", diagnostics)
        _write_csv_rows(output_dir / "budget_curve.csv", diagnostics["budget_curve"])
        _write_csv_rows(output_dir / "mask_family_table.csv", diagnostics["mask_family_table"])
        _write_csv_rows(output_dir / "variant_table.csv", diagnostics["variant_table"])
        _write_csv_rows(output_dir / "model_ablation_table.csv", diagnostics["model_ablation_table"])
        _write_csv_rows(output_dir / "mask_family_psnr_gaps.csv", diagnostics["mask_family_psnr_gaps"])
        _write_json(output_dir / "summary.json", payload)
        with open(output_dir / "eval_records.jsonl", "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record.to_dict(), ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
