"""Multi-file selector validation for standalone Stage2-J pixel flow loops."""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Iterable


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

from evaluate_pixel_rectified_flow_loop_stage2 import (  # noqa: E402
    _load_flow,
    _parse_csv_ints,
    _parse_csv_strings,
    _parse_float_weights,
    _run_one_family,
    _write_csv,
)
from train_pixel_temporal_prior_stage2 import (  # noqa: E402
    _load_pbu,
    _prepare_output_dir,
    _resolve_data_root,
    _write_json,
)
from ulsa.pixel_pbu_stage1 import VariableMaskBudgetSampler  # noqa: E402
from ulsa.pixel_pbu_stage1_dataset import list_stage1_hdf5_files, map_range_tensor  # noqa: E402
from ulsa.pixel_temporal_prior import mean_or_none  # noqa: E402


@dataclass(frozen=True)
class SelectorSpec:
    name: str
    group: str
    family: str
    seed: int
    roll_stride: int = 1
    roll_offset: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-file Stage2-J selector validation.")
    parser.add_argument("--flow_checkpoint", type=str, required=True)
    parser.add_argument("--pbu_checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--allow_overwrite", action="store_true")
    parser.add_argument("--data_root", type=str, default="processed_echonet")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--key", type=str, default="data/image_sc")
    parser.add_argument("--file_offset", type=int, default=0)
    parser.add_argument("--num_files", type=int, default=5)
    parser.add_argument("--history_size", type=int, default=3)
    parser.add_argument("--rollout_frames", type=int, default=171)
    parser.add_argument("--rollout_history_source", type=str, default="pbu", choices=("pbu", "flow_prior", "blend"))
    parser.add_argument("--history_blend_alpha", type=float, default=0.5)
    parser.add_argument("--short_lengths", type=_parse_csv_ints, default=(2, 3, 5, 10, 20, 50, 171))
    parser.add_argument("--budget", type=int, default=14)
    parser.add_argument(
        "--selectors",
        type=_parse_csv_strings,
        default=("fixed104", "rolled_s1", "rolled_s2", "rolled_s3", "rolled_s4", "heuristic_rbf", "random"),
    )
    parser.add_argument("--random_seeds", type=_parse_csv_ints, default=(910, 911, 912, 913, 914))
    parser.add_argument("--n_lines", type=int, default=112)
    parser.add_argument("--seed", type=int, default=914)
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


def _sample_std_or_none(values: Iterable[Any]) -> float | None:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return None
    if len(vals) == 1:
        return 0.0
    mean = sum(vals) / len(vals)
    return float((sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5)


def _min_or_none(values: Iterable[Any]) -> float | None:
    vals = [float(v) for v in values if v is not None]
    return min(vals) if vals else None


def _max_or_none(values: Iterable[Any]) -> float | None:
    vals = [float(v) for v in values if v is not None]
    return max(vals) if vals else None


def _first_bad_min(values: Iterable[Any]) -> int | None:
    vals = [int(v) for v in values if v is not None and str(v) != ""]
    return min(vals) if vals else None


def _json_safe_args(args: argparse.Namespace) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in vars(args).items():
        if isinstance(value, tuple):
            out[key] = list(value)
        elif isinstance(value, Path):
            out[key] = str(value)
        else:
            out[key] = value
    return out


def _write_source_snapshot(output_dir: Path) -> list[str]:
    snapshot_dir = output_dir / "source_snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    sources = [
        Path(__file__).resolve(),
        THIS_DIR / "evaluate_pixel_rectified_flow_loop_stage2.py",
        THIS_DIR / "train_pixel_rectified_flow_prior_stage2.py",
        THIS_DIR / "train_pixel_temporal_prior_stage2.py",
        THIS_DIR / "ulsa" / "pixel_acquisition_stage3.py",
        THIS_DIR / "ulsa" / "pixel_measurement.py",
        THIS_DIR / "ulsa" / "pixel_pbu_stage1.py",
        THIS_DIR / "ulsa" / "pixel_rectified_flow_prior.py",
        THIS_DIR / "ulsa" / "pixel_temporal_prior.py",
        THIS_DIR / "ulsa" / "pixel_timing.py",
    ]
    copied: list[str] = []
    for source in sources:
        if not source.exists():
            continue
        rel = source.relative_to(THIS_DIR)
        destination = snapshot_dir / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        copied.append(str(rel))
    return copied


def _load_sequence_file(path: Path, args: argparse.Namespace) -> torch.Tensor:
    with h5py.File(path, "r") as f:
        data = torch.as_tensor(f[str(args.key)][:], dtype=torch.float32)
    return map_range_tensor(data, source_range=(-60.0, 0.0), target_range=(-1.0, 1.0)).unsqueeze(1)


def _selector_specs(args: argparse.Namespace) -> list[SelectorSpec]:
    specs: list[SelectorSpec] = []
    for raw in args.selectors:
        token = str(raw).strip()
        if not token:
            continue
        if token == "fixed104":
            specs.append(SelectorSpec(name="fixed104", group="fixed104", family="fixed104", seed=int(args.seed)))
        elif token == "heuristic_rbf":
            specs.append(
                SelectorSpec(name="heuristic_rbf", group="heuristic_rbf", family="heuristic_rbf", seed=int(args.seed))
            )
        elif token in {"random", "random_b14"}:
            for seed in args.random_seeds:
                specs.append(
                    SelectorSpec(
                        name=f"random_b{int(args.budget)}_seed{int(seed)}",
                        group=f"random_b{int(args.budget)}",
                        family="random",
                        seed=int(seed),
                    )
                )
        elif token.startswith("rolled_s"):
            stride = int(token.removeprefix("rolled_s"))
            specs.append(
                SelectorSpec(
                    name=f"rolled_s{stride}",
                    group=f"rolled_s{stride}",
                    family="rolled_equispaced",
                    seed=int(args.seed),
                    roll_stride=stride,
                )
            )
        elif token == "rolled_equispaced":
            specs.append(
                SelectorSpec(
                    name=f"rolled_s{int(args.seed)}",
                    group="rolled_equispaced",
                    family="rolled_equispaced",
                    seed=int(args.seed),
                )
            )
        else:
            raise ValueError(f"Unsupported selector token {token!r}")
    if not specs:
        raise ValueError("No selector specs requested")
    return specs


def _run_args(args: argparse.Namespace, spec: SelectorSpec) -> argparse.Namespace:
    run_args = argparse.Namespace(**vars(args))
    run_args.seed = int(spec.seed)
    run_args.roll_stride = int(spec.roll_stride)
    run_args.roll_offset = int(spec.roll_offset)
    return run_args


def _per_sample_row(
    *,
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
    spec: SelectorSpec,
    file_index: int,
    file_path: Path,
) -> dict[str, Any]:
    timing = summary.get("timing", {})
    return {
        "file_index": int(file_index),
        "file_name": file_path.name,
        "file_path": str(file_path),
        "selector_name": str(spec.name),
        "selector_group": str(spec.group),
        "mask_family": str(spec.family),
        "seed": int(spec.seed),
        "roll_stride": int(spec.roll_stride),
        "roll_offset": int(spec.roll_offset),
        "budget": int(summary["budget"]),
        "frames": int(summary["frames"]),
        "psnr_mean": summary["rollout_pbu_psnr_mean"],
        "ssim_mean": mean_or_none([row["rollout_pbu_ssim"] for row in rows]),
        "flow_prior_psnr_mean": summary["flow_prior_psnr_mean"],
        "teacher_forced_psnr_mean": summary["teacher_forced_pbu_psnr_mean"],
        "drop_vs_teacher_mean": summary["mean_drop_vs_teacher_forced"],
        "first_bad_frame_drop_gt_1db": summary["first_bad_frame_drop_gt_1db"],
        "first10_psnr": summary["first_10_psnr"],
        "last10_psnr": summary["last_10_psnr"],
        "drop_first10_to_last10": summary["drop_first10_to_last10"],
        "frames_1_5_psnr": summary["frames_1_5_psnr"],
        "frames_16_20_psnr": summary["frames_16_20_psnr"],
        "drop_frames_1_5_to_16_20": summary["drop_frames_1_5_to_16_20"],
        "steady_fps_excl_first": timing.get("steady_fps_excl_first"),
        "max_observed_consistency_l1": summary["max_observed_consistency_l1"],
        "selected_count_all_match_budget": summary["selected_count_all_match_budget"],
    }


def _aggregate(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)

    metric_cols = (
        "psnr_mean",
        "ssim_mean",
        "flow_prior_psnr_mean",
        "teacher_forced_psnr_mean",
        "drop_vs_teacher_mean",
        "first10_psnr",
        "last10_psnr",
        "drop_first10_to_last10",
        "drop_frames_1_5_to_16_20",
        "steady_fps_excl_first",
    )
    out: list[dict[str, Any]] = []
    for group_key, group_rows in sorted(groups.items(), key=lambda item: tuple(str(x) for x in item[0])):
        item = {key: value for key, value in zip(keys, group_key, strict=True)}
        item["n"] = len(group_rows)
        item["file_count"] = len({row["file_index"] for row in group_rows})
        item["seed_count"] = len({row["seed"] for row in group_rows})
        for col in metric_cols:
            vals = [row.get(col) for row in group_rows]
            item[f"{col}_mean"] = mean_or_none(vals)
            item[f"{col}_std"] = _sample_std_or_none(vals)
            item[f"{col}_min"] = _min_or_none(vals)
            item[f"{col}_max"] = _max_or_none(vals)
        item["first_bad_frame_min"] = _first_bad_min(row.get("first_bad_frame_drop_gt_1db") for row in group_rows)
        item["first_bad_frame_mean"] = mean_or_none(
            [row.get("first_bad_frame_drop_gt_1db") for row in group_rows if row.get("first_bad_frame_drop_gt_1db")]
        )
        item["all_selected_count_match_budget"] = all(bool(row["selected_count_all_match_budget"]) for row in group_rows)
        item["max_observed_consistency_l1"] = _max_or_none(row.get("max_observed_consistency_l1") for row in group_rows)
        out.append(item)
    return out


def main() -> None:
    args = parse_args()
    device = torch.device("cuda:0" if args.device == "auto" and torch.cuda.is_available() else "cpu")
    if args.device != "auto":
        device = torch.device(args.device)
    output_dir = _prepare_output_dir(args.output_dir, allow_overwrite=bool(args.allow_overwrite))

    files = list_stage1_hdf5_files(_resolve_data_root(args.data_root), split=str(args.split))
    start = int(args.file_offset)
    stop = start + int(args.num_files)
    selected_files = files[start:stop]
    if not selected_files:
        raise RuntimeError(f"No files selected from split={args.split!r} offset={start} num_files={args.num_files}")

    specs = _selector_specs(args)
    flow_model, ckpt = _load_flow(args, device=device)
    pbu_model = _load_pbu(args, device=device)
    if pbu_model is None:
        raise RuntimeError("multi-file selector validation requires --pbu_checkpoint")

    all_frame_rows: list[dict[str, Any]] = []
    per_sample_rows: list[dict[str, Any]] = []
    actual_frames_by_file: list[dict[str, Any]] = []
    for local_index, file_path in enumerate(selected_files):
        file_index = start + local_index
        data = _load_sequence_file(file_path, args)
        actual_frames_by_file.append(
            {
                "file_index": int(file_index),
                "file_name": file_path.name,
                "file_path": str(file_path),
                "source_frames": int(data.shape[0]),
                "target_frames": int(min(int(args.rollout_frames), int(data.shape[0]) - int(args.history_size))),
            }
        )
        for spec in specs:
            run_args = _run_args(args, spec)
            torch.manual_seed(int(spec.seed))
            sampler = VariableMaskBudgetSampler(
                image_shape=(int(data.shape[-2]), int(data.shape[-1])),
                n_lines=int(args.n_lines),
                device=device,
                dtype=torch.float32,
                seed=int(spec.seed),
            )
            rows, summary = _run_one_family(
                args=run_args,
                family=str(spec.family),
                data=data,
                flow_model=flow_model,
                pbu_model=pbu_model,
                sampler=sampler,
                device=device,
            )
            per_sample_rows.append(
                _per_sample_row(
                    summary=summary,
                    rows=rows,
                    spec=spec,
                    file_index=file_index,
                    file_path=file_path,
                )
            )
            for row in rows:
                item = dict(row)
                item.update(
                    {
                        "file_index": int(file_index),
                        "file_name": file_path.name,
                        "file_path": str(file_path),
                        "selector_name": str(spec.name),
                        "selector_group": str(spec.group),
                        "seed": int(spec.seed),
                        "roll_stride": int(spec.roll_stride),
                        "roll_offset": int(spec.roll_offset),
                    }
                )
                all_frame_rows.append(item)
            print(
                json.dumps(
                    {
                        "file_index": file_index,
                        "file_name": file_path.name,
                        "selector": spec.name,
                        "psnr": summary["rollout_pbu_psnr_mean"],
                        "ssim": per_sample_rows[-1]["ssim_mean"],
                        "fps": per_sample_rows[-1]["steady_fps_excl_first"],
                        "first_bad": summary["first_bad_frame_drop_gt_1db"],
                    },
                    ensure_ascii=False,
                    allow_nan=False,
                ),
                flush=True,
            )

    selector_aggregates = _aggregate(per_sample_rows, ("selector_name",))
    group_aggregates = _aggregate(per_sample_rows, ("selector_group",))
    per_file_group_aggregates = _aggregate(per_sample_rows, ("file_index", "selector_group"))
    source_snapshot = _write_source_snapshot(output_dir)

    _write_csv(output_dir / "rectified_flow_loop_frame_metrics.csv", all_frame_rows)
    _write_csv(output_dir / "selector_per_sample_metrics.csv", per_sample_rows)
    _write_csv(output_dir / "selector_aggregate_metrics.csv", selector_aggregates)
    _write_csv(output_dir / "selector_group_aggregate_metrics.csv", group_aggregates)
    _write_csv(output_dir / "selector_per_file_group_metrics.csv", per_file_group_aggregates)

    basic = {
        "psnr_mean": mean_or_none([row["psnr_mean"] for row in per_sample_rows]),
        "ssim_mean": mean_or_none([row["ssim_mean"] for row in per_sample_rows]),
        "flow_prior_psnr_mean": mean_or_none([row["flow_prior_psnr_mean"] for row in per_sample_rows]),
        "teacher_forced_psnr_mean": mean_or_none([row["teacher_forced_psnr_mean"] for row in per_sample_rows]),
        "rollout_drop_psnr_mean": mean_or_none([row["drop_vs_teacher_mean"] for row in per_sample_rows]),
        "observed_consistency_l1": _max_or_none(row["max_observed_consistency_l1"] for row in per_sample_rows),
        "first_bad_frame": _first_bad_min(row["first_bad_frame_drop_gt_1db"] for row in per_sample_rows),
        "aggregation_unit": "per_file_selector_seed_run",
        "aggregation_note": (
            "Top-level metrics are sweep/global means across all per-sample selector specs. "
            "Use selector_group_aggregate_metrics.csv for selector claims."
        ),
    }
    timing = {
        "steady_fps_excl_first": mean_or_none([row["steady_fps_excl_first"] for row in per_sample_rows]),
        "note": "Mean over per-file/per-selector online full-loop runs; teacher-forced branch is excluded by inner profiler.",
        "selector_group_steady_fps_excl_first": {
            row["selector_group"]: row["steady_fps_excl_first_mean"] for row in group_aggregates
        },
    }
    summary_payload = {
        "mode": "stage2j_pixel_rectified_flow_selector_multifile_validation",
        "args": _json_safe_args(args),
        "flow_checkpoint": str(args.flow_checkpoint),
        "pbu_checkpoint": str(args.pbu_checkpoint),
        "flow_checkpoint_keys": sorted(ckpt.keys()) if isinstance(ckpt, dict) else None,
        "data_root": str(args.data_root),
        "split": str(args.split),
        "file_offset": int(args.file_offset),
        "num_files": int(len(selected_files)),
        "files": [str(path) for path in selected_files],
        "actual_frames_by_file": actual_frames_by_file,
        "budget": int(args.budget),
        "rollout_frames": int(args.rollout_frames),
        "selectors": [spec.__dict__ for spec in specs],
        "random_seeds": [int(seed) for seed in args.random_seeds],
        "selector_aggregates": selector_aggregates,
        "selector_group_aggregates": group_aggregates,
        "no_base_prior_checkpoint": True,
        "no_model_training": True,
        "max_observed_consistency_l1": basic["observed_consistency_l1"],
        "source_snapshot_dir": str(output_dir / "source_snapshot"),
        "source_snapshot_files": source_snapshot,
        "top_level_basic_metrics_are_sweep_means": True,
    }
    _write_json(output_dir / "basic_metrics.json", basic)
    _write_json(output_dir / "timing.json", timing)
    _write_json(output_dir / "summary.json", summary_payload)
    print(json.dumps(summary_payload, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
