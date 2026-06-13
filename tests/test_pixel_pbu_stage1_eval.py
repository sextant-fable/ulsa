import py_compile
from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile

import h5py
import torch

from ulsa.pixel_pbu_stage1 import (
    MASK_FIXED104,
    MASK_RANDOM,
    MASK_ROLLED_EQUISPACED,
    MODEL_ABLATION_FULL,
    MODEL_ABLATION_NO_BUDGET_TOKEN,
    MODEL_ABLATION_NO_HARD_PROJECTION,
    MODEL_ABLATION_NO_RESIDUAL,
    PRIOR_COPY_LAST,
    PRIOR_WEAK_INTERPOLATION,
    Stage1PBUWrapper,
    VariableMaskBudgetSampler,
    make_stage1_pbu_batch,
)
from research_harness.parsers import summarize_output
from ulsa.pixel_pbu_stage1_eval import (
    STAGE1_EVAL_VARIANT_NO_OBSERVATION,
    STAGE1_EVAL_VARIANT_NORMAL,
    STAGE1_EVAL_VARIANT_SHUFFLED_OBS,
    STAGE1_EVAL_VARIANT_WRONG_MASK,
    compute_stage1_eval_metrics,
    _spearman_1d,
    count_monotonicity_violations,
    run_stage1_contract_eval,
    summarize_stage1_gate_diagnostics,
    summarize_stage1_records,
)


def _target(batch_size=2, height=32, width=32):
    torch.manual_seed(71)
    return torch.randn(batch_size, 1, height, width).clamp(-1.0, 1.0)


def _make_tiny_hdf5(root: Path) -> None:
    split_dir = root / "val"
    split_dir.mkdir(parents=True)
    path = split_dir / "tiny.hdf5"
    values = torch.linspace(-60.0, 0.0, steps=5 * 8 * 8, dtype=torch.float32).reshape(5, 8, 8).numpy()
    with h5py.File(path, "w") as f:
        group = f.create_group("data")
        group.create_dataset("image_sc", data=values)


def test_stage1_eval_modules_compile():
    root = Path(__file__).resolve().parents[1]
    py_compile.compile(str(root / "ulsa/pixel_pbu_stage1_eval.py"), doraise=True)
    py_compile.compile(str(root / "evaluate_pixel_pbu_stage1.py"), doraise=True)


def test_stage1_contract_eval_records_budget_mask_and_variants():
    target = _target()
    sampler = VariableMaskBudgetSampler(image_shape=(32, 32), n_lines=32, seed=73)
    model = Stage1PBUWrapper(
        n_lines=32,
        base_channels=8,
        channel_mult=(1, 2),
        num_res_blocks=1,
        groupnorm_groups=4,
        delta_clip_scale=0.05,
    )
    records = run_stage1_contract_eval(
        model=model,
        target=target,
        sampler=sampler,
        budgets=(0, 2, 14),
        mask_families=(MASK_RANDOM, MASK_ROLLED_EQUISPACED, MASK_FIXED104),
        prior_kind=PRIOR_COPY_LAST,
        variants=(
            STAGE1_EVAL_VARIANT_NORMAL,
            STAGE1_EVAL_VARIANT_SHUFFLED_OBS,
            STAGE1_EVAL_VARIANT_WRONG_MASK,
            STAGE1_EVAL_VARIANT_NO_OBSERVATION,
        ),
        model_ablations=(
            MODEL_ABLATION_FULL,
            MODEL_ABLATION_NO_RESIDUAL,
            MODEL_ABLATION_NO_BUDGET_TOKEN,
            MODEL_ABLATION_NO_HARD_PROJECTION,
        ),
    )
    assert records
    assert any(record.budget == 14 and record.mask_family == MASK_FIXED104 for record in records)
    assert {
        MODEL_ABLATION_FULL,
        MODEL_ABLATION_NO_RESIDUAL,
        MODEL_ABLATION_NO_BUDGET_TOKEN,
        MODEL_ABLATION_NO_HARD_PROJECTION,
    }.issubset({record.model_ablation for record in records})
    for record in records:
        assert "metadata" in record.to_dict()
        assert record.variant in {
            STAGE1_EVAL_VARIANT_NORMAL,
            STAGE1_EVAL_VARIANT_SHUFFLED_OBS,
            STAGE1_EVAL_VARIANT_WRONG_MASK,
            STAGE1_EVAL_VARIANT_NO_OBSERVATION,
        }
        assert record.model_ablation in {
            MODEL_ABLATION_FULL,
            MODEL_ABLATION_NO_RESIDUAL,
            MODEL_ABLATION_NO_BUDGET_TOKEN,
            MODEL_ABLATION_NO_HARD_PROJECTION,
        }
        for key in [
            "psnr",
            "ssim",
            "unobserved_psnr",
            "prior_psnr",
            "observed_consistency_l1",
            "observed_raw_l1",
            "uncertainty_error_spearman",
        ]:
            assert key in record.metrics
        if (
            record.model_ablation == MODEL_ABLATION_NO_HARD_PROJECTION
            and record.budget > 0
            and record.variant != STAGE1_EVAL_VARIANT_NO_OBSERVATION
        ):
            assert record.metrics["observed_consistency_l1"] > 1.0e-8
        else:
            assert record.metrics["observed_consistency_l1"] <= 1.0e-6

    summary = summarize_stage1_records(records)
    assert summary["num_records"] == len(records)
    assert "psnr_monotonicity_violations" in summary
    assert "ssim_monotonicity_violations" in summary
    assert count_monotonicity_violations(records, metric_name="psnr") >= 0
    diagnostics = summarize_stage1_gate_diagnostics(records, required_budgets=(0, 2, 14), target_budget=14)
    assert diagnostics["num_records"] == len(records)
    assert diagnostics["num_primary_records"] > 0
    assert diagnostics["missing_required_budgets"] == []
    assert diagnostics["target_budget"] == 14
    assert diagnostics["target_budget_degradation_deltas"]["shuffled_observation_psnr_drop_vs_normal"] is not None
    assert diagnostics["budget_curve"]
    assert diagnostics["mask_family_table"]
    assert diagnostics["variant_table"]
    assert diagnostics["model_ablation_table"]
    assert diagnostics["mask_family_psnr_gaps"]


def test_stage1_eval_handles_full_budget_and_constant_spearman_without_nan_json():
    target = _target()
    sampler = VariableMaskBudgetSampler(image_shape=(32, 32), n_lines=32, seed=79)
    model = Stage1PBUWrapper(
        n_lines=32,
        base_channels=8,
        channel_mult=(1, 2),
        num_res_blocks=1,
        groupnorm_groups=4,
        delta_clip_scale=0.05,
    )
    records = run_stage1_contract_eval(
        model=model,
        target=target,
        sampler=sampler,
        budgets=(32,),
        mask_families=(MASK_RANDOM,),
        prior_kind=PRIOR_COPY_LAST,
        variants=(STAGE1_EVAL_VARIANT_NORMAL,),
    )
    assert len(records) == 1
    metrics = records[0].metrics
    assert metrics["unobserved_psnr"] is None
    assert metrics["unobserved_l1"] is None
    assert metrics["unobserved_u_post"] is None
    assert metrics["uncertainty_error_spearman"] is None
    assert _spearman_1d(torch.ones(4), torch.arange(4, dtype=torch.float32)) is None
    json.dumps({"summary": summarize_stage1_records(records), "records": [r.to_dict() for r in records]}, allow_nan=False)


def test_stage1_metrics_can_skip_heavy_ssim_and_spearman():
    target = _target()
    sampler = VariableMaskBudgetSampler(image_shape=(32, 32), n_lines=32, seed=81)
    model = Stage1PBUWrapper(
        n_lines=32,
        base_channels=8,
        channel_mult=(1, 2),
        num_res_blocks=1,
        groupnorm_groups=4,
        delta_clip_scale=0.05,
    )
    records = run_stage1_contract_eval(
        model=model,
        target=target,
        sampler=sampler,
        budgets=(2,),
        mask_families=(MASK_RANDOM,),
        prior_kind=PRIOR_COPY_LAST,
        variants=(STAGE1_EVAL_VARIANT_NORMAL,),
    )
    record = records[0]
    # Rebuild the same lightweight metric path directly so train code can skip
    # expensive SSIM/Spearman while keeping PSNR and consistency diagnostics.
    batch = make_stage1_pbu_batch(
        target=target,
        sampler=sampler,
        budget=2,
        mask_family=MASK_RANDOM,
        prior_kind=PRIOR_COPY_LAST,
    )
    output = model(batch)
    metrics = compute_stage1_eval_metrics(
        batch=batch,
        x_raw=output.x_raw,
        x_final=output.x_final,
        u_post=output.u_post,
        observed_raw_l1=output.observed_raw_l1,
        observed_consistency_l1=output.observed_consistency_l1,
        include_ssim=False,
        include_spearman=False,
    )
    assert metrics["psnr"] is not None
    assert metrics["prior_psnr"] is not None
    assert metrics["observed_consistency_l1"] <= 1.0e-6
    assert metrics["ssim"] is None
    assert metrics["prior_ssim"] is None
    assert metrics["ssim_gain_over_prior"] is None
    assert metrics["uncertainty_error_spearman"] is None
    assert record.metrics["ssim"] is not None


def test_stage1_eval_recomputes_observation_dependent_prior_for_no_observation():
    target = _target()
    sampler = VariableMaskBudgetSampler(image_shape=(32, 32), n_lines=32, seed=83)
    model = Stage1PBUWrapper(
        n_lines=32,
        base_channels=8,
        channel_mult=(1, 2),
        num_res_blocks=1,
        groupnorm_groups=4,
        delta_clip_scale=0.05,
    )
    records = run_stage1_contract_eval(
        model=model,
        target=target,
        sampler=sampler,
        budgets=(14,),
        mask_families=(MASK_RANDOM,),
        prior_kind=PRIOR_WEAK_INTERPOLATION,
        variants=(STAGE1_EVAL_VARIANT_NORMAL, STAGE1_EVAL_VARIANT_NO_OBSERVATION),
    )
    assert len(records) == 2
    normal, no_obs = records
    assert normal.variant == STAGE1_EVAL_VARIANT_NORMAL
    assert no_obs.variant == STAGE1_EVAL_VARIANT_NO_OBSERVATION
    assert no_obs.metrics["observed_consistency_l1"] <= 1.0e-6
    assert normal.metrics["prior_psnr"] != no_obs.metrics["prior_psnr"]


def test_stage1_eval_rejects_unsupported_mask_family_before_compatibility_skip():
    target = _target()
    sampler = VariableMaskBudgetSampler(image_shape=(32, 32), n_lines=32, seed=89)
    model = Stage1PBUWrapper(
        n_lines=32,
        base_channels=8,
        channel_mult=(1, 2),
        num_res_blocks=1,
        groupnorm_groups=4,
        delta_clip_scale=0.05,
    )
    try:
        run_stage1_contract_eval(
            model=model,
            target=target,
            sampler=sampler,
            budgets=(0,),
            mask_families=("typo_mask",),
            prior_kind=PRIOR_COPY_LAST,
            variants=(STAGE1_EVAL_VARIANT_NORMAL,),
        )
    except ValueError as exc:
        assert "Unsupported mask family" in str(exc)
    else:
        raise AssertionError("unsupported mask family typo must fail loudly")


def test_stage1_eval_real_data_checkpoint_cli_writes_harness_visible_metrics():
    repo_root = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        data_root = base / "data"
        out = base / "eval_out"
        checkpoint = base / "stage1_wrapper.pt"
        _make_tiny_hdf5(data_root)
        model = Stage1PBUWrapper(
            n_lines=8,
            base_channels=8,
            channel_mult=(1, 2),
            num_res_blocks=1,
            groupnorm_groups=4,
            delta_clip_scale=0.05,
        )
        torch.save(
            {
                "stage1_pbu_wrapper_state_dict": model.state_dict(),
                "model_state_dict": model.model.state_dict(),
            },
            checkpoint,
        )
        env = dict(os.environ)
        env["PYTHONPATH"] = str(repo_root / "ulsa")
        result = subprocess.run(
            [
                sys.executable,
                str(repo_root / "ulsa/evaluate_pixel_pbu_stage1.py"),
                "--real_data",
                "--checkpoint",
                str(checkpoint),
                "--output_dir",
                str(out),
                "--data_root",
                str(data_root),
                "--split",
                "val",
                "--max_files",
                "1",
                "--max_items",
                "2",
                "--max_frames_per_file",
                "2",
                "--batch_size",
                "2",
                "--budgets",
                "0,2",
                "--mask_families",
                MASK_RANDOM,
                "--variants",
                "normal,no_observation",
                "--model_ablations",
                "full,no_hard_projection",
                "--prior_kind",
                PRIOR_COPY_LAST,
                "--n_lines",
                "8",
                "--base_channels",
                "8",
                "--channel_mult",
                "1,2",
                "--num_res_blocks",
                "1",
                "--groupnorm_groups",
                "4",
                "--delta_clip_scale",
                "0.05",
                "--device",
                "cpu",
            ],
            cwd=repo_root,
            env=env,
            text=True,
            capture_output=True,
            check=True,
        )
        summary_stdout = json.loads(result.stdout)
        assert summary_stdout["num_records"] == 8
        assert sorted(path.name for path in out.iterdir()) == [
            "basic_metrics.json",
            "budget_curve.csv",
            "diagnostics.json",
            "eval_records.jsonl",
            "mask_family_psnr_gaps.csv",
            "mask_family_table.csv",
            "model_ablation_table.csv",
            "summary.json",
            "timing.json",
            "variant_table.csv",
        ]
        payload = json.loads((out / "summary.json").read_text())
        assert payload["mode"] == "real_data_checkpoint_eval"
        assert payload["checkpoint"] == str(checkpoint)
        assert payload["real_data"] is True
        assert payload["summary"]["num_records"] == 8
        assert payload["primary_summary"]["num_records"] == 2
        assert payload["diagnostics"]["num_records"] == 8
        assert payload["diagnostics"]["num_primary_records"] == 2
        assert payload["diagnostics"]["missing_required_budgets"] == [4, 7, 10, 14, 21, 28]
        lines = (out / "eval_records.jsonl").read_text().strip().splitlines()
        assert len(lines) == 8
        first_record = json.loads(lines[0])
        assert first_record["metadata"]["batch_index"] == 0
        assert first_record["metadata"]["frame_idx"] == [2, 3]
        assert len(first_record["metadata"]["file_path"]) == 2
        assert (out / "budget_curve.csv").read_text().splitlines()[0]
        assert (out / "model_ablation_table.csv").read_text().splitlines()[0]
        basic_metrics = json.loads((out / "basic_metrics.json").read_text())
        assert basic_metrics["psnr_mean"] == payload["primary_summary"]["mean_metrics"]["psnr"]
        assert "normal/full subset" in basic_metrics["note"]
        timing = json.loads((out / "timing.json").read_text())
        assert timing["n_frames"] == 2
        assert timing["n_batches"] == 1
        assert timing["n_records"] == 8
        assert timing["first_batch_n_frames"] == 2
        assert timing["first_batch_s"] >= 0.0
        assert timing["first_frame_s"] >= 0.0
        assert timing["steady_fps_excl_first"] is None
        parsed = summarize_output(out)
        assert parsed["primary"]["psnr_mean"] == basic_metrics["psnr_mean"]
        assert parsed["primary"]["n_frames"] == 2


def test_stage1_eval_real_data_requires_checkpoint_before_output_dir_creation():
    repo_root = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        data_root = base / "data"
        out = base / "eval_out"
        _make_tiny_hdf5(data_root)
        env = dict(os.environ)
        env["PYTHONPATH"] = str(repo_root / "ulsa")
        result = subprocess.run(
            [
                sys.executable,
                str(repo_root / "ulsa/evaluate_pixel_pbu_stage1.py"),
                "--real_data",
                "--output_dir",
                str(out),
                "--data_root",
                str(data_root),
                "--split",
                "val",
                "--max_files",
                "1",
                "--max_items",
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
        assert "--real_data Stage 1 eval requires an explicit --checkpoint" in result.stderr
        assert not out.exists()


def test_stage1_eval_bad_checkpoint_path_does_not_create_output_dir():
    repo_root = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        data_root = base / "data"
        out = base / "eval_out"
        missing_checkpoint = base / "missing.pt"
        _make_tiny_hdf5(data_root)
        env = dict(os.environ)
        env["PYTHONPATH"] = str(repo_root / "ulsa")
        result = subprocess.run(
            [
                sys.executable,
                str(repo_root / "ulsa/evaluate_pixel_pbu_stage1.py"),
                "--real_data",
                "--checkpoint",
                str(missing_checkpoint),
                "--output_dir",
                str(out),
                "--data_root",
                str(data_root),
                "--split",
                "val",
                "--max_files",
                "1",
                "--max_items",
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
        assert not out.exists()
