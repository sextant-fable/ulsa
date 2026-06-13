from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import torch
import torch.nn.functional as F

from ulsa.pixel_pbu_stage1 import (
    MASK_CLUSTERED_LOCAL,
    MASK_HEURISTIC,
    MASK_RANDOM,
    MASK_ROLLED_EQUISPACED,
    MODEL_ABLATION_FULL,
    PRIOR_CACHED_DPF,
    PRIOR_COPY_LAST,
    PRIOR_WEAK_INTERPOLATION,
    PRIOR_ZERO_FILL,
    PixelPriorMixture,
    STAGE1_BUDGETS,
    SUPPORTED_STAGE1_MASK_FAMILIES,
    SUPPORTED_STAGE1_MODEL_ABLATIONS,
    SUPPORTED_STAGE1_PRIORS,
    Stage1PBUBatch,
    Stage1PBUWrapper,
    VariableMaskBudgetSampler,
    make_shuffled_observation_batch,
    make_stage1_pbu_batch,
    make_wrong_mask_batch,
)


STAGE1_EVAL_VARIANT_NORMAL = "normal"
STAGE1_EVAL_VARIANT_SHUFFLED_OBS = "shuffled_observation"
STAGE1_EVAL_VARIANT_WRONG_MASK = "wrong_mask"
STAGE1_EVAL_VARIANT_NO_OBSERVATION = "no_observation"
SUPPORTED_STAGE1_EVAL_VARIANTS = (
    STAGE1_EVAL_VARIANT_NORMAL,
    STAGE1_EVAL_VARIANT_SHUFFLED_OBS,
    STAGE1_EVAL_VARIANT_WRONG_MASK,
    STAGE1_EVAL_VARIANT_NO_OBSERVATION,
)


@dataclass(frozen=True)
class Stage1EvalRecord:
    budget: int
    mask_family: str
    prior_kind: str
    variant: str
    model_ablation: str
    metrics: dict[str, float | None]
    metadata: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "budget": int(self.budget),
            "mask_family": str(self.mask_family),
            "prior_kind": str(self.prior_kind),
            "variant": str(self.variant),
            "model_ablation": str(self.model_ablation),
            "metrics": dict(self.metrics),
            "metadata": None if self.metadata is None else dict(self.metadata),
        }


def _masked_mean(value: torch.Tensor, mask: torch.Tensor, *, eps: float = 1.0e-6) -> torch.Tensor | None:
    denom = torch.sum(mask)
    if float(denom.detach().cpu().item()) <= 0.0:
        return None
    return torch.sum(value * mask) / denom.clamp_min(float(eps))


def _psnr_from_mse(mse: torch.Tensor, *, data_range: float) -> torch.Tensor:
    return 20.0 * torch.log10(torch.as_tensor(float(data_range), device=mse.device, dtype=mse.dtype)) - 10.0 * torch.log10(
        mse.clamp_min(1.0e-12)
    )


def _ssim_torch(x: torch.Tensor, y: torch.Tensor, *, data_range: float) -> torch.Tensor:
    c1 = (0.01 * float(data_range)) ** 2
    c2 = (0.03 * float(data_range)) ** 2
    mu_x = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
    mu_y = F.avg_pool2d(y, kernel_size=3, stride=1, padding=1)
    sigma_x = F.avg_pool2d(x * x, kernel_size=3, stride=1, padding=1) - mu_x * mu_x
    sigma_y = F.avg_pool2d(y * y, kernel_size=3, stride=1, padding=1) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(x * y, kernel_size=3, stride=1, padding=1) - mu_x * mu_y
    numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
    return torch.mean(numerator / denominator.clamp_min(1.0e-12))


def _rankdata_1d(values: torch.Tensor) -> torch.Tensor:
    values = values.flatten()
    sorted_values, order = torch.sort(values)
    ranks_sorted = torch.empty(values.numel(), device=values.device, dtype=torch.float32)
    start = 0
    while start < int(values.numel()):
        end = start + 1
        while end < int(values.numel()) and bool(sorted_values[end] == sorted_values[start]):
            end += 1
        rank = 0.5 * float(start + end - 1)
        ranks_sorted[start:end] = rank
        start = end
    ranks = torch.empty_like(ranks_sorted)
    ranks[order] = ranks_sorted
    return ranks


def _spearman_1d(x: torch.Tensor, y: torch.Tensor) -> float | None:
    x = x.flatten()
    y = y.flatten()
    valid = torch.isfinite(x) & torch.isfinite(y)
    x = x[valid]
    y = y[valid]
    if int(x.numel()) < 2:
        return None
    if bool(torch.all(x == x[0])) or bool(torch.all(y == y[0])):
        return None
    rx = _rankdata_1d(x)
    ry = _rankdata_1d(y)
    rx = rx - torch.mean(rx)
    ry = ry - torch.mean(ry)
    denom = torch.sqrt(torch.sum(rx * rx) * torch.sum(ry * ry)).clamp_min(1.0e-12)
    if float(denom.detach().cpu().item()) <= 0.0:
        return None
    return float((torch.sum(rx * ry) / denom).detach().cpu().item())


def _recompute_observation_dependent_prior(batch: Stage1PBUBatch) -> Stage1PBUBatch:
    if batch.prior_kind not in {PRIOR_ZERO_FILL, PRIOR_WEAK_INTERPOLATION}:
        return batch
    x_prior = PixelPriorMixture.make(
        kind=batch.prior_kind,
        target=batch.x_gt,
        prev_x_final=batch.prev_x_final,
        obs=batch.obs,
        mask=batch.mask,
    )
    u_prior = torch.clamp(torch.abs(batch.prev_x_final - x_prior), min=0.0)
    return batch.replace(x_prior=x_prior, u_prior=u_prior)


def apply_stage1_eval_variant(batch: Stage1PBUBatch, variant: str) -> Stage1PBUBatch:
    if variant == STAGE1_EVAL_VARIANT_NORMAL:
        return _recompute_observation_dependent_prior(batch)
    if variant == STAGE1_EVAL_VARIANT_SHUFFLED_OBS:
        return _recompute_observation_dependent_prior(make_shuffled_observation_batch(batch))
    if variant == STAGE1_EVAL_VARIANT_WRONG_MASK:
        return _recompute_observation_dependent_prior(make_wrong_mask_batch(batch))
    if variant == STAGE1_EVAL_VARIANT_NO_OBSERVATION:
        return _recompute_observation_dependent_prior(
            batch.replace(obs=torch.zeros_like(batch.obs), mask=torch.zeros_like(batch.mask))
        )
    raise ValueError(f"Unsupported Stage 1 eval variant {variant!r}")


def compute_stage1_eval_metrics(
    *,
    batch: Stage1PBUBatch,
    x_raw: torch.Tensor,
    x_final: torch.Tensor,
    u_post: torch.Tensor,
    observed_raw_l1: torch.Tensor,
    observed_consistency_l1: torch.Tensor,
    input_range: tuple[float, float] = (-1.0, 1.0),
    include_ssim: bool = True,
    include_spearman: bool = True,
) -> dict[str, float | None]:
    data_range = max(float(input_range[1] - input_range[0]), 1.0e-6)
    error = torch.abs(x_final - batch.x_gt)
    sq_error = (x_final - batch.x_gt) ** 2
    raw_sq_error = (x_raw - batch.x_gt) ** 2
    unobs = 1.0 - batch.mask

    mse = torch.mean(sq_error)
    raw_mse = torch.mean(raw_sq_error)
    unobs_mse = _masked_mean(sq_error, unobs)
    prior_mse = torch.mean((batch.x_prior - batch.x_gt) ** 2)
    psnr = _psnr_from_mse(mse, data_range=data_range)
    raw_psnr = _psnr_from_mse(raw_mse, data_range=data_range)
    unobs_psnr = _psnr_from_mse(unobs_mse, data_range=data_range) if unobs_mse is not None else None
    prior_psnr = _psnr_from_mse(prior_mse, data_range=data_range)
    ssim = _ssim_torch(x_final, batch.x_gt, data_range=data_range) if bool(include_ssim) else None
    prior_ssim = _ssim_torch(batch.x_prior, batch.x_gt, data_range=data_range) if bool(include_ssim) else None
    spearman = _spearman_1d(u_post[unobs > 0], error[unobs > 0]) if bool(include_spearman) else None

    unobserved_l1 = _masked_mean(error, unobs)
    unobserved_u_post = _masked_mean(u_post, unobs)
    metrics: dict[str, float | None] = {
        "psnr": float(psnr.detach().cpu().item()),
        "ssim": float(ssim.detach().cpu().item()) if ssim is not None else None,
        "unobserved_psnr": float(unobs_psnr.detach().cpu().item()) if unobs_psnr is not None else None,
        "raw_psnr": float(raw_psnr.detach().cpu().item()),
        "prior_psnr": float(prior_psnr.detach().cpu().item()),
        "prior_ssim": float(prior_ssim.detach().cpu().item()) if prior_ssim is not None else None,
        "psnr_gain_over_prior": float((psnr - prior_psnr).detach().cpu().item()),
        "ssim_gain_over_prior": (
            float((ssim - prior_ssim).detach().cpu().item())
            if ssim is not None and prior_ssim is not None
            else None
        ),
        "observed_consistency_l1": float(observed_consistency_l1.detach().cpu().item()),
        "observed_raw_l1": float(observed_raw_l1.detach().cpu().item()),
        "mean_abs_error": float(torch.mean(error).detach().cpu().item()),
        "unobserved_l1": float(unobserved_l1.detach().cpu().item()) if unobserved_l1 is not None else None,
        "mean_u_post": float(torch.mean(u_post).detach().cpu().item()),
        "unobserved_u_post": float(unobserved_u_post.detach().cpu().item()) if unobserved_u_post is not None else None,
        "uncertainty_error_spearman": spearman,
    }
    return metrics


def _mean_or_none(values: Iterable[float | None]) -> float | None:
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not clean:
        return None
    return float(sum(clean) / len(clean))


def _records_mean(records: Iterable[Stage1EvalRecord], metric_name: str) -> float | None:
    return _mean_or_none(record.metrics.get(metric_name) for record in records)


def _metric_summary_for_records(records: list[Stage1EvalRecord]) -> dict[str, float | None]:
    keys = (
        "psnr",
        "ssim",
        "unobserved_psnr",
        "raw_psnr",
        "prior_psnr",
        "observed_consistency_l1",
        "observed_raw_l1",
        "mean_abs_error",
        "uncertainty_error_spearman",
        "mean_u_post",
        "unobserved_u_post",
    )
    return {key: _records_mean(records, key) for key in keys}


def _model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def count_monotonicity_violations(records: list[Stage1EvalRecord], *, metric_name: str) -> int:
    normal_records = [
        record
        for record in records
        if record.variant == STAGE1_EVAL_VARIANT_NORMAL and record.model_ablation == "full"
    ]
    by_budget: dict[int, list[float]] = {}
    for record in normal_records:
        value = record.metrics.get(metric_name)
        if value is not None:
            by_budget.setdefault(int(record.budget), []).append(value)
    ordered = sorted((budget, _mean_or_none(values)) for budget, values in by_budget.items())
    finite = [(budget, value) for budget, value in ordered if value is not None and math.isfinite(value)]
    return sum(1 for (_prev_b, prev), (_cur_b, cur) in zip(finite, finite[1:]) if cur + 1.0e-8 < prev)


def summarize_stage1_records(records: list[Stage1EvalRecord]) -> dict[str, object]:
    metrics: dict[str, list[float | None]] = {}
    for record in records:
        for key, value in record.metrics.items():
            metrics.setdefault(key, []).append(None if value is None else float(value))
    return {
        "num_records": int(len(records)),
        "mean_metrics": {key: _mean_or_none(values) for key, values in sorted(metrics.items())},
        "psnr_monotonicity_violations": int(count_monotonicity_violations(records, metric_name="psnr")),
        "ssim_monotonicity_violations": int(count_monotonicity_violations(records, metric_name="ssim")),
    }


def _primary_records(records: list[Stage1EvalRecord]) -> list[Stage1EvalRecord]:
    return [
        record
        for record in records
        if record.variant == STAGE1_EVAL_VARIANT_NORMAL and record.model_ablation == MODEL_ABLATION_FULL
    ]


def _table_by_fields(records: list[Stage1EvalRecord], fields: tuple[str, ...]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[Stage1EvalRecord]] = {}
    for record in records:
        key = tuple(getattr(record, field) for field in fields)
        grouped.setdefault(key, []).append(record)
    rows: list[dict[str, object]] = []
    for key, group in sorted(grouped.items(), key=lambda item: tuple(str(part) for part in item[0])):
        row = {field: key[idx] for idx, field in enumerate(fields)}
        row["num_records"] = int(len(group))
        row["metrics"] = _metric_summary_for_records(group)
        rows.append(row)
    return rows


def _budget_curve(primary: list[Stage1EvalRecord]) -> list[dict[str, object]]:
    rows = []
    by_budget: dict[int, list[Stage1EvalRecord]] = {}
    for record in primary:
        by_budget.setdefault(int(record.budget), []).append(record)
    for budget in sorted(by_budget):
        group = by_budget[budget]
        metrics = _metric_summary_for_records(group)
        rows.append(
            {
                "budget": int(budget),
                "num_records": int(len(group)),
                "psnr": metrics.get("psnr"),
                "ssim": metrics.get("ssim"),
                "uncertainty_error_spearman": metrics.get("uncertainty_error_spearman"),
                "observed_consistency_l1": metrics.get("observed_consistency_l1"),
            }
        )
    return rows


def _degradation_deltas(records: list[Stage1EvalRecord], *, budget: int = 14) -> dict[str, float | None]:
    normal = [
        record
        for record in records
        if int(record.budget) == int(budget)
        and record.variant == STAGE1_EVAL_VARIANT_NORMAL
        and record.model_ablation == MODEL_ABLATION_FULL
    ]
    normal_psnr = _records_mean(normal, "psnr")
    out: dict[str, float | None] = {}
    for variant in (STAGE1_EVAL_VARIANT_SHUFFLED_OBS, STAGE1_EVAL_VARIANT_WRONG_MASK, STAGE1_EVAL_VARIANT_NO_OBSERVATION):
        variant_records = [
            record
            for record in records
            if int(record.budget) == int(budget)
            and record.variant == variant
            and record.model_ablation == MODEL_ABLATION_FULL
        ]
        variant_psnr = _records_mean(variant_records, "psnr")
        out[f"{variant}_psnr_drop_vs_normal"] = (
            float(normal_psnr - variant_psnr) if normal_psnr is not None and variant_psnr is not None else None
        )
    for ablation in ("no_residual", "no_budget_token", "no_hard_projection"):
        ablation_records = [
            record
            for record in records
            if int(record.budget) == int(budget)
            and record.variant == STAGE1_EVAL_VARIANT_NORMAL
            and record.model_ablation == ablation
        ]
        ablation_psnr = _records_mean(ablation_records, "psnr")
        out[f"{ablation}_psnr_drop_vs_full"] = (
            float(normal_psnr - ablation_psnr) if normal_psnr is not None and ablation_psnr is not None else None
        )
    return out


def _mask_family_gaps(primary: list[Stage1EvalRecord]) -> list[dict[str, object]]:
    rows = []
    target_families = (MASK_RANDOM, MASK_ROLLED_EQUISPACED, MASK_CLUSTERED_LOCAL)
    by_budget: dict[int, list[Stage1EvalRecord]] = {}
    for record in primary:
        by_budget.setdefault(int(record.budget), []).append(record)
    for budget in sorted(by_budget):
        group = by_budget[budget]
        mean_psnr = _records_mean(group, "psnr")
        for family in target_families:
            family_records = [record for record in group if record.mask_family == family]
            family_psnr = _records_mean(family_records, "psnr")
            rows.append(
                {
                    "budget": int(budget),
                    "mask_family": str(family),
                    "psnr": family_psnr,
                    "psnr_gap_vs_budget_average": (
                        float(family_psnr - mean_psnr) if family_psnr is not None and mean_psnr is not None else None
                    ),
                    "num_records": int(len(family_records)),
                }
            )
    return rows


def summarize_stage1_gate_diagnostics(
    records: list[Stage1EvalRecord],
    *,
    required_budgets: Iterable[int] = STAGE1_BUDGETS,
    target_budget: int = 14,
) -> dict[str, object]:
    primary = _primary_records(records)
    present_budgets = sorted({int(record.budget) for record in primary})
    required = [int(budget) for budget in required_budgets]
    observed_values = [
        record.metrics.get("observed_consistency_l1")
        for record in records
        if record.metrics.get("observed_consistency_l1") is not None
    ]
    return {
        "note": (
            "Diagnostic summaries for Stage 1 hard-gate auditing. These are not a pass/fail verdict "
            "and must be interpreted with checkpoint, split, timing, review, and evidence-audit context."
        ),
        "primary_filter": {"variant": STAGE1_EVAL_VARIANT_NORMAL, "model_ablation": MODEL_ABLATION_FULL},
        "num_records": int(len(records)),
        "num_primary_records": int(len(primary)),
        "required_budgets": required,
        "present_required_budgets": [budget for budget in required if budget in present_budgets],
        "missing_required_budgets": [budget for budget in required if budget not in present_budgets],
        "budget_curve": _budget_curve(primary),
        "psnr_monotonicity_violations": int(count_monotonicity_violations(primary, metric_name="psnr")),
        "ssim_monotonicity_violations": int(count_monotonicity_violations(primary, metric_name="ssim")),
        "mask_family_table": _table_by_fields(primary, ("budget", "mask_family")),
        "variant_table": _table_by_fields(
            [record for record in records if record.model_ablation == MODEL_ABLATION_FULL],
            ("budget", "variant"),
        ),
        "model_ablation_table": _table_by_fields(
            [record for record in records if record.variant == STAGE1_EVAL_VARIANT_NORMAL],
            ("budget", "model_ablation"),
        ),
        "mask_family_psnr_gaps": _mask_family_gaps(primary),
        "target_budget": int(target_budget),
        "target_budget_degradation_deltas": _degradation_deltas(records, budget=int(target_budget)),
        "max_observed_consistency_l1_all_records": max((float(v) for v in observed_values), default=None),
        "primary_uncertainty_error_spearman_mean": _records_mean(primary, "uncertainty_error_spearman"),
    }


def run_stage1_contract_eval(
    *,
    model: Stage1PBUWrapper,
    target: torch.Tensor,
    sampler: VariableMaskBudgetSampler,
    budgets: Iterable[int],
    mask_families: Iterable[str] = SUPPORTED_STAGE1_MASK_FAMILIES,
    prior_kind: str = PRIOR_COPY_LAST,
    variants: Iterable[str] = (STAGE1_EVAL_VARIANT_NORMAL,),
    model_ablations: Iterable[str] = ("full",),
    prev_x_final: torch.Tensor | None = None,
    prev_prev_x_final: torch.Tensor | None = None,
    cached_dpf_prior: torch.Tensor | None = None,
    input_range: tuple[float, float] = (-1.0, 1.0),
    record_metadata: dict[str, object] | None = None,
) -> list[Stage1EvalRecord]:
    if prior_kind not in SUPPORTED_STAGE1_PRIORS:
        raise ValueError(f"Unsupported prior kind {prior_kind!r}")
    if prior_kind == PRIOR_CACHED_DPF and cached_dpf_prior is None:
        raise ValueError("cached_dpf prior requires cached_dpf_prior for Stage 1 eval")
    for family in mask_families:
        if family not in SUPPORTED_STAGE1_MASK_FAMILIES:
            raise ValueError(f"Unsupported mask family {family!r}")
    for variant in variants:
        if variant not in SUPPORTED_STAGE1_EVAL_VARIANTS:
            raise ValueError(f"Unsupported eval variant {variant!r}")
    for model_ablation in model_ablations:
        if model_ablation not in SUPPORTED_STAGE1_MODEL_ABLATIONS:
            raise ValueError(f"Unsupported Stage 1 model ablation {model_ablation!r}")
    records: list[Stage1EvalRecord] = []
    device = _model_device(model)
    if sampler.generator.device != device:
        raise ValueError(
            "Stage 1 eval requires sampler and model on the same device, "
            f"got sampler={sampler.generator.device} and model={device}"
        )
    target = target.to(device=device, dtype=sampler.generator.dtype)
    if prev_x_final is None:
        heuristic_map = torch.zeros_like(target)
    else:
        heuristic_map = torch.abs(prev_x_final.to(device=device, dtype=sampler.generator.dtype))
    for budget in budgets:
        compatible = set(sampler.compatible_families_for_budget(int(budget)))
        for family in mask_families:
            if family not in compatible:
                continue
            base_batch = make_stage1_pbu_batch(
                target=target,
                sampler=sampler,
                budget=int(budget),
                mask_family=family,
                prior_kind=prior_kind,
                prev_x_final=prev_x_final,
                prev_prev_x_final=prev_prev_x_final,
                cached_dpf_prior=cached_dpf_prior,
                heuristic_map=heuristic_map if family == MASK_HEURISTIC else None,
            )
            for variant in variants:
                batch = apply_stage1_eval_variant(base_batch, variant)
                for model_ablation in model_ablations:
                    with torch.no_grad():
                        output = model(batch, ablation=model_ablation)
                    records.append(
                        Stage1EvalRecord(
                            budget=int(budget),
                            mask_family=str(family),
                            prior_kind=str(prior_kind),
                            variant=str(variant),
                            model_ablation=str(model_ablation),
                            metrics=compute_stage1_eval_metrics(
                                batch=batch,
                                x_raw=output.x_raw,
                                x_final=output.x_final,
                                u_post=output.u_post,
                                observed_raw_l1=output.observed_raw_l1,
                                observed_consistency_l1=output.observed_consistency_l1,
                                input_range=input_range,
                            ),
                            metadata=record_metadata,
                        )
                    )
    return records
