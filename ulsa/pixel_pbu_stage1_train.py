from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
import torch.nn.functional as F

from ulsa.pixel_pbu_stage1 import Stage1PBUBatch, Stage1PBUOutput, Stage1PBUWrapper
from ulsa.pixel_pbu_stage1_eval import compute_stage1_eval_metrics


@dataclass(frozen=True)
class Stage1PBULossWeights:
    reconstruction_unobserved: float = 1.0
    observed_raw_consistency: float = 0.1
    uncertainty_nll: float = 0.01
    observed_uncertainty: float = 0.01
    delta_l1: float = 0.001


@dataclass(frozen=True)
class Stage1TrainStepResult:
    output: Stage1PBUOutput
    total_loss: torch.Tensor
    losses: dict[str, torch.Tensor]
    metrics: dict[str, float | None]
    grad_norm: float | None = None


def _safe_masked_mean(value: torch.Tensor, mask: torch.Tensor, *, eps: float = 1.0e-6) -> torch.Tensor:
    denom = torch.sum(mask)
    if float(denom.detach().cpu().item()) <= 0.0:
        return torch.mean(value * 0.0)
    return torch.sum(value * mask) / denom.clamp_min(float(eps))


def compute_stage1_pbu_losses(
    *,
    batch: Stage1PBUBatch,
    output: Stage1PBUOutput,
    weights: Stage1PBULossWeights = Stage1PBULossWeights(),
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    unobs = 1.0 - batch.mask
    abs_error = torch.abs(output.x_final - batch.x_gt)
    loss_unobs = _safe_masked_mean(abs_error, unobs)
    loss_obs_raw = _safe_masked_mean(torch.abs(output.x_raw - batch.obs), batch.mask)

    # Use logvar directly for observed uncertainty so this loss still has a gradient
    # even though output.u_post is hard-zeroed on observed pixels after projection.
    observed_uncertainty = _safe_masked_mean(F.softplus(output.logvar), batch.mask)
    variance = F.softplus(output.logvar).clamp_min(1.0e-6)
    nll_map = abs_error / variance + torch.log1p(variance)
    loss_uncertainty = _safe_masked_mean(nll_map, unobs)
    loss_delta = torch.mean(torch.abs(output.delta_x))

    losses = {
        "loss_unobserved_l1": loss_unobs,
        "loss_observed_raw_l1": loss_obs_raw,
        "loss_uncertainty_nll": loss_uncertainty,
        "loss_observed_uncertainty": observed_uncertainty,
        "loss_delta_l1": loss_delta,
    }
    total = (
        float(weights.reconstruction_unobserved) * loss_unobs
        + float(weights.observed_raw_consistency) * loss_obs_raw
        + float(weights.uncertainty_nll) * loss_uncertainty
        + float(weights.observed_uncertainty) * observed_uncertainty
        + float(weights.delta_l1) * loss_delta
    )
    losses["total_loss"] = total
    return total, losses


def _grad_norm(parameters) -> float:
    sq_sum = 0.0
    for param in parameters:
        if param.grad is None:
            continue
        if not torch.isfinite(param.grad).all():
            raise FloatingPointError("Non-finite gradient encountered in Stage 1 backward smoke")
        value = float(torch.sum(param.grad.detach().float() ** 2).cpu().item())
        if not math.isfinite(value):
            raise FloatingPointError("Non-finite gradient norm contribution in Stage 1 backward smoke")
        sq_sum += value
    return float(math.sqrt(sq_sum))


def run_stage1_pbu_train_step(
    *,
    batch: Stage1PBUBatch,
    model: Stage1PBUWrapper,
    optimizer: torch.optim.Optimizer | None = None,
    weights: Stage1PBULossWeights = Stage1PBULossWeights(),
    backward: bool = False,
    grad_clip: float | None = 1.0,
    input_range: tuple[float, float] = (-1.0, 1.0),
    include_heavy_metrics: bool = True,
) -> Stage1TrainStepResult:
    if backward and optimizer is None:
        raise ValueError("optimizer is required when backward=True")
    model.train(mode=bool(backward))
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)

    output = model(batch)
    total_loss, losses = compute_stage1_pbu_losses(batch=batch, output=output, weights=weights)
    grad_norm = None
    if backward:
        total_loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=float(grad_clip),
                error_if_nonfinite=True,
            )
        grad_norm = _grad_norm(model.parameters())
        optimizer.step()

    metrics = compute_stage1_eval_metrics(
        batch=batch,
        x_raw=output.x_raw,
        x_final=output.x_final,
        u_post=output.u_post,
        observed_raw_l1=output.observed_raw_l1,
        observed_consistency_l1=output.observed_consistency_l1,
        input_range=input_range,
        include_ssim=bool(include_heavy_metrics),
        include_spearman=bool(include_heavy_metrics),
    )
    for key, value in losses.items():
        metrics[key] = float(value.detach().cpu().item())
    if grad_norm is not None:
        metrics["grad_norm"] = float(grad_norm)
    return Stage1TrainStepResult(
        output=output,
        total_loss=total_loss,
        losses=losses,
        metrics=metrics,
        grad_norm=grad_norm,
    )


def stage1_train_metrics_to_jsonable(metrics: dict[str, Any]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for key, value in metrics.items():
        if value is None:
            out[key] = None
            continue
        numeric = float(value)
        out[key] = numeric if math.isfinite(numeric) else None
    return out
