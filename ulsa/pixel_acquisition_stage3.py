from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

from ulsa.pixel_measurement import LineMaskGenerator
from ulsa.pixel_temporal_prior import rankdata_1d, spearman_1d


def _as_nchw(name: str, value: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim == 3:
        value = value[:, None, :, :]
    if value.ndim != 4 or int(value.shape[1]) != 1:
        raise ValueError(f"{name} must have shape [B,1,H,W] or [B,H,W], got {tuple(value.shape)}")
    return value


def edge_magnitude(value: torch.Tensor) -> torch.Tensor:
    value = _as_nchw("value", value)
    dx = F.pad(torch.abs(value[..., :, 1:] - value[..., :, :-1]), (0, 1, 0, 0))
    dy = F.pad(torch.abs(value[..., 1:, :] - value[..., :-1, :]), (0, 0, 0, 1))
    return dx + dy


def normalize_line_scores(scores: torch.Tensor) -> torch.Tensor:
    if scores.ndim == 1:
        scores = scores.view(1, -1)
    lo = torch.amin(scores, dim=1, keepdim=True)
    hi = torch.amax(scores, dim=1, keepdim=True)
    return (scores - lo) / (hi - lo).clamp_min(1.0e-6)


def heuristic_line_scores(
    *,
    generator: LineMaskGenerator,
    u_prior: torch.Tensor,
    change_map: torch.Tensor | None = None,
    prior_image: torch.Tensor | None = None,
    line_history: torch.Tensor | None = None,
    previous_residual_ema: torch.Tensor | None = None,
    weights: dict[str, float] | None = None,
) -> torch.Tensor:
    weights = {
        "uncertainty": 1.0,
        "change": 0.35,
        "edge": 0.20,
        "previous_residual": 0.20,
        "history": 0.35,
        **({} if weights is None else {str(k): float(v) for k, v in weights.items()}),
    }
    u_prior = _as_nchw("u_prior", u_prior)
    scores = weights["uncertainty"] * normalize_line_scores(generator.linewise_mean(torch.abs(u_prior)))
    if change_map is not None:
        scores = scores + weights["change"] * normalize_line_scores(
            generator.linewise_mean(torch.abs(_as_nchw("change_map", change_map)))
        )
    if prior_image is not None:
        scores = scores + weights["edge"] * normalize_line_scores(
            generator.linewise_mean(edge_magnitude(_as_nchw("prior_image", prior_image)))
        )
    if previous_residual_ema is not None:
        scores = scores + weights["previous_residual"] * normalize_line_scores(
            generator.linewise_mean(torch.abs(_as_nchw("previous_residual_ema", previous_residual_ema)))
        )
    if line_history is not None:
        if line_history.ndim == 1:
            line_history = line_history.view(1, -1)
        if tuple(line_history.shape) != tuple(scores.shape):
            raise ValueError(f"line_history must match score shape, got {tuple(line_history.shape)} and {tuple(scores.shape)}")
        scores = scores - weights["history"] * normalize_line_scores(line_history.to(device=scores.device, dtype=scores.dtype))
    return scores


def rbf_greedy_topk(
    scores: torch.Tensor,
    *,
    k: int,
    available: torch.Tensor | None = None,
    sigma: float = 3.0,
    suppression: float = 0.45,
) -> torch.Tensor:
    if scores.ndim == 1:
        scores = scores.view(1, -1)
    if scores.ndim != 2:
        raise ValueError(f"scores must have shape [B,L], got {tuple(scores.shape)}")
    b, n_lines = int(scores.shape[0]), int(scores.shape[1])
    k = int(k)
    if k < 0 or k > n_lines:
        raise ValueError(f"k must be in [0,{n_lines}], got {k}")
    if available is None:
        available = torch.ones_like(scores, dtype=torch.bool)
    else:
        available = available.to(device=scores.device, dtype=torch.bool)
        if tuple(available.shape) != tuple(scores.shape):
            raise ValueError(f"available must match scores, got {tuple(available.shape)} and {tuple(scores.shape)}")
    selected = torch.zeros_like(available, dtype=torch.bool)
    working = scores.clone()
    line_pos = torch.arange(n_lines, device=scores.device, dtype=scores.dtype).view(1, -1)
    sigma = max(float(sigma), 1.0e-6)
    for _ in range(k):
        masked = torch.where(available & ~selected, working, torch.full_like(working, -torch.inf))
        idx = torch.argmax(masked, dim=1)
        finite = torch.isfinite(masked.gather(1, idx.view(b, 1))).view(b)
        if not bool(torch.any(finite)):
            break
        selected[finite, idx[finite]] = True
        center = idx.to(dtype=scores.dtype).view(b, 1)
        penalty = float(suppression) * torch.exp(-0.5 * ((line_pos - center) / sigma) ** 2)
        working = torch.where(finite.view(b, 1), working - penalty, working)
    return selected


def selected_indices_from_mask(selected: torch.Tensor) -> torch.Tensor:
    if selected.ndim != 2:
        raise ValueError(f"selected must be [B,L], got {tuple(selected.shape)}")
    rows = []
    max_count = int(torch.sum(selected, dim=1).max().detach().cpu().item()) if selected.numel() else 0
    for batch_idx in range(int(selected.shape[0])):
        idx = torch.nonzero(selected[batch_idx], as_tuple=False).flatten()
        if int(idx.numel()) < max_count:
            pad = torch.full((max_count - int(idx.numel()),), -1, device=selected.device, dtype=torch.long)
            idx = torch.cat([idx, pad], dim=0)
        rows.append(idx.to(dtype=torch.long))
    return torch.stack(rows, dim=0) if rows else torch.empty((0, 0), device=selected.device, dtype=torch.long)


def action_from_heuristic_scores(
    *,
    generator: LineMaskGenerator,
    scores: torch.Tensor,
    budget: int,
    available: torch.Tensor | None = None,
    sigma: float = 3.0,
    suppression: float = 0.45,
):
    selected = rbf_greedy_topk(
        scores.to(device=generator.device),
        k=int(budget),
        available=available,
        sigma=float(sigma),
        suppression=float(suppression),
    )
    indices = selected_indices_from_mask(selected)
    return generator.action_from_indices(indices, batch_size=int(selected.shape[0]))


@dataclass(frozen=True)
class OracleRankingMetrics:
    spearman: float | None
    topb_recall: float | None
    ndcg_at_b: float | None
    selected_gain: float | None
    oracle_gain: float | None
    random_gain: float | None
    equispaced_gain: float | None
    oracle_regret: float | None
    random_regret: float | None
    equispaced_regret: float | None

    def to_dict(self) -> dict[str, float | None]:
        return {
            "spearman": self.spearman,
            "topb_recall": self.topb_recall,
            "ndcg_at_b": self.ndcg_at_b,
            "selected_gain": self.selected_gain,
            "oracle_gain": self.oracle_gain,
            "random_gain": self.random_gain,
            "equispaced_gain": self.equispaced_gain,
            "oracle_regret": self.oracle_regret,
            "random_regret": self.random_regret,
            "equispaced_regret": self.equispaced_regret,
        }


def _dcg(gains: torch.Tensor) -> torch.Tensor:
    if int(gains.numel()) == 0:
        return torch.zeros((), device=gains.device, dtype=gains.dtype)
    discounts = torch.log2(torch.arange(2, int(gains.numel()) + 2, device=gains.device, dtype=gains.dtype))
    return torch.sum(gains / discounts)


def _mean_or_none(values: list[float | None]) -> float | None:
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not clean:
        return None
    return float(sum(clean) / len(clean))


def oracle_ranking_metrics(
    *,
    scores: torch.Tensor,
    oracle_gain: torch.Tensor,
    selected: torch.Tensor,
    budget: int,
    random_selected: torch.Tensor | None = None,
    equispaced_selected: torch.Tensor | None = None,
) -> OracleRankingMetrics:
    if scores.ndim == 1:
        scores = scores.view(1, -1)
    if oracle_gain.ndim == 1:
        oracle_gain = oracle_gain.view(1, -1)
    if selected.ndim == 1:
        selected = selected.view(1, -1)
    if tuple(scores.shape) != tuple(oracle_gain.shape) or tuple(selected.shape) != tuple(scores.shape):
        raise ValueError("scores, oracle_gain, and selected must share [B,L] shape")
    b, n_lines = int(scores.shape[0]), int(scores.shape[1])
    k = min(max(0, int(budget)), n_lines)
    spearman_values: list[float | None] = []
    recall_values: list[float | None] = []
    ndcg_values: list[float | None] = []
    selected_gain_values: list[float | None] = []
    oracle_gain_values: list[float | None] = []
    random_gain_values: list[float | None] = []
    equi_gain_values: list[float | None] = []
    for batch_idx in range(b):
        score_row = scores[batch_idx]
        gain_row = oracle_gain[batch_idx]
        selected_row = selected[batch_idx].to(dtype=torch.bool)
        spearman_values.append(spearman_1d(score_row, gain_row))
        if k == 0:
            continue
        oracle_idx = torch.topk(gain_row, k=k, dim=0).indices
        oracle_mask = torch.zeros_like(selected_row)
        oracle_mask[oracle_idx] = True
        overlap = torch.sum(selected_row & oracle_mask).to(dtype=torch.float32)
        recall_values.append(float((overlap / float(k)).detach().cpu().item()))
        selected_gains = gain_row[selected_row]
        selected_gain_values.append(float(torch.sum(selected_gains).detach().cpu().item()))
        oracle_gains = gain_row[oracle_mask]
        oracle_gain_values.append(float(torch.sum(oracle_gains).detach().cpu().item()))
        selected_order = torch.argsort(score_row[selected_row], descending=True)
        selected_sorted = selected_gains[selected_order] if int(selected_gains.numel()) else selected_gains
        ideal_sorted = torch.sort(gain_row, descending=True).values[:k]
        denom = _dcg(ideal_sorted).clamp_min(1.0e-12)
        ndcg_values.append(float((_dcg(selected_sorted[:k]) / denom).detach().cpu().item()))
        if random_selected is not None:
            random_gain_values.append(float(torch.sum(gain_row[random_selected[batch_idx].to(dtype=torch.bool)]).detach().cpu().item()))
        if equispaced_selected is not None:
            equi_gain_values.append(float(torch.sum(gain_row[equispaced_selected[batch_idx].to(dtype=torch.bool)]).detach().cpu().item()))
    selected_gain = _mean_or_none(selected_gain_values)
    oracle_gain_mean = _mean_or_none(oracle_gain_values)
    random_gain = _mean_or_none(random_gain_values)
    equi_gain = _mean_or_none(equi_gain_values)
    return OracleRankingMetrics(
        spearman=_mean_or_none(spearman_values),
        topb_recall=_mean_or_none(recall_values),
        ndcg_at_b=_mean_or_none(ndcg_values),
        selected_gain=selected_gain,
        oracle_gain=oracle_gain_mean,
        random_gain=random_gain,
        equispaced_gain=equi_gain,
        oracle_regret=(
            float(oracle_gain_mean - selected_gain) if oracle_gain_mean is not None and selected_gain is not None else None
        ),
        random_regret=(
            float(oracle_gain_mean - random_gain) if oracle_gain_mean is not None and random_gain is not None else None
        ),
        equispaced_regret=(
            float(oracle_gain_mean - equi_gain) if oracle_gain_mean is not None and equi_gain is not None else None
        ),
    )
