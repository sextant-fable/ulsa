from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ulsa.pixel_acquisition_stage3 import edge_magnitude, normalize_line_scores, rbf_greedy_topk, selected_indices_from_mask
from ulsa.pixel_measurement import LineMaskGenerator


@dataclass(frozen=True)
class PixelLineFeatureSpec:
    names: tuple[str, ...] = (
        "u_prior",
        "abs_prior_delta",
        "prior_edge",
        "prev_u",
        "prev_mask",
        "line_pos",
    )


def _as_nchw(name: str, value: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim == 3:
        value = value[:, None, :, :]
    if value.ndim != 4 or int(value.shape[1]) != 1:
        raise ValueError(f"{name} must have shape [B,1,H,W] or [B,H,W], got {tuple(value.shape)}")
    return value


def build_stage2j_line_features(
    *,
    generator: LineMaskGenerator,
    mu_prior: torch.Tensor,
    u_prior: torch.Tensor,
    prev_x: torch.Tensor,
    prev_u: torch.Tensor | None = None,
    prev_mask: torch.Tensor | None = None,
    normalize: bool = True,
) -> tuple[torch.Tensor, PixelLineFeatureSpec]:
    mu_prior = _as_nchw("mu_prior", mu_prior)
    u_prior = _as_nchw("u_prior", u_prior).to(device=mu_prior.device, dtype=mu_prior.dtype)
    prev_x = _as_nchw("prev_x", prev_x).to(device=mu_prior.device, dtype=mu_prior.dtype)
    if prev_u is None:
        prev_u = torch.zeros_like(mu_prior)
    else:
        prev_u = _as_nchw("prev_u", prev_u).to(device=mu_prior.device, dtype=mu_prior.dtype)
    if prev_mask is None:
        prev_mask = torch.zeros_like(mu_prior)
    else:
        prev_mask = _as_nchw("prev_mask", prev_mask).to(device=mu_prior.device, dtype=mu_prior.dtype)
    values = [
        generator.linewise_mean(torch.abs(u_prior)),
        generator.linewise_mean(torch.abs(mu_prior - prev_x)),
        generator.linewise_mean(edge_magnitude(mu_prior)),
        generator.linewise_mean(torch.abs(prev_u)),
        generator.linewise_mean(prev_mask),
    ]
    if normalize:
        values = [normalize_line_scores(v) for v in values]
    b, n_lines = int(values[0].shape[0]), int(values[0].shape[1])
    line_pos = torch.linspace(-1.0, 1.0, steps=n_lines, device=mu_prior.device, dtype=mu_prior.dtype)
    line_pos = line_pos.view(1, n_lines).expand(b, n_lines)
    values.append(line_pos)
    features = torch.stack(values, dim=-1)
    return features, PixelLineFeatureSpec()


class PixelLineMLP(nn.Module):
    def __init__(self, *, in_features: int = 6, hidden_features: int = 32, num_layers: int = 2) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        cur = int(in_features)
        for _ in range(max(1, int(num_layers))):
            layers.append(nn.Linear(cur, int(hidden_features)))
            layers.append(nn.GELU())
            cur = int(hidden_features)
        layers.append(nn.Linear(cur, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError(f"features must have shape [B,L,F], got {tuple(features.shape)}")
        return self.net(features).squeeze(-1)


def action_from_line_scores(
    *,
    generator: LineMaskGenerator,
    scores: torch.Tensor,
    budget: int,
    use_rbf: bool = True,
    rbf_sigma: float = 3.0,
    rbf_suppression: float = 0.45,
):
    if bool(use_rbf):
        selected = rbf_greedy_topk(
            scores.to(device=generator.device),
            k=int(budget),
            sigma=float(rbf_sigma),
            suppression=float(rbf_suppression),
        )
        indices = selected_indices_from_mask(selected)
        return generator.action_from_indices(indices, batch_size=int(selected.shape[0]))
    return generator.action_from_scores(scores.to(device=generator.device), k=int(budget))
