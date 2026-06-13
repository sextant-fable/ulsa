from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ulsa.physics_conditioned_corrector import ResidualBlock


def _as_nchw(name: str, value: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim == 3:
        value = value[:, None, :, :]
    if value.ndim != 4 or int(value.shape[1]) != 1:
        raise ValueError(f"{name} must have shape [B,1,H,W] or [B,H,W], got {tuple(value.shape)}")
    return value


def _as_bkchw(name: str, value: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim == 4:
        value = value[:, :, None, :, :]
    if value.ndim != 5 or int(value.shape[2]) != 1:
        raise ValueError(f"{name} must have shape [B,K,1,H,W] or [B,K,H,W], got {tuple(value.shape)}")
    return value


@dataclass(frozen=True)
class PixelTemporalPriorOutput:
    mu_prior: torch.Tensor
    u_prior: torch.Tensor
    delta: torch.Tensor
    logvar: torch.Tensor
    delta_raw: torch.Tensor


def temporal_difference_maps(history_x: torch.Tensor) -> torch.Tensor:
    history_x = _as_bkchw("history_x", history_x)
    diffs = torch.zeros_like(history_x)
    if int(history_x.shape[1]) > 1:
        diffs[:, 1:] = history_x[:, 1:] - history_x[:, :-1]
    return diffs


def assemble_temporal_prior_inputs(
    *,
    history_x: torch.Tensor,
    history_u: torch.Tensor | None = None,
    history_mask: torch.Tensor | None = None,
    history_selected: torch.Tensor | None = None,
    change_maps: torch.Tensor | None = None,
) -> torch.Tensor:
    history_x = _as_bkchw("history_x", history_x)
    b, k, _c, h, w = history_x.shape
    if history_u is None:
        history_u = torch.zeros_like(history_x)
    else:
        history_u = _as_bkchw("history_u", history_u).to(device=history_x.device, dtype=history_x.dtype)
    if history_mask is None:
        history_mask = torch.ones_like(history_x)
    else:
        history_mask = _as_bkchw("history_mask", history_mask).to(device=history_x.device, dtype=history_x.dtype)
    if history_selected is None:
        history_selected = history_mask
    else:
        history_selected = _as_bkchw("history_selected", history_selected).to(
            device=history_x.device,
            dtype=history_x.dtype,
        )
    if change_maps is None:
        change_maps = torch.abs(temporal_difference_maps(history_x))
    else:
        change_maps = _as_bkchw("change_maps", change_maps).to(device=history_x.device, dtype=history_x.dtype)
    expected = tuple(history_x.shape)
    for name, value in (
        ("history_u", history_u),
        ("history_mask", history_mask),
        ("history_selected", history_selected),
        ("change_maps", change_maps),
    ):
        if tuple(value.shape) != expected:
            raise ValueError(f"{name} shape must match history_x, got {tuple(value.shape)} and {expected}")

    x_coord = torch.linspace(-1.0, 1.0, steps=w, device=history_x.device, dtype=history_x.dtype).view(1, 1, 1, w)
    x_coord = x_coord.expand(b, 1, h, w)
    y_coord = torch.linspace(-1.0, 1.0, steps=h, device=history_x.device, dtype=history_x.dtype).view(1, 1, h, 1)
    y_coord = y_coord.expand(b, 1, h, w)
    tensors = [
        history_x.reshape(b, k, h, w),
        history_u.reshape(b, k, h, w),
        history_mask.reshape(b, k, h, w),
        history_selected.reshape(b, k, h, w),
        change_maps.reshape(b, k, h, w),
        x_coord,
        y_coord,
    ]
    return torch.cat(tensors, dim=1)


class ResidualPixelTemporalPrior(nn.Module):
    """Small residual U-Net pixel temporal prior for Stage 2 velocity smoke."""

    def __init__(
        self,
        *,
        history_size: int = 3,
        base_channels: int = 16,
        channel_mult: tuple[int, ...] = (1, 2),
        num_res_blocks: int = 1,
        groupnorm_groups: int = 4,
        delta_clip_scale: float = 0.25,
        prediction_mode: str = "residual",
    ) -> None:
        super().__init__()
        self.history_size = int(history_size)
        if self.history_size <= 0:
            raise ValueError(f"history_size must be positive, got {history_size}")
        self.delta_clip_scale = float(delta_clip_scale)
        self.prediction_mode = str(prediction_mode)
        if self.prediction_mode not in {"residual", "direct"}:
            raise ValueError(f"prediction_mode must be 'residual' or 'direct', got {prediction_mode!r}")
        in_channels = 5 * self.history_size + 2
        widths = [int(base_channels * m) for m in channel_mult]
        if len(widths) <= 0:
            raise ValueError("channel_mult must define at least one level")
        self.stem = nn.Conv2d(in_channels, widths[0], kernel_size=3, padding=1)
        self.enc_blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        cur_ch = widths[0]
        for level, width in enumerate(widths):
            blocks = nn.ModuleList()
            for _ in range(int(num_res_blocks)):
                blocks.append(ResidualBlock(cur_ch, width, groupnorm_groups=groupnorm_groups))
                cur_ch = width
            self.enc_blocks.append(blocks)
            if level < len(widths) - 1:
                self.downs.append(nn.Conv2d(cur_ch, cur_ch, kernel_size=3, stride=2, padding=1))

        self.mid1 = ResidualBlock(cur_ch, cur_ch, groupnorm_groups=groupnorm_groups)
        self.mid2 = ResidualBlock(cur_ch, cur_ch, groupnorm_groups=groupnorm_groups)

        self.up_convs = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for level in reversed(range(len(widths) - 1)):
            up_out = widths[level]
            self.up_convs.append(nn.ConvTranspose2d(cur_ch, up_out, kernel_size=2, stride=2))
            cur_ch = up_out + widths[level]
            blocks = nn.ModuleList()
            for _ in range(int(num_res_blocks)):
                blocks.append(ResidualBlock(cur_ch, widths[level], groupnorm_groups=groupnorm_groups))
                cur_ch = widths[level]
            self.dec_blocks.append(blocks)

        self.out_norm = nn.GroupNorm(groupnorm_groups, cur_ch)
        self.out_act = nn.SiLU()
        self.delta_head = nn.Conv2d(cur_ch, 1, kernel_size=3, padding=1)
        self.logvar_head = nn.Conv2d(cur_ch, 1, kernel_size=3, padding=1)

    def forward(
        self,
        *,
        history_x: torch.Tensor,
        history_u: torch.Tensor | None = None,
        history_mask: torch.Tensor | None = None,
        history_selected: torch.Tensor | None = None,
        change_maps: torch.Tensor | None = None,
    ) -> PixelTemporalPriorOutput:
        history_x = _as_bkchw("history_x", history_x)
        if int(history_x.shape[1]) != self.history_size:
            raise ValueError(
                f"history_x K must match history_size={self.history_size}, got {int(history_x.shape[1])}"
            )
        h = assemble_temporal_prior_inputs(
            history_x=history_x,
            history_u=history_u,
            history_mask=history_mask,
            history_selected=history_selected,
            change_maps=change_maps,
        )
        h = self.stem(h)
        skips = []
        for level, blocks in enumerate(self.enc_blocks):
            for block in blocks:
                h = block(h)
            skips.append(h)
            if level < len(self.downs):
                h = self.downs[level](h)
        h = self.mid1(h)
        h = self.mid2(h)
        for idx, up in enumerate(self.up_convs):
            h = up(h)
            skip = skips[-(idx + 2)]
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = torch.cat([h, skip], dim=1)
            for block in self.dec_blocks[idx]:
                h = block(h)
        h = self.out_act(self.out_norm(h))
        delta_raw = self.delta_head(h)
        logvar = self.logvar_head(h).clamp(-8.0, 4.0)
        last_x = history_x[:, -1]
        if self.prediction_mode == "direct":
            mu_prior = (1.25 * torch.tanh(delta_raw)).clamp(-1.25, 1.25)
            delta = (mu_prior - last_x).clamp(-2.5, 2.5)
        else:
            delta = self.delta_clip_scale * torch.tanh(delta_raw)
            mu_prior = (last_x + delta).clamp(-1.25, 1.25)
        u_prior = F.softplus(logvar)
        return PixelTemporalPriorOutput(
            mu_prior=mu_prior,
            u_prior=u_prior,
            delta=delta,
            logvar=logvar,
            delta_raw=delta_raw,
        )


def psnr_from_mse(mse: torch.Tensor, *, data_range: float = 2.0) -> torch.Tensor:
    return 20.0 * torch.log10(torch.as_tensor(float(data_range), device=mse.device, dtype=mse.dtype)) - 10.0 * torch.log10(
        mse.clamp_min(1.0e-12)
    )


def ssim_torch(x: torch.Tensor, y: torch.Tensor, *, data_range: float = 2.0) -> torch.Tensor:
    x = _as_nchw("x", x)
    y = _as_nchw("y", y).to(device=x.device, dtype=x.dtype)
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


def rankdata_1d(values: torch.Tensor) -> torch.Tensor:
    values = values.flatten()
    sorted_values, order = torch.sort(values)
    ranks_sorted = torch.empty(values.numel(), device=values.device, dtype=torch.float32)
    start = 0
    while start < int(values.numel()):
        end = start + 1
        while end < int(values.numel()) and bool(sorted_values[end] == sorted_values[start]):
            end += 1
        ranks_sorted[start:end] = 0.5 * float(start + end - 1)
        start = end
    ranks = torch.empty_like(ranks_sorted)
    ranks[order] = ranks_sorted
    return ranks


def spearman_1d(x: torch.Tensor, y: torch.Tensor) -> float | None:
    x = x.flatten()
    y = y.flatten()
    valid = torch.isfinite(x) & torch.isfinite(y)
    x = x[valid]
    y = y[valid]
    if int(x.numel()) < 2:
        return None
    if bool(torch.all(x == x[0])) or bool(torch.all(y == y[0])):
        return None
    rx = rankdata_1d(x)
    ry = rankdata_1d(y)
    rx = rx - torch.mean(rx)
    ry = ry - torch.mean(ry)
    denom = torch.sqrt(torch.sum(rx * rx) * torch.sum(ry * ry)).clamp_min(1.0e-12)
    if float(denom.detach().cpu().item()) <= 0.0:
        return None
    return float((torch.sum(rx * ry) / denom).detach().cpu().item())


def prior_prediction_metrics(
    *,
    prediction: torch.Tensor,
    target: torch.Tensor,
    uncertainty: torch.Tensor | None = None,
    data_range: float = 2.0,
) -> dict[str, float | None]:
    prediction = _as_nchw("prediction", prediction)
    target = _as_nchw("target", target).to(device=prediction.device, dtype=prediction.dtype)
    err = prediction - target
    mse = torch.mean(err * err)
    metrics: dict[str, float | None] = {
        "psnr": float(psnr_from_mse(mse, data_range=data_range).detach().cpu().item()),
        "ssim": float(ssim_torch(prediction, target, data_range=data_range).detach().cpu().item()),
        "mae": float(torch.mean(torch.abs(err)).detach().cpu().item()),
        "mse": float(mse.detach().cpu().item()),
    }
    if uncertainty is not None:
        uncertainty = _as_nchw("uncertainty", uncertainty).to(device=prediction.device, dtype=prediction.dtype)
        metrics["uncertainty_error_spearman"] = spearman_1d(uncertainty, torch.abs(err))
    else:
        metrics["uncertainty_error_spearman"] = None
    return metrics


def mean_or_none(values: list[float | None]) -> float | None:
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not clean:
        return None
    return float(sum(clean) / len(clean))
