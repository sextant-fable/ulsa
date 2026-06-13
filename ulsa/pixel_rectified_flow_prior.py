from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ulsa.physics_conditioned_corrector import ResidualBlock
from ulsa.pixel_temporal_prior import _as_bkchw, _as_nchw, assemble_temporal_prior_inputs


@dataclass(frozen=True)
class PixelRectifiedFlowTemporalOutput:
    mu_prior: torch.Tensor
    u_prior: torch.Tensor
    velocity: torch.Tensor
    logvar: torch.Tensor
    x_start: torch.Tensor


def _as_time_map(t: torch.Tensor | float, *, like: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(t):
        t = torch.full((int(like.shape[0]), 1, 1, 1), float(t), device=like.device, dtype=like.dtype)
    else:
        t = t.to(device=like.device, dtype=like.dtype)
        if t.ndim == 0:
            t = t.view(1, 1, 1, 1)
        elif t.ndim == 1:
            t = t.view(-1, 1, 1, 1)
        elif t.ndim == 2:
            t = t.view(int(t.shape[0]), 1, 1, 1)
        elif t.ndim == 4:
            pass
        else:
            raise ValueError(f"t must be scalar, [B], [B,1], or [B,1,H,W], got {tuple(t.shape)}")
    if int(t.shape[0]) == 1 and int(like.shape[0]) > 1:
        t = t.expand(int(like.shape[0]), -1, -1, -1)
    if int(t.shape[0]) != int(like.shape[0]):
        raise ValueError(f"t batch size must match x_t, got {int(t.shape[0])} and {int(like.shape[0])}")
    return t.expand(-1, 1, int(like.shape[-2]), int(like.shape[-1]))


def assemble_rectified_flow_temporal_inputs(
    *,
    x_t: torch.Tensor,
    t: torch.Tensor | float,
    x_start: torch.Tensor,
    history_x: torch.Tensor,
    history_u: torch.Tensor | None = None,
    history_mask: torch.Tensor | None = None,
    history_selected: torch.Tensor | None = None,
) -> torch.Tensor:
    x_t = _as_nchw("x_t", x_t)
    x_start = _as_nchw("x_start", x_start).to(device=x_t.device, dtype=x_t.dtype)
    history_x = _as_bkchw("history_x", history_x).to(device=x_t.device, dtype=x_t.dtype)
    if tuple(x_start.shape) != tuple(x_t.shape):
        raise ValueError(f"x_start must match x_t shape, got {tuple(x_start.shape)} and {tuple(x_t.shape)}")
    temporal = assemble_temporal_prior_inputs(
        history_x=history_x,
        history_u=history_u,
        history_mask=history_mask,
        history_selected=history_selected,
    )
    last_x = history_x[:, -1]
    t_map = _as_time_map(t, like=x_t)
    start_delta = x_start - last_x
    current_delta = x_t - last_x
    return torch.cat([temporal, x_t, x_start, start_delta, current_delta, t_map], dim=1)


class StandalonePixelRectifiedFlowPrior(nn.Module):
    """History-conditioned pixel rectified-flow prior.

    This is the actual pixel-flow temporal prior path: it does not depend on a
    frozen R2/base UNet prior. The model learns a flow-matching velocity field
    that transports an anchor/noisy previous frame to the next frame.
    """

    def __init__(
        self,
        *,
        history_size: int = 3,
        base_channels: int = 32,
        channel_mult: tuple[int, ...] = (1, 2),
        num_res_blocks: int = 1,
        groupnorm_groups: int = 4,
        velocity_clip_scale: float = 1.25,
    ) -> None:
        super().__init__()
        self.history_size = int(history_size)
        if self.history_size <= 0:
            raise ValueError(f"history_size must be positive, got {history_size}")
        self.velocity_clip_scale = float(velocity_clip_scale)
        in_channels = 5 * self.history_size + 2 + 5
        widths = [int(base_channels * mult) for mult in channel_mult]
        if not widths:
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
        self.velocity_head = nn.Conv2d(cur_ch, 1, kernel_size=3, padding=1)
        self.logvar_head = nn.Conv2d(cur_ch, 1, kernel_size=3, padding=1)

    def predict_velocity(
        self,
        *,
        x_t: torch.Tensor,
        t: torch.Tensor | float,
        x_start: torch.Tensor,
        history_x: torch.Tensor,
        history_u: torch.Tensor | None = None,
        history_mask: torch.Tensor | None = None,
        history_selected: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        history_x = _as_bkchw("history_x", history_x)
        if int(history_x.shape[1]) != self.history_size:
            raise ValueError(
                f"history_x K must match history_size={self.history_size}, got {int(history_x.shape[1])}"
            )
        h = assemble_rectified_flow_temporal_inputs(
            x_t=x_t,
            t=t,
            x_start=x_start,
            history_x=history_x,
            history_u=history_u,
            history_mask=history_mask,
            history_selected=history_selected,
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
        velocity = self.velocity_clip_scale * torch.tanh(self.velocity_head(h))
        logvar = self.logvar_head(h).clamp(-8.0, 4.0)
        return velocity, logvar

    def forward(
        self,
        *,
        x_t: torch.Tensor,
        t: torch.Tensor | float,
        x_start: torch.Tensor,
        history_x: torch.Tensor,
        history_u: torch.Tensor | None = None,
        history_mask: torch.Tensor | None = None,
        history_selected: torch.Tensor | None = None,
    ) -> PixelRectifiedFlowTemporalOutput:
        x_t = _as_nchw("x_t", x_t)
        x_start = _as_nchw("x_start", x_start).to(device=x_t.device, dtype=x_t.dtype)
        velocity, logvar = self.predict_velocity(
            x_t=x_t,
            t=t,
            x_start=x_start,
            history_x=history_x,
            history_u=history_u,
            history_mask=history_mask,
            history_selected=history_selected,
        )
        mu_prior = (x_t + (1.0 - _as_time_map(t, like=x_t)) * velocity).clamp(-1.25, 1.25)
        u_prior = F.softplus(logvar) + 0.05 * torch.abs(velocity.detach())
        return PixelRectifiedFlowTemporalOutput(
            mu_prior=mu_prior,
            u_prior=u_prior,
            velocity=velocity,
            logvar=logvar,
            x_start=x_start,
        )

    def integrate(
        self,
        *,
        x_start: torch.Tensor,
        steps: int,
        history_x: torch.Tensor,
        history_u: torch.Tensor | None = None,
        history_mask: torch.Tensor | None = None,
        history_selected: torch.Tensor | None = None,
    ) -> PixelRectifiedFlowTemporalOutput:
        steps = max(1, int(steps))
        x = _as_nchw("x_start", x_start)
        start = x
        logvar = None
        velocity = None
        dt = 1.0 / float(steps)
        for step_idx in range(steps):
            t = float(step_idx) / float(steps)
            velocity, logvar = self.predict_velocity(
                x_t=x,
                t=t,
                x_start=start,
                history_x=history_x,
                history_u=history_u,
                history_mask=history_mask,
                history_selected=history_selected,
            )
            x = (x + dt * velocity).clamp(-1.25, 1.25)
        if logvar is None or velocity is None:
            raise RuntimeError("Pixel rectified-flow integration produced no step")
        u_prior = F.softplus(logvar) + 0.05 * torch.abs(x - start)
        return PixelRectifiedFlowTemporalOutput(
            mu_prior=x,
            u_prior=u_prior,
            velocity=velocity,
            logvar=logvar,
            x_start=start,
        )
