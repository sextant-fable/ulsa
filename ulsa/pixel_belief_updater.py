from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ulsa.physics_conditioned_corrector import ResidualBlock


def apply_pixel_belief_update(
    *,
    x_dc: torch.Tensor,
    obs_mask: torch.Tensor,
    obs_buffer: torch.Tensor,
    delta_x: torch.Tensor,
) -> torch.Tensor:
    """Apply Stage 3A update and restore observed pixels exactly."""
    x_pre = x_dc + (1.0 - obs_mask) * delta_x
    return obs_mask * obs_buffer + (1.0 - obs_mask) * x_pre


def build_pixel_belief_condition_maps(
    *,
    obs_mask: torch.Tensor,
    total_budget: int | torch.Tensor,
    observed_count: int | torch.Tensor | None = None,
    group_index: int | torch.Tensor | None = None,
    n_possible_actions: int = 112,
) -> dict[str, torch.Tensor]:
    """Build mask/budget conditioning maps for PixelBeliefUpdater inputs."""
    if obs_mask.ndim != 4:
        raise ValueError(f"obs_mask must be [B,1,H,W], got shape={tuple(obs_mask.shape)}")
    b, _c, h, w = obs_mask.shape
    device = obs_mask.device
    dtype = obs_mask.dtype
    denom = float(max(1, int(n_possible_actions)))

    mask_density = torch.mean(obs_mask, dim=(1, 2, 3), keepdim=True).expand(b, 1, h, w)

    def _as_batch_map(value, *, default: float) -> torch.Tensor:
        if value is None:
            base = torch.full((b, 1, 1, 1), float(default), device=device, dtype=dtype)
        elif torch.is_tensor(value):
            base = value.to(device=device, dtype=dtype)
            if base.ndim == 0:
                base = base.view(1, 1, 1, 1).expand(b, 1, 1, 1)
            elif base.ndim == 1:
                if int(base.numel()) != b:
                    raise ValueError(f"Expected {b} conditioning values, got {int(base.numel())}")
                base = base.view(b, 1, 1, 1)
            else:
                base = base.reshape(b, 1, 1, 1)
        else:
            base = torch.full((b, 1, 1, 1), float(value), device=device, dtype=dtype)
        return base.expand(b, 1, h, w)

    budget_map = _as_batch_map(total_budget, default=0.0) / denom
    if observed_count is None:
        observed_count_map = mask_density
    else:
        observed_count_map = _as_batch_map(observed_count, default=0.0) / denom
    group_index_map = _as_batch_map(group_index, default=0.0) / 8.0

    x_coord = torch.linspace(-1.0, 1.0, steps=w, device=device, dtype=dtype).view(1, 1, 1, w)
    line_coord = x_coord.expand(b, 1, h, w)

    observed_cols = (torch.amax(obs_mask, dim=2, keepdim=True) > 0).to(dtype=dtype)
    col_idx = torch.arange(w, device=device, dtype=dtype).view(1, 1, 1, w)
    distance_maps = []
    for batch_idx in range(b):
        cols = torch.nonzero(observed_cols[batch_idx, 0, 0] > 0, as_tuple=False).flatten()
        if int(cols.numel()) == 0:
            dist = torch.ones((1, 1, h, w), device=device, dtype=dtype)
        else:
            obs_idx = cols.to(device=device, dtype=dtype).view(-1, 1)
            min_dist = torch.min(torch.abs(obs_idx - col_idx[0, 0, 0].view(1, -1)), dim=0).values
            min_dist = min_dist / float(max(1, w - 1))
            dist = min_dist.view(1, 1, 1, w).expand(1, 1, h, w)
        distance_maps.append(dist)
    distance_to_observed_line = torch.cat(distance_maps, dim=0)

    return {
        "mask_density": mask_density,
        "budget_token": budget_map,
        "observed_count_token": observed_count_map,
        "group_index_token": group_index_map,
        "distance_to_observed_line": distance_to_observed_line,
        "line_coord": line_coord,
    }


def assemble_pixel_belief_inputs(
    *,
    x_prior: torch.Tensor,
    x_dc: torch.Tensor,
    obs_buffer: torch.Tensor,
    obs_mask: torch.Tensor,
    residual_obs: torch.Tensor,
    history_map: torch.Tensor,
    mask_density: torch.Tensor,
    budget_token: torch.Tensor,
    observed_count_token: torch.Tensor,
    group_index_token: torch.Tensor,
    distance_to_observed_line: torch.Tensor,
    line_coord: torch.Tensor,
) -> torch.Tensor:
    """Assemble Stage 3A main inputs without LBF teacher diagnostics."""
    abs_residual_obs = torch.abs(residual_obs)
    tensors = [
        x_prior,
        x_dc,
        obs_buffer,
        obs_mask,
        residual_obs,
        abs_residual_obs,
        history_map,
        mask_density,
        budget_token,
        observed_count_token,
        group_index_token,
        distance_to_observed_line,
        line_coord,
    ]
    shapes = {tuple(t.shape[-2:]) for t in tensors}
    if len(shapes) != 1:
        raise ValueError(f"All PixelBeliefUpdater inputs must share H/W, got {sorted(shapes)}")
    return torch.cat(tensors, dim=1)


class PixelBeliefUpdater(nn.Module):
    """Stage 3A replay-mask pixel observation updater.

    This module is separate from PixelCorrector. It consumes the prior,
    observed-buffer context, mask/history/budget conditioning, and predicts an
    update for unobserved pixels plus a belief log-variance map.
    """

    def __init__(
        self,
        *,
        base_channels: int = 32,
        channel_mult: tuple[int, ...] = (1, 2),
        num_res_blocks: int = 1,
        groupnorm_groups: int = 8,
        delta_clip_scale: float = 0.1,
    ):
        super().__init__()
        self.delta_clip_scale = float(delta_clip_scale)
        in_channels = 13
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
        x_prior: torch.Tensor,
        x_dc: torch.Tensor,
        obs_buffer: torch.Tensor,
        obs_mask: torch.Tensor,
        residual_obs: torch.Tensor,
        history_map: torch.Tensor,
        mask_density: torch.Tensor,
        budget_token: torch.Tensor,
        observed_count_token: torch.Tensor,
        group_index_token: torch.Tensor,
        distance_to_observed_line: torch.Tensor,
        line_coord: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = assemble_pixel_belief_inputs(
            x_prior=x_prior,
            x_dc=x_dc,
            obs_buffer=obs_buffer,
            obs_mask=obs_mask,
            residual_obs=residual_obs,
            history_map=history_map,
            mask_density=mask_density,
            budget_token=budget_token,
            observed_count_token=observed_count_token,
            group_index_token=group_index_token,
            distance_to_observed_line=distance_to_observed_line,
            line_coord=line_coord,
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
        logvar_belief = self.logvar_head(h)
        delta_x = self.delta_clip_scale * torch.tanh(delta_raw)
        return delta_x, logvar_belief, delta_raw
