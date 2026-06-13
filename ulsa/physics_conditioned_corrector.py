from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, groupnorm_groups: int = 8):
        super().__init__()
        self.norm1 = nn.GroupNorm(groupnorm_groups, in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.norm2 = nn.GroupNorm(groupnorm_groups, out_channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act1(self.norm1(x)))
        h = self.conv2(self.act2(self.norm2(h)))
        return h + self.skip(x)


class PhysicsConditionedCorrector(nn.Module):
    """
    Latent UNet corrector.
    Input: concat([z_pred, mask_low, res_low, abs_res_low, hist_low], dim=1)
    Output heads: delta_z_raw, logvar_low
    """

    def __init__(
        self,
        latent_channels: int,
        base_channels: int = 64,
        channel_mult: tuple[int, int, int] = (1, 2, 4),
        num_res_blocks: int = 2,
        groupnorm_groups: int = 8,
        delta_clip_scale: float = 0.1,
    ):
        super().__init__()
        self.latent_channels = int(latent_channels)
        self.delta_clip_scale = float(delta_clip_scale)

        in_channels = self.latent_channels + 4
        widths = [int(base_channels * m) for m in channel_mult]

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
            self.up_convs.append(
                nn.ConvTranspose2d(cur_ch, up_out, kernel_size=2, stride=2)
            )
            cur_ch = up_out + widths[level]  # concat skip
            blocks = nn.ModuleList()
            for _ in range(int(num_res_blocks)):
                blocks.append(ResidualBlock(cur_ch, widths[level], groupnorm_groups=groupnorm_groups))
                cur_ch = widths[level]
            self.dec_blocks.append(blocks)

        self.out_norm = nn.GroupNorm(groupnorm_groups, cur_ch)
        self.out_act = nn.SiLU()
        self.delta_head = nn.Conv2d(cur_ch, self.latent_channels, kernel_size=3, padding=1)
        self.logvar_head = nn.Conv2d(cur_ch, 1, kernel_size=3, padding=1)

    def forward(
        self,
        z_pred: torch.Tensor,
        mask_low: torch.Tensor,
        res_low: torch.Tensor,
        abs_res_low: torch.Tensor,
        hist_low: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.cat([z_pred, mask_low, res_low, abs_res_low, hist_low], dim=1)
        h = self.stem(x)

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
        delta_z_raw = self.delta_head(h)
        logvar_low = self.logvar_head(h)

        delta_z = self.delta_clip_scale * torch.tanh(delta_z_raw)
        z_corr = z_pred + delta_z
        z_logvar = logvar_low

        return z_corr, z_logvar, delta_z_raw
