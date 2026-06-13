from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ulsa.pixel_belief_state import ActionSet


def _as_nchw(name: str, value: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim == 3:
        value = value[:, None, :, :]
    if value.ndim != 4 or int(value.shape[1]) != 1:
        raise ValueError(f"{name} must have shape [B,1,H,W] or [B,H,W], got {tuple(value.shape)}")
    return value


@dataclass(frozen=True)
class Measurement:
    obs: torch.Tensor
    mask: torch.Tensor
    action: ActionSet

    @property
    def obs_buffer(self) -> torch.Tensor:
        return self.obs

    @property
    def obs_mask(self) -> torch.Tensor:
        return self.mask


class LineMaskGenerator:
    def __init__(
        self,
        *,
        image_shape: tuple[int, int],
        n_lines: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.image_shape = (int(image_shape[0]), int(image_shape[1]))
        self.n_lines = int(n_lines)
        if self.image_shape[0] <= 0 or self.image_shape[1] <= 0:
            raise ValueError(f"image_shape must be positive, got {self.image_shape}")
        if self.n_lines <= 0:
            raise ValueError(f"n_lines must be positive, got {self.n_lines}")
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.dtype = dtype
        self.line_masks = self._build_line_masks()
        self.line_masks_flat = self.line_masks.reshape(self.n_lines, -1).contiguous()
        self.line_denoms = torch.sum(self.line_masks_flat, dim=1).clamp_min(1.0)

    def _build_line_masks(self) -> torch.Tensor:
        h, w = self.image_shape
        cols = torch.arange(w, device=self.device)
        line_for_col = torch.div(cols * self.n_lines, w, rounding_mode="floor").clamp_max(self.n_lines - 1)
        line_cols = F.one_hot(line_for_col, num_classes=self.n_lines).to(dtype=self.dtype)
        line_cols = line_cols.transpose(0, 1).contiguous()
        return line_cols[:, None, None, :].expand(self.n_lines, 1, h, w).contiguous()

    def to(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "LineMaskGenerator":
        return LineMaskGenerator(
            image_shape=self.image_shape,
            n_lines=self.n_lines,
            device=device if device is not None else self.device,
            dtype=dtype if dtype is not None else self.dtype,
        )

    def selected_lines_from_indices(
        self,
        line_indices: torch.Tensor | list[int] | tuple[int, ...],
        *,
        batch_size: int | None = None,
    ) -> torch.Tensor:
        indices = torch.as_tensor(line_indices, dtype=torch.long, device=self.device)
        if indices.ndim == 0:
            indices = indices.view(1)
        if indices.ndim == 1:
            batch_size = 1 if batch_size is None else int(batch_size)
            indices = indices.view(1, -1).expand(batch_size, -1)
        elif indices.ndim == 2:
            if batch_size is not None and int(indices.shape[0]) != int(batch_size):
                raise ValueError(
                    f"line_indices batch size must be {int(batch_size)}, got {int(indices.shape[0])}"
                )
        else:
            raise ValueError(f"line_indices must have shape [K] or [B,K], got {tuple(indices.shape)}")
        if indices.numel() > 0:
            if torch.any(indices < 0) or torch.any(indices >= self.n_lines):
                raise ValueError(f"line_indices must be in [0,{self.n_lines}), got {indices}")
        selected = torch.zeros((int(indices.shape[0]), self.n_lines), dtype=torch.bool, device=self.device)
        if indices.numel() > 0:
            selected.scatter_(1, indices, True)
        return selected

    def pixel_mask_from_lines(self, selected_lines: torch.Tensor) -> torch.Tensor:
        if selected_lines.ndim == 1:
            selected_lines = selected_lines.view(1, -1)
        if selected_lines.ndim != 2 or int(selected_lines.shape[1]) != self.n_lines:
            raise ValueError(
                f"selected_lines must have shape [B,{self.n_lines}], got {tuple(selected_lines.shape)}"
            )
        line_weights = selected_lines.to(device=self.device, dtype=self.dtype)
        flat = torch.matmul(line_weights, self.line_masks_flat)
        b = int(selected_lines.shape[0])
        h, w = self.image_shape
        return (flat.reshape(b, 1, h, w) > 0).to(dtype=self.dtype)

    def action_from_indices(
        self,
        line_indices: torch.Tensor | list[int] | tuple[int, ...],
        *,
        batch_size: int | None = None,
    ) -> ActionSet:
        selected = self.selected_lines_from_indices(line_indices, batch_size=batch_size)
        return ActionSet(
            selected_lines=selected,
            pixel_mask=self.pixel_mask_from_lines(selected),
            line_indices=torch.as_tensor(line_indices, dtype=torch.long, device=self.device),
        )

    def action_from_scores(
        self,
        scores: torch.Tensor,
        *,
        k: int,
        available: torch.Tensor | None = None,
    ) -> ActionSet:
        if scores.ndim == 1:
            scores = scores.view(1, -1)
        if scores.ndim != 2 or int(scores.shape[1]) != self.n_lines:
            raise ValueError(f"scores must have shape [B,{self.n_lines}], got {tuple(scores.shape)}")
        k = int(k)
        if k < 0:
            raise ValueError(f"k must be non-negative, got {k}")
        if k > self.n_lines:
            raise ValueError(f"k must be <= n_lines, got {k}>{self.n_lines}")
        scores = scores.to(device=self.device)
        if available is None:
            available = torch.ones_like(scores, dtype=torch.bool, device=self.device)
        else:
            available = available.to(device=self.device, dtype=torch.bool)
            if tuple(available.shape) != tuple(scores.shape):
                raise ValueError(
                    f"available must match scores shape, got {tuple(available.shape)} and {tuple(scores.shape)}"
                )
        selected = torch.zeros_like(available, dtype=torch.bool)
        if k > 0:
            masked_scores = torch.where(
                available,
                scores,
                torch.full_like(scores, -torch.inf),
            )
            _values, indices = torch.topk(masked_scores, k=k, dim=1)
            selected.scatter_(1, indices, True)
            selected = selected & available
        return ActionSet(
            selected_lines=selected,
            pixel_mask=self.pixel_mask_from_lines(selected),
            line_indices=None,
        )

    def linewise_mean(self, value: torch.Tensor) -> torch.Tensor:
        value = _as_nchw("value", value).to(device=self.device, dtype=self.dtype)
        flat = value.reshape(int(value.shape[0]), -1)
        sums = torch.matmul(flat, self.line_masks_flat.transpose(0, 1))
        return sums / self.line_denoms.view(1, -1)


class MeasurementOperator:
    def measure(
        self,
        target: torch.Tensor,
        action: ActionSet,
        *,
        prev_mask: torch.Tensor | None = None,
        prev_obs: torch.Tensor | None = None,
    ) -> Measurement:
        target = _as_nchw("target", target)
        action_mask = action.pixel_mask.to(device=target.device, dtype=target.dtype)
        if tuple(action_mask.shape) != tuple(target.shape):
            raise ValueError(
                f"action pixel mask must match target shape, got {tuple(action_mask.shape)} and {tuple(target.shape)}"
            )
        if prev_mask is None:
            prev_mask = torch.zeros_like(target)
        else:
            prev_mask = _as_nchw("prev_mask", prev_mask).to(device=target.device, dtype=target.dtype)
        if prev_obs is None:
            prev_obs = torch.zeros_like(target)
        else:
            prev_obs = _as_nchw("prev_obs", prev_obs).to(device=target.device, dtype=target.dtype)
        new_mask = torch.maximum(prev_mask, action_mask)
        new_obs = torch.where(action_mask > 0, target, prev_obs)
        new_obs = torch.where(new_mask > 0, new_obs, torch.zeros_like(new_obs))
        return Measurement(obs=new_obs, mask=new_mask, action=action)


class HardProjection:
    @staticmethod
    def apply(*, x_raw: torch.Tensor, obs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x_raw = _as_nchw("x_raw", x_raw)
        obs = _as_nchw("obs", obs).to(device=x_raw.device, dtype=x_raw.dtype)
        mask = _as_nchw("mask", mask).to(device=x_raw.device, dtype=x_raw.dtype)
        if tuple(obs.shape) != tuple(x_raw.shape) or tuple(mask.shape) != tuple(x_raw.shape):
            raise ValueError(
                f"x_raw, obs, and mask must share shape, got {tuple(x_raw.shape)}, "
                f"{tuple(obs.shape)}, {tuple(mask.shape)}"
            )
        return torch.where(mask > 0, obs, x_raw)
