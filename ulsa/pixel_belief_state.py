from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import torch


def _expect_nchw(name: str, value: torch.Tensor) -> None:
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != 4 or int(value.shape[1]) != 1:
        raise ValueError(f"{name} must have shape [B,1,H,W], got {tuple(value.shape)}")


def _expect_line_map(name: str, value: torch.Tensor, *, batch_size: int) -> None:
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != 2:
        raise ValueError(f"{name} must have shape [B,L], got {tuple(value.shape)}")
    if int(value.shape[0]) != int(batch_size):
        raise ValueError(
            f"{name} batch size must match image tensors, got {int(value.shape[0])} "
            f"and {int(batch_size)}"
        )


@dataclass(frozen=True)
class BudgetState:
    total: int
    used: int = 0
    phase: int = 0
    max_updates: int = 1

    def __post_init__(self) -> None:
        total = int(self.total)
        used = int(self.used)
        phase = int(self.phase)
        max_updates = int(self.max_updates)
        if total < 0:
            raise ValueError(f"budget total must be non-negative, got {total}")
        if used < 0:
            raise ValueError(f"budget used must be non-negative, got {used}")
        if used > total:
            raise ValueError(f"budget used must be <= total, got {used}>{total}")
        if phase < 0:
            raise ValueError(f"budget phase must be non-negative, got {phase}")
        if max_updates <= 0:
            raise ValueError(f"budget max_updates must be positive, got {max_updates}")
        object.__setattr__(self, "total", total)
        object.__setattr__(self, "used", used)
        object.__setattr__(self, "phase", phase)
        object.__setattr__(self, "max_updates", max_updates)

    @property
    def remaining(self) -> int:
        return int(self.total - self.used)

    def after_select(self, n_lines: int, *, advance_phase: bool = True) -> "BudgetState":
        n_lines = int(n_lines)
        if n_lines < 0:
            raise ValueError(f"n_lines must be non-negative, got {n_lines}")
        return BudgetState(
            total=self.total,
            used=min(self.total, self.used + n_lines),
            phase=self.phase + (1 if advance_phase else 0),
            max_updates=self.max_updates,
        )

    def token_maps(self, reference: torch.Tensor, *, n_lines: int) -> torch.Tensor:
        _expect_nchw("reference", reference)
        denom = float(max(1, int(n_lines)))
        values = torch.tensor(
            [
                float(self.total) / denom,
                float(self.used) / denom,
                float(self.remaining) / denom,
                float(self.phase) / float(max(1, self.max_updates)),
            ],
            dtype=reference.dtype,
            device=reference.device,
        )
        b, _c, h, w = reference.shape
        return values.view(1, 4, 1, 1).expand(int(b), 4, int(h), int(w))


@dataclass(frozen=True)
class ActionSet:
    selected_lines: torch.Tensor
    pixel_mask: torch.Tensor
    line_indices: torch.Tensor | None = None

    def __post_init__(self) -> None:
        _expect_line_map("selected_lines", self.selected_lines, batch_size=int(self.pixel_mask.shape[0]))
        _expect_nchw("pixel_mask", self.pixel_mask)
        if int(self.selected_lines.shape[0]) != int(self.pixel_mask.shape[0]):
            raise ValueError(
                "selected_lines and pixel_mask must share batch size, "
                f"got {tuple(self.selected_lines.shape)} and {tuple(self.pixel_mask.shape)}"
            )
        if self.line_indices is not None:
            if not torch.is_tensor(self.line_indices):
                raise TypeError("line_indices must be a torch.Tensor when provided")
            if self.line_indices.ndim not in {1, 2}:
                raise ValueError(
                    f"line_indices must have shape [K] or [B,K], got {tuple(self.line_indices.shape)}"
                )

    @property
    def batch_size(self) -> int:
        return int(self.selected_lines.shape[0])

    @property
    def n_lines(self) -> int:
        return int(self.selected_lines.shape[1])

    @property
    def counts(self) -> torch.Tensor:
        return torch.sum(self.selected_lines.to(dtype=torch.int64), dim=1)


@dataclass(frozen=True)
class PixelBeliefState:
    x_final: torch.Tensor
    u_post: torch.Tensor
    mask: torch.Tensor
    obs: torch.Tensor
    selected_lines: torch.Tensor
    line_history: torch.Tensor
    budget: BudgetState
    x_prior: torch.Tensor | None = None
    u_prior: torch.Tensor | None = None
    memory: Any | None = None
    frame_idx: int = 0

    def __post_init__(self) -> None:
        for name in ["x_final", "u_post", "mask", "obs"]:
            _expect_nchw(name, getattr(self, name))
        base_shape = tuple(self.x_final.shape)
        for name in ["u_post", "mask", "obs"]:
            value = getattr(self, name)
            if tuple(value.shape) != base_shape:
                raise ValueError(f"{name} shape must match x_final, got {tuple(value.shape)} and {base_shape}")
        if self.x_prior is not None:
            _expect_nchw("x_prior", self.x_prior)
            if tuple(self.x_prior.shape) != base_shape:
                raise ValueError("x_prior shape must match x_final")
        if self.u_prior is not None:
            _expect_nchw("u_prior", self.u_prior)
            if tuple(self.u_prior.shape) != base_shape:
                raise ValueError("u_prior shape must match x_final")
        _expect_line_map("selected_lines", self.selected_lines, batch_size=self.batch_size)
        _expect_line_map("line_history", self.line_history, batch_size=self.batch_size)
        if int(self.line_history.shape[1]) != int(self.selected_lines.shape[1]):
            raise ValueError(
                f"line_history must have the same number of lines as selected_lines, "
                f"got {int(self.line_history.shape[1])} and {int(self.selected_lines.shape[1])}"
            )

    @property
    def batch_size(self) -> int:
        return int(self.x_final.shape[0])

    @property
    def spatial_shape(self) -> tuple[int, int]:
        return int(self.x_final.shape[-2]), int(self.x_final.shape[-1])

    @property
    def n_lines(self) -> int:
        return int(self.selected_lines.shape[1])

    @property
    def obs_mask(self) -> torch.Tensor:
        return self.mask

    @property
    def obs_buffer(self) -> torch.Tensor:
        return self.obs

    def replace(self, **updates: Any) -> "PixelBeliefState":
        return replace(self, **updates)


def make_empty_pixel_belief_state(
    *,
    batch_size: int,
    image_shape: tuple[int, int],
    n_lines: int,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
    budget: BudgetState | None = None,
    frame_idx: int = 0,
) -> PixelBeliefState:
    h, w = int(image_shape[0]), int(image_shape[1])
    batch_size = int(batch_size)
    n_lines = int(n_lines)
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if h <= 0 or w <= 0:
        raise ValueError(f"image_shape must be positive, got {image_shape}")
    if n_lines <= 0:
        raise ValueError(f"n_lines must be positive, got {n_lines}")
    zeros = torch.zeros((batch_size, 1, h, w), dtype=dtype, device=device)
    selected = torch.zeros((batch_size, n_lines), dtype=torch.bool, device=device)
    history = torch.zeros((batch_size, n_lines), dtype=dtype, device=device)
    return PixelBeliefState(
        x_final=zeros.clone(),
        u_post=torch.ones_like(zeros),
        mask=zeros.clone(),
        obs=zeros.clone(),
        selected_lines=selected,
        line_history=history,
        budget=budget or BudgetState(total=0),
        x_prior=zeros.clone(),
        u_prior=torch.ones_like(zeros),
        memory=None,
        frame_idx=int(frame_idx),
    )
