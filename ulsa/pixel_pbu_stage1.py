from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from ulsa.pixel_belief_state import ActionSet, BudgetState, PixelBeliefState
from ulsa.pixel_belief_updater import PixelBeliefUpdater, build_pixel_belief_condition_maps
from ulsa.pixel_measurement import HardProjection, LineMaskGenerator, MeasurementOperator


STAGE1_BUDGETS = (0, 2, 4, 7, 10, 14, 21, 28)

MASK_FIXED104 = "fixed104"
MASK_FIXED14 = "fixed14"
MASK_FIXED10 = "fixed10"
MASK_RANDOM = "random"
MASK_ROLLED_EQUISPACED = "rolled_equispaced"
MASK_CLUSTERED_LOCAL = "clustered_local"
MASK_HEURISTIC = "heuristic_generated"
MASK_ADVERSARIAL_GAP = "adversarial_gap"
SUPPORTED_STAGE1_MASK_FAMILIES = (
    MASK_FIXED104,
    MASK_FIXED14,
    MASK_FIXED10,
    MASK_RANDOM,
    MASK_ROLLED_EQUISPACED,
    MASK_CLUSTERED_LOCAL,
    MASK_HEURISTIC,
    MASK_ADVERSARIAL_GAP,
)

PRIOR_COPY_LAST = "copy_last"
PRIOR_EMA = "ema"
PRIOR_DEGRADED_GT = "degraded_gt"
PRIOR_WEAK_INTERPOLATION = "weak_interpolation"
PRIOR_ZERO_FILL = "zero_fill"
PRIOR_CACHED_DPF = "cached_dpf"
SUPPORTED_STAGE1_PRIORS = (
    PRIOR_COPY_LAST,
    PRIOR_EMA,
    PRIOR_DEGRADED_GT,
    PRIOR_WEAK_INTERPOLATION,
    PRIOR_ZERO_FILL,
    PRIOR_CACHED_DPF,
)
TRAINING_ONLY_PRIORS = (PRIOR_DEGRADED_GT,)

MODEL_ABLATION_FULL = "full"
MODEL_ABLATION_NO_RESIDUAL = "no_residual"
MODEL_ABLATION_NO_BUDGET_TOKEN = "no_budget_token"
MODEL_ABLATION_NO_HARD_PROJECTION = "no_hard_projection"
SUPPORTED_STAGE1_MODEL_ABLATIONS = (
    MODEL_ABLATION_FULL,
    MODEL_ABLATION_NO_RESIDUAL,
    MODEL_ABLATION_NO_BUDGET_TOKEN,
    MODEL_ABLATION_NO_HARD_PROJECTION,
)

Stage1MaskFamily = Literal[
    "fixed104",
    "fixed14",
    "fixed10",
    "random",
    "rolled_equispaced",
    "clustered_local",
    "heuristic_generated",
    "adversarial_gap",
]

Stage1PriorKind = Literal[
    "copy_last",
    "ema",
    "degraded_gt",
    "weak_interpolation",
    "zero_fill",
    "cached_dpf",
]

Stage1ModelAblationName = Literal[
    "full",
    "no_residual",
    "no_budget_token",
    "no_hard_projection",
]


@dataclass(frozen=True)
class Stage1PBUAblation:
    name: str = MODEL_ABLATION_FULL
    disable_residual: bool = False
    disable_budget_tokens: bool = False
    disable_hard_projection: bool = False


def resolve_stage1_pbu_ablation(
    ablation: Stage1PBUAblation | Stage1ModelAblationName | str | None,
) -> Stage1PBUAblation:
    if ablation is None:
        return Stage1PBUAblation()
    if isinstance(ablation, Stage1PBUAblation):
        expected_flags = {
            MODEL_ABLATION_FULL: (False, False, False),
            MODEL_ABLATION_NO_RESIDUAL: (True, False, False),
            MODEL_ABLATION_NO_BUDGET_TOKEN: (False, True, False),
            MODEL_ABLATION_NO_HARD_PROJECTION: (False, False, True),
        }.get(ablation.name)
        actual_flags = (
            bool(ablation.disable_residual),
            bool(ablation.disable_budget_tokens),
            bool(ablation.disable_hard_projection),
        )
        if expected_flags is None or actual_flags != expected_flags:
            raise ValueError(
                "Stage1PBUAblation must match one supported named smoke control, "
                f"got name={ablation.name!r} flags={actual_flags}"
            )
        return ablation
    if ablation == MODEL_ABLATION_FULL:
        return Stage1PBUAblation(name=MODEL_ABLATION_FULL)
    if ablation == MODEL_ABLATION_NO_RESIDUAL:
        return Stage1PBUAblation(name=MODEL_ABLATION_NO_RESIDUAL, disable_residual=True)
    if ablation == MODEL_ABLATION_NO_BUDGET_TOKEN:
        return Stage1PBUAblation(name=MODEL_ABLATION_NO_BUDGET_TOKEN, disable_budget_tokens=True)
    if ablation == MODEL_ABLATION_NO_HARD_PROJECTION:
        return Stage1PBUAblation(name=MODEL_ABLATION_NO_HARD_PROJECTION, disable_hard_projection=True)
    raise ValueError(f"Unsupported Stage 1 model ablation {ablation!r}")


def _as_nchw(name: str, value: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim == 3:
        value = value[:, None, :, :]
    if value.ndim != 4 or int(value.shape[1]) != 1:
        raise ValueError(f"{name} must have shape [B,1,H,W] or [B,H,W], got {tuple(value.shape)}")
    return value


def _masked_l1(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = torch.sum(mask).clamp_min(1.0)
    return torch.sum(torch.abs(value) * mask) / denom


def _equispaced_indices(n_lines: int, budget: int, *, roll: int = 0, device=None) -> torch.Tensor:
    budget = int(budget)
    if budget <= 0:
        return torch.empty((0,), dtype=torch.long, device=device)
    if budget > int(n_lines):
        raise ValueError(f"budget must be <= n_lines, got {budget}>{n_lines}")
    idx = torch.div(
        torch.arange(budget, dtype=torch.long, device=device) * int(n_lines),
        budget,
        rounding_mode="floor",
    )
    return torch.remainder(idx + int(roll), int(n_lines))


def _dedupe_and_fill(indices: torch.Tensor, *, n_lines: int, budget: int) -> torch.Tensor:
    seen: set[int] = set()
    ordered: list[int] = []
    for idx in indices.detach().cpu().tolist():
        item = int(idx)
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    fill_idx = 0
    while len(ordered) < int(budget):
        if fill_idx not in seen:
            seen.add(fill_idx)
            ordered.append(fill_idx)
        fill_idx += 1
        if fill_idx >= int(n_lines) and len(ordered) < int(budget):
            raise ValueError("Could not build a unique fixed mask")
    return torch.as_tensor(ordered[: int(budget)], dtype=torch.long, device=indices.device)


def fixed104_indices(*, n_lines: int, device=None) -> torch.Tensor:
    """Static fixed104-compatible mask family, not the protected DPF greedy trace."""
    if int(n_lines) < 14:
        raise ValueError(f"fixed104 requires at least 14 candidate lines, got {n_lines}")
    first = _equispaced_indices(int(n_lines), 10, roll=0, device=device)
    second_roll = max(1, int(n_lines) // 8)
    second = _equispaced_indices(int(n_lines), 4, roll=second_roll, device=device)
    return _dedupe_and_fill(torch.cat([first, second], dim=0), n_lines=int(n_lines), budget=14)


class VariableMaskBudgetSampler:
    def __init__(
        self,
        *,
        image_shape: tuple[int, int],
        n_lines: int = 112,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
        seed: int = 0,
    ) -> None:
        self.generator = LineMaskGenerator(
            image_shape=image_shape,
            n_lines=int(n_lines),
            device=device,
            dtype=dtype,
        )
        self.seed = int(seed)
        self._sample_count = 0

    @property
    def n_lines(self) -> int:
        return int(self.generator.n_lines)

    @property
    def image_shape(self) -> tuple[int, int]:
        return tuple(self.generator.image_shape)

    def _randperm(self, batch_idx: int) -> torch.Tensor:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(self.seed + self._sample_count * 997 + batch_idx * 101))
        return torch.randperm(self.n_lines, generator=gen).to(device=self.generator.device)

    def compatible_families_for_budget(self, budget: int) -> tuple[str, ...]:
        budget = int(budget)
        families = [MASK_RANDOM, MASK_ROLLED_EQUISPACED, MASK_CLUSTERED_LOCAL, MASK_HEURISTIC, MASK_ADVERSARIAL_GAP]
        if budget == 14:
            families.extend([MASK_FIXED104, MASK_FIXED14])
        if budget == 10:
            families.append(MASK_FIXED10)
        return tuple(families)

    def sample_indices(
        self,
        *,
        budget: int,
        family: Stage1MaskFamily,
        batch_size: int = 1,
        heuristic_map: torch.Tensor | None = None,
        roll: int = 0,
    ) -> torch.Tensor:
        budget = int(budget)
        batch_size = int(batch_size)
        if family not in SUPPORTED_STAGE1_MASK_FAMILIES:
            raise ValueError(f"Unsupported mask family {family!r}")
        if budget < 0 or budget > self.n_lines:
            raise ValueError(f"budget must be in [0,{self.n_lines}], got {budget}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        if family == MASK_FIXED104:
            if budget != 14:
                raise ValueError("fixed104 mask family is only defined for budget=14")
            base = fixed104_indices(n_lines=self.n_lines, device=self.generator.device)
            return base.view(1, -1).expand(batch_size, -1)
        if family == MASK_FIXED14:
            if budget != 14:
                raise ValueError("fixed14 mask family is only defined for budget=14")
            base = _equispaced_indices(self.n_lines, 14, roll=0, device=self.generator.device)
            return base.view(1, -1).expand(batch_size, -1)
        if family == MASK_FIXED10:
            if budget != 10:
                raise ValueError("fixed10 mask family is only defined for budget=10")
            base = _equispaced_indices(self.n_lines, 10, roll=0, device=self.generator.device)
            return base.view(1, -1).expand(batch_size, -1)
        if family == MASK_RANDOM:
            rows = [self._randperm(batch_idx)[:budget] for batch_idx in range(batch_size)]
            self._sample_count += 1
            return torch.stack(rows, dim=0) if budget > 0 else torch.empty(
                (batch_size, 0), dtype=torch.long, device=self.generator.device
            )
        if family == MASK_ROLLED_EQUISPACED:
            rows = [
                _equispaced_indices(self.n_lines, budget, roll=int(roll + batch_idx), device=self.generator.device)
                for batch_idx in range(batch_size)
            ]
            return torch.stack(rows, dim=0) if budget > 0 else torch.empty(
                (batch_size, 0), dtype=torch.long, device=self.generator.device
            )
        if family == MASK_CLUSTERED_LOCAL:
            if budget == 0:
                return torch.empty((batch_size, 0), dtype=torch.long, device=self.generator.device)
            rows = []
            for batch_idx in range(batch_size):
                start = int(roll + batch_idx * max(1, budget // 2)) % self.n_lines
                rows.append(torch.remainder(torch.arange(budget, device=self.generator.device) + start, self.n_lines))
            return torch.stack(rows, dim=0).to(dtype=torch.long)
        if family == MASK_ADVERSARIAL_GAP:
            if budget == 0:
                return torch.empty((batch_size, 0), dtype=torch.long, device=self.generator.device)
            left_count = (budget + 1) // 2
            right_count = budget - left_count
            left = torch.arange(left_count, device=self.generator.device)
            right = torch.arange(self.n_lines - right_count, self.n_lines, device=self.generator.device)
            base = torch.cat([left, right], dim=0).to(dtype=torch.long)
            return base.view(1, -1).expand(batch_size, -1)
        if family == MASK_HEURISTIC:
            if heuristic_map is None:
                raise ValueError("heuristic_generated mask family requires heuristic_map")
            scores = self.generator.linewise_mean(heuristic_map)
            action = self.generator.action_from_scores(scores, k=budget)
            selected = torch.nonzero(action.selected_lines, as_tuple=False)
            rows = []
            for batch_idx in range(int(scores.shape[0])):
                row = selected[selected[:, 0] == batch_idx, 1]
                rows.append(row.to(device=self.generator.device, dtype=torch.long))
            return torch.stack(rows, dim=0) if budget > 0 else torch.empty(
                (int(scores.shape[0]), 0), dtype=torch.long, device=self.generator.device
            )
        raise ValueError(f"Unhandled mask family {family!r}")

    def sample_action(
        self,
        *,
        budget: int,
        family: Stage1MaskFamily,
        batch_size: int = 1,
        heuristic_map: torch.Tensor | None = None,
        roll: int = 0,
    ) -> ActionSet:
        indices = self.sample_indices(
            budget=budget,
            family=family,
            batch_size=batch_size,
            heuristic_map=heuristic_map,
            roll=roll,
        )
        selected = self.generator.selected_lines_from_indices(indices, batch_size=batch_size)
        return ActionSet(
            selected_lines=selected,
            pixel_mask=self.generator.pixel_mask_from_lines(selected),
            line_indices=indices,
        )


class PixelPriorMixture:
    @staticmethod
    def make(
        *,
        kind: Stage1PriorKind,
        target: torch.Tensor,
        prev_x_final: torch.Tensor | None = None,
        prev_prev_x_final: torch.Tensor | None = None,
        obs: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        cached_dpf_prior: torch.Tensor | None = None,
        ema_alpha: float = 0.8,
        degraded_noise_scale: float = 0.03,
    ) -> torch.Tensor:
        if kind not in SUPPORTED_STAGE1_PRIORS:
            raise ValueError(f"Unsupported prior kind {kind!r}")
        target = _as_nchw("target", target)
        if kind == PRIOR_COPY_LAST:
            return torch.zeros_like(target) if prev_x_final is None else _as_nchw("prev_x_final", prev_x_final).to(
                device=target.device, dtype=target.dtype
            )
        if kind == PRIOR_EMA:
            prev = PixelPriorMixture.make(kind=PRIOR_COPY_LAST, target=target, prev_x_final=prev_x_final)
            if prev_prev_x_final is None:
                return prev
            prev_prev = _as_nchw("prev_prev_x_final", prev_prev_x_final).to(device=target.device, dtype=target.dtype)
            alpha = float(ema_alpha)
            if not 0.0 <= alpha <= 1.0:
                raise ValueError(f"ema_alpha must be in [0,1], got {alpha}")
            return alpha * prev + (1.0 - alpha) * prev_prev
        if kind == PRIOR_DEGRADED_GT:
            blurred = F.avg_pool2d(target, kernel_size=3, stride=1, padding=1)
            return blurred + float(degraded_noise_scale) * torch.randn_like(target)
        if kind in {PRIOR_WEAK_INTERPOLATION, PRIOR_ZERO_FILL}:
            if obs is None or mask is None:
                raise ValueError(f"{kind} prior requires obs and mask")
            obs = _as_nchw("obs", obs).to(device=target.device, dtype=target.dtype)
            mask = _as_nchw("mask", mask).to(device=target.device, dtype=target.dtype)
            zero_fill = torch.where(mask > 0, obs, torch.zeros_like(obs))
            if kind == PRIOR_ZERO_FILL:
                return zero_fill
            blurred = F.avg_pool2d(zero_fill, kernel_size=5, stride=1, padding=2)
            return torch.where(mask > 0, obs, blurred)
        if kind == PRIOR_CACHED_DPF:
            if cached_dpf_prior is None:
                raise NotImplementedError("cached_dpf prior requires an externally supplied cached_dpf_prior tensor")
            return _as_nchw("cached_dpf_prior", cached_dpf_prior).to(device=target.device, dtype=target.dtype)
        raise ValueError(f"Unhandled prior kind {kind!r}")


@dataclass(frozen=True)
class Stage1PBUBatch:
    x_gt: torch.Tensor
    x_prior: torch.Tensor
    u_prior: torch.Tensor
    obs: torch.Tensor
    mask: torch.Tensor
    selected_lines: torch.Tensor
    line_history: torch.Tensor
    budget: BudgetState
    prev_x_final: torch.Tensor
    prev_u_post: torch.Tensor
    change_map: torch.Tensor
    mask_family: str
    prior_kind: str

    def __post_init__(self) -> None:
        base_shape = tuple(_as_nchw("x_gt", self.x_gt).shape)
        for name in [
            "x_prior",
            "u_prior",
            "obs",
            "mask",
            "prev_x_final",
            "prev_u_post",
            "change_map",
        ]:
            value = _as_nchw(name, getattr(self, name))
            if tuple(value.shape) != base_shape:
                raise ValueError(f"{name} must match x_gt shape, got {tuple(value.shape)} and {base_shape}")
        if self.selected_lines.ndim != 2:
            raise ValueError(f"selected_lines must be [B,L], got {tuple(self.selected_lines.shape)}")
        if tuple(self.line_history.shape) != tuple(self.selected_lines.shape):
            raise ValueError("line_history must match selected_lines")
        if int(self.selected_lines.shape[0]) != int(base_shape[0]):
            raise ValueError("selected_lines batch must match x_gt batch")

    @property
    def residual_obs(self) -> torch.Tensor:
        return self.mask * (self.obs - self.x_prior)

    @property
    def x_dc(self) -> torch.Tensor:
        return HardProjection.apply(x_raw=self.x_prior, obs=self.obs, mask=self.mask)

    def budget_embedding(self, *, n_lines: int) -> torch.Tensor:
        return self.budget.token_maps(self.x_gt, n_lines=int(n_lines))

    @property
    def prior_is_training_only(self) -> bool:
        return self.prior_kind in TRAINING_ONLY_PRIORS

    def replace(self, **updates) -> "Stage1PBUBatch":
        return replace(self, **updates)


@dataclass(frozen=True)
class Stage1PBUOutput:
    x_raw: torch.Tensor
    x_final: torch.Tensor
    u_post: torch.Tensor
    delta_x: torch.Tensor
    logvar: torch.Tensor
    delta_raw: torch.Tensor
    observed_consistency_l1: torch.Tensor
    observed_raw_l1: torch.Tensor


@dataclass(frozen=True)
class Stage1PBUStateOutput:
    state: PixelBeliefState
    aux: Stage1PBUOutput


def make_stage1_pbu_batch(
    *,
    target: torch.Tensor,
    sampler: VariableMaskBudgetSampler,
    budget: int,
    mask_family: Stage1MaskFamily,
    prior_kind: Stage1PriorKind,
    prev_x_final: torch.Tensor | None = None,
    prev_prev_x_final: torch.Tensor | None = None,
    prev_u_post: torch.Tensor | None = None,
    cached_dpf_prior: torch.Tensor | None = None,
    heuristic_map: torch.Tensor | None = None,
    roll: int = 0,
) -> Stage1PBUBatch:
    target = _as_nchw("target", target).to(device=sampler.generator.device, dtype=sampler.generator.dtype)
    batch_size = int(target.shape[0])
    action = sampler.sample_action(
        budget=int(budget),
        family=mask_family,
        batch_size=batch_size,
        heuristic_map=heuristic_map,
        roll=roll,
    )
    measurement = MeasurementOperator().measure(target, action)
    prev = torch.zeros_like(target) if prev_x_final is None else _as_nchw("prev_x_final", prev_x_final).to(
        device=target.device, dtype=target.dtype
    )
    prev_u = torch.ones_like(target) if prev_u_post is None else _as_nchw("prev_u_post", prev_u_post).to(
        device=target.device, dtype=target.dtype
    )
    x_prior = PixelPriorMixture.make(
        kind=prior_kind,
        target=target,
        prev_x_final=prev,
        prev_prev_x_final=prev_prev_x_final,
        obs=measurement.obs,
        mask=measurement.mask,
        cached_dpf_prior=cached_dpf_prior,
    )
    u_prior = torch.clamp(torch.abs(prev - x_prior), min=0.0)
    change_map = torch.abs(target - prev)
    return Stage1PBUBatch(
        x_gt=target,
        x_prior=x_prior,
        u_prior=u_prior,
        obs=measurement.obs,
        mask=measurement.mask,
        selected_lines=action.selected_lines,
        line_history=action.selected_lines.to(dtype=target.dtype),
        budget=BudgetState(total=int(budget), used=int(budget), phase=0, max_updates=1),
        prev_x_final=prev,
        prev_u_post=prev_u,
        change_map=change_map,
        mask_family=str(mask_family),
        prior_kind=str(prior_kind),
    )


def make_stage1_pbu_batch_from_state(
    state: PixelBeliefState,
    *,
    target_for_metrics: torch.Tensor | None = None,
    mask_family: str = "from_state",
    prior_kind: str = "from_state",
) -> Stage1PBUBatch:
    if state.x_prior is None:
        raise ValueError("PixelBeliefState.x_prior is required for Stage 1 PBU")
    if state.u_prior is None:
        raise ValueError("PixelBeliefState.u_prior is required for Stage 1 PBU")
    x_gt = state.x_final if target_for_metrics is None else _as_nchw("target_for_metrics", target_for_metrics)
    change_map = torch.abs(state.x_final - state.x_prior)
    return Stage1PBUBatch(
        x_gt=x_gt,
        x_prior=state.x_prior,
        u_prior=state.u_prior,
        obs=state.obs,
        mask=state.mask,
        selected_lines=state.selected_lines,
        line_history=state.line_history,
        budget=state.budget,
        prev_x_final=state.x_final,
        prev_u_post=state.u_post,
        change_map=change_map,
        mask_family=str(mask_family),
        prior_kind=str(prior_kind),
    )


def make_shuffled_observation_batch(batch: Stage1PBUBatch) -> Stage1PBUBatch:
    flat = batch.obs.reshape(int(batch.obs.shape[0]), -1)
    shuffled = torch.roll(flat, shifts=1, dims=1).reshape_as(batch.obs)
    shuffled = torch.where(batch.mask > 0, shuffled, torch.zeros_like(shuffled))
    return batch.replace(obs=shuffled)


def make_wrong_mask_batch(batch: Stage1PBUBatch) -> Stage1PBUBatch:
    wrong_mask = torch.roll(batch.mask, shifts=1, dims=-1)
    return batch.replace(mask=wrong_mask)


class Stage1PBUWrapper(nn.Module):
    def __init__(
        self,
        model: PixelBeliefUpdater | None = None,
        *,
        n_lines: int = 112,
        base_channels: int = 32,
        channel_mult: tuple[int, ...] = (1, 2),
        num_res_blocks: int = 1,
        groupnorm_groups: int = 8,
        delta_clip_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_lines = int(n_lines)
        self.model = model or PixelBeliefUpdater(
            base_channels=base_channels,
            channel_mult=channel_mult,
            num_res_blocks=num_res_blocks,
            groupnorm_groups=groupnorm_groups,
            delta_clip_scale=delta_clip_scale,
        )
        self.projection = HardProjection()

    def forward(
        self,
        batch: Stage1PBUBatch,
        *,
        ablation: Stage1PBUAblation | Stage1ModelAblationName | str | None = None,
    ) -> Stage1PBUOutput:
        ablation_cfg = resolve_stage1_pbu_ablation(ablation)
        observed_count = torch.full(
            (int(batch.selected_lines.shape[0]),),
            float(batch.budget.used),
            device=batch.selected_lines.device,
            dtype=torch.float32,
        )
        cond = build_pixel_belief_condition_maps(
            obs_mask=batch.mask,
            total_budget=batch.budget.total,
            observed_count=observed_count,
            group_index=batch.budget.phase,
            n_possible_actions=self.n_lines,
        )
        if ablation_cfg.disable_budget_tokens:
            for key in ("budget_token", "observed_count_token", "group_index_token"):
                cond[key] = torch.zeros_like(cond[key])
        residual_obs = torch.zeros_like(batch.residual_obs) if ablation_cfg.disable_residual else batch.residual_obs
        delta_x, logvar, delta_raw = self.model(
            x_prior=batch.x_prior,
            x_dc=batch.x_dc,
            obs_buffer=batch.obs,
            obs_mask=batch.mask,
            residual_obs=residual_obs,
            history_map=batch.mask,
            **cond,
        )
        x_raw = batch.x_prior + delta_x
        if ablation_cfg.disable_hard_projection:
            x_final = x_raw
            u_post = F.softplus(logvar)
        else:
            x_final = self.projection.apply(x_raw=x_raw, obs=batch.obs, mask=batch.mask)
            u_post = torch.where(batch.mask > 0, torch.zeros_like(logvar), F.softplus(logvar))
        return Stage1PBUOutput(
            x_raw=x_raw,
            x_final=x_final,
            u_post=u_post,
            delta_x=delta_x,
            logvar=logvar,
            delta_raw=delta_raw,
            observed_consistency_l1=_masked_l1(x_final - batch.obs, batch.mask),
            observed_raw_l1=_masked_l1(x_raw - batch.obs, batch.mask),
        )

    def forward_state(self, state: PixelBeliefState) -> Stage1PBUStateOutput:
        batch = make_stage1_pbu_batch_from_state(state)
        aux = self.forward(batch)
        next_state = state.replace(
            x_final=aux.x_final,
            u_post=aux.u_post,
            x_prior=batch.x_prior,
            u_prior=batch.u_prior,
            mask=batch.mask,
            obs=batch.obs,
            selected_lines=batch.selected_lines,
            line_history=batch.line_history,
            budget=batch.budget,
        )
        return Stage1PBUStateOutput(state=next_state, aux=aux)
