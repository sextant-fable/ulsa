from __future__ import annotations

from dataclasses import dataclass
from typing import Any


FIXED104_TOTAL_BUDGET = 14
FIXED104_GROUP_PLAN = (10, 4)

SAMPLING_MODE_FIXED104_BASELINE = "fixed104_baseline"
SAMPLING_MODE_RECONSTRUCT_GIVEN_MASK = "reconstruct_given_mask"
SAMPLING_MODE_ADAPTIVE_VARIABLE_BUDGET = "adaptive_variable_budget"
SUPPORTED_SAMPLING_MODES = {
    SAMPLING_MODE_FIXED104_BASELINE,
    SAMPLING_MODE_RECONSTRUCT_GIVEN_MASK,
    SAMPLING_MODE_ADAPTIVE_VARIABLE_BUDGET,
}

GROUPING_MODE_FIXED104 = "fixed104"
GROUPING_MODE_ONESHOT = "oneshot"
GROUPING_MODE_RATIO2 = "ratio2"
GROUPING_MODE_RATIO3 = "ratio3"
SUPPORTED_GROUPING_MODES = {
    GROUPING_MODE_FIXED104,
    GROUPING_MODE_ONESHOT,
    GROUPING_MODE_RATIO2,
    GROUPING_MODE_RATIO3,
}


@dataclass(frozen=True)
class SamplingPlan:
    mode: str
    total_budget: int
    grouping_mode: str
    max_updates: int
    group_schedule: tuple[int, ...]
    source: str
    explicit_group_plan: tuple[int, ...] | None = None
    legacy_fixed_group_schedule: tuple[int, ...] | None = None
    is_fixed104_compat: bool = False
    uses_active_selection: bool = True
    uses_external_mask: bool = False


def _coerce_nonnegative_int(value: Any, *, name: str) -> int:
    try:
        resolved = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc
    if resolved < 0:
        raise ValueError(f"{name} must be non-negative, got {resolved}")
    return int(resolved)


def _coerce_positive_int(value: Any, *, name: str) -> int:
    resolved = _coerce_nonnegative_int(value, name=name)
    if resolved <= 0:
        raise ValueError(f"{name} must be positive, got {resolved}")
    return int(resolved)


def parse_group_plan(value: Any) -> tuple[int, ...] | None:
    if value is None:
        return None
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        if raw.startswith("[") and raw.endswith("]"):
            raw = raw[1:-1]
        tokens = [tok.strip() for tok in raw.split(",") if tok.strip()]
        if not tokens:
            return None
        parsed = tuple(int(tok) for tok in tokens)
    elif isinstance(value, (list, tuple)):
        if len(value) == 0:
            return None
        parsed = tuple(int(x) for x in value)
    else:
        raise ValueError(f"group_plan must be a comma string, list, or tuple, got {type(value).__name__}")
    if any(x <= 0 for x in parsed):
        raise ValueError(f"group_plan must contain positive integers, got {list(parsed)}")
    return parsed


def _allocate_by_ratios(total_budget: int, ratios: list[float]) -> list[int]:
    if total_budget == 0:
        return []
    n_groups = len(ratios)
    if n_groups <= 0:
        raise ValueError("ratios must contain at least one value")
    if total_budget < n_groups:
        n_groups = int(total_budget)
        ratios = ratios[:n_groups]
    if n_groups == 1:
        return [int(total_budget)]

    ratio_sum = float(sum(ratios[:n_groups]))
    if ratio_sum <= 0.0:
        raise ValueError(f"ratios must sum to a positive value, got {ratios}")
    normalized = [float(x) / ratio_sum for x in ratios[:n_groups]]
    targets = [float(total_budget) * x for x in normalized]
    allocations = [max(1, int(target)) for target in targets]

    while sum(allocations) < total_budget:
        fractions = [targets[i] - int(targets[i]) for i in range(n_groups)]
        order = sorted(range(n_groups), key=lambda i: (fractions[i], normalized[i]), reverse=True)
        for idx in order:
            if sum(allocations) >= total_budget:
                break
            allocations[idx] += 1

    while sum(allocations) > total_budget:
        fractions = [targets[i] - int(targets[i]) for i in range(n_groups)]
        candidates = [i for i in range(n_groups) if allocations[i] > 1]
        if not candidates:
            break
        idx = sorted(candidates, key=lambda i: (fractions[i], normalized[i]))[0]
        allocations[idx] -= 1

    if sum(allocations) != total_budget:
        raise ValueError(
            f"failed to allocate total_budget={total_budget} across ratios={ratios}; "
            f"got {allocations}"
        )
    return [int(x) for x in allocations]


def make_group_schedule(
    total_budget: int,
    grouping_mode: str,
    max_updates: int,
    first_ratio: float = 0.7,
) -> list[int]:
    total_budget = _coerce_nonnegative_int(total_budget, name="total_budget")
    grouping_mode = str(grouping_mode).strip().lower()
    if grouping_mode not in SUPPORTED_GROUPING_MODES:
        raise ValueError(
            f"grouping_mode must be one of {sorted(SUPPORTED_GROUPING_MODES)}, got {grouping_mode!r}"
        )
    if total_budget == 0:
        return []
    max_updates = _coerce_positive_int(max_updates, name="max_updates")

    if grouping_mode == GROUPING_MODE_FIXED104:
        if total_budget != FIXED104_TOTAL_BUDGET:
            raise ValueError(
                f"grouping_mode='fixed104' requires total_budget={FIXED104_TOTAL_BUDGET}, "
                f"got {total_budget}"
            )
        if max_updates < len(FIXED104_GROUP_PLAN):
            raise ValueError(
                f"grouping_mode='fixed104' requires max_updates>={len(FIXED104_GROUP_PLAN)}, "
                f"got {max_updates}"
            )
        return [int(x) for x in FIXED104_GROUP_PLAN]

    if grouping_mode == GROUPING_MODE_ONESHOT or max_updates <= 1:
        return [int(total_budget)]

    if grouping_mode == GROUPING_MODE_RATIO2:
        first_ratio = float(first_ratio)
        if not 0.0 < first_ratio < 1.0:
            raise ValueError(f"first_ratio must be in (0, 1), got {first_ratio}")
        ratios = [first_ratio, 1.0 - first_ratio]
        effective = min(int(max_updates), len(ratios), int(total_budget))
        if effective < len(ratios):
            ratios = ratios[: max(1, effective)]
            if len(ratios) > 1:
                ratios[-1] = 1.0 - sum(ratios[:-1])
        return _allocate_by_ratios(total_budget, ratios)

    if grouping_mode == GROUPING_MODE_RATIO3:
        ratios = [0.5, 0.3, 0.2]
        effective = min(int(max_updates), len(ratios), int(total_budget))
        if effective < len(ratios):
            kept = ratios[: max(1, effective)]
            if len(kept) > 1:
                kept[-1] = 1.0 - sum(kept[:-1])
            ratios = kept
        return _allocate_by_ratios(total_budget, ratios)

    raise ValueError(f"unsupported grouping_mode={grouping_mode!r}")


def _legacy_group_schedule(total_budget: int, line_update_batch_size: int) -> tuple[int, ...]:
    total_budget = _coerce_nonnegative_int(total_budget, name="total_budget")
    if total_budget == 0:
        return tuple()
    line_update_batch_size = _coerce_positive_int(line_update_batch_size, name="line_update_batch_size")
    groups: list[int] = []
    remaining = int(total_budget)
    while remaining > 0:
        group_size = min(int(line_update_batch_size), int(remaining))
        groups.append(int(group_size))
        remaining -= int(group_size)
    return tuple(groups)


def _validate_schedule(
    schedule: tuple[int, ...],
    *,
    total_budget: int,
    max_updates: int | None,
    field_name: str,
) -> None:
    if any(x <= 0 for x in schedule):
        raise ValueError(f"{field_name} must contain positive integers, got {list(schedule)}")
    schedule_sum = int(sum(schedule))
    if schedule_sum != int(total_budget):
        raise ValueError(
            f"{field_name} sum must equal total_budget. "
            f"got schedule={list(schedule)}, sum={schedule_sum}, total_budget={total_budget}"
        )
    if max_updates is not None and len(schedule) > int(max_updates):
        raise ValueError(
            f"{field_name} length must be <= max_updates. "
            f"got schedule={list(schedule)}, max_updates={max_updates}"
        )


def _has_explicit_sampling_values(sampling_cfg: dict[str, Any]) -> bool:
    return any(value is not None for value in sampling_cfg.values())


def _sampling_value(sampling_cfg: dict[str, Any], key: str, default: Any) -> Any:
    value = sampling_cfg.get(key, default)
    return default if value is None else value


def resolve_sampling_plan(
    dynamic_cfg: dict[str, Any] | None,
    *,
    n_possible_actions: int | None = None,
    legacy_line_update_batch_size: int | None = None,
) -> SamplingPlan:
    cfg = dict(dynamic_cfg or {})
    sampling_cfg_raw = cfg.get("sampling", None)
    sampling_cfg = sampling_cfg_raw if isinstance(sampling_cfg_raw, dict) else None

    if sampling_cfg is None or not _has_explicit_sampling_values(sampling_cfg):
        total_budget = max(1, int(cfg.get("fixed_budget_lines", 7)))
        line_update_batch_size = (
            int(legacy_line_update_batch_size)
            if legacy_line_update_batch_size is not None
            else int(cfg.get("line_update_batch_size", 1))
        )
        legacy_fixed_schedule = parse_group_plan(cfg.get("fixed_group_schedule", None))
        if legacy_fixed_schedule is not None:
            _validate_schedule(
                legacy_fixed_schedule,
                total_budget=total_budget,
                max_updates=None,
                field_name="fixed_group_schedule",
            )
            group_schedule = legacy_fixed_schedule
        else:
            group_schedule = _legacy_group_schedule(total_budget, line_update_batch_size)

        if n_possible_actions is not None and total_budget > int(n_possible_actions):
            raise ValueError(
                f"total_budget must be <= n_possible_actions, got {total_budget}>{n_possible_actions}"
            )
        is_fixed104 = (
            int(total_budget) == FIXED104_TOTAL_BUDGET
            and tuple(group_schedule) == FIXED104_GROUP_PLAN
        )
        return SamplingPlan(
            mode=SAMPLING_MODE_FIXED104_BASELINE if is_fixed104 else SAMPLING_MODE_ADAPTIVE_VARIABLE_BUDGET,
            total_budget=int(total_budget),
            grouping_mode=GROUPING_MODE_FIXED104 if is_fixed104 else "legacy",
            max_updates=int(len(group_schedule)),
            group_schedule=tuple(int(x) for x in group_schedule),
            source="legacy",
            explicit_group_plan=None,
            legacy_fixed_group_schedule=legacy_fixed_schedule,
            is_fixed104_compat=bool(is_fixed104),
            uses_active_selection=True,
            uses_external_mask=False,
        )

    mode = str(_sampling_value(sampling_cfg, "mode", SAMPLING_MODE_FIXED104_BASELINE)).strip().lower()
    if mode not in SUPPORTED_SAMPLING_MODES:
        raise ValueError(f"sampling.mode must be one of {sorted(SUPPORTED_SAMPLING_MODES)}, got {mode!r}")

    default_total = FIXED104_TOTAL_BUDGET if mode == SAMPLING_MODE_FIXED104_BASELINE else int(
        cfg.get("fixed_budget_lines", 7)
    )
    total_budget = _coerce_nonnegative_int(
        _sampling_value(sampling_cfg, "total_budget", default_total),
        name="sampling.total_budget",
    )
    if n_possible_actions is not None and total_budget > int(n_possible_actions):
        raise ValueError(
            f"sampling.total_budget must be <= n_possible_actions, got {total_budget}>{n_possible_actions}"
        )

    default_grouping = GROUPING_MODE_FIXED104 if mode == SAMPLING_MODE_FIXED104_BASELINE else GROUPING_MODE_ONESHOT
    grouping_mode = str(_sampling_value(sampling_cfg, "grouping_mode", default_grouping)).strip().lower()
    if grouping_mode not in SUPPORTED_GROUPING_MODES:
        raise ValueError(
            f"sampling.grouping_mode must be one of {sorted(SUPPORTED_GROUPING_MODES)}, "
            f"got {grouping_mode!r}"
        )

    explicit_group_plan = parse_group_plan(sampling_cfg.get("group_plan", None))
    default_max_updates = len(explicit_group_plan) if explicit_group_plan is not None else (
        len(FIXED104_GROUP_PLAN) if grouping_mode == GROUPING_MODE_FIXED104 else 1
    )
    max_updates = _coerce_positive_int(
        _sampling_value(sampling_cfg, "max_updates", default_max_updates),
        name="sampling.max_updates",
    )
    first_ratio = float(_sampling_value(sampling_cfg, "first_ratio", 0.7))

    if explicit_group_plan is not None:
        _validate_schedule(
            explicit_group_plan,
            total_budget=total_budget,
            max_updates=max_updates,
            field_name="sampling.group_plan",
        )
        group_schedule = explicit_group_plan
    else:
        group_schedule = tuple(
            make_group_schedule(
                total_budget=total_budget,
                grouping_mode=grouping_mode,
                max_updates=max_updates,
                first_ratio=first_ratio,
            )
        )

    if mode == SAMPLING_MODE_FIXED104_BASELINE and tuple(group_schedule) != FIXED104_GROUP_PLAN:
        raise ValueError(
            "sampling.mode='fixed104_baseline' requires the fixed104 compatibility schedule"
        )

    is_fixed104 = (
        mode == SAMPLING_MODE_FIXED104_BASELINE
        or (
            int(total_budget) == FIXED104_TOTAL_BUDGET
            and tuple(group_schedule) == FIXED104_GROUP_PLAN
            and grouping_mode == GROUPING_MODE_FIXED104
        )
    )
    return SamplingPlan(
        mode=mode,
        total_budget=int(total_budget),
        grouping_mode=grouping_mode,
        max_updates=int(max_updates),
        group_schedule=tuple(int(x) for x in group_schedule),
        source="sampling",
        explicit_group_plan=explicit_group_plan,
        legacy_fixed_group_schedule=None,
        is_fixed104_compat=bool(is_fixed104),
        uses_active_selection=mode != SAMPLING_MODE_RECONSTRUCT_GIVEN_MASK,
        uses_external_mask=mode == SAMPLING_MODE_RECONSTRUCT_GIVEN_MASK,
    )


def validate_reconstruct_given_mask(
    obs_mask: Any,
    obs_buffer: Any,
    *,
    image_shape: tuple[int, int] | None = None,
) -> dict[str, Any]:
    mask_shape = tuple(int(x) for x in getattr(obs_mask, "shape", ()))
    buffer_shape = tuple(int(x) for x in getattr(obs_buffer, "shape", ()))
    if not mask_shape:
        raise ValueError("obs_mask must have a shape")
    if not buffer_shape:
        raise ValueError("obs_buffer must have a shape")
    if mask_shape != buffer_shape:
        raise ValueError(f"obs_mask and obs_buffer shapes must match, got {mask_shape} and {buffer_shape}")
    if len(mask_shape) not in {2, 3, 4}:
        raise ValueError(f"obs_mask must be 2D, 3D, or 4D, got shape={mask_shape}")
    if image_shape is not None:
        h, w = (int(image_shape[0]), int(image_shape[1]))
        if len(mask_shape) == 2:
            spatial_shape = mask_shape
        elif len(mask_shape) == 3:
            spatial_shape = mask_shape[:2]
        else:
            spatial_shape = mask_shape[-2:]
        if tuple(spatial_shape) != (h, w):
            raise ValueError(
                f"obs_mask spatial shape must match image_shape={(h, w)}, got {spatial_shape}"
            )
    return {
        "mask_shape": mask_shape,
        "buffer_shape": buffer_shape,
    }
