import numpy as np
import pytest

from ulsa.sampling_schedule import (
    FIXED104_GROUP_PLAN,
    FIXED104_TOTAL_BUDGET,
    GROUPING_MODE_FIXED104,
    GROUPING_MODE_ONESHOT,
    GROUPING_MODE_RATIO2,
    GROUPING_MODE_RATIO3,
    SAMPLING_MODE_ADAPTIVE_VARIABLE_BUDGET,
    SAMPLING_MODE_FIXED104_BASELINE,
    SAMPLING_MODE_RECONSTRUCT_GIVEN_MASK,
    make_group_schedule,
    resolve_sampling_plan,
    validate_reconstruct_given_mask,
)


def test_fixed104_schedule_is_explicit_compatibility_mode():
    assert make_group_schedule(
        total_budget=FIXED104_TOTAL_BUDGET,
        grouping_mode=GROUPING_MODE_FIXED104,
        max_updates=2,
    ) == list(FIXED104_GROUP_PLAN)

    with pytest.raises(ValueError, match="requires total_budget"):
        make_group_schedule(total_budget=10, grouping_mode=GROUPING_MODE_FIXED104, max_updates=2)


def test_ratio2_schedule_examples_sum_to_budget_without_empty_groups():
    expected = {
        0: [],
        2: [1, 1],
        4: [3, 1],
        7: [5, 2],
        10: [7, 3],
        14: [10, 4],
        21: [15, 6],
        28: [20, 8],
    }
    for budget, schedule in expected.items():
        assert make_group_schedule(
            total_budget=budget,
            grouping_mode=GROUPING_MODE_RATIO2,
            max_updates=2,
        ) == schedule
        assert sum(schedule) == budget
        assert all(group > 0 for group in schedule)


def test_ratio3_and_oneshot_schedules_cover_stage05_budgets():
    budgets = [0, 2, 4, 7, 10, 14, 21, 28]
    for budget in budgets:
        ratio3 = make_group_schedule(
            total_budget=budget,
            grouping_mode=GROUPING_MODE_RATIO3,
            max_updates=3,
        )
        assert sum(ratio3) == budget
        assert all(group > 0 for group in ratio3)
        if budget > 0:
            assert len(ratio3) <= min(3, budget)

        oneshot = make_group_schedule(
            total_budget=budget,
            grouping_mode=GROUPING_MODE_ONESHOT,
            max_updates=1,
        )
        assert oneshot == ([] if budget == 0 else [budget])


def test_legacy_fixed104_resolves_to_protected_compatibility_plan():
    plan = resolve_sampling_plan(
        {
            "fixed_budget_lines": 14,
            "fixed_group_schedule": "10,4",
            "line_update_batch_size": 10,
        },
        n_possible_actions=112,
    )
    assert plan.source == "legacy"
    assert plan.mode == SAMPLING_MODE_FIXED104_BASELINE
    assert plan.total_budget == 14
    assert plan.group_schedule == FIXED104_GROUP_PLAN
    assert plan.legacy_fixed_group_schedule == FIXED104_GROUP_PLAN
    assert plan.is_fixed104_compat
    assert plan.uses_active_selection


def test_null_sampling_block_preserves_legacy_behavior():
    plan = resolve_sampling_plan(
        {
            "fixed_budget_lines": 7,
            "line_update_batch_size": 3,
            "fixed_group_schedule": None,
            "sampling": {
                "mode": None,
                "total_budget": None,
                "grouping_mode": None,
                "max_updates": None,
                "group_plan": None,
            },
        },
        n_possible_actions=112,
    )
    assert plan.source == "legacy"
    assert plan.total_budget == 7
    assert plan.group_schedule == (3, 3, 1)
    assert plan.legacy_fixed_group_schedule is None


def test_explicit_adaptive_variable_budget_plan_is_update_count_controlled():
    plan = resolve_sampling_plan(
        {
            "sampling": {
                "mode": SAMPLING_MODE_ADAPTIVE_VARIABLE_BUDGET,
                "total_budget": 28,
                "grouping_mode": GROUPING_MODE_RATIO2,
                "max_updates": 2,
            }
        },
        n_possible_actions=112,
    )
    assert plan.mode == SAMPLING_MODE_ADAPTIVE_VARIABLE_BUDGET
    assert plan.total_budget == 28
    assert plan.group_schedule == (20, 8)
    assert plan.max_updates == 2
    assert plan.uses_active_selection


def test_explicit_group_plan_validation():
    plan = resolve_sampling_plan(
        {
            "sampling": {
                "mode": SAMPLING_MODE_ADAPTIVE_VARIABLE_BUDGET,
                "total_budget": 10,
                "grouping_mode": GROUPING_MODE_RATIO2,
                "max_updates": 3,
                "group_plan": [4, 3, 3],
            }
        },
        n_possible_actions=112,
    )
    assert plan.group_schedule == (4, 3, 3)

    with pytest.raises(ValueError, match="sum must equal total_budget"):
        resolve_sampling_plan(
            {
                "sampling": {
                    "mode": SAMPLING_MODE_ADAPTIVE_VARIABLE_BUDGET,
                    "total_budget": 10,
                    "group_plan": [4, 4],
                }
            },
            n_possible_actions=112,
        )


def _simulate_unique_greedy_selection(group_schedule, *, n_possible_actions):
    scores = np.arange(n_possible_actions, dtype=np.float32)
    available = np.ones((n_possible_actions,), dtype=bool)
    selected = []
    groups = []
    for group_size in group_schedule:
        current_group = []
        for _ in range(int(group_size)):
            masked_scores = np.where(available, scores, -np.inf)
            line_idx = int(np.argmax(masked_scores))
            available[line_idx] = False
            selected.append(line_idx)
            current_group.append(line_idx)
        groups.append(current_group)
    return selected, groups


def test_adaptive_variable_budget_smoke_selects_exact_unique_budget():
    for budget in [2, 4, 7, 10, 14, 21, 28]:
        plan = resolve_sampling_plan(
            {
                "sampling": {
                    "mode": SAMPLING_MODE_ADAPTIVE_VARIABLE_BUDGET,
                    "total_budget": budget,
                    "grouping_mode": GROUPING_MODE_RATIO2,
                    "max_updates": 2,
                }
            },
            n_possible_actions=112,
        )
        selected, groups = _simulate_unique_greedy_selection(
            plan.group_schedule,
            n_possible_actions=112,
        )
        assert len(selected) == budget
        assert len(set(selected)) == budget
        assert [len(group) for group in groups] == list(plan.group_schedule)


def test_reconstruct_given_mask_accepts_random_equispaced_and_clustered_masks():
    h, w = 112, 112
    rng = np.random.default_rng(123)
    random_mask = (rng.random((h, w)) > 0.95).astype(np.float32)
    equispaced_mask = np.zeros((h, w), dtype=np.float32)
    equispaced_mask[:, ::16] = 1.0
    clustered_mask = np.zeros((h, w), dtype=np.float32)
    clustered_mask[40:60, 48:72] = 1.0

    for mask in [random_mask, equispaced_mask, clustered_mask]:
        buffer = rng.normal(size=(h, w)).astype(np.float32) * mask
        result = validate_reconstruct_given_mask(mask, buffer, image_shape=(h, w))
        assert result["mask_shape"] == (h, w)
        assert result["buffer_shape"] == (h, w)

    plan = resolve_sampling_plan(
        {
            "sampling": {
                "mode": SAMPLING_MODE_RECONSTRUCT_GIVEN_MASK,
                "total_budget": 0,
                "grouping_mode": GROUPING_MODE_ONESHOT,
                "max_updates": 1,
            }
        },
        n_possible_actions=112,
    )
    assert plan.mode == SAMPLING_MODE_RECONSTRUCT_GIVEN_MASK
    assert plan.group_schedule == tuple()
    assert plan.uses_external_mask
    assert not plan.uses_active_selection
