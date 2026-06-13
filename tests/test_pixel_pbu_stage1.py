import py_compile
from pathlib import Path

import torch

from ulsa.pixel_pbu_stage1 import (
    MASK_ADVERSARIAL_GAP,
    MASK_CLUSTERED_LOCAL,
    MASK_FIXED10,
    MASK_FIXED104,
    MASK_FIXED14,
    MASK_HEURISTIC,
    MASK_RANDOM,
    MASK_ROLLED_EQUISPACED,
    MODEL_ABLATION_NO_BUDGET_TOKEN,
    MODEL_ABLATION_NO_HARD_PROJECTION,
    MODEL_ABLATION_NO_RESIDUAL,
    PRIOR_CACHED_DPF,
    PRIOR_COPY_LAST,
    PRIOR_DEGRADED_GT,
    PRIOR_EMA,
    PRIOR_WEAK_INTERPOLATION,
    PRIOR_ZERO_FILL,
    STAGE1_BUDGETS,
    TRAINING_ONLY_PRIORS,
    PixelPriorMixture,
    Stage1PBUAblation,
    Stage1PBUWrapper,
    VariableMaskBudgetSampler,
    make_shuffled_observation_batch,
    make_stage1_pbu_batch,
    make_wrong_mask_batch,
)
from ulsa.pixel_belief_state import PixelBeliefState


def _target(batch_size=2, height=32, width=32):
    torch.manual_seed(41)
    return torch.randn(batch_size, 1, height, width)


def test_stage1_modules_compile():
    root = Path(__file__).resolve().parents[1]
    py_compile.compile(str(root / "ulsa/pixel_pbu_stage1.py"), doraise=True)


def test_variable_mask_sampler_covers_required_budgets():
    sampler = VariableMaskBudgetSampler(image_shape=(32, 32), n_lines=32, seed=13)
    for budget in STAGE1_BUDGETS:
        for family in [MASK_RANDOM, MASK_ROLLED_EQUISPACED, MASK_CLUSTERED_LOCAL, MASK_ADVERSARIAL_GAP]:
            action = sampler.sample_action(budget=budget, family=family, batch_size=2, roll=3)
            assert tuple(action.selected_lines.shape) == (2, 32)
            assert tuple(action.pixel_mask.shape) == (2, 1, 32, 32)
            assert action.pixel_mask.dtype == torch.float32
            assert action.counts.tolist() == [budget, budget]
            assert torch.isfinite(action.pixel_mask).all()


def test_stage1_named_mask_families_shape_device_dtype():
    sampler = VariableMaskBudgetSampler(image_shape=(112, 112), n_lines=112, seed=7)
    heuristic_map = torch.linspace(0.0, 1.0, steps=112).view(1, 1, 1, 112).expand(2, 1, 112, 112)
    cases = [
        (MASK_FIXED104, 14, None),
        (MASK_FIXED14, 14, None),
        (MASK_FIXED10, 10, None),
        (MASK_RANDOM, 28, None),
        (MASK_ROLLED_EQUISPACED, 21, None),
        (MASK_CLUSTERED_LOCAL, 7, None),
        (MASK_ADVERSARIAL_GAP, 4, None),
        (MASK_HEURISTIC, 14, heuristic_map),
    ]
    for family, budget, hmap in cases:
        action = sampler.sample_action(
            budget=budget,
            family=family,
            batch_size=2,
            heuristic_map=hmap,
            roll=5,
        )
        assert tuple(action.selected_lines.shape) == (2, 112)
        assert tuple(action.pixel_mask.shape) == (2, 1, 112, 112)
        assert action.pixel_mask.device.type == "cpu"
        assert action.pixel_mask.dtype == torch.float32
        assert action.counts.tolist() == [budget, budget]
        assert torch.all(action.selected_lines.sum(dim=1) == budget)


def test_x_prior_mixture_outputs_and_cached_dpf_interface():
    target = _target()
    prev = target * 0.5
    prev_prev = target * -0.25
    obs = torch.zeros_like(target)
    mask = torch.zeros_like(target)
    mask[:, :, :, ::4] = 1.0
    obs = mask * target

    for kind in [PRIOR_COPY_LAST, PRIOR_EMA, PRIOR_DEGRADED_GT, PRIOR_ZERO_FILL, PRIOR_WEAK_INTERPOLATION]:
        prior = PixelPriorMixture.make(
            kind=kind,
            target=target,
            prev_x_final=prev,
            prev_prev_x_final=prev_prev,
            obs=obs,
            mask=mask,
        )
        assert tuple(prior.shape) == tuple(target.shape)
        assert torch.isfinite(prior).all()

    cached = torch.randn_like(target)
    assert torch.equal(
        PixelPriorMixture.make(kind=PRIOR_CACHED_DPF, target=target, cached_dpf_prior=cached),
        cached,
    )
    try:
        PixelPriorMixture.make(kind=PRIOR_CACHED_DPF, target=target)
    except NotImplementedError as exc:
        assert "cached_dpf_prior" in str(exc)
    else:
        raise AssertionError("cached_dpf without external tensor must fail loudly")


def test_stage1_pbu_batch_channels_budget_embedding_and_projection_inputs():
    target = _target()
    prev = target * 0.25
    sampler = VariableMaskBudgetSampler(image_shape=(32, 32), n_lines=32, seed=19)
    batch = make_stage1_pbu_batch(
        target=target,
        sampler=sampler,
        budget=14,
        mask_family=MASK_RANDOM,
        prior_kind=PRIOR_COPY_LAST,
        prev_x_final=prev,
    )

    assert tuple(batch.x_prior.shape) == tuple(target.shape)
    assert tuple(batch.u_prior.shape) == tuple(target.shape)
    assert tuple(batch.obs.shape) == tuple(target.shape)
    assert tuple(batch.mask.shape) == tuple(target.shape)
    assert tuple(batch.residual_obs.shape) == tuple(target.shape)
    assert tuple(batch.x_dc.shape) == tuple(target.shape)
    assert tuple(batch.selected_lines.shape) == (2, 32)
    assert tuple(batch.line_history.shape) == (2, 32)
    assert batch.selected_lines.sum(dim=1).tolist() == [14, 14]
    assert torch.equal(batch.residual_obs, batch.mask * (batch.obs - batch.x_prior))
    assert torch.equal(batch.x_dc[batch.mask.bool()], batch.obs[batch.mask.bool()])
    assert torch.equal(batch.u_prior, torch.abs(prev - batch.x_prior))
    assert batch.prior_is_training_only is False
    assert torch.isfinite(batch.budget_embedding(n_lines=32)).all()
    assert tuple(batch.budget_embedding(n_lines=32).shape) == (2, 4, 32, 32)


def test_stage1_pbu_wrapper_smoke_required_budgets_no_nan_and_projection_exactness():
    target = _target()
    prev = torch.zeros_like(target)
    sampler = VariableMaskBudgetSampler(image_shape=(32, 32), n_lines=32, seed=23)
    model = Stage1PBUWrapper(
        n_lines=32,
        base_channels=8,
        channel_mult=(1, 2),
        num_res_blocks=1,
        groupnorm_groups=4,
        delta_clip_scale=0.05,
    )

    for budget in STAGE1_BUDGETS:
        batch = make_stage1_pbu_batch(
            target=target,
            sampler=sampler,
            budget=budget,
            mask_family=MASK_RANDOM,
            prior_kind=PRIOR_DEGRADED_GT,
            prev_x_final=prev,
        )
        output = model(batch)
        assert batch.prior_kind in TRAINING_ONLY_PRIORS
        assert batch.prior_is_training_only is True
        assert tuple(output.x_raw.shape) == tuple(target.shape)
        assert tuple(output.x_final.shape) == tuple(target.shape)
        assert tuple(output.u_post.shape) == tuple(target.shape)
        for value in [output.x_raw, output.x_final, output.u_post, output.delta_x, output.logvar]:
            assert torch.isfinite(value).all()
        assert output.observed_consistency_l1.item() <= 1.0e-6
        assert torch.equal(output.x_final[batch.mask.bool()], batch.obs[batch.mask.bool()])
        if budget == 0:
            assert torch.count_nonzero(batch.mask).item() == 0


def test_stage1_pbu_wrapper_model_ablation_switches_are_explicit_and_finite():
    target = _target()
    sampler = VariableMaskBudgetSampler(image_shape=(32, 32), n_lines=32, seed=29)
    model = Stage1PBUWrapper(
        n_lines=32,
        base_channels=8,
        channel_mult=(1, 2),
        num_res_blocks=1,
        groupnorm_groups=4,
        delta_clip_scale=0.05,
    )
    batch = make_stage1_pbu_batch(
        target=target,
        sampler=sampler,
        budget=14,
        mask_family=MASK_RANDOM,
        prior_kind=PRIOR_COPY_LAST,
        prev_x_final=torch.zeros_like(target),
    )

    full = model(batch)
    assert full.observed_consistency_l1.item() <= 1.0e-6
    assert torch.equal(full.x_final[batch.mask.bool()], batch.obs[batch.mask.bool()])
    assert torch.count_nonzero(full.u_post[batch.mask.bool()]).item() == 0

    for ablation in [MODEL_ABLATION_NO_RESIDUAL, MODEL_ABLATION_NO_BUDGET_TOKEN]:
        output = model(batch, ablation=ablation)
        assert torch.isfinite(output.x_final).all()
        assert torch.isfinite(output.u_post).all()
        assert output.observed_consistency_l1.item() <= 1.0e-6

    no_projection = model(batch, ablation=MODEL_ABLATION_NO_HARD_PROJECTION)
    assert torch.equal(no_projection.x_final, no_projection.x_raw)
    assert no_projection.observed_consistency_l1.item() > 1.0e-8
    assert torch.count_nonzero(no_projection.u_post[batch.mask.bool()]).item() > 0

    try:
        model(batch, ablation=Stage1PBUAblation(name="full", disable_hard_projection=True))
    except ValueError as exc:
        assert "supported named smoke control" in str(exc)
    else:
        raise AssertionError("custom ablation flag combinations must fail loudly")


def test_stage1_shuffled_and_wrong_observation_sanity_interfaces_run():
    target = _target()
    sampler = VariableMaskBudgetSampler(image_shape=(32, 32), n_lines=32, seed=31)
    model = Stage1PBUWrapper(
        n_lines=32,
        base_channels=8,
        channel_mult=(1, 2),
        num_res_blocks=1,
        groupnorm_groups=4,
        delta_clip_scale=0.05,
    )
    batch = make_stage1_pbu_batch(
        target=target,
        sampler=sampler,
        budget=14,
        mask_family=MASK_ROLLED_EQUISPACED,
        prior_kind=PRIOR_WEAK_INTERPOLATION,
        prev_x_final=torch.zeros_like(target),
    )
    for variant in [batch, make_shuffled_observation_batch(batch), make_wrong_mask_batch(batch)]:
        output = model(variant)
        assert torch.isfinite(output.x_final).all()
        assert torch.isfinite(output.u_post).all()
        assert output.observed_consistency_l1.item() <= 1.0e-6
        assert tuple(variant.residual_obs.shape) == tuple(target.shape)
    wrong = make_wrong_mask_batch(batch)
    assert not torch.equal(wrong.mask, batch.mask)
    assert torch.equal(wrong.obs, batch.obs)


def test_stage1_wrapper_accepts_and_returns_pixel_belief_state_contract():
    target = _target()
    sampler = VariableMaskBudgetSampler(image_shape=(32, 32), n_lines=32, seed=37)
    batch = make_stage1_pbu_batch(
        target=target,
        sampler=sampler,
        budget=14,
        mask_family=MASK_CLUSTERED_LOCAL,
        prior_kind=PRIOR_COPY_LAST,
        prev_x_final=torch.zeros_like(target),
    )
    state = PixelBeliefState(
        x_final=batch.prev_x_final,
        u_post=batch.prev_u_post,
        x_prior=batch.x_prior,
        u_prior=batch.u_prior,
        mask=batch.mask,
        obs=batch.obs,
        selected_lines=batch.selected_lines,
        line_history=batch.line_history,
        budget=batch.budget,
    )
    model = Stage1PBUWrapper(
        n_lines=32,
        base_channels=8,
        channel_mult=(1, 2),
        num_res_blocks=1,
        groupnorm_groups=4,
        delta_clip_scale=0.05,
    )
    result = model.forward_state(state)

    assert isinstance(result.state, PixelBeliefState)
    assert tuple(result.state.x_final.shape) == tuple(target.shape)
    assert tuple(result.state.u_post.shape) == tuple(target.shape)
    assert torch.equal(result.state.x_final[batch.mask.bool()], batch.obs[batch.mask.bool()])
    assert result.aux.observed_consistency_l1.item() <= 1.0e-6
