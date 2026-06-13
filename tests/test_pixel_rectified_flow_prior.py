from __future__ import annotations

from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
ZEA_TOOL_ROOT = ROOT.parent / "zea_tool"
for path in [ROOT, ZEA_TOOL_ROOT]:
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ulsa.pixel_rectified_flow_prior import StandalonePixelRectifiedFlowPrior


def test_standalone_pixel_rectified_flow_shapes_and_finite() -> None:
    torch.manual_seed(7)
    model = StandalonePixelRectifiedFlowPrior(
        history_size=3,
        base_channels=8,
        channel_mult=(1,),
        num_res_blocks=1,
        groupnorm_groups=4,
    )
    history_x = torch.randn(2, 3, 1, 32, 32).clamp(-1.0, 1.0)
    history_u = torch.zeros_like(history_x)
    history_mask = torch.ones_like(history_x)
    x_start = history_x[:, -1] + 0.05 * torch.randn_like(history_x[:, -1])
    target = torch.randn_like(x_start).clamp(-1.0, 1.0)
    t = torch.rand(2, 1, 1, 1)
    x_t = (1.0 - t) * x_start + t * target

    out = model(
        x_t=x_t,
        t=t,
        x_start=x_start,
        history_x=history_x,
        history_u=history_u,
        history_mask=history_mask,
        history_selected=history_mask,
    )
    assert tuple(out.mu_prior.shape) == (2, 1, 32, 32)
    assert tuple(out.u_prior.shape) == (2, 1, 32, 32)
    assert torch.isfinite(out.mu_prior).all()
    assert torch.isfinite(out.u_prior).all()
    assert torch.isfinite(out.velocity).all()

    integrated = model.integrate(
        x_start=x_start,
        steps=2,
        history_x=history_x,
        history_u=history_u,
        history_mask=history_mask,
        history_selected=history_mask,
    )
    assert tuple(integrated.mu_prior.shape) == (2, 1, 32, 32)
    assert torch.isfinite(integrated.mu_prior).all()
    assert torch.isfinite(integrated.u_prior).all()


if __name__ == "__main__":
    test_standalone_pixel_rectified_flow_shapes_and_finite()
