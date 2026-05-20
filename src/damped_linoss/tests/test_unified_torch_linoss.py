import pytest
import torch

from damped_linoss import LinOSSBackbone, LinOSSSequenceMixer, TRITON_AVAILABLE
from damped_linoss.models.TorchLinOSS import DampedLayer, IMEXLayer, IMLayer


def _copy_legacy_layer(legacy_layer, unified_layer):
    with torch.no_grad():
        unified_layer.steps[0].copy_(legacy_layer.steps)
        unified_layer.A_diag[0].copy_(legacy_layer.A_diag)
        unified_layer.B[0].copy_(legacy_layer.B)
        unified_layer.C[0].copy_(legacy_layer.C)
        unified_layer.D[0].copy_(legacy_layer.D)
        unified_layer.gamma_log.zero_()
        if hasattr(legacy_layer, "G_diag"):
            unified_layer.G_diag[0].copy_(legacy_layer.G_diag)
        else:
            unified_layer.G_diag.zero_()


@pytest.mark.parametrize(
    ("legacy_cls", "discretization", "damping"),
    [
        (IMLayer, "IM", False),
        (IMEXLayer, "IMEX", False),
        (DampedLayer, "IMEX", True),
    ],
)
def test_unified_layer_matches_legacy_modes(legacy_cls, discretization, damping):
    torch.manual_seed(123)
    state_dim = 8
    hidden_dim = 5
    x = torch.randn(3, 11, hidden_dim)

    legacy = legacy_cls(state_dim, hidden_dim, 0.9, 1.0, torch.pi, use_triton=False)
    unified = LinOSSSequenceMixer(
        in_features=hidden_dim,
        state_dim=state_dim,
        num_heads=1,
        discretization=discretization,
        damping=damping,
        stability="oscillatory",
        projection_eps=0.0,
        use_triton=False,
    )
    _copy_legacy_layer(legacy, unified)

    actual = unified(x)
    expected = legacy(x)

    assert torch.allclose(actual, expected, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("discretization", ["IM", "IMEX", "IMEX2", "IMEX3", "EX"])
@pytest.mark.parametrize("stability", ["oscillatory", "stable"])
def test_unified_layer_new_discretizations_are_finite_and_differentiable(
    discretization, stability
):
    torch.manual_seed(321)
    layer = LinOSSSequenceMixer(
        in_features=6,
        state_dim=10,
        num_heads=2,
        discretization=discretization,
        damping=True,
        stability=stability,
        input_normalization=True,
        use_head_gating=True,
        use_head_output_projection=True,
        use_triton=False,
    )
    x = torch.randn(4, 9, 6, requires_grad=True)

    y = layer(x)
    loss = y.square().mean()
    loss.backward()

    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    assert torch.isfinite(x.grad).all()
    for parameter in layer.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_unified_layer_validates_multi_head_partitions():
    with pytest.raises(ValueError, match="in_features=7 must be divisible by num_heads=2"):
        LinOSSSequenceMixer(in_features=7, state_dim=8, num_heads=2)

    with pytest.raises(ValueError, match="state_dim=9 must be divisible by num_heads=2"):
        LinOSSSequenceMixer(in_features=8, state_dim=9, num_heads=2)


def test_unified_layer_rejects_invalid_input_normalization():
    with pytest.raises(ValueError, match="input_normalization requires damping=True"):
        LinOSSSequenceMixer(in_features=4, state_dim=8, damping=False, input_normalization=True)


def test_unified_layer_rejects_forced_triton_on_cpu():
    layer = LinOSSSequenceMixer(in_features=4, state_dim=8, use_triton=True)
    x = torch.randn(3, 4)

    with pytest.raises(RuntimeError, match="input tensors are not on CUDA"):
        layer(x)


def test_backbone_validates_multi_head_partitions():
    with pytest.raises(ValueError, match="hidden_dim=7 must be divisible by num_heads=2"):
        LinOSSBackbone(hidden_dim=7, state_dim=8, num_heads=2)

    with pytest.raises(ValueError, match="state_dim=9 must be divisible by num_heads=2"):
        LinOSSBackbone(hidden_dim=8, state_dim=9, num_heads=2)


def test_single_head_merge_flags_preserve_plain_path():
    torch.manual_seed(654)
    keyed = LinOSSSequenceMixer(
        in_features=6,
        state_dim=8,
        num_heads=1,
        use_head_gating=True,
        use_head_output_projection=True,
        use_triton=False,
    )
    torch.manual_seed(654)
    plain = LinOSSSequenceMixer(
        in_features=6,
        state_dim=8,
        num_heads=1,
        use_head_gating=False,
        use_head_output_projection=False,
        use_triton=False,
    )
    x = torch.randn(5, 6)

    assert keyed.head_gate is None
    assert keyed.head_output_projection is None
    assert torch.allclose(keyed(x), plain(x), atol=0.0, rtol=0.0)


@pytest.mark.parametrize("discretization", ["IM", "IMEX", "IMEX2", "IMEX3", "EX"])
def test_rt_initialization_executes(discretization):
    torch.manual_seed(987)
    layer = LinOSSSequenceMixer(
        in_features=4,
        state_dim=6,
        discretization=discretization,
        initialization="RT",
        damping=True,
        use_triton=False,
    )
    x = torch.randn(2, 5, 4)

    y = layer(x)

    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_backbone_matches_discretax_style_shape_contract():
    torch.manual_seed(456)
    model = LinOSSBackbone(
        hidden_dim=8,
        num_blocks=2,
        state_dim=12,
        num_heads=2,
        use_head_gating=True,
        use_head_output_projection=True,
        discretization="IMEX3",
        drop_rate=0.0,
        use_triton=False,
    )
    x = torch.randn(3, 7, 8)

    y = model(x)

    assert y.shape == x.shape
    assert torch.isfinite(y).all()


@pytest.mark.skipif(
    not torch.cuda.is_available() or not TRITON_AVAILABLE,
    reason="Triton parity requires CUDA and Triton",
)
@pytest.mark.parametrize("discretization", ["IM", "IMEX", "IMEX2", "IMEX3", "EX"])
def test_unified_layer_triton_matches_native_multihead_projection(discretization):
    torch.manual_seed(789)
    native = LinOSSSequenceMixer(
        in_features=8,
        state_dim=16,
        num_heads=4,
        discretization=discretization,
        damping=True,
        use_head_output_projection=True,
        use_triton=False,
    ).cuda()
    triton = LinOSSSequenceMixer(
        in_features=8,
        state_dim=16,
        num_heads=4,
        discretization=discretization,
        damping=True,
        use_head_output_projection=True,
        use_triton=True,
    ).cuda()
    triton.load_state_dict(native.state_dict())
    assert native.head_gate is None
    assert triton.head_gate is None
    assert native.head_output_projection is not None
    assert triton.head_output_projection is not None

    x = torch.randn(3, 33, 8, device="cuda", requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)

    y_native = native(x_ref)
    y_triton = triton(x)
    y_native.square().mean().backward()
    y_triton.square().mean().backward()

    assert torch.allclose(y_triton, y_native, atol=5e-4, rtol=5e-4)
    assert torch.allclose(x.grad, x_ref.grad, atol=2e-2, rtol=2e-2)
