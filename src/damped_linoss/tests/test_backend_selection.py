import subprocess
import sys

import pytest
import torch

from damped_linoss.models.TorchLinOSS import DampedLayer, LinOSS


def test_package_imports_outside_repo_root():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from damped_linoss.models.TorchLinOSS import LinOSS; print(LinOSS.__name__)",
        ],
        cwd="/tmp",
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "LinOSS"


@pytest.mark.parametrize("layer_name", ["IM", "IMEX", "Damped"])
def test_use_triton_propagates_to_layers(layer_name):
    model = LinOSS(
        layer_name=layer_name,
        input_dim=2,
        state_dim=4,
        hidden_dim=3,
        output_dim=2,
        num_blocks=2,
        use_triton=False,
    )
    assert [block.layer.use_triton for block in model.blocks] == [False, False]


def test_forced_triton_requires_cuda_tensor():
    layer = LinOSS(
        layer_name="IM",
        input_dim=2,
        state_dim=4,
        hidden_dim=3,
        output_dim=2,
        num_blocks=1,
        use_triton=True,
    )
    with pytest.raises(RuntimeError, match="input tensors are not on CUDA"):
        layer(torch.randn(5, 2))


def test_damped_layer_clamps_a_diag_before_recurrence(monkeypatch):
    layer = DampedLayer(3, 2, 0.9, 1.0, 3.14, use_triton=False)
    with torch.no_grad():
        layer.steps.fill_(0.0)
        layer.G_diag.fill_(0.0)
        layer.A_diag.copy_(torch.tensor([-10.0, 4.0, 100.0]))

    captured = {}

    def fake_recurrence(A_diag, G_diag, B, input_sequence, step):
        captured["A_diag"] = A_diag.detach().clone()
        return input_sequence.new_zeros(input_sequence.shape[0], 3, 2)

    monkeypatch.setattr(layer, "_recurrence", fake_recurrence)
    layer(torch.randn(4, 2))

    assert torch.allclose(captured["A_diag"], torch.tensor([0.0, 4.0, 16.0]))
