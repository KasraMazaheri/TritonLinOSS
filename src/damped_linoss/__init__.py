"""
Damped Linear Oscillatory State-Space Models (D-LinOSS)

A PyTorch implementation of Damped Linear Oscillatory State-Space Models
with automatic backend selection between Triton (CUDA) and torch.compile.
"""

__version__ = "0.1.0"

__all__ = [
    "LinOSS",
    "LinOSSBlock",
    "IMLayer",
    "IMEXLayer",
    "DampedLayer",
    "LinOSSSequenceMixer",
    "LinOSSBackboneBlock",
    "LinOSSBackbone",
    "TRITON_AVAILABLE",
]


def __getattr__(name):
    if name == "TRITON_AVAILABLE":
        try:
            from .parallel_scan.torch_interface import TRITON_AVAILABLE
        except ImportError:
            TRITON_AVAILABLE = False
        return TRITON_AVAILABLE

    if name in {
        "LinOSS",
        "LinOSSBlock",
        "IMLayer",
        "IMEXLayer",
        "DampedLayer",
        "LinOSSSequenceMixer",
        "LinOSSBackboneBlock",
        "LinOSSBackbone",
    }:
        from .models import TorchLinOSS

        return getattr(TorchLinOSS, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
