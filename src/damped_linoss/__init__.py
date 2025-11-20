"""
Damped Linear Oscillatory State-Space Models (D-LinOSS)

A PyTorch implementation of Damped Linear Oscillatory State-Space Models
with automatic backend selection between Triton (CUDA) and torch.compile.
"""

__version__ = "0.1.0"

# Check what backends are available
try:
    from .parallel_scan.torch_interface import TRITON_AVAILABLE
except ImportError:
    TRITON_AVAILABLE = False

# Export main models
from .models.TorchLinOSS import LinOSS, LinOSSBlock, IMLayer, IMEXLayer, DampedLayer

__all__ = [
    "LinOSS",
    "LinOSSBlock",
    "IMLayer",
    "IMEXLayer",
    "DampedLayer",
    "TRITON_AVAILABLE",
]
