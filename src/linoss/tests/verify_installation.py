#!/usr/bin/env python
"""
Simple script to verify the installation of damped-linoss.
Run this after installation to check which backend is being used.
"""

import torch
import damped_linoss


def main():
    print("=" * 60)
    print("Damped-LinOSS Installation Verification")
    print("=" * 60)
    print(f"Package version: {damped_linoss.__version__}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU device: {torch.cuda.get_device_name(0)}")

    print(f"Triton available: {damped_linoss.TRITON_AVAILABLE}")
    print()

    # Determine which backend will be used
    if damped_linoss.TRITON_AVAILABLE and torch.cuda.is_available():
        backend = "Triton (optimized CUDA kernels)"
    else:
        backend = "torch.compile (pure PyTorch)"

    print(f"Active backend: {backend}")
    print()

    # Create a simple model to test
    print("Testing model creation...")
    try:
        model = damped_linoss.LinOSS(
            layer_name="Damped",
            input_dim=10,
            state_dim=32,
            hidden_dim=64,
            output_dim=5,
            num_blocks=2,
            classification=True,
            tanh_output=False,
            output_step=1,
        )
        print("✓ Model created successfully")

        # Test a forward pass
        print("Testing forward pass...")
        batch_size = 4
        seq_length = 100
        input_dim = 10

        x = torch.randn(batch_size, seq_length, input_dim)
        if torch.cuda.is_available():
            x = x.cuda()
            model = model.cuda()

        with torch.no_grad():
            output = model(x)

        print(f"✓ Forward pass successful")
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {output.shape}")

        print()
        print("=" * 60)
        print("Installation verified successfully!")
        print("=" * 60)

        return True

    except Exception as e:
        print(f"✗ Error during testing: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
