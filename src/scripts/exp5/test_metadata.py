#!/usr/bin/env python3
"""
Quick test to verify metadata files are generated correctly.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import torch

from src.datamodelopt.tracking.hessian import HessianTracker
from src.datamodelopt.tracking.weights import WeightHistoryTracker


class MockContext:
    """Mock context for testing."""
    def __init__(self):
        self.stage_name = "test_stage"
        self.stage_idx = 0
        self.step_in_stage = 0


def test_weight_metadata():
    """Test that weight metadata is saved."""
    print("Testing weight metadata...")

    # Create tracker
    tracker = WeightHistoryTracker(every=1, filename="weights.pt")

    # Simulate stage
    ctx = MockContext()
    tracker.on_stage_start(ctx)

    # Add some fake weights
    for i in range(10):
        ctx.step_in_stage = i
        fake_weight = torch.randn(100)  # 100 params
        tracker.weights.append(fake_weight)
        tracker.steps.append(i)

    # Flush to temp dir
    save_dir = "/tmp/test_metadata"
    Path(save_dir).mkdir(exist_ok=True)
    tracker.flush(save_dir)

    # Check files exist
    weight_file = Path(save_dir) / "weights_test_stage.pt"
    info_file = Path(save_dir) / "weights_test_stage_info.txt"

    assert weight_file.exists(), f"Weight file not found: {weight_file}"
    assert info_file.exists(), f"Info file not found: {info_file}"

    # Read and verify metadata
    with open(info_file) as f:
        content = f.read()
        print("\nMetadata file content:")
        print("=" * 50)
        print(content)
        print("=" * 50)

        # Verify key info is present
        assert "test_stage" in content
        assert "n_snapshots: 10" in content
        assert "n_params: 100" in content
        assert "First step: 0" in content
        assert "Last step: 9" in content

    print("✅ Weight metadata test passed!")


def test_hessian_metadata():
    """Test that Hessian metadata is saved."""
    print("\nTesting Hessian metadata...")

    # Create tracker
    tracker = HessianTracker(every=1, filename="hessians.pt", max_params=200)

    # Simulate stage
    ctx = MockContext()
    ctx.model = None  # Not actually computing Hessians for this test
    tracker.on_stage_start(ctx)
    tracker._n_params = 50  # Mock param count

    # Add some fake Hessians
    for i in range(5):
        fake_hessian = torch.randn(50, 50)
        fake_eigenvalues = torch.randn(50)
        tracker.hessians.append(fake_hessian)
        tracker.eigenvalues.append(fake_eigenvalues)
        tracker.steps.append(i)

    # Flush to temp dir
    save_dir = "/tmp/test_metadata"
    Path(save_dir).mkdir(exist_ok=True)
    tracker.flush(save_dir)

    # Check files exist
    hessian_file = Path(save_dir) / "hessians_test_stage.pt"
    info_file = Path(save_dir) / "hessians_test_stage_info.txt"

    assert hessian_file.exists(), f"Hessian file not found: {hessian_file}"
    assert info_file.exists(), f"Info file not found: {info_file}"

    # Read and verify metadata
    with open(info_file) as f:
        content = f.read()
        print("\nMetadata file content:")
        print("=" * 50)
        print(content)
        print("=" * 50)

        # Verify key info is present
        assert "test_stage" in content
        assert "Model parameters: 50" in content
        assert "Number of Hessians: 5" in content
        assert "Matrix size: 50 x 50" in content
        assert "Eigenvalues computed: 5" in content

    print("✅ Hessian metadata test passed!")


if __name__ == "__main__":
    test_weight_metadata()
    test_hessian_metadata()
    print("\n✅ All metadata tests passed!")
