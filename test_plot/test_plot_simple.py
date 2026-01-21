"""
Simple test module for plotting functions that doesn't require importing the actual plotting module.
This tests the core logic used in prediction vs ground truth plots.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import tempfile


def test_mae_calculation():
    """Test MAE calculation logic."""
    print("Testing MAE calculation...")
    
    # Test with perfect predictions
    true = np.array([1.0, 2.0, 3.0])
    pred = np.array([1.0, 2.0, 3.0])
    mae = np.mean(np.abs(true - pred))
    assert mae == 0.0, f"Expected MAE 0.0, got {mae}"
    print("✓ Perfect predictions MAE test passed")
    
    # Test with errors
    true = np.array([1.0, 2.0, 3.0])
    pred = np.array([1.5, 2.5, 3.5])
    mae = np.mean(np.abs(true - pred))
    assert mae == 0.5, f"Expected MAE 0.5, got {mae}"
    print("✓ Error MAE test passed")
    
    # Test with mixed errors
    true = np.array([1.0, 2.0, 3.0])
    pred = np.array([2.0, 1.0, 4.0])
    mae = np.mean(np.abs(true - pred))
    expected_mae = (1.0 + 1.0 + 1.0) / 3.0
    assert abs(mae - expected_mae) < 1e-10, f"Expected MAE {expected_mae}, got {mae}"
    print("✓ Mixed errors MAE test passed")
    
    return True


def test_bdi_regime_classification():
    """Test BDI regime classification logic."""
    print("\nTesting BDI regime classification...")
    
    # Test ductile regime (BDI > 1)
    bdi_ductile = np.array([1.1, 1.5, 2.0, 100.0])
    regime_ductile = bdi_ductile > 1.0
    assert np.all(regime_ductile == True), "Ductile regime classification failed"
    print("✓ Ductile regime classification test passed")
    
    # Test brittle regime (BDI < 1)
    bdi_brittle = np.array([0.1, 0.5, 0.9, 0.999])
    regime_brittle = bdi_brittle > 1.0
    assert np.all(regime_brittle == False), "Brittle regime classification failed"
    print("✓ Brittle regime classification test passed")
    
    # Test boundary case (BDI = 1)
    bdi_boundary = np.array([1.0])
    regime_boundary = bdi_boundary > 1.0
    assert regime_boundary[0] == False, "BDI = 1 should be classified as brittle"
    print("✓ Boundary case test passed")
    
    # Test mixed regimes
    bdi_mixed = np.array([1.5, 0.5, 1.2, 0.8])
    regime_mixed = bdi_mixed > 1.0
    expected = np.array([True, False, True, False])
    assert np.array_equal(regime_mixed, expected), "Mixed regime classification failed"
    print("✓ Mixed regime classification test passed")
    
    return True


def test_regime_transition_detection():
    """Test detection of transitions between BDI regimes."""
    print("\nTesting regime transition detection...")
    
    # Test case with no transitions (all ductile)
    bdi_all_ductile = np.array([1.5, 1.5, 1.5, 1.5])
    regime_all_ductile = bdi_all_ductile > 1.0
    changes = np.where(np.diff(regime_all_ductile.astype(int)) != 0)[0] + 1
    assert len(changes) == 0, f"Expected 0 transitions, got {len(changes)}"
    print("✓ No transitions (all ductile) test passed")
    
    # Test case with no transitions (all brittle)
    bdi_all_brittle = np.array([0.5, 0.5, 0.5, 0.5])
    regime_all_brittle = bdi_all_brittle > 1.0
    changes = np.where(np.diff(regime_all_brittle.astype(int)) != 0)[0] + 1
    assert len(changes) == 0, f"Expected 0 transitions, got {len(changes)}"
    print("✓ No transitions (all brittle) test passed")
    
    # Test case with one transition
    bdi_one_transition = np.array([1.5, 1.5, 0.5, 0.5])
    regime_one_transition = bdi_one_transition > 1.0
    changes = np.where(np.diff(regime_one_transition.astype(int)) != 0)[0] + 1
    assert len(changes) == 1, f"Expected 1 transition, got {len(changes)}"
    assert changes[0] == 2, f"Expected transition at index 2, got {changes[0]}"
    print("✓ One transition test passed")
    
    # Test case with multiple transitions
    # Array: [True, False, True, False, True] has transitions at indices 1, 2, 3, 4
    # But we want to test 3 transitions, so use [True, False, True, False]
    bdi_multiple = np.array([1.5, 0.5, 1.5, 0.5])
    regime_multiple = bdi_multiple > 1.0  # [True, False, True, False]
    changes = np.where(np.diff(regime_multiple.astype(int)) != 0)[0] + 1
    # diff of [1, 0, 1, 0] -> [-1, 1, -1]
    # where finds indices 0, 2
    # +1 gives indices 1, 3
    # But we expect transitions at 1, 2, 3
    # Actually, let me think: transitions occur BETWEEN elements
    # Element 0-1: True->False transition at index 1
    # Element 1-2: False->True transition at index 2
    # Element 2-3: True->False transition at index 3
    # So we expect 3 transitions at indices 1, 2, 3
    # The issue is that diff gives -1, 1, -1 (indices 0, 1, 2)
    # where finds indices 0, 1, 2
    # +1 gives indices 1, 2, 3 ✓
    assert len(changes) == 3, f"Expected 3 transitions, got {len(changes)}"
    expected_changes = [1, 2, 3]
    assert np.array_equal(changes, expected_changes), \
        f"Expected transitions at {expected_changes}, got {changes}"
    print("✓ Multiple transitions test passed")
    
    return True


def test_plot_creation():
    """Test basic plot creation functionality."""
    print("\nTesting plot creation...")
    
    # Create a simple plot similar to what the prediction plot would look like
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create synthetic data
    n_samples = 20
    sample_indices = np.arange(n_samples)
    true_values = np.random.uniform(0.1, 2.0, n_samples)
    predictions = true_values + np.random.normal(0, 0.1, n_samples)
    bdi_values = np.concatenate([
        np.random.uniform(1.1, 2.0, n_samples // 2),
        np.random.uniform(0.5, 0.9, n_samples // 2)
    ])
    np.random.shuffle(bdi_values)
    
    # Plot ground truth and predictions
    ax.plot(sample_indices, true_values, 'o-', label='Ground Truth', 
            color='black', alpha=0.8, markersize=4, linewidth=1.5)
    ax.plot(sample_indices, predictions, 's-', label='Prediction', 
            color='red', alpha=0.8, markersize=4, linewidth=1.5)
    
    # Calculate MAE
    mae = np.mean(np.abs(true_values - predictions))
    
    # Create background colors based on BDI regime
    bdi_regime = bdi_values > 1.0
    
    # Group consecutive samples with same regime
    regime_changes = np.where(np.diff(bdi_regime.astype(int)) != 0)[0] + 1
    regime_starts = np.concatenate(([0], regime_changes))
    regime_ends = np.concatenate((regime_changes, [len(bdi_regime)]))
    
    # Color background based on BDI regime
    for start, end in zip(regime_starts, regime_ends):
        regime = bdi_regime[start]
        color = 'lightblue' if regime else 'lightcoral'
        alpha = 0.3 if regime else 0.2
        
        # Extend slightly beyond data range for visual clarity
        # Convert to integer indices
        x_start_idx = max(0, int(start-0.5))
        x_end_idx = min(len(sample_indices)-1, int(end-0.5))
        x_start = sample_indices[x_start_idx]
        x_end = sample_indices[x_end_idx]
        
        ax.axvspan(x_start, x_end, ymin=0, ymax=1, alpha=alpha, color=color)
    
    # Customize plot
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Surface Roughness Ra ($\\mu$m)')
    ax.set_title('Test Prediction vs Ground Truth with Physical Context')
    
    # Add MAE annotation
    ax.text(0.02, 0.98, f'MAE = {mae:.2f} μm', 
            transform=ax.transAxes, fontsize=12, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            verticalalignment='top')
    
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    # Verify plot elements
    assert ax.get_xlabel() == 'Sample Index'
    assert ax.get_ylabel() == 'Surface Roughness Ra ($\\mu$m)'
    assert 'Test Prediction' in ax.get_title()
    assert ax.xaxis.get_gridlines()[0].get_visible() == True
    
    # Clean up
    plt.close(fig)
    print("✓ Plot creation test passed")
    
    return True


def test_plot_save_functionality():
    """Test that plots can be saved to file."""
    print("\nTesting plot save functionality...")
    
    # Create a simple plot
    fig, ax = plt.subplots()
    ax.plot([0, 1, 2], [0, 1, 4])
    ax.set_title("Test Plot")
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        temp_path = tmp.name
    
    try:
        fig.savefig(temp_path, dpi=300, bbox_inches='tight')
        
        # Verify file was created
        assert os.path.exists(temp_path), f"File not created: {temp_path}"
        assert os.path.getsize(temp_path) > 0, f"File is empty: {temp_path}"
        
        print(f"✓ Plot saved successfully to {temp_path}")
        print(f"  File size: {os.path.getsize(temp_path)} bytes")
        
    finally:
        # Clean up
        plt.close(fig)
        if os.path.exists(temp_path):
            os.unlink(temp_path)
            print("✓ Temporary file cleaned up")
    
    return True


def run_all_tests():
    """Run all tests and report results."""
    print("=" * 60)
    print("Running Simple Plot Function Tests")
    print("=" * 60)
    
    tests = [
        ("MAE Calculation", test_mae_calculation),
        ("BDI Regime Classification", test_bdi_regime_classification),
        ("Regime Transition Detection", test_regime_transition_detection),
        ("Plot Creation", test_plot_creation),
        ("Plot Save Functionality", test_plot_save_functionality),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                print(f"\n✓ {test_name}: PASSED")
                passed += 1
            else:
                print(f"\n✗ {test_name}: FAILED")
                failed += 1
        except Exception as e:
            print(f"\n✗ {test_name}: ERROR - {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed == 0:
        print("\nAll tests passed successfully! ✓")
        return True
    else:
        print("\nSome tests failed. ✗")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
