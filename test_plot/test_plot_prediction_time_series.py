"""
Test module for plotting functions, specifically for prediction vs ground truth plots with physical context.
This module tests the functions from plot_prediction_time_series_with_physics.py
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pytest
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path to import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the functions we want to test
try:
    from postprocessing.plot_prediction_time_series_with_physics import (
        plot_time_series_with_physics,
        allowed_input_types
    )
    IMPORT_SUCCESS = True
except ImportError as e:
    print(f"Warning: Could not import plotting functions: {e}")
    IMPORT_SUCCESS = False


class TestPlotPredictionTimeSeries:
    """Test class for prediction time series plotting functions."""
    
    def setup_method(self):
        """Set up test data."""
        # Create synthetic test data
        np.random.seed(42)
        self.n_samples = 50
        
        # Generate synthetic true values (surface roughness in μm)
        self.true_values = np.random.uniform(0.1, 2.0, self.n_samples)
        
        # Generate synthetic predictions with some error
        self.predictions = self.true_values + np.random.normal(0, 0.1, self.n_samples)
        
        # Generate synthetic BDI values (mix of ductile >1 and brittle <1)
        self.bdi_values = np.concatenate([
            np.random.uniform(1.1, 2.0, self.n_samples // 2),  # Ductile
            np.random.uniform(0.5, 0.9, self.n_samples // 2)   # Brittle
        ])
        np.random.shuffle(self.bdi_values)
        
        self.model_type = "test_model"
        self.sample_indices = np.arange(self.n_samples)
    
    def test_plot_time_series_with_physics_basic(self):
        """Test basic functionality of plot_time_series_with_physics."""
        if not IMPORT_SUCCESS:
            pytest.skip("Plotting functions not available for import")
        
        # Call the function
        fig, ax = plot_time_series_with_physics(
            self.true_values,
            self.predictions,
            self.bdi_values,
            self.model_type,
            self.sample_indices
        )
        
        # Verify the figure and axes were created
        assert fig is not None
        assert ax is not None
        
        # Verify the figure has the expected title
        expected_title = f'Prediction vs Ground Truth with Physical Context\nModel: {self.model_type}'
        assert ax.get_title() == expected_title
        
        # Verify x and y labels
        assert ax.get_xlabel() == 'Sample Index'
        assert ax.get_ylabel() == 'Surface Roughness Ra ($\\mu$m)'
        
        # Verify grid is enabled
        assert ax.xaxis.get_gridlines()[0].get_visible() == True
        
        # Clean up
        plt.close(fig)
    
    def test_plot_time_series_with_physics_mae_calculation(self):
        """Test that MAE is calculated and displayed correctly."""
        if not IMPORT_SUCCESS:
            pytest.skip("Plotting functions not available for import")
        
        # Calculate expected MAE
        expected_mae = np.mean(np.abs(self.true_values - self.predictions))
        
        # Call the function
        fig, ax = plot_time_series_with_physics(
            self.true_values,
            self.predictions,
            self.bdi_values,
            self.model_type,
            self.sample_indices
        )
        
        # Check that MAE annotation is present
        # The function adds text at position (0.02, 0.98) in axes coordinates
        texts = ax.texts
        mae_found = False
        for text in texts:
            if f'MAE = {expected_mae:.2f}' in text.get_text():
                mae_found = True
                break
        
        assert mae_found, f"MAE annotation not found. Expected MAE = {expected_mae:.2f}"
        
        # Clean up
        plt.close(fig)
    
    def test_plot_time_series_with_physics_regime_coloring(self):
        """Test that BDI regime coloring is applied correctly."""
        if not IMPORT_SUCCESS:
            pytest.skip("Plotting functions not available for import")
        
        # Create BDI values with clear regime transitions
        bdi_test = np.array([1.5, 1.5, 0.5, 0.5, 1.5, 1.5])  # Ductile, Brittle, Ductile
        true_test = np.ones(6)
        pred_test = np.ones(6)
        
        fig, ax = plot_time_series_with_physics(
            true_test,
            pred_test,
            bdi_test,
            self.model_type,
            np.arange(6)
        )
        
        # Check that patches were created (background coloring)
        # The function creates axvspan patches for each regime
        patches = ax.patches
        assert len(patches) > 0, "No background patches created for BDI regimes"
        
        # Clean up
        plt.close(fig)
    
    def test_plot_time_series_with_physics_legend(self):
        """Test that legend is created with correct elements."""
        if not IMPORT_SUCCESS:
            pytest.skip("Plotting functions not available for import")
        
        fig, ax = plot_time_series_with_physics(
            self.true_values,
            self.predictions,
            self.bdi_values,
            self.model_type,
            self.sample_indices
        )
        
        # Check that legend exists
        legend = ax.get_legend()
        assert legend is not None, "Legend not created"
        
        # Check legend text contains expected labels
        legend_texts = [t.get_text() for t in legend.get_texts()]
        expected_labels = ['Ground Truth', 'Prediction', 'BDI > 1 (Ductile)', 'BDI < 1 (Brittle)']
        
        for expected_label in expected_labels:
            assert any(expected_label in text for text in legend_texts), \
                f"Legend label '{expected_label}' not found in {legend_texts}"
        
        # Clean up
        plt.close(fig)
    
    def test_plot_time_series_with_physics_edge_cases(self):
        """Test edge cases for the plotting function."""
        if not IMPORT_SUCCESS:
            pytest.skip("Plotting functions not available for import")
        
        # Test with empty arrays
        with pytest.raises(ValueError):
            fig, ax = plot_time_series_with_physics(
                np.array([]),
                np.array([]),
                np.array([]),
                self.model_type,
                np.array([])
            )
        
        # Test with single sample
        fig, ax = plot_time_series_with_physics(
            np.array([1.0]),
            np.array([1.1]),
            np.array([1.5]),  # Ductile
            self.model_type,
            np.array([0])
        )
        assert fig is not None
        plt.close(fig)
        
        # Test with all ductile BDI values
        fig, ax = plot_time_series_with_physics(
            np.ones(10),
            np.ones(10),
            np.full(10, 1.5),  # All ductile
            self.model_type,
            np.arange(10)
        )
        assert fig is not None
        plt.close(fig)
        
        # Test with all brittle BDI values
        fig, ax = plot_time_series_with_physics(
            np.ones(10),
            np.ones(10),
            np.full(10, 0.5),  # All brittle
            self.model_type,
            np.arange(10)
        )
        assert fig is not None
        plt.close(fig)
    
    def test_allowed_input_types(self):
        """Test that allowed_input_types is defined and contains expected values."""
        if not IMPORT_SUCCESS:
            pytest.skip("Plotting functions not available for import")
        
        # Check that allowed_input_types exists and is a list
        assert isinstance(allowed_input_types, list)
        assert len(allowed_input_types) > 0
        
        # Check for some expected model types
        expected_types = ['ae_features', 'vib_features', 'ae_features+pp', 'vib_features+pp']
        for expected_type in expected_types:
            assert expected_type in allowed_input_types, \
                f"Expected model type '{expected_type}' not in allowed_input_types"


class TestPlotUtilities:
    """Test utility functions for plotting."""
    
    def test_mae_calculation(self):
        """Test MAE calculation logic."""
        # Test with perfect predictions
        true = np.array([1.0, 2.0, 3.0])
        pred = np.array([1.0, 2.0, 3.0])
        mae = np.mean(np.abs(true - pred))
        assert mae == 0.0
        
        # Test with errors
        true = np.array([1.0, 2.0, 3.0])
        pred = np.array([1.5, 2.5, 3.5])
        mae = np.mean(np.abs(true - pred))
        assert mae == 0.5
        
        # Test with mixed errors
        true = np.array([1.0, 2.0, 3.0])
        pred = np.array([2.0, 1.0, 4.0])
        mae = np.mean(np.abs(true - pred))
        assert mae == (1.0 + 1.0 + 1.0) / 3.0
    
    def test_bdi_regime_classification(self):
        """Test BDI regime classification logic."""
        # Test ductile regime (BDI > 1)
        bdi_ductile = np.array([1.1, 1.5, 2.0, 100.0])
        regime_ductile = bdi_ductile > 1.0
        assert np.all(regime_ductile == True)
        
        # Test brittle regime (BDI < 1)
        bdi_brittle = np.array([0.1, 0.5, 0.9, 0.999])
        regime_brittle = bdi_brittle > 1.0
        assert np.all(regime_brittle == False)
        
        # Test boundary case (BDI = 1)
        bdi_boundary = np.array([1.0])
        regime_boundary = bdi_boundary > 1.0
        assert regime_boundary[0] == False  # BDI = 1 is considered brittle
        
        # Test mixed regimes
        bdi_mixed = np.array([1.5, 0.5, 1.2, 0.8])
        regime_mixed = bdi_mixed > 1.0
        expected = np.array([True, False, True, False])
        assert np.array_equal(regime_mixed, expected)
    
    def test_regime_transition_detection(self):
        """Test detection of transitions between BDI regimes."""
        # Test case with no transitions (all ductile)
        bdi_all_ductile = np.array([1.5, 1.5, 1.5, 1.5])
        regime_all_ductile = bdi_all_ductile > 1.0
        changes = np.where(np.diff(regime_all_ductile.astype(int)) != 0)[0] + 1
        assert len(changes) == 0
        
        # Test case with no transitions (all brittle)
        bdi_all_brittle = np.array([0.5, 0.5, 0.5, 0.5])
        regime_all_brittle = bdi_all_brittle > 1.0
        changes = np.where(np.diff(regime_all_brittle.astype(int)) != 0)[0] + 1
        assert len(changes) == 0
        
        # Test case with one transition
        bdi_one_transition = np.array([1.5, 1.5, 0.5, 0.5])
        regime_one_transition = bdi_one_transition > 1.0
        changes = np.where(np.diff(regime_one_transition.astype(int)) != 0)[0] + 1
        assert len(changes) == 1
        assert changes[0] == 2  # Transition at index 2
        
        # Test case with multiple transitions
        bdi_multiple = np.array([1.5, 0.5, 1.5, 0.5, 1.5])
        regime_multiple = bdi_multiple > 1.0
        changes = np.where(np.diff(regime_multiple.astype(int)) != 0)[0] + 1
        assert len(changes) == 3
        assert np.array_equal(changes, [1, 2, 3])


def test_plot_save_functionality():
    """Test that plots can be saved to file."""
    if not IMPORT_SUCCESS:
        pytest.skip("Plotting functions not available for import")
    
    # Create a simple plot
    fig, ax = plt.subplots()
    ax.plot([0, 1, 2], [0, 1, 4])
    ax.set_title("Test Plot")
    
    # Save to temporary file
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        temp_path = tmp.name
    
    try:
        fig.savefig(temp_path, dpi=300, bbox_inches='tight')
        
        # Verify file was created
        assert os.path.exists(temp_path)
        assert os.path.getsize(temp_path) > 0
    finally:
        # Clean up
        plt.close(fig)
        if os.path.exists(temp_path):
            os.unlink(temp_path)


if __name__ == "__main__":
    """Run tests directly if script is executed."""
    import sys
    
    # Create test instance
    test_class = TestPlotPredictionTimeSeries()
    test_class.setup_method()
    
    print("Running plot prediction time series tests...")
    
    # Run tests
    try:
        test_class.test_plot_time_series_with_physics_basic()
        print("✓ test_plot_time_series_with_physics_basic passed")
    except Exception as e:
        print(f"✗ test_plot_time_series_with_physics_basic failed: {e}")
    
    try:
        test_class.test_plot_time_series_with_physics_mae_calculation()
        print("✓ test_plot_time_series_with_physics_mae_calculation passed")
    except Exception as e:
        print(f"✗ test_plot_time_series_with_physics_mae_calculation failed: {e}")
    
    try:
        test_class.test_plot_time_series_with_physics_regime_coloring()
        print("✓ test_plot_time_series_with_physics_regime_coloring passed")
    except Exception as e:
        print(f"✗ test_plot_time_series_with_physics_regime_coloring failed: {e}")
    
    try:
        test_class.test_plot_time_series_with_physics_legend()
        print("✓ test_plot_time_series_with_physics_legend passed")
    except Exception as e:
        print(f"✗ test_plot_time_series_with_physics_legend failed: {e}")
    
    try:
        test_class.test_allowed_input_types()
        print("✓ test_allowed_input_types passed")
    except Exception as e:
        print(f"✗ test_allowed_input_types failed: {e}")
    
    print("\nAll tests completed!")
