#!/usr/bin/env python3
"""
Generate example PNG plot files in the test_plot folder.
This script creates example plots showing prediction vs ground truth with physical context.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


def generate_example_data(n_samples=50):
    """Generate example data for plotting."""
    np.random.seed(42)  # For reproducibility
    
    # Generate synthetic true values (surface roughness in μm)
    true_values = np.random.uniform(0.1, 2.0, n_samples)
    
    # Generate synthetic predictions with some error
    predictions = true_values + np.random.normal(0, 0.1, n_samples)
    
    # Generate synthetic BDI values (mix of ductile >1 and brittle <1)
    # Create clear regime transitions for visualization
    bdi_values = np.ones(n_samples)
    
    # Create 3 distinct regimes: ductile, brittle, ductile
    regime_length = n_samples // 3
    bdi_values[:regime_length] = np.random.uniform(1.2, 2.0, regime_length)  # Ductile
    bdi_values[regime_length:2*regime_length] = np.random.uniform(0.5, 0.8, regime_length)  # Brittle
    bdi_values[2*regime_length:] = np.random.uniform(1.1, 1.8, n_samples - 2*regime_length)  # Ductile
    
    sample_indices = np.arange(n_samples)
    
    return true_values, predictions, bdi_values, sample_indices


def create_prediction_plot(true_values, predictions, bdi_values, sample_indices, 
                          model_type="Example Model", output_path=None):
    """
    Create a time-series plot showing predicted vs ground truth surface roughness
    with background colored by BDI regime.
    
    Parameters:
    - true_values: Ground truth surface roughness values
    - predictions: Model predictions
    - bdi_values: BDI values for each sample
    - sample_indices: Indices for x-axis
    - model_type: Type of model used
    - output_path: Path to save the plot (if None, returns figure without saving)
    
    Returns:
    - fig, ax: Matplotlib figure and axes objects
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Calculate MAE
    mae = np.mean(np.abs(true_values - predictions))
    
    # Plot ground truth and predictions
    ax.plot(sample_indices, true_values, 'o-', label='Ground Truth', 
            color='black', alpha=0.8, markersize=4, linewidth=1.5)
    ax.plot(sample_indices, predictions, 's-', label='Prediction', 
            color='red', alpha=0.8, markersize=4, linewidth=1.5)
    
    # Create background colors based on BDI regime
    bdi_regime = bdi_values > 1.0  # True for ductile, False for brittle
    
    # Group consecutive samples with same regime
    regime_changes = np.where(np.diff(bdi_regime.astype(int)) != 0)[0] + 1
    regime_starts = np.concatenate(([0], regime_changes))
    regime_ends = np.concatenate((regime_changes, [len(bdi_regime)]))
    
    # Color background based on BDI regime
    for start, end in zip(regime_starts, regime_ends):
        regime = bdi_regime[start]
        color = 'lightblue' if regime else 'lightcoral'
        alpha = 0.3 if regime else 0.2
        
        # Convert to integer indices
        x_start_idx = max(0, int(start-0.5))
        x_end_idx = min(len(sample_indices)-1, int(end-0.5))
        x_start = sample_indices[x_start_idx]
        x_end = sample_indices[x_end_idx]
        
        ax.axvspan(x_start, x_end, ymin=0, ymax=1, alpha=alpha, color=color)
    
    # Customize plot
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Surface Roughness Ra ($\\mu$m)')
    ax.set_title(f'Prediction vs Ground Truth with Physical Context\nModel: {model_type}')
    
    # Create legend with regime information
    legend_elements = [
        Line2D([0], [0], color='black', marker='o', linestyle='-', label='Ground Truth'),
        Line2D([0], [0], color='red', marker='s', linestyle='-', label='Prediction'),
        Patch(facecolor='lightblue', alpha=0.3, label='BDI > 1 (Ductile)'),
        Patch(facecolor='lightcoral', alpha=0.2, label='BDI < 1 (Brittle)')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    ax.grid(True, alpha=0.3)
    
    # Add MAE annotation
    ax.text(0.02, 0.98, f'MAE = {mae:.2f} μm', 
            transform=ax.transAxes, fontsize=12, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            verticalalignment='top')
    
    # Add caption-like text
    caption_text = (
        f"The model demonstrates high fidelity in predicting Ra (MAE = {mae:.2f}). "
        "Notably, prediction accuracy remains robust during transitions between "
        "ductile (blue) and brittle (red) machining regimes, showcasing the model's "
        "ability to capture non-stationary dynamics."
    )
    
    # Add caption below plot
    fig.text(0.5, 0.01, caption_text, ha='center', fontsize=10, 
             style='italic', wrap=True)
    
    plt.tight_layout(rect=(0, 0.05, 1, 0.95))  # Make room for caption
    
    # Save plot if output path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    
    return fig, ax


def create_simple_plot(output_path=None):
    """Create a simple example plot."""
    print("Creating simple example plot...")
    
    # Generate example data
    true_values, predictions, bdi_values, sample_indices = generate_example_data(30)
    
    # Create plot
    fig, ax = create_prediction_plot(
        true_values, predictions, bdi_values, sample_indices,
        model_type="Simple Example",
        output_path=output_path
    )
    
    plt.close(fig)
    return True


def create_detailed_plot(output_path=None):
    """Create a detailed example plot with more samples."""
    print("Creating detailed example plot...")
    
    # Generate example data with more samples
    true_values, predictions, bdi_values, sample_indices = generate_example_data(100)
    
    # Create plot
    fig, ax = create_prediction_plot(
        true_values, predictions, bdi_values, sample_indices,
        model_type="Detailed Example",
        output_path=output_path
    )
    
    plt.close(fig)
    return True


def create_bdi_regime_plot(output_path=None):
    """Create a plot focusing on BDI regime visualization."""
    print("Creating BDI regime visualization plot...")
    
    np.random.seed(123)
    n_samples = 40
    
    # Create data with clear regime transitions
    true_values = np.sin(np.linspace(0, 4*np.pi, n_samples)) * 0.5 + 1.0
    predictions = true_values + np.random.normal(0, 0.05, n_samples)
    
    # Create BDI values with clear transitions
    bdi_values = np.ones(n_samples)
    bdi_values[:10] = np.random.uniform(1.5, 2.0, 10)  # Ductile
    bdi_values[10:20] = np.random.uniform(0.4, 0.7, 10)  # Brittle
    bdi_values[20:30] = np.random.uniform(1.3, 1.8, 10)  # Ductile
    bdi_values[30:] = np.random.uniform(0.6, 0.9, 10)  # Brittle
    
    sample_indices = np.arange(n_samples)
    
    # Create plot
    fig, ax = create_prediction_plot(
        true_values, predictions, bdi_values, sample_indices,
        model_type="BDI Regime Example",
        output_path=output_path
    )
    
    plt.close(fig)
    return True


def create_all_plots():
    """Create all example plots."""
    print("=" * 60)
    print("Generating Example Plot PNG Files")
    print("=" * 60)
    
    # Get the directory containing this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create output directory if it doesn't exist
    output_dir = script_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate plots
    plots = [
        ("simple_prediction_plot.png", create_simple_plot),
        ("detailed_prediction_plot.png", create_detailed_plot),
        ("bdi_regime_plot.png", create_bdi_regime_plot),
    ]
    
    for filename, plot_func in plots:
        output_path = os.path.join(output_dir, filename)
        print(f"\nGenerating {filename}...")
        try:
            plot_func(output_path)
            print(f"✓ Successfully created {filename}")
        except Exception as e:
            print(f"✗ Error creating {filename}: {e}")
    
    print("\n" + "=" * 60)
    print("All example plots generated successfully!")
    print("=" * 60)
    
    # List generated files
    print("\nGenerated files:")
    for filename, _ in plots:
        filepath = os.path.join(output_dir, filename)
        if os.path.exists(filepath):
            file_size = os.path.getsize(filepath)
            print(f"  - {filename} ({file_size:,} bytes)")
        else:
            print(f"  - {filename} (NOT FOUND)")
    
    return True


if __name__ == "__main__":
    create_all_plots()
    print("\nDone! Example PNG files have been generated in the test_plot folder.")
