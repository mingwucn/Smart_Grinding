import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

# Add parent directory to path to import GrindingData
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_physics_data_only():
    """
    Load only physics data using GrindingData's _load_all_physics_data method.
    This avoids loading the full dataset with spectrograms and other large data.
    """
    from GrindingData import GrindingData
    from MyDataset import project_dir
    
    # Create GrindingData instance
    grinding_data = GrindingData(project_dir)
    
    # Load only physics data (much more efficient)
    print("Loading physics data only...")
    grinding_data._load_all_physics_data()
    
    # Extract the data we need
    true_values = grinding_data.sr * 1e3  # Convert to μm
    bdi_values = grinding_data.bid
    
    # Convert to numpy arrays and ensure proper shape
    true_values = np.array(true_values).flatten()
    bdi_values = np.array(bdi_values).flatten()
    
    print(f"Loaded physics data for {len(true_values)} samples")
    print(f"BDI range: {np.min(bdi_values):.3f} to {np.max(bdi_values):.3f}")
    print(f"Surface roughness range: {np.min(true_values):.3f} to {np.max(true_values):.3f} μm")
    
    return true_values, bdi_values

def plot_prediction_overlay_efficient(true_values, predictions, bdi_values):
    """
    Plot predictions overlaying measurements with background colored by BDI regime.
    Uses efficient data loading.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sort data by BDI for better visualization
    sort_idx = np.argsort(bdi_values)
    bdi_sorted = bdi_values[sort_idx]
    true_sorted = true_values[sort_idx]
    pred_sorted = predictions[sort_idx]
    
    # Plot measurements and predictions
    ax.plot(bdi_sorted, true_sorted, 'o-', label='Measurement', color='blue', alpha=0.7, markersize=4)
    ax.plot(bdi_sorted, pred_sorted, 's-', label='Prediction', color='red', alpha=0.7, markersize=4)
    
    # Add vertical line at BDI=1
    ax.axvline(x=1.0, color='black', linestyle='--', alpha=0.8, linewidth=1)
    
    # Color background based on BDI regime
    bdi_min, bdi_max = np.min(bdi_sorted), np.max(bdi_sorted)
    
    # Fill area for BDI < 1
    if bdi_min < 1.0:
        ax.axvspan(bdi_min, 1.0, alpha=0.2, color='lightcoral', label='BDI < 1')
    
    # Fill area for BDI > 1  
    if bdi_max > 1.0:
        ax.axvspan(1.0, bdi_max, alpha=0.2, color='lightgreen', label='BDI > 1')
    
    # Customize plot
    ax.set_xlabel('BDI Value')
    ax.set_ylabel('Surface Roughness ($\mu$m)')
    ax.set_title('Prediction vs Measurement with BDI Regime Background (Efficient Loading)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add text annotation for BDI=1 line
    ax.text(1.02, ax.get_ylim()[1] * 0.95, 'BDI = 1', rotation=90, 
            verticalalignment='top', fontsize=10)
    
    return fig, ax

def load_predictions_from_file(prediction_file=None):
    """
    Load predictions from a file. If no file provided, use true values as placeholder.
    In practice, you would load your model predictions here.
    """
    if prediction_file and os.path.exists(prediction_file):
        # Load predictions from file (adjust format as needed)
        predictions = np.load(prediction_file)
        print(f"Loaded predictions from {prediction_file}")
    else:
        # Use true values as placeholder for predictions
        predictions = None
        print("Using true values as placeholder for predictions")
    
    return predictions

def main_efficient(prediction_file=None):
    """Main function using efficient data loading."""
    print("Loading physics data efficiently...")
    true_values, bdi_values = load_physics_data_only()
    
    # Load or create predictions
    predictions = load_predictions_from_file(prediction_file)
    if predictions is None:
        predictions = true_values.copy()  # Placeholder
    
    print("Creating plot with efficient data loading...")
    fig, ax = plot_prediction_overlay_efficient(true_values, predictions, bdi_values)
    
    # Save the plot
    output_path = os.path.join(os.path.dirname(__file__), 'prediction_overlay_bdi_efficient.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    # Show the plot
    plt.show()

def compare_data_loading():
    """Compare memory usage between full dataset and physics-only loading."""
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    
    print("Memory usage comparison:")
    print(f"Initial memory: {process.memory_info().rss / 1024 / 1024:.2f} MB")
    
    # Test efficient loading
    true_values, bdi_values = load_physics_data_only()
    mem_after_efficient = process.memory_info().rss / 1024 / 1024
    print(f"After efficient loading: {mem_after_efficient:.2f} MB")
    
    # Clean up
    del true_values, bdi_values
    
    return mem_after_efficient

if __name__ == "__main__":
    # Compare memory usage
    mem_usage = compare_data_loading()
    print(f"\nEfficient loading uses approximately {mem_usage:.2f} MB")
    
    # Run main function
    main_efficient()
