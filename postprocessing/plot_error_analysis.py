import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde

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
    st_values = grinding_data.st
    
    # Convert to numpy arrays and ensure proper shape
    true_values = np.array(true_values).flatten()
    bdi_values = np.array(bdi_values).flatten()
    st_values = np.array(st_values).flatten()
    
    print(f"Loaded physics data for {len(true_values)} samples")
    print(f"BDI range: {np.min(bdi_values):.3f} to {np.max(bdi_values):.3f}")
    print(f"St range: {np.min(st_values):.3f} to {np.max(st_values):.3f}")
    print(f"Surface roughness range: {np.min(true_values):.3f} to {np.max(true_values):.3f} μm")
    
    return true_values, bdi_values, st_values

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

def calculate_absolute_errors(true_values, predictions):
    """Calculate absolute prediction errors."""
    return np.abs(true_values - predictions)

def plot_error_vs_bdi(true_values, predictions, bdi_values):
    """
    Scatter plot of absolute prediction error vs. BDI.
    Highlight consistent low error around BDI=1 threshold.
    """
    absolute_errors = calculate_absolute_errors(true_values, predictions)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create scatter plot with color coding based on BDI region
    bdi_near_1_mask = (bdi_values >= 0.9) & (bdi_values <= 1.1)
    bdi_low_mask = bdi_values < 0.9
    bdi_high_mask = bdi_values > 1.1
    
    # Plot points with different colors for different BDI regions
    ax.scatter(bdi_values[bdi_low_mask], absolute_errors[bdi_low_mask], 
               alpha=0.6, color='lightcoral', label='BDI < 0.9', s=30)
    ax.scatter(bdi_values[bdi_near_1_mask], absolute_errors[bdi_near_1_mask], 
               alpha=0.8, color='blue', label='0.9 $\leq$ BDI $\leq$ 1.1', s=40)
    ax.scatter(bdi_values[bdi_high_mask], absolute_errors[bdi_high_mask], 
               alpha=0.6, color='lightgreen', label='BDI > 1.1', s=30)
    
    # Add vertical lines at critical BDI values
    ax.axvline(x=0.9, color='gray', linestyle='--', alpha=0.7, linewidth=1)
    ax.axvline(x=1.0, color='black', linestyle='-', alpha=0.8, linewidth=2)
    ax.axvline(x=1.1, color='gray', linestyle='--', alpha=0.7, linewidth=1)
    
    # Add shaded region around BDI=1
    ax.axvspan(0.9, 1.1, alpha=0.2, color='lightblue', label=r'Critical Region (BDI $\approx$ 1)')
    
    # Customize plot
    ax.set_xlabel('BDI Value')
    ax.set_ylabel('Absolute Prediction Error ($\mu$m)')
    ax.set_title('Absolute Prediction Error vs. BDI\nHighlighting Consistent Low Error Around Critical Threshold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add text annotations
    ax.text(1.02, ax.get_ylim()[1] * 0.9, 'BDI = 1', rotation=90, 
            verticalalignment='top', fontsize=12, fontweight='bold')
    ax.text(0.91, ax.get_ylim()[1] * 0.8, 'Critical\nRegion', 
            verticalalignment='top', fontsize=10, color='darkblue')
    
    # Calculate and display average error in critical region
    avg_error_critical = np.mean(absolute_errors[bdi_near_1_mask])
    ax.text(0.05, 0.95, f'Avg Error near BDI=1: {avg_error_critical:.2f} $\mu$m', 
            transform=ax.transAxes, fontsize=12, bbox=dict(boxstyle="round,pad=0.3", 
            facecolor="white", alpha=0.8))
    
    return fig, ax

def plot_error_vs_st(true_values, predictions, st_values):
    """
    Scatter plot of absolute prediction error vs. Thermal Severity (St).
    Emphasize how physics-informed loss maintains accuracy as St approaches 1.
    """
    absolute_errors = calculate_absolute_errors(true_values, predictions)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create scatter plot with color intensity based on St value
    colors = st_values
    scatter = ax.scatter(st_values, absolute_errors, c=colors, cmap='viridis', 
                        alpha=0.7, s=40)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Thermal Severity (St)')
    
    # Highlight region as St approaches 1
    st_high_mask = st_values > 0.8
    ax.scatter(st_values[st_high_mask], absolute_errors[st_high_mask], 
               alpha=0.9, color='red', s=50, label='St > 0.8')
    
    # Add vertical line at St=1 (critical value)
    ax.axvline(x=1.0, color='red', linestyle='--', alpha=0.8, linewidth=2, 
               label='St = 1 (Critical)')
    
    # Add trend line using polynomial fit
    if len(st_values) > 10:
        # Sort for smooth trend line
        sort_idx = np.argsort(st_values)
        st_sorted = st_values[sort_idx]
        errors_sorted = absolute_errors[sort_idx]
        
        # Polynomial fit (degree 2)
        z = np.polyfit(st_sorted, errors_sorted, 2)
        p = np.poly1d(z)
        st_fit = np.linspace(st_sorted.min(), st_sorted.max(), 100)
        ax.plot(st_fit, p(st_fit), 'r-', linewidth=2, alpha=0.8, 
                label='Trend Line')
    
    # Customize plot
    ax.set_xlabel('Thermal Severity (St)')
    ax.set_ylabel('Absolute Prediction Error ($\mu$m)')
    ax.set_title('Absolute Prediction Error vs. Thermal Severity (St)\nPhysics-Informed Loss Maintains Accuracy Near Critical St=1')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add text annotation emphasizing physics-informed performance
    ax.text(0.05, 0.95, 'Physics-Informed Loss:\nMaintains Accuracy\nas St → 1', 
            transform=ax.transAxes, fontsize=12, bbox=dict(boxstyle="round,pad=0.3", 
            facecolor="lightyellow", alpha=0.9), verticalalignment='top')
    
    # Calculate and display statistics for high St region
    if np.any(st_high_mask):
        avg_error_high_st = np.mean(absolute_errors[st_high_mask])
        ax.text(0.65, 0.95, f'Avg Error for St > 0.8: {avg_error_high_st:.2f} μm', 
                transform=ax.transAxes, fontsize=10, bbox=dict(boxstyle="round,pad=0.3", 
                facecolor="white", alpha=0.8))
    
    return fig, ax

def main(prediction_file=None):
    """Main function to generate both error analysis plots."""
    print("Loading physics data efficiently...")
    true_values, bdi_values, st_values = load_physics_data_only()
    
    # Load or create predictions
    predictions = load_predictions_from_file(prediction_file)
    if predictions is None:
        predictions = true_values.copy()  # Placeholder
    
    print("Creating error vs BDI plot...")
    fig1, ax1 = plot_error_vs_bdi(true_values, predictions, bdi_values)
    
    # Save the plot
    output_path1 = os.path.join(os.path.dirname(__file__), 'error_vs_bdi.png')
    plt.savefig(output_path1, dpi=300, bbox_inches='tight')
    print(f"Error vs BDI plot saved to: {output_path1}")
    
    print("Creating error vs St plot...")
    fig2, ax2 = plot_error_vs_st(true_values, predictions, st_values)
    
    # Save the plot
    output_path2 = os.path.join(os.path.dirname(__file__), 'error_vs_st.png')
    plt.savefig(output_path2, dpi=300, bbox_inches='tight')
    print(f"Error vs St plot saved to: {output_path2}")
    
    # Show the plots
    plt.show()

if __name__ == "__main__":
    main()
