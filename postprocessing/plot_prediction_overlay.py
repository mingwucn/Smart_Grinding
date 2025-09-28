import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Patch

# Add parent directory to path to import MyDataset
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from MyDataset import get_dataset

def load_data_and_predictions():
    """
    Load dataset and get true values, BDI values, and predictions.
    For now, using true values as predictions placeholder.
    """
    # Load the dataset
    dataset = get_dataset(input_type="all", dataset_mode="classical")
    
    true_values = []
    bdi_values = []
    
    # Extract true labels and BDI values from the dataset
    for i in range(len(dataset)):
        item = dataset[i]
        true_values.append(item['label'].item())
        # BDI is the third element in features_pp [ec, st, bid]
        bdi_values.append(item['features_pp'][2].item())
    
    true_values = np.array(true_values)
    bdi_values = np.array(bdi_values)
    
    # For demonstration, use true values as predictions
    # In practice, replace this with actual model predictions
    predictions = true_values.copy()
    
    return true_values, predictions, bdi_values

def plot_prediction_overlay(true_values, predictions, bdi_values):
    """
    Plot predictions overlaying measurements with background colored by BDI regime.
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
    ax.set_title('Prediction vs Measurement with BDI Regime Background')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add text annotation for BDI=1 line
    ax.text(1.02, ax.get_ylim()[1] * 0.95, 'BDI = 1', rotation=90, 
            verticalalignment='top', fontsize=10)
    
    return fig, ax

def main():
    """Main function to generate the plot."""
    print("Loading data and generating predictions...")
    true_values, predictions, bdi_values = load_data_and_predictions()
    
    print(f"Loaded {len(true_values)} data points")
    print(f"BDI range: {np.min(bdi_values):.3f} to {np.max(bdi_values):.3f}")
    print(f"Surface roughness range: {np.min(true_values):.3f} to {np.max(true_values):.3f} μm")
    
    print("Creating plot...")
    fig, ax = plot_prediction_overlay(true_values, predictions, bdi_values)
    
    # Save the plot
    output_path = os.path.join(os.path.dirname(__file__), 'prediction_overlay_bdi.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    main()
