import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import matplotlib as mpl
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# Add project root and utils to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "utils"))

from MyModels import GrindingPredictor
from MyDataset import get_dataset, get_collate_fn

# Set up plotting style for publications
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14
mpl.rcParams['figure.dpi'] = 300

def load_best_model(model_type="all", fold=0):
    """Load the trained model."""
    model_path = os.path.join(project_root, "lfs", "checkpoints", f"{model_type}_fold{fold}_of_folds10.pt")
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return None
    
    model = GrindingPredictor(input_type=model_type)
    checkpoint = torch.load(model_path, map_location='cpu')
    
    if 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
    elif 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()
    print(f"Successfully loaded model from {model_path}")
    return model

def get_predictions(model, dataset, num_samples=200):
    """Generate predictions and extract physical indicators."""
    collate_fn = get_collate_fn(model.input_type)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    
    predictions = []
    ground_truth = []
    bdi_values = []
    st_values = []
    
    print(f"Generating predictions for {num_samples} samples...")
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_samples:
                break
            
            pred = model(batch)
            if isinstance(pred, tuple):
                pred = pred[0]
            
            predictions.append(pred.item())
            ground_truth.append(batch['label'].item())
            
            # Extract BDI and St from features_pp [ec, st, bid]
            pp = batch['features_pp'].squeeze().numpy()
            st_values.append(pp[1])
            bdi_values.append(pp[2])
            
    return np.array(predictions), np.array(ground_truth), np.array(bdi_values), np.array(st_values)

def plot_panel_a(true_values, predictions, bdi_values, output_path):
    """Panel A: Prediction vs Ground Truth with BDI context."""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    indices = np.arange(len(true_values))
    mae = np.mean(np.abs(true_values - predictions))
    
    # Plot data
    ax.plot(indices, true_values, 'k-', label='Measured $R_a$', alpha=0.8, linewidth=1.5)
    ax.plot(indices, predictions, 'r--', label='Predicted $R_a$', alpha=0.8, linewidth=1.5)
    
    # Background coloring based on BDI
    threshold = np.median(bdi_values) 
    
    bdi_regime = bdi_values > threshold
    regime_changes = np.where(np.diff(bdi_regime.astype(int)) != 0)[0] + 1
    regime_starts = np.concatenate(([0], regime_changes))
    regime_ends = np.concatenate((regime_changes, [len(bdi_regime)]))
    
    for start, end in zip(regime_starts, regime_ends):
        regime = bdi_regime[start]
        color = 'skyblue' if regime else 'salmon'
        ax.axvspan(start, end, alpha=0.2, color=color)
        
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Surface Roughness $R_a$ ($\mu$m)')
    
    legend_elements = [
        Line2D([0], [0], color='black', linestyle='-', label='Measured $R_a$'),
        Line2D([0], [0], color='red', linestyle='--', label='Predicted $R_a$'),
        Patch(facecolor='skyblue', alpha=0.3, label='Ductile-dominated'),
        Patch(facecolor='salmon', alpha=0.3, label='Brittle-dominated')
    ]
    ax.legend(handles=legend_elements, loc='upper right', frameon=True)
    
    plt.title(f'Prediction Fidelity across Machining Regimes (MAE: {mae:.3f} $\mu$m)')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Panel A saved to {output_path}")

def plot_panel_b(true_values, predictions, bdi_values, output_path):
    """Panel B: Error Analysis vs BDI."""
    errors = np.abs(true_values - predictions)
    fig, ax = plt.subplots(figsize=(8, 6))
    
    scatter = ax.scatter(bdi_values, errors, c='blue', alpha=0.5, s=30, edgecolors='none')
    
    # Add trend line
    z = np.polyfit(bdi_values, errors, 1)
    p = np.poly1d(z)
    ax.plot(np.sort(bdi_values), p(np.sort(bdi_values)), "r--", alpha=0.8, linewidth=2)
    
    ax.set_xlabel('Brittle-Ductile Indicator (BDI)')
    ax.set_ylabel('Absolute Error ($\mu$m)')
    ax.set_title('Prediction Error Stability vs. BDI')
    ax.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Panel B saved to {output_path}")

def plot_panel_c(true_values, predictions, st_values, output_path):
    """Panel C: Error Analysis vs St."""
    errors = np.abs(true_values - predictions)
    fig, ax = plt.subplots(figsize=(8, 6))
    
    scatter = ax.scatter(st_values, errors, c='darkgreen', alpha=0.5, s=30, edgecolors='none')
    
    # Add trend line
    z = np.polyfit(st_values, errors, 1)
    p = np.poly1d(z)
    ax.plot(np.sort(st_values), p(np.sort(st_values)), "r--", alpha=0.8, linewidth=2)
    
    ax.set_xlabel('Thermal Severity ($S_t$)')
    ax.set_ylabel('Absolute Error ($\mu$m)')
    ax.set_title('Robustness under Thermal Loads')
    ax.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Panel C saved to {output_path}")

def main():
    # 1. Setup paths
    image_dir = os.path.join(project_root, "Grinding Fusion", "images")
    os.makedirs(image_dir, exist_ok=True)
    
    # 2. Load model and data
    model = load_best_model(model_type="all", fold=0)
    if model is None:
        return
    
    dataset = get_dataset(input_type="all", dataset_mode="classical")
    
    # 3. Generate predictions
    predictions, true_values, bdi_values, st_values = get_predictions(model, dataset, num_samples=50)
    
    # 4. Generate plots
    plot_panel_a(true_values, predictions, bdi_values, os.path.join(image_dir, "phys_regime_time_series.png"))
    plot_panel_b(true_values, predictions, bdi_values, os.path.join(image_dir, "error_vs_bdi.png"))
    plot_panel_c(true_values, predictions, st_values, os.path.join(image_dir, "error_vs_st.png"))
    
    print("\nAll physical regime analysis plots generated successfully.")

if __name__ == "__main__":
    main()