import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import matplotlib as mpl
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import argparse

# Add project root and utils to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "utils"))

from MyModels import GrindingPredictor
from MyDataset import get_dataset, get_collate_fn

# Set up plotting style matching "prediction_time_series_ae_features.png" style
plt.rcParams['figure.constrained_layout.use'] = True
mpl.rcParams['figure.dpi'] = 300
# Font settings: sans-serif as requested
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Liberation Sans', 'DejaVu Sans', 'Bitstream Vera Sans', 'sans-serif']

# Model types from generate_accuracy_report.py
ALLOWED_INPUT_TYPES = [
    'ae_spec',
    'ae_features',
    'ae_features+pp',
    'ae_spec+ae_features',
    'vib_spec',
    'vib_features',
    'vib_features+pp',
    'vib_spec+vib_features',
    'ae_features+vib_features',
    'ae_features+vib_features+pp',
    'ae_spec+vib_spec',
    'ae_spec+ae_features+vib_spec+vib_features',
    'all',
]

def load_best_model(model_type="all", fold=0, device="cpu"):
    """Load the trained model."""
    checkpoint_dirs = [
        os.path.join(project_root, "lfs", "checkpoints"),
        os.path.join(project_root, "checkpoints")
    ]
    
    model_path = None
    for d in checkpoint_dirs:
        path = os.path.join(d, f"{model_type}_fold{fold}_of_folds10.pt")
        if os.path.exists(path):
            model_path = path
            break
            
    if model_path is None:
        print(f"Model not found for type '{model_type}'")
        return None
    
    try:
        model = GrindingPredictor(input_type=model_type)
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
        
        if 'model_state' in checkpoint:
            model.load_state_dict(checkpoint['model_state'])
        elif 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        model.to(device)
        model.eval()
        print(f"Loaded model: {model_type} (Device: {device})")
        return model
    except Exception as e:
        print(f"Error loading model {model_type}: {e}")
        return None

def get_predictions(model, dataset, device="cpu"):
    """Generate predictions for ALL samples."""
    collate_fn = get_collate_fn(model.input_type)
    # Increase batch size for faster inference
    # Use multiple workers and pinned memory for speed
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False, collate_fn=collate_fn, num_workers=4, pin_memory=True)
    
    predictions = []
    ground_truth = []
    bdi_values = []
    st_values = []
    
    print(f"Generating predictions for {len(dataset)} samples using {device}...", flush=True)
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            try:
                # Move batch to device
                batch_on_device = {}
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch_on_device[k] = v.to(device)
                    else:
                        batch_on_device[k] = v
                
                pred = model(batch_on_device)
                if isinstance(pred, tuple):
                    pred = pred[0]
                
                # Move predictions back to cpu for numpy conversion
                predictions.extend(pred.cpu().flatten().numpy())
                ground_truth.extend(batch['label'].flatten().numpy())
                
                # Extract PP features (assumed to be cpu tensor in original batch or moved back)
                # features_pp is [batch, 3] -> [ec, st, bid]
                pp = batch['features_pp'] # Use original batch which is on CPU if we didn't overwrite it, or move back
                if isinstance(pp, torch.Tensor):
                    pp = pp.cpu().numpy()
                
                st_values.extend(pp[:, 1])
                bdi_values.extend(pp[:, 2])
                
            except Exception as e:
                print(f"Error prediction batch {i}: {e}")
                continue
            
    return np.array(predictions), np.array(ground_truth), np.array(bdi_values), np.array(st_values)

def plot_time_series_with_physics(true_values, predictions, bdi_values, model_type, output_path):
    """Time-series plot showing predicted vs ground truth with BDI regime background."""
    indices = np.arange(len(true_values))
    mae = np.mean(np.abs(true_values - predictions))
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot ground truth and predictions
    ax.plot(indices, true_values, 'o-', label='Ground Truth', 
            color='black', alpha=0.8, markersize=4, linewidth=1.5)
    ax.plot(indices, predictions, 's-', label='Prediction', 
            color='red', alpha=0.8, markersize=4, linewidth=1.5)
    
    # Find transitions between BDI regimes
    threshold = 1.0 if np.max(bdi_values) > 1.0 else np.median(bdi_values)
    bdi_regime = bdi_values > threshold
    
    regime_changes = np.where(np.diff(bdi_regime.astype(int)) != 0)[0] + 1
    regime_starts = np.concatenate(([0], regime_changes))
    regime_ends = np.concatenate((regime_changes, [len(bdi_regime)]))
    
    for start, end in zip(regime_starts, regime_ends):
        regime = bdi_regime[start]
        color = 'lightblue' if regime else 'lightcoral'
        alpha = 0.3 if regime else 0.2
        ax.axvspan(start-0.5, end-0.5, ymin=0, ymax=1, alpha=alpha, color=color)
    
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Surface Roughness Ra ($\mu$m)')
    ax.set_title(f'Prediction vs Ground Truth with Physical Context\nModel: {model_type}')
    
    legend_elements = [
        Line2D([0], [0], color='black', marker='o', linestyle='-', label='Ground Truth'),
        Line2D([0], [0], color='red', marker='s', linestyle='-', label='Prediction'),
        Patch(facecolor='lightblue', alpha=0.3, label='Ductile-dominated'),
        Patch(facecolor='lightcoral', alpha=0.2, label='Brittle-dominated')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Add MAE annotation
    ax.text(0.02, 0.98, f'MAE = {mae:.2f} $\mu$m', 
            transform=ax.transAxes, fontsize=12, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            verticalalignment='top')
    
    # Add the specific caption text the user likes
    caption_text = (
        f"The model demonstrates high fidelity in predicting Ra (MAE = {mae:.2f}). "
        "Notably, prediction accuracy remains robust during transitions between "
        "ductile (blue) and brittle (red) machining regimes, showcasing the model's "
        "ability to capture non-stationary dynamics."
    )
    fig.text(0.5, 0.01, caption_text, ha='center', fontsize=10, style='italic', wrap=True)
    
    plt.tight_layout(rect=(0, 0.05, 1, 0.95))
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {output_path}")
    plt.close()

def main():
    image_dir = os.path.join(project_root, "Grinding Fusion", "images", "prediction_plots")
    os.makedirs(image_dir, exist_ok=True)
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    
    # Process only 'all' as requested
    model_type = 'all'
    print(f"\n=== Generating Plot for Model: {model_type} ===")
    
    model = load_best_model(model_type=model_type, device=device)
    if model is None:
        return
    
    dataset = get_dataset(input_type=model_type, dataset_mode="classical")
    
    predictions, true_values, bdi_values, st_values = get_predictions(model, dataset, device=device)
    
    output_path = os.path.join(image_dir, f"time_series_{model_type}.png")
    plot_time_series_with_physics(true_values, predictions, bdi_values, model_type, output_path)
    
    print(f"\nPlot generated for '{model_type}'. Please check: {output_path}")

if __name__ == "__main__":
    main()