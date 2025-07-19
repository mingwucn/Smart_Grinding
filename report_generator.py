import os
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import torch
import subprocess
from utils.preprocessing import one_column
from utils.XAI import GradCAM, get_conv_layer_names
from utils.postprocessing import calculate_regression_metrics
from MyModels import GrindingPredictor
from XAI_ModelWrapper import XAI_ModelWrapper
from GrindingData import GrindingData
from MyDataset import get_dataset, get_collate_fn
from torch.utils.data import DataLoader

# Configuration
REPORT_PATH = "report"
SNAPSHOT_DIR = "snapshots"
HISTORY_DIR = "history"
LFS_PATH = "lfs"

# Dynamically determine input combinations from LFS directory
def get_input_combinations():
    """Scan LFS directory to find available input combinations"""
    input_combinations = set()
    train_his_dir = os.path.join(LFS_PATH, "train_his")
    
    # Process training history files
    if os.path.exists(train_his_dir):
        for filename in os.listdir(train_his_dir):
            if filename.endswith(".csv"):
                # Extract input combination from filename pattern:
                # {input_type}_fold... or {input_type}.csv
                parts = filename.split('_fold')
                if len(parts) > 0:
                    input_comb = parts[0]
                    input_combinations.add(input_comb)
    
    return list(input_combinations)

INPUT_COMBINATIONS = get_input_combinations()
print(f"Found input combinations: {INPUT_COMBINATIONS}")

# Fallback to default combinations if none found
if not INPUT_COMBINATIONS:
    print("No input combinations found in LFS directory. Using default combinations.")
    INPUT_COMBINATIONS = [
        "ae",
        "mic",
        "power",
        "ae+mic",
        "ae+power",
        "mic+power",
        "ae+mic+power"
    ]

def resolve_model_path(input_type):
    """Get model path for given input combination"""
    # First check local snapshots directory
    local_path = os.path.join(SNAPSHOT_DIR, f"Res15_classification_input_{input_type}_output_regime_roi_time10_roi_radius3_fold0_of_folds10.pt")
    if os.path.exists(local_path):
        return local_path
    
    # Check LFS checkpoints directory
    checkpoints_dir = os.path.join(LFS_PATH, "checkpoints")
    if os.path.exists(checkpoints_dir):
        # Find the first model file matching the input_type pattern
        for filename in os.listdir(checkpoints_dir):
            if filename.startswith(f"{input_type}_") and filename.endswith(".pt"):
                return os.path.join(checkpoints_dir, filename)
    
    return None

def load_history(input_type):
    """Load training history for given input combination"""
    # First check local history directory
    local_path = os.path.join(HISTORY_DIR, f"{input_type}_fold0_of_folds10.csv")
    if os.path.exists(local_path):
        df = pd.read_csv(local_path, index_col=0)
        if "Test Accuracy" in df.columns:
            return df["Test Accuracy"].iloc[-1]
        elif "accuracy" in df.columns:
            return df["accuracy"].iloc[-1]
        return 0.0
    
    # Check LFS train_his directory
    train_his_dir = os.path.join(LFS_PATH, "train_his")
    if os.path.exists(train_his_dir):
        # Find the first history file matching the input_type pattern
        for filename in os.listdir(train_his_dir):
            if filename.startswith(f"{input_type}_") and filename.endswith(".csv"):
                file_path = os.path.join(train_his_dir, filename)
                df = pd.read_csv(file_path, index_col=0)
                if "Test Accuracy" in df.columns:
                    return df["Test Accuracy"].iloc[-1]
                elif "accuracy" in df.columns:
                    return df["accuracy"].iloc[-1]
                return 0.0
    
    # Check for JSON results files
    results_path = os.path.join(LFS_PATH, "checkpoints", f"results_{input_type}.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)
            if "test_accuracy" in results:
                return results["test_accuracy"]
            elif "accuracy" in results:
                return results["accuracy"]
    
    return 0.0

def generate_accuracy_report():
    """Generate detailed statistical report in markdown format."""
    os.makedirs(os.path.join(REPORT_PATH, "accuracy_reports"), exist_ok=True)
    md_content = "# Detailed Statistical Performance Report\n\n"

    for combo in INPUT_COMBINATIONS:
        try:
            metrics = get_metrics_for_combo(combo)
            if not metrics:
                continue

            md_content += f"## {combo.replace('_', ' ').title()} Input Combination\n\n"
            
            for metric_name, stats_dict in metrics[0].items():
                if metric_name == "Input Combination":
                    continue
                
                md_content += f"### {metric_name}\n"
                df = pd.DataFrame.from_dict(stats_dict, orient='index', columns=['Value'])
                md_content += df.to_markdown() + "\n\n"

        except Exception as e:
            print(f"Failed to generate detailed report for {combo}: {e}")

    report_path = os.path.join(REPORT_PATH, "accuracy_reports", "summary.md")
    with open(report_path, "w") as f:
        f.write(md_content)
    
    print(f"Saved detailed accuracy report to {report_path}")
    return md_content

def generate_shap_report(model, sample_input, output_path):
    """Generate SHAP report for a model"""
    # Initialize SHAP explainer
    explainer = shap.DeepExplainer(model, sample_input)
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(sample_input)
    
    # Generate visualization
    plt.figure()
    shap.summary_plot(shap_values, sample_input, show=False)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

def generate_gradcam_report(model, sample_input, raw_signal, target_layer, output_path):
    """Generate Grad-CAM report with detailed plots."""
    # Initialize Grad-CAM
    cam = GradCAM(model, target_layer)
    
    # Generate heatmap
    heatmap = cam(sample_input)
    
    # Create visualization
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    
    # Original 1D Signal
    axs[0].plot(raw_signal)
    axs[0].set_title('Original 1D Signal')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Amplitude')

    # Original 2D Spectrogram
    axs[1].imshow(sample_input[0, 0].cpu().detach().numpy(), cmap='gray', aspect='auto')
    axs[1].set_title('Original Spectrogram')
    axs[1].axis('off')
    
    # Heatmap
    axs[2].imshow(heatmap, cmap='viridis', aspect='auto')
    axs[2].set_title('Grad-CAM Heatmap')
    axs[2].axis('off')
    
    # Overlay
    axs[3].imshow(sample_input[0, 0].cpu().detach().numpy(), cmap='gray', aspect='auto')
    axs[3].imshow(heatmap, cmap='viridis', alpha=0.5, aspect='auto')
    axs[3].set_title('Overlay')
    axs[3].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Grad-CAM report saved to {output_path}")

def generate_xai_reports():
    """Generate XAI reports for all models"""
    for combo in INPUT_COMBINATIONS:
        model_path = resolve_model_path(combo)
        if model_path and os.path.exists(model_path):
            try:
                # Initialize base model
                base_model = GrindingPredictor(interp=False, input_type=combo)
                
                # Load model state
                checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
                
                # Handle different checkpoint formats
                if 'model_state' in checkpoint:
                    base_model.load_state_dict(checkpoint['model_state'])
                elif 'state_dict' in checkpoint:
                    base_model.load_state_dict(checkpoint['state_dict'])
                else:
                    # Direct state_dict loading
                    base_model.load_state_dict(checkpoint)
                
                base_model.eval()
                
                # Wrap model for XAI compatibility
                model = XAI_ModelWrapper(base_model)
                model.eval()
                
                # Generate SHAP report
                try:
                    # SHAP expects multiple samples - create a batch of 10
                    sample_input_dict = get_sample_input(combo)
                    if sample_input_dict is not None:
                        generate_shap_report(
                            model=model,
                            sample_input=sample_input_dict,
                            output_path=os.path.join(REPORT_PATH, "xai_reports", f"shap_{combo}.png")
                        )
                except Exception as e:
                    print(f"SHAP report generation failed for {combo}: {str(e)}")

                # Generate Grad-CAM report
                if "spec" in combo:
                    try:
                        sample_input_dict = get_sample_input(combo)
                        if sample_input_dict is None:
                            continue
                        
                        if "ae" in combo:
                            target_layer = 'base_model.ae_spec_processor.conv.0'
                            raw_signal = sample_input_dict['spec_ae'][0,0,0].numpy()
                            sample_input = sample_input_dict['spec_ae']
                        elif "vib" in combo:
                            target_layer = 'base_model.vib_spec_processor.conv.0'
                            raw_signal = sample_input_dict['spec_vib'][0,0,0].numpy()
                            sample_input = sample_input_dict['spec_vib']
                        else:
                            continue
                        
                        generate_gradcam_report(
                            model=model,
                            sample_input=sample_input,
                            raw_signal=raw_signal,
                            target_layer=target_layer,
                            output_path=os.path.join(REPORT_PATH, "xai_reports", f"gradcam_{combo}.png")
                        )
                    except Exception as e:
                        print(f"Grad-CAM report generation failed for {combo}: {str(e)}")
            except Exception as e:
                print(f"Error loading model for {combo}: {str(e)}")

def generate_summary_md(accuracy_report):
    """Generate summary markdown report with all content embedded"""
    md_content = "# Smart Grinding Model Report\n\n"
    
    # Include accuracy reports
    md_content += "## Model Performance Reports\n"
    md_content += accuracy_report + "\n\n"
    
    # XAI section
    md_content += "## Explainable AI Reports\n"
    for combo in INPUT_COMBINATIONS:
        md_content += f"### {combo.replace('_', ' ').title()} Inputs\n"
        if os.path.exists(os.path.join(REPORT_PATH, "xai_reports", f"shap_{combo}.png")):
            md_content += f"![SHAP Summary Plot](xai_reports/shap_{combo}.png)\n\n"
        if os.path.exists(os.path.join(REPORT_PATH, "xai_reports", f"gradcam_{combo}.png")):
            md_content += f"![Grad-CAM Visualization](xai_reports/gradcam_{combo}.png)\n\n"
    
    # Save report
    with open(os.path.join(REPORT_PATH, "summary.md"), "w") as f:
        f.write(md_content)

def get_sample_input(input_type="all"):
    """Load a real data sample for XAI analysis."""
    try:
        dataset = get_dataset(input_type=input_type, dataset_mode="classical")
        sample = next(iter(DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=get_collate_fn(input_type))))
        return sample
    except Exception as e:
        print(f"Could not load real data sample: {e}. Using random data instead.")
        return None

def get_metrics_for_combo(input_type):
    """Get detailed statistics for a given input combination."""
    train_mse, test_mse, train_mae, test_mae = [], [], [], []

    pattern = os.path.join(LFS_PATH, "train_his", f"{input_type}*.csv")
    matching_files = glob.glob(pattern)

    if not matching_files:
        print(f"No history files found for {input_type}")
        return []

    for file_path in matching_files:
        try:
            df = pd.read_csv(file_path, index_col=0)
            if "train_mse" in df.columns:
                train_mse.append(df["train_mse"].iloc[-1])
            if "test_mse" in df.columns:
                test_mse.append(df["test_mse"].iloc[-1])
            if "train_mae" in df.columns:
                train_mae.append(df["train_mae"].iloc[-1])
            if "test_mae" in df.columns:
                test_mae.append(df["test_mae"].iloc[-1])
        except Exception:
            pass

    stats = {
        "Input Combination": input_type,
        "Train MSE": pd.Series(train_mse).describe().to_dict(),
        "Test MSE": pd.Series(test_mse).describe().to_dict(),
        "Train MAE": pd.Series(train_mae).describe().to_dict(),
        "Test MAE": pd.Series(test_mae).describe().to_dict()
    }
    return [stats]

if __name__ == "__main__":
    # Ensure report directories exist
    os.makedirs(os.path.join(REPORT_PATH, "accuracy_reports"), exist_ok=True)
    os.makedirs(os.path.join(REPORT_PATH, "xai_reports"), exist_ok=True)
    
    # Generate reports
    accuracy_report = generate_accuracy_report()
    
    # Only generate XAI reports if we have input combinations
    if INPUT_COMBINATIONS:
        try:
            generate_xai_reports()
        except Exception as e:
            print(f"XAI report generation failed: {e}")
    else:
        print("Skipping XAI reports - no input combinations available")
    
    generate_summary_md(accuracy_report)
    
    print("Report generation complete!")
