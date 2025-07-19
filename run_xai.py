import os
import torch
import matplotlib.pyplot as plt
from MyModels import GrindingPredictor
from utils.XAI import GradCAM, get_conv_layer_names
from XAI_ModelWrapper import XAI_ModelWrapper

# Configuration
REPORT_PATH = "report"
SNAPSHOT_DIR = "snapshots"
LFS_PATH = "/mnt/c/Users/Public/Documents/IntermediateData/SmartGrinding"

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

def run_grad_cam(input_type="ae_spec"):
    """Run Grad-CAM for a given input type."""
    model_path = resolve_model_path(input_type)
    if not model_path or not os.path.exists(model_path):
        print(f"Model not found for input type: {input_type}")
        return

    # Load model
    base_model = GrindingPredictor(interp=False, input_type=input_type)
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    
    if 'model_state' in checkpoint:
        base_model.load_state_dict(checkpoint['model_state'])
    elif 'state_dict' in checkpoint:
        base_model.load_state_dict(checkpoint['state_dict'])
    else:
        base_model.load_state_dict(checkpoint)
    
    base_model.eval()
    model = XAI_ModelWrapper(base_model)
    model.eval()

    # Get sample input
    sample_input = torch.randn(1, 5, 128, 128)  # (batch, channels, height, width)

    # Get convolutional layers and select the correct one
    conv_layers = get_conv_layer_names(model)
    print(f"Found convolutional layers: {conv_layers}")
    if not conv_layers:
        print("No convolutional layers found.")
        return

    if "ae" in input_type:
        target_layer = 'base_model.ae_spec_processor.conv.0'
    elif "vib" in input_type:
        target_layer = 'base_model.vib_spec_processor.conv.0'
    else:
        print(f"Could not determine target layer for input_type '{input_type}'")
        return
    
    print(f"Using target layer: {target_layer}")

    # Run Grad-CAM
    cam = GradCAM(model, target_layer)
    heatmap = cam(sample_input)

    # Save heatmap
    output_path = os.path.join(REPORT_PATH, "xai_reports", f"gradcam_{input_type}.png")
    plt.imshow(heatmap, cmap='viridis')
    plt.colorbar()
    plt.savefig(output_path)
    plt.close()
    print(f"Grad-CAM heatmap saved to {output_path}")

if __name__ == "__main__":
    os.makedirs(os.path.join(REPORT_PATH, "xai_reports"), exist_ok=True)
    run_grad_cam()
