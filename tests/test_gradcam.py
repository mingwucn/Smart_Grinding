import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.xai.gradcam import GradCAMExplainer

# Simple test model
class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=3, padding=1)
        self.fc = nn.Linear(300*47, 1)
        self.flatten = nn.Flatten()
        
    def forward(self, input_dict):
        # Extract spectrogram from input dictionary
        x = input_dict['spec_ae']
        # Remove sequence dimension (batch, seq=1, channels, time, freq)
        x = x.squeeze(1)  # now (batch, channels, time, freq)
        x = self.conv(x)
        x = self.flatten(x)
        return self.fc(x)
# Test GradCAM
def test_gradcam():
    model = TestModel()
    target_layer = model.conv
    
    # Create test input
    input_data = {
        'spec_ae': torch.randn(1, 1, 2, 300, 47)  # (batch, seq, channels, time, freq)
    }
    
    # Initialize explainer
    explainer = GradCAMExplainer(model, target_layer)
    
    # Run explanation
    heatmap = explainer.explain(input_data)
    
    # Verify output
    assert heatmap.shape == (300, 47), "Heatmap shape incorrect"
    assert not np.isnan(heatmap).any(), "Heatmap contains NaN values"
    print("GradCAM test passed!")

if __name__ == "__main__":
    test_gradcam()
