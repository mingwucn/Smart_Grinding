import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.xai.shap import SHAPExplainer

# Simple test model
class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=3, padding=1)
        self.fc = nn.Linear(10*10, 1)
        self.flatten = nn.Flatten()
        
    def forward(self, input_dict):
        # Extract spectrogram from input dictionary
        x = input_dict['spec_ae']
        # Remove sequence dimension (batch, seq=1, channels, time, freq)
        x = x.squeeze(1)  # now (batch, channels, time, freq)
        x = self.conv(x)
        x = self.flatten(x)
        return self.fc(x)

# Test SHAP
def test_shap():
    model = TestModel()
    
    # Create test input
    input_data = {
        'spec_ae': torch.randn(1, 1, 2, 10, 10)  # (batch, seq, channels, time, freq)
    }
    
    # Initialize explainer
    explainer = SHAPExplainer(model)
    
    # Run explanation
    shap_values = explainer.explain(input_data)
    
    # Verify output
    # SHAP returns a list of arrays for each output class
    # Since our model has one output, we take the first element
    assert len(shap_values) == 1, "SHAP should return one array per output class"
    shap_array = shap_values[0]
    assert shap_array.shape == (1, 2, 10, 10, 1), f"SHAP values shape {shap_array.shape} incorrect"
    assert not np.isnan(shap_array).any(), "SHAP values contain NaN values"
    print("SHAP test passed!")

if __name__ == "__main__":
    test_shap()
