import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.xai.integrated_gradients import IntegratedGradientsExplainer

# Simple test model
class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(5, 3)
        self.fc2 = nn.Linear(3, 1)
        
    def forward(self, input_dict):
        # Extract features from input dictionary
        x = input_dict['features']
        x = self.fc1(x)
        return self.fc2(x)

# Test Integrated Gradients
def test_integrated_gradients():
    model = TestModel()
    
    # Create test input
    input_data = {
        'features': torch.randn(1, 5)  # (batch, features)
    }
    
    # Initialize explainer
    explainer = IntegratedGradientsExplainer(model)
    
    # Run explanation
    attribution = explainer.explain(input_data)
    
    # Verify output
    assert 'features' in attribution, "Features attribution missing"
    features_attribution = attribution['features']
    assert features_attribution.shape == (1, 5), f"Attribution shape {features_attribution.shape} incorrect"
    assert not np.isnan(features_attribution.numpy()).any(), "Attribution contains NaN values"
    print("Integrated Gradients test passed!")

if __name__ == "__main__":
    test_integrated_gradients()
