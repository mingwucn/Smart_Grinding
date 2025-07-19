import torch
import torch.nn as nn
from MyModels import GrindingPredictor

class XAI_ModelWrapper(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        
    def forward(self, x):
        return self.base_model(x)
    
    def get_expected_input_shape(self):
        """Return the expected input shape as (channels, height, width)"""
        return (3, 128, 128)  # Matches the sample input dimensions
