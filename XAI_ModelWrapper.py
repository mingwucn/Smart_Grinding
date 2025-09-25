import torch
import torch.nn as nn
from MyModels import GrindingPredictor

class XAI_ModelWrapper(nn.Module):
    def __init__(self, base_model, input_type):
        super().__init__()
        self.base_model = base_model
        self.input_type = input_type
        
    def forward(self, batch):
        # Pass the full batch dictionary directly to the base model
        return self.base_model(batch)
    
    def get_expected_input_shape(self):
        """Return the expected input shape as (channels, height, width)"""
        return (3, 128, 128)  # Matches the sample input dimensions
