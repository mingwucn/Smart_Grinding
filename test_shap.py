import torch
import numpy as np
from utils.xai_manager import SHAPExplainer

class MockModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 8, kernel_size=3, padding=1)
        self.flatten = torch.nn.Flatten()
        self.fc = torch.nn.Linear(8*300*47, 1)
        
    def forward(self, input_dict):
        x = input_dict['spec_ae']
        # Remove sequence dimension (batch, seq, channels, time, freq) -> (batch, channels, time, freq)
        x = x.squeeze(1)
        x = self.conv(x)
        x = torch.relu(x)
        x = self.flatten(x)
        return self.fc(x)

# Create model and explainer
model = MockModel()
explainer = SHAPExplainer(model)

# Create input dictionary
input_dict = {
    'spec_ae': torch.randn(1, 1, 2, 300, 47)  # Batch, seq, channels, time, freq
}

# Generate explanation
explanation = explainer.explain(input_dict)

# Print results
print(f"SHAP explanation type: {type(explanation)}")
print(f"SHAP explanation shape: {explanation[0].shape}")

# Basic validation
if explanation is not None and explanation[0].size > 0:
    print("✅ SHAP implementation works!")
    print(f"SHAP values range: {np.min(explanation[0])} to {np.max(explanation[0])}")
else:
    print("❌ SHAP implementation failed")
