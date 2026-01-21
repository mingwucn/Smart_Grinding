import torch
import numpy as np
import matplotlib.pyplot as plt
from .base_explainer import BaseExplainer

class IntegratedGradients:
    def __init__(self, model):
        self.model = model
        self.model.eval()
        
    def attribute(self, input_dict, target_output_idx=None, n_steps=50):
        # Create baseline (zero input)
        baseline = {k: torch.zeros_like(v) for k, v in input_dict.items()}
        
        # Generate alphas
        alphas = torch.linspace(0, 1, n_steps)
        
        # Initialize gradients dictionary
        gradients = {k: torch.zeros_like(v) for k, v in input_dict.items()}
        
        # Compute integrated gradients
        for alpha in alphas:
            # Create interpolated input
            interpolated_input = {}
            for key in input_dict:
                interpolated_input[key] = baseline[key] + alpha * (input_dict[key] - baseline[key])
            
            # Set requires_grad for all inputs
            for key in interpolated_input:
                interpolated_input[key].requires_grad = True
                
            # Forward pass
            output = self.model(interpolated_input)
            
            # If target_output_idx not specified, use the predicted class
            if target_output_idx is None:
                target_output_idx = torch.argmax(output, dim=1)
            
            # Create one-hot tensor for backward
            one_hot = torch.zeros_like(output)
            one_hot[0, target_output_idx] = 1
            
            # Backward pass
            output.backward(gradient=one_hot)
            
        # Accumulate gradients (only if gradient exists)
        for key in interpolated_input:
            grad = interpolated_input[key].grad
            if grad is not None:
                gradients[key] += grad
            
            # Zero gradients
            self.model.zero_grad()
        
        # Compute integrated gradients
        integrated_gradients = {}
        for key in gradients:
            gradients[key] /= n_steps
            integrated_gradients[key] = (input_dict[key] - baseline[key]) * gradients[key]
            
        return integrated_gradients

class IntegratedGradientsExplainer(BaseExplainer):
    def __init__(self, model):
        super().__init__(model)
        self.ig = IntegratedGradients(model)
        
    def explain(self, input_dict, target_output_idx=None):
        return self.ig.attribute(input_dict, target_output_idx)
    
    def visualize(self, explanation, input_data):
        fig, ax = plt.subplots(figsize=(10, 5))
        
        if 'spec_ae' in explanation:
            spec_data = explanation['spec_ae'][0,0].abs().mean(dim=0).detach().numpy()
            im = ax.imshow(spec_data, aspect='auto', cmap='viridis')
            ax.set_title('Integrated Gradients - AE Spec Importance')
        elif 'spec_vib' in explanation:
            spec_data = explanation['spec_vib'][0,0].abs().mean(dim=0).detach().numpy()
            im = ax.imshow(spec_data, aspect='auto', cmap='viridis')
            ax.set_title('Integrated Gradients - VIB Spec Importance')
        else:
            ax.text(0.5, 0.5, 'No spectrogram data available', 
                    horizontalalignment='center', verticalalignment='center')
        
        plt.colorbar(im, ax=ax)
        return fig
    
    def metrics(self, explanation):
        metrics = {}
        for key, tensor in explanation.items():
            if isinstance(tensor, torch.Tensor):
                data = tensor.detach().cpu().numpy()
                metrics[f"{key}_max"] = float(np.max(data))
                metrics[f"{key}_min"] = float(np.min(data))
                metrics[f"{key}_mean"] = float(np.mean(data))
                metrics[f"{key}_std"] = float(np.std(data))
        return metrics
