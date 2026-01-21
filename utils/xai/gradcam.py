import torch
import matplotlib.pyplot as plt
import numpy as np
from .base_explainer import BaseExplainer

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        
        # Register hooks
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def __call__(self, input_dict, target_output_idx=None):
        # Forward pass
        output = self.model(input_dict)
        
        # Zero gradients
        self.model.zero_grad()
        
        # If target output not specified, use the predicted output
        if target_output_idx is None:
            target_output_idx = torch.argmax(output, dim=1)
        
        # Create one-hot tensor for backward
        one_hot = torch.zeros_like(output)
        one_hot[0, target_output_idx] = 1
        
        # Backward pass
        output.backward(gradient=one_hot)
        
        # Global average pooling of gradients
        pooled_gradients = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        
        # Weight the activations by pooled gradients
        weighted_activations = self.activations * pooled_gradients
        
        # Average over channels
        heatmap = torch.mean(weighted_activations, dim=1, keepdim=True)
        
        # ReLU and normalize
        heatmap = torch.relu(heatmap)
        heatmap = heatmap / torch.max(heatmap)
        
        return heatmap.squeeze().cpu().numpy()

class GradCAMExplainer(BaseExplainer):
    def __init__(self, model, target_layer):
        super().__init__(model)
        self.cam = GradCAM(model, target_layer)
        
    def explain(self, input_dict, target_output_idx=None):
        return self.cam(input_dict, target_output_idx)
    
    def visualize(self, heatmap, input_data):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot input spectrogram
        spec_key = 'spec_ae' if 'spec_ae' in input_data else 'spec_vib'
        if spec_key in input_data:
            spec_data = input_data[spec_key][0,0].detach().cpu()
            magnitude = torch.sqrt(spec_data[0]**2 + spec_data[1]**2).numpy()
            im1 = ax1.imshow(magnitude.T, aspect='auto', origin='lower', cmap='viridis')
            ax1.set_title(f'Input Spectrogram ({spec_key})')
            plt.colorbar(im1, ax=ax1)
        else:
            ax1.text(0.5, 0.5, 'Spectrogram Not Available', 
                     horizontalalignment='center', verticalalignment='center',
                     transform=ax1.transAxes, fontsize=12)
            ax1.set_title('Input Spectrogram')
            ax1.axis('off')
        
        # Plot Grad-CAM heatmap
        im2 = ax2.imshow(heatmap, aspect='auto', cmap='viridis')
        ax2.set_title('Grad-CAM Heatmap')
        plt.colorbar(im2, ax=ax2)
        
        return fig
    
    def metrics(self, heatmap):
        return {
            "max_value": float(np.max(heatmap)),
            "min_value": float(np.min(heatmap)),
            "mean_value": float(np.mean(heatmap)),
            "std_dev": float(np.std(heatmap))
        }
