import os
import numpy as np
import matplotlib.pyplot as plt
import shap
from shap import Explanation
import torch
from .base_explainer import BaseExplainer

class SHAPExplainer(BaseExplainer):
    def __init__(self, model):
        super().__init__(model)
        self.explainer = None
        
    def explain(self, input_dict, target_output_idx=None):
        # Focus on spectrogram explanations
        spec_key = 'spec_ae' if 'spec_ae' in input_dict else 'spec_vib'
        if not spec_key:
            return None
            
        # Prepare background data
        background = torch.mean(input_dict[spec_key], dim=0, keepdim=True)
        
        # Model wrapper for SHAP
        class ModelWrapper(torch.nn.Module):
            def __init__(self, model, spec_key):
                super().__init__()
                self.model = model
                self.spec_key = spec_key
                
            def forward(self, x):
                input_dict = {
                    self.spec_key: x,
                    'features_pp': torch.zeros(x.shape[0], 1, 3, device=x.device),
                    'features_ae': torch.zeros(x.shape[0], 1, 4, device=x.device),
                    'features_vib': torch.zeros(x.shape[0], 1, 4, device=x.device)
                }
                return self.model(input_dict)
        
        # Create SHAP explainer
        wrapped_model = ModelWrapper(self.model, spec_key)
        self.explainer = shap.DeepExplainer(wrapped_model, background)
        
        # Calculate SHAP values
        return self.explainer.shap_values(input_dict[spec_key])
    
    def visualize(self, explanation, input_data):
        if explanation is None:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.text(0.5, 0.5, 'No SHAP explanation available', 
                    horizontalalignment='center', verticalalignment='center')
            return fig
            
        fig = plt.figure(figsize=(15, 12), layout='constrained')
        gs = fig.add_gridspec(3, 2)
        
        # Get spectrogram data
        spec_key = 'spec_ae' if 'spec_ae' in input_data else 'spec_vib'
        spec_data = input_data[spec_key][0,0].detach().cpu().numpy()
        magnitude = np.sqrt(spec_data[0]**2 + spec_data[1]**2)
        
        # Input spectrogram
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.imshow(magnitude.T, aspect='auto', origin='lower', cmap='viridis')
        ax1.set_title(f'Input Spectrogram ({spec_key})')
        plt.colorbar(im1, ax=ax1)
        
        # SHAP feature importance heatmap
        ax2 = fig.add_subplot(gs[0, 1])
        shap_magnitude = np.abs(explanation[0]).sum(axis=0).squeeze()
        if len(shap_magnitude.shape) == 1:
            shap_magnitude = shap_magnitude.reshape(1, -1)
        im2 = ax2.imshow(shap_magnitude.T, aspect='auto', origin='lower', cmap='viridis')
        ax2.set_title('SHAP Feature Importance')
        plt.colorbar(im2, ax=ax2)
        
        # Top 20 important features
        ax3 = fig.add_subplot(gs[1, :])
        flat_shap = explanation[0].reshape(-1)
        feature_names = [f'bin_{i//47}_{i%47}' for i in range(flat_shap.size)]
        shap.summary_plot(
            explanation[0].reshape(1, -1), 
            feature_names=feature_names,
            plot_type="bar",
            show=False,
            max_display=20
        )
        plt.savefig('temp_summary.png')
        plt.close()
        ax3.imshow(plt.imread('temp_summary.png'))
        ax3.axis('off')
        ax3.set_title('Top 20 Important Features')
        if os.path.exists('temp_summary.png'):
            os.remove('temp_summary.png')
        
        # Dependence plot
        ax4 = fig.add_subplot(gs[2, 0])
        if len(explanation) > 0 and explanation[0].size > 0:
            abs_shap = np.abs(explanation[0][0])
            max_idx = np.unravel_index(np.argmax(abs_shap), abs_shap.shape)
            bin_values = input_data[spec_key][0,0,:,max_idx[1], max_idx[2]].detach().cpu().numpy()
            shap_values = explanation[0][0,:,max_idx[1],max_idx[2]]
            
            for ch in range(2):
                ax4.scatter(bin_values[ch], shap_values[ch], label=f'Channel {ch}')
            ax4.axhline(0, color='gray', linestyle='--', alpha=0.5)
            ax4.set_xlabel(f'Bin Value (Time: {max_idx[1]}, Freq: {max_idx[2]})')
            ax4.set_ylabel('SHAP Value')
            ax4.set_title('Dependence Plot')
            ax4.legend()
        else:
            ax4.text(0.5, 0.5, 'No SHAP values for dependence plot', 
                     horizontalalignment='center', verticalalignment='center')
        
        return fig
    
    def metrics(self, explanation):
        if explanation is None:
            return {}
        shap_values = explanation[0]
        return {
            "max_shap": float(np.max(shap_values)),
            "min_shap": float(np.min(shap_values)),
            "mean_shap": float(np.mean(shap_values)),
            "std_shap": float(np.std(shap_values)),
            "positive_ratio": float(np.mean(shap_values > 0))
        }
