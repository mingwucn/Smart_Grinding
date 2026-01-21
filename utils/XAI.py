import torch.nn as nn
import numpy as np

def get_conv_layer_names(model):
    conv_layer_names = []
    for name, module in model.named_modules():
        # Check if the module is an instance of nn.Conv2d
        if isinstance(module, nn.Conv2d):
            conv_layer_names.append(name)
    return conv_layer_names

class GradCAM:
    def __init__(self, model, target_layer_name):
        self.model = model
        self.target_layer_name = target_layer_name
        self.gradients = None
        self.activations = None

        # Register hooks to store activations and gradients
        # We need to find the actual module based on its name (e.g., 'S_conv.conv_sequence')
        # Iterate through named modules to find the target
        target_layer = None
        for name, module in self.model.named_modules():
            if name == target_layer_name:
                target_layer = module
                break

        if target_layer is None:
            raise ValueError(f"Target layer '{target_layer_name}' not found in the model.")

        target_layer.register_forward_hook(self._save_activations)
        target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output

    def _save_gradients(self, module, grad_in, grad_out):
        # grad_out[0] contains the gradient w.r.t. the output of the module
        self.gradients = grad_out[0]

    def __call__(self, input_batch, target_output_idx=None):
        """
        Generates Grad-CAM heatmap for a given input.
        Args:
            input_batch (dict): Dictionary of input tensors, e.g., {"S": tensor_S, ...}.
                                Expects batch size 1 for visualization.
            target_output_idx (int, optional): For regression, which output neuron's gradient
                                               to use. If None, sums gradients of all outputs.
        Returns:
            torch.Tensor: Normalized heatmap.
        """
        self.model.eval() # Set model to evaluation mode

        # Ensure input_batch is on the correct device
        input_on_device = {k: v.to(next(self.model.parameters()).device) for k, v in input_batch.items() if k != "labels"}
        
        # We need to make sure the input for Grad-CAM has requires_grad=True
        # This is often already true for the data loader output, but good to be explicit
        if "S" in input_on_device:
            input_on_device["S"].requires_grad_(True)

        # Forward pass
        outputs = self.model(input_on_device)
        # predictions = outputs["final_prediction"]
        predictions = outputs

        if predictions.dim() == 1: # If output_size=1, predictions might be (batch_size,)
            predictions = predictions.unsqueeze(1) # Make it (batch_size, 1)

        # Zero gradients
        self.model.zero_grad()

        # Backward pass based on the target output neuron
        if target_output_idx is not None:
            # Backpropagate through a specific output neuron
            if target_output_idx >= predictions.size(1) or target_output_idx < 0:
                raise ValueError(f"target_output_idx {target_output_idx} out of range for predictions of size {predictions.size(1)}")
            target_score = predictions[:, target_output_idx]
        else:
            # Backpropagate through the sum of all outputs (for overall importance)
            target_score = predictions.sum()

        target_score.backward(retain_graph=True) # retain_graph=True if you need to do multiple backward passes

        # Get feature map and gradients
        guided_gradients = self.gradients.cpu().data.numpy()[0] # [0] for batch size 1
        activations = self.activations.cpu().data.numpy()[0]    # [0] for batch size 1

        # Global average pooling of gradients
        weights = np.mean(guided_gradients, axis=(1, 2)) # Average over H and W

        # Create heatmap
        heatmap = np.zeros(activations.shape[1:], dtype=np.float32) # (C, H, W) -> (H, W) for first channel's shape
        for i, w in enumerate(weights):
            heatmap += w * activations[i] # Weighted sum of activations

        # ReLU for heatmap (important: only positive influences)
        heatmap = np.maximum(heatmap, 0)

        # Normalize heatmap to [0, 1]
        heatmap = heatmap / (np.max(heatmap) + 1e-8) # Add small epsilon to avoid division by zero

        return heatmap
