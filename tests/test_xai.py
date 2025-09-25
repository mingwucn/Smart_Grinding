import os
import torch
import unittest
from unittest.mock import MagicMock, patch
from utils.XAI import GradCAM, get_conv_layer_names
from XAI_ModelWrapper import XAI_ModelWrapper
from MyModels import GrindingPredictor

class TestXAI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a real model instance for testing
        cls.real_model = GrindingPredictor(interp=False, input_type="ae_spec")
        
        # Add parameters to the model
        cls.real_model.conv_test = torch.nn.Conv2d(5, 3, kernel_size=3)
        
        # Create wrapper
        cls.wrapped_model = XAI_ModelWrapper(cls.real_model)
        
        # Sample input tensor
        cls.sample_input = torch.randn(1, 5, 128, 128)
        
        # Mock convolutional layers
        cls.conv_layers = ['conv1', 'conv2', 'conv3']
        
        # Patch the GradCAM class to avoid device issues
        cls.patcher = patch('utils.XAI.GradCAM', autospec=True)
        cls.mock_gradcam = cls.patcher.start()
        
    @classmethod
    def tearDownClass(cls):
        cls.patcher.stop()
        
    def test_gradcam_heatmap_generation(self):
        """Test that Grad-CAM is called with correct parameters"""
        # Configure mock GradCAM
        mock_cam_instance = self.mock_gradcam.return_value
        mock_cam_instance.return_value = torch.zeros(128, 128)
        
        # Create Grad-CAM instance
        cam = GradCAM(self.wrapped_model, 'conv_test')
        
        # Generate heatmap
        cam(self.sample_input)
        
        # Verify Grad-CAM was called with correct parameters
        self.mock_gradcam.assert_called_once()
        mock_cam_instance.assert_called_once_with(self.sample_input)
        
    def test_conv_layer_detection(self):
        """Test that convolutional layers are correctly identified"""
        # Mock the model's named_modules method
        self.wrapped_model.named_modules = MagicMock(return_value=[
            ('conv1', MagicMock(__class__=torch.nn.Conv2d)),
            ('pool1', MagicMock(__class__=torch.nn.MaxPool2d)),
            ('conv2', MagicMock(__class__=torch.nn.Conv2d)),
            ('fc', MagicMock(__class__=torch.nn.Linear))
        ])
        
        # Get convolutional layers
        conv_layers = get_conv_layer_names(self.wrapped_model)
        
        # Verify correct layers are detected
        self.assertEqual(conv_layers, ['conv1', 'conv2'])
        
    def test_model_wrapper_forward(self):
        """Test that model wrapper correctly passes through forward calls"""
        # Call wrapped model
        output = self.wrapped_model(self.sample_input)
        
        # Verify output shape
        self.assertEqual(output.shape, (1, 2))
        
        # Verify the input shape to the base model matches
        self.assertEqual(self.real_model.call_args[0][0].shape, self.sample_input.shape)

if __name__ == '__main__':
    unittest.main()
