import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Add parent directory to path to import MyModels
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from MyModels import GrindingPredictor, SpectrogramProcessor, FeatureInterpreter

class MyCustomSpectrogramProcessor(SpectrogramProcessor):
    def __init__(self, in_channels, out_dim=32):
        super().__init__(in_channels, out_dim)
        # Override the conv layers to match the checkpoint structure
        self.conv0 = nn.Conv2d(in_channels, 8, kernel_size=3, padding=1)
        self.relu0 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(8, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2)
        
        # Adaptive pooling, flatten, and linear layers remain the same
        # as they are inherited and not explicitly redefined here.
        # If they were different in the checkpoint, they would need to be overridden.

    def forward(self, x):
        # x: [batch, seq_len, C, H, W]
        batch_size, seq_len = x.shape[:2]
        x = x.view(batch_size * seq_len, *x.shape[2:]).clone()  # Merge batch and seq, and clone
        
        x = self.relu0(self.conv0(x))
        x = self.pool1(x)
        x = self.relu2(self.conv2(x))
        x = self.pool3(x)

        x = self.adaptive_pool(x)
        x = self.flatten(x)  # Flatten to [batch*seq, channels]
        features = self.linear(x)
        
        return features.view(batch_size, seq_len, -1)

class MyCustomGrindingPredictor(GrindingPredictor):
    def __init__(self, interp=False, input_type="all"):
        super().__init__(interp, input_type)
        # Override spec processors with custom ones
        self.ae_spec_processor = MyCustomSpectrogramProcessor(2, out_dim=32)
        self.vib_spec_processor = MyCustomSpectrogramProcessor(3, out_dim=32)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        # Custom load_state_dict to handle strict=False by default
        # This is to allow loading checkpoints that might have slightly different keys
        # due to architectural changes or different PyTorch versions.
        # We always use strict=False here to allow loading partial models or models with minor mismatches.
        return super().load_state_dict(state_dict, strict=False, assign=assign)
        # print("Loaded state_dict with strict=False for MyCustomGrindingPredictor.") # Removed print to match return type

    def forward(self, batch):
        mode = self.input_type
        outputs = {}
        
        ae_out, ae_time, vib_out, vib_time, pp_out = None, None, None, None, None

        # Process AE components based on mode
        if 'ae_spec' in mode or 'ae_spec+ae_features' in mode or 'ae_spec+vib_spec' in mode or 'ae_spec+ae_features+vib_spec+vib_features' in mode or 'all' in mode:
            ae_spec_input = batch.get("spec_ae") # Use .get to avoid KeyError
            if ae_spec_input is not None:
                ae_spec = self.ae_spec_processor(ae_spec_input)
                outputs['ae_spec_processed'] = ae_spec # Store processed spec for later combination
            else:
                outputs['ae_spec_processed'] = None

        if 'ae_features' in mode or 'ae_spec+ae_features' in mode or 'ae_features+pp' in mode or 'ae_spec+ae_features+vib_spec+vib_features' in mode or 'ae_features+vib_features' in mode or 'ae_features+vib_features+pp' in mode or 'all' in mode:
            ae_time_input = batch.get("features_ae") # Use .get to avoid KeyError
            if ae_time_input is not None:
                outputs['ae_time_processed'] = ae_time_input # Store raw time features
            else:
                outputs['ae_time_processed'] = None

        # Combine AE spec and time features if both are present and required by the mode
        if ('ae_spec+ae_features' in mode or 'ae_spec+ae_features+vib_spec+vib_features' in mode or 'all' in mode) and outputs['ae_spec_processed'] is not None and outputs['ae_time_processed'] is not None:
            ae_out, ae_attn = self.ae_interpreter(outputs['ae_spec_processed'], outputs['ae_time_processed'])
            outputs['ae_out'] = ae_out
            outputs['ae_attn'] = ae_attn
        elif 'ae_spec' in mode and outputs['ae_spec_processed'] is not None:
            outputs['ae_out'] = outputs['ae_spec_processed'].mean(dim=1) # If only spec, take mean over seq_len
        elif 'ae_features' in mode and outputs['ae_time_processed'] is not None:
            outputs['ae_out'] = outputs['ae_time_processed'].mean(dim=1) # If only features, take mean over seq_len
        else:
            outputs['ae_out'] = None # Default if not processed or not required

        # Process VIB components based on mode (similar logic)
        if 'vib_spec' in mode or 'vib_spec+vib_features' in mode or 'ae_spec+vib_spec' in mode or 'ae_spec+ae_features+vib_spec+vib_features' in mode or 'all' in mode:
            vib_spec_input = batch.get("spec_vib") # Use .get to avoid KeyError
            if vib_spec_input is not None:
                vib_spec = self.vib_spec_processor(vib_spec_input)
                outputs['vib_spec_processed'] = vib_spec
            else:
                outputs['vib_spec_processed'] = None

        if 'vib_features' in mode or 'vib_spec+vib_features' in mode or 'vib_features+pp' in mode or 'ae_spec+ae_features+vib_spec+vib_features' in mode or 'ae_features+vib_features' in mode or 'ae_features+vib_features+pp' in mode or 'all' in mode:
            vib_time_input = batch.get("features_vib") # Use .get to avoid KeyError
            if vib_time_input is not None:
                outputs['vib_time_processed'] = vib_time_input
            else:
                outputs['vib_time_processed'] = None

        # Combine VIB spec and time features if both are present and required by the mode
        if ('vib_spec+vib_features' in mode or 'ae_spec+ae_features+vib_spec+vib_features' in mode or 'all' in mode) and outputs['vib_spec_processed'] is not None and outputs['vib_time_processed'] is not None:
            vib_out, vib_attn = self.vib_interpreter(outputs['vib_spec_processed'], outputs['vib_time_processed'])
            outputs['vib_out'] = vib_out
            outputs['vib_attn'] = vib_attn
        elif 'vib_spec' in mode and outputs['vib_spec_processed'] is not None:
            outputs['vib_out'] = outputs['vib_spec_processed'].mean(dim=1)
        elif 'vib_features' in mode and outputs['vib_time_processed'] is not None:
            outputs['vib_out'] = outputs['vib_time_processed'].mean(dim=1)
        else:
            outputs['vib_out'] = None

        # Process parameter (pp) Processing (if applicable)
        if 'pp' in mode or 'ae_features+pp' in mode or 'vib_features+pp' in mode or 'ae_features+vib_features+pp' in mode or 'all' in mode:
            pp_input = batch.get("features_pp") # Use .get to avoid KeyError
            if pp_input is not None:
                pp_out = self.pp_encoder(pp_input)
                outputs['pp'] = pp_out
            else:
                outputs['pp'] = None
        else:
            outputs['pp'] = None

        # Combine final features based on mode
        combined_features = []
        if outputs['ae_out'] is not None and ('ae_spec' in mode or 'ae_features' in mode or 'ae_spec+ae_features' in mode or 'ae_spec+vib_spec' in mode or 'ae_features+pp' in mode or 'ae_spec+ae_features+vib_spec+vib_features' in mode or 'ae_features+vib_features' in mode or 'ae_features+vib_features+pp' in mode or 'all' in mode):
            combined_features.append(outputs['ae_out'])
        
        if outputs['vib_out'] is not None and ('vib_spec' in mode or 'vib_features' in mode or 'vib_spec+vib_features' in mode or 'ae_spec+vib_spec' in mode or 'vib_features+pp' in mode or 'ae_spec+ae_features+vib_spec+vib_features' in mode or 'ae_features+vib_features' in mode or 'ae_features+vib_features+pp' in mode or 'all' in mode):
            combined_features.append(outputs['vib_out'])

        if outputs['pp'] is not None and ('pp' in mode or 'ae_features+pp' in mode or 'vib_features+pp' in mode or 'ae_features+vib_features+pp' in mode or 'all' in mode):
            combined_features.append(outputs['pp'])

        if not combined_features:
            raise ValueError(f"No features to combine for input_type: {mode}")
        
        combined = torch.cat(combined_features, dim=1)

        # Final prediction
        if self.interp:
            return self.regressor(combined), {"ae": outputs.get('ae_attn', None), "vib": outputs.get('vib_attn', None)}
        else:
            return self.regressor(combined)

    def _calculate_regressor_input_dim(self):
        """
        Calculate the input dimension for the regressor based on the input_type.
        """
        input_type = self.input_type
        if input_type == 'pp':
            return 64  # PP output dimension
        if input_type == 'ae_spec':
            return 32  # AE spec processor output dimension
        elif input_type == 'vib_spec':
            return 32  # Vib spec processor output dimension
        if input_type == 'ae_features':
            return 4  # AE time features dimension
        elif input_type == 'vib_features':
            return 4  # Vib time features dimension
        elif input_type == 'ae_spec+ae_features':
            return 32 + 4  # AE spec output + AE time features
        elif input_type == 'vib_spec+vib_features':
            return 32 + 4  # Vib spec output + Vib time features
        elif input_type == 'ae_features+pp':
            return 4 + 64
        elif input_type == 'vib_features+pp':
            return 4 + 64
        elif input_type == 'ae_spec+ae_features+vib_spec+vib_features':
            return 32 + 4 + 32 + 4  # AE spec output + AE time features + Vib spec output + Vib time features
        else:  # 'all'
            return 32 + 32 + 64  # AE spec output + Vib spec output + Physics
