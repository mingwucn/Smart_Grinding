import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureInterpreter(nn.Module):
    def __init__(self, spec_feat_dim, time_feat_dim, hidden_dim=64):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(spec_feat_dim + time_feat_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, spec_feat_dim + time_feat_dim),
            nn.Softmax(dim=-1),
        )
        self.feature_processor = nn.Linear(spec_feat_dim + time_feat_dim, hidden_dim)

    def forward(self, spec_feats, time_feats):
        """
        spec_feats: [batch, seq_len, spec_features]
        time_feats: [batch, seq_len, time_features]
        """
        combined = torch.cat([spec_feats, time_feats], dim=-1)
        attn_weights = self.attention(combined.mean(dim=1))
        weighted = combined * attn_weights.unsqueeze(1)
        processed = F.relu(self.feature_processor(weighted))
        return processed.mean(dim=1), attn_weights


class GrindingPredictor(nn.Module):
    def __init__(self,interp=False, input_type="all"):
        super().__init__()
        self.input_type = input_type
        # AE Pathway (2 spec channels + 4 time features)
        self.ae_spec_processor = SpectrogramProcessor(2, out_dim=32)
        self.interp = interp

        # Vib Pathway (3 spec channels + 4 time features)
        self.vib_spec_processor = SpectrogramProcessor(3, out_dim=32)

        self.ae_interpreter = FeatureInterpreter(spec_feat_dim=32, time_feat_dim=4)
        self.vib_interpreter = FeatureInterpreter(spec_feat_dim=32, time_feat_dim=4)

        # Physics Processor
        self.pp_encoder = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(), nn.LayerNorm(64)
        )

        self.regressor_input_dim = self._calculate_regressor_input_dim()

        # Final Fusion
        self.regressor = nn.Sequential(
            nn.Linear(self.regressor_input_dim, 128),  # 64(ae) + 64(vib) + 64(physics)
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, batch):
        mode = self.input_type
        outputs = {}

        # AE Processing (if applicable)
        if 'ae' in mode or 'all' in mode:
            # Extract tensor from dictionary if needed
            ae_spec_input = batch["spec_ae"] if not isinstance(batch["spec_ae"], dict) else batch["spec_ae"]['spec_ae']
            ae_spec = self.ae_spec_processor(ae_spec_input)  # [batch, seq_len, 32]
            ae_time = batch["features_ae"]  # [batch, seq_len, 4]
            ae_out, ae_attn = self.ae_interpreter(ae_spec, ae_time)
            outputs['ae_out'] = ae_out
            outputs['ae_attn'] = ae_attn

        # Vib Processing (if applicable)
        if 'vib' in mode or 'all' in mode:
            # Extract tensor from dictionary if needed
            vib_spec_input = batch["spec_vib"] if not isinstance(batch["spec_vib"], dict) else batch["spec_vib"]['spec_vib']
            vib_spec = self.vib_spec_processor(vib_spec_input)  # [batch, seq_len, 32]
            vib_time = batch["features_vib"]  # [batch, seq_len, 4]
            vib_out, vib_attn = self.vib_interpreter(vib_spec, vib_time)
            outputs['vib_out'] = vib_out
            outputs['vib_attn'] = vib_attn

        # Process parameter (pp) Processing (if applicable)
        if 'all' in mode or 'pp' in mode:
            pp = self.pp_encoder(batch["features_pp"])
            outputs['pp'] = pp

        # Combine features based on mode
        if mode == 'pp':
            combined = outputs['pp']
        elif mode == 'ae_features':
            combined = ae_time.mean(dim=1)
        elif mode == 'vib_features':
            combined = vib_time.mean(dim=1)
        elif mode == 'ae_spec':
            combined = outputs['ae_out']
        elif mode == 'vib_spec':
            combined = outputs['vib_out']
        elif mode == 'ae_spec+ae_features':
            combined = torch.cat([outputs['ae_out'], ae_time.mean(dim=1)], dim=1)
        elif mode == 'vib_spec+vib_features':
            combined = torch.cat([outputs['vib_out'], vib_time.mean(dim=1)], dim=1)
        elif mode == 'ae_spec+vib_spec':
            combined = torch.cat([outputs['ae_out'], outputs['vib_out']], dim=1)
        elif mode == 'ae_features+pp':
            combined = torch.cat([ae_time.mean(dim=1),outputs['pp']], dim=1)
        elif mode == 'vib_features+pp':
            combined = torch.cat([vib_time.mean(dim=1),outputs['pp']], dim=1)
        elif mode == 'ae_spec+ae_features+vib_spec+vib_features':
            combined = torch.cat([outputs['ae_out'], ae_time.mean(dim=1), outputs['vib_out'], vib_time.mean(dim=1)], dim=1)
        elif mode == 'ae_features+vib_features':
            combined = torch.cat([ae_time.mean(dim=1), vib_time.mean(dim=1)], dim=1)
        elif mode == 'ae_features+vib_features+pp':
            combined = torch.cat([ae_time.mean(dim=1), vib_time.mean(dim=1), outputs['pp']], dim=1)
        elif mode == 'all':  # 'all'
            combined = torch.cat([outputs['ae_out'], outputs['vib_out'], outputs['pp']], dim=1)

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
            return 64  # Only PP output
        if input_type == 'ae_spec':
            return 64  # Only AE output
        elif input_type == 'vib_spec':
            return 64  # Only Vib output
        if input_type == 'ae_features':
            return 4  # Only AE output
        elif input_type == 'vib_features':
            return 4  # Only Vib output
        elif input_type == 'ae_spec+ae_features':
            return 64 + 4  # AE output + AE time features
        elif input_type == 'vib_spec+vib_features':
            return 64 + 4  # Vib output + Vib time features
        elif input_type == 'ae_features+pp':
            return 4 + 64
        elif input_type == 'vib_features+pp':
            return 4 + 64
        elif input_type == 'ae_spec+ae_features+vib_spec+vib_features':
            return 64 + 4 + 64 + 4  # AE output + AE time features + Vib output + Vib time features
        else:  # 'all'
            return 64 + 64 + 64  # AE output + Vib output + Physics


    def _init_weights(self, m):
        """
        Initialize weights for all layers in the model.
        """
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    def initialize_weights(self):
        """
        Apply weight initialization to all submodules.
        """
        self.apply(self._init_weights)

# Modified Spectrogram Processor
class SpectrogramProcessor(nn.Module):
    def __init__(self, in_channels, out_dim=32):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # Adaptive pooling to (1,1) preserves channel dimension
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        # Input features should be 32 (number of channels)
        self.linear = nn.Linear(32, out_dim)
        
    def forward(self, x):
        # x: [batch, seq_len, C, H, W]
        batch_size, seq_len = x.shape[:2]
        x = x.view(batch_size * seq_len, *x.shape[2:])  # Merge batch and seq
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.adaptive_pool(x)
        x = self.flatten(x)  # Flatten to [batch*seq, channels]
        features = self.linear(x)
        
        return features.view(batch_size, seq_len, -1)
