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
        self.physics_encoder = nn.Sequential(
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
            ae_spec = self.ae_spec_processor(batch["spec_ae"])  # [batch, seq_len, 32]
            ae_time = batch["features_ae"]  # [batch, seq_len, 4]
            ae_out, ae_attn = self.ae_interpreter(ae_spec, ae_time)
            outputs['ae_out'] = ae_out
            outputs['ae_attn'] = ae_attn

        # Vib Processing (if applicable)
        if 'vib' in mode or 'all' in mode:
            vib_spec = self.vib_spec_processor(batch["spec_vib"])  # [batch, seq_len, 32]
            vib_time = batch["features_vib"]  # [batch, seq_len, 4]
            vib_out, vib_attn = self.vib_interpreter(vib_spec, vib_time)
            outputs['vib_out'] = vib_out
            outputs['vib_attn'] = vib_attn

        # Physics Processing (if applicable)
        if 'all' in mode or 'pp' in mode:
            physics = self.physics_encoder(batch["features_pp"])
            outputs['physics'] = physics

        # Combine features based on mode
        if mode == 'ae_spec':
            combined = outputs['ae_out']
        elif mode == 'vib_spec':
            combined = outputs['vib_out']
        elif mode == 'ae_spec+ae_features':
            combined = torch.cat([outputs['ae_out'], ae_time.mean(dim=1)], dim=1)
        elif mode == 'vib_spec+vib_features':
            combined = torch.cat([outputs['vib_out'], vib_time.mean(dim=1)], dim=1)
        elif mode == 'ae_spec+ae_features+vib_spec+vib_features':
            combined = torch.cat([outputs['ae_out'], ae_time.mean(dim=1), outputs['vib_out'], vib_time.mean(dim=1)], dim=1)
        elif mode == 'pp':
            combined = outputs['physics']
        else:  # 'all'
            combined = torch.cat([outputs['ae_out'], outputs['vib_out'], outputs['physics']], dim=1)

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
        if input_type == 'ae_spec':
            return 64  # Only AE output
        elif input_type == 'vib_spec':
            return 64  # Only Vib output
        elif input_type == 'ae_spec+ae_features':
            return 64 + 4  # AE output + AE time features
        elif input_type == 'vib_spec+vib_features':
            return 64 + 4  # Vib output + Vib time features
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
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(8 * 4 * 4, out_dim),
        )

    def forward(self, x):
        # x: [batch, seq_len, C, H, W]
        batch_size, seq_len = x.shape[:2]
        x = x.view(-1, *x.shape[2:])  # Merge batch and seq
        features = self.conv(x)
        return features.view(batch_size, seq_len, -1)