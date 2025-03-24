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
    def __init__(self,interp=False):
        super().__init__()
        # AE Pathway (2 spec channels + 4 time features)
        self.ae_spec_processor = SpectrogramProcessor(2, out_dim=32)
        self.interp = interp

        # Vib Pathway (3 spec channels + 4 time features)
        self.vib_spec_processor = SpectrogramProcessor(3, out_dim=32)

        if self.interp == True:
            self.ae_interpreter = FeatureInterpreter(spec_feat_dim=32, time_feat_dim=4)
            self.vib_interpreter = FeatureInterpreter(spec_feat_dim=32, time_feat_dim=4)

        # Physics Processor
        self.physics_encoder = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(), nn.LayerNorm(64)
        )

        # Final Fusion
        self.regressor = nn.Sequential(
            nn.Linear(64 * 2 + 64, 128),  # 64(ae) + 64(vib) + 64(physics)
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, batch):
        # AE Processing
        ae_spec = self.ae_spec_processor(batch["spec_ae"])  # [batch, seq_len, 32]
        ae_time = batch["features_ae"]  # [batch, seq_len, 4]
        ae_out, ae_attn = self.ae_interpreter(ae_spec, ae_time)

        # Vib Processing
        vib_spec = self.vib_spec_processor(batch["spec_vib"])  # [batch, seq_len, 32]
        vib_time = batch["features_vib"]  # [batch, seq_len, 4]
        vib_out, vib_attn = self.vib_interpreter(vib_spec, vib_time)

        # Physics
        physics = self.physics_encoder(batch["features_pp"])

        # Final prediction
        combined = torch.cat([ae_out, vib_out, physics], dim=1)
        if self.interp == True:
            return self.regressor(combined), {"ae": ae_attn, "vib": vib_attn}
        else:
            return self.regressor(combined)


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