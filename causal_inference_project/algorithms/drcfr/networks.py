"""
Neural Network Components for MIM-DRCFR Model.

Includes MLP and Attention-based Encoders, PredictionHead,
with optional Dropout and BatchNorm.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def _build_block(in_dim, out_dim, use_batchnorm=False, dropout=0.0):
    """Linear → (BatchNorm) → ELU → (Dropout)."""
    layers = [nn.Linear(in_dim, out_dim)]
    if use_batchnorm:
        layers.append(nn.BatchNorm1d(out_dim))
    layers.append(nn.ELU())
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    return layers


class Encoder(nn.Module):
    """Encoder phi(x) → (z_y, z_s) with optional Dropout/BatchNorm."""

    def __init__(self, input_dim, hidden_dims_phi, latent_dim_zy, latent_dim_zs,
                 dropout=0.0, use_batchnorm=False):
        super(Encoder, self).__init__()

        layers = []
        current_dim = input_dim
        for hidden_dim in hidden_dims_phi:
            layers.extend(_build_block(current_dim, hidden_dim, use_batchnorm, dropout))
            current_dim = hidden_dim

        self.shared_layers = nn.Sequential(*layers)
        self.fc_zy = nn.Linear(current_dim, latent_dim_zy)
        self.fc_zs = nn.Linear(current_dim, latent_dim_zs)

    def forward(self, x):
        h = self.shared_layers(x)
        return self.fc_zy(h), self.fc_zs(h)


class PredictionHead(nn.Module):
    """Prediction head h(z_y) → outcome, with optional Dropout/BatchNorm."""

    def __init__(self, latent_dim_zy, hidden_dims_h, output_dim=1,
                 dropout=0.0, use_batchnorm=False):
        super(PredictionHead, self).__init__()

        layers = []
        current_dim = latent_dim_zy
        for hidden_dim in hidden_dims_h:
            layers.extend(_build_block(current_dim, hidden_dim, use_batchnorm, dropout))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, z_y):
        return self.network(z_y)


class AttentionEncoder(nn.Module):
    """Attention-based encoder that treats each feature as a token.

    Projects input features to an embedding space, applies multi-head
    self-attention to capture feature interactions, then produces
    (z_y, z_s) via linear heads.
    """

    def __init__(self, input_dim, embed_dim, n_heads, n_layers,
                 latent_dim_zy, latent_dim_zs, dropout=0.0):
        super().__init__()
        self.input_proj = nn.Linear(1, embed_dim)
        self.pos_emb = nn.Parameter(torch.randn(1, input_dim, embed_dim) * 0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=embed_dim * 2,
            dropout=dropout, activation='gelu', batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.pool_fc = nn.Linear(input_dim * embed_dim, embed_dim)
        self.fc_zy = nn.Linear(embed_dim, latent_dim_zy)
        self.fc_zs = nn.Linear(embed_dim, latent_dim_zs)

    def forward(self, x):
        B, D = x.shape
        tokens = x.unsqueeze(-1)           # (B, D, 1)
        tokens = self.input_proj(tokens)    # (B, D, embed)
        tokens = tokens + self.pos_emb[:, :D, :]
        h = self.transformer(tokens)        # (B, D, embed)
        h_flat = h.reshape(B, -1)           # (B, D*embed)
        h_pool = F.elu(self.pool_fc(h_flat))
        return self.fc_zy(h_pool), self.fc_zs(h_pool)

# Example Usage (for testing purposes, can be removed or commented out)
if __name__ == '__main__':
    # Example parameters based on the paper
    input_dim_example = 25 # Example: IHDP dataset has 25 features
    hidden_dims_phi_example = [100, 100, 100]
    latent_dim_zy_example = 50
    latent_dim_zs_example = 50
    
    hidden_dims_h_example = [100, 100, 100]
    output_dim_example = 1 # Predicting a single outcome value

    # Instantiate Encoder
    encoder = Encoder(input_dim_example, hidden_dims_phi_example, latent_dim_zy_example, latent_dim_zs_example)
    
    # Instantiate Prediction Heads (one for control, one for treated)
    h0_network = PredictionHead(latent_dim_zy_example, hidden_dims_h_example, output_dim_example)
    h1_network = PredictionHead(latent_dim_zy_example, hidden_dims_h_example, output_dim_example)
    
    # Create dummy input data
    batch_size = 4
    dummy_x = torch.randn(batch_size, input_dim_example)
    
    # Forward pass through encoder
    z_y_out, z_s_out = encoder(dummy_x)
    print("Encoder outputs:")
    print("z_y shape:", z_y_out.shape) # Expected: [batch_size, latent_dim_zy_example]
    print("z_s shape:", z_s_out.shape) # Expected: [batch_size, latent_dim_zs_example]
    
    # Forward pass through prediction heads
    y0_pred = h0_network(z_y_out)
    y1_pred = h1_network(z_y_out)
    print("\nPrediction Head outputs:")
    print("y0_pred shape:", y0_pred.shape) # Expected: [batch_size, output_dim_example]
    print("y1_pred shape:", y1_pred.shape) # Expected: [batch_size, output_dim_example]

    # Check parameters are being updated (simple check)
    print(f"\nEncoder has {sum(p.numel() for p in encoder.parameters() if p.requires_grad)} trainable parameters.")
    print(f"PredictionHead (h0) has {sum(p.numel() for p in h0_network.parameters() if p.requires_grad)} trainable parameters.")
