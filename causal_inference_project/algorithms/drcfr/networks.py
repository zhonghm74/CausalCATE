"""
Neural Network Components for MIM-DRCFR Model.

This module defines the core neural network architectures used in the
Mutual Information Regularized Disentangled Representation for Counterfactual Regression (MIM-DRCFR) model.
It includes:
- Encoder: Maps input features to disentangled latent representations (z_y and z_s).
- PredictionHead: Predicts outcomes from the z_y latent representation.
"""
import torch
import torch.nn as nn

class Encoder(nn.Module):
    """
    Encoder network phi(x) that maps input features x to two latent representations:
    z_y (for outcome prediction) and z_s (for treatment mechanism/similarity).

    The architecture consists of shared hidden layers followed by two separate linear
    layers to produce z_y and z_s. This structure is based on Figure 2 and
    Section 3.1 of the MIM-DRCFR paper, where the paper's phi_z and phi_s
    are implemented here as two heads of a single encoder.

    Attributes:
        shared_layers (nn.Sequential): Shared hidden layers.
        fc_zy (nn.Linear): Linear layer to produce z_y.
        fc_zs (nn.Linear): Linear layer to produce z_s.
    """
    def __init__(self, input_dim, hidden_dims_phi, latent_dim_zy, latent_dim_zs):
        """
        Initializes the Encoder network.

        Args:
            input_dim (int): Dimensionality of the input features x.
            hidden_dims_phi (list of int): List of hidden layer sizes for the shared part.
                                           Example: `[100, 100, 100]`
            latent_dim_zy (int): Dimensionality of the latent representation z_y,
                                 which is used for outcome prediction. Example: 50.
            latent_dim_zs (int): Dimensionality of the latent representation z_s,
                                 which captures treatment assignment related information. Example: 50.
        """
        super(Encoder, self).__init__()
        
        layers = []
        current_dim = input_dim
        for hidden_dim in hidden_dims_phi:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ELU()) # As specified in MIM-DRCFR (typically used in CFR-type nets)
            current_dim = hidden_dim
        
        self.shared_layers = nn.Sequential(*layers)
        
        # Output layer for z_y
        self.fc_zy = nn.Linear(current_dim, latent_dim_zy)
        # Output layer for z_s
        self.fc_zs = nn.Linear(current_dim, latent_dim_zs)

    def forward(self, x):
        """
        Forward pass of the encoder.

        Args:
            x (torch.Tensor): Input feature tensor of shape (batch_size, input_dim).
            
        Returns:
            tuple:
                - z_y (torch.Tensor): Latent representation for outcome prediction,
                                      shape (batch_size, latent_dim_zy).
                - z_s (torch.Tensor): Latent representation for similarity/treatment mechanism,
                                      shape (batch_size, latent_dim_zs).
        """
        h = self.shared_layers(x)
        z_y = self.fc_zy(h)
        z_s = self.fc_zs(h)
        return z_y, z_s

class PredictionHead(nn.Module):
    """
    Prediction head network h(z_y) that predicts an outcome given the
    latent representation z_y.

    In the MIM-DRCFR model, two instances of this class are typically used:
    - h0(z_y): Predicts the potential outcome Y(0) (control group).
    - h1(z_y): Predicts the potential outcome Y(1) (treated group).
    This architecture is based on Figure 2 and Section 3.1 of the MIM-DRCFR paper.

    Attributes:
        network (nn.Sequential): The sequence of layers forming the prediction head.
    """
    def __init__(self, latent_dim_zy, hidden_dims_h, output_dim=1):
        """
        Initializes the PredictionHead network.

        Args:
            latent_dim_zy (int): Dimensionality of the input latent representation z_y.
                                 This should match `latent_dim_zy` from the `Encoder`. Example: 50.
            hidden_dims_h (list of int): List of hidden layer sizes for this head.
                                         Example: `[100, 100, 100]`
            output_dim (int): Dimensionality of the predicted outcome. Typically 1 for
                              predicting a single outcome value (e.g., Y(0) or Y(1)). Default is 1.
        """
        super(PredictionHead, self).__init__()
        
        layers = []
        current_dim = latent_dim_zy
        for hidden_dim in hidden_dims_h:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ELU()) # As specified in MIM-DRCFR
            current_dim = hidden_dim
            
        # Output layer
        layers.append(nn.Linear(current_dim, output_dim))
        
        self.network = nn.Sequential(*layers)

    def forward(self, z_y):
        """
        Forward pass of the prediction head.
        
        Args:
            z_y (torch.Tensor): Latent representation z_y from the Encoder,
                                shape (batch_size, latent_dim_zy).
            
        Returns:
            torch.Tensor: Predicted outcome y_pred, shape (batch_size, output_dim).
        """
        y_pred = self.network(z_y)
        return y_pred

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
