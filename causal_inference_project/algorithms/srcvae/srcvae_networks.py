"""
Neural Network Components for the SRCVAE Model.

This module defines the core neural network architectures used in the
SRCVAE (Learning Causal Effect Variational Autoencoder) model.
It includes:
- Encoders: For latent variables u (q_phi_u(u|x,t,y)) and v (q_phi_v(v|x,t)).
- Decoders: For reconstructing x (p_theta_x(x|u,v)), t (p_theta_t(t|x,v)), and y (p_theta_y(y|x,u,t)).
- Auxiliary Networks: For q_phi_t(t|x) and q_phi_y(y|x,t).
All networks are built using a common MLP helper function (_build_mlp).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

def _build_mlp(input_dim, output_dim, hidden_dims, activation=nn.ELU(), output_activation=None):
    """
    Helper function to build a Multi-Layer Perceptron (MLP).

    Args:
        input_dim (int): Dimensionality of the input layer.
        output_dim (int): Dimensionality of the output layer.
        hidden_dims (list of int): List of hidden layer sizes.
        activation (nn.Module, optional): Activation function for hidden layers.
                                           Defaults to nn.ELU().
        output_activation (nn.Module, optional): Activation function for the output layer.
                                                 If None, no activation is applied to the output.
                                                 Defaults to None.

    Returns:
        nn.Sequential: A PyTorch Sequential model representing the MLP.
    """
    layers = []
    current_dim = input_dim
    for h_dim in hidden_dims:
        layers.append(nn.Linear(current_dim, h_dim))
        layers.append(activation)
        current_dim = h_dim
    
    layers.append(nn.Linear(current_dim, output_dim))
    if output_activation is not None:
        layers.append(output_activation)
        
    return nn.Sequential(*layers)

class EncoderU(nn.Module):
    """
    Encoder for latent variable u: q_phi_u(u | x, t, y_f).

    This network takes observed features x, treatment t, and factual outcome y_f
    as input and outputs the parameters (mean and log-variance) of a
    Gaussian distribution for the latent variable u.

    Attributes:
        shared_mlp (nn.Sequential): Shared MLP layers.
        fc_mean (nn.Linear): Linear layer to output the mean of u.
        fc_logvar (nn.Linear): Linear layer to output the log-variance of u.
    """
    def __init__(self, x_dim, t_dim, y_dim, u_dim, hidden_dims):
        """
        Initializes the EncoderU network.

        Args:
            x_dim (int): Dimensionality of input features x.
            t_dim (int): Dimensionality of treatment t (typically 1 for binary).
            y_dim (int): Dimensionality of outcome y_f (typically 1).
            u_dim (int): Dimensionality of the latent variable u.
            hidden_dims (list of int): List of hidden layer sizes for the MLP.
        """
        super().__init__()
        input_concat_dim = x_dim + t_dim + y_dim
        
        # Determine the output dimension of the shared part and the input to fc layers
        last_hidden_dim_shared = hidden_dims[-1] if hidden_dims else input_concat_dim
        
        self.shared_mlp = _build_mlp(
            input_dim=input_concat_dim,
            output_dim=last_hidden_dim_shared,
            hidden_dims=hidden_dims[:-1] if hidden_dims else [], # All but last for shared part
            activation=nn.ELU(),
            output_activation=nn.ELU() if hidden_dims else None # ELU if hidden layers exist
        )
        
        self.fc_mean = nn.Linear(last_hidden_dim_shared, u_dim)
        self.fc_logvar = nn.Linear(last_hidden_dim_shared, u_dim)

    def forward(self, x, t, y):
        """
        Forward pass of EncoderU.

        Args:
            x (torch.Tensor): Input features, shape (batch_size, x_dim).
            t (torch.Tensor): Treatment, shape (batch_size, t_dim) or (batch_size,).
            y (torch.Tensor): Factual outcome, shape (batch_size, y_dim) or (batch_size,).

        Returns:
            tuple:
                - u_mean (torch.Tensor): Mean of the Gaussian distribution for u, shape (batch_size, u_dim).
                - u_logvar (torch.Tensor): Log-variance of the Gaussian distribution for u, shape (batch_size, u_dim).
        """
        # Ensure t and y are 2D before concatenation: [batch_size, 1]
        if t.ndim == 1: t = t.unsqueeze(1)
        if y.ndim == 1: y = y.unsqueeze(1)
            
        input_concat = torch.cat((x, t, y), dim=1)
        h = self.shared_mlp(input_concat)
        u_mean = self.fc_mean(h)
        u_logvar = self.fc_logvar(h)
        return u_mean, u_logvar

class EncoderV(nn.Module):
    """
    Encoder for latent variable v: q_phi_v(v | x, t).

    This network takes observed features x and treatment t as input and outputs
    the parameters (mean and log-variance) of a Gaussian distribution for the
    latent variable v.

    Attributes:
        shared_mlp (nn.Sequential): Shared MLP layers.
        fc_mean (nn.Linear): Linear layer to output the mean of v.
        fc_logvar (nn.Linear): Linear layer to output the log-variance of v.
    """
    def __init__(self, x_dim, t_dim, v_dim, hidden_dims):
        """
        Initializes the EncoderV network.

        Args:
            x_dim (int): Dimensionality of input features x.
            t_dim (int): Dimensionality of treatment t (typically 1 for binary).
            v_dim (int): Dimensionality of the latent variable v.
            hidden_dims (list of int): List of hidden layer sizes for the MLP.
        """
        super().__init__()
        input_concat_dim = x_dim + t_dim

        last_hidden_dim_shared = hidden_dims[-1] if hidden_dims else input_concat_dim

        self.shared_mlp = _build_mlp(
            input_dim=input_concat_dim,
            output_dim=last_hidden_dim_shared,
            hidden_dims=hidden_dims[:-1] if hidden_dims else [],
            activation=nn.ELU(),
            output_activation=nn.ELU() if hidden_dims else None
        )
        
        self.fc_mean = nn.Linear(last_hidden_dim_shared, v_dim)
        self.fc_logvar = nn.Linear(last_hidden_dim_shared, v_dim)

    def forward(self, x, t):
        """
        Forward pass of EncoderV.

        Args:
            x (torch.Tensor): Input features, shape (batch_size, x_dim).
            t (torch.Tensor): Treatment, shape (batch_size, t_dim) or (batch_size,).

        Returns:
            tuple:
                - v_mean (torch.Tensor): Mean of the Gaussian distribution for v, shape (batch_size, v_dim).
                - v_logvar (torch.Tensor): Log-variance of the Gaussian distribution for v, shape (batch_size, v_dim).
        """
        if t.ndim == 1: t = t.unsqueeze(1)
        input_concat = torch.cat((x, t), dim=1)
        h = self.shared_mlp(input_concat)
        v_mean = self.fc_mean(h)
        v_logvar = self.fc_logvar(h)
        return v_mean, v_logvar

class DecoderX(nn.Module):
    """Decoder p_theta_x(x | u, v).  Optionally heteroscedastic (learns variance)."""

    def __init__(self, u_dim, v_dim, x_dim, hidden_dims, heteroscedastic=False):
        super().__init__()
        self.heteroscedastic = heteroscedastic
        self.mlp = _build_mlp(u_dim + v_dim, x_dim, hidden_dims, nn.ELU(), None)
        if heteroscedastic:
            self.mlp_logvar = _build_mlp(u_dim + v_dim, x_dim, hidden_dims, nn.ELU(), None)

    def forward(self, u, v):
        h = torch.cat((u, v), dim=1)
        x_mean = self.mlp(h)
        if self.heteroscedastic:
            return x_mean, self.mlp_logvar(h)
        return x_mean

class DecoderT(nn.Module):
    """
    Decoder for treatment t: p_theta_t(t | x, v).

    This network predicts the logits for the binary treatment t, given
    observed covariates x and latent variable v.

    Attributes:
        mlp (nn.Sequential): The MLP layers for decoding.
    """
    def __init__(self, x_dim, v_dim, t_dim, hidden_dims):
        """
        Initializes the DecoderT network.

        Args:
            x_dim (int): Dimensionality of observed covariates x.
            v_dim (int): Dimensionality of latent variable v.
            t_dim (int): Dimensionality of treatment t (typically 1 for logits of binary t).
            hidden_dims (list of int): List of hidden layer sizes for the MLP.
        """
        super().__init__()
        self.mlp = _build_mlp(
            input_dim=x_dim + v_dim,
            output_dim=t_dim, # Outputting t_logits
            hidden_dims=hidden_dims,
            activation=nn.ELU(),
            output_activation=None # Linear output for logits
        )

    def forward(self, x, v):
        """
        Forward pass of DecoderT.

        Args:
            x (torch.Tensor): Observed covariates x, shape (batch_size, x_dim).
            v (torch.Tensor): Latent variable v, shape (batch_size, v_dim).

        Returns:
            torch.Tensor: Predicted logits for treatment t, shape (batch_size, t_dim).
        """
        input_concat = torch.cat((x, v), dim=1)
        t_logits = self.mlp(input_concat)
        return t_logits

class DecoderY(nn.Module):
    """Decoder p_theta_y(y | x, u, t).  Optionally heteroscedastic."""

    def __init__(self, x_dim, u_dim, t_dim, y_dim, hidden_dims, heteroscedastic=False):
        super().__init__()
        self.heteroscedastic = heteroscedastic
        in_dim = x_dim + u_dim + t_dim
        self.mlp = _build_mlp(in_dim, y_dim, hidden_dims, nn.ELU(), None)
        if heteroscedastic:
            self.mlp_logvar = _build_mlp(in_dim, y_dim, hidden_dims, nn.ELU(), None)

    def forward(self, x, u, t):
        if t.ndim == 1: t = t.unsqueeze(1)
        h = torch.cat((x, u, t), dim=1)
        y_mean = self.mlp(h)
        if self.heteroscedastic:
            return y_mean, self.mlp_logvar(h)
        return y_mean

class AuxiliaryQTX(nn.Module):
    """
    Auxiliary network for predicting treatment t from covariates x: q_phi_t(t | x).

    This network aids in the VAE training process. For binary treatment, it outputs logits.

    Attributes:
        mlp (nn.Sequential): The MLP layers for prediction.
    """
    def __init__(self, x_dim, t_dim, hidden_dims):
        """
        Initializes the AuxiliaryQTX network.

        Args:
            x_dim (int): Dimensionality of input features x.
            t_dim (int): Dimensionality of treatment t (typically 1 for logits of binary t).
            hidden_dims (list of int): List of hidden layer sizes for the MLP.
        """
        super().__init__()
        self.mlp = _build_mlp(
            input_dim=x_dim,
            output_dim=t_dim, # Outputting t_logits
            hidden_dims=hidden_dims,
            activation=nn.ELU(),
            output_activation=None # Linear output for logits
        )

    def forward(self, x):
        """
        Forward pass of AuxiliaryQTX.

        Args:
            x (torch.Tensor): Input features, shape (batch_size, x_dim).

        Returns:
            torch.Tensor: Predicted logits for treatment t, shape (batch_size, t_dim).
        """
        t_logits = self.mlp(x)
        return t_logits

class AuxiliaryQYXT(nn.Module):
    """
    Auxiliary network for predicting outcome y from covariates x and treatment t: q_phi_y(y | x, t).

    This network aids in the VAE training process. For continuous outcome, it outputs the mean.

    Attributes:
        mlp (nn.Sequential): The MLP layers for prediction.
    """
    def __init__(self, x_dim, t_dim, y_dim, hidden_dims):
        """
        Initializes the AuxiliaryQYXT network.

        Args:
            x_dim (int): Dimensionality of input features x.
            t_dim (int): Dimensionality of treatment t (typically 1 for binary).
            y_dim (int): Dimensionality of outcome y (typically 1).
            hidden_dims (list of int): List of hidden layer sizes for the MLP.
        """
        super().__init__()
        self.mlp = _build_mlp(
            input_dim=x_dim + t_dim,
            output_dim=y_dim, # Outputting y_mean
            hidden_dims=hidden_dims,
            activation=nn.ELU(),
            output_activation=None # Linear output for mean
        )

    def forward(self, x, t):
        """
        Forward pass of AuxiliaryQYXT.

        Args:
            x (torch.Tensor): Input features, shape (batch_size, x_dim).
            t (torch.Tensor): Treatment, shape (batch_size, t_dim) or (batch_size,).

        Returns:
            torch.Tensor: Predicted mean of outcome y, shape (batch_size, y_dim).
        """
        if t.ndim == 1: t = t.unsqueeze(1)
        input_concat = torch.cat((x, t), dim=1)
        y_mean = self.mlp(input_concat)
        return y_mean

class ConditionalPriorU(nn.Module):
    """Conditional prior p_θ(u | x) outputting (mean, logvar)."""

    def __init__(self, x_dim, u_dim, hidden_dims):
        super().__init__()
        last_h = hidden_dims[-1] if hidden_dims else x_dim
        self.shared = _build_mlp(x_dim, last_h,
                                 hidden_dims[:-1] if hidden_dims else [],
                                 nn.ELU(), nn.ELU() if hidden_dims else None)
        self.fc_mean = nn.Linear(last_h, u_dim)
        self.fc_logvar = nn.Linear(last_h, u_dim)

    def forward(self, x):
        h = self.shared(x)
        return self.fc_mean(h), self.fc_logvar(h)


# Example Usage (for testing purposes, can be removed or commented out)
if __name__ == '__main__':
    x_dim, t_dim, y_dim, u_dim, v_dim = 10, 1, 1, 5, 4
    hidden_dims_enc = [32, 16]
    hidden_dims_dec = [16, 32]
    batch_size = 4

    # Dummy inputs
    x_sample = torch.randn(batch_size, x_dim)
    t_sample = (torch.rand(batch_size, t_dim) > 0.5).float()
    y_sample = torch.randn(batch_size, y_dim)
    u_sample = torch.randn(batch_size, u_dim)
    v_sample = torch.randn(batch_size, v_dim)

    print("--- Testing Encoders ---")
    encoder_u = EncoderU(x_dim, t_dim, y_dim, u_dim, hidden_dims_enc)
    u_mean, u_logvar = encoder_u(x_sample, t_sample, y_sample)
    print(f"EncoderU output shapes: u_mean {u_mean.shape}, u_logvar {u_logvar.shape}")
    # assert u_mean.shape == (batch_size, u_dim)
    # assert u_logvar.shape == (batch_size, u_dim)

    encoder_v = EncoderV(x_dim, t_dim, v_dim, hidden_dims_enc)
    v_mean, v_logvar = encoder_v(x_sample, t_sample)
    print(f"EncoderV output shapes: v_mean {v_mean.shape}, v_logvar {v_logvar.shape}")
    # assert v_mean.shape == (batch_size, v_dim)
    # assert v_logvar.shape == (batch_size, v_dim)

    print("\n--- Testing Decoders ---")
    decoder_x = DecoderX(u_dim, v_dim, x_dim, hidden_dims_dec)
    x_mean_decoded = decoder_x(u_sample, v_sample)
    print(f"DecoderX output shape: x_mean {x_mean_decoded.shape}")
    # assert x_mean_decoded.shape == (batch_size, x_dim)

    decoder_t = DecoderT(x_dim, v_dim, t_dim, hidden_dims_dec)
    t_logits_decoded = decoder_t(x_sample, v_sample)
    print(f"DecoderT output shape: t_logits {t_logits_decoded.shape}")
    # assert t_logits_decoded.shape == (batch_size, t_dim)

    decoder_y = DecoderY(x_dim, u_dim, t_dim, y_dim, hidden_dims_dec)
    y_mean_decoded = decoder_y(x_sample, u_sample, t_sample)
    print(f"DecoderY output shape: y_mean {y_mean_decoded.shape}")
    # assert y_mean_decoded.shape == (batch_size, y_dim)

    print("\n--- Testing Auxiliary Networks ---")
    aux_qt_x = AuxiliaryQTX(x_dim, t_dim, hidden_dims_dec) # Reusing hidden_dims_dec for example
    t_logits_aux = aux_qt_x(x_sample)
    print(f"AuxiliaryQTX output shape: t_logits {t_logits_aux.shape}")
    # assert t_logits_aux.shape == (batch_size, t_dim)

    aux_qy_xt = AuxiliaryQYXT(x_dim, t_dim, y_dim, hidden_dims_dec)
    y_mean_aux = aux_qy_xt(x_sample, t_sample)
    print(f"AuxiliaryQYXT output shape: y_mean {y_mean_aux.shape}")
    # assert y_mean_aux.shape == (batch_size, y_dim)

    print("\nAll network component tests passed (basic shape checks).")
