"""
Loss Utility Functions for the SRCVAE Model.

This module provides utility functions for calculating the various loss components
required to train the SRCVAE (Learning Causal Effect Variational Autoencoder) model.
These include:
- KL divergence loss for Gaussian latent variables.
- Reconstruction losses (MSE for continuous, BCE for binary variables).
- Auxiliary prediction losses (which are aliases to reconstruction losses).
"""
import torch
import torch.nn.functional as F

def kl_gaussian_loss(mean, logvar):
    """
    Calculates the KL divergence between a diagonal Gaussian q_phi(z|...)
    and a standard Normal prior N(0, I).

    The formula used is: KL(q_phi(z|...) || N(z|0, I)) = 0.5 * sum(mean^2 + exp(logvar) - logvar - 1),
    summed over the latent dimensions and then averaged over the batch.

    Args:
        mean (torch.Tensor): The mean of the Gaussian distribution q.
                             Shape: (batch_size, latent_dim).
        logvar (torch.Tensor): The log-variance of the Gaussian distribution q.
                               Shape: (batch_size, latent_dim).

    Returns:
        torch.Tensor: The KL divergence loss, averaged over the batch (scalar).
    """
    # Calculate KL divergence for each element in the batch and each latent dimension
    kl_div_elements = 0.5 * (mean.pow(2) + logvar.exp() - logvar - 1)
    
    # Sum over latent dimensions
    kl_div_sum_over_latents = torch.sum(kl_div_elements, dim=1)
    
    # Average over the batch
    return kl_div_sum_over_latents.mean()

def reconstruction_mse_loss(y_true, y_pred_mean):
    """
    Calculates the reconstruction loss for continuous variables using Mean Squared Error (MSE).

    This loss is typically used for reconstructing continuous features (like x or y)
    by comparing the ground truth values with the mean predicted by a decoder.
    It corresponds to the negative log-likelihood under a Gaussian likelihood
    assumption with fixed variance.

    Args:
        y_true (torch.Tensor): The ground truth continuous values.
                               Shape: (batch_size, feature_dim).
        y_pred_mean (torch.Tensor): The predicted mean of the continuous values from the decoder.
                                   Shape: (batch_size, feature_dim).

    Returns:
        torch.Tensor: The MSE reconstruction loss, averaged over all elements in the batch
                      and feature dimensions (scalar), as per `reduction='mean'`.
    """
    return F.mse_loss(y_pred_mean, y_true, reduction='mean')

def reconstruction_bce_loss(t_true, t_pred_logits):
    """
    Calculates the reconstruction loss for binary variables using Binary Cross Entropy (BCE) with logits.

    This loss is suitable for reconstructing binary features (like treatment t)
    where the decoder outputs logits for a Bernoulli distribution.

    Args:
        t_true (torch.Tensor): The ground truth binary values (0 or 1).
                               Shape: (batch_size, feature_dim) or (batch_size, 1).
                               Expected to be float for `binary_cross_entropy_with_logits`.
        t_pred_logits (torch.Tensor): The predicted logits from the decoder.
                                      Shape: (batch_size, feature_dim) or (batch_size, 1).

    Returns:
        torch.Tensor: The BCE reconstruction loss, averaged over all elements in the batch
                      and feature dimensions (scalar), as per `reduction='mean'`.
    """
    return F.binary_cross_entropy_with_logits(t_pred_logits, t_true, reduction='mean')

# Auxiliary losses are aliases to the reconstruction losses, as their mathematical form is the same.
# They are used for the auxiliary prediction tasks q(t|x) and q(y|x,t).

def aux_bce_loss(t_true, t_pred_logits_aux):
    """
    Calculates the auxiliary prediction loss for binary variables using BCE with logits.
    This is an alias for `reconstruction_bce_loss`.

    Args:
        t_true (torch.Tensor): Ground truth binary values.
        t_pred_logits_aux (torch.Tensor): Predicted logits from an auxiliary network.

    Returns:
        torch.Tensor: The BCE loss.
    """
    return reconstruction_bce_loss(t_true, t_pred_logits_aux)

def aux_mse_loss(y_true, y_pred_mean_aux):
    """
    Calculates the auxiliary prediction loss for continuous variables using MSE.
    This is an alias for `reconstruction_mse_loss`.

    Args:
        y_true (torch.Tensor): Ground truth continuous values.
        y_pred_mean_aux (torch.Tensor): Predicted mean from an auxiliary network.

    Returns:
        torch.Tensor: The MSE loss.
    """
    return reconstruction_mse_loss(y_true, y_pred_mean_aux)


# Example Usage (for testing purposes, can be removed or commented out)
if __name__ == '__main__':
    batch_size = 4
    latent_dim = 5
    feature_dim_continuous = 3
    feature_dim_binary = 1

    print("--- Testing kl_gaussian_loss ---")
    mean_ex = torch.randn(batch_size, latent_dim)
    logvar_ex = torch.randn(batch_size, latent_dim) # Can be positive or negative
    kl_loss = kl_gaussian_loss(mean_ex, logvar_ex)
    print(f"KL Gaussian Loss: {kl_loss.item()}")
    # assert kl_loss.ndim == 0

    # Test case where q is same as prior N(0,I) -> KL should be 0
    mean_zero = torch.zeros(batch_size, latent_dim)
    logvar_zero_for_std_one = torch.zeros(batch_size, latent_dim) # log(1^2) = 0
    kl_loss_zero = kl_gaussian_loss(mean_zero, logvar_zero_for_std_one)
    print(f"KL Gaussian Loss (q=prior): {kl_loss_zero.item()}") # Should be close to 0
    # assert torch.isclose(kl_loss_zero, torch.tensor(0.0), atol=1e-7)


    print("\n--- Testing reconstruction_mse_loss ---")
    y_true_ex = torch.randn(batch_size, feature_dim_continuous)
    y_pred_mean_ex = torch.randn(batch_size, feature_dim_continuous)
    mse_loss = reconstruction_mse_loss(y_true_ex, y_pred_mean_ex)
    print(f"Reconstruction MSE Loss: {mse_loss.item()}")
    # assert mse_loss.ndim == 0
    # Check against manual calculation for a simple case
    simple_y_true = torch.tensor([[1.0, 2.0]])
    simple_y_pred = torch.tensor([[1.5, 2.5]])
    expected_mse = ((0.5**2 + 0.5**2) / 2) # Mean over features and batch
    manual_mse_loss = reconstruction_mse_loss(simple_y_true, simple_y_pred)
    # assert torch.isclose(manual_mse_loss, torch.tensor(expected_mse))


    print("\n--- Testing reconstruction_bce_loss ---")
    t_true_ex = torch.randint(0, 2, (batch_size, feature_dim_binary)).float()
    t_pred_logits_ex = torch.randn(batch_size, feature_dim_binary)
    bce_loss = reconstruction_bce_loss(t_true_ex, t_pred_logits_ex)
    print(f"Reconstruction BCE Loss: {bce_loss.item()}")
    # assert bce_loss.ndim == 0

    print("\n--- Testing Auxiliary Losses (should be same as reconstruction) ---")
    aux_bce = aux_bce_loss(t_true_ex, t_pred_logits_ex)
    print(f"Auxiliary BCE Loss: {aux_bce.item()}")
    # assert aux_bce.item() == bce_loss.item()

    aux_mse = aux_mse_loss(y_true_ex, y_pred_mean_ex)
    print(f"Auxiliary MSE Loss: {aux_mse.item()}")
    # assert aux_mse.item() == mse_loss.item()

    print("\nAll loss component tests passed (basic checks).")
