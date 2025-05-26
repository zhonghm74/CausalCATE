"""
Loss Functions for the MIM-DRCFR Model.

This module provides various loss functions used in the training of the
Mutual Information Regularized Disentangled Representation for Counterfactual Regression (MIM-DRCFR) model.
These include:
- factual_mse_loss: For supervised learning of potential outcomes.
- mmd_loss: Maximum Mean Discrepancy for comparing distributions.
- ipm_loss_zy: An IPM based on MMD to balance z_y distributions.
- mi_loss_zs_t: An MMD-based estimator for mutual information I(z_s; t).
"""
import torch
import torch.nn.functional as F

def factual_mse_loss(y_true, t_true, y_pred_h0, y_pred_h1):
    """
    Calculates the factual mean squared error loss (L_R in the MIM-DRCFR paper).

    This loss is the sum of the mean squared error (MSE) for the control group's
    predicted outcomes (using h0) and the MSE for the treated group's predicted
    outcomes (using h1).
    L_R = (1/N_c) * sum_{i where t_i=0} (y_i - h0(phi_z(x_i)))^2 +
          (1/N_t) * sum_{i where t_i=1} (y_i - h1(phi_z(x_i)))^2
    where N_c and N_t are the number of control and treated units, respectively.

    Args:
        y_true (torch.Tensor): Observed factual outcomes. Shape: (batch_size,) or (batch_size, 1).
        t_true (torch.Tensor): Observed binary treatments (0 for control, 1 for treated).
                               Shape: (batch_size,) or (batch_size, 1). Must be boolean or convertible to boolean.
        y_pred_h0 (torch.Tensor): Predicted outcomes from prediction head h0 (E[Y(0)|z_y]).
                                  Shape: (batch_size,) or (batch_size, 1).
        y_pred_h1 (torch.Tensor): Predicted outcomes from prediction head h1 (E[Y(1)|z_y]).
                                  Shape: (batch_size,) or (batch_size, 1).

    Returns:
        torch.Tensor: The factual MSE loss (a scalar). If one group is empty,
                      the loss for that group is 0. If both are empty, loss is 0.
    """
    loss = torch.tensor(0.0, device=y_true.device, dtype=y_true.dtype)
    
    # Ensure t_true is boolean for indexing. Squeeze if it's [N,1] to be [N] for boolean indexing.
    if t_true.ndim > 1 and t_true.shape[1] == 1:
        t_true_bool = t_true.squeeze(1).bool()
    else:
        # Ensure it's boolean if it's already 1D but float/int
        t_true_bool = t_true.bool()


    # Control group (t_true_bool is False)
    y_control_true = y_true[~t_true_bool]
    y_control_pred = y_pred_h0[~t_true_bool]
    
    if y_control_true.numel() > 0:
        loss += F.mse_loss(y_control_pred.squeeze(), y_control_true.squeeze(), reduction='mean')

    # Treated group (t_true_bool is True)
    y_treated_true = y_true[t_true_bool]
    y_treated_pred = y_pred_h1[t_true_bool]

    if y_treated_true.numel() > 0:
        loss += F.mse_loss(y_treated_pred.squeeze(), y_treated_true.squeeze(), reduction='mean')
        
    return loss


def rbf_kernel_matrix(x1, x2, sigma):
    """
    Computes the Radial Basis Function (RBF) kernel matrix between two sets of samples.

    The RBF kernel is defined as: k(x_i, x_j) = exp(-||x_i - x_j||^2 / (2 * sigma^2)),
    where ||.||^2 is the squared Euclidean distance and sigma is the bandwidth parameter.

    Args:
        x1 (torch.Tensor): First set of samples. Shape: (n1, dim), where n1 is the number
                           of samples and dim is the feature dimensionality.
        x2 (torch.Tensor): Second set of samples. Shape: (n2, dim), where n2 is the number
                           of samples and dim is the feature dimensionality.
        sigma (float): Bandwidth of the RBF kernel. Must be positive.

    Returns:
        torch.Tensor: The RBF kernel matrix of shape (n1, n2).
    """
    # Pairwise squared Euclidean distances: D_ij = ||x1_i - x2_j||^2
    x1_sq_norms = x1.pow(2).sum(dim=1, keepdim=True)  # Shape: (n1, 1)
    x2_sq_norms = x2.pow(2).sum(dim=1, keepdim=True)  # Shape: (n2, 1)
    
    dist_sq = x1_sq_norms + x2_sq_norms.t() - 2 * torch.mm(x1, x2.t())
    
    # Apply RBF kernel
    gamma = 1.0 / (2 * sigma**2)
    kernel_matrix = torch.exp(-gamma * dist_sq)
    return kernel_matrix


def mmd_loss(x1, x2, sigmas):
    """
    Calculates the Maximum Mean Discrepancy (MMD) between two sets of samples,
    x1 (from distribution P) and x2 (from distribution Q).

    This implementation uses a sum of RBF kernels with different bandwidths (`sigmas`)
    to make the MMD estimate more robust. The MMD^2 is computed as:
    MMD^2(P, Q) = E[k(x, x')] + E[k(y, y')] - 2E[k(x, y)],
    where x, x' ~ P and y, y' ~ Q.

    Args:
        x1 (torch.Tensor): Samples from distribution P. Shape: (n1, dim).
        x2 (torch.Tensor): Samples from distribution Q. Shape: (n2, dim).
        sigmas (list of float): List of RBF kernel bandwidths. Each sigma must be positive.

    Returns:
        torch.Tensor: The estimated MMD^2 loss (a scalar). Returns 0.0 if either
                      `x1` or `x2` is empty, or if no valid positive sigmas are provided.
    """
    if x1.numel() == 0 or x2.numel() == 0:
        return torch.tensor(0.0, device=x1.device, dtype=x1.dtype)

    total_mmd_sq = torch.tensor(0.0, device=x1.device, dtype=x1.dtype)
    valid_sigmas_found = False
    for sigma in sigmas:
        if sigma <= 0: # Sigma must be positive
            continue
        valid_sigmas_found = True
            
        k_x1x1 = rbf_kernel_matrix(x1, x1, sigma)
        k_x2x2 = rbf_kernel_matrix(x2, x2, sigma)
        k_x1x2 = rbf_kernel_matrix(x1, x2, sigma)
        
        # Biased MMD^2 estimate for a single kernel
        mmd2_single_sigma = k_x1x1.mean() + k_x2x2.mean() - 2 * k_x1x2.mean()
        total_mmd_sq += mmd2_single_sigma
    
    if not valid_sigmas_found: # Avoid division by zero if sigmas list was empty or all non-positive
        return torch.tensor(0.0, device=x1.device, dtype=x1.dtype)
        
    return total_mmd_sq / len(sigmas) # Average MMD^2 over sigmas


def ipm_loss_zy(z_y_treated, z_y_control, sigmas):
    """
    Calculates the Integral Probability Metric (IPM) loss (L_IPM in the MIM-DRCFR paper)
    between the latent representations z_y of the treated and control groups.

    This loss aims to balance the distributions of z_y for the treated and control
    groups, i.e., P(z_y|t=1) and P(z_y|t=0). It is implemented using MMD.

    Args:
        z_y_treated (torch.Tensor): Latent variable z_y for the treated group.
                                    Shape: (n_treated, dim_zy).
        z_y_control (torch.Tensor): Latent variable z_y for the control group.
                                    Shape: (n_control, dim_zy).
        sigmas (list of float): List of RBF kernel bandwidths for the MMD calculation.

    Returns:
        torch.Tensor: The IPM loss (a scalar), which is the MMD^2 between
                      z_y_treated and z_y_control.
    """
    return mmd_loss(z_y_treated, z_y_control, sigmas)


def mi_loss_zs_t(z_s, t_true, sigmas):
    """
    Calculates the Mutual Information minimization loss (L_I in the MIM-DRCFR paper)
    between the latent representation z_s and the treatment variable t.

    This loss aims to make z_s independent of t, effectively minimizing I(z_s; t).
    It is estimated using MMD between samples from the joint distribution p(z_s, t)
    and samples from the product of marginal distributions p(z_s)p(t).

    Args:
        z_s (torch.Tensor): Latent variable z_s. Shape: (batch_size, dim_zs).
        t_true (torch.Tensor): Observed binary treatments (0 or 1).
                               Shape: (batch_size,) or (batch_size, 1).
        sigmas (list of float): List of RBF kernel bandwidths for the MMD calculation.

    Returns:
        torch.Tensor: The MI minimization loss (a scalar), which is the MMD^2
                      between p(z_s, t) and p(z_s)p(t).
    """
    if z_s.numel() == 0:
        return torch.tensor(0.0, device=z_s.device, dtype=z_s.dtype)

    # Ensure t_true is [batch_size, 1] and float for concatenation
    if t_true.ndim == 1:
        t_true_reshaped = t_true.float().unsqueeze(-1)
    else:
        t_true_reshaped = t_true.float() # Assuming it's already [N,1] if not 1D

    # Samples from p(z_s, t) - the joint distribution
    z_s_t_joint = torch.cat((z_s, t_true_reshaped), dim=1)

    # Samples from p(z_s)p(t) - product of marginals (achieved by permuting t)
    perm_indices = torch.randperm(t_true_reshaped.size(0), device=t_true_reshaped.device)
    t_permuted = t_true_reshaped[perm_indices]
    z_s_t_product_marginals = torch.cat((z_s, t_permuted), dim=1)
    
    return mmd_loss(z_s_t_joint, z_s_t_product_marginals, sigmas)


# Example Usage (for testing purposes, can be removed or commented out)
if __name__ == '__main__':
    # --- Test factual_mse_loss ---
    print("--- Testing factual_mse_loss ---")
    y_true_ex = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).float()
    t_true_ex = torch.tensor([0, 1, 0, 1, 0, 1]).bool() # 3 control, 3 treated
    y_pred_h0_ex = torch.tensor([1.1, 0.0, 3.1, 0.0, 5.1, 0.0]).float() # Predictions for t=0
    y_pred_h1_ex = torch.tensor([0.0, 2.2, 0.0, 4.2, 0.0, 6.2]).float() # Predictions for t=1

    loss_r = factual_mse_loss(y_true_ex, t_true_ex, y_pred_h0_ex, y_pred_h1_ex)
    # Expected:
    # Control: ( (1.1-1.0)^2 + (3.1-3.0)^2 + (5.1-5.0)^2 ) / 3 = (0.01 + 0.01 + 0.01) / 3 = 0.01
    # Treated: ( (2.2-2.0)^2 + (4.2-4.0)^2 + (6.2-6.0)^2 ) / 3 = (0.04 + 0.04 + 0.04) / 3 = 0.04
    # Total L_R = 0.01 + 0.04 = 0.05
    print(f"Factual MSE Loss (L_R): {loss_r.item()}") # Expected: approx 0.05

    # Test with only one group
    t_true_ex_all_control = torch.tensor([0, 0, 0]).bool()
    y_true_ex_ac = y_true_ex[:3]
    y_pred_h0_ex_ac = y_pred_h0_ex[:3]
    y_pred_h1_ex_ac = y_pred_h1_ex[:3] # Should not be used
    loss_r_ac = factual_mse_loss(y_true_ex_ac, t_true_ex_all_control, y_pred_h0_ex_ac, y_pred_h1_ex_ac)
    print(f"Factual MSE Loss (all control): {loss_r_ac.item()}") # Expected: approx 0.01

    # --- Test mmd_loss ---
    print("\n--- Testing mmd_loss ---")
    torch.manual_seed(0)
    sample_p1 = torch.randn(100, 10)
    sample_p2 = torch.randn(100, 10) + 0.5 # Shifted distribution
    sigmas_ex = [0.1, 1.0, 10.0]
    
    loss_mmd_same = mmd_loss(sample_p1, sample_p1, sigmas_ex) # MMD between same dist should be near 0
    loss_mmd_diff = mmd_loss(sample_p1, sample_p2, sigmas_ex) # MMD between diff dist should be > 0
    print(f"MMD Loss (p1 vs p1): {loss_mmd_same.item()}")
    print(f"MMD Loss (p1 vs p2): {loss_mmd_diff.item()}")

    # --- Test ipm_loss_zy ---
    print("\n--- Testing ipm_loss_zy ---")
    z_y_treated_ex = torch.randn(50, 20) + 0.3
    z_y_control_ex = torch.randn(60, 20)
    loss_ipm = ipm_loss_zy(z_y_treated_ex, z_y_control_ex, sigmas_ex)
    print(f"IPM Loss (L_IPM): {loss_ipm.item()}")

    # --- Test mi_loss_zs_t ---
    print("\n--- Testing mi_loss_zs_t ---")
    z_s_ex = torch.randn(120, 15)
    t_true_mi_ex = (torch.rand(120) > 0.5).bool() # Random treatments
    
    # Case 1: z_s and t are independent (generate z_s independently of t)
    loss_mi_indep = mi_loss_zs_t(z_s_ex, t_true_mi_ex, sigmas_ex)
    print(f"MI Loss (L_I, z_s indep of t): {loss_mi_indep.item()}") # Should be small

    # Case 2: z_s and t are dependent (make z_s dependent on t)
    z_s_dependent_ex = z_s_ex.clone()
    z_s_dependent_ex[t_true_mi_ex, :5] += 1.0 # Add offset to some dims of z_s for treated group
    loss_mi_dep = mi_loss_zs_t(z_s_dependent_ex, t_true_mi_ex, sigmas_ex)
    print(f"MI Loss (L_I, z_s dep on t): {loss_mi_dep.item()}") # Should be larger than indep case
