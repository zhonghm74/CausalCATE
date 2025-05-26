import torch
import numpy as np
from sklearn.model_selection import train_test_split

# Adjust sys.path to import SRCVAEModel from the project
import sys
import os
# Get the absolute path of the current script
current_script_path = os.path.abspath(__file__)
# Get the directory containing the current script (examples)
current_script_dir = os.path.dirname(current_script_path)
# Get the project root directory (causal_inference_project)
project_root = os.path.dirname(current_script_dir)
# Add the project root to sys.path
sys.path.insert(0, project_root)

from algorithms.srcvae.srcvae_model import SRCVAEModel

def generate_srcvae_synthetic_data(n_samples=2000, x_dim=5, u_true_dim=1, v_true_dim=1, true_ate=2.0, random_seed=42):
    """
    Generates synthetic data based on a specific Structural Causal Model (SCM).
    - U: Unobserved confounder
    - V: Instrumental variable (affects X, T, but not Y directly)
    - X: Observed covariates
    - T: Treatment
    - Y: Outcome
    """
    np.random.seed(random_seed)

    # Generate latent variables
    U = np.random.normal(0, 1, (n_samples, u_true_dim))
    V = np.random.normal(0, 1, (n_samples, v_true_dim))

    # Generate covariates X
    # X depends on U and V
    # X depends on U and V. U and V are (n_samples, 1).
    # Their sum (U * 0.7 + V * 0.5) will be (n_samples, 1).
    # This sum will be broadcast across all x_dim features of X.
    X_base_effect = U * 0.7 + V * 0.5  # Shape: (n_samples, 1)
    X_noise = np.random.normal(0, 0.2, (n_samples, x_dim))
    X = X_base_effect + X_noise # Broadcasting X_base_effect to (n_samples, x_dim)


    # Generate treatment T (binary)
    # T depends on X (e.g., X[:,0]), U, and V
    t_logits = X[:, 0] * 0.5 + U[:,0] * 0.6 + V[:,0] * 0.8 - 0.5 
    t_probs = 1 / (1 + np.exp(-t_logits))
    T_binary = np.random.binomial(1, t_probs, n_samples).astype(np.float32) # Shape (n_samples,)

    # Generate potential outcomes Y0 and Y1
    # Y depends on X (e.g., X[:,1]) and U. V does not directly affect Y.
    Y0_underlying = X[:, 1] * 0.4 + U[:,0] * 1.0 + np.random.normal(0, 0.1, n_samples)
    Y1_underlying = Y0_underlying + true_ate # Constant ATE for simplicity

    # Observed outcome Y_f
    Y_f = T_binary * Y1_underlying + (1 - T_binary) * Y0_underlying # Shape (n_samples,)

    return X, T_binary, Y_f, Y0_underlying, Y1_underlying, U, V


def main():
    # 1. Generate Synthetic Data
    n_samples = 2000
    x_true_dim = 5 # True dimensionality of X
    u_true_dim_gen = 1
    v_true_dim_gen = 1
    ate_gen = 2.0

    X_np, T_np, Yf_np, Y0_np, Y1_np, U_np, V_np = generate_srcvae_synthetic_data(
        n_samples=n_samples, x_dim=x_true_dim, u_true_dim=u_true_dim_gen, 
        v_true_dim=v_true_dim_gen, true_ate=ate_gen, random_seed=42
    )
    
    # 2. Data Preparation
    # Split data (80% train, 20% validation)
    X_train_np, X_val_np, \
    T_train_np, T_val_np, \
    Yf_train_np, Yf_val_np, \
    Y0_val_np, Y1_val_np = train_test_split( # Y0_train, Y1_train not used by model, only for eval
        X_np, T_np, Yf_np, Y0_np, Y1_np, test_size=0.2, random_state=123
    )

    # Convert to PyTorch Tensors
    x_train_tensor = torch.tensor(X_train_np, dtype=torch.float32)
    t_train_tensor = torch.tensor(T_train_np, dtype=torch.float32) # Kept as 1D
    yf_train_tensor = torch.tensor(Yf_train_np, dtype=torch.float32) # Kept as 1D

    x_val_tensor = torch.tensor(X_val_np, dtype=torch.float32)
    t_val_tensor = torch.tensor(T_val_np, dtype=torch.float32)   # Kept as 1D
    yf_val_tensor = torch.tensor(Yf_val_np, dtype=torch.float32) # Kept as 1D
    
    # For evaluation
    y0_val_tensor = torch.tensor(Y0_val_np, dtype=torch.float32).unsqueeze(1) # Ensure [N,1] for PEHE calc
    y1_val_tensor = torch.tensor(Y1_val_np, dtype=torch.float32).unsqueeze(1) # Ensure [N,1] for PEHE calc

    # 3. Model Instantiation
    # Model's latent dimensions (can be different from true u_dim, v_dim used in generation)
    model_u_latent_dim = 5 
    model_v_latent_dim = 5 
    
    # Simplified hidden dimensions for example
    h_dims_u_enc = [64, 32]
    h_dims_v_enc = [64, 32]
    h_dims_x_dec = [32, 64]
    h_dims_t_dec = [32, 64]
    h_dims_y_dec = [32, 64]
    h_dims_aux_t = [32]
    h_dims_aux_y = [32]

    model = SRCVAEModel(
        x_dim=x_true_dim, t_dim=1, y_dim=1, 
        u_dim=model_u_latent_dim, v_dim=model_v_latent_dim,
        hidden_dims_encoder_u=h_dims_u_enc, hidden_dims_encoder_v=h_dims_v_enc,
        hidden_dims_decoder_x=h_dims_x_dec, hidden_dims_decoder_t=h_dims_t_dec, 
        hidden_dims_decoder_y=h_dims_y_dec,
        hidden_dims_aux_qtx=h_dims_aux_t, hidden_dims_aux_qyxt=h_dims_aux_y,
        alpha_x=1.0, alpha_t=1.0, alpha_y=1.0,
        beta_u=0.1, beta_v=0.1,
        gamma_t=1.0, gamma_y=1.0, # Increased aux weights as per some VAE practices
        learning_rate=1e-3, weight_decay=1e-5
    )
    print(f"SRCVAEModel instantiated. Moving to device: {model.device}")

    # 4. Model Training
    num_epochs = 50 # Adjust for reasonable runtime in an example (may need more for convergence)
    batch_size = 128
    print_every_epochs = 10

    print(f"\nStarting SRCVAE training for {num_epochs} epochs...")
    model.fit(
        x_train=x_train_tensor, t_train=t_train_tensor, y_train=yf_train_tensor,
        num_epochs=num_epochs, batch_size=batch_size,
        x_val=x_val_tensor, t_val=t_val_tensor, y_val=yf_val_tensor,
        print_every_epochs=print_every_epochs
    )
    print("SRCVAE Training finished.")

    # 5. Effect Estimation and Evaluation (on validation set)
    print("\n--- Evaluating SRCVAE on Validation Set ---")
    
    ite_preds_val, y0_preds_val, y1_preds_val = model.estimate_causal_effect(
        x_val_tensor, n_samples_u=100
    ) # ite_preds_val will be [N_val, 1]

    # True ATE from generated validation data
    true_ate_val = (Y1_val_np - Y0_val_np).mean()
    
    # Estimated ATE from model predictions
    estimated_ate_val = ite_preds_val.mean().item()

    # PEHE: sqrt(mean(((Y1_val - Y0_val) - pred_ITE)^2))
    # Ensure true_ite_val_tensor is also [N_val, 1]
    true_ite_val_tensor = (y1_val_tensor - y0_val_tensor) # Already [N,1] from earlier unsqueeze
    
    # Check shapes before PEHE calculation
    # print(f"Shape of true_ite_val_tensor: {true_ite_val_tensor.shape}")
    # print(f"Shape of ite_preds_val: {ite_preds_val.shape}")

    pehe_val = torch.sqrt(torch.mean((true_ite_val_tensor.to(model.device) - ite_preds_val)**2)).item()

    print(f"True ATE (from validation data Y0, Y1): {true_ate_val:.4f}")
    print(f"Estimated ATE (SRCVAE on validation data): {estimated_ate_val:.4f}")
    print(f"PEHE (SRCVAE on validation data): {pehe_val:.4f}")
    print(f"(Data generation ATE parameter was: {ate_gen:.4f})")

if __name__ == "__main__":
    main()
