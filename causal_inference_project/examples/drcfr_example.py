import torch
import numpy as np
from sklearn.model_selection import train_test_split

# Adjust sys.path to import DRCFRModel from the project
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

from algorithms.drcfr.drcfr_model import DRCFRModel

def generate_drcfr_synthetic_data(n_samples=2000, n_features=10, true_ate_baseline=2.0, random_seed=42):
    """
    Generates synthetic data for MIM-DRCFR example.
    Args:
        n_samples (int): Number of samples.
        n_features (int): Number of features for X.
        true_ate_baseline (float): Baseline average treatment effect.
        random_seed (int): Random seed for reproducibility.
    Returns:
        tuple: (X, T, Y_f, Y0, Y1) as numpy arrays.
               T is 1D (n_samples,). Y_f, Y0, Y1 are 1D (n_samples,).
    """
    np.random.seed(random_seed)
    X = np.random.randn(n_samples, n_features)

    # Treatment assignment (T) depends on X[:,0] and X[:,1]
    treatment_propensity = 1 / (1 + np.exp(-(X[:, 0] + X[:, 1]))) # Simplified as per instructions
    T = np.random.binomial(1, treatment_propensity, n_samples) # Shape: (n_samples,)

    # Potential outcomes (Y0, Y1)
    # Y0 depends on X[:,0], X[:,1]
    Y0 = X[:, 0] * 0.5 + X[:, 1] * 0.2 + np.random.normal(0, 0.2, n_samples)
    
    # Y1 depends on X[:,0], X[:,1] (like Y0) and also X[:,2], X[:,3] (treatment effect modifiers)
    # plus the true_ate_baseline
    Y1 = X[:, 0] * 0.5 + X[:, 1] * 0.2 + \
         X[:, 2] * 0.6 + X[:, 3] * 0.3 + \
         true_ate_baseline + np.random.normal(0, 0.2, n_samples)

    # Factual outcome
    Y_f = T * Y1 + (1 - T) * Y0 # Shape: (n_samples,)
    
    return X, T, Y_f, Y0, Y1


def main():
    # 1. Generate Synthetic Data
    n_samples = 2000
    n_features = 10
    true_ate_param = 1.5 # This is the 'true_ate_baseline' in data generation

    X_np, T_np, Yf_np, Y0_np, Y1_np = generate_drcfr_synthetic_data(
        n_samples=n_samples, n_features=n_features, true_ate_baseline=true_ate_param, random_seed=42
    )

    # 2. Data Preparation: Split and Convert to Tensors
    # Split data (e.g., 80% train, 20% validation/test)
    X_train_np, X_val_np, \
    T_train_np, T_val_np, \
    Yf_train_np, Yf_val_np, \
    Y0_train_np, Y0_val_np, \
    Y1_train_np, Y1_val_np = train_test_split(
        X_np, T_np, Yf_np, Y0_np, Y1_np, test_size=0.2, random_state=123
    )

    # Convert to PyTorch Tensors
    # Features
    x_train_tensor = torch.tensor(X_train_np, dtype=torch.float32)
    x_val_tensor = torch.tensor(X_val_np, dtype=torch.float32)
    # Treatments (should be 1D for DRCFRModel fit and loss computation)
    t_train_tensor = torch.tensor(T_train_np, dtype=torch.float32) # Will be unsqueezed in fit if needed
    t_val_tensor = torch.tensor(T_val_np, dtype=torch.float32)   # Will be unsqueezed in fit if needed
    # Factual Outcomes (should be [N,1] for DRCFRModel fit)
    yf_train_tensor = torch.tensor(Yf_train_np, dtype=torch.float32).unsqueeze(1)
    yf_val_tensor = torch.tensor(Yf_val_np, dtype=torch.float32).unsqueeze(1)
    # Potential Outcomes (for evaluation)
    y0_val_tensor = torch.tensor(Y0_val_np, dtype=torch.float32).unsqueeze(1)
    y1_val_tensor = torch.tensor(Y1_val_np, dtype=torch.float32).unsqueeze(1)


    # 3. Model Instantiation
    input_dim = n_features
    # Network architecture parameters (example values)
    hidden_dims_phi = [128, 64, 32] # Encoder hidden layers
    latent_dim_zy = 20             # Latent dim for outcome prediction
    latent_dim_zs = 20             # Latent dim for similarity
    hidden_dims_h = [64, 32]       # Prediction head hidden layers
    output_dim = 1                 # Outcome dimension
    
    # Loss hyperparameters
    alpha = 1.0  # Weight for IPM loss on z_y
    beta = 0.1   # Weight for MI loss on z_s and t
    
    # Optimizer parameters
    learning_rate = 1e-3
    weight_decay = 1e-5

    model = DRCFRModel(
        input_dim=input_dim,
        hidden_dims_phi=hidden_dims_phi,
        latent_dim_zy=latent_dim_zy,
        latent_dim_zs=latent_dim_zs,
        hidden_dims_h=hidden_dims_h,
        output_dim=output_dim,
        alpha=alpha,
        beta=beta,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        sigmas_ipm = [0.1, 1.0, 10.0], # Default MMD sigmas for IPM loss
        sigmas_mi = [0.1, 1.0, 10.0]   # Default MMD sigmas for MI loss
    )
    print(f"DRCFRModel instantiated. Moving to device: {next(model.parameters()).device}")

    # 4. Model Training
    num_epochs = 100 # Increased epochs for better convergence demonstration
    batch_size = 128
    print_every_epochs = 20

    print(f"\nStarting training for {num_epochs} epochs...")
    model.fit(
        x_train=x_train_tensor, 
        y_train=yf_train_tensor, # Factual outcomes for training
        t_train=t_train_tensor,  # Treatment assignments for training
        num_epochs=num_epochs, 
        batch_size=batch_size,
        x_val=x_val_tensor,      # Validation features
        y_val=yf_val_tensor,     # Validation factual outcomes
        t_val=t_val_tensor,      # Validation treatment assignments
        print_every_epochs=print_every_epochs
    )
    print("Training finished.")

    # 5. Effect Estimation and Evaluation (on validation set)
    print("\n--- Evaluating on Validation Set ---")
    
    # Predict ITE on the validation set
    ite_preds_val_tensor = model.predict_ite(x_val_tensor) # Shape: [n_val_samples, 1]

    # True ITE from validation set potential outcomes
    true_ite_val_tensor = y1_val_tensor - y0_val_tensor # Shape: [n_val_samples, 1]

    # Calculate true ATE from the validation data generation process
    true_ate_from_data = (Y1_val_np - Y0_val_np).mean()
    
    # Calculate estimated ATE from model predictions on validation set
    estimated_ate = ite_preds_val_tensor.mean().item()

    # Calculate PEHE (Precision in Estimating Heterogeneous Effect)
    # PEHE = sqrt( MSE_ITE ) = sqrt( mean( (true_ITE_i - pred_ITE_i)^2 ) )
    mse_ite = ((true_ite_val_tensor - ite_preds_val_tensor)**2).mean().item()
    pehe = np.sqrt(mse_ite)

    print(f"True ATE (from validation data Y0, Y1): {true_ate_from_data:.4f}")
    print(f"Estimated ATE (from model on validation data): {estimated_ate:.4f}")
    print(f"PEHE (Precision in Estimating Heterogeneous Effect) on validation data: {pehe:.4f}")
    
    # For reference, the 'true_ate_baseline' parameter used in data generation was: {true_ate_param}
    # The actual ATE in the data depends on the average effect of X[:,2]*0.6 + X[:,3]*0.3

if __name__ == "__main__":
    main()
