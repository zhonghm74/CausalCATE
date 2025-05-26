import numpy as np
import pandas as pd # Optional, but good to have for potential future use
from sklearn.linear_model import Lasso, LogisticRegression 
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# Adjust path to import double_ml if necessary, assuming causal_inference_project is in PYTHONPATH
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

from algorithms.dml.dml_core import double_ml


def generate_synthetic_data(n_samples=1000, n_features=5, true_causal_effect=2.0, binary_treatment=True, random_seed=42):
    """Generates synthetic data for DML example."""
    np.random.seed(random_seed)
    X = np.random.rand(n_samples, n_features)

    if binary_treatment:
        # Example propensity based on first two features
        treatment_propensity = 1 / (1 + np.exp(-X[:, 0] - X[:, 1])) 
        T = np.random.binomial(1, treatment_propensity, n_samples)
        
        # Outcome model based on potential outcomes
        # y0 is the outcome if T=0, affected by X[:, 2], X[:, 3]
        y0 = X[:, 2] + X[:, 3] + np.random.normal(0, 0.1, n_samples)
        # y1 is the outcome if T=1
        y1 = y0 + true_causal_effect 
        
        y = T * y1 + (1 - T) * y0
    else: # Continuous treatment
        # Example continuous treatment based on X[:,0] and X[:,1]
        T = X[:, 0] + 0.5 * X[:, 1] + np.random.normal(0, 0.1, n_samples)
        
        # Confounders (X[:,2], X[:,3]) also affect Y directly
        confounders_effect_on_y = X[:, 2] * 0.5 + X[:, 3] * 0.3 
        y = confounders_effect_on_y + true_causal_effect * T + np.random.normal(0, 0.1, n_samples)
        
    return X, y, T

def main():
    # --- Binary Treatment Example ---
    print("--- Binary Treatment Example ---")
    true_effect_binary = 2.5
    X_binary, y_binary, T_binary = generate_synthetic_data(
        n_samples=2000, 
        n_features=5, 
        true_causal_effect=true_effect_binary, 
        binary_treatment=True,
        random_seed=42
    )

    # Instantiate machine learning models
    model_y_binary = RandomForestRegressor(random_state=42)
    model_t_binary = RandomForestClassifier(random_state=42)

    # Call the double_ml function
    estimated_effect_binary = double_ml(
        X_binary, y_binary, T_binary, 
        model_y_binary, model_t_binary, 
        treatment_is_binary=True
    )

    print(f"True causal effect (binary treatment): {true_effect_binary}")
    print(f"DML estimated causal effect (binary treatment): {estimated_effect_binary:.4f}")
    print("-" * 30)

    # --- Continuous Treatment Example ---
    print("\n--- Continuous Treatment Example ---")
    true_effect_continuous = -1.5
    X_continuous, y_continuous, T_continuous = generate_synthetic_data(
        n_samples=2000, 
        n_features=5, 
        true_causal_effect=true_effect_continuous, 
        binary_treatment=False,
        random_seed=123
    )

    # Instantiate machine learning models
    model_y_continuous = Lasso(alpha=0.1) 
    model_t_continuous = Lasso(alpha=0.1)

    # Call the double_ml function
    estimated_effect_continuous = double_ml(
        X_continuous, y_continuous, T_continuous,
        model_y_continuous, model_t_continuous,
        treatment_is_binary=False
    )

    print(f"True causal effect (continuous treatment): {true_effect_continuous}")
    print(f"DML estimated causal effect (continuous treatment): {estimated_effect_continuous:.4f}")
    print("-" * 30)

if __name__ == "__main__":
    main()
