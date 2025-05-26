import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

# Assuming dml_core.py is in causal_inference_project/algorithms/dml/
# and this test file is in causal_inference_project/tests/
# We need to adjust sys.path for pytest to find the modules
import sys
import os

# Get the absolute path of the current test script
current_script_path = os.path.abspath(__file__)
# Get the directory containing the test script (tests)
current_script_dir = os.path.dirname(current_script_path)
# Get the project root directory (causal_inference_project) by going up one level
project_root = os.path.dirname(current_script_dir)
# Add the project root to sys.path to allow imports from 'algorithms' and 'examples'
sys.path.insert(0, project_root)

from algorithms.dml.dml_core import double_ml

# Copied from causal_inference_project/examples/dml_example.py to avoid import complexities for this task
def generate_synthetic_data(n_samples=1000, n_features=5, true_causal_effect=2.0, binary_treatment=True, random_seed=42):
    """Generates synthetic data for DML example."""
    np.random.seed(random_seed)
    X = np.random.rand(n_samples, n_features)

    if binary_treatment:
        # Example propensity based on first two features
        treatment_propensity = 1 / (1 + np.exp(-X[:, 0] - X[:, 1])) 
        T = np.random.binomial(1, treatment_propensity, n_samples)
        
        # Outcome model based on potential outcomes
        y0 = X[:, 2] + X[:, 3] + np.random.normal(0, 0.1, n_samples)
        y1 = y0 + true_causal_effect 
        y = T * y1 + (1 - T) * y0
    else: # Continuous treatment
        T = X[:, 0] + 0.5 * X[:, 1] + np.random.normal(0, 0.1, n_samples)
        confounders_effect_on_y = X[:, 2] * 0.5 + X[:, 3] * 0.3 
        y = confounders_effect_on_y + true_causal_effect * T + np.random.normal(0, 0.1, n_samples)
        
    return X, y, T

def test_dml_binary_treatment():
    """Tests the double_ml function with binary treatment."""
    true_causal_effect = 0.75
    n_samples = 1000  # Using a reasonable number of samples for stable estimates
    
    X, y, T = generate_synthetic_data(
        n_samples=n_samples,
        n_features=5,
        true_causal_effect=true_causal_effect,
        binary_treatment=True,
        random_seed=42
    )

    # Using simple models for testing purposes
    model_y = LinearRegression()
    model_t = LogisticRegression(solver='liblinear', random_state=42) # liblinear is robust for small datasets

    estimated_effect = double_ml(X, y, T, model_y, model_t, treatment_is_binary=True)

    # Assert that the estimated effect is close to the true effect
    # The tolerance (abs) can be adjusted based on expected precision with simple models and n_samples
    assert estimated_effect == pytest.approx(true_causal_effect, abs=0.2) 

def test_dml_continuous_treatment():
    """Tests the double_ml function with continuous treatment."""
    true_causal_effect = -0.5
    n_samples = 1000 # Using a reasonable number of samples for stable estimates

    X, y, T = generate_synthetic_data(
        n_samples=n_samples,
        n_features=5,
        true_causal_effect=true_causal_effect,
        binary_treatment=False,
        random_seed=123
    )

    # Using simple models for testing purposes
    model_y = LinearRegression()
    model_t = LinearRegression()

    estimated_effect = double_ml(X, y, T, model_y, model_t, treatment_is_binary=False)

    # Assert that the estimated effect is close to the true effect
    # The tolerance (abs) can be adjusted
    assert estimated_effect == pytest.approx(true_causal_effect, abs=0.2)

# To run these tests, navigate to the project root in the terminal and run:
# python -m pytest causal_inference_project/tests/test_dml_core.py
# Ensure scikit-learn and pytest are installed.
