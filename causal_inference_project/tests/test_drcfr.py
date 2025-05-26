import torch
import torch.nn as nn
import numpy as np
import pytest

# Adjust sys.path for imports
import sys
import os

# Get the absolute path of the current test script
current_script_path = os.path.abspath(__file__)
# Get the directory containing the test script (tests)
current_script_dir = os.path.dirname(current_script_path)
# Get the project root directory (causal_inference_project) by going up one level
project_root = os.path.dirname(current_script_dir)
# Add the project root to sys.path
sys.path.insert(0, project_root)

from algorithms.drcfr.networks import Encoder, PredictionHead
from algorithms.drcfr.losses import factual_mse_loss, mmd_loss, ipm_loss_zy, mi_loss_zs_t
from algorithms.drcfr.drcfr_model import DRCFRModel

# Seed for reproducibility
torch.manual_seed(0)
np.random.seed(0)

# Global device for tests (CPU)
DEVICE = torch.device("cpu")

@pytest.fixture
def default_model_params():
    return {
        'input_dim': 5,
        'hidden_dims_phi': [10, 8],
        'latent_dim_zy': 4,
        'latent_dim_zs': 3,
        'hidden_dims_h': [6, 4],
        'output_dim': 1,
        'alpha': 0.5,
        'beta': 0.5,
        'learning_rate': 1e-4, # Small LR for tests
        'weight_decay': 1e-5,
        'sigmas_ipm': [0.1, 1.0], # Simplified sigmas for tests
        'sigmas_mi': [0.1, 1.0]
    }

@pytest.fixture
def dummy_data():
    n_samples = 10
    input_dim = 5
    x = torch.randn(n_samples, input_dim, device=DEVICE)
    # y_f needs to be [N,1] for factual_mse_loss via DRCFRModel
    y_f = torch.randn(n_samples, 1, device=DEVICE) 
    # t_f needs to be [N] or [N,1] for boolean indexing in losses/model
    t_f = (torch.rand(n_samples, 1, device=DEVICE) > 0.5).float() 
    return x, y_f, t_f

# --- Tests for Network Components ---
def test_encoder_output_shape(default_model_params):
    encoder = Encoder(
        input_dim=default_model_params['input_dim'],
        hidden_dims_phi=default_model_params['hidden_dims_phi'],
        latent_dim_zy=default_model_params['latent_dim_zy'],
        latent_dim_zs=default_model_params['latent_dim_zs']
    ).to(DEVICE)
    x_dummy = torch.randn(10, default_model_params['input_dim'], device=DEVICE)
    z_y, z_s = encoder(x_dummy)
    assert z_y.shape == (10, default_model_params['latent_dim_zy'])
    assert z_s.shape == (10, default_model_params['latent_dim_zs'])

def test_prediction_head_output_shape(default_model_params):
    pred_head = PredictionHead(
        latent_dim_zy=default_model_params['latent_dim_zy'],
        hidden_dims_h=default_model_params['hidden_dims_h'],
        output_dim=default_model_params['output_dim']
    ).to(DEVICE)
    z_y_dummy = torch.randn(10, default_model_params['latent_dim_zy'], device=DEVICE)
    y_pred = pred_head(z_y_dummy)
    assert y_pred.shape == (10, default_model_params['output_dim'])

# --- Tests for Loss Functions ---
def test_factual_mse_loss():
    y_true = torch.tensor([1.0, 2.0, 3.0, 4.0], device=DEVICE)
    t_true = torch.tensor([0, 1, 0, 1], dtype=torch.bool, device=DEVICE) # Needs to be bool
    y_pred_h0 = torch.tensor([1.1, 0.0, 3.1, 0.0], device=DEVICE) # Predictions for t=0
    y_pred_h1 = torch.tensor([0.0, 2.2, 0.0, 4.2], device=DEVICE) # Predictions for t=1

    # Expected: control: ((1.1-1)^2 + (3.1-3)^2)/2 = (0.01+0.01)/2 = 0.01
    #           treated: ((2.2-2)^2 + (4.2-4)^2)/2 = (0.04+0.04)/2 = 0.04
    #           total = 0.01 + 0.04 = 0.05
    expected_loss = ((0.1**2 + 0.1**2)/2) + ((0.2**2 + 0.2**2)/2) # Sum of means
    loss = factual_mse_loss(y_true, t_true, y_pred_h0, y_pred_h1)
    assert torch.isclose(loss, torch.tensor(expected_loss, device=DEVICE), atol=1e-5)

    # Test with only control
    t_true_control = torch.tensor([0, 0], dtype=torch.bool, device=DEVICE)
    y_true_c = y_true[:2]
    y_pred_h0_c = y_pred_h0[:2]
    y_pred_h1_c = y_pred_h1[:2] # Should not be used
    expected_loss_c = (0.1**2)/1 # Only one sample (1.1-1.0)^2 / 1 (Oops, my y_true was for 4 samples)
                               # Corrected: y_true_c=[1,2], t_true_c=[0,0], y_pred_h0_c=[1.1, 0]
                               # This test is tricky, let's simplify:
    y_true_c = torch.tensor([1.0, 3.0], device=DEVICE)
    t_true_c = torch.tensor([0, 0], dtype=torch.bool, device=DEVICE)
    y_pred_h0_c = torch.tensor([1.1, 3.1], device=DEVICE)
    y_pred_h1_c = torch.zeros_like(y_pred_h0_c) # Not used
    expected_loss_c = (0.1**2 + 0.1**2)/2
    loss_c = factual_mse_loss(y_true_c, t_true_c, y_pred_h0_c, y_pred_h1_c)
    assert torch.isclose(loss_c, torch.tensor(expected_loss_c, device=DEVICE), atol=1e-5)

def test_mmd_loss():
    x1 = torch.randn(10, 5, device=DEVICE)
    x2 = torch.randn(10, 5, device=DEVICE) + 0.5 # Shifted
    sigmas = [0.1, 1.0]
    
    loss_val_diff = mmd_loss(x1, x2, sigmas)
    assert loss_val_diff.item() > 0

    loss_val_same = mmd_loss(x1, x1, sigmas) # Should be close to 0
    assert torch.isclose(loss_val_same, torch.tensor(0.0, device=DEVICE), atol=1e-4) # Increased atol for biased MMD

    # Test empty inputs
    assert mmd_loss(torch.empty(0, 5, device=DEVICE), x2, sigmas).item() == 0.0
    assert mmd_loss(x1, torch.empty(0, 5, device=DEVICE), sigmas).item() == 0.0

def test_ipm_loss_zy(default_model_params):
    z_y_treated = torch.randn(7, default_model_params['latent_dim_zy'], device=DEVICE)
    z_y_control = torch.randn(8, default_model_params['latent_dim_zy'], device=DEVICE)
    loss_val = ipm_loss_zy(z_y_treated, z_y_control, default_model_params['sigmas_ipm'])
    assert loss_val.item() >= 0

def test_mi_loss_zs_t(default_model_params):
    z_s = torch.randn(10, default_model_params['latent_dim_zs'], device=DEVICE)
    t_true = (torch.rand(10, 1, device=DEVICE) > 0.5).float()
    loss_val = mi_loss_zs_t(z_s, t_true, default_model_params['sigmas_mi'])
    assert loss_val.item() >= 0

# --- Tests for DRCFRModel ---
@pytest.fixture
def initialized_model(default_model_params):
    model = DRCFRModel(**default_model_params)
    model.to(DEVICE) # Ensure model is on CPU for tests
    return model

def test_drcfr_model_init(initialized_model):
    assert isinstance(initialized_model.encoder, Encoder)
    assert isinstance(initialized_model.h0, PredictionHead)
    assert isinstance(initialized_model.h1, PredictionHead)
    assert initialized_model.alpha == 0.5
    assert next(initialized_model.parameters()).device == DEVICE


def test_drcfr_model_forward_pass(initialized_model, dummy_data, default_model_params):
    x_dummy, _, _ = dummy_data
    y_pred_h0, y_pred_h1, z_y, z_s = initialized_model.forward(x_dummy)
    
    n_samples = x_dummy.shape[0]
    assert y_pred_h0.shape == (n_samples, default_model_params['output_dim'])
    assert y_pred_h1.shape == (n_samples, default_model_params['output_dim'])
    assert z_y.shape == (n_samples, default_model_params['latent_dim_zy'])
    assert z_s.shape == (n_samples, default_model_params['latent_dim_zs'])

def test_drcfr_model_compute_loss(initialized_model, dummy_data):
    x_dummy, y_f_dummy, t_f_dummy = dummy_data
    # Ensure t_f_dummy is boolean for the test, model handles conversion but good to be specific for test
    t_f_dummy_bool = t_f_dummy.squeeze().bool() if t_f_dummy.ndim >1 else t_f_dummy.bool()

    y_pred_h0, y_pred_h1, z_y, z_s = initialized_model.forward(x_dummy)
    total_loss, loss_r, loss_ipm, loss_mi = initialized_model.compute_loss(
        x_dummy, y_f_dummy, t_f_dummy_bool, y_pred_h0, y_pred_h1, z_y, z_s
    )
    assert total_loss.ndim == 0 # Scalar tensor
    assert loss_r.ndim == 0
    assert loss_ipm.ndim == 0
    assert loss_mi.ndim == 0
    assert total_loss.item() >= 0

def test_drcfr_model_fit_smoke_test(initialized_model, dummy_data):
    x_dummy, y_f_dummy, t_f_dummy = dummy_data
    
    # Very small training run
    try:
        initialized_model.fit(
            x_train=x_dummy, y_train=y_f_dummy, t_train=t_f_dummy,
            num_epochs=2, batch_size=5, print_every_epochs=100 # Don't print during smoke test
        )
    except Exception as e:
        pytest.fail(f"model.fit() smoke test failed with exception: {e}")

def test_drcfr_model_predict_ite(initialized_model, dummy_data):
    x_dummy, _, _ = dummy_data
    # Model needs to be "trained" a bit for predict_ite to be meaningful, 
    # but for shape test, just running fit for one step is enough or even not needed if params are init
    # For robustness, let's run a minimal fit.
    y_f_dummy_fit = torch.randn(x_dummy.shape[0], 1, device=DEVICE)
    t_f_dummy_fit = (torch.rand(x_dummy.shape[0], 1, device=DEVICE) > 0.5).float()
    initialized_model.fit(x_dummy, y_f_dummy_fit, t_f_dummy_fit, num_epochs=1, batch_size=x_dummy.shape[0], print_every_epochs=2)

    ite_preds = initialized_model.predict_ite(x_dummy)
    assert ite_preds.shape == (x_dummy.shape[0], initialized_model.h0.network[-1].out_features) # out_features of last layer

    pot_outcomes_0, pot_outcomes_1 = initialized_model.predict_potential_outcomes(x_dummy)
    assert pot_outcomes_0.shape == (x_dummy.shape[0], initialized_model.h0.network[-1].out_features)
    assert pot_outcomes_1.shape == (x_dummy.shape[0], initialized_model.h1.network[-1].out_features)

# To run these tests:
# Ensure pytest is installed: pip install pytest
# Navigate to the project root directory in the terminal.
# Run: python -m pytest causal_inference_project/tests/test_drcfr.py
# Or simply: pytest causal_inference_project/tests/test_drcfr.py (if paths are set up)
