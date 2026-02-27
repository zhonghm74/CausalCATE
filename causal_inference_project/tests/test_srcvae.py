import torch
import torch.nn as nn
import torch.nn.functional as F
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

from algorithms.srcvae.srcvae_networks import (
    EncoderU, EncoderV, DecoderX, DecoderT, DecoderY, AuxiliaryQTX, AuxiliaryQYXT
)
from algorithms.srcvae.srcvae_losses import (
    kl_gaussian_loss, reconstruction_mse_loss, reconstruction_bce_loss
)
from algorithms.srcvae.srcvae_model import SRCVAEModel

# Global Settings/Fixtures
DEVICE = torch.device("cpu")
torch.manual_seed(0)
np.random.seed(0)

@pytest.fixture
def default_srcvae_params():
    return {
        'x_dim': 5, 't_dim': 1, 'y_dim': 1, 
        'u_dim': 3, 'v_dim': 3, # Renamed from u_dim_model, v_dim_model for consistency with SRCVAEModel args
        'hidden_dims_encoder_u': [8, 6],
        'hidden_dims_encoder_v': [8, 6],
        'hidden_dims_decoder_x': [6, 8],
        'hidden_dims_decoder_t': [6, 8],
        'hidden_dims_decoder_y': [6, 8],
        'hidden_dims_aux_qtx': [6],
        'hidden_dims_aux_qyxt': [6],
        'alpha_x': 1.0, 'alpha_t': 1.0, 'alpha_y': 1.0,
        'beta_u': 0.1, 'beta_v': 0.1,
        'gamma_t': 1.0, 'gamma_y': 1.0,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'device': DEVICE # Ensure model is created on CPU
    }

@pytest.fixture
def dummy_srcvae_data(default_srcvae_params):
    n_samples = 10
    x_dim = default_srcvae_params['x_dim']
    
    x = torch.randn(n_samples, x_dim, device=DEVICE)
    # t should be binary {0,1} and float, 1D for SRCVAEModel.fit
    t = torch.randint(0, 2, (n_samples,), device=DEVICE, dtype=torch.float32)
    # y_f should be continuous, 1D for SRCVAEModel.fit
    y_f = torch.randn(n_samples, device=DEVICE, dtype=torch.float32)
    return x, t, y_f

# --- Tests for Network Components ---
@pytest.fixture
def dummy_latent_vars(default_srcvae_params):
    n_samples = 10 # Should match dummy_srcvae_data n_samples if used together
    u_sampled = torch.randn(n_samples, default_srcvae_params['u_dim'], device=DEVICE)
    v_sampled = torch.randn(n_samples, default_srcvae_params['v_dim'], device=DEVICE)
    return u_sampled, v_sampled

def test_encoder_u_output_shape(default_srcvae_params, dummy_srcvae_data):
    params = default_srcvae_params
    x, t, y_f = dummy_srcvae_data
    encoder = EncoderU(params['x_dim'], params['t_dim'], params['y_dim'], params['u_dim'], params['hidden_dims_encoder_u']).to(DEVICE)
    u_mean, u_logvar = encoder(x, t, y_f)
    assert u_mean.shape == (x.shape[0], params['u_dim'])
    assert u_logvar.shape == (x.shape[0], params['u_dim'])

def test_encoder_v_output_shape(default_srcvae_params, dummy_srcvae_data):
    params = default_srcvae_params
    x, t, _ = dummy_srcvae_data
    encoder = EncoderV(params['x_dim'], params['t_dim'], params['v_dim'], params['hidden_dims_encoder_v']).to(DEVICE)
    v_mean, v_logvar = encoder(x, t)
    assert v_mean.shape == (x.shape[0], params['v_dim'])
    assert v_logvar.shape == (x.shape[0], params['v_dim'])

def test_decoder_x_output_shape(default_srcvae_params, dummy_latent_vars):
    params = default_srcvae_params
    u_sampled, v_sampled = dummy_latent_vars
    decoder = DecoderX(params['u_dim'], params['v_dim'], params['x_dim'], params['hidden_dims_decoder_x']).to(DEVICE)
    x_mean = decoder(u_sampled, v_sampled)
    assert x_mean.shape == (u_sampled.shape[0], params['x_dim'])

def test_decoder_t_output_shape(default_srcvae_params, dummy_srcvae_data, dummy_latent_vars):
    params = default_srcvae_params
    x, _, _ = dummy_srcvae_data
    _, v_sampled = dummy_latent_vars
    decoder = DecoderT(params['x_dim'], params['v_dim'], params['t_dim'], params['hidden_dims_decoder_t']).to(DEVICE)
    t_logits = decoder(x, v_sampled)
    assert t_logits.shape == (x.shape[0], params['t_dim'])

def test_decoder_y_output_shape(default_srcvae_params, dummy_srcvae_data, dummy_latent_vars):
    params = default_srcvae_params
    x, t, _ = dummy_srcvae_data
    u_sampled, _ = dummy_latent_vars
    decoder = DecoderY(params['x_dim'], params['u_dim'], params['t_dim'], params['y_dim'], params['hidden_dims_decoder_y']).to(DEVICE)
    y_mean = decoder(x, u_sampled, t)
    assert y_mean.shape == (x.shape[0], params['y_dim'])

def test_auxiliary_qtx_output_shape(default_srcvae_params, dummy_srcvae_data):
    params = default_srcvae_params
    x, _, _ = dummy_srcvae_data
    aux_net = AuxiliaryQTX(params['x_dim'], params['t_dim'], params['hidden_dims_aux_qtx']).to(DEVICE)
    t_logits = aux_net(x)
    assert t_logits.shape == (x.shape[0], params['t_dim'])

def test_auxiliary_qyxt_output_shape(default_srcvae_params, dummy_srcvae_data):
    params = default_srcvae_params
    x, t, _ = dummy_srcvae_data
    aux_net = AuxiliaryQYXT(params['x_dim'], params['t_dim'], params['y_dim'], params['hidden_dims_aux_qyxt']).to(DEVICE)
    y_mean = aux_net(x, t)
    assert y_mean.shape == (x.shape[0], params['y_dim'])

# --- Tests for Loss Functions ---
def test_kl_gaussian_loss_zero_and_positive():
    batch_size, latent_dim = 4, 3
    # Test case where q is same as prior N(0,I) -> KL should be 0
    mean_zero = torch.zeros(batch_size, latent_dim, device=DEVICE)
    logvar_zero_for_std_one = torch.zeros(batch_size, latent_dim, device=DEVICE) # log(1^2) = 0
    kl_loss_val_zero = kl_gaussian_loss(mean_zero, logvar_zero_for_std_one)
    assert torch.isclose(kl_loss_val_zero, torch.tensor(0.0, device=DEVICE), atol=1e-7)

    # Test with non-zero mean/logvar
    mean_ex = torch.randn(batch_size, latent_dim, device=DEVICE)
    logvar_ex = torch.randn(batch_size, latent_dim, device=DEVICE)
    kl_loss_val_positive = kl_gaussian_loss(mean_ex, logvar_ex)
    assert kl_loss_val_positive.item() > 0 # KL div should be positive if not identical to prior

def test_reconstruction_mse_loss():
    y_true = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=DEVICE)
    y_pred_mean = torch.tensor([[1.5, 2.5], [3.0, 3.5]], device=DEVICE)
    # MSE = ( (0.5^2+0.5^2)/2 + (0^2+(-0.5)^2)/2 ) / 2 (mean over samples, then mean over features)
    # No, mse_loss with reduction='mean' averages over all elements.
    # (0.5^2 + 0.5^2 + 0^2 + (-0.5)^2) / 4 = (0.25 + 0.25 + 0 + 0.25) / 4 = 0.75 / 4 = 0.1875
    expected_mse = 0.1875
    loss = reconstruction_mse_loss(y_true, y_pred_mean)
    assert torch.isclose(loss, torch.tensor(expected_mse, device=DEVICE))

def test_reconstruction_bce_loss():
    t_true = torch.tensor([[1.0], [0.0], [1.0]], device=DEVICE) # Needs to be float for BCE
    t_pred_logits = torch.tensor([[2.0], [-1.0], [0.0]], device=DEVICE) # Logits
    # F.binary_cross_entropy_with_logits will calculate this
    expected_loss = F.binary_cross_entropy_with_logits(t_pred_logits, t_true, reduction='mean')
    loss = reconstruction_bce_loss(t_true, t_pred_logits)
    assert torch.isclose(loss, expected_loss)

# --- Tests for SRCVAEModel ---
@pytest.fixture
def initialized_model(default_srcvae_params):
    model = SRCVAEModel(**default_srcvae_params)
    # Model already moves to device in its __init__
    return model

def test_srcvae_model_init(initialized_model):
    assert isinstance(initialized_model.encoder_u, EncoderU)
    assert isinstance(initialized_model.aux_qyxt, AuxiliaryQYXT)
    assert next(initialized_model.parameters()).device == DEVICE

def test_srcvae_reparameterize(initialized_model):
    mean = torch.randn(5, 3, device=DEVICE)
    logvar = torch.randn(5, 3, device=DEVICE)
    z = initialized_model.reparameterize(mean, logvar)
    assert z.shape == mean.shape
    assert not torch.equal(z, mean) # Stochastic component should make them different

def test_srcvae_model_forward_pass(initialized_model, dummy_srcvae_data, default_srcvae_params):
    params = default_srcvae_params
    x, t, y_f = dummy_srcvae_data
    
    # Ensure y_f is also passed to model.forward as per SRCVAEModel definition
    outputs = initialized_model.forward(x, t, y_f)
    (u_mean, u_logvar, v_mean, v_logvar,
     x_recon_mean, t_recon_logits, y_recon_mean,
     t_aux_logits, y_aux_mean, u_sampled, v_sampled,
     x_recon_logvar, y_recon_logvar) = outputs
    
    n_samples = x.shape[0]
    assert u_mean.shape == (n_samples, params['u_dim'])
    assert u_logvar.shape == (n_samples, params['u_dim'])
    assert v_mean.shape == (n_samples, params['v_dim'])
    assert v_logvar.shape == (n_samples, params['v_dim'])
    assert x_recon_mean.shape == (n_samples, params['x_dim'])
    assert t_recon_logits.shape == (n_samples, params['t_dim'])
    assert y_recon_mean.shape == (n_samples, params['y_dim'])
    assert t_aux_logits.shape == (n_samples, params['t_dim'])
    assert y_aux_mean.shape == (n_samples, params['y_dim'])
    assert u_sampled.shape == (n_samples, params['u_dim'])
    assert v_sampled.shape == (n_samples, params['v_dim'])
    assert x_recon_logvar is None
    assert y_recon_logvar is None


def test_srcvae_model_compute_loss(initialized_model, dummy_srcvae_data):
    x, t, y_f = dummy_srcvae_data # y_f is factual y
    
    outputs = initialized_model.forward(x, t, y_f)
    (u_mean, u_logvar, v_mean, v_logvar,
     x_recon_mean, t_recon_logits, y_recon_mean,
     t_aux_logits, y_aux_mean, _, _,
     x_recon_logvar, y_recon_logvar) = outputs

    total_loss, loss_components = initialized_model.compute_loss(
        x, t, y_f,
        u_mean, u_logvar, v_mean, v_logvar,
        x_recon_mean, t_recon_logits, y_recon_mean,
        t_aux_logits, y_aux_mean,
    )
    assert total_loss.ndim == 0
    assert isinstance(loss_components, dict)
    assert 'loss_kl_u' in loss_components
    assert total_loss.item() >= 0

def test_srcvae_model_fit_smoke_test(initialized_model, dummy_srcvae_data):
    x, t, y_f = dummy_srcvae_data
    try:
        initialized_model.fit(
            x_train=x, t_train=t, y_train=y_f,
            num_epochs=2, batch_size=x.shape[0], # Use full batch for simplicity
            print_every_epochs=100 # Don't print
        )
    except Exception as e:
        pytest.fail(f"model.fit() smoke test failed with exception: {e}")

def test_srcvae_model_estimate_causal_effect(initialized_model, dummy_srcvae_data):
    x, t, y_f = dummy_srcvae_data # We only need x for estimation
    
    # Run a minimal fit to ensure parameters are not completely random (though not strictly necessary for shape checks)
    initialized_model.fit(x, t, y_f, num_epochs=1, batch_size=x.shape[0], print_every_epochs=2)

    ite_preds, y0_preds, y1_preds = initialized_model.estimate_causal_effect(x, n_samples_u=5)
    
    n_samples = x.shape[0]
    output_dim = initialized_model.y_dim # y_dim from model params
    assert ite_preds.shape == (n_samples, output_dim)
    assert y0_preds.shape == (n_samples, output_dim)
    assert y1_preds.shape == (n_samples, output_dim)

# pytest causal_inference_project/tests/test_srcvae.py
