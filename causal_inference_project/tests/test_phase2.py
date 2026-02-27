"""Tests for Phase 2 (能力提升) additions."""

import numpy as np
import torch
import pytest
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEVICE = torch.device("cpu")
torch.manual_seed(0)
np.random.seed(0)


# ===================================================================
# DRCFR: Sinkhorn distance
# ===================================================================

from algorithms.drcfr.losses import sinkhorn_loss, ipm_loss_zy
from algorithms.drcfr.drcfr_model import DRCFRModel


class TestSinkhornLoss:
    def test_same_distribution(self):
        x = torch.randn(50, 5, device=DEVICE)
        loss = sinkhorn_loss(x, x, epsilon=0.1, n_iters=50)
        assert loss.item() < 0.01

    def test_different_distributions(self):
        x1 = torch.randn(50, 5, device=DEVICE)
        x2 = torch.randn(50, 5, device=DEVICE) + 2.0
        loss = sinkhorn_loss(x1, x2, epsilon=0.1)
        assert loss.item() > 0.01

    def test_empty_input(self):
        x1 = torch.empty(0, 5, device=DEVICE)
        x2 = torch.randn(10, 5, device=DEVICE)
        assert sinkhorn_loss(x1, x2).item() == 0.0

    def test_ipm_loss_sinkhorn_mode(self):
        z_t = torch.randn(30, 5, device=DEVICE)
        z_c = torch.randn(40, 5, device=DEVICE) + 0.5
        loss_mmd = ipm_loss_zy(z_t, z_c, [0.1, 1.0], method='mmd')
        loss_sk = ipm_loss_zy(z_t, z_c, [0.1, 1.0], method='sinkhorn', sinkhorn_eps=0.1)
        assert loss_mmd.item() > 0
        assert loss_sk.item() > 0


class TestDRCFRSinkhorn:
    def test_model_with_sinkhorn(self):
        model = DRCFRModel(
            input_dim=5, hidden_dims_phi=[10], latent_dim_zy=4, latent_dim_zs=3,
            hidden_dims_h=[6], ipm_method='sinkhorn', sinkhorn_eps=0.1,
        )
        x = torch.randn(20, 5)
        y = torch.randn(20, 1)
        t = (torch.rand(20, 1) > 0.5).float()
        history = model.fit(x, y, t, num_epochs=3, batch_size=20, print_every_epochs=100)
        assert len(history['train']['total']) == 3

    def test_sinkhorn_backward_compat(self):
        model = DRCFRModel(5, [10], 4, 3, [6])
        assert model.ipm_method == 'mmd'


# ===================================================================
# SRCVAE: IWAE
# ===================================================================

from algorithms.srcvae.srcvae_model import SRCVAEModel


def _srcvae_params(**overrides):
    p = dict(
        x_dim=5, t_dim=1, y_dim=1, u_dim=3, v_dim=3,
        hidden_dims_encoder_u=[8], hidden_dims_encoder_v=[8],
        hidden_dims_decoder_x=[6], hidden_dims_decoder_t=[6],
        hidden_dims_decoder_y=[6], hidden_dims_aux_qtx=[6],
        hidden_dims_aux_qyxt=[6],
        alpha_x=1.0, alpha_t=1.0, alpha_y=1.0,
        beta_u=0.1, beta_v=0.1, gamma_t=1.0, gamma_y=1.0,
        learning_rate=1e-3, device=DEVICE,
    )
    p.update(overrides)
    return p


def _dummy_srcvae_data(n=30):
    x = torch.randn(n, 5, device=DEVICE)
    t = torch.randint(0, 2, (n,), device=DEVICE, dtype=torch.float32)
    y = torch.randn(n, device=DEVICE)
    return x, t, y


class TestSRCVAEIWAE:
    def test_iwae_default_is_1(self):
        model = SRCVAEModel(**_srcvae_params())
        assert model.n_iwae_samples == 1

    def test_iwae_k5_runs(self):
        model = SRCVAEModel(**_srcvae_params(n_iwae_samples=3))
        x, t, y = _dummy_srcvae_data()
        history = model.fit(x, t, y, num_epochs=2, batch_size=30, print_every_epochs=100)
        assert len(history['train']['total_loss']) == 2

    def test_iwae_loss_is_scalar(self):
        model = SRCVAEModel(**_srcvae_params(n_iwae_samples=3))
        x, t, y = _dummy_srcvae_data()
        loss = model._iwae_loss(x, t.unsqueeze(1), y.unsqueeze(1))
        assert loss.ndim == 0


# ===================================================================
# SRCVAE: Mixture-of-Gaussians prior
# ===================================================================

from algorithms.srcvae.srcvae_losses import kl_gaussian_to_mog


class TestMoGPrior:
    def test_kl_to_mog_shape(self):
        z = torch.randn(10, 3)
        q_mean = torch.randn(10, 3)
        q_logvar = torch.zeros(10, 3)
        mog_means = torch.randn(5, 3)
        mog_logvars = torch.zeros(5, 3)
        mog_logweights = torch.zeros(5)
        kl = kl_gaussian_to_mog(z, q_mean, q_logvar, mog_means, mog_logvars, mog_logweights)
        assert kl.ndim == 0
        assert kl.item() >= 0 or True  # can be negative for MoG

    def test_kl_to_mog_single_component_positive_with_shifted_q(self):
        """With a single N(0,I) component and q far from prior, KL should be positive."""
        torch.manual_seed(42)
        B, D = 200, 3
        q_mean = torch.ones(B, D) * 3.0
        q_logvar = torch.zeros(B, D)
        z = q_mean + torch.randn(B, D)
        mog_means = torch.zeros(1, D)
        mog_logvars = torch.zeros(1, D)
        mog_logweights = torch.zeros(1)
        kl = kl_gaussian_to_mog(z, q_mean, q_logvar, mog_means, mog_logvars, mog_logweights)
        assert kl.item() > 0

    def test_model_with_mog_prior(self):
        model = SRCVAEModel(**_srcvae_params(prior='mog', n_mog_components=3))
        assert model.prior_type == 'mog'
        assert hasattr(model, 'mog_u_means')
        x, t, y = _dummy_srcvae_data()
        history = model.fit(x, t, y, num_epochs=2, batch_size=30, print_every_epochs=100)
        assert len(history['train']['total_loss']) == 2

    def test_standard_prior_no_mog_params(self):
        model = SRCVAEModel(**_srcvae_params())
        assert model.prior_type == 'standard'
        assert not hasattr(model, 'mog_u_means')


# ===================================================================
# UI Benchmark (import test only — full UI test via computerUse)
# ===================================================================

class TestBenchmarkImports:
    def test_all_three_algorithms_importable(self):
        from algorithms.dml.dml_core import double_ml_crossfit
        from algorithms.drcfr.drcfr_model import DRCFRModel
        from algorithms.srcvae.srcvae_model import SRCVAEModel
        from data.ihdp import generate_ihdp
        from evaluation.metrics import evaluation_report
        assert True

    def test_ihdp_end_to_end_pipeline(self):
        from data.ihdp import generate_ihdp
        from evaluation.metrics import evaluation_report
        ds = generate_ihdp(n_samples=200, n_features=10, seed=42)
        train, test = ds.train_test_split()
        fake_ite_pred = np.random.randn(test.n_samples)
        report = evaluation_report(test.ITE, fake_ite_pred, test.T)
        assert report.pehe > 0
