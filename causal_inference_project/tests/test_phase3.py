"""Tests for Phase 3 (前沿扩展) additions."""

import numpy as np
import torch
import pytest
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEVICE = torch.device("cpu")
torch.manual_seed(0)
np.random.seed(0)


# ===================================================================
# DRCFR: MC Dropout uncertainty
# ===================================================================
from algorithms.drcfr.drcfr_model import DRCFRModel


class TestMCDropout:
    def test_predict_with_uncertainty(self):
        model = DRCFRModel(5, [10], 4, 3, [6], dropout=0.3, learning_rate=1e-3)
        x = torch.randn(20, 5)
        y = torch.randn(20, 1)
        t = (torch.rand(20, 1) > 0.5).float()
        model.fit(x, y, t, num_epochs=3, batch_size=20, print_every_epochs=100)
        ite_mean, ite_std = model.predict_ite_with_uncertainty(x, n_mc=10)
        assert ite_mean.shape == (20, 1)
        assert ite_std.shape == (20, 1)
        assert (ite_std >= 0).all()

    def test_uncertainty_increases_with_ood(self):
        model = DRCFRModel(5, [10], 4, 3, [6], dropout=0.3)
        x = torch.randn(50, 5)
        y = torch.randn(50, 1)
        t = (torch.rand(50, 1) > 0.5).float()
        model.fit(x, y, t, num_epochs=5, batch_size=50, print_every_epochs=100)
        _, std_in = model.predict_ite_with_uncertainty(x[:10], n_mc=30)
        x_ood = torch.randn(10, 5) * 10
        _, std_ood = model.predict_ite_with_uncertainty(x_ood, n_mc=30)
        assert std_ood.mean().item() >= std_in.mean().item() * 0.5


# ===================================================================
# DRCFR: Attention encoder
# ===================================================================
from algorithms.drcfr.networks import AttentionEncoder


class TestAttentionEncoder:
    def test_output_shape(self):
        enc = AttentionEncoder(10, embed_dim=16, n_heads=2, n_layers=1,
                               latent_dim_zy=4, latent_dim_zs=3)
        x = torch.randn(8, 10)
        zy, zs = enc(x)
        assert zy.shape == (8, 4)
        assert zs.shape == (8, 3)

    def test_model_with_attention(self):
        model = DRCFRModel(
            input_dim=10, hidden_dims_phi=[16], latent_dim_zy=4, latent_dim_zs=3,
            hidden_dims_h=[8], encoder_type='attention',
            attn_embed_dim=16, attn_n_heads=2, attn_n_layers=1,
        )
        x = torch.randn(12, 10)
        y = torch.randn(12, 1)
        t = (torch.rand(12, 1) > 0.5).float()
        history = model.fit(x, y, t, num_epochs=2, batch_size=12, print_every_epochs=100)
        ite = model.predict_ite(x)
        assert ite.shape == (12, 1)
        assert len(history['train']['total']) == 2


# ===================================================================
# SRCVAE: Conditional prior
# ===================================================================
from algorithms.srcvae.srcvae_model import SRCVAEModel
from algorithms.srcvae.srcvae_networks import ConditionalPriorU


class TestConditionalPrior:
    def test_prior_network_shape(self):
        net = ConditionalPriorU(5, 3, [8, 6])
        x = torch.randn(10, 5)
        mean, logvar = net(x)
        assert mean.shape == (10, 3)
        assert logvar.shape == (10, 3)

    def test_model_with_conditional_prior(self):
        model = SRCVAEModel(
            x_dim=5, t_dim=1, y_dim=1, u_dim=3, v_dim=3,
            hidden_dims_encoder_u=[8], hidden_dims_encoder_v=[8],
            hidden_dims_decoder_x=[6], hidden_dims_decoder_t=[6],
            hidden_dims_decoder_y=[6], hidden_dims_aux_qtx=[6],
            hidden_dims_aux_qyxt=[6],
            alpha_x=1.0, alpha_t=1.0, alpha_y=1.0,
            beta_u=0.1, beta_v=0.1, gamma_t=1.0, gamma_y=1.0,
            learning_rate=1e-3, device=DEVICE,
            conditional_prior_u=True, hidden_dims_prior_u=[8],
        )
        assert model.conditional_prior_u
        x = torch.randn(15, 5, device=DEVICE)
        t = torch.randint(0, 2, (15,), device=DEVICE, dtype=torch.float32)
        y = torch.randn(15, device=DEVICE)
        history = model.fit(x, t, y, num_epochs=2, batch_size=15, print_every_epochs=100)
        assert len(history['train']['total_loss']) == 2


# ===================================================================
# SRCVAE: Semi-supervised
# ===================================================================
from algorithms.srcvae.srcvae_semi import semi_supervised_fit


class TestSemiSupervised:
    def test_semi_fit_runs(self):
        model = SRCVAEModel(
            x_dim=5, t_dim=1, y_dim=1, u_dim=3, v_dim=3,
            hidden_dims_encoder_u=[8], hidden_dims_encoder_v=[8],
            hidden_dims_decoder_x=[6], hidden_dims_decoder_t=[6],
            hidden_dims_decoder_y=[6], hidden_dims_aux_qtx=[6],
            hidden_dims_aux_qyxt=[6],
            alpha_x=1.0, alpha_t=1.0, alpha_y=1.0,
            beta_u=0.1, beta_v=0.1, gamma_t=1.0, gamma_y=1.0,
            learning_rate=1e-3, device=DEVICE,
        )
        x_lab = torch.randn(20, 5, device=DEVICE)
        t_lab = torch.randint(0, 2, (20,), device=DEVICE, dtype=torch.float32)
        y_lab = torch.randn(20, device=DEVICE)
        x_unlab = torch.randn(30, 5, device=DEVICE)
        t_unlab = torch.randint(0, 2, (30,), device=DEVICE, dtype=torch.float32)

        history = semi_supervised_fit(
            model, x_lab, t_lab, y_lab, x_unlab, t_unlab,
            num_epochs=3, batch_size=20, print_every_epochs=100,
        )
        assert 'lab_loss' in history
        assert 'unlab_loss' in history
        assert len(history['lab_loss']) == 3


# ===================================================================
# TARNet
# ===================================================================
from algorithms.tarnet.tarnet_model import TARNetModel


class TestTARNet:
    def test_init_and_forward(self):
        model = TARNetModel(5, [10], [6], output_dim=1)
        x = torch.randn(10, 5, device=model.device)
        y0, y1 = model(x)
        assert y0.shape == (10, 1)
        assert y1.shape == (10, 1)

    def test_fit_and_predict(self):
        model = TARNetModel(5, [10], [6], learning_rate=1e-3)
        x = torch.randn(50, 5)
        y = torch.randn(50, 1)
        t = (torch.rand(50, 1) > 0.5).float()
        history = model.fit(x, y, t, num_epochs=5, batch_size=25, print_every_epochs=100)
        assert 'train_loss' in history
        assert len(history['train_loss']) == 5
        ite = model.predict_ite(x)
        assert ite.shape == (50, 1)

    def test_potential_outcomes(self):
        model = TARNetModel(5, [10], [6])
        x = torch.randn(10, 5)
        y0, y1 = model.predict_potential_outcomes(x)
        assert y0.shape == (10, 1)
        assert y1.shape == (10, 1)

    def test_early_stopping(self):
        model = TARNetModel(5, [10], [6], learning_rate=1e-2)
        x = torch.randn(80, 5)
        y = torch.randn(80, 1)
        t = (torch.rand(80, 1) > 0.5).float()
        xv = torch.randn(20, 5)
        yv = torch.randn(20, 1)
        tv = (torch.rand(20, 1) > 0.5).float()
        history = model.fit(x, y, t, num_epochs=200, batch_size=40,
                            x_val=xv, y_val=yv, t_val=tv,
                            patience=5, print_every_epochs=1000)
        assert len(history['train_loss']) <= 200


# ===================================================================
# GANITE
# ===================================================================
from algorithms.ganite.ganite_model import GANITEModel


class TestGANITE:
    def test_init(self):
        model = GANITEModel(5, hidden_dims=[10, 8], noise_dim=4)
        assert model.noise_dim == 4

    def test_fit(self):
        model = GANITEModel(5, hidden_dims=[10], noise_dim=4, learning_rate=1e-3)
        x = torch.randn(40, 5)
        y = torch.randn(40, 1)
        t = (torch.rand(40, 1) > 0.5).float()
        history = model.fit(x, y, t, num_epochs=6, batch_size=20, print_every_epochs=100)
        assert 'g_loss' in history
        assert 'i_loss' in history
        assert len(history['g_loss']) > 0

    def test_predict_ite(self):
        model = GANITEModel(5, hidden_dims=[10], noise_dim=4)
        x = torch.randn(30, 5)
        y = torch.randn(30, 1)
        t = (torch.rand(30, 1) > 0.5).float()
        model.fit(x, y, t, num_epochs=4, batch_size=30, print_every_epochs=100)
        ite = model.predict_ite(x)
        assert ite.shape == (30, 1)

    def test_predict_potential_outcomes(self):
        model = GANITEModel(5, hidden_dims=[10], noise_dim=4)
        x = torch.randn(10, 5)
        y = torch.randn(10, 1)
        t = (torch.rand(10, 1) > 0.5).float()
        model.fit(x, y, t, num_epochs=4, batch_size=10, print_every_epochs=100)
        y0, y1 = model.predict_potential_outcomes(x)
        assert y0.shape == (10, 1)
        assert y1.shape == (10, 1)
