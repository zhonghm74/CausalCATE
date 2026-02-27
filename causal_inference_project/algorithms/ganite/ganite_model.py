"""
GANITE — Generative Adversarial Nets for Inference of
Individualised Treatment Effects.

Two-stage architecture:
  Stage 1 (Counterfactual block): A generator G produces counterfactual
    outcomes and a discriminator D distinguishes real from generated
    (factual, counterfactual) pairs.
  Stage 2 (ITE block): A second generator I takes x and predicts ITE,
    trained on pseudo-outcomes from stage 1.

Reference:
    Yoon, J., Jordon, J., & van der Schaar, M. (2018). "GANITE:
    Estimation of Individualized Treatment Effects Using Generative
    Adversarial Nets." ICLR.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


def _mlp(in_d, out_d, hiddens, dropout=0.0):
    layers = []
    d = in_d
    for h in hiddens:
        layers += [nn.Linear(d, h), nn.LeakyReLU(0.2)]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        d = h
    layers.append(nn.Linear(d, out_d))
    return nn.Sequential(*layers)


class GANITEModel(nn.Module):
    """GANITE: two-stage GAN for ITE estimation."""

    def __init__(self, input_dim, hidden_dims=None, noise_dim=8,
                 output_dim=1, dropout=0.0,
                 learning_rate=1e-3, weight_decay=1e-4):
        super().__init__()
        h = hidden_dims or [64, 32]
        self.noise_dim = noise_dim
        self.output_dim = output_dim

        # Stage 1: Counterfactual generator & discriminator
        self.G = _mlp(input_dim + 1 + 1 + noise_dim, output_dim, h, dropout)
        self.D = _mlp(input_dim + 1 + output_dim + output_dim, 1, h, dropout)

        # Stage 2: ITE predictor
        self.I_net = _mlp(input_dim, output_dim, h, dropout)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

        self.opt_G = optim.Adam(self.G.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.opt_D = optim.Adam(self.D.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.opt_I = optim.Adam(self.I_net.parameters(), lr=learning_rate, weight_decay=weight_decay)

    def _generate_cf(self, x, t, y_f, z):
        """Generate counterfactual outcome."""
        inp = torch.cat([x, t, y_f, z], dim=1)
        return self.G(inp)

    def fit(self, x_train, y_train, t_train, num_epochs, batch_size,
            print_every_epochs=10):
        """Two-stage training.  Returns history dict."""
        x = x_train.to(self.device)
        y = y_train.to(self.device)
        t = t_train.to(self.device)
        if y.ndim == 1: y = y.unsqueeze(1)
        if t.ndim == 1: t = t.unsqueeze(1)

        dl = DataLoader(TensorDataset(x, y, t), batch_size=batch_size, shuffle=True)

        history = {'g_loss': [], 'd_loss': [], 'i_loss': []}

        # --- Stage 1: Train G and D ---
        s1_epochs = num_epochs // 2 if num_epochs > 4 else max(num_epochs - 1, 1)
        for epoch in range(s1_epochs):
            self.train()
            eg, ed = 0.0, 0.0
            for bx, by, bt in dl:
                n = bx.shape[0]
                z = torch.randn(n, self.noise_dim, device=self.device)
                y_cf_gen = self._generate_cf(bx, bt, by, z)

                # Construct (y0, y1) pairs
                y0_real = torch.where(bt == 0, by, torch.zeros_like(by))
                y1_real = torch.where(bt == 1, by, torch.zeros_like(by))
                y0_gen = torch.where(bt == 0, by, y_cf_gen)
                y1_gen = torch.where(bt == 1, by, y_cf_gen)

                # D: real = factual pair, fake = with generated cf
                d_real = self.D(torch.cat([bx, bt, y0_real, y1_real], dim=1))
                d_fake = self.D(torch.cat([bx, bt, y0_gen.detach(), y1_gen.detach()], dim=1))
                loss_d = F.binary_cross_entropy_with_logits(d_real, torch.ones_like(d_real)) + \
                         F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake))
                self.opt_D.zero_grad()
                loss_d.backward()
                self.opt_D.step()

                # G: fool D + supervised on factual
                z2 = torch.randn(n, self.noise_dim, device=self.device)
                y_cf_gen2 = self._generate_cf(bx, bt, by, z2)
                y0_g = torch.where(bt == 0, by, y_cf_gen2)
                y1_g = torch.where(bt == 1, by, y_cf_gen2)
                d_g = self.D(torch.cat([bx, bt, y0_g, y1_g], dim=1))
                loss_g = F.binary_cross_entropy_with_logits(d_g, torch.ones_like(d_g))
                # Supervised: factual reconstruction
                y_f_pred = torch.where(bt == 1, y1_g, y0_g)
                loss_g = loss_g + F.mse_loss(y_f_pred, by)
                self.opt_G.zero_grad()
                loss_g.backward()
                self.opt_G.step()

                eg += loss_g.item()
                ed += loss_d.item()
            history['g_loss'].append(eg / len(dl))
            history['d_loss'].append(ed / len(dl))
            if (epoch + 1) % print_every_epochs == 0:
                print(f"Stage1 Epoch {epoch+1}/{s1_epochs} - G: {history['g_loss'][-1]:.4f}  D: {history['d_loss'][-1]:.4f}")

        # --- Stage 2: Train ITE predictor on G's outputs ---
        s2_epochs = num_epochs - s1_epochs
        self.G.eval()
        for epoch in range(s2_epochs):
            self.I_net.train()
            ei = 0.0
            for bx, by, bt in dl:
                n = bx.shape[0]
                with torch.no_grad():
                    z = torch.randn(n, self.noise_dim, device=self.device)
                    y_cf = self._generate_cf(bx, bt, by, z)
                    y0_pseudo = torch.where(bt == 0, by, y_cf)
                    y1_pseudo = torch.where(bt == 1, by, y_cf)
                    ite_target = y1_pseudo - y0_pseudo

                ite_pred = self.I_net(bx)
                loss_i = F.mse_loss(ite_pred, ite_target)
                self.opt_I.zero_grad()
                loss_i.backward()
                self.opt_I.step()
                ei += loss_i.item()
            history['i_loss'].append(ei / len(dl))
            if (epoch + 1) % print_every_epochs == 0:
                print(f"Stage2 Epoch {epoch+1}/{s2_epochs} - I: {history['i_loss'][-1]:.4f}")

        return history

    def predict_ite(self, x):
        self.eval()
        x = x.to(self.device)
        with torch.no_grad():
            return self.I_net(x)

    def predict_potential_outcomes(self, x):
        """Approximate Y(0), Y(1) via generator averaging."""
        self.eval()
        x = x.to(self.device)
        n = x.shape[0]
        y0s, y1s = [], []
        with torch.no_grad():
            for _ in range(20):
                z = torch.randn(n, self.noise_dim, device=self.device)
                t0 = torch.zeros(n, 1, device=self.device)
                t1 = torch.ones(n, 1, device=self.device)
                y_dummy = torch.zeros(n, self.output_dim, device=self.device)
                y0s.append(self._generate_cf(x, t1, y_dummy, z))
                y1s.append(self._generate_cf(x, t0, y_dummy, z))
        y0 = torch.stack(y0s).mean(0)
        y1 = torch.stack(y1s).mean(0)
        return y0, y1
