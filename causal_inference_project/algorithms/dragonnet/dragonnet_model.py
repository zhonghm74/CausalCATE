"""
DragonNet — Adapting Neural Networks for the Estimation of Treatment Effects.

Extends TARNet with a third propensity-score head and targeted
regularization, ensuring that the shared representation captures all
confounding information necessary for unbiased effect estimation.

Architecture::

        x
        │
    ┌───┴───┐
    │ shared │  Φ(x)
    └───┬───┘
    ┌───┼───────┐
    │   │       │
   h₀  h₁     ε(x)
  Y(0) Y(1)  P(T=1|X)

Loss = L_outcome + α·L_propensity + β·L_targeted

Reference:
    Shi, C., Blei, D. M., & Veitch, V. (2019). "Adapting Neural Networks
    for the Estimation of Treatment Effects." NeurIPS.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


class DragonNetModel(nn.Module):
    """DragonNet: TARNet + propensity head + targeted regularization."""

    def __init__(self, input_dim, hidden_dims_shared, hidden_dims_head,
                 output_dim=1, dropout=0.0,
                 alpha=1.0, beta=1.0,
                 learning_rate=1e-3, weight_decay=1e-4):
        """
        Args:
            input_dim: Number of input features.
            hidden_dims_shared: Hidden layer sizes for shared representation Φ.
            hidden_dims_head: Hidden layer sizes for each outcome head.
            output_dim: Outcome dimension (typically 1).
            dropout: Dropout rate for regularization.
            alpha: Weight for propensity-score loss.
            beta: Weight for targeted regularization loss.
            learning_rate: Adam learning rate.
            weight_decay: Adam L2 penalty.
        """
        super().__init__()

        self.alpha = alpha
        self.beta = beta

        # --- shared representation ---
        layers = []
        d = input_dim
        for h in hidden_dims_shared:
            layers += [nn.Linear(d, h), nn.ELU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = h
        self.shared = nn.Sequential(*layers)
        self._repr_dim = d

        # --- outcome heads ---
        def _head(in_d):
            ls = []
            cd = in_d
            for h in hidden_dims_head:
                ls += [nn.Linear(cd, h), nn.ELU()]
                if dropout > 0:
                    ls.append(nn.Dropout(dropout))
                cd = h
            ls.append(nn.Linear(cd, output_dim))
            return nn.Sequential(*ls)

        self.h0 = _head(d)
        self.h1 = _head(d)

        # --- propensity head (ε) ---
        self.propensity_head = nn.Sequential(
            nn.Linear(d, d // 2 if d > 2 else 1),
            nn.ELU(),
            nn.Linear(d // 2 if d > 2 else 1, 1),
        )

        # --- epsilon layer for targeted reg (learnable scalar) ---
        self.epsilon = nn.Parameter(torch.zeros(1))

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate,
                                    weight_decay=weight_decay)

    def forward(self, x):
        """Returns (y0_pred, y1_pred, propensity_logit)."""
        phi = self.shared(x)
        y0 = self.h0(phi)
        y1 = self.h1(phi)
        eps_logit = self.propensity_head(phi)
        return y0, y1, eps_logit

    def _compute_loss(self, bx, by, bt):
        """Compute the three DragonNet loss components.

        Returns:
            (total_loss, outcome_loss, propensity_loss, targeted_loss)
        """
        y0, y1, eps_logit = self.forward(bx)
        t_float = bt.float()
        t_bool = bt.squeeze(-1).bool() if bt.ndim > 1 else bt.bool()

        # 1. Factual outcome loss (MSE on observed arm only)
        loss_out = torch.tensor(0.0, device=self.device)
        if (~t_bool).any():
            loss_out = loss_out + F.mse_loss(y0[~t_bool], by[~t_bool])
        if t_bool.any():
            loss_out = loss_out + F.mse_loss(y1[t_bool], by[t_bool])

        # 2. Propensity loss (BCE)
        loss_prop = F.binary_cross_entropy_with_logits(
            eps_logit, t_float if t_float.ndim == eps_logit.ndim else t_float.unsqueeze(-1))

        # 3. Targeted regularization (TMLE-style)
        eps_prob = torch.sigmoid(eps_logit).squeeze(-1)  # (B,)
        eps_prob = eps_prob.clamp(0.01, 0.99)
        t_sq = bt.squeeze(-1).float() if bt.ndim > 1 else bt.float()

        y_pred_factual = t_sq * y1.squeeze(-1) + (1 - t_sq) * y0.squeeze(-1)
        y_true_sq = by.squeeze(-1) if by.ndim > 1 else by

        # Doubly-robust-style pseudo-residual
        h_term = t_sq / eps_prob - (1 - t_sq) / (1 - eps_prob)
        loss_targeted = F.mse_loss(
            y_pred_factual + self.epsilon * h_term,
            y_true_sq,
        )

        total = loss_out + self.alpha * loss_prop + self.beta * loss_targeted
        return total, loss_out, loss_prop, loss_targeted

    def fit(self, x_train, y_train, t_train, num_epochs, batch_size,
            x_val=None, y_val=None, t_val=None,
            print_every_epochs=10, patience=None, lr_scheduler=None):
        """Train DragonNet.  Returns history dict."""
        x_train = x_train.to(self.device)
        y_train = y_train.to(self.device)
        t_train = t_train.to(self.device)
        if y_train.ndim == 1: y_train = y_train.unsqueeze(1)
        if t_train.ndim == 1: t_train = t_train.unsqueeze(1)

        dl = DataLoader(TensorDataset(x_train, y_train, t_train),
                        batch_size=batch_size, shuffle=True)

        val_dl = None
        if x_val is not None and y_val is not None and t_val is not None:
            xv = x_val.to(self.device)
            yv = y_val.to(self.device) if y_val.ndim > 1 else y_val.unsqueeze(1).to(self.device)
            tv = t_val.to(self.device) if t_val.ndim > 1 else t_val.unsqueeze(1).to(self.device)
            val_dl = DataLoader(TensorDataset(xv, yv, tv), batch_size=batch_size)

        scheduler = None
        if lr_scheduler == 'plateau' and val_dl:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=max(1, (patience or 10) // 2))
        elif lr_scheduler == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=num_epochs)

        best_val = float('inf')
        no_imp = 0
        best_state = None

        history = {
            'train_loss': [], 'train_outcome': [], 'train_propensity': [], 'train_targeted': [],
            'val_loss': [],
        }

        for epoch in range(num_epochs):
            self.train()
            ep_t = ep_o = ep_p = ep_tr = 0.0
            for bx, by, bt in dl:
                self.optimizer.zero_grad()
                total, l_out, l_prop, l_targ = self._compute_loss(bx, by, bt)
                total.backward()
                self.optimizer.step()
                ep_t += total.item()
                ep_o += l_out.item()
                ep_p += l_prop.item()
                ep_tr += l_targ.item()
            nb = len(dl)
            history['train_loss'].append(ep_t / nb)
            history['train_outcome'].append(ep_o / nb)
            history['train_propensity'].append(ep_p / nb)
            history['train_targeted'].append(ep_tr / nb)

            vl = None
            if val_dl:
                self.eval()
                vloss = 0.0
                with torch.no_grad():
                    for bx, by, bt in val_dl:
                        tl, _, _, _ = self._compute_loss(bx, by, bt)
                        vloss += tl.item()
                vl = vloss / len(val_dl)
                history['val_loss'].append(vl)

            if scheduler is not None:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau) and vl is not None:
                    scheduler.step(vl)
                elif not isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step()

            if patience and vl is not None:
                if vl < best_val - 1e-6:
                    best_val = vl
                    no_imp = 0
                    best_state = {k: v.cpu().clone() for k, v in self.state_dict().items()}
                else:
                    no_imp += 1
                if no_imp >= patience:
                    if best_state:
                        self.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})
                    break

            if (epoch + 1) % print_every_epochs == 0:
                msg = (f"Epoch {epoch+1}/{num_epochs} - Loss: {history['train_loss'][-1]:.4f} "
                       f"(out: {history['train_outcome'][-1]:.4f}, "
                       f"prop: {history['train_propensity'][-1]:.4f}, "
                       f"targ: {history['train_targeted'][-1]:.4f})")
                if vl is not None:
                    msg += f"  Val: {vl:.4f}"
                print(msg)

        return history

    def predict_ite(self, x):
        """Predict ITE = E[Y(1)|x] - E[Y(0)|x]."""
        self.eval()
        x = x.to(self.device)
        with torch.no_grad():
            y0, y1, _ = self.forward(x)
        return y1 - y0

    def predict_potential_outcomes(self, x):
        """Predict (Y(0), Y(1))."""
        self.eval()
        x = x.to(self.device)
        with torch.no_grad():
            y0, y1, _ = self.forward(x)
        return y0, y1

    def predict_propensity(self, x):
        """Predict P(T=1|x)."""
        self.eval()
        x = x.to(self.device)
        with torch.no_grad():
            _, _, eps_logit = self.forward(x)
        return torch.sigmoid(eps_logit)
