"""
TARNet — Treatment-Agnostic Representation Network.

A simpler baseline for ITE estimation: a shared representation network
followed by two treatment-specific outcome heads.  No distributional
balancing (no IPM / MMD).  Serves as the baseline that CFRNet and
MIM-DRCFR build upon.

Reference:
    Shalit, U., Johansson, F. D., & Sontag, D. (2017). "Estimating
    individual treatment effect: generalization bounds and algorithms." ICML.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


class TARNetModel(nn.Module):
    """TARNet: shared representation → two outcome heads."""

    def __init__(self, input_dim, hidden_dims_shared, hidden_dims_head,
                 output_dim=1, dropout=0.0,
                 learning_rate=1e-3, weight_decay=1e-4):
        super().__init__()
        layers = []
        d = input_dim
        for h in hidden_dims_shared:
            layers.append(nn.Linear(d, h))
            layers.append(nn.ELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = h
        self.shared = nn.Sequential(*layers)

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

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate,
                                    weight_decay=weight_decay)

    def forward(self, x):
        phi = self.shared(x)
        return self.h0(phi), self.h1(phi)

    def fit(self, x_train, y_train, t_train, num_epochs, batch_size,
            x_val=None, y_val=None, t_val=None,
            print_every_epochs=10, patience=None):
        """Train TARNet.  Returns history dict."""
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

        best_val = float('inf')
        no_imp = 0
        best_state = None
        history = {'train_loss': [], 'val_loss': []}

        for epoch in range(num_epochs):
            self.train()
            ep_loss = 0.0
            for bx, by, bt in dl:
                self.optimizer.zero_grad()
                y0, y1 = self.forward(bx)
                t_bool = bt.squeeze().bool()
                loss = torch.tensor(0.0, device=self.device)
                if (~t_bool).any():
                    loss = loss + F.mse_loss(y0[~t_bool], by[~t_bool])
                if t_bool.any():
                    loss = loss + F.mse_loss(y1[t_bool], by[t_bool])
                loss.backward()
                self.optimizer.step()
                ep_loss += loss.item()
            history['train_loss'].append(ep_loss / len(dl))

            vl = None
            if val_dl:
                self.eval()
                vloss = 0.0
                with torch.no_grad():
                    for bx, by, bt in val_dl:
                        y0, y1 = self.forward(bx)
                        t_bool = bt.squeeze().bool()
                        l = torch.tensor(0.0, device=self.device)
                        if (~t_bool).any():
                            l = l + F.mse_loss(y0[~t_bool], by[~t_bool])
                        if t_bool.any():
                            l = l + F.mse_loss(y1[t_bool], by[t_bool])
                        vloss += l.item()
                vl = vloss / len(val_dl)
                history['val_loss'].append(vl)

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
                msg = f"Epoch {epoch+1}/{num_epochs} - Loss: {history['train_loss'][-1]:.4f}"
                if vl is not None:
                    msg += f"  Val: {vl:.4f}"
                print(msg)

        return history

    def predict_ite(self, x):
        self.eval()
        x = x.to(self.device)
        with torch.no_grad():
            y0, y1 = self.forward(x)
        return y1 - y0

    def predict_potential_outcomes(self, x):
        self.eval()
        x = x.to(self.device)
        with torch.no_grad():
            y0, y1 = self.forward(x)
        return y0, y1
