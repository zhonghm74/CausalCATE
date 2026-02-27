"""
Semi-supervised SRCVAE extension.

When some observations have missing outcomes (Y), the standard ELBO
cannot be computed.  This module provides a training routine that:
- Uses the full ELBO for labelled data (x, t, y observed).
- Uses a marginalised ELBO for unlabelled data (x, t observed; y missing),
  where y is sampled from the auxiliary predictor q(y|x,t).
"""

import torch
from torch.utils.data import TensorDataset, DataLoader
from copy import deepcopy


def semi_supervised_fit(
    model, x_lab, t_lab, y_lab, x_unlab, t_unlab,
    num_epochs, batch_size, alpha_unsup=1.0,
    print_every_epochs=10, kl_warmup_epochs=0,
):
    """Train an SRCVAEModel in semi-supervised fashion.

    For each mini-batch the loss is:
        L = L_lab + α_unsup · L_unlab

    where L_unlab uses pseudo-outcomes ŷ ~ q(y|x,t) from the auxiliary
    network to fill in the missing y.

    Args:
        model: An SRCVAEModel instance.
        x_lab, t_lab, y_lab: Labelled data tensors.
        x_unlab, t_unlab: Unlabelled data tensors (y missing).
        num_epochs: Training epochs.
        batch_size: Mini-batch size.
        alpha_unsup: Weight for the unsupervised loss term.
        print_every_epochs: Logging frequency.
        kl_warmup_epochs: Linear KL annealing warmup.

    Returns:
        history dict with 'lab_loss' and 'unlab_loss' per epoch.
    """
    device = model.device
    x_lab = x_lab.to(device)
    t_lab = t_lab.to(device)
    y_lab = y_lab.to(device)
    x_unlab = x_unlab.to(device)
    t_unlab = t_unlab.to(device)

    if t_lab.ndim == 1: t_lab = t_lab.unsqueeze(1)
    if y_lab.ndim == 1: y_lab = y_lab.unsqueeze(1)
    if t_unlab.ndim == 1: t_unlab = t_unlab.unsqueeze(1)

    lab_ds = TensorDataset(x_lab, t_lab, y_lab)
    lab_dl = DataLoader(lab_ds, batch_size=batch_size, shuffle=True)

    unlab_ds = TensorDataset(x_unlab, t_unlab)
    unlab_dl = DataLoader(unlab_ds, batch_size=batch_size, shuffle=True)

    history = {'lab_loss': [], 'unlab_loss': []}

    for epoch in range(num_epochs):
        model.train()
        kl_w = min(1.0, (epoch + 1) / max(kl_warmup_epochs, 1)) if kl_warmup_epochs > 0 else 1.0

        ep_lab = 0.0
        ep_unlab = 0.0
        n_lab = 0
        n_unlab = 0

        unlab_iter = iter(unlab_dl)

        for bx, bt, by in lab_dl:
            model.optimizer.zero_grad()

            # Labelled loss
            fwd = model._unpack_forward(model.forward(bx, bt, by))
            (u_m, u_lv, v_m, v_lv, xr, tr, yr, ta, ya, xlv, ylv) = fwd
            loss_lab, _ = model.compute_loss(
                bx, bt, by, u_m, u_lv, v_m, v_lv,
                xr, tr, yr, ta, ya, xlv, ylv, kl_weight=kl_w)

            # Unlabelled loss (cycle the iterator)
            try:
                bx_u, bt_u = next(unlab_iter)
            except StopIteration:
                unlab_iter = iter(unlab_dl)
                bx_u, bt_u = next(unlab_iter)

            with torch.no_grad():
                y_pseudo = model.aux_qyxt(bx_u, bt_u).detach()

            fwd_u = model._unpack_forward(model.forward(bx_u, bt_u, y_pseudo))
            (u_m2, u_lv2, v_m2, v_lv2, xr2, tr2, yr2, ta2, ya2, xlv2, ylv2) = fwd_u
            loss_unlab, _ = model.compute_loss(
                bx_u, bt_u, y_pseudo, u_m2, u_lv2, v_m2, v_lv2,
                xr2, tr2, yr2, ta2, ya2, xlv2, ylv2, kl_weight=kl_w)

            total = loss_lab + alpha_unsup * loss_unlab
            total.backward()
            model.optimizer.step()

            ep_lab += loss_lab.item()
            ep_unlab += loss_unlab.item()
            n_lab += 1
            n_unlab += 1

        history['lab_loss'].append(ep_lab / max(n_lab, 1))
        history['unlab_loss'].append(ep_unlab / max(n_unlab, 1))

        if (epoch + 1) % print_every_epochs == 0:
            print(f"Epoch {epoch+1}/{num_epochs} - "
                  f"Lab: {history['lab_loss'][-1]:.4f}  "
                  f"Unlab: {history['unlab_loss'][-1]:.4f}")

    return history
