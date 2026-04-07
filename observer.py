"""Bandwidth-limited persistent observer model.

The observer is a per-entity recurrent model that processes episodic
observations of a physical domain. It supports multiple accumulation
modes to test whether temporal memory helps prediction.

Reference: "Persistence Structure of Bandwidth-Limited Observation"
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PersistentObserver(nn.Module):
    """GRU-based bandwidth-limited observer.

    Parameters
    ----------
    obs_dim : int
        Dimensionality of per-variable observation vectors.
    query_dim : int
        Dimensionality of query vectors.
    n_vars : int
        Number of variables per episode.
    n_ops : int
        Number of operator classes for edge prediction.
    hidden_dim : int
        Width of encoder layers.
    latent_dim : int
        Dimensionality of per-variable latent state.
    target_dim : int
        Dimensionality of prediction targets.
    accum_mode : str
        Temporal accumulation mechanism: "gru" (persistent) or "last"
        (memoryless baseline).
    """

    def __init__(self, *, obs_dim, query_dim, n_vars, n_ops,
                 hidden_dim=128, latent_dim=64, target_dim=1,
                 accum_mode="gru"):
        super().__init__()
        self.n_vars = n_vars
        self.n_ops = n_ops
        self.latent_dim = latent_dim
        self.target_dim = target_dim
        self.accum_mode = accum_mode

        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.to_update = nn.Linear(hidden_dim, latent_dim)
        self.gru = nn.GRUCell(latent_dim, latent_dim)

        pair_dim = latent_dim * 3
        self.edge_head = nn.Sequential(
            nn.Linear(pair_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.op_head = nn.Sequential(
            nn.Linear(pair_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, n_ops),
        )
        q_dim = max(1, hidden_dim // 4)
        self.query_enc = nn.Sequential(
            nn.Linear(query_dim, q_dim), nn.ReLU(),
        )
        self.next_head = nn.Sequential(
            nn.Linear(latent_dim + hidden_dim + q_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, target_dim),
        )

    def init_latent(self, batch_size):
        return torch.zeros(batch_size, self.n_vars, self.latent_dim,
                           device=DEVICE)

    def observe(self, obs, latent):
        b, n, d = latent.shape
        upd = self.to_update(self.encoder(obs))
        if self.accum_mode == "gru":
            nxt = self.gru(upd.reshape(b * n, d), latent.reshape(b * n, d))
            return nxt.reshape(b, n, d)
        if self.accum_mode == "last":
            return upd
        raise ValueError(f"Unknown accum_mode: {self.accum_mode}")

    def predict(self, latent, obs, query):
        b, n, d = latent.shape
        left = latent[:, :, None, :].expand(b, n, n, d)
        right = latent[:, None, :, :].expand(b, n, n, d)
        pair = torch.cat([left, right, torch.abs(left - right)], dim=-1)
        edge_logits = self.edge_head(pair).squeeze(-1)
        operator_logits = self.op_head(pair)
        eye = torch.eye(n, device=edge_logits.device, dtype=torch.bool)[None]
        edge_logits = edge_logits.masked_fill(eye, -8.0)
        node_emb = self.encoder(obs)
        q_emb = self.query_enc(query)
        next_pred = self.next_head(
            torch.cat([latent, node_emb, q_emb], dim=-1))
        return {
            "edge_logits": edge_logits,
            "operator_logits": operator_logits,
            "next_pred": next_pred,
        }


def _edge_f1(logits, labels):
    pred = (torch.sigmoid(logits) > 0.5).float()
    truth = labels.float()
    tp = (pred * truth).sum()
    fp = (pred * (1.0 - truth)).sum()
    fn = ((1.0 - pred) * truth).sum()
    pr = tp / (tp + fp + 1e-8)
    rc = tp / (tp + fn + 1e-8)
    return float((2.0 * pr * rc / (pr + rc + 1e-8)).item())


def _op_accuracy(logits, labels):
    mask = labels >= 0
    if mask.sum() == 0:
        return 0.0
    return float(
        (logits.argmax(dim=-1)[mask] == labels[mask]).float().mean().item())


def loss_fn(out, target, edges, operators,
            *, edge_weight=0.35, op_weight=0.25):
    next_loss = F.mse_loss(out["next_pred"], target)
    pos = edges.sum().clamp_min(1.0)
    neg = (1.0 - edges).sum().clamp_min(1.0)
    edge_loss = F.binary_cross_entropy_with_logits(
        out["edge_logits"], edges,
        pos_weight=(neg / pos).detach())
    mask = operators >= 0
    if mask.any():
        op_loss = F.cross_entropy(
            out["operator_logits"][mask], operators[mask])
    else:
        op_loss = torch.tensor(0.0, device=DEVICE)
    return next_loss + edge_weight * edge_loss + op_weight * op_loss


def train_model(model, data, *, epochs=30, batch_size=32, lr=2e-3,
                max_observe=15, verbose=True):
    n_worlds = data["obs"].shape[0]
    total_eps = data["obs"].shape[1]
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    rng = np.random.RandomState(42)
    best_state = None
    best_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_worlds)
        losses = []
        for start in range(0, n_worlds, batch_size):
            idx = perm[start:start + batch_size]
            b = len(idx)
            n_obs = (0 if rng.random() < 0.15
                     else int(rng.randint(1, max_observe + 1)))
            latent = model.init_latent(b)
            for ei in range(n_obs):
                latent = model.observe(
                    data["obs"][idx, ei].to(DEVICE), latent)
            tep = total_eps - 1
            out = model.predict(
                latent,
                data["obs"][idx, tep].to(DEVICE),
                data["query"][idx, tep].to(DEVICE))
            loss = loss_fn(
                out,
                data["target"][idx, tep].to(DEVICE),
                data["edges"][idx].to(DEVICE),
                data["operators"][idx].to(DEVICE))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.item()))
        ml = float(np.mean(losses))
        if ml < best_loss:
            best_loss = ml
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        if verbose and (epoch + 1) % max(1, epochs // 5) == 0:
            print(f"  epoch {epoch + 1}/{epochs}  loss={ml:.4f}")
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.inference_mode()
def evaluate(model, data, *, n_observe, batch_size=32):
    n_worlds = data["obs"].shape[0]
    total_eps = data["obs"].shape[1]
    next_vals = []
    model.eval()
    for start in range(0, n_worlds, batch_size):
        end = min(start + batch_size, n_worlds)
        idx = torch.arange(start, end)
        latent = model.init_latent(len(idx))
        for ei in range(min(n_observe, total_eps - 1)):
            latent = model.observe(
                data["obs"][idx, ei].to(DEVICE), latent)
        tep = total_eps - 1
        out = model.predict(
            latent,
            data["obs"][idx, tep].to(DEVICE),
            data["query"][idx, tep].to(DEVICE))
        next_vals.append(float(F.mse_loss(
            out["next_pred"],
            data["target"][idx, tep].to(DEVICE)).item()))
    return round(float(np.mean(next_vals)), 6)
