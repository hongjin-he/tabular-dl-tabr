"""
A minimal, faiss-free illustration of TabR's core idea (differentiable retrieval
over training examples) applied to a synthetic cross-sectional factor model.

This is NOT a reproduction of the TabR paper's results and does not use the
official pipeline in bin/tabr.py (which relies on faiss + a config-driven
experiment runner). It exists to make the *mechanism* -- "predict using an
attention-weighted average of similar training examples' labels" -- runnable
and inspectable in a few seconds, before diving into the full official
pipeline described in the main README.

Usage:
    python quant_demo/retrieval_augmented_factor_demo.py
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def make_synthetic_factor_data(
    n_assets: int = 500, n_dates: int = 300, n_factors: int = 10, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = n_assets * n_dates
    x = rng.normal(size=(n, n_factors)).astype(np.float32)
    w = rng.normal(size=n_factors) * np.array([1.0 / (i + 1) for i in range(n_factors)])
    y = (x @ w + 0.4 * np.tanh(x[:, 0] * x[:, 1]) + rng.normal(scale=1.2, size=n)).astype(
        np.float32
    )
    return x, y


class TinyRetrievalAugmentedRegressor(nn.Module):
    """A stripped-down analog of TabR: encode features, attend over a bank of
    candidate (encoded feature, label) pairs from the training set, and blend
    the retrieved labels with a direct MLP prediction.
    """

    def __init__(self, n_features: int, d_embedding: int = 32, top_k: int = 16):
        super().__init__()
        self.top_k = top_k
        self.encoder = nn.Sequential(
            nn.Linear(n_features, 64), nn.ReLU(), nn.Linear(64, d_embedding)
        )
        self.direct_head = nn.Sequential(
            nn.Linear(n_features, 64), nn.ReLU(), nn.Linear(64, 1)
        )
        self.mix_logit = nn.Parameter(torch.tensor(0.0))  # learned blend weight

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.encoder(x), dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        candidate_x: torch.Tensor,
        candidate_y: torch.Tensor,
    ) -> torch.Tensor:
        q = self.encode(x)  # (batch, d)
        k = self.encode(candidate_x)  # (n_candidates, d)

        sim = q @ k.T  # (batch, n_candidates) cosine similarity
        top_sim, top_idx = sim.topk(min(self.top_k, k.shape[0]), dim=-1)
        weights = F.softmax(top_sim * 8.0, dim=-1)  # temperature-scaled attention
        retrieved_y = candidate_y[top_idx]  # (batch, top_k)
        retrieval_pred = (weights * retrieved_y).sum(dim=-1, keepdim=True)

        direct_pred = self.direct_head(x)
        alpha = torch.sigmoid(self.mix_logit)
        return alpha * retrieval_pred + (1 - alpha) * direct_pred


def main() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    X, Y = make_synthetic_factor_data()
    X = StandardScaler().fit_transform(X).astype(np.float32)

    X_trainval, X_test, Y_trainval, Y_test = train_test_split(
        X, Y, train_size=0.8, random_state=0
    )
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_trainval, Y_trainval, train_size=0.8, random_state=0
    )

    to_t = lambda a: torch.as_tensor(a, device=device)  # noqa: E731
    X_train_t, Y_train_t = to_t(X_train), to_t(Y_train)
    X_val_t, Y_val_t = to_t(X_val), to_t(Y_val)
    X_test_t, Y_test_t = to_t(X_test), to_t(Y_test)

    model = TinyRetrievalAugmentedRegressor(n_features=X.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)

    # The "candidate bank" TabR retrieves from -- here, a random subset of the
    # training set, refreshed each epoch (TabR uses the full/near-full train set
    # with faiss for efficient nearest-neighbor search at scale).
    candidate_pool_size = 2000
    batch_size = 256

    @torch.no_grad()
    def evaluate(x: torch.Tensor, y: torch.Tensor) -> tuple[float, float]:
        model.eval()
        pool_idx = torch.randperm(len(X_train_t), device=device)[:candidate_pool_size]
        pred = model(x, X_train_t[pool_idx], Y_train_t[pool_idx]).squeeze(-1)
        rmse = float(torch.sqrt(F.mse_loss(pred, y)).item())
        ic = float(spearmanr(pred.cpu().numpy(), y.cpu().numpy()).correlation)
        return rmse, ic

    rmse0, ic0 = evaluate(X_test_t, Y_test_t)
    print(f"Before training | test RMSE {rmse0:.4f} | test Rank-IC {ic0:.4f}")

    best_val = float("inf")
    patience, remaining = 10, 10
    for epoch in range(100):
        model.train()
        perm = torch.randperm(len(X_train_t), device=device)
        for start in range(0, len(perm), batch_size):
            batch_idx = perm[start : start + batch_size]
            pool_idx = torch.randperm(len(X_train_t), device=device)[:candidate_pool_size]
            optimizer.zero_grad()
            pred = model(
                X_train_t[batch_idx], X_train_t[pool_idx], Y_train_t[pool_idx]
            ).squeeze(-1)
            loss = F.mse_loss(pred, Y_train_t[batch_idx])
            loss.backward()
            optimizer.step()

        val_rmse, val_ic = evaluate(X_val_t, Y_val_t)
        improved = val_rmse < best_val
        print(f'{"*" if improved else " "} epoch {epoch:<3} val RMSE {val_rmse:.4f} val Rank-IC {val_ic:.4f}')
        if improved:
            best_val = val_rmse
            remaining = patience
        else:
            remaining -= 1
        if remaining < 0:
            break

    test_rmse, test_ic = evaluate(X_test_t, Y_test_t)
    print("\n[Summary]")
    print(f"test RMSE:    {test_rmse:.4f}")
    print(f"test Rank-IC: {test_ic:.4f}")
    print(f"learned blend weight (retrieval vs. direct MLP): alpha={torch.sigmoid(model.mix_logit).item():.3f}")


if __name__ == "__main__":
    main()
