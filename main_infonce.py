# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python (fmri)
#     language: python
#     name: fmri
# ---

# +
# We want to maximize the dot product
# Step 1: Add the position encoding (content + position)
# INFONCE: A and B are complete unconstrained, instead of optimizing over pi
# Check: Does the model localize the right memory

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import time
import os
import copy
from typing import Optional

class DAM(nn.Module):
    def __init__(self, N, H, eta=1.0, init_std=1e-2):
        """
        Args:
            N: Sequence length (bits).
            H: Hidden dimension (number of heads).
            eta: Inverse temperature parameter.
        """
        super().__init__()
        self.N = N
        self.H = H
        self.eta = eta
        self.init_std = init_std

        # A(n) parameters: (N, H, N). We will mask unused parts.
        # We need independent parameters for each length n.
        # For length n (predicting n-th bit, 0-indexed), we use inputs 0..n-1.
        # So we have separate weights for each n.
        self.A_logits = nn.Parameter(torch.randn(N, H, N) * self.init_std) # small positive weights

        # B parameters: (H, N).
        self.B_logits = nn.Parameter(torch.randn(H, N) * self.init_std) # small positive weights

        # Dataset (acts as the "memory bank"): (K, N) with entries in {-1, +1}
        # Stored as a buffer so `model.to(device)` moves it with the module.
        self.register_buffer("dataset", torch.zeros(0, N))

    def set_dataset(self, sequences: torch.Tensor):
        """
        Set the retrieval set (dataset) used for InfoNCE-style training.

        Args:
            sequences: (K, N) tensor of binary sequences (-1, +1).
        """
        if sequences.ndim != 2 or sequences.shape[1] != self.N:
            raise ValueError(f"Expected sequences with shape (K, {self.N}), got {tuple(sequences.shape)}")
        self.dataset = sequences.detach().to(self.dataset.device)

    def get_A(self, n):
        """
        Returns normalized A(n) matrix of shape (H, n).
        """
        if n == 0:
            return torch.zeros(self.H, 0, device=self.A_logits.device)

        # Select params for predictng n+1 th bit (index n).
        # We use slice 0:n because we attend to first n bits.
        logits = self.A_logits[n, :, :n] # (H, n)
        return F.softmax(logits, dim=-1)

    def get_B(self):
        """
        Returns normalized B matrix of shape (H, N).
        """
        return F.softmax(self.B_logits, dim=-1)

    def retrieval_logits(self, zeta: torch.Tensor, n: int, phi_all: Optional[torch.Tensor] = None):
        """
        Compute K-way logits for retrieving the correct sequence from the dataset given a cue.

        Args:
            zeta: (Batch, N) full sequences (we only use the prefix zeta[:, :n]).
            n: int, prefix length (0..N-1). We use cue zeta[:, :n].
            phi_all: optional precomputed dataset keys (K, H).

        Returns:
            logits: (Batch, K) where logits[b, mu] = eta * <hat_phi_b, phi_mu>
        """
        if self.dataset.numel() == 0:
            raise RuntimeError("Dataset is empty. Call set_dataset(sequences) before training/evaluation.")

        A_n = self.get_A(n)  # (H, n)
        context = zeta[:, :n]  # (Batch, n)
        hat_phi = torch.einsum("bi,hi->bh", context, A_n)  # (Batch, H)

        if phi_all is None:
            # phi_all: (K, H) = (H, N) x (K, N)^T
            phi_all = torch.einsum("hn,kn->kh", self.get_B(), self.dataset)

        logits = self.eta * torch.einsum("bh,kh->bk", hat_phi, phi_all)
        return logits

    def train_batch(self, sequences: torch.Tensor, indices: torch.Tensor, optimizer):
        """
        InfoNCE-style training: maximize probability of retrieving the correct sequence index.

        For each prefix length n, compute logits over all K dataset sequences and apply
        multiclass cross entropy against the true index.

        Args:
            sequences: (Batch, N) sampled sequences from the dataset.
            indices: (Batch,) int64 indices into the dataset identifying the correct class.
            optimizer: torch optimizer (used only when model is in training mode).

        Returns:
            loss: float
            accuracy: float (top-1 retrieval accuracy averaged over n)
        """
        if self.dataset.numel() == 0:
            raise RuntimeError("Dataset is empty. Call set_dataset(sequences) before training/evaluation.")

        if indices.dtype != torch.long:
            indices = indices.long()

        if self.training:
            if optimizer is None:
                raise ValueError("optimizer must be provided when model is in training mode.")
            optimizer.zero_grad()

        loss_batch = 0.0
        total_accuracy = 0.0

        # Precompute dataset keys once per batch: phi_all is (K, H)
        phi_all = torch.einsum("hn,kn->kh", self.get_B(), self.dataset)

        for n in range(1, self.N):
            logits = self.retrieval_logits(sequences, n, phi_all=phi_all)  # (Batch, K)
            loss_n = F.cross_entropy(logits, indices)
            loss_batch = loss_batch + loss_n

            preds = logits.argmax(dim=1)
            total_accuracy += (preds == indices).float().mean()

        loss_final = loss_batch / (self.N - 1)

        if self.training:
            loss_final.backward()
            optimizer.step()

        return loss_final.item(), (total_accuracy / (self.N - 1)).item()

def initialize_record(args):
    record = {
        "args": args,
        "logs": [],
    }
    return record


@torch.no_grad()
def compute_model_stats(model: DAM, args):
    """
    Computes per-position summary stats for monitoring.

    Returns:
        dict with:
          - A_logits_mean: (N,) cpu tensor
          - B_logits_mean: (N,) cpu tensor
          - A_logits_mean_mean: float
          - B_logits_mean_mean: float
          - A_sharpness: float
          - A_correlation: (N, H, H) cpu tensor
          - A_correlation_offdiag_mean: (N,) cpu tensor
          - A_correlation_offdiag_mean_mean: float
    """
    # A_logits: (N, H, N) -> mean over last dim -> (N, H) -> mean over heads -> (N,)
    A_logits_mean = model.A_logits.detach().mean(dim=-1).mean(dim=1).cpu()
    A_logits_mean_mean = A_logits_mean.mean().item()

    # B_logits: (H, N) -> mean over heads -> (N,)
    B_logits_mean = model.B_logits.detach().mean(dim=0).cpu()
    B_logits_mean_mean = B_logits_mean.mean().item()

    # Just take mean max attention as proxy for "sharpness"
    A_sharpness = 0.0
    for n in range(1, args.N):
        A_n = model.get_A(n)  # (H, n)
        A_sharpness += A_n.max(dim=1).values.mean().item()
    A_sharpness /= (args.N - 1)

    # Pearson correlation across the last dim (N) between heads (rows).
    # X: (N, H, N)
    X = model.A_logits.detach()  # shape: (N, H, N)
    X = X - X.mean(dim=-1, keepdim=True)  # center over last dim
    X = X / (X.std(dim=-1, keepdim=True, unbiased=False) + 1e-8)  # normalize variance
    A_correlation = torch.matmul(X, X.transpose(-1, -2)) / X.shape[-1]  # (N, H, H)

    # Off-diagonal mean per n: (N,)
    offdiag_mask = ~torch.eye(args.H, dtype=torch.bool, device=A_correlation.device)
    A_correlation_offdiag_mean = (
        A_correlation.masked_select(offdiag_mask).view(args.N, -1).mean(dim=1).cpu()
    ) # shape: 
    A_correlation_offdiag_mean_mean = A_correlation_offdiag_mean.mean().item()

    return {
        "A_logits_mean": A_logits_mean,
        "B_logits_mean": B_logits_mean,
        "A_logits_mean_mean": A_logits_mean_mean,
        "B_logits_mean_mean": B_logits_mean_mean,
        "A_sharpness": A_sharpness,
        # "A_correlation": A_correlation.detach().cpu(),
        "A_correlation_offdiag_mean": A_correlation_offdiag_mean,
        "A_correlation_offdiag_mean_mean": A_correlation_offdiag_mean_mean,
    }


# +

if __name__ == "__main__":
    print("Running DAM...")

    parser = argparse.ArgumentParser()
    parser.add_argument("--eta", type=float, default=1e1)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--INIT_STD", type=float, default=1e-1)
    parser.add_argument("--N", type=int, default=50)
    parser.add_argument("--H", type=int, default=30)
    parser.add_argument("--K", type=int, default=500)
    parser.add_argument("--BATCH_SIZE", type=int, default=50)
    parser.add_argument("--NUM_STEPS", type=int, default=50000)
    parser.add_argument("--NUM_ITERS_PER_LOG", type=int, default=100)
    parser.add_argument("--savedir", type=str, default="./results")
    parser.add_argument("--prefix", type=str, default="")
    parser.add_argument("--is_freeze_A", type=str, default="False")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (e.g. 'cuda', 'cuda:0', 'cpu'). Defaults to cuda if available.",
    )
    try:
        args = parser.parse_args()
    except:
        args = parser.parse_args([])

    device = (
        torch.device(args.device)
        if args.device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")
    print(f"args: {args}")

    # Create savedir if it doesn't exist
    os.makedirs(f"{args.savedir}/{args.prefix}", exist_ok=True) 
    exp_name = f"{args.prefix}/dam_{time.time()}"
    model = DAM(args.N, args.H, eta=args.eta, init_std=args.INIT_STD)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    record = initialize_record(args)

    # Generate random sequences
    sequences = torch.sign(torch.randn(args.K, args.N, device=device))
    model.set_dataset(sequences)

    print(f"Dataset initialized with shape: {sequences.shape}")

    # Training Loop Simulation
    model.train()
    for step in range(args.NUM_STEPS):

        # Sample batch from sequences
        indices = torch.randint(0, args.K, (args.BATCH_SIZE,), device=device)
        batch = sequences[indices]

        loss, accuracy = model.train_batch(batch, indices, optimizer)
        # Compute stats
        stats = compute_model_stats(model, args)
        logs = {
            "step": step,
            "loss": loss,
            "accuracy": accuracy,
            **stats,
            "phase": "train",
        }
        record["logs"].append(logs)

        if step % args.NUM_ITERS_PER_LOG == 0:
            print(f"Step {step}: Loss = {loss:.4f}, Accuracy = {accuracy:.4f}")
            
            print(
                f"Stats: A_logits_mean={stats['A_logits_mean_mean']:.4f}, "
                f"B_logits_mean={stats['B_logits_mean_mean']:.4f}, "
                f"A_corr_offdiag={stats['A_correlation_offdiag_mean_mean']:.4f}, "
                f"A_sharp={stats['A_sharpness']:.4f}"
            )

    print("Verification complete.")
    # save model.state_dict() 
    record["model_state_dict"] = copy.deepcopy(model.state_dict())

    # resample sequences 
    sequences = torch.sign(torch.randn(args.K, args.N, device=device))
    print(f"Sequences resampled with shape: {sequences.shape}")
    model.set_dataset(sequences)
    if args.is_freeze_A == "FreezeA":
        model.A_logits.requires_grad = False
        optimizer = torch.optim.Adam([model.B_logits], lr=args.lr)
    elif args.is_freeze_A == "BothAB":
        model.eval() 
    elif args.is_freeze_A == "LearnAB":
        pass

    for step in range(args.NUM_STEPS):

        # Sample batch from sequences
        indices = torch.randint(0, args.K, (args.BATCH_SIZE,), device=device)
        batch = sequences[indices]

        loss, accuracy = model.train_batch(batch, indices, optimizer)

        # Compute stats
        stats = compute_model_stats(model, args)
        logs = {
            "step": step,
            "loss": loss,
            "accuracy": accuracy,
            **stats,
            "phase": "eval",
        }
        record["logs"].append(logs)
    # save model.state_dict() as A_logits and B_logits
    torch.save(record, f"{args.savedir}/{exp_name}.pt")
    