#!/usr/bin/env python3
"""
Benchmark: O(L^2) vs O(L*K) implementations for the three hot paths changed
in this PR.  Reproduces the original logic inline so no checkpoint is needed.

Usage:
    python benchmark.py                    # CPU
    python benchmark.py --device cuda      # GPU
    python benchmark.py --lengths 50 100 200 400 600 800
"""

import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Shared utility
# ---------------------------------------------------------------------------

def gather_nodes(nodes, neighbor_idx):
    """[B, L, nv] + [B, L, K] -> [B, L, K, nv]"""
    B, L, nv = nodes.shape
    K = neighbor_idx.shape[2]
    idx = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, nv)
    return torch.gather(nodes.unsqueeze(2).expand(-1, -1, K, -1), 1, idx)


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def measure(fn, n_warmup=5, n_runs=20):
    for _ in range(n_warmup):
        fn()
        _sync()
    times = []
    for _ in range(n_runs):
        _sync()
        t0 = time.perf_counter()
        fn()
        _sync()
        times.append(time.perf_counter() - t0)
    return np.median(times) * 1e3  # ms


# ---------------------------------------------------------------------------
# 1. _get_rbf: pairwise distance RBF features
# ---------------------------------------------------------------------------

def get_rbf_before(A, B_coords, num_rbf=16):
    """Original: allocates full [B, L, L] distance matrix."""
    D = torch.sqrt(
        torch.sum((A[:, :, None, :] - B_coords[:, None, :, :]) ** 2, -1) + 1e-6
    )  # [B, L, L]
    D_mu = torch.linspace(2.0, 22.0, num_rbf, device=A.device)
    D_sigma = (22.0 - 2.0) / num_rbf
    return torch.exp(-(((D.unsqueeze(-1) - D_mu) / D_sigma) ** 2))


def get_rbf_after(A, B_coords, E_idx, num_rbf=16):
    """New: gathers K neighbours first, only computes K distances per residue."""
    D_mu = torch.linspace(2.0, 22.0, num_rbf, device=A.device).view(1, 1, 1, -1)
    D_sigma = (22.0 - 2.0) / num_rbf
    B_nbrs = gather_nodes(B_coords, E_idx)               # [B, L, K, 3]
    D = torch.sqrt(
        torch.sum((A[:, :, None, :] - B_nbrs) ** 2, -1) + 1e-6
    )                                                      # [B, L, K]
    return torch.exp(-(((D.unsqueeze(-1) - D_mu) / D_sigma) ** 2))


# ---------------------------------------------------------------------------
# 2. Causal mask: which neighbours were decoded before position i?
# ---------------------------------------------------------------------------

def causal_mask_before(decoding_order, L):
    """Original: L×L one-hot matrix + O(L^3) einsum."""
    perm = F.one_hot(decoding_order.long(), num_classes=L).float()  # [B, L, L]
    tril = 1 - torch.triu(torch.ones(L, L, device=decoding_order.device))
    return torch.einsum("ij,biq,bjp->bqp", tril, perm, perm)       # [B, L, L]


def causal_mask_after(decoding_order, E_idx):
    """New: argsort gives rank; gather neighbour ranks; scalar compare."""
    rank = torch.argsort(decoding_order, dim=1).float()             # [B, L]
    rank_nbrs = gather_nodes(rank.unsqueeze(-1), E_idx).squeeze(-1) # [B, L, K]
    return (rank_nbrs < rank.unsqueeze(-1)).float()                 # [B, L, K]


# ---------------------------------------------------------------------------
# 3. 14×14 sidechain RBF (features_decode inner loop)
# ---------------------------------------------------------------------------

def sc_rbf_before(X, X_m, E_idx, num_rbf=16, lo=2.0, hi=22.0):
    """Original: 196 sequential gather_nodes + RBF calls."""
    B, L, _, _ = X.shape
    X_m_g = gather_nodes(X_m, E_idx)                               # [B, L, K, 14]
    D_mu = torch.linspace(lo, hi, num_rbf, device=X.device).view(1, 1, 1, -1)
    D_sigma = (hi - lo) / num_rbf
    out = []
    for i in range(14):
        for j in range(14):
            nbrs = gather_nodes(X[:, :, j, :], E_idx)              # [B, L, K, 3]
            D = torch.sqrt(
                torch.sum((X[:, :, i, :].unsqueeze(2) - nbrs) ** 2, -1) + 1e-6
            )
            rbf = torch.exp(-(((D.unsqueeze(-1) - D_mu) / D_sigma) ** 2))
            rbf = rbf * X_m[:, :, i, None, None] * X_m_g[:, :, :, j, None]
            out.append(rbf)
    return torch.cat(out, dim=-1)                                   # [B, L, K, 196*num_rbf]


def sc_rbf_after(X, X_m, E_idx, num_rbf=16, lo=2.0, hi=22.0):
    """New: single [B, L, 14_i, K, 14_j] distance tensor."""
    B, L, _, _ = X.shape
    K = E_idx.shape[2]
    X_m_g = gather_nodes(X_m, E_idx)                               # [B, L, K, 14]
    D_mu = torch.linspace(lo, hi, num_rbf, device=X.device).view(1, 1, 1, 1, 1, -1)
    D_sigma = (hi - lo) / num_rbf
    X_nbrs = gather_nodes(X.view(B, L, -1), E_idx).view(B, L, K, 14, 3)
    D = torch.sqrt(
        torch.sum(
            (X[:, :, :, None, None, :] - X_nbrs[:, :, None, :, :, :]) ** 2, -1
        ) + 1e-6
    )                                                               # [B, L, 14_i, K, 14_j]
    rbf = torch.exp(-(((D.unsqueeze(-1) - D_mu) / D_sigma) ** 2))
    rbf = rbf * X_m[:, :, :, None, None, None] * X_m_g[:, :, None, :, :, None]
    return rbf.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, L, K, -1)


# ---------------------------------------------------------------------------
# Correctness check
# ---------------------------------------------------------------------------

def check_correctness(device):
    torch.manual_seed(42)
    B, L, K = 1, 64, 32
    A = torch.randn(B, L, 3, device=device)
    B_ = torch.randn(B, L, 3, device=device)
    E_idx = torch.randint(0, L, (B, L, K), device=device)

    rbf_b = get_rbf_before(A, B_)[:, :, E_idx[0], :]   # index into full matrix
    rbf_a = get_rbf_after(A, B_, E_idx)
    # rbf_b indexed via E_idx should match rbf_a
    rbf_b2 = torch.stack(
        [get_rbf_before(A, B_)[0, i, E_idx[0, i], :] for i in range(L)]
    ).unsqueeze(0)
    assert torch.allclose(rbf_a, rbf_b2, atol=1e-5), "_get_rbf mismatch"

    dec = torch.argsort(torch.rand(B, L, device=device), dim=1).float()
    full = causal_mask_before(dec, L)                   # [B, L, L]
    sparse = causal_mask_after(dec, E_idx)              # [B, L, K]
    # sparse[b, i, k] should equal full[b, i, E_idx[b, i, k]]
    expected = torch.stack(
        [full[0, i, E_idx[0, i]] for i in range(L)]
    ).unsqueeze(0)
    assert torch.allclose(sparse, expected, atol=1e-5), "causal mask mismatch"

    X = torch.randn(B, L, 14, 3, device=device)
    X_m = (torch.rand(B, L, 14, device=device) > 0.3).float()
    E_sc = torch.randint(0, L, (B, L, 48), device=device)
    sc_b = sc_rbf_before(X, X_m, E_sc)
    sc_a = sc_rbf_after(X, X_m, E_sc)
    assert torch.allclose(sc_b, sc_a, atol=1e-5), "sidechain RBF mismatch"

    print("Correctness: all checks passed.\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(device, lengths):
    B = 1
    K_mpnn = 32    # k-NN for LigandMPNN
    K_sc   = 48    # k-NN for side-chain packing
    num_rbf = 16

    print(f"Device : {device}")
    print(f"B={B}  K(mpnn)={K_mpnn}  K(sc)={K_sc}  num_rbf={num_rbf}")
    print(f"Median over 20 runs (ms).  Warmup: 5 runs.\n")

    # ---- _get_rbf ----
    hdr = f"{'L':>6}  {'before':>10}  {'after':>9}  {'speedup':>8}"
    sep = "-" * len(hdr)
    print("_get_rbf  — pairwise distance RBF features")
    print(hdr); print(sep)
    for L in lengths:
        A      = torch.randn(B, L, 3, device=device)
        Bc     = torch.randn(B, L, 3, device=device)
        E_idx  = torch.randint(0, L, (B, L, K_mpnn), device=device)
        tb = measure(lambda: get_rbf_before(A, Bc))
        ta = measure(lambda: get_rbf_after(A, Bc, E_idx))
        print(f"{L:>6}  {tb:>10.2f}  {ta:>9.2f}  {tb/ta:>7.1f}x")

    # ---- causal mask ----
    print(f"\nCausal mask  — backward/forward neighbour masks for decoding")
    print(hdr); print(sep)
    for L in lengths:
        dec   = torch.argsort(torch.rand(B, L, device=device), dim=1).float()
        E_idx = torch.randint(0, L, (B, L, K_mpnn), device=device)
        tb = measure(lambda: causal_mask_before(dec, L))
        ta = measure(lambda: causal_mask_after(dec, E_idx))
        print(f"{L:>6}  {tb:>10.2f}  {ta:>9.2f}  {tb/ta:>7.1f}x")

    # ---- 14x14 sidechain RBF ----
    print(
        f"\n14×14 sidechain RBF  — features_decode (K={K_sc})\n"
        f"  Note: speedup is primarily on GPU (196 kernel launches → 1 fused op).\n"
        f"  On CPU the large intermediate tensor (~345 MB at L=600) creates cache\n"
        f"  pressure; CPU timings are comparable or slightly worse at long sequences."
    )
    print(hdr); print(sep)
    for L in lengths:
        X     = torch.randn(B, L, 14, 3, device=device)
        X_m   = (torch.rand(B, L, 14, device=device) > 0.3).float()
        E_idx = torch.randint(0, L, (B, L, K_sc), device=device)
        tb = measure(lambda: sc_rbf_before(X, X_m, E_idx))
        ta = measure(lambda: sc_rbf_after(X, X_m, E_idx))
        print(f"{L:>6}  {tb:>10.2f}  {ta:>9.2f}  {tb/ta:>7.1f}x")

    print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument(
        "--lengths", nargs="+", type=int, default=[50, 100, 200, 400, 600]
    )
    ap.add_argument("--skip-correctness", action="store_true")
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.\n")
        args.device = "cpu"

    if not args.skip_correctness:
        check_correctness(args.device)

    run(args.device, args.lengths)
