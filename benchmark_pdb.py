#!/usr/bin/env python3
"""
Benchmark on real PDB structures using actual backbone coordinates.

Loads the three bundled input PDBs (inputs/1BC8.pdb, 2GFB.pdb, 4GYT.pdb),
builds the real k-NN graph from Cα coordinates, and times the three hot
paths changed in this PR.  No model checkpoint needed.

Usage:
    python benchmark_pdb.py
    python benchmark_pdb.py --device cuda
    python benchmark_pdb.py --pdbs inputs/1BC8.pdb inputs/4GYT.pdb
"""

import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F

from data_utils import parse_PDB, featurize


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def gather_nodes(nodes, neighbor_idx):
    """[B, L, nv] + [B, L, K] -> [B, L, K, nv]"""
    B, L, nv = nodes.shape
    K = neighbor_idx.shape[2]
    idx = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, nv)
    return torch.gather(nodes.unsqueeze(2).expand(-1, -1, K, -1), 1, idx)


def build_knn(CA, K):
    """Build k-NN graph from Cα coordinates. Returns E_idx [1, L, K]."""
    D = torch.cdist(CA, CA)  # [1, L, L]
    return torch.topk(D, K + 1, dim=-1, largest=False).indices[:, :, 1:]  # exclude self


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def measure(fn, n_warmup=5, n_runs=20):
    for _ in range(n_warmup):
        fn(); _sync()
    t = []
    for _ in range(n_runs):
        _sync()
        t0 = time.perf_counter()
        fn(); _sync()
        t.append(time.perf_counter() - t0)
    return np.median(t) * 1e3  # ms


# ---------------------------------------------------------------------------
# Before/after implementations (same as benchmark.py)
# ---------------------------------------------------------------------------

def get_rbf_before(A, B_coords, num_rbf=16):
    D = torch.sqrt(
        torch.sum((A[:, :, None, :] - B_coords[:, None, :, :]) ** 2, -1) + 1e-6
    )
    D_mu = torch.linspace(2.0, 22.0, num_rbf, device=A.device)
    D_sigma = (22.0 - 2.0) / num_rbf
    return torch.exp(-(((D.unsqueeze(-1) - D_mu) / D_sigma) ** 2))


def get_rbf_after(A, B_coords, E_idx, num_rbf=16):
    D_mu = torch.linspace(2.0, 22.0, num_rbf, device=A.device).view(1, 1, 1, -1)
    D_sigma = (22.0 - 2.0) / num_rbf
    B_nbrs = gather_nodes(B_coords, E_idx)
    D = torch.sqrt(torch.sum((A[:, :, None, :] - B_nbrs) ** 2, -1) + 1e-6)
    return torch.exp(-(((D.unsqueeze(-1) - D_mu) / D_sigma) ** 2))


def causal_mask_before(decoding_order, L):
    perm = F.one_hot(decoding_order.long(), num_classes=L).float()
    tril = 1 - torch.triu(torch.ones(L, L, device=decoding_order.device))
    return torch.einsum("ij,biq,bjp->bqp", tril, perm, perm)


def causal_mask_after(decoding_order, E_idx):
    rank = torch.argsort(decoding_order, dim=1).float()
    rank_nbrs = gather_nodes(rank.unsqueeze(-1), E_idx).squeeze(-1)
    return (rank_nbrs < rank.unsqueeze(-1)).float()


def sc_rbf_before(X, X_m, E_idx, num_rbf=16, lo=2.0, hi=22.0):
    B, L, _, _ = X.shape
    X_m_g = gather_nodes(X_m, E_idx)
    D_mu = torch.linspace(lo, hi, num_rbf, device=X.device).view(1, 1, 1, -1)
    D_sigma = (hi - lo) / num_rbf
    out = []
    for i in range(14):
        for j in range(14):
            nbrs = gather_nodes(X[:, :, j, :], E_idx)
            D = torch.sqrt(
                torch.sum((X[:, :, i, :].unsqueeze(2) - nbrs) ** 2, -1) + 1e-6
            )
            rbf = torch.exp(-(((D.unsqueeze(-1) - D_mu) / D_sigma) ** 2))
            out.append(rbf * X_m[:, :, i, None, None] * X_m_g[:, :, :, j, None])
    return torch.cat(out, dim=-1)


def sc_rbf_after(X, X_m, E_idx, num_rbf=16, lo=2.0, hi=22.0):
    B, L, _, _ = X.shape
    K = E_idx.shape[2]
    X_m_g = gather_nodes(X_m, E_idx)
    D_mu = torch.linspace(lo, hi, num_rbf, device=X.device).view(1, 1, 1, 1, 1, -1)
    D_sigma = (hi - lo) / num_rbf
    X_nbrs = gather_nodes(X.view(B, L, -1), E_idx).view(B, L, K, 14, 3)
    D = torch.sqrt(
        torch.sum(
            (X[:, :, :, None, None, :] - X_nbrs[:, :, None, :, :, :]) ** 2, -1
        ) + 1e-6
    )
    rbf = torch.exp(-(((D.unsqueeze(-1) - D_mu) / D_sigma) ** 2))
    rbf = rbf * X_m[:, :, :, None, None, None] * X_m_g[:, :, None, :, :, None]
    return rbf.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, L, K, -1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def bench_pdb(pdb_path, device, K_mpnn=32, K_sc=48):
    protein_list = parse_PDB(pdb_path)
    feature_dict = featurize(protein_list[0])

    X_raw = feature_dict["X"].to(device)   # [1, L, 4, 3]  N,CA,C,O
    mask  = feature_dict["mask"].to(device) # [1, L]
    B, L, _, _ = X_raw.shape

    # Cα coordinates for k-NN
    CA = X_raw[:, :, 1, :]  # [1, L, 3]

    E_idx_mpnn = build_knn(CA, K_mpnn)  # [1, L, K_mpnn]
    E_idx_sc   = build_knn(CA, K_sc)   # [1, L, K_sc]

    decoding_order = torch.argsort(torch.rand(1, L, device=device), dim=1).float()

    # For sc_rbf we need 14-atom coords; pad with zeros beyond the 4 backbone atoms
    X_14 = torch.zeros(1, L, 14, 3, device=device)
    X_14[:, :, :4, :] = X_raw
    X_m_14 = torch.zeros(1, L, 14, device=device)
    X_m_14[:, :, :4] = mask.unsqueeze(-1)

    results = {}

    tb = measure(lambda: get_rbf_before(CA, CA))
    ta = measure(lambda: get_rbf_after(CA, CA, E_idx_mpnn))
    results["rbf"] = (tb, ta)

    tb = measure(lambda: causal_mask_before(decoding_order, L))
    ta = measure(lambda: causal_mask_after(decoding_order, E_idx_mpnn))
    results["mask"] = (tb, ta)

    tb = measure(lambda: sc_rbf_before(X_14, X_m_14, E_idx_sc))
    ta = measure(lambda: sc_rbf_after(X_14, X_m_14, E_idx_sc))
    results["sc"] = (tb, ta)

    return L, results


def run(pdbs, device):
    K_mpnn, K_sc = 32, 48
    print(f"\nDevice: {device}   K(mpnn)={K_mpnn}  K(sc)={K_sc}")
    print(f"Median of 20 runs (ms), 5 warmup runs.\n")

    hdr = f"{'PDB':<12} {'L':>5}  {'rbf before':>11} {'rbf after':>10} {'rbf speedup':>11}  "
    hdr += f"{'mask before':>12} {'mask after':>11} {'mask speedup':>12}  "
    hdr += f"{'sc before':>10} {'sc after':>9} {'sc speedup':>10}"
    print(hdr)
    print("-" * len(hdr))

    for pdb in pdbs:
        name = pdb.split("/")[-1].replace(".pdb", "")
        try:
            L, r = bench_pdb(pdb, device, K_mpnn, K_sc)
            rb, ra = r["rbf"];  mb, ma = r["mask"];  sb, sa = r["sc"]
            row  = f"{name:<12} {L:>5}  {rb:>11.2f} {ra:>10.2f} {rb/ra:>10.1f}x  "
            row += f"{mb:>12.2f} {ma:>11.2f} {mb/ma:>11.1f}x  "
            row += f"{sb:>10.2f} {sa:>9.2f} {sb/sa:>9.1f}x"
            print(row)
        except Exception as e:
            print(f"{name:<12}  ERROR: {e}")

    print()
    print("Columns: _get_rbf, causal mask, 14×14 sidechain RBF (features_decode)")
    print(f"sc uses K={K_sc} and backbone coords padded to 14 atoms.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument(
        "--pdbs",
        nargs="+",
        default=["inputs/1BC8.pdb", "inputs/2GFB.pdb", "inputs/4GYT.pdb"],
    )
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.\n")
        args.device = "cpu"

    run(args.pdbs, args.device)
