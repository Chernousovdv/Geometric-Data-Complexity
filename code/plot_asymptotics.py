"""
Plot asymptotic behaviour of C(X) for different data regimes.

Three regimes from the paper:
  1. N identical points:        C = log(1+N) ~ log N
  2. M clusters of m points:    C = M log(1+m), varying N = M*m
  3. N maximally diverse points: C = N log 2

Also plots:
  4. Growth rate comparison on one axis
  5. Per-point contribution C(N)/N as N grows
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.complexity import (
    score_embedding,
    pairwise_distance,
    gaussian_kernel,
    complexity,
)


def build_K(sv, lam):
    emb = score_embedding(sv)
    D = pairwise_distance(emb)
    return gaussian_kernel(D, lam)


def C(K):
    val, _ = complexity(K)
    return val


# ============================================================
# Data generators (return score vectors directly)
# ============================================================

def identical_points(N, L=3, D=2):
    """N copies of the same point in embedding space."""
    return np.ones((N, L, D)) * 0.5


def maximally_diverse(N, L=3, D=2):
    """N points infinitely far apart (K -> I)."""
    sv = np.zeros((N, L, D))
    for i in range(N):
        sv[i, :, :] = i * 1e6
    return sv


def m_clusters(M, m, L=3, D=2):
    """M clusters, m identical copies each."""
    N = M * m
    sv = np.zeros((N, L, D))
    for c in range(M):
        sv[c * m:(c + 1) * m, :, :] = c * 1e6
    return sv


def growing_clusters(N, M, L=3, D=2):
    """Fixed M clusters, growing total N (m = N/M per cluster)."""
    m = N // M
    actual_N = m * M
    return m_clusters(M, m, L, D), actual_N


# ============================================================
# Experiments
# ============================================================

def experiment_three_regimes(Ns, M_fixed=5):
    """Compute C for identical, M-cluster, and diverse regimes."""
    C_identical = []
    C_diverse = []
    C_clustered = []
    Ns_clustered = []

    lam = 1.0

    for N in Ns:
        # Identical
        sv = identical_points(N)
        C_identical.append(C(build_K(sv, lam)))

        # Diverse
        sv = maximally_diverse(N)
        C_diverse.append(C(build_K(sv, lam)))

        # M clusters (N must be divisible by M)
        m = max(1, N // M_fixed)
        actual_N = m * M_fixed
        sv = m_clusters(M_fixed, m)
        C_clustered.append(C(build_K(sv, lam)))
        Ns_clustered.append(actual_N)

    return (np.array(C_identical), np.array(C_diverse),
            np.array(C_clustered), np.array(Ns_clustered))


def experiment_varying_clusters(N_total, Ms):
    """Fixed total N, vary number of clusters M."""
    lam = 1.0
    Cs = []
    actual_Ms = []

    for M in Ms:
        m = N_total // M
        if m < 1:
            continue
        actual_N = m * M
        sv = m_clusters(M, m)
        Cs.append(C(build_K(sv, lam)))
        actual_Ms.append(M)

    return np.array(actual_Ms), np.array(Cs)


# ============================================================
# Plotting
# ============================================================

def plot_three_regimes(out_dir):
    """Main plot: three growth regimes on one axis."""
    Ns = np.arange(2, 102, 2)
    M_fixed = 5
    C_ident, C_div, C_clust, Ns_clust = experiment_three_regimes(Ns, M_fixed)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(Ns, C_div, 'o-', ms=3, label=r'Diverse: $C = N\log 2$', color='C0')
    ax.plot(Ns_clust, C_clust, 's-', ms=3,
            label=rf'{M_fixed} clusters: $C = M\log(1+N/M)$', color='C1')
    ax.plot(Ns, C_ident, '^-', ms=3, label=r'Identical: $C = \log(1+N)$', color='C2')

    # Theoretical curves
    ax.plot(Ns, Ns * np.log(2), '--', alpha=0.4, color='C0')
    ax.plot(Ns, M_fixed * np.log(1 + Ns / M_fixed), '--', alpha=0.4, color='C1')
    ax.plot(Ns, np.log(1 + Ns), '--', alpha=0.4, color='C2')

    ax.set_xlabel('N (number of points)')
    ax.set_ylabel(r'$\mathcal{C}(\mathcal{X})$')
    ax.set_title('Three asymptotic regimes of dataset complexity')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / 'three_regimes.png', dpi=150)
    plt.close(fig)
    print(f'Saved {out_dir / "three_regimes.png"}')


def plot_per_point_contribution(out_dir):
    """C(N)/N: per-point contribution decays for redundant data."""
    Ns = np.arange(2, 102, 2)
    M_fixed = 5
    C_ident, C_div, C_clust, Ns_clust = experiment_three_regimes(Ns, M_fixed)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(Ns, C_div / Ns, 'o-', ms=3, label='Diverse', color='C0')
    ax.plot(Ns_clust, C_clust / Ns_clust, 's-', ms=3,
            label=f'{M_fixed} clusters', color='C1')
    ax.plot(Ns, C_ident / Ns, '^-', ms=3, label='Identical', color='C2')

    ax.axhline(np.log(2), ls=':', color='gray', alpha=0.5, label=r'$\log 2$ (max)')
    ax.set_xlabel('N')
    ax.set_ylabel(r'$\mathcal{C}(\mathcal{X}) / N$')
    ax.set_title('Per-point complexity contribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / 'per_point.png', dpi=150)
    plt.close(fig)
    print(f'Saved {out_dir / "per_point.png"}')


def plot_varying_clusters(out_dir):
    """Fixed N, vary M: complexity increases with number of clusters."""
    N_total = 100
    Ms = [1, 2, 4, 5, 10, 20, 25, 50, 100]
    actual_Ms, Cs = experiment_varying_clusters(N_total, Ms)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(actual_Ms, Cs, 'o-', ms=5, color='C3')

    # Theoretical: M log(1 + N/M)
    M_cont = np.linspace(1, N_total, 200)
    ax.plot(M_cont, M_cont * np.log(1 + N_total / M_cont),
            '--', alpha=0.4, color='C3', label=r'$M\log(1+N/M)$')

    ax.set_xlabel('M (number of clusters)')
    ax.set_ylabel(r'$\mathcal{C}(\mathcal{X})$')
    ax.set_title(f'Complexity vs. cluster count (N={N_total} fixed)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / 'varying_clusters.png', dpi=150)
    plt.close(fig)
    print(f'Saved {out_dir / "varying_clusters.png"}')


def plot_marginal_gains(out_dir):
    """Marginal gain of adding the N-th point in each regime."""
    Ns = np.arange(2, 82)
    lam = 1.0

    gains_ident = []
    gains_diverse = []
    gains_clust = []
    M_fixed = 5

    for N in Ns:
        # Identical
        K_prev = build_K(identical_points(N - 1), lam)
        K_curr = build_K(identical_points(N), lam)
        gains_ident.append(C(K_curr) - C(K_prev))

        # Diverse
        K_prev = build_K(maximally_diverse(N - 1), lam)
        K_curr = build_K(maximally_diverse(N), lam)
        gains_diverse.append(C(K_curr) - C(K_prev))

        # Clusters: add to existing cluster (just grow m for last cluster)
        # N-1 points in M clusters, then add one more to the last cluster
        m_base = (N - 1) // M_fixed
        remainder = (N - 1) % M_fixed
        # Approximate: just use floor clusters
        if m_base >= 1:
            m_curr = N // M_fixed
            if m_curr >= 1:
                K_prev_c = build_K(m_clusters(M_fixed, m_base), lam)
                K_curr_c = build_K(m_clusters(M_fixed, m_curr), lam)
                gains_clust.append(
                    (C(K_curr_c) - C(K_prev_c)) / max(1, m_curr - m_base)
                )
            else:
                gains_clust.append(np.nan)
        else:
            gains_clust.append(np.nan)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(Ns, gains_diverse, 'o-', ms=2, label='Diverse', color='C0')
    ax.plot(Ns, gains_ident, '^-', ms=2, label='Identical', color='C2')

    ax.axhline(np.log(2), ls=':', color='gray', alpha=0.5, label=r'$\log 2$')
    ax.axhline(0, ls='-', color='black', alpha=0.2)
    ax.set_xlabel('N')
    ax.set_ylabel('Marginal gain')
    ax.set_title(r'Marginal gain $\mathcal{C}(N) - \mathcal{C}(N-1)$')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / 'marginal_gains.png', dpi=150)
    plt.close(fig)
    print(f'Saved {out_dir / "marginal_gains.png"}')


def plot_log_scale(out_dir):
    """Log-log plot to see growth exponents clearly."""
    Ns = np.arange(2, 202, 2)
    M_values = [1, 3, 10, 50]
    lam = 1.0

    fig, ax = plt.subplots(figsize=(8, 5))

    # Diverse
    C_div = [C(build_K(maximally_diverse(N), lam)) for N in Ns]
    ax.plot(Ns, C_div, '-', lw=2, label='Diverse (slope 1)', color='C0')

    # M clusters for several M
    for M in M_values:
        Cs = []
        Ns_actual = []
        for N in Ns:
            m = N // M
            if m < 1:
                continue
            sv = m_clusters(M, m)
            Cs.append(C(build_K(sv, lam)))
            Ns_actual.append(m * M)
        ax.plot(Ns_actual, Cs, '--', lw=1.5, label=f'M={M} clusters')

    # Identical
    C_id = [C(build_K(identical_points(N), lam)) for N in Ns]
    ax.plot(Ns, C_id, '-', lw=2, label='Identical (slope 0)', color='C2')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('N')
    ax.set_ylabel(r'$\mathcal{C}(\mathcal{X})$')
    ax.set_title('Growth rate on log-log scale')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which='both')

    fig.tight_layout()
    fig.savefig(out_dir / 'log_scale.png', dpi=150)
    plt.close(fig)
    print(f'Saved {out_dir / "log_scale.png"}')


# ============================================================
# Main
# ============================================================

def main():
    out_dir = Path(__file__).resolve().parent.parent / 'results' / 'exp1_asymptotics'
    out_dir.mkdir(parents=True, exist_ok=True)

    print('Plotting asymptotic behaviour...\n')

    plot_three_regimes(out_dir)
    plot_per_point_contribution(out_dir)
    plot_varying_clusters(out_dir)
    plot_marginal_gains(out_dir)
    plot_log_scale(out_dir)

    print(f'\nAll plots saved to {out_dir}')


if __name__ == '__main__':
    main()
