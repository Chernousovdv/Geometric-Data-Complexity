"""
Compare Gaussian kernel vs linear kernel for C(X) = log det(I + K).

Gaussian: K_{ij} = exp(-lambda * D^2_{ij})  (distance-based, needs CND)
Linear:   K = Phi @ Phi.T / c               (Gram-based, always PSD)

Both should give the same qualitative behavior on low-D synthetic data.
The linear kernel should also work in high-D where Gaussian fails.

Experiments:
1. Growth rate (N-cluster vs single-cluster) in D=2
2. Growth rate in D=100 (high-dimensional GMM)
3. Varying separation
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.model import AnalyticScoreModel, VPNoiseSchedule
from core.trajectory import ode_trajectories, trajectory_embedding
from core.complexity import calibrate_bandwidth, compute_complexity

SEED = 42
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

ns = VPNoiseSchedule()
L = 20
t_eval = np.linspace(0.01, 0.99, L)


def C_linear(emb, c=1.0):
    """C(X) = log det(I + Phi @ Phi.T / c) via eigenvalues."""
    K = emb @ emb.T / c
    eigvals = np.linalg.eigvalsh(K)
    eigvals = np.clip(eigvals, 0, None)
    return np.sum(np.log(1 + eigvals))


def C_gaussian(emb, lam):
    """C(X) with Gaussian kernel (existing pipeline)."""
    return compute_complexity(emb, lam)['C']


# ============================================================
# Experiment 1: Growth rate in D=2
# ============================================================

print("=" * 60)
print("Experiment 1: Growth rate, D=2")
print("=" * 60)

D = 2
spacing = 20.0
Ns = [2, 3, 4, 6, 8, 10, 15, 20, 25, 30]

# Calibrate lambda for Gaussian kernel
model_ref = AnalyticScoreModel(
    np.zeros((1, D)), np.tile(np.eye(D), (1, 1, 1)),
    np.ones(1), noise_schedule=ns)
x_cal = np.random.default_rng(99).multivariate_normal(np.zeros(D), np.eye(D), 300)
traj_cal = ode_trajectories(x_cal, model_ref, t_eval)
emb_cal = trajectory_embedding(traj_cal)
lam = calibrate_bandwidth(emb_cal)

# Calibrate c for linear kernel: use same reference embeddings
# c = median of ||Phi_i||^2 so that eigenvalues are O(1)
norms_sq = np.sum(emb_cal ** 2, axis=1)
c_lin = np.median(norms_sq)
print(f"lambda = {lam:.6f}, c = {c_lin:.4f}")

# N-cluster
C_nclust_gauss = []
C_nclust_lin = []

for N in Ns:
    means = np.zeros((N, D))
    means[:, 0] = np.arange(N) * spacing
    covs = np.tile(np.eye(D), (N, 1, 1))
    model = AnalyticScoreModel(means, covs, np.ones(N) / N, noise_schedule=ns)

    rng = np.random.default_rng(SEED)
    x = np.array([rng.multivariate_normal(means[k], covs[k]) for k in range(N)])
    traj = ode_trajectories(x, model, t_eval)
    emb = trajectory_embedding(traj)

    C_nclust_gauss.append(C_gaussian(emb, lam))
    C_nclust_lin.append(C_linear(emb, c_lin))

# Single cluster (tight)
C_tight_gauss = []
C_tight_lin = []
sig = 0.1

model_s = AnalyticScoreModel(
    np.zeros((1, D)), np.tile(np.eye(D), (1, 1, 1)),
    np.ones(1), noise_schedule=ns)

for N in Ns:
    x = np.random.default_rng(SEED).multivariate_normal(
        np.zeros(D), sig ** 2 * np.eye(D), N)
    traj = ode_trajectories(x, model_s, t_eval)
    emb = trajectory_embedding(traj)

    C_tight_gauss.append(C_gaussian(emb, lam))
    C_tight_lin.append(C_linear(emb, c_lin))

# Print
print(f"\n{'N':>5} {'Gauss_clust':>12} {'Lin_clust':>12} {'Gauss_tight':>12} {'Lin_tight':>12}")
for i, N in enumerate(Ns):
    print(f"{N:>5} {C_nclust_gauss[i]:>12.3f} {C_nclust_lin[i]:>12.3f} "
          f"{C_tight_gauss[i]:>12.3f} {C_tight_lin[i]:>12.3f}")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
Ns_arr = np.array(Ns)

ax = axes[0]
ax.set_title("Gaussian kernel (D=2)")
ax.plot(Ns_arr, C_nclust_gauss, 'o-', label='N-cluster')
ax.plot(Ns_arr, C_tight_gauss, 's-', label=f'Single cluster (σ={sig})')
ax.plot(Ns_arr, Ns_arr * np.log(2), '--', alpha=0.5, label='N log 2')
ax.plot(Ns_arr, np.log(1 + Ns_arr), '--', alpha=0.5, label='log(1+N)')
ax.set_xlabel('N')
ax.set_ylabel('C(X)')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.set_title("Linear kernel (D=2)")
ax.plot(Ns_arr, C_nclust_lin, 'o-', label='N-cluster')
ax.plot(Ns_arr, C_tight_lin, 's-', label=f'Single cluster (σ={sig})')
ax.set_xlabel('N')
ax.set_ylabel('C(X)')
ax.legend()
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(RESULTS_DIR / "linear_vs_gaussian_D2.png", dpi=150)
print(f"\nSaved: {RESULTS_DIR / 'linear_vs_gaussian_D2.png'}")
plt.close()


# ============================================================
# Experiment 2: Growth rate in D=100
# ============================================================

print("\n" + "=" * 60)
print("Experiment 2: Growth rate, D=100")
print("=" * 60)

D = 100
spacing = 20.0
Ns2 = [2, 5, 10, 15, 20, 30]

# Reference model (single unit Gaussian in R^100)
model_ref = AnalyticScoreModel(
    np.zeros((1, D)), np.tile(np.eye(D), (1, 1, 1)),
    np.ones(1), noise_schedule=ns)
x_cal = np.random.default_rng(99).multivariate_normal(np.zeros(D), np.eye(D), 100)
traj_cal = ode_trajectories(x_cal, model_ref, t_eval)
emb_cal = trajectory_embedding(traj_cal)
lam = calibrate_bandwidth(emb_cal)
norms_sq = np.sum(emb_cal ** 2, axis=1)
c_lin = np.median(norms_sq)
print(f"lambda = {lam:.8f}, c = {c_lin:.2f}")

C_nclust_gauss2 = []
C_nclust_lin2 = []
C_tight_gauss2 = []
C_tight_lin2 = []

for N in Ns2:
    # N-cluster on a line
    means = np.zeros((N, D))
    means[:, 0] = np.arange(N) * spacing
    covs = np.tile(np.eye(D), (N, 1, 1))
    model = AnalyticScoreModel(means, covs, np.ones(N) / N, noise_schedule=ns)

    rng = np.random.default_rng(SEED)
    x = np.array([rng.multivariate_normal(means[k], covs[k]) for k in range(N)])
    traj = ode_trajectories(x, model, t_eval)
    emb = trajectory_embedding(traj)

    C_nclust_gauss2.append(C_gaussian(emb, lam))
    C_nclust_lin2.append(C_linear(emb, c_lin))
    print(f"N-cluster  N={N:3d}: Gauss={C_nclust_gauss2[-1]:.3f}, Linear={C_nclust_lin2[-1]:.3f}")

    # Single cluster tight
    x = np.random.default_rng(SEED).multivariate_normal(
        np.zeros(D), 0.1 ** 2 * np.eye(D), N)
    traj = ode_trajectories(x, model_ref, t_eval)
    emb = trajectory_embedding(traj)

    C_tight_gauss2.append(C_gaussian(emb, lam))
    C_tight_lin2.append(C_linear(emb, c_lin))
    print(f"Tight      N={N:3d}: Gauss={C_tight_gauss2[-1]:.3f}, Linear={C_tight_lin2[-1]:.3f}")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
Ns2_arr = np.array(Ns2)

ax = axes[0]
ax.set_title("Gaussian kernel (D=100)")
ax.plot(Ns2_arr, C_nclust_gauss2, 'o-', label='N-cluster')
ax.plot(Ns2_arr, C_tight_gauss2, 's-', label='Single cluster (σ=0.1)')
ax.set_xlabel('N')
ax.set_ylabel('C(X)')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.set_title("Linear kernel (D=100)")
ax.plot(Ns2_arr, C_nclust_lin2, 'o-', label='N-cluster')
ax.plot(Ns2_arr, C_tight_lin2, 's-', label='Single cluster (σ=0.1)')
ax.set_xlabel('N')
ax.set_ylabel('C(X)')
ax.legend()
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(RESULTS_DIR / "linear_vs_gaussian_D100.png", dpi=150)
print(f"\nSaved: {RESULTS_DIR / 'linear_vs_gaussian_D100.png'}")
plt.close()


# ============================================================
# Experiment 3: Varying separation in D=2
# ============================================================

print("\n" + "=" * 60)
print("Experiment 3: Varying separation, D=2")
print("=" * 60)

D = 2
N = 20
deltas = [0.5, 1, 2, 3, 5, 8, 10, 15, 20, 30]

# Reference for Gaussian kernel
model_ref = AnalyticScoreModel(
    np.zeros((1, D)), np.tile(np.eye(D), (1, 1, 1)),
    np.ones(1), noise_schedule=ns)
x_cal = np.random.default_rng(99).multivariate_normal(np.zeros(D), np.eye(D), 300)
traj_cal = ode_trajectories(x_cal, model_ref, t_eval)
emb_cal = trajectory_embedding(traj_cal)
lam = calibrate_bandwidth(emb_cal)
norms_sq = np.sum(emb_cal ** 2, axis=1)
c_lin = np.median(norms_sq)

C_sep_gauss = []
C_sep_lin = []

for delta in deltas:
    means = np.array([[0, 0], [delta, 0]], dtype=float)
    covs = np.tile(np.eye(D), (2, 1, 1))
    model = AnalyticScoreModel(means, covs, np.ones(2) / 2, noise_schedule=ns)

    rng = np.random.default_rng(SEED)
    x = np.vstack([
        rng.multivariate_normal(means[0], covs[0], N // 2),
        rng.multivariate_normal(means[1], covs[1], N // 2),
    ])
    traj = ode_trajectories(x, model, t_eval)
    emb = trajectory_embedding(traj)

    C_sep_gauss.append(C_gaussian(emb, lam))
    C_sep_lin.append(C_linear(emb, c_lin))
    print(f"delta={delta:5.1f}: Gauss={C_sep_gauss[-1]:.3f}, Linear={C_sep_lin[-1]:.3f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.set_title("Gaussian kernel (D=2)")
ax.plot(deltas, C_sep_gauss, 'o-')
ax.set_xlabel('Cluster separation δ')
ax.set_ylabel('C(X)')
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.set_title("Linear kernel (D=2)")
ax.plot(deltas, C_sep_lin, 'o-')
ax.set_xlabel('Cluster separation δ')
ax.set_ylabel('C(X)')
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(RESULTS_DIR / "linear_vs_gaussian_separation.png", dpi=150)
print(f"\nSaved: {RESULTS_DIR / 'linear_vs_gaussian_separation.png'}")
plt.close()
