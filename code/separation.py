"""
Experiment: Varying cluster separation with fixed lambda.

Lambda is calibrated from a single-cluster reference so it captures
within-cluster scale and doesn't get washed out by between-cluster distances.
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


def make_gmm_means(K, D, delta):
    angles = np.linspace(0, 2 * np.pi, K, endpoint=False)
    return delta * np.column_stack([np.cos(angles), np.sin(angles)])


def sample_gmm_roundrobin(means, covs, N, seed=None):
    rng = np.random.default_rng(seed)
    K = len(means)
    x = np.zeros((N, means.shape[1]))
    labels = np.zeros(N, dtype=int)
    for i in range(N):
        k = i % K
        labels[i] = k
        x[i] = rng.multivariate_normal(means[k], covs[k])
    return x, labels


# ----- Setup -----
K, D, N = 3, 2, 30
deltas = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0]

# Calibrate lambda once from a SINGLE cluster (unit Gaussian at origin)
# This captures within-cluster scale, independent of separation.
mean_ref = np.array([[0.0, 0.0]])
cov_ref = np.tile(np.eye(D), (1, 1, 1))
w_ref = np.ones(1)
model_ref = AnalyticScoreModel(mean_ref, cov_ref, w_ref, noise_schedule=ns)

x_cal = np.random.default_rng(99).multivariate_normal(mean_ref[0], cov_ref[0], 300)
traj_cal = ode_trajectories(x_cal, model_ref, t_eval)
emb_cal = trajectory_embedding(traj_cal)
lam = calibrate_bandwidth(emb_cal)
print(f"Lambda (single-cluster calibration): {lam:.6f}")

# ----- Run -----
C_singles = []
C_mixtures = []

for delta in deltas:
    means = make_gmm_means(K, D, delta)
    covs = np.tile(np.eye(D), (K, 1, 1))
    w = np.ones(K) / K
    model = AnalyticScoreModel(means, covs, w, noise_schedule=ns)

    # Single cluster
    x_s = np.random.default_rng(SEED).multivariate_normal(means[0], covs[0], N)
    traj_s = ode_trajectories(x_s, model, t_eval)
    emb_s = trajectory_embedding(traj_s)
    res_s = compute_complexity(emb_s, lam)

    # Mixture (round-robin)
    x_m, _ = sample_gmm_roundrobin(means, covs, N, seed=SEED)
    traj_m = ode_trajectories(x_m, model, t_eval)
    emb_m = trajectory_embedding(traj_m)
    res_m = compute_complexity(emb_m, lam)

    C_singles.append(res_s['C'])
    C_mixtures.append(res_m['C'])
    ratio = res_m['C'] / res_s['C']
    print(f"delta={delta:5.1f}: C_single={res_s['C']:.3f}, C_mix={res_m['C']:.3f}, ratio={ratio:.3f}")

# ----- Plot -----
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(deltas, C_mixtures, 's-', label=f'Mixture ({K} clusters)')
ax.set_xlabel('Separation $\\delta$')
ax.set_ylabel('$C(X)$')
ax.set_title(f'Complexity vs cluster separation')
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
out = RESULTS_DIR / "varying_separation.png"
fig.savefig(out, dpi=150)
print(f"\nSaved: {out}")
plt.close()
