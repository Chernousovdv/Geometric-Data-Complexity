"""
Experiment: Growth rate — linear vs logarithmic.

- New cluster per point (well-separated): C ~ N log 2
- Single cluster (tight variance): C ~ log(1 + N)

Lambda is calibrated from a single unit-Gaussian reference.
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
D = 2

# Calibrate lambda from single unit-Gaussian
mean_ref = np.array([[0.0, 0.0]])
cov_ref = np.tile(np.eye(D), (1, 1, 1))
model_ref = AnalyticScoreModel(mean_ref, cov_ref, np.ones(1), noise_schedule=ns)
x_cal = np.random.default_rng(99).multivariate_normal(mean_ref[0], cov_ref[0], 300)
traj_cal = ode_trajectories(x_cal, model_ref, t_eval)
emb_cal = trajectory_embedding(traj_cal)
lam = calibrate_bandwidth(emb_cal)
print(f"Lambda: {lam:.6f}")


# ---- N-cluster case: place clusters on a LINE so they stay separated ----

Ns = [2, 3, 4, 6, 8, 10, 15, 20, 25, 30]
spacing = 20.0  # distance between adjacent clusters

C_nclust = []
for N in Ns:
    # N clusters along x-axis, spaced by `spacing`
    means_n = np.zeros((N, D))
    means_n[:, 0] = np.arange(N) * spacing
    covs_n = np.tile(np.eye(D), (N, 1, 1))
    w_n = np.ones(N) / N
    model_n = AnalyticScoreModel(means_n, covs_n, w_n, noise_schedule=ns)

    rng = np.random.default_rng(SEED)
    x = np.array([rng.multivariate_normal(means_n[k], covs_n[k]) for k in range(N)])
    traj = ode_trajectories(x, model_n, t_eval)
    emb = trajectory_embedding(traj)
    res = compute_complexity(emb, lam)
    C_nclust.append(res['C'])
    print(f"N-cluster  N={N:3d}: C={res['C']:.3f}, theory={N * np.log(2):.3f}")


# ---- Single-cluster cases ----

K_model = 3
means_s = np.array([[5., 0.], [-2.5, 4.33], [-2.5, -4.33]])
covs_s = np.tile(np.eye(D), (K_model, 1, 1))
model_s = AnalyticScoreModel(means_s, covs_s, np.ones(K_model) / K_model,
                              noise_schedule=ns)

sigmas = [0.1, 1.0]
C_single = {sig: [] for sig in sigmas}

for sig in sigmas:
    for N in Ns:
        x = np.random.default_rng(SEED).multivariate_normal(
            means_s[0], sig**2 * np.eye(D), N)
        traj = ode_trajectories(x, model_s, t_eval)
        emb = trajectory_embedding(traj)
        res = compute_complexity(emb, lam)
        C_single[sig].append(res['C'])
        print(f"Single(σ={sig})  N={N:3d}: C={res['C']:.3f}")


# ---- Plot ----

Ns_arr = np.array(Ns)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(Ns_arr, C_nclust, 'o-', label='New cluster per point')
ax.plot(Ns_arr, C_single[1.0], 's-', label='Single cluster ($\\sigma = 1$)')
ax.plot(Ns_arr, C_single[0.1], '^-', label='Single cluster ($\\sigma = 0.1$)')
ax.plot(Ns_arr, Ns_arr * np.log(2), '--', color='C0', alpha=0.5,
        label='$N \\log 2$')
ax.plot(Ns_arr, np.log(1 + Ns_arr), '--', color='C2', alpha=0.5,
        label='$\\log(1+N)$')
ax.set_xlabel('$N$')
ax.set_ylabel('$C(X)$')
ax.set_title('Complexity growth rate')
ax.legend()
ax.grid(True, alpha=0.3)

fig.tight_layout()
out = RESULTS_DIR / "growth_rate.png"
fig.savefig(out, dpi=150)
print(f"\nSaved: {out}")
plt.close()
