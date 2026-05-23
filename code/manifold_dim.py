"""
Experiment: Complexity vs N for different intrinsic dimensions.

Data lives on d-dimensional subspaces of R^10 (d = 1, 2, 3, 5).
Each dataset is a single Gaussian with diagonal covariance:
    Sigma = diag(sigma^2, ..., sigma^2, eps, ..., eps)
              --- d active ---   -- D-d near-zero --

To control for distance scale, we set sigma^2 = sigma0^2 / d so that
E[||x_i - x_j||^2] = 2 * d * sigma^2 = 2 * sigma0^2 = const.

The model matches the data distribution in each case (analytic score).
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

D_ambient = 10
sigma_active = 1.0  # variance per active direction (same for all d)
eps_floor = 1e-4  # small variance in inactive directions (avoids singular cov)
dims = [1, 2, 3, 5]
Ns = [2, 4, 6, 8, 10, 15, 20, 30, 40, 50]

# Calibrate lambda from a mid-dimensional reference (d=3).
d_ref = 3
cov_ref = eps_floor * np.eye(D_ambient)
for j in range(d_ref):
    cov_ref[j, j] = sigma_active
mean_ref = np.zeros((1, D_ambient))
cov_ref_3d = cov_ref[np.newaxis, :, :]
model_ref = AnalyticScoreModel(mean_ref, cov_ref_3d, np.ones(1), noise_schedule=ns)

x_cal = np.random.default_rng(99).multivariate_normal(mean_ref[0], cov_ref, 300)
traj_cal = ode_trajectories(x_cal, model_ref, t_eval)
emb_cal = trajectory_embedding(traj_cal)
lam = calibrate_bandwidth(emb_cal)
print(f"Lambda (calibrated from d={d_ref} reference): {lam:.6f}")
print()

# Run
results = {}
for d in dims:
    cov = eps_floor * np.eye(D_ambient)
    for j in range(d):
        cov[j, j] = sigma_active

    mean = np.zeros((1, D_ambient))
    cov_3d = cov[np.newaxis, :, :]
    model = AnalyticScoreModel(mean, cov_3d, np.ones(1), noise_schedule=ns)

    Cs = []
    for N in Ns:
        x = np.random.default_rng(SEED).multivariate_normal(mean[0], cov, N)
        traj = ode_trajectories(x, model, t_eval)
        emb = trajectory_embedding(traj)
        res = compute_complexity(emb, lam)
        Cs.append(res['C'])

    results[d] = Cs
    print(f"d={d}: {['%.2f' % c for c in Cs]}")

# Plot
Ns_arr = np.array(Ns)

fig, ax = plt.subplots(figsize=(8, 5))
for d in dims:
    ax.plot(Ns_arr, results[d], 'o-', label=f'd = {d}')

ax.plot(Ns_arr, np.log(1 + Ns_arr), '--', color='gray', alpha=0.5,
        label='$\\log(1+N)$ (identical)')
ax.set_xlabel('$N$')
ax.set_ylabel('$C(X)$')
ax.set_title(
    f'Complexity vs $N$ for different intrinsic dimensions '
    f'($D = {D_ambient}$, matched distances)')
ax.legend()
ax.grid(True, alpha=0.3)

fig.tight_layout()
out = RESULTS_DIR / "manifold_dimension.png"
fig.savefig(out, dpi=150)
print(f"\nSaved: {out}")
plt.close()
