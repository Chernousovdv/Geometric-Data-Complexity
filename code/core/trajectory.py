"""
ODE trajectory embedding for dataset complexity.

Pipeline: data points (N, D) -> probability flow ODE -> trajectories (N, L, D)
-> weighted embedding (N, D*L).

Supports two noise schedules:

VE (variance-exploding):
    dz/dt = -ln(sigma_max/sigma_min) * sigma(t)^2 * s(z, t)

VP (variance-preserving):
    dz/dt = -1/2 * beta(t) * [z + s(z, t)]

Both with z(0) = x.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import beta as beta_dist


# ============================================================
# ODE solver
# ============================================================

def ode_trajectories(x, model, t_eval, rtol=1e-5, atol=1e-7):
    """
    Compute probability flow ODE trajectories for each data point.

    Automatically selects the correct ODE based on the model's noise schedule:
    - VE: dz/dt = -ln(sigma_max/sigma_min) * sigma(t)^2 * s(z, t)
    - VP: dz/dt = -1/2 * beta(t) * [z + s(z, t)]

    Args:
        x: (N, D) data points
        model: score model with .score(x, t) and .noise_schedule
        t_eval: (L,) time points at which to record z(t)
        rtol, atol: ODE solver tolerances

    Returns:
        trajectories: (N, L, D) array of z_x(t_l)
    """
    N, D = x.shape
    ns = model.noise_schedule
    is_vp = getattr(ns, 'schedule_type', 've') == 'vp'
    t_eval = np.sort(t_eval)
    L = len(t_eval)

    if not is_vp:
        log_ratio = np.log(ns.sigma_max / ns.sigma_min)

    trajectories = np.zeros((N, L, D))

    for i in range(N):
        def rhs(t, z):
            z_2d = z.reshape(1, D)
            t_arr = np.array([t])
            s = model.score(z_2d, t_arr)[0, 0, :]  # (D,)
            if is_vp:
                return -0.5 * ns.beta(t) * (z + s)
            else:
                return -log_ratio * ns.sigma(t) ** 2 * s

        sol = solve_ivp(
            rhs, [0, t_eval[-1]], x[i],
            t_eval=t_eval, method='RK45',
            rtol=rtol, atol=atol,
        )

        if sol.success:
            trajectories[i] = sol.y.T  # (L, D)
        else:
            print(f'  ODE failed for point {i}: {sol.message}')

    return trajectories


# ============================================================
# Weighting
# ============================================================

def beta_weights(t_eval, alpha, beta):
    """
    Compute Beta(alpha, beta) pdf weights at given time points.

    The trajectory distance becomes an expectation:
        D^2(x_i, x_j) = E_{t ~ Beta(alpha, beta)}[||z_{x_i}(t) - z_{x_j}(t)||^2]

    Weights are normalized to sum to 1.

    Args:
        t_eval: (L,) time points in (0, 1)
        alpha: Beta distribution shape parameter (> 0)
        beta: Beta distribution shape parameter (> 0)

    Returns:
        weights: (L,) normalized weights
    """
    w = beta_dist.pdf(t_eval, alpha, beta)
    w_sum = w.sum()
    if w_sum == 0:
        return np.ones(len(t_eval)) / len(t_eval)
    return w / w_sum


# ============================================================
# Embedding
# ============================================================

def trajectory_embedding(trajectories, weights=None):
    """
    Build embedding from ODE trajectories.

    f(x) = (sqrt(w_1) * z_x(t_1), ..., sqrt(w_L) * z_x(t_L))

    The squared distance ||f(x_i) - f(x_j)||^2 = sum_l w_l ||z_{x_i}(t_l) - z_{x_j}(t_l)||^2
    is a weighted trajectory distance (quadrature approximation to the continuous form).

    Args:
        trajectories: (N, L, D) ODE trajectories
        weights: (L,) time-weighting, or None for uniform

    Returns:
        embeddings: (N, D*L) flattened weighted trajectories
    """
    N, L, D = trajectories.shape
    if weights is None:
        weights = np.ones(L)
    weights = weights / weights.sum()  # normalize

    # Scale each time slice by sqrt(weight)
    w_sqrt = np.sqrt(weights)  # (L,)
    weighted = trajectories * w_sqrt[np.newaxis, :, np.newaxis]  # (N, L, D)
    return weighted.reshape(N, L * D)
