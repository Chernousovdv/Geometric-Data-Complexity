"""
Growth rate experiment on MNIST with pretrained diffusion model.

Two curves:
  1. Single-class: samples are all zeros (digit 0)
  2. Multi-class: samples cycle through 0, 1, 2, ..., 9, 0, 1, ...

Uses batched Euler integration for the probability flow ODE, which is
much faster than per-point adaptive RK45 for neural network score models.

Plots C(X) vs N for both.
"""

import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from exp4_mnist.pretrained import PretrainedMNISTModel
from core.trajectory import trajectory_embedding, beta_weights
from core.complexity import compute_complexity, calibrate_bandwidth

RESULTS_DIR = Path(__file__).parent / 'results'
RESULTS_DIR.mkdir(exist_ok=True)


def load_mnist_by_digit():
    """Load MNIST and group by digit. Returns dict: digit -> (M, 784) arrays."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    ds = datasets.MNIST(str(Path(__file__).parent / 'data'),
                        train=True, download=True, transform=transform)

    by_digit = {d: [] for d in range(10)}
    for img, label in ds:
        by_digit[label].append(img.numpy().flatten())

    return {d: np.array(v) for d, v in by_digit.items()}


@torch.no_grad()
def batched_ode_trajectories(x, model, t_eval, n_euler_steps=200):
    """
    Compute probability flow ODE trajectories using batched Euler integration.

    Much faster than per-point adaptive RK45 for neural network models,
    since we can batch all N points through the network at once.

    VP ODE: dz/dt = -1/2 * beta(t) * [z + s(z, t)]
    where s(z, t) = -eps_pred(z, t) / sigma(t)

    Args:
        x: (N, 784) numpy array, images in [-1, 1]
        model: PretrainedMNISTModel
        t_eval: (L,) time points to record
        n_euler_steps: number of Euler steps from t=0 to t=1

    Returns:
        trajectories: (N, L, 784) numpy array
    """
    device = model.device
    ns = model.noise_schedule
    N, D = x.shape
    L = len(t_eval)

    # Euler time grid
    dt = 1.0 / n_euler_steps
    t_grid = np.linspace(0, 1.0, n_euler_steps + 1)

    # Find which Euler steps are closest to t_eval
    record_indices = [np.argmin(np.abs(t_grid - te)) for te in t_eval]

    # Initialize z = x as torch tensor (N, 1, 28, 28)
    z = torch.from_numpy(x.reshape(N, 1, 28, 28).astype(np.float32)).to(device)

    trajectories = np.zeros((N, L, D), dtype=np.float32)

    for step in range(n_euler_steps):
        t = t_grid[step]
        t_next = t_grid[step + 1]

        # Record if this step matches a t_eval point
        if step in record_indices:
            j = record_indices.index(step)
            trajectories[:, j, :] = z.reshape(N, D).cpu().numpy()

        # VP ODE: dz/dt = -1/2 * beta(t) * [z + s(z, t)]
        # s(z, t) = -eps_pred / sigma(t)
        k = ns._t_to_k(t)
        timestep = torch.full((N,), k, device=device, dtype=torch.long)
        eps_pred = model.unet(z, timestep).sample

        sig = ns.sigma(t)
        score = -eps_pred / sig

        beta_t = ns.beta(t)
        dz_dt = -0.5 * beta_t * (z + score)
        z = z + dz_dt * dt

    # Record last point if needed
    if n_euler_steps in record_indices:
        j = record_indices.index(n_euler_steps)
        trajectories[:, j, :] = z.reshape(N, D).cpu().numpy()

    return trajectories


def build_cycling_sample(by_digit, N):
    """Build a sample of size N cycling through digits 0,1,...,9,0,1,..."""
    counts = {d: 0 for d in range(10)}
    samples = []
    for i in range(N):
        d = i % 10
        samples.append(by_digit[d][counts[d]])
        counts[d] += 1
    return np.array(samples)


def main():
    print('Loading MNIST...')
    by_digit = load_mnist_by_digit()

    print('\nLoading pretrained model...')
    model = PretrainedMNISTModel()

    # Trajectory parameters
    L = 10
    t_eval = np.linspace(0.05, 0.95, L)
    weights = beta_weights(t_eval, alpha=2.0, beta=2.0)

    N_values = [5, 10, 15, 20, 30, 40, 50]

    # Calibrate lambda from a reference sample (30 zeros, held out)
    print('\nCalibrating lambda...')
    ref_data = by_digit[0][200:230]
    ref_trajs = batched_ode_trajectories(ref_data, model, t_eval, n_euler_steps=200)
    ref_emb = trajectory_embedding(ref_trajs, weights)
    lambda_ = calibrate_bandwidth(ref_emb)
    print(f'  lambda = {lambda_:.4f}')

    C_single = []
    C_multi = []

    for N in N_values:
        print(f'\n--- N = {N} ---')

        # Single class (zeros), use images starting at index 0
        x_single = by_digit[0][:N]
        print(f'  Single-class trajectories ({N} zeros)...')
        trajs_s = batched_ode_trajectories(x_single, model, t_eval, n_euler_steps=200)
        emb_s = trajectory_embedding(trajs_s, weights)
        res_s = compute_complexity(emb_s, lambda_)
        C_single.append(res_s['C'])
        print(f'  C_single = {res_s["C"]:.4f}')

        # Multi-class (cycling)
        x_multi = build_cycling_sample(by_digit, N)
        print(f'  Multi-class trajectories ({N} cycling digits)...')
        trajs_m = batched_ode_trajectories(x_multi, model, t_eval, n_euler_steps=200)
        emb_m = trajectory_embedding(trajs_m, weights)
        res_m = compute_complexity(emb_m, lambda_)
        C_multi.append(res_m['C'])
        print(f'  C_multi  = {res_m["C"]:.4f}')

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(N_values, C_single, 'o-', color='steelblue', lw=2,
            label='Single class (digit 0)')
    ax.plot(N_values, C_multi, 's-', color='firebrick', lw=2,
            label='Multi-class (cycling 0–9)')
    ax.set_xlabel('Sample size N', fontsize=12)
    ax.set_ylabel('C(X)', fontsize=12)
    ax.set_title('Dataset Complexity: Growth Rate on MNIST', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = RESULTS_DIR / 'mnist_growth.png'
    plt.savefig(path, dpi=150)
    print(f'\nSaved: {path}')
    plt.close()

    # Summary
    print('\nSummary:')
    print(f'{"N":>5} {"C_single":>10} {"C_multi":>10} {"ratio":>8}')
    for i, N in enumerate(N_values):
        ratio = C_multi[i] / C_single[i] if C_single[i] > 0 else float('inf')
        print(f'{N:>5} {C_single[i]:>10.4f} {C_multi[i]:>10.4f} {ratio:>8.2f}')


if __name__ == '__main__':
    main()
