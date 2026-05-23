"""
Score function models for dataset complexity computation.

Two implementations behind the same interface:
- AnalyticScoreModel: closed-form score for Gaussian mixtures (no training)
- LearnedScoreModel: neural network trained via denoising score matching

Both expose .score(x, t) -> (N, L, D) numpy arrays.

Architecture registry:
    Use @register_architecture("name") to register new score network classes.
    LearnedScoreModel picks the architecture by name from the registry.
"""

import numpy as np
from datetime import datetime
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# --- Architecture registry ---

_ARCHITECTURE_REGISTRY = {}


def register_architecture(name):
    """
    Decorator to register a score network architecture.

    The registered class must be an nn.Module with:
        __init__(self, data_dim, **kwargs)
        forward(self, x, sigma) -> (B, data_dim)

    Example:
        @register_architecture("my_net")
        class MyScoreNet(nn.Module):
            def __init__(self, data_dim, hidden=128):
                ...
            def forward(self, x, sigma):
                ...
    """
    def decorator(cls):
        _ARCHITECTURE_REGISTRY[name] = cls
        return cls
    return decorator


def list_architectures():
    """Return list of registered architecture names."""
    return list(_ARCHITECTURE_REGISTRY.keys())


# --- Noise schedule ---

class NoiseSchedule:
    """
    Variance-exploding (VE) noise schedule.
    sigma(t) = sigma_min * (sigma_max / sigma_min)^t  for t in [0, 1].

    Forward kernel: q(x_t | x_0) = N(x_0, sigma(t)^2 I).
    """
    schedule_type = 've'

    def __init__(self, sigma_min=0.01, sigma_max=50.0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def sigma(self, t):
        """Return sigma(t). t can be scalar or array."""
        t = np.asarray(t, dtype=np.float64)
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** t

    def uniform_grid(self, L):
        """
        Return L noise levels uniformly spaced in [0, 1].
        These correspond to log-spaced sigma values.
        """
        return np.linspace(0, 1, L)

    def sigma_grid(self, L):
        """Return sigma values at L uniformly spaced time points."""
        return self.sigma(self.uniform_grid(L))


class VPNoiseSchedule:
    """
    Variance-preserving (VP) noise schedule.

    beta(t) = beta_min + (beta_max - beta_min) * t
    alpha_bar(t) = exp(-beta_min * t - 0.5 * (beta_max - beta_min) * t^2)

    Forward kernel: q(x_t | x_0) = N(sqrt(alpha_bar(t)) * x_0, (1 - alpha_bar(t)) I).

    At t=0: alpha_bar=1, so x_t = x_0 (clean data).
    At t=1: alpha_bar ~ 0, so x_t ~ N(0, I) (standard normal).
    """
    schedule_type = 'vp'

    def __init__(self, beta_min=0.1, beta_max=20.0):
        self.beta_min = beta_min
        self.beta_max = beta_max

    def beta(self, t):
        """Return beta(t). t can be scalar or array."""
        t = np.asarray(t, dtype=np.float64)
        return self.beta_min + (self.beta_max - self.beta_min) * t

    def alpha_bar(self, t):
        """Return alpha_bar(t) = exp(-integral of beta)."""
        t = np.asarray(t, dtype=np.float64)
        log_ab = -self.beta_min * t - 0.5 * (self.beta_max - self.beta_min) * t ** 2
        return np.exp(log_ab)

    def sigma(self, t):
        """Return sigma(t) = sqrt(1 - alpha_bar(t))."""
        return np.sqrt(1.0 - self.alpha_bar(t))

    def signal_rate(self, t):
        """Return sqrt(alpha_bar(t)) — the signal scaling factor."""
        return np.sqrt(self.alpha_bar(t))

    def uniform_grid(self, L):
        return np.linspace(0, 1, L)

    def sigma_grid(self, L):
        return self.sigma(self.uniform_grid(L))


# --- Analytic score model ---

class AnalyticScoreModel:
    """
    Exact score function for Gaussian mixture distributions.

    Given p(x) = sum_k pi_k N(x; mu_k, Sigma_k), the noisy marginal is:
        p_t(x) = sum_k pi_k N(x; mu_k, Sigma_k + sigma(t)^2 I)

    The score is:
        s(x, t) = sum_k w_k(x,t) * (Sigma_k + sigma(t)^2 I)^{-1} (mu_k - x)

    where w_k are posterior component weights.
    """

    def __init__(self, means, covariances=None, weights=None,
                 noise_schedule=None):
        """
        Args:
            means: (K, D) array of component means
            covariances: (K, D, D) array of covariance matrices,
                         or (K,) array of scalar variances (isotropic),
                         or None (unit covariance for all components)
            weights: (K,) array of mixture weights, or None (uniform)
            noise_schedule: NoiseSchedule instance, or None (default)
        """
        self.means = np.asarray(means, dtype=np.float64)
        self.n_components, self.dim = self.means.shape

        # Covariances
        if covariances is None:
            self.covs = np.tile(np.eye(self.dim), (self.n_components, 1, 1))
        else:
            covariances = np.asarray(covariances, dtype=np.float64)
            if covariances.ndim == 1:
                # Scalar variances -> isotropic
                self.covs = covariances[:, None, None] * np.eye(self.dim)
            else:
                self.covs = covariances

        # Weights
        if weights is None:
            self.weights = np.ones(self.n_components) / self.n_components
        else:
            self.weights = np.asarray(weights, dtype=np.float64)
            self.weights /= self.weights.sum()

        # Noise schedule
        self.noise_schedule = noise_schedule or NoiseSchedule()

    def score(self, x, t):
        """
        Compute s(x, t) = nabla_x log p_t(x).

        Supports both VE and VP noise schedules:
        - VE: p_t(x) = sum_k pi_k N(x; mu_k, Sigma_k + sigma(t)^2 I)
        - VP: p_t(x) = sum_k pi_k N(x; sqrt(abar)*mu_k, abar*Sigma_k + (1-abar)*I)

        Args:
            x: (N, D) array of data points
            t: (L,) array of noise levels (in [0, 1])

        Returns:
            scores: (N, L, D) array of score vectors
        """
        x = np.asarray(x, dtype=np.float64)
        t = np.asarray(t, dtype=np.float64)
        N, D = x.shape
        L = len(t)

        is_vp = getattr(self.noise_schedule, 'schedule_type', 've') == 'vp'
        scores = np.zeros((N, L, D))

        for l in range(L):
            if is_vp:
                abar = self.noise_schedule.alpha_bar(t[l])
                # q(x_t | x_0) = N(sqrt(abar)*x_0, (1-abar)*I)
                # Marginal: sum_k pi_k N(x; sqrt(abar)*mu_k, abar*Sigma_k + (1-abar)*I)
                scaled_means = np.sqrt(abar) * self.means  # (K, D)
            else:
                sig2 = self.noise_schedule.sigma(t[l]) ** 2

            log_probs = np.zeros((N, self.n_components))
            diffs = np.zeros((N, self.n_components, D))

            for k in range(self.n_components):
                if is_vp:
                    cov_noisy = abar * self.covs[k] + (1 - abar) * np.eye(D)
                    mean_k = scaled_means[k]
                else:
                    cov_noisy = self.covs[k] + sig2 * np.eye(D)
                    mean_k = self.means[k]

                cov_inv = np.linalg.inv(cov_noisy)
                diff = mean_k - x  # (N, D)
                diffs[:, k, :] = diff

                sign, logdet = np.linalg.slogdet(cov_noisy)
                mahal = np.sum(diff @ cov_inv * diff, axis=1)  # (N,)
                log_probs[:, k] = (np.log(self.weights[k])
                                   - 0.5 * D * np.log(2 * np.pi)
                                   - 0.5 * logdet
                                   - 0.5 * mahal)

            # Posterior weights via log-sum-exp
            log_probs_max = log_probs.max(axis=1, keepdims=True)
            w = np.exp(log_probs - log_probs_max)
            w /= w.sum(axis=1, keepdims=True)  # (N, K)

            # Score = sum_k w_k * cov_noisy_k^{-1} (mean_k - x)
            for k in range(self.n_components):
                if is_vp:
                    cov_noisy = abar * self.covs[k] + (1 - abar) * np.eye(D)
                else:
                    cov_noisy = self.covs[k] + sig2 * np.eye(D)
                cov_inv = np.linalg.inv(cov_noisy)
                scores[:, l, :] += (w[:, k:k+1]
                                    * (diffs[:, k, :] @ cov_inv.T))

        return scores.astype(np.float32)


# --- Built-in architectures ---

if HAS_TORCH:

    @register_architecture("mlp")
    class MLPScoreNet(nn.Module):
        """Simple MLP for score prediction on low-dimensional data."""

        def __init__(self, data_dim, hidden_dim=256, n_layers=4):
            super().__init__()
            self.data_dim = data_dim
            layers = [nn.Linear(data_dim + 1, hidden_dim), nn.SiLU()]
            for _ in range(n_layers - 1):
                layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
            layers.append(nn.Linear(hidden_dim, data_dim))
            self.net = nn.Sequential(*layers)

        def forward(self, x, sigma):
            """
            Args:
                x: (B, D) noisy data
                sigma: (B, 1) noise std
            Returns:
                score: (B, D) predicted score
            """
            return self.net(torch.cat([x, sigma], dim=-1))


    @register_architecture("mlp_small")
    class MLPScoreNetSmall(nn.Module):
        """Smaller MLP for very low-dimensional data."""

        def __init__(self, data_dim, hidden_dim=128, n_layers=3):
            super().__init__()
            self.data_dim = data_dim
            layers = [nn.Linear(data_dim + 1, hidden_dim), nn.SiLU()]
            for _ in range(n_layers - 1):
                layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
            layers.append(nn.Linear(hidden_dim, data_dim))
            self.net = nn.Sequential(*layers)

        def forward(self, x, sigma):
            return self.net(torch.cat([x, sigma], dim=-1))


    # --- Learned score model ---

    class LearnedScoreModel:
        """
        Neural network score model trained via denoising score matching.

        Training objective (noise prediction form):
            L = E_{t, x_0, eps} [ ||net(x_0 + sigma(t)*eps, sigma(t)) + eps/sigma(t)||^2 ]

        Usage:
            model = LearnedScoreModel(data_dim=2, arch="mlp")
            model.train(dataset, n_epochs=1000)
            model.save("results/checkpoints")
            scores = model.score(x, t)
        """

        def __init__(self, data_dim, arch="mlp", arch_kwargs=None,
                     noise_schedule=None, device='cpu', name=None):
            """
            Args:
                data_dim: dimensionality of data points
                arch: architecture name from registry (see list_architectures())
                arch_kwargs: dict of extra kwargs for the architecture
                noise_schedule: NoiseSchedule instance
                device: 'cpu' or 'cuda'
                name: model name (used in checkpoint filenames)
            """
            self.data_dim = data_dim
            self.arch_name = arch
            self.arch_kwargs = arch_kwargs or {}
            self.device = torch.device(device)
            self.noise_schedule = noise_schedule or NoiseSchedule()
            self.name = name or f"score_{arch}"

            if arch not in _ARCHITECTURE_REGISTRY:
                raise ValueError(
                    f"Unknown architecture '{arch}'. "
                    f"Available: {list_architectures()}"
                )

            net_cls = _ARCHITECTURE_REGISTRY[arch]
            self.net = net_cls(data_dim, **self.arch_kwargs)
            self.net.to(self.device)

            # Training state
            self._epoch = 0

        def train(self, dataset, n_epochs=1000, lr=1e-3, batch_size=256,
                  log_dir=None, log_every=100):
            """
            Train the score model on the given dataset.

            Args:
                dataset: Dataset instance (uses train split)
                n_epochs: number of training epochs
                lr: learning rate
                batch_size: batch size
                log_dir: directory for tensorboard logs (None = no logging)
                log_every: print/log frequency in epochs
            """
            writer = None
            if log_dir is not None:
                from torch.utils.tensorboard import SummaryWriter
                log_path = Path(log_dir) / self.name
                writer = SummaryWriter(str(log_path))

            loader = dataset.torch_loader(split='train', batch_size=batch_size)
            optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
            self.net.train()

            for epoch in range(n_epochs):
                total_loss = 0.0
                n_batches = 0

                for batch in loader:
                    x0 = batch.to(self.device)
                    B = x0.shape[0]

                    # Sample random noise levels
                    t = torch.rand(B, 1, device=self.device)
                    sigma_min = self.noise_schedule.sigma_min
                    sigma_max = self.noise_schedule.sigma_max
                    sigma = sigma_min * (sigma_max / sigma_min) ** t

                    # Add noise
                    eps = torch.randn_like(x0)
                    x_noisy = x0 + sigma * eps

                    # Predict score (target: -eps / sigma)
                    pred = self.net(x_noisy, sigma)
                    target = -eps / sigma
                    loss = ((pred - target) ** 2).mean()

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    n_batches += 1

                self._epoch += 1
                avg_loss = total_loss / n_batches

                if writer is not None:
                    writer.add_scalar('loss/train', avg_loss, self._epoch)

                if self._epoch % log_every == 0:
                    print(f"[{self.name}] Epoch {self._epoch}, "
                          f"loss: {avg_loss:.6f}")

            if writer is not None:
                writer.close()

        def score(self, x, t):
            """
            Evaluate s_theta(x, t).

            Args:
                x: (N, D) numpy array of data points
                t: (L,) numpy array of noise levels in [0, 1]

            Returns:
                scores: (N, L, D) numpy array of score vectors
            """
            self.net.eval()
            x_torch = torch.from_numpy(np.asarray(x, dtype=np.float32))
            x_torch = x_torch.to(self.device)

            sigmas = self.noise_schedule.sigma(t)
            N = x_torch.shape[0]
            L = len(t)
            D = self.data_dim

            scores = np.zeros((N, L, D), dtype=np.float32)

            with torch.no_grad():
                for l in range(L):
                    sigma_l = torch.full((N, 1), sigmas[l],
                                        dtype=torch.float32,
                                        device=self.device)
                    pred = self.net(x_torch, sigma_l)
                    scores[:, l, :] = pred.cpu().numpy()

            return scores

        # --- Save / Load ---

        def save(self, checkpoint_dir="results/checkpoints"):
            """
            Save model checkpoint.

            Saves to: {checkpoint_dir}/{name}_{timestamp}.pt
            The checkpoint contains everything needed to reconstruct
            and resume training.

            Returns:
                path: Path to saved checkpoint file
            """
            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.name}_{timestamp}.pt"
            path = checkpoint_dir / filename

            checkpoint = {
                'net_state_dict': self.net.state_dict(),
                'data_dim': self.data_dim,
                'arch_name': self.arch_name,
                'arch_kwargs': self.arch_kwargs,
                'noise_schedule': {
                    'sigma_min': self.noise_schedule.sigma_min,
                    'sigma_max': self.noise_schedule.sigma_max,
                },
                'name': self.name,
                'epoch': self._epoch,
            }

            torch.save(checkpoint, path)
            print(f"Saved checkpoint: {path}")
            return path

        @classmethod
        def load(cls, path, device='cpu'):
            """
            Load model from checkpoint.

            Args:
                path: path to .pt checkpoint file
                device: device to load onto

            Returns:
                model: LearnedScoreModel with weights restored
            """
            checkpoint = torch.load(path, map_location=device,
                                    weights_only=False)

            ns = NoiseSchedule(**checkpoint['noise_schedule'])
            model = cls(
                data_dim=checkpoint['data_dim'],
                arch=checkpoint['arch_name'],
                arch_kwargs=checkpoint['arch_kwargs'],
                noise_schedule=ns,
                device=device,
                name=checkpoint['name'],
            )
            model.net.load_state_dict(checkpoint['net_state_dict'])
            model._epoch = checkpoint['epoch']

            print(f"Loaded checkpoint: {path} (epoch {model._epoch})")
            return model
