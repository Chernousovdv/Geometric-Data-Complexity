"""
Wrapper around HuggingFace pretrained DDPM for MNIST.

Adapts the diffusers UNet2DModel to our score interface so it can be
used with core/trajectory.py for ODE trajectory computation.

The pretrained model uses:
    - Discrete linear beta schedule: beta_start=0.0001, beta_end=0.02, T=1000
    - eps-prediction: model(x_t, k) predicts noise eps
    - Score extraction: s(x, t) = -eps / sigma(t)

We map continuous time t in [0, 1] to discrete timestep k in {0, ..., 999}
via matching alpha_bar values.
"""

import numpy as np
import torch
from diffusers import DDPMPipeline


class DDPMNoiseSchedule:
    """
    Discrete linear beta schedule matching the pretrained model.

    Precomputes alpha_bar[k] for k = 0, ..., T-1 and provides
    continuous-time interface via interpolation.
    """

    schedule_type = 'vp'

    def __init__(self, beta_start=0.0001, beta_end=0.02, T=1000):
        self.T = T
        self.beta_start = beta_start
        self.beta_end = beta_end

        # Discrete betas and alpha_bars
        betas = np.linspace(beta_start, beta_end, T)
        alphas = 1.0 - betas
        self.alpha_bars = np.cumprod(alphas)  # (T,)
        self.betas_discrete = betas

    def _t_to_k(self, t):
        """Map continuous t in [0, 1] to discrete timestep k."""
        k = int(round(t * (self.T - 1)))
        return max(0, min(self.T - 1, k))

    def alpha_bar(self, t):
        """alpha_bar at continuous time t."""
        k = self._t_to_k(t)
        return self.alpha_bars[k]

    def sigma(self, t):
        """sigma(t) = sqrt(1 - alpha_bar(t))"""
        return np.sqrt(1.0 - self.alpha_bar(t))

    def beta(self, t):
        """Instantaneous beta for the VP ODE drift."""
        # Linear interpolation matching beta_start -> beta_end over [0, 1]
        # Scale by T to get the continuous-time rate
        # In discrete: x_{k+1} = sqrt(1-beta_k) x_k + sqrt(beta_k) eps
        # The continuous VP ODE uses beta(t) where integral matches discrete
        # For linear schedule: beta(t) ≈ T * (beta_start + (beta_end - beta_start) * t)
        return self.T * (self.beta_start + (self.beta_end - self.beta_start) * t)


class PretrainedMNISTModel:
    """
    Pretrained DDPM model for MNIST with score interface.

    Compatible with core/trajectory.py's ode_trajectories().

    Usage:
        model = PretrainedMNISTModel()
        # model.score(x, t) returns (N, L, D) numpy scores
        # model.noise_schedule has .beta(t), .sigma(t), schedule_type='vp'
    """

    def __init__(self, model_id='1aurent/ddpm-mnist', device=None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device

        pipe = DDPMPipeline.from_pretrained(model_id)
        self.unet = pipe.unet.to(device).eval()
        self.noise_schedule = DDPMNoiseSchedule(
            beta_start=pipe.scheduler.config.beta_start,
            beta_end=pipe.scheduler.config.beta_end,
            T=pipe.scheduler.config.num_train_timesteps,
        )

    def score(self, x, t):
        """
        Compute score s(x, t) = -eps_pred / sigma(t).

        Args:
            x: (N, D) numpy array, D = 784 (flattened 28x28 images in [-1, 1])
            t: (L,) numpy array of times in [0, 1]

        Returns:
            scores: (N, L, D) numpy array
        """
        x = np.asarray(x, dtype=np.float32)
        t = np.asarray(t, dtype=np.float64)
        N, D = x.shape
        L = len(t)

        assert D == 784, f'Expected D=784 (28x28), got D={D}'

        scores = np.zeros((N, L, D), dtype=np.float32)

        with torch.no_grad():
            # Reshape to image format: (N, 1, 28, 28)
            x_img = torch.from_numpy(x.reshape(N, 1, 28, 28)).to(self.device)

            for j, tj in enumerate(t):
                # Map continuous t to discrete timestep
                k = self.noise_schedule._t_to_k(tj)
                timestep = torch.full((N,), k, device=self.device, dtype=torch.long)

                # Get eps prediction
                eps_pred = self.unet(x_img, timestep).sample  # (N, 1, 28, 28)

                # Score = -eps / sigma(t)
                sig = self.noise_schedule.sigma(tj)
                score_img = -eps_pred / sig

                # Flatten back to (N, D)
                scores[:, j, :] = score_img.reshape(N, D).cpu().numpy()

        return scores
