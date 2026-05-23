"""
Variance-preserving diffusion utilities for training and sampling.

Matches the VP schedule in core/model.py:
    beta(t) = beta_min + (beta_max - beta_min) * t
    alpha_bar(t) = exp(-beta_min * t - 0.5 * (beta_max - beta_min) * t^2)

Training: eps-prediction loss
Sampling: DDPM reverse process
Score extraction: s(x, t) = -eps_pred / sigma(t)
"""

import torch
import torch.nn as nn
import numpy as np


class VPDiffusion:
    """
    VP diffusion process for training and sampling.

    Images are expected in [-1, 1].

    Args:
        beta_min: minimum noise rate
        beta_max: maximum noise rate
        device: torch device
    """

    def __init__(self, beta_min=0.1, beta_max=20.0, device='cpu'):
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.device = device

    def alpha_bar(self, t):
        """alpha_bar(t) = exp(-beta_min*t - 0.5*(beta_max-beta_min)*t^2)"""
        return torch.exp(-self.beta_min * t - 0.5 * (self.beta_max - self.beta_min) * t ** 2)

    def sigma(self, t):
        """sigma(t) = sqrt(1 - alpha_bar(t))"""
        return torch.sqrt(1.0 - self.alpha_bar(t))

    def q_sample(self, x0, t, noise=None):
        """
        Forward process: sample x_t ~ q(x_t | x_0).

        x_t = sqrt(alpha_bar(t)) * x0 + sqrt(1 - alpha_bar(t)) * eps

        Args:
            x0: (B, C, H, W) clean images in [-1, 1]
            t: (B,) times in [0, 1]
            noise: optional pre-sampled noise

        Returns:
            x_t: (B, C, H, W) noisy images
            noise: (B, C, H, W) the noise that was added
        """
        if noise is None:
            noise = torch.randn_like(x0)
        abar = self.alpha_bar(t)[:, None, None, None]
        x_t = torch.sqrt(abar) * x0 + torch.sqrt(1.0 - abar) * noise
        return x_t, noise

    def training_loss(self, model, x0):
        """
        Compute denoising score matching loss.

        L = E_{t, eps}[||model(x_t, t) - eps||^2]

        Args:
            model: UNet that predicts noise
            x0: (B, C, H, W) clean images in [-1, 1]

        Returns:
            loss: scalar
        """
        B = x0.shape[0]
        t = torch.rand(B, device=x0.device)
        # Avoid t=0 and t=1 for numerical stability
        t = t * 0.998 + 0.001
        x_t, noise = self.q_sample(x0, t)
        noise_pred = model(x_t, t)
        return nn.functional.mse_loss(noise_pred, noise)

    @torch.no_grad()
    def sample_ddpm(self, model, shape, n_steps=1000):
        """
        Generate samples via DDPM reverse process.

        Uses discrete-time approximation with n_steps steps.

        Args:
            model: trained UNet
            shape: (B, C, H, W) output shape
            n_steps: number of reverse steps

        Returns:
            x: (B, C, H, W) generated images in [-1, 1]
        """
        device = next(model.parameters()).device
        dt = 1.0 / n_steps
        x = torch.randn(shape, device=device)

        for i in reversed(range(n_steps)):
            t = torch.full((shape[0],), (i + 1) * dt, device=device)
            t_prev = torch.full((shape[0],), i * dt, device=device)

            abar = self.alpha_bar(t)[:, None, None, None]
            abar_prev = self.alpha_bar(t_prev)[:, None, None, None]

            # Predict noise
            eps_pred = model(x, t)

            # Predict x0
            x0_pred = (x - torch.sqrt(1 - abar) * eps_pred) / torch.sqrt(abar)
            x0_pred = x0_pred.clamp(-1, 1)

            # Compute x_{t-1}
            if i > 0:
                beta_t = self.beta_min + (self.beta_max - self.beta_min) * t[0]
                beta_t = beta_t * dt
                sigma_t = torch.sqrt(beta_t)
                noise = torch.randn_like(x)
                # Posterior mean
                coeff1 = torch.sqrt(abar_prev) * beta_t / (1 - abar)
                coeff2 = torch.sqrt(1 - beta_t) * (1 - abar_prev) / (1 - abar)
                x = coeff1 * x0_pred + coeff2 * x + sigma_t * noise
            else:
                x = x0_pred

        return x.clamp(-1, 1)

    @torch.no_grad()
    def sample_ddim(self, model, shape, n_steps=50):
        """
        Generate samples via DDIM (deterministic, faster).

        Args:
            model: trained UNet
            shape: (B, C, H, W) output shape
            n_steps: number of reverse steps

        Returns:
            x: (B, C, H, W) generated images in [-1, 1]
        """
        device = next(model.parameters()).device
        # Time steps from 1 to ~0
        times = torch.linspace(1.0, 0.001, n_steps + 1, device=device)

        x = torch.randn(shape, device=device)

        for i in range(n_steps):
            t = times[i].expand(shape[0])
            t_next = times[i + 1].expand(shape[0])

            abar = self.alpha_bar(t)[:, None, None, None]
            abar_next = self.alpha_bar(t_next)[:, None, None, None]

            eps_pred = model(x, t)

            # Predict x0
            x0_pred = (x - torch.sqrt(1 - abar) * eps_pred) / torch.sqrt(abar)
            x0_pred = x0_pred.clamp(-1, 1)

            # DDIM step (eta=0, deterministic)
            x = torch.sqrt(abar_next) * x0_pred + torch.sqrt(1 - abar_next) * eps_pred

        return x.clamp(-1, 1)

    def score(self, model, x, t):
        """
        Extract score s(x, t) = -eps_pred / sigma(t) from the trained model.

        This is the interface used by core/trajectory.py for ODE integration.

        Args:
            model: trained UNet
            x: (B, C, H, W) images
            t: (B,) or scalar time

        Returns:
            score: (B, C, H, W) score vectors
        """
        if not isinstance(t, torch.Tensor):
            t = torch.full((x.shape[0],), t, device=x.device)
        eps_pred = model(x, t)
        sig = self.sigma(t)[:, None, None, None]
        return -eps_pred / sig


class EMA:
    """
    Exponential moving average for model parameters.

    Usage:
        ema = EMA(model, decay=0.9999)
        # In training loop:
        ema.update()
        # For evaluation:
        ema.apply()
        model.eval(); sample(model)
        ema.restore()
    """

    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1 - self.decay)

    def apply(self):
        """Replace model params with EMA params."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self):
        """Restore original model params."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])
        self.backup = {}
