"""DocDiff diffusion sampler adapted from https://github.com/Royalvice/DocDiff.

Licensed under the MIT License; see third_party_licenses/DocDiff-MIT.txt.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


def extract_(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    out = torch.gather(v, index=t, dim=0).float()
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


class GaussianDiffusion(nn.Module):
    def __init__(self, model, T, schedule):
        super().__init__()
        self.visual = False
        if self.visual:
            self.num = 0
        self.model = model
        self.T = T
        self.schedule = schedule
        betas = self.schedule.get_betas()
        self.register_buffer('betas', betas.float())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]
        gammas = alphas_bar

        self.register_buffer('coeff1', torch.sqrt(1. / alphas))
        self.register_buffer('coeff2', self.coeff1 * (1. - alphas) / torch.sqrt(1. - alphas_bar))
        self.register_buffer('posterior_var', self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))

        # calculation for q(y_t|y_{t-1})
        self.register_buffer('gammas', gammas)
        self.register_buffer('sqrt_one_minus_gammas', torch.sqrt(1 - gammas))
        self.register_buffer('sqrt_gammas', torch.sqrt(gammas))

    def noisy_image(self, t, y):
        """ Compute y_noisy according to (6) p15 of [2]"""
        noise = torch.randn_like(y)
        y_noisy = extract_(self.sqrt_gammas, t, y.shape) * y + extract_(self.sqrt_one_minus_gammas, t, noise.shape) * noise
        return y_noisy, noise

    def forward(self, x_T, cond):
        x_t = x_T
        cond_ = cond
        for time_step in tqdm(reversed(range(self.T)), total=self.T, desc="time_step"):
            
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step

            if time_step > 0:
                ori = self.model(torch.cat((x_t, cond_), dim=1), t)
                eps = x_t - extract_(self.sqrt_gammas, t, ori.shape) * ori
                eps = eps / extract_(self.sqrt_one_minus_gammas, t, eps.shape)
                x_t = extract_(self.sqrt_gammas, t - 1, ori.shape) * ori + extract_(self.sqrt_one_minus_gammas, t - 1, eps.shape) * eps
            else:
                x_t = self.model(torch.cat((x_t, cond_), dim=1), t)

        x_0 = x_t
        return x_0
