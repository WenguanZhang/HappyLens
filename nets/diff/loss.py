"""DocDiff training loss adapted from https://github.com/Royalvice/DocDiff.

Licensed under the MIT License; see third_party_licenses/DocDiff-MIT.txt.
"""

import torch.nn as nn
from .sobel import Laplacian

class Loss(nn.modules.loss._Loss):
    def __init__(self, time_steps=100, beta=50):
        super(Loss, self).__init__()
        self.loss = nn.MSELoss()
        # self.loss = nn.L1Loss()
        self.high_filter = Laplacian()

        self.time_steps = time_steps
        self.beta = beta
        
    def forward(self, gt, init_predict, noise_pred, noise_ref):
        residual_high = self.high_filter(gt - init_predict)
        ddpm_loss = 2 * self.loss(self.high_filter(noise_pred), residual_high) + self.loss(noise_pred, gt - init_predict)
        
        low_high_loss = self.loss(init_predict, gt)
        low_freq_loss = self.loss(init_predict - self.high_filter(init_predict), gt - self.high_filter(gt))
        pixel_loss = low_high_loss + 2 * low_freq_loss
        
        loss = ddpm_loss + self.beta * (pixel_loss) / self.time_steps
        return loss
