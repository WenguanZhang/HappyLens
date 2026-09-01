"""FSNet training loss adapted for HappyLens.

Upstream: https://github.com/c-yn/FSNet
License: MIT; see third_party_licenses/FSNet-MIT.txt.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Loss(nn.modules.loss._Loss):
    def __init__(self):
        super(Loss, self).__init__()
        self.criterion = nn.L1Loss()
    
    def forward(self, recov, images):
        label_img2 = F.interpolate(images, scale_factor=0.5, mode='bilinear')
        label_img4 = F.interpolate(images, scale_factor=0.25, mode='bilinear')
        l1 = self.criterion(recov[0], label_img4)
        l2 = self.criterion(recov[1], label_img2)
        l3 = self.criterion(recov[2], images)
        loss_content = l1+l2+l3
        
        label_fft1 = torch.fft.fft2(label_img4, dim=(-2,-1))
        label_fft1 = torch.stack((label_fft1.real, label_fft1.imag), -1)

        pred_fft1 = torch.fft.fft2(recov[0], dim=(-2,-1))
        pred_fft1 = torch.stack((pred_fft1.real, pred_fft1.imag), -1)

        label_fft2 = torch.fft.fft2(label_img2, dim=(-2,-1))
        label_fft2 = torch.stack((label_fft2.real, label_fft2.imag), -1)

        pred_fft2 = torch.fft.fft2(recov[1], dim=(-2,-1))
        pred_fft2 = torch.stack((pred_fft2.real, pred_fft2.imag), -1)

        label_fft3 = torch.fft.fft2(images, dim=(-2,-1))
        label_fft3 = torch.stack((label_fft3.real, label_fft3.imag), -1)

        pred_fft3 = torch.fft.fft2(recov[2], dim=(-2,-1))
        pred_fft3 = torch.stack((pred_fft3.real, pred_fft3.imag), -1)

        f1 = self.criterion(pred_fft1, label_fft1)
        f2 = self.criterion(pred_fft2, label_fft2)
        f3 = self.criterion(pred_fft3, label_fft3)
        loss_fft = f1+f2+f3

        loss = loss_content + 0.1 * loss_fft
        return loss
