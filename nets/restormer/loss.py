"""Restormer training loss adapted for HappyLens.

Upstream: https://github.com/swz30/Restormer
License: MIT; see third_party_licenses/Restormer-MIT.txt.
"""

import torch.nn as nn

class Loss(nn.modules.loss._Loss):
    def __init__(self):
        super(Loss, self).__init__()
    
    def forward(self, recov, images):
        return nn.functional.l1_loss(recov, images, reduction='mean')
