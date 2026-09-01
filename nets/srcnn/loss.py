import torch.nn as nn

class Loss(nn.modules.loss._Loss):
    def __init__(self):
        super(Loss, self).__init__()
    
    def forward(self, recov, images):
        return nn.functional.mse_loss(recov, images)