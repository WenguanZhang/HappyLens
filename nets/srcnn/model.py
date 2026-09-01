from torch import nn

class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, stride=1, padding=4),
            nn.ReLU(True)
        )
        
        self.map = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True)
        )
        
        self.reconstruction = nn.Conv2d(32, 3, kernel_size=5, stride=1, padding=2)
        
        
    def forward(self, x):
        x = self.features(x)
        x = self.map(x)
        x = self.reconstruction(x)
        return x
    