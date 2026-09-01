"""DocDiff model adapted from https://github.com/Royalvice/DocDiff.

Licensed under the MIT License; see third_party_licenses/DocDiff-MIT.txt.
"""

import math
from typing import Tuple, Union, List
import numpy as np
import torch
from torch import nn
from .schedule import Schedule
from .diffusionSample import GaussianDiffusion


def get_pad(in_,  ksize, stride, atrous=1):
    out_ = np.ceil(float(in_)/stride)
    return int(((out_ - 1) * stride + atrous*(ksize-1) + 1 - in_)/2)


class Swish(nn.Module):
    """
    ### Swish activation function
    """

    def forward(self, x):
        return x * torch.sigmoid(x)


class TimeEmbedding(nn.Module):
    """
    ### Embeddings for t
    """

    def __init__(self, n_channels: int):
        """
        * `n_channels` is the number of dimensions in the embedding
        """
        super().__init__()
        self.n_channels = n_channels
        # First linear layer
        self.lin1 = nn.Linear(self.n_channels // 4, self.n_channels)
        # Activation
        self.act = Swish()
        # Second linear layer
        self.lin2 = nn.Linear(self.n_channels, self.n_channels)

    def forward(self, t: torch.Tensor):
        # Create sinusoidal position embeddings
        # [same as those from the transformer](../../transformers/positional_encoding.html)
        #
        # \begin{align}
        # PE^{(1)}_{t,i} &= sin\Bigg(\frac{t}{10000^{\frac{i}{d - 1}}}\Bigg) \\
        # PE^{(2)}_{t,i} &= cos\Bigg(\frac{t}{10000^{\frac{i}{d - 1}}}\Bigg)
        # \end{align}
        #
        # where $d$ is `half_dim`
        half_dim = self.n_channels // 8
        emb = math.log(10_000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)

        # Transform with the MLP
        emb = self.act(self.lin1(emb))
        emb = self.lin2(emb)

        #
        return emb


class ResidualBlock(nn.Module):
    """
    ### Residual block
    A residual block has two convolution layers with group normalization.
    Each resolution is processed with two residual blocks.
    """

    def __init__(self, in_channels: int, out_channels: int, time_channels: int,
                 dropout: float = 0.1, is_noise: bool = True):
        """
        * `in_channels` is the number of input channels
        * `out_channels` is the number of input channels
        * `time_channels` is the number channels in the time step ($t$) embeddings
        * `n_groups` is the number of groups for [group normalization](../../normalization/group_norm/index.html)
        * `dropout` is the dropout rate
        """
        super().__init__()
        # Group normalization and the first convolution layer
        self.is_noise = is_noise
        self.act1 = Swish()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))

        # Group normalization and the second convolution layer

        self.act2 = Swish()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))

        # If the number of input channels is not equal to the number of output channels we have to
        # project the shortcut connection
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        else:
            self.shortcut = nn.Identity()

        # Linear layer for time embeddings
        if self.is_noise:
            self.time_emb = nn.Linear(time_channels, out_channels)
            self.time_act = Swish()

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        * `x` has shape `[batch_size, in_channels, height, width]`
        * `t` has shape `[batch_size, time_channels]`
        """
        # First convolution layer
        h = self.conv1(self.act1(x))
        # Add time embeddings
        if self.is_noise:
            h += self.time_emb(self.time_act(t))[:, :, None, None]
        # Second convolution layer
        h = self.conv2(self.dropout(self.act2(h)))

        # Add the shortcut connection and return
        return h + self.shortcut(x)


class DownBlock(nn.Module):
    """
    ### Down block
    This combines `ResidualBlock` and `AttentionBlock`. These are used in the first half of U-Net at each resolution.
    """

    def __init__(self, in_channels: int, out_channels: int, time_channels: int, is_noise: bool = True):
        super().__init__()
        self.res = ResidualBlock(in_channels, out_channels, time_channels, is_noise=is_noise)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res(x, t)
        return x


class UpBlock(nn.Module):
    """
    ### Up block
    This combines `ResidualBlock` and `AttentionBlock`. These are used in the second half of U-Net at each resolution.
    """

    def __init__(self, in_channels: int, out_channels: int, time_channels: int, is_noise: bool = True):
        super().__init__()
        # The input has `in_channels + out_channels` because we concatenate the output of the same resolution
        # from the first half of the U-Net
        self.res = ResidualBlock(in_channels + out_channels, out_channels, time_channels, is_noise=is_noise)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res(x, t)
        return x


class MiddleBlock(nn.Module):
    """
    ### Middle block
    It combines a `ResidualBlock`, `AttentionBlock`, followed by another `ResidualBlock`.
    This block is applied at the lowest resolution of the U-Net.
    """

    def __init__(self, n_channels: int, time_channels: int, is_noise: bool = True):
        super().__init__()
        self.res1 = ResidualBlock(n_channels, n_channels, time_channels, is_noise=is_noise)
        self.dia1 = nn.Conv2d(n_channels, n_channels, 3, 1, dilation=2, padding=get_pad(16, 3, 1, 2))
        self.dia2 = nn.Conv2d(n_channels, n_channels, 3, 1, dilation=4, padding=get_pad(16, 3, 1, 4))
        self.dia3 = nn.Conv2d(n_channels, n_channels, 3, 1, dilation=8, padding=get_pad(16, 3, 1, 8))
        self.dia4 = nn.Conv2d(n_channels, n_channels, 3, 1, dilation=16, padding=get_pad(16, 3, 1, 16))
        self.res2 = ResidualBlock(n_channels, n_channels, time_channels, is_noise=is_noise)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res1(x, t)
        x = self.dia1(x)
        x = self.dia2(x)
        x = self.dia3(x)
        x = self.dia4(x)
        x = self.res2(x, t)
        return x


class Upsample(nn.Module):
    """
    ### Scale up the feature map by $2 \times$
    """

    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.ConvTranspose2d(n_channels, n_channels, (4, 4), (2, 2), (1, 1))

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        # `t` is not used, but it's kept in the arguments because for the attention layer function signature
        # to match with `ResidualBlock`.
        _ = t
        return self.conv(x)


class Downsample(nn.Module):
    """
    ### Scale down the feature map by $\frac{1}{2} \times$
    """

    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.Conv2d(n_channels, n_channels, (3, 3), (2, 2), (1, 1))

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        # `t` is not used, but it's kept in the arguments because for the attention layer function signature
        # to match with `ResidualBlock`.
        _ = t
        return self.conv(x)


class UNet(nn.Module):
    """
    ## U-Net
    """

    def __init__(self, input_channels: int = 2, output_channels: int = 1, n_channels: int = 32,
                 ch_mults: Union[Tuple[int, ...], List[int]] = (1, 2, 2, 4),
                 n_blocks: int = 2, is_noise: bool = True):
        """
        * `image_channels` is the number of channels in the image. $3$ for RGB.
        * `n_channels` is number of channels in the initial feature map that we transform the image into
        * `ch_mults` is the list of channel numbers at each resolution. The number of channels is `ch_mults[i] * n_channels`
        * `is_attn` is a list of booleans that indicate whether to use attention at each resolution
        * `n_blocks` is the number of `UpDownBlocks` at each resolution
        """
        super().__init__()

        # Number of resolutions
        n_resolutions = len(ch_mults)

        # Project image into feature map
        self.image_proj = nn.Conv2d(input_channels, n_channels, kernel_size=(3, 3), padding=(1, 1))

        # Time embedding layer. Time embedding has `n_channels * 4` channels
        self.is_noise = is_noise
        if is_noise:
            self.time_emb = TimeEmbedding(n_channels * 4)

        # #### First half of U-Net - decreasing resolution
        down = []
        # Number of channels
        out_channels = in_channels = n_channels
        # For each resolution
        for i in range(n_resolutions):
            # Number of output channels at this resolution
            out_channels = n_channels * ch_mults[i]
            # Add `n_blocks`
            for _ in range(n_blocks):
                down.append(DownBlock(in_channels, out_channels, n_channels * 4, is_noise=is_noise))
                in_channels = out_channels
            # Down sample at all resolutions except the last
            if i < n_resolutions - 1:
                down.append(Downsample(in_channels))

        # Combine the set of modules
        self.down = nn.ModuleList(down)

        # Middle block
        self.middle = MiddleBlock(out_channels, n_channels * 4, is_noise=False)

        # #### Second half of U-Net - increasing resolution
        up = []
        # Number of channels
        in_channels = out_channels
        # For each resolution
        for i in reversed(range(n_resolutions)):
            # `n_blocks` at the same resolution
            out_channels = n_channels * ch_mults[i]
            for _ in range(n_blocks):
                up.append(UpBlock(in_channels, out_channels, n_channels * 4, is_noise=is_noise))
            # Final block to reduce the number of channels
            in_channels = n_channels * (ch_mults[i-1] if i >= 1 else 1)
            up.append(UpBlock(in_channels, out_channels, n_channels * 4, is_noise=is_noise))
            in_channels = out_channels
            # Up sample at all resolutions except last
            if i > 0:
                up.append(Upsample(in_channels))

        # Combine the set of modules
        self.up = nn.ModuleList(up)

        self.act = Swish()
        self.final = nn.Conv2d(in_channels, output_channels, kernel_size=(3, 3), padding=(1, 1))

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        * `x` has shape `[batch_size, in_channels, height, width]`
        * `t` has shape `[batch_size]`
        """

        # Get time-step embeddings
        if self.is_noise:
            t = self.time_emb(t)
        else:
            t = None
        # Get image projection
        x = self.image_proj(x)

        # `h` will store outputs at each resolution for skip connection
        h = [x]
        # First half of U-Net
        for m in self.down:
            x = m(x, t)
            h.append(x)

        # Middle (bottom)
        x = self.middle(x, t)

        # Second half of U-Net
        for m in self.up:
            if isinstance(m, Upsample):
                x = m(x, t)
            else:
                # Get the skip connection from first half of U-Net and concatenate
                s = h.pop()
                # print(x.shape, s.shape)
                x = torch.cat((x, s), dim=1)
                #
                x = m(x, t)

        # Final normalization and convolution
        return self.final(self.act(x))


class DocDiff(nn.Module):
    def __init__(self, input_channels:int=3, output_channels:int=3, n_channels:int=32,
                 ch_mults:Union[Tuple[int, ...], List[int]]=(1, 2, 3, 4),
                 n_blocks:int=1, time_steps:int=100):
        super(DocDiff, self).__init__()
        self.denoiser = UNet(input_channels * 2, output_channels, n_channels, ch_mults, n_blocks, is_noise=True)
        self.init_predictor = UNet(input_channels, output_channels, n_channels, ch_mults, n_blocks, is_noise=False)

        self.time_steps = time_steps
        self.schedule = Schedule('linear', self.time_steps)
        self.sampler = GaussianDiffusion(self.denoiser, self.time_steps, self.schedule)

    def forward(self, gt, img):
        t = torch.randint(0, self.time_steps, (gt.shape[0],)).long()
        init_predict = self.init_predictor(img, t)
        residual = gt - init_predict
        noisy_image, noise_ref = self.sampler.noisy_image(t, residual)
        noise_pred = self.denoiser(torch.cat((noisy_image, init_predict.clone().detach()), dim=1), t)
        return init_predict, noise_pred, noise_ref

    @torch.no_grad()
    def inference(self, img):
        noisyImage = torch.randn_like(img)
        init_predict = self.init_predictor(img, 0)
        sampledImgs = self.sampler(noisyImage, init_predict)
        finalImgs = (sampledImgs + init_predict)
        return finalImgs
