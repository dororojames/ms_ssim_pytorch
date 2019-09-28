'''
© 2019, JamesChan
forked from https://github.com/One-sixth/ms_ssim_pytorch/ssim.py
'''
import torch
import torch.jit
import torch.nn.functional as F
from torch import arange, exp, mean, prod, stack, sum


@torch.jit.script
def rerange(tensor):
    if torch.max(tensor) > 1.5:
        return tensor
    scaled = tensor * 255.
    return scaled - (scaled - scaled.round()).detach()


@torch.jit.script
def create_window(window_size: int, sigma: float, channel: int):
    '''
    Create 1-D gauss kernel
    :param window_size: the size of gauss kernel
    :param sigma: sigma of normal distribution
    :param channel: input channel
    :return: 1D kernel
    '''
    half_window = window_size // 2
    coords = arange(-half_window, half_window+1, dtype=torch.float)

    g = exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()

    return g.reshape(1, 1, 1, -1).repeat(channel, 1, 1, 1)


@torch.jit.script
def _gaussian_filter(x, window_1d, use_padding: bool):
    '''
    Blur input with 1-D kernel
    :param x: batch of tensors to be blured
    :param window_1d: 1-D gauss kernel
    :param use_padding: padding image before conv
    :return: blured tensors
    '''
    C = x.size(1)
    padding = 0 if not use_padding else window_1d.size(3) // 2
    out = F.conv2d(x, window_1d, stride=1, padding=(0, padding), groups=C)
    out = F.conv2d(out, window_1d.transpose(2, 3),
                   stride=1, padding=(padding, 0), groups=C)
    return out


@torch.jit.script
def ssim(X, Y, window, data_range: float, use_padding: bool = False):
    '''
    Calculate ssim index for X and Y
    :param X: images
    :param Y: images
    :param window: 1-D gauss kernel
    :param data_range: value range of input images. (usually 1.0 or 255)
    :param use_padding: padding image before conv
    :return:
    '''

    K1, K2 = 0.01, 0.03
    compensation = 1.0

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    mu1 = _gaussian_filter(X, window, use_padding)
    mu2 = _gaussian_filter(Y, window, use_padding)
    sigma1_sq = _gaussian_filter(X * X, window, use_padding)
    sigma2_sq = _gaussian_filter(Y * Y, window, use_padding)
    sigma12 = _gaussian_filter(X * Y, window, use_padding)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = compensation * (sigma1_sq - mu1_sq)
    sigma2_sq = compensation * (sigma2_sq - mu2_sq)
    sigma12 = compensation * (sigma12 - mu1_mu2)

    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map

    ssim_val = ssim_map.mean(dim=(1, 2, 3))  # reduce along CHW
    cs = cs_map.mean(dim=(1, 2, 3))

    return ssim_val, cs


@torch.jit.script
def ms_ssim(X, Y, window, data_range: float, weights, use_padding: bool = False):
    '''
    interface of ms-ssim
    :param X: a batch of images, (N,C,H,W)
    :param Y: a batch of images, (N,C,H,W)
    :param window: 1-D gauss kernel
    :param data_range: value range of input images. (usually 1.0 or 255)
    :param weights: weights for different levels
    :param use_padding: padding image before conv
    :return:
    '''
    levels = weights.size(0)
    cs_vals, ssim_vals = [], []
    for _ in range(levels):
        ssim_val, cs = ssim(X, Y, window=window,
                            data_range=data_range, use_padding=use_padding)
        cs_vals.append(cs)
        ssim_vals.append(ssim_val)
        padding = (X.size(-2) % 2, X.size(-1) % 2)
        X = F.avg_pool2d(X, kernel_size=2, padding=padding)
        Y = F.avg_pool2d(Y, kernel_size=2, padding=padding)

    ms_cs_vals = stack(cs_vals[:-1], dim=0) ** weights[:-1].unsqueeze(1)
    ms_ssim_val = prod(ms_cs_vals * (ssim_vals[-1] ** weights[-1]), dim=0)
    return ms_ssim_val


class SSIM(torch.jit.ScriptModule):
    __constants__ = ['scale', 'data_range', 'use_padding', 'reduction']

    def __init__(self, window_size=11, window_sigma=1.5, scale=True, data_range=255., channel=3, use_padding=False, reduction="none"):
        '''
        :param window_size: the size of gauss kernel
        :param window_sigma: sigma of normal distribution
        :param scale: rerange to 255.
        :param data_range: value range of input images. (usually 1.0 or 255)
        :param channel: input channels (default: 3)
        :param use_padding: padding image before conv
        :param reduction: reduction mode
        '''
        super().__init__()
        assert window_size % 2 == 1, 'Window size must be odd.'
        window = create_window(window_size, window_sigma, channel)
        self.register_buffer('window', window)
        self.scale = scale
        self.data_range = 255. if scale else data_range
        self.use_padding = use_padding
        self.reduction = reduction

    @torch.jit.script_method
    def forward(self, input, target):
        if self.scale:
            input, target = rerange(input), rerange(target)

        ret = ssim(input, target, window=self.window,
                   data_range=self.data_range, use_padding=self.use_padding)[0]
        if self.reduction != 'none':
            ret = mean(ret) if self.reduction == 'mean' else sum(ret)
        return ret


class MS_SSIM(torch.jit.ScriptModule):
    __constants__ = ['scale', 'data_range', 'use_padding', 'reduction']

    def __init__(self, window_size=11, window_sigma=1.5, scale=True, data_range=255., channel=3, use_padding=False, weights=None, levels=None, reduction="none"):
        '''
        :param window_size: the size of gauss kernel
        :param window_sigma: sigma of normal distribution
        :param scale: rerange to 255.
        :param data_range: value range of input images. (usually 1.0 or 255)
        :param channel: input channels
        :param use_padding: padding image before conv
        :param weights: weights for different levels. (default [0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
        :param levels: number of downsampling
        :param reduction: reduction mode
        '''
        super().__init__()
        assert window_size % 2 == 1, 'Window size must be odd.'
        self.scale = scale
        self.data_range = 255. if scale else data_range
        self.use_padding = use_padding
        self.reduction = reduction

        window = create_window(window_size, window_sigma, channel)
        self.register_buffer('window', window)

        if weights is None:
            weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
        weights = torch.tensor(weights, dtype=torch.float)

        if levels is not None:
            weights = weights[:levels]
            weights = weights / weights.sum()

        self.register_buffer('weights', weights)

    @torch.jit.script_method
    def forward(self, input, target):
        if self.scale:
            input, target = rerange(input), rerange(target)

        ret = ms_ssim(input, target, window=self.window, data_range=self.data_range,
                      weights=self.weights, use_padding=self.use_padding)
        if self.reduction != 'none':
            ret = mean(ret) if self.reduction == 'mean' else sum(ret)
        return ret
