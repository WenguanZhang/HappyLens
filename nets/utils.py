import torch
import torch.nn.functional as F
from torch.fft import fft2, ifft2, fftshift, ifftshift
import os
from torchvision.transforms.functional import to_pil_image

class Adder(object):
    def __init__(self):
        self.count = 0
        self.num = float(0)

    def reset(self):
        self.count = 0
        self.num = float(0)

    def __call__(self, num):
        if num != float('inf'):
            self.count += 1
            self.num += num
        return num

    def average(self):
        return self.num / self.count
    
    
def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def postprocess(img, rgb_range):
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)


def save_images(img, path, name:str):
    save_pth = os.path.join(path, f'{name}.png')
    img_clip = torch.clamp(img, 0, 1)        
    out = to_pil_image(img_clip.squeeze(0).cpu(), 'RGB')
    out.save(save_pth)

def crop_or_pad_tensor(img_tensor:torch.Tensor, target_h:int, target_w:int):
    # img_tensor: [c, h, w]
    _, h, w = img_tensor.shape

    pad_top = max((target_h - h) // 2, 0)
    pad_bottom = max(target_h - h - pad_top, 0)
    pad_left = max((target_w - w) // 2, 0)
    pad_right = max(target_w - w - pad_left, 0)

    if pad_top + pad_bottom + pad_left + pad_right > 0:
        img_tensor = F.pad(
            img_tensor, 
            pad=(pad_left, pad_right, pad_top, pad_bottom),
            mode="reflect",
        )

    _, new_h, new_w = img_tensor.shape
    start_h = (new_h - target_h) // 2
    start_w = (new_w - target_w) // 2
    img_tensor = img_tensor[:, start_h : start_h + target_h, start_w : start_w + target_w]
    return img_tensor

def img_conv_mul(img:torch.Tensor, psf:torch.Tensor, mode='fft'):
    # psf: [B, C, 2M + 1, 2N + 1]
    # img: [B, C, H, W]
    # cut shade edge
    cut_x = psf.shape[-2] // 2
    cut_y = psf.shape[-1] // 2
    
    if mode == 'conv2d':
        #! conv2d
        img_blur = img.new_zeros(img.shape)
        # zero pad
        img_pad = torch.nn.functional.pad(img, (cut_y, cut_y, cut_x, cut_x))
        for i in range(img.shape[0]):
            img_blur[i] = torch.nn.functional.conv2d(img_pad[i], psf[i].flip(1).flip(2).unsqueeze(1), groups=psf.shape[1]).squeeze()
            
    elif mode == 'fft':
        #! fft
        padx = 2 * max([img.shape[-1], psf.shape[-1]]) - img.shape[-1]
        pady = 2 * max([img.shape[-2], psf.shape[-2]]) - img.shape[-2]
        
        img_pad = torch.nn.functional.pad(img, (0, padx, 0, pady))
    
        psf_pad = torch.nn.functional.pad(psf, (0, img_pad.shape[-1]-psf.shape[-1], 0, img_pad.shape[-2]-psf.shape[-2]))
        shiftx = - (psf.shape[-2] // 2)
        shifty = - (psf.shape[-1] // 2)
        psf_pad = torch.roll(psf_pad, (shiftx, shifty), (-2, -1))
        
        img_pad_f = fftshift(fft2(img_pad))
        psf_pad_f = fftshift(fft2(psf_pad))
        img_blur = torch.real(ifft2(ifftshift(img_pad_f * psf_pad_f)))
        img_blur = img_blur[:, :, 0:img.shape[-2], 0:img.shape[-1]]
    else:
        raise NotImplementedError
    
    img_label = img[:, :, cut_x:-cut_x, cut_y:-cut_y]
    img_blur = img_blur[:, :, cut_x:-cut_x, cut_y:-cut_y]
    return img_blur, img_label

def simulate_rgb(gts:torch.Tensor, psfs:torch.Tensor, rl:torch.Tensor, sigma:torch.Tensor, lamb:torch.Tensor, mode='fft'):
    """
    gts: [B, 3, H+2M, W+2N]
    psfs: [B, 3, 2M+1, 2N+1]
    rl: [B] / [B, H, W] for relative illumination
    sigma: [B, 1, 1, 1] for gauss noise
    lamb: [B, 1, 1, 1] for poisson noise
    """
    blurs, labels = img_conv_mul(gts, psfs, mode) # [B, C, H, W]
    cut_x = psfs.shape[-2] // 2
    cut_y = psfs.shape[-1] // 2
    rl = rl[:, None, None, None] if rl.ndim == 1 else rl[:, None, cut_x:-cut_x, cut_y:-cut_y]
    blurs = blurs * rl
    with torch.no_grad():
        noise = torch.poisson(blurs.clip(0., 1.) / lamb) * lamb + torch.randn_like(blurs) * sigma - blurs
    blurs = blurs + noise                
    blurs, labels = blurs.clip(0., 1.), labels.clip(0., 1.)
    return blurs, labels

def simulate_raw(raw:torch.Tensor, color:torch.Tensor, psf:torch.Tensor, rl:torch.Tensor, sigma:torch.Tensor, lamb:torch.Tensor, mode='fft'):
    """
    raw: [B, H+2M, W+2N]
    color: [B, H+2M, W+2N]
    psf: [B, 3, 2M+1, 2N+1]
    rl: [B] / [B, H, W] for relative illumination
    sigma: [B, 1, 1] for gauss noise
    lamb: [B, 1, 1] for poisson noise
    """
    # get raw demosaic
    with torch.no_grad():
        _raw_label = demosaic(raw, color, method='Malvar') # [B, 3, H, W]
    
    # get color (cut)
    cut_x = psf.shape[-2] // 2
    cut_y = psf.shape[-1] // 2
    color = color[:, cut_x:-cut_x, cut_y:-cut_y]

    # get raw label (cut)
    raw_label = raw[:, cut_x:-cut_x, cut_y:-cut_y]
    
    # get degraded raw (cut)
    raw_de_r, _ = img_conv_mul(_raw_label[:, 0, :, :][:, None, :, :], psf[:, 0, :, :][:, None, :, :], mode=mode) # [B, 1, H, W]
    raw_de_g, _ = img_conv_mul(_raw_label[:, 1, :, :][:, None, :, :], psf[:, 1, :, :][:, None, :, :], mode=mode) # [B, 1, H, W]
    raw_de_b, _ = img_conv_mul(_raw_label[:, 2, :, :][:, None, :, :], psf[:, 2, :, :][:, None, :, :], mode=mode) # [B, 1, H, W]
    
    mask_r = torch.zeros_like(color)
    mask_g = torch.zeros_like(color)
    mask_b = torch.zeros_like(color)

    for i, col in enumerate(b'RGBG'):
        if chr(col) == 'R':
            mask_r = mask_r.masked_fill(torch.eq(color, i), 1)
        if chr(col) == 'G':
            mask_g = mask_g.masked_fill(torch.eq(color, i), 1)
        if chr(col) == 'B':
            mask_b = mask_b.masked_fill(torch.eq(color, i), 1)
    
    raw_r = torch.mul(raw_de_r[:, 0, :, :], mask_r) # [B, H, W]
    raw_g = torch.mul(raw_de_g[:, 0, :, :], mask_g)
    raw_b = torch.mul(raw_de_b[:, 0, :, :], mask_b)
    
    rl = rl[:, None, None] if rl.ndim == 1 else rl[:, cut_x:-cut_x, cut_y:-cut_y]
    raw_blur = (raw_r + raw_g + raw_b).clip(0.) * rl # [B, H, W]
    with torch.no_grad(): noise = torch.randn_like(raw_blur) * sigma + torch.poisson(raw_blur / lamb) * lamb - raw_blur # [B, H, W]
    raw_blur = raw_blur + noise
    raw_blur = raw_blur.clip(0., 1.)
    return raw_blur, raw_label, color


def isp(raw:torch.Tensor, rlc:torch.Tensor, color:torch.Tensor, wb:torch.Tensor, cm:torch.Tensor, alpha:torch.Tensor=None):
    """
    :param raw: raw_img # [B, H, W]
    :param rlc: rlc_block # [B, H, W] / [B]
    :param color: raw_color # [B, H, W]
    :param wb: wb_matrix # [B, 4]
    :param cm: color_matrix # [B, 3, 3]
    :param alpha: alpha for tone mapping # [B]
    """
    img = raw
    
    # 0. lens shading correction
    img = lsc(img, rlc)
    
    # 1. denoise
    img = denoise(img)
    
    # 2. white balance
    img = white_balance(img, color, wb)
    
    # 3. demosaic
    img = demosaic(img, color, method='Malvar')
    
    # 4. ccm
    img = ccm(img, cm)
    
    # 5. gamma
    img = gamma(img, gamma_type='Rec709')
    
    # 6. tone mapping
    img = tone_mapping(img, alpha)

    return img # [B, 3, H, W]

def isp_raw(raw:torch.Tensor, rlc:torch.Tensor, color:torch.Tensor, wb:torch.Tensor):
    """
    :param raw: raw_img # [B, H, W]
    :param rlc: rlc_block # [B, H, W] / [B]
    :param color: raw_color # [B, H, W]
    :param wb: wb_matrix # [B, 4]
    """
    img = raw
    
    # 0. lens shading correction
    img = lsc(img, rlc)
    
    # 1. denoise
    img = denoise(img)
    
    # 2. white balance
    img = white_balance(img, color, wb)
    
    # 3. demosaic
    img = demosaic(img, color, method='Malvar')
    
    return img # [B, 3, H, W]

def isp_rgb(raw:torch.Tensor, cm:torch.Tensor, alpha:torch.Tensor=None):
    """
    :param raw: raw_img # [B, 3, H, W]
    :param cm: color_matrix # [B, 3, 3]
    :param alpha: alpha for tone mapping # [B]
    """
    img = raw
    
    # 1. ccm
    img = ccm(img, cm)
    
    # 2. gamma
    img = gamma(img, gamma_type='Rec709')
    
    # 3. tone mapping
    img = tone_mapping(img, alpha)
    
    return img # [B, 3, H, W]

def lsc(raw:torch.Tensor, rlc:torch.Tensor):
    """
    :param raw: raw_img [B, H, W]
    :param rlc: rl_block [B, H, W] / [B]
    """
    rlc = rlc[:, None, None] if rlc.ndim == 1 else rlc
    raw_correct = raw * rlc
    return raw_correct.clip(1e-8, 1.0) # [B, H, W]

def denoise(raw:torch.Tensor):
    """
    :param raw: raw_img [B, H, W]
    :param color: raw_color [B, H, W]
    """
    A = raw[:, None, 0::2, 0::2]
    B = raw[:, None, 0::2, 1::2]
    C = raw[:, None, 1::2, 0::2]
    D = raw[:, None, 1::2, 1::2]
    
    def guide_filter(raw_noisy, guide, radius=1, eps=5e-6):
        mean_filter = raw.new_ones(1, 1, 2 * radius + 1, 2 * radius + 1) / ((2 * radius + 1) ** 2)
        
        mean_I = F.conv2d(F.pad(guide, (radius, radius, radius, radius), mode='reflect'), mean_filter)
        mean_p = F.conv2d(F.pad(raw_noisy, (radius, radius, radius, radius), mode='reflect'), mean_filter)
        mean_Ip = F.conv2d(F.pad(guide * raw_noisy, (radius, radius, radius, radius), mode='reflect'), mean_filter)
        
        cov_Ip = mean_Ip - mean_I * mean_p
        
        mean_II = F.conv2d(F.pad(guide * guide, (radius, radius, radius, radius), mode='reflect'), mean_filter)
        var_I = mean_II - mean_I * mean_I
        
        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I
        
        mean_a = F.conv2d(F.pad(a, (radius, radius, radius, radius), mode='reflect'), mean_filter)
        mean_b = F.conv2d(F.pad(b, (radius, radius, radius, radius), mode='reflect'), mean_filter)
        
        raw_denoised = mean_a * guide + mean_b
        return raw_denoised
    
    guide_img = (A + B + C + D) / 4.
    A = guide_filter(A, guide_img)
    B = guide_filter(B, guide_img)
    C = guide_filter(C, guide_img)
    D = guide_filter(D, guide_img)
    
    new_raw = torch.zeros_like(raw) 
    new_raw[:, 0::2, 0::2] = A[:, 0]
    new_raw[:, 0::2, 1::2] = B[:, 0]
    new_raw[:, 1::2, 0::2] = C[:, 0]
    new_raw[:, 1::2, 1::2] = D[:, 0]
    return new_raw.clip(1e-8, 1.0)

def white_balance(raw:torch.Tensor, color:torch.Tensor, wb:torch.Tensor):
    """
    :param raw: raw_img [B, H, W]
    :param color: raw_color [B, H, W]
    :param wb: wb_matrix [B, 4]
    """
    wb_mask = torch.zeros_like(raw)
    wb = wb / wb.amin(dim=-1, keepdim=True)
    for i in range(wb.shape[0]):
        for j in range(wb.shape[1]):
            if torch.any(torch.eq(color[i], j)):
                wb_mask[i] = wb_mask[i].masked_fill(torch.eq(color[i], j), wb[i][j])

    wb_raw = torch.mul(raw, wb_mask).clip(1e-8, 1.0)
    return wb_raw # [B, H, W]

def demosaic(raw:torch.Tensor, color:torch.Tensor, method:str='Malvar'):
    """
    :param raw: raw_img [B, H, W]
    :param color: raw_color [B, H, W]
    :param method: "Malvar" or "Bilinear"
    """
    raw = raw[:, None, :, :] # [B, 1, H, W]
    color = color[:, None, :, :] # [B, 1, H, W]
    
    mask_r = torch.zeros_like(raw) # [B, 1, H, W]
    mask_g = torch.zeros_like(raw)
    mask_b = torch.zeros_like(raw)

    for i, col in enumerate(b'RGBG'):
        if chr(col) == 'R':
            mask_r = mask_r.masked_fill(torch.eq(color, i), 1)
        if chr(col) == 'G':
            mask_g = mask_g.masked_fill(torch.eq(color, i), 1)
        if chr(col) == 'B':
            mask_b = mask_b.masked_fill(torch.eq(color, i), 1)
            
    image_r = torch.mul(raw, mask_r) # [B, 1, H, W]
    image_g = torch.mul(raw, mask_g)
    image_b = torch.mul(raw, mask_b)
    
    if method == 'Bilinear':
        r_b_kernel1 = torch.tensor([[1, 0, 1],
                                    [0, 0, 0],
                                    [1, 0, 1]], dtype=torch.float32)[None, None, :, :] / 4.0
        r_b_kernel2 = torch.tensor([[0, 1, 0],
                                    [1, 0, 1],
                                    [0, 1, 0]], dtype=torch.float32)[None, None, :, :] / 2.0
        g_kernel = torch.tensor([[0, 1, 0],
                                 [1, 0, 1],
                                 [0, 1, 0]], dtype=torch.float32)[None, None, :, :] / 4.0

        image_r = image_r + torch.nn.functional.conv2d(image_r, r_b_kernel1, stride=(1, 1), padding="same") + torch.nn.functional.conv2d(image_r, r_b_kernel2, stride=(1, 1), padding="same")
        image_g = image_g + torch.nn.functional.conv2d(image_g, g_kernel, stride=(1, 1), padding="same")
        image_b = image_b + torch.nn.functional.conv2d(image_b, r_b_kernel1, stride=(1, 1), padding="same") + torch.nn.functional.conv2d(image_b, r_b_kernel2, stride=(1, 1), padding="same")

    elif method == 'Malvar':
        f0 = torch.tensor([[0, 0, -1, 0, 0],
                           [0, 0, 2, 0, 0],
                           [-1, 2, 4, 2, -1],
                           [0, 0, 2, 0, 0],
                           [0, 0, -1, 0, 0]], dtype=torch.float32)[None, None, :, :] / 8
        f1 = torch.tensor([[0, 0, 1, 0, 0],
                           [0, -2, 0, -2, 0],
                           [-2, 8, 10, 8, -2],
                           [0, -2, 0, -2, 0],
                           [0, 0, 1, 0, 0]], dtype=torch.float32)[None, None, :, :] / 16
        f2 = torch.transpose(f1, dim0=-2, dim1=-1)
        f3 = torch.tensor([[0, 0, -3, 0, 0],
                           [0, 4, 0, 4, 0],
                           [-3, 0, 12, 0, -3],
                           [0, 4, 0, 4, 0],
                           [0, 0, -3, 0, 0]], dtype=torch.float32)[None, None, :, :] / 16

        d0 = torch.nn.functional.conv2d(raw, f0, stride=1, padding="same")
        d1 = torch.nn.functional.conv2d(raw, f1, stride=1, padding="same")
        d2 = torch.nn.functional.conv2d(raw, f2, stride=1, padding="same")
        d3 = torch.nn.functional.conv2d(raw, f3, stride=1, padding="same")

        mask_r_g_r_row = torch.roll(mask_r, shifts=1, dims=-1)
        mask_r_g_r_col = torch.roll(mask_r, shifts=1, dims=-2)
        image_r = image_r + torch.mul(d1, mask_r_g_r_row) + torch.mul(d2, mask_r_g_r_col) + torch.mul(d3, mask_b)

        image_g = image_g + torch.mul(d0, mask_r) + torch.mul(d0, mask_b)

        mask_b_g_b_row = torch.roll(mask_b, shifts=1, dims=-1)
        mask_b_g_b_col = torch.roll(mask_b, shifts=1, dims=-2)
        image_b = image_b + torch.mul(d1, mask_b_g_b_row) + torch.mul(d2, mask_b_g_b_col) + torch.mul(d3, mask_r)

    demosaic_image = torch.cat([image_r, image_g, image_b], dim=1).clip(1e-8, 1.0)
    return demosaic_image # [B, 3, H, W]

def ccm(raw:torch.Tensor, cm:torch.Tensor):
    """
    :param raw: raw_img [B, 3, H, W]
    :param cm: color_matrix [B, 3, 3]
    """
    img = torch.einsum('bij, bjkl -> bikl', cm, raw).clip(1e-8, 1.0)
    return img # [B, 3, H, W]

def gamma(image:torch.Tensor, gamma_type:str='Rec709'):
    """
    :param image: [B, 3, H, W]
    :param gamma_type: 'Rec709' or '2.2'
    """
    if gamma_type == 'Rec709':
        gamma_image1 = 4.5 * image.masked_fill(image >= 0.018, 0)
        mask = torch.zeros_like(image)
        gamma_image2 = 1.099 * torch.pow(image.masked_fill(image < 0.018, 0), 0.45) + mask.masked_fill(image >= 0.018, -0.099)
        gamma_image = gamma_image1 + gamma_image2

    elif gamma_type == '2.2':
        gamma_image = torch.pow(torch.clamp(image, min=1e-8), 1/2.2)
    return gamma_image # [B, 3, H, W]

def tone_mapping(raw:torch.Tensor, alpha:torch.Tensor=None):
    """
    :param raw: raw_img [B, 3, H, W]
    :param alpha: alpha [B]
    """
    if alpha is None:
        alpha = torch.ones(raw.shape[0]).to(raw.dtype) * 0. # [B]
        
    mid_image = raw + alpha[:, None, None, None] * (raw) * (1 - raw)
    tone_mapping_image = mid_image + alpha[:, None, None, None] * (mid_image) * (1 - mid_image)
    return tone_mapping_image # [B, 3, H, W]