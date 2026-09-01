# %%
import torch
opt_dtype = torch.float64
net_dtype = torch.float32
torch.set_default_dtype(opt_dtype)
torch.set_printoptions(precision=10)
from torchvision import utils as vutils
import torchmetrics.image as metrics

from tqdm import tqdm
import os
from glob import glob
import h5py
import logging
import shutil
import matplotlib.pyplot as plt

import sys as system
system.path.append('..')
import lens
import nets
# %%
#----------------------------------------------------------------------------------------------#
name = 'g_014_co'
# name = 'g_014_ref'

# name = 'g_015_co'
# name = 'g_015_ref'

# name = 'l_004_co'
# name = 'l_004_ref'

# name = 'l_022_co'
# name = 'l_022_ref'

result_path = r'xxxxxxxxx'

args = lens.GetYaml(r'xxxxxxxxx.yaml')
lens.configure_material_catalog(getattr(args, 'MATERIAL_CATALOG', None))
torch.set_default_device(f'{args.DEVICE}')
lens.set_random_seed(args.SEED)

model_path = r'xxxxxxxxx.pt'
model = nets.model.Model(args.NET).to(net_dtype)
model.load_state_dict(torch.load(model_path, map_location=torch.device(args.DEVICE))['model'])

file = r'xxxxxxxxx.json'
file = lens.System(wavelengths=args.WAVELENGTHS, 
                   waveweights=args.WAVEWEIGHTS,
                   p_wvl=args.P_WAVE, 
                   max_view=args.MAX_VIEW, 
                   sys_num=args.SYS_NUM, 
                   cfg_num=args.CFG_NUM, 
                   pre_samp=args.PRE_SAMP, 
                   norm_views=args.NORM_VIEWS, 
                   azimuths=args.AZIMUTHS, 
                   file=file)
file.vig_chief = True
print(f'EFFL: {file.EFFL}')
print(f'FNO: {file.FNO}')
print(f'ENPD: {file.ENPD}')
print(f'ENPP: {file.ENPP}')
print(f'EXPD: {file.EXPD}')
print(f'EXPP: {file.EXPP}')
print(f'TOTR: {file.TOTR}')
# %%
# #----------------------------------------------------------------------------------------------#
# name = 'g_014'
# name = 'g_015'
# name = 'l_004'
# name = 'l_022'

# result_path = r'xxxxxxxxx'

# args = lens.GetYaml(r'xxxxxxxxx.yaml')
# torch.set_default_device(f'{args.DEVICE}')
# lens.set_random_seed(args.SEED)

# file = r'xxxxxxxxx.json'
# file = lens.System(wavelengths=args.WAVELENGTHS, 
#                    waveweights=args.WAVEWEIGHTS,
#                    p_wvl=args.P_WAVE, 
#                    max_view=args.MAX_VIEW, 
#                    sys_num=args.SYS_NUM, 
#                    cfg_num=args.CFG_NUM, 
#                    pre_samp=args.PRE_SAMP, 
#                    norm_views=args.NORM_VIEWS, 
#                    azimuths=args.AZIMUTHS, 
#                    file=file)
# file.vig_chief = True
# print(f'EFFL: {file.EFFL}')
# print(f'FNO: {file.FNO}')
# print(f'ENPD: {file.ENPD}')
# print(f'ENPP: {file.ENPP}')
# print(f'EXPD: {file.EXPD}')
# print(f'EXPP: {file.EXPP}')
# print(f'TOTR: {file.TOTR}')

# model = None
# %%
sys_id, cfg_id = 0, 0
valid_path = f'../../Data/{args.DATASET}/valid'
ratio = 3840 / 2160
pixel_size = args.PSF_DELTA * 1e-3 # mm
patch_size = args.VALID_PATCH_SIZE # Even number
psf_size = args.PSF_SIZE # Odd number
psf_sampling = args.PSF_SAMPLING # Odd number
psf_delta = args.PSF_DELTA # um
rl_sampling = args.RL_SAMPLING
if patch_size%2!=0 or psf_size%2!=1 or psf_sampling%2!=1:
    raise ValueError("VALID_PATCH_SIZE: must even! / PSF_SIZE: must odd! / PSF_SAMPLING: must odd!")
# %%
analysis = lens.Analysis(file)
analysis.save_analysis_results(result_path)
# %%
img_R = args.RENDER_R
img_h = (2 * img_R) / (ratio ** 2 + 1) ** 0.5
img_w = img_h * ratio
# %%
patch_w_num = int((img_w / pixel_size / patch_size) // 2 * 2) + 1
patch_h_num = int((img_h / pixel_size / patch_size) // 2 * 2) + 1
pixel_w = patch_w_num * patch_size
pixel_h = patch_h_num * patch_size
cut_size = (((patch_size + psf_size * 1) // 8 + 1) * 8 - patch_size) // 2
# %%
print(f'patch_w_num: {patch_w_num:.0f}, patch_h_num: {patch_h_num:.0f}')
print(f'pixel_w: {pixel_w:.0f} px, pixel_h: {pixel_h:.0f} px')
print(f'img_w: {pixel_w * pixel_size} mm, img_h: {pixel_h * pixel_size} mm')
delta_R = img_R - (pixel_w ** 2 + pixel_h ** 2) ** 0.5 * pixel_size / 2
print(f'delta_R: {delta_R:.3f} mm')
# %%
target_x = torch.linspace(-(patch_w_num // 2), (patch_w_num // 2), patch_w_num) * patch_size * pixel_size
target_y = torch.linspace((patch_h_num // 2), -(patch_h_num // 2), patch_h_num) * patch_size * pixel_size
target_X, target_Y = torch.meshgrid(target_x, target_y, indexing='ij')
target_X = target_X.reshape(-1)
target_Y = target_Y.reshape(-1)
target_XY = torch.stack((target_X, target_Y), dim=-1)
# %%
dX = torch.zeros(patch_w_num * patch_h_num)
dY = torch.zeros(patch_w_num * patch_h_num)

for i in range(len(target_XY)):
    xy = lens.pupil_distribution(args.SAMP_RAYS, 0, 'ring') # [M, 2]
    o = torch.tensor([target_XY[i, 0],	   target_XY[i, 1],	   0.0000000000])[None, None, None, None, None, :].repeat(1, 1, 1, 1, xy.shape[0], 1) # [sys, cfg, 1, 1, M, 3]
    o_pp = file.EXPD[:, :, None, None, None, None] * 0.5 * xy[None, None, None, None, :, :] # [sys, cfg, 1, 1, M, 2]
    o_pp = torch.stack([o_pp[..., 0], o_pp[..., 1], -file.EXPP[:, :, None, None, None].repeat(1, 1, 1, 1, xy.shape[0])], dim=-1) # [sys, cfg, 1, 1, M, 3]
    d = lens.normalize(o_pp - o)
    
    ray = lens.Ray(o, d, wavelength=file.wavelengths[file.p_wvl])
    ray = file.reverse_propagate(ray)
    dx = -ray.d[..., 0][ray.valid].mean()
    dy = -ray.d[..., 1][ray.valid].mean()
    print(f'dx: {dx:.6f}, dy: {dy:.6f}')
    dX[i] = dx
    dY[i] = dy
# %%
angle = torch.where(torch.sign(target_Y) != 0, torch.sign(target_Y), 1) * torch.rad2deg(torch.arcsin(torch.sqrt(dX ** 2 + dY ** 2))).detach()
azimuth = torch.rad2deg(torch.arctan(dX / dY)).detach()
angle = (angle.reshape(patch_w_num, patch_h_num) / args.MAX_VIEW).clip(-1., 1.)
azimuth = azimuth.reshape(patch_w_num, patch_h_num)
if patch_h_num % 2 == 1:
    azimuth[0:(patch_w_num-1)//2, (patch_h_num-1)//2] = -90.
    azimuth[(patch_w_num-1)//2+1:, (patch_h_num-1)//2] = 90.
if patch_w_num % 2 == 1: azimuth[(patch_w_num-1)//2, :] = 0.
# %%
merit = lens.Merit(file, args.SAMP_RAYS)
waveweights_rgb = torch.tensor([args.WAVEWEIGHTS_R, args.WAVEWEIGHTS_G, args.WAVEWEIGHTS_B])

match name:
    case 'g_014_co' | 'g_014_ref':
        tols_dict = {
            # 1st tol
            '1_2': {"decenter": [0.0, 0.0], "tilt": [0.0, 0.0, 0.]},
            '3_4': {"decenter": [0.0, 0.0], "tilt": [0.0, 0.0, 0.]},
            '5_6': {"decenter": [0.0, 0.0], "tilt": [0.0, 0.0, 0.]},
            '7_8': {"decenter": [0.0, 0.0], "tilt": [0.0, 0.0, 0.]},
        }
        file.ini_tol_sys(tols_dict)
    case 'g_015_co' | 'g_015_ref':
        tols_dict = {
            # 1st tol
            '1_2': {"decenter": [0.0, 0.0], "tilt": [0.0, 0.0, 0.]},
            '3_4': {"decenter": [0.0, 0.0], "tilt": [0.0, 0.0, 0.]},
            '5_6': {"decenter": [0.0, 0.0], "tilt": [0.0, 0.0, 0.]},
            '7_8': {"decenter": [0.0, 0.0], "tilt": [0.0, 0.0, 0.]},
            '9_10': {"decenter": [0.0, 0.0], "tilt": [0.0, 0.0, 0.]},
            '12_13': {"decenter": [0.0, 0.0], "tilt": [0.0, 0.0, 0.]},
        }
        file.ini_tol_sys(tols_dict)
    case 'l_004_co' | 'l_004_ref':
        tols_dict = {
            # 1st tol
            '1_2': {"decenter": [0.0, 0.0], "tilt": [0.0, 0.0, 0.]},
            '3_5': {"decenter": [0.0, 0.0], "tilt": [0.0, 0.0, 0.]},
            '7_8': {"decenter": [0.0, 0.0], "tilt": [0.0, 0.0, 0.]},
        }
        file.ini_tol_sys(tols_dict)
    case 'l_022_co' | 'l_022_ref':
        tols_dict = {
            # 1st tol
            '1_2': {"decenter": [0.0, 0.0], "tilt": [0.0, 0.0, 0.]},
            '3_5': {"decenter": [0.0, 0.0], "tilt": [0.0, 0.0, 0.]},
            '7_9': {"decenter": [0.0, 0.0], "tilt": [0.0, 0.0, 0.]},
        }
        file.ini_tol_sys(tols_dict)
    case 'g_014':
        tols_dict = {
            # 1st tol
            '1_2': {"decenter": [0.0, 0.0], "tilt": [0.0, 0.0, 0.]},
            '3_4': {"decenter": [0.0, 0.0], "tilt": [0.0, 0.0, 0.]},
            '5_6': {"decenter": [0.0, 0.0], "tilt": [0.0, 0.0, 0.]},
            '7_8': {"decenter": [0.0, 0.0], "tilt": [0.0, 0.0, 0.]},
            '9_10': {"decenter": [0.0, 0.0], "tilt": [0.0, 0.0, 0.]},
        }
        file.ini_tol_sys(tols_dict)
    case 'g_015':
        tols_dict = {
            # 1st tol
            '1_2': {"decenter": [0.0, 0.0], "tilt": [0.0, 0.0, 0.]},
            '3_4': {"decenter": [0.0, 0.0], "tilt": [0.0, 0.0, 0.]},
            '5_6': {"decenter": [0.0, 0.0], "tilt": [0.0, 0.0, 0.]},
            '7_8': {"decenter": [0.0, 0.0], "tilt": [0.0, 0.0, 0.]},
            '9_10': {"decenter": [0.0, 0.0], "tilt": [0.0, 0.0, 0.]},
            '12_13': {"decenter": [0.0, 0.0], "tilt": [0.0, 0.0, 0.]},
            '14_15': {"decenter": [0.0, 0.0], "tilt": [0.0, 0.0, 0.]},
        }
        file.ini_tol_sys(tols_dict)
    case 'l_004':
        tols_dict = {
            # 1st tol
            '1_2': {"decenter": [0.0, 0.0], "tilt": [0.0, 0.0, 0.]},
            '3_5': {"decenter": [0.0, 0.0], "tilt": [0.0, 0.0, 0.]},
            '7_9': {"decenter": [0.0, 0.0], "tilt": [0.0, 0.0, 0.]},
            '10_11': {"decenter": [0.0, 0.0], "tilt": [0.0, 0.0, 0.]},
        }
        file.ini_tol_sys(tols_dict)
    case 'l_022':
        tols_dict = {
            # 1st tol
            '1_2': {"decenter": [0.0, 0.0], "tilt": [0.0, 0.0, 0.]},
            '3_5': {"decenter": [0.0, 0.0], "tilt": [0.0, 0.0, 0.]},
            '7_9': {"decenter": [0.0, 0.0], "tilt": [0.0, 0.0, 0.]},
            '10_11': {"decenter": [0.0, 0.0], "tilt": [0.0, 0.0, 0.]},
        }
        file.ini_tol_sys(tols_dict)
# %%
@torch.no_grad()
def simulate_images(raw_data, save_path:str, sim_dist=False, zernike_err:dict=None):
    # psf calculation
    psf_data = torch.zeros([patch_h_num, patch_w_num, 3, psf_size, psf_size])
    psf_pool = torch.zeros(3, pixel_h, pixel_w)
    for i in tqdm(range(patch_h_num), desc='h'):
        for j in tqdm(range(patch_w_num), desc='w'):
            if zernike_err is not None:
                psf = merit.psf_rs_err(psf_sampling, psf_size, psf_delta, angle[j, i].item(), azimuth[j, i].item(), zernike_err, auto=False)
            else:
                psf = merit.psf_rs(psf_sampling, psf_size, psf_delta, angle[j, i].item(), azimuth[j, i].item(), auto=False)
            psf = merit.psf_to_rgb(psf, waveweights_rgb, False)[:, 0, 0, 0, 0, :, :] # [3, M, M]
            psf_data[i, j, :, :, :] = psf
            psf_pool[:, i * patch_size:i * patch_size + psf_size, j * patch_size:j * patch_size + psf_size] = psf
    psf_map = torch.zeros_like(psf_pool)
    psf_map[:, (patch_size-psf_size)//2+1:, (patch_size-psf_size)//2+1:] = psf_pool[:, :-(patch_size-psf_size)//2, :-(patch_size-psf_size)//2]
    vutils.save_image(psf_map / psf_map.max(), f'{save_path}/psf.png')
    
    # relative illumination calculation
    rl_data = torch.zeros([patch_h_num, patch_w_num])
    for i in tqdm(range(patch_h_num), desc='h'):
        for j in tqdm(range(patch_w_num), desc='w'):
            rl = merit.relative_illumination(rl_sampling, angle[j, i].item(), azimuth[j, i].item())
            rl_data[i, j] = rl
    rl_map = torch.nn.functional.interpolate(rl_data[None, None, ...], size=[pixel_h, pixel_w], mode='bilinear', align_corners=True)[0, 0, ...]
    vutils.save_image(rl_map, f'{save_path}/rl.png')
    
    if sim_dist:
        # distortion calculation
        dist_data = torch.zeros([patch_h_num, patch_w_num, 2])
        for i in tqdm(range(patch_h_num), desc='h'):
            for j in tqdm(range(patch_w_num), desc='w'):
                dist = merit.distortion(rl_sampling, angle[j, i].item(), azimuth[j, i].item())
                dist_data[i, j] = dist
        X = torch.linspace(-pixel_w * pixel_size / 2, pixel_w * pixel_size / 2, patch_w_num)
        Y = torch.linspace(pixel_h * pixel_size / 2, -pixel_h * pixel_size / 2, patch_h_num)
        plt.quiver(X.cpu(), Y.cpu(), dist_data[:, :, 0].cpu(), dist_data[:, :, 1].cpu(), scale=4)
        plt.savefig(f'{save_path}/dist.png')
    
    raw_img, raw_color, raw_wb, raw_cm = raw_data
    alpha = torch.tensor([0.5]).to(net_dtype)
    
    # calculate image size
    h = pixel_h + psf_size - 1 + cut_size * 2
    w = pixel_w + psf_size - 1 + cut_size * 2
    
    raw_label = raw_img[:h, :w] # [h, w]
    raw_color = raw_color[:h, :w] # [h, w]
    
    #*----------------------------------------------- get simulation raw image -----------------------------------------------*#
    if sim_dist:
        # 1. pre distortion
        dist_data_x = torch.nn.functional.interpolate(dist_data[None, None, :, :, 0], [h, w], mode='bilinear', align_corners=True)[0, 0] / pixel_size
        dist_data_y = torch.nn.functional.interpolate(dist_data[None, None, :, :, 1], [h, w], mode='bilinear', align_corners=True)[0, 0] / pixel_size
        
        samp_data_x = torch.linspace(-1., 1., w)
        samp_data_y = torch.linspace(1., -1., h)
        samp_data_y, samp_data_x = torch.meshgrid(samp_data_y, samp_data_x, indexing='ij')
        
        grid_data_x = samp_data_x - dist_data_x * 2 / (w - 1)
        grid_data_y = samp_data_y - dist_data_y * 2 / (h - 1)
        grid_data_xy = torch.stack([grid_data_x, grid_data_y], dim=-1).to(net_dtype)
        
        with torch.no_grad():
            _raw_demosaic = nets.demosaic(raw_label[None, ...], raw_color[None, ...], method='Malvar') # [1, 3, H, W]
            _raw_dist = torch.nn.functional.grid_sample(_raw_demosaic.flip(-2), grid_data_xy[None, ...], padding_mode='reflection', align_corners=True)[0] # [3, H, W]
        
        mask_r = torch.zeros_like(raw_color)
        mask_g = torch.zeros_like(raw_color)
        mask_b = torch.zeros_like(raw_color)
        
        for i, col in enumerate(b'RGBG'):
            if chr(col) == 'R':
                mask_r = mask_r.masked_fill(torch.eq(raw_color, i), 1)
            if chr(col) == 'G':
                mask_g = mask_g.masked_fill(torch.eq(raw_color, i), 1)
            if chr(col) == 'B':
                mask_b = mask_b.masked_fill(torch.eq(raw_color, i), 1)
                
        raw_r = torch.mul(_raw_dist[0, :, :], mask_r) # [H, W]
        raw_g = torch.mul(_raw_dist[1, :, :], mask_g)
        raw_b = torch.mul(_raw_dist[2, :, :], mask_b)
        raw_img = (raw_r + raw_g + raw_b).clip(0.)
    else:
        raw_img = raw_label
        
    # 2. illumination calculation
    rl_data = torch.nn.functional.interpolate(rl_data[None, None, ...], size=[h, w], mode='bilinear', align_corners=True)[0, 0, ...].to(net_dtype)
    
    # blur
    raw_img_blur = torch.zeros(pixel_h, pixel_w).to(net_dtype)
    raw_color_blur = torch.zeros(pixel_h, pixel_w).to(net_dtype)
    blur_img = torch.zeros(3, pixel_h, pixel_w).to(net_dtype)
    # clear
    label_img = torch.zeros(3, pixel_h, pixel_w).to(net_dtype)
    
    for h in range(patch_h_num):
        for w in range(patch_w_num):
            # 3. get psf and raw patches
            psf = psf_data[h, w][None, ...].to(net_dtype) # [1, 3, psf_size, psf_size]
            img_patch = raw_img[h * patch_size:(h + 1) * patch_size + psf_size - 1 + cut_size * 2, w * patch_size:(w + 1) * patch_size + psf_size - 1 + cut_size * 2][None, ...]
            rl_patch = rl_data[h * patch_size:(h + 1) * patch_size + psf_size - 1 + cut_size * 2, w * patch_size:(w + 1) * patch_size + psf_size - 1 + cut_size * 2][None, ...]
            color_patch = raw_color[h * patch_size:(h + 1) * patch_size + psf_size - 1 + cut_size * 2, w * patch_size:(w + 1) * patch_size + psf_size - 1 + cut_size * 2][None, ...]
            
            # 4. get simulation raw patches 
            sigma = torch.ones(1, dtype=net_dtype)[:, None, None] * args.NOISE_G
            lamb = torch.ones(1, dtype=net_dtype)[:, None, None] * args.NOISE_P
            _blur, _label, _color = nets.utils.simulate_raw(img_patch, color_patch, psf, rl_patch, sigma, lamb)
            # return _blur, _color
            
            # 5. lsc -> denoise -> wb -> demosaic
            rl_correct = rl_patch[:, (psf_size - 1) // 2:-(psf_size - 1) // 2, (psf_size - 1) // 2:-(psf_size - 1) // 2] ** -1
            blur = nets.utils.isp_raw(_blur, rl_correct, _color, raw_wb[None, ...]) # [B, 3, H, W]
            label = nets.utils.isp_raw(_label, torch.ones_like(rl_correct), _color, raw_wb[None, ...]) # [B, 3, H, W]
            
            # 6. ccm -> gamma -> tone mapping
            blur = nets.utils.isp_rgb(blur, raw_cm[None, ...], alpha)
            label = nets.utils.isp_rgb(label, raw_cm[None,...], alpha)
            
            # 7. save patch
            # blur
            raw_img_blur[h * patch_size:(h + 1) * patch_size, w * patch_size:(w + 1) * patch_size] = _blur[0, cut_size:-cut_size, cut_size:-cut_size]
            raw_color_blur[h * patch_size:(h + 1) * patch_size, w * patch_size:(w + 1) * patch_size] = _color[0, cut_size:-cut_size, cut_size:-cut_size]
            blur_img[:, h * patch_size:(h + 1) * patch_size, w * patch_size:(w + 1) * patch_size] = blur[0, :, cut_size:-cut_size, cut_size:-cut_size]
            # clear
            label_img[:, h * patch_size:(h + 1) * patch_size, w * patch_size:(w + 1) * patch_size] = label[0, :, cut_size:-cut_size, cut_size:-cut_size]

    if sim_dist:
        # 8. lens distortion correction
        dist_data_x = torch.nn.functional.interpolate(dist_data[None, None, :, :, 0], [pixel_h, pixel_w], mode='bilinear', align_corners=True)[0, 0] / pixel_size
        dist_data_y = torch.nn.functional.interpolate(dist_data[None, None, :, :, 1], [pixel_h, pixel_w], mode='bilinear', align_corners=True)[0, 0] / pixel_size
        
        samp_data_x = torch.linspace(-1., 1., pixel_w)
        samp_data_y = torch.linspace(1., -1., pixel_h)
        samp_data_y, samp_data_x = torch.meshgrid(samp_data_y, samp_data_x, indexing='ij')

        grid_data_x = samp_data_x + dist_data_x * 2 / (pixel_w - 1)
        grid_data_y = samp_data_y + dist_data_y * 2 / (pixel_h - 1)
        grid_data_xy = torch.stack([grid_data_x, grid_data_y], dim=-1).to(net_dtype)
        
        blur_img = torch.nn.functional.grid_sample(blur_img.flip(-2)[None, ...], grid_data_xy[None, ...], padding_mode='reflection', align_corners=True)[0] # [3, H, W]
        label_img = torch.nn.functional.grid_sample(label_img.flip(-2)[None, ...], grid_data_xy[None, ...], padding_mode='reflection', align_corners=True)[0] # [3, H, W]
    
    blur_img = nets.utils.postprocess(blur_img, 1.)
    label_img = nets.utils.postprocess(label_img, 1.)
    
    raw_data = (raw_img_blur, raw_color_blur, raw_wb, raw_cm)
    if sim_dist:
        return blur_img, label_img, raw_data, psf_data, rl_data, dist_data
    else:
        return blur_img, label_img, raw_data, psf_data, rl_data, None

@torch.no_grad()
def process_images(raw_data, net:torch.nn.Module, psf_data=None, rl_data=None, dist_data=None):
    raw_img_blur, raw_color_blur, raw_wb, raw_cm = raw_data
    alpha = torch.tensor([0.5]).to(net_dtype)
    
    # calculate size
    h = pixel_h + cut_size * 2
    w = pixel_w + cut_size * 2
    raw_img = nets.crop_or_pad_tensor(raw_img_blur[None, ...], h, w)[0]
    raw_color = nets.crop_or_pad_tensor(raw_color_blur[None, ...], h, w)[0]
    rl_data = torch.ones_like(raw_img_blur).to(net_dtype) if rl_data is None else nets.crop_or_pad_tensor(rl_data[None, ...], h, w)[0]
    
    rec_img = torch.zeros(3, pixel_h, pixel_w).to(net_dtype)
    for h in range(patch_h_num):
        for w in range(patch_w_num):
            img_patch = raw_img[h * patch_size:(h + 1) * patch_size + cut_size * 2, w * patch_size:(w + 1) * patch_size + cut_size * 2][None, ...]
            rl_patch = rl_data[h * patch_size:(h + 1) * patch_size + cut_size * 2, w * patch_size:(w + 1) * patch_size + cut_size * 2][None, ...]
            color_patch = raw_color[h * patch_size:(h + 1) * patch_size + cut_size * 2, w * patch_size:(w + 1) * patch_size + cut_size * 2][None, ...]
            
            # 1. raw processing
            rl_correct = rl_patch ** -1
            img_patch = nets.utils.isp_raw(img_patch, rl_correct, color_patch, raw_wb[None, ...]) # [B, 3, H, W]
            if net:
                # 2. aberration compensation
                match args.NET:
                    case 'DWDN' | 'CDWDN':
                        if psf_data is None: raise ValueError("psf_data is None!")
                        psf = psf_data[h, w][None, ...].to(net_dtype) # [1, 3, psf_size, psf_size]
                        recov = net(img_patch, psf)
                        recov = recov[-1].clip(0., 1.)
                    case 'SRCNN' | 'DEEPSN':
                        recov = net(img_patch)
                        recov = recov.clip(0., 1.)
                    case 'MIMOUNET' | 'FSNET':
                        recov = net(img_patch)
                        recov = recov[-1].clip(0., 1.)
            else:
                recov = img_patch
            
            # 3. rgb processing
            recov = nets.utils.isp_rgb(recov, raw_cm[None,...], alpha)
            rec_img[:, h * patch_size:(h + 1) * patch_size, w * patch_size:(w + 1) * patch_size] = recov[0, :, cut_size:-cut_size, cut_size:-cut_size]
            
    if dist_data is not None:
        # 4. lens distortion correction
        dist_data_x = torch.nn.functional.interpolate(dist_data[None, None, :, :, 0], [pixel_h, pixel_w], mode='bilinear', align_corners=True)[0, 0] / pixel_size
        dist_data_y = torch.nn.functional.interpolate(dist_data[None, None, :, :, 1], [pixel_h, pixel_w], mode='bilinear', align_corners=True)[0, 0] / pixel_size
        
        samp_data_x = torch.linspace(-1., 1., pixel_w)
        samp_data_y = torch.linspace(1., -1., pixel_h)
        samp_data_y, samp_data_x = torch.meshgrid(samp_data_y, samp_data_x, indexing='ij')

        grid_data_x = samp_data_x + dist_data_x * 2 / (pixel_w - 1)
        grid_data_y = samp_data_y + dist_data_y * 2 / (pixel_h - 1)
        grid_data_xy = torch.stack([grid_data_x, grid_data_y], dim=-1).to(net_dtype)
        
        rec_img = torch.nn.functional.grid_sample(rec_img.flip(-2)[None, ...], grid_data_xy[None, ...], padding_mode='reflection', align_corners=True)[0] # [3, H, W]
    
    rec_img = nets.utils.postprocess(rec_img, 1.)
    return rec_img
# %%
logger = logging.getLogger()
logger.setLevel(logging.INFO)
fh = logging.FileHandler(f'{result_path}/record.log')
fh.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)
# %%
lens.set_random_seed(args.SEED)

psnr_measure = metrics.PeakSignalNoiseRatio().to(args.DEVICE)
ssim_measure = metrics.StructuralSimilarityIndexMeasure().to(args.DEVICE)
lpips_measure = metrics.LearnedPerceptualImagePatchSimilarity().to(args.DEVICE)

psnr_adder_blur = nets.utils.Adder()
ssim_adder_blur = nets.utils.Adder()
lpips_adder_blur = nets.utils.Adder()

psnr_adder_rec = nets.utils.Adder()
ssim_adder_rec = nets.utils.Adder()
lpips_adder_rec = nets.utils.Adder()

pic_path = f'{result_path}/pic'
os.makedirs(pic_path, exist_ok=True)

for idx, img_name in enumerate(glob(os.path.join(valid_path, '*.h5'))):
    name = os.path.basename(img_name)
    name, _ = os.path.splitext(name)
    
    pic_path = f'{result_path}/pic/{idx}'
    os.makedirs(pic_path, exist_ok=True)
    
    with h5py.File(img_name, 'r') as f:
        raw_img = torch.from_numpy(f['raw_img'][:]).to(args.DEVICE).to(net_dtype) # [H, W]
        raw_color = torch.from_numpy(f['raw_color'][:]).to(args.DEVICE).to(net_dtype) # [H, W]
        raw_wb = torch.from_numpy(f['raw_wb_matrix'][:]).to(args.DEVICE).to(net_dtype) # [4]
        raw_cm = torch.from_numpy(f['raw_color_matrix'][:]).to(args.DEVICE).to(net_dtype) # [3, 3]
        raw_data = (raw_img, raw_color, raw_wb, raw_cm)
    
    if hasattr(args, "TOL"):
        file.rand_decenter_tilt_thick_param(args.TOL['decenter'], args.TOL['tilt'], args.TOL['thick'])
        merit.quickfocus()
        
    blur_img, label_img, raw_data, psf_data, rl_data, dist_data = simulate_images(raw_data, pic_path)
    rec_img = process_images(raw_data, model, psf_data=psf_data, rl_data=rl_data, dist_data=dist_data)
    
    vutils.save_image(blur_img, f'{pic_path}/{name}_blur.png')
    vutils.save_image(rec_img, f'{pic_path}/{name}_rec.png')
    vutils.save_image(label_img, f'{pic_path}/{name}_label.png')
    
    # calculate metrics
    psnr_blur = psnr_adder_blur(psnr_measure(blur_img[None, ...], label_img[None, ...]))
    ssim_blur = ssim_adder_blur(ssim_measure(blur_img[None, ...], label_img[None, ...]))
    lpips_blur = lpips_adder_blur(lpips_measure(blur_img[None, ...], label_img[None, ...]))
    
    psnr_rec = psnr_adder_rec(psnr_measure(rec_img[None, ...], label_img[None, ...]))
    ssim_rec = ssim_adder_rec(ssim_measure(rec_img[None, ...], label_img[None, ...]))
    lpips_rec = lpips_adder_rec(lpips_measure(rec_img[None, ...], label_img[None, ...]))
    
    print(f'{name} - rec=(PSNR: {psnr_rec:.4f} dB, SSIM: {ssim_rec:.4f}, LPIPS: {lpips_rec:.4f}) | blur=(PSNR: {psnr_blur:.4f} dB, SSIM: {ssim_blur:.4f}, LPIPS: {lpips_blur:.4f})')
    print(f'average - blur=(PSNR: {psnr_adder_blur.average():.4f} dB, SSIM: {ssim_adder_blur.average():.4f}, LPIPS: {lpips_adder_blur.average():.4f})')
    print(f'average - rec=(PSNR: {psnr_adder_rec.average():.4f} dB, SSIM: {ssim_adder_rec.average():.4f}, LPIPS: {lpips_adder_rec.average():.4f})')
    logger.info(f'{name} - rec=(PSNR: {psnr_rec:.4f} dB, SSIM: {ssim_rec:.4f}, LPIPS: {lpips_rec:.4f}) | blur=(PSNR: {psnr_blur:.4f} dB, SSIM: {ssim_blur:.4f}, LPIPS: {lpips_blur:.4f})')
# %%
print(f'average - blur=(PSNR: {psnr_adder_blur.average():.4f} dB, SSIM: {ssim_adder_blur.average():.4f}, LPIPS: {lpips_adder_blur.average():.4f})')
print(f'average - rec=(PSNR: {psnr_adder_rec.average():.4f} dB, SSIM: {ssim_adder_rec.average():.4f}, LPIPS: {lpips_adder_rec.average():.4f})')

logger.info(f'average - blur=(PSNR: {psnr_adder_blur.average():.4f} dB, SSIM: {ssim_adder_blur.average():.4f}, LPIPS: {lpips_adder_blur.average():.4f})')
logger.info(f'average - rec=(PSNR: {psnr_adder_rec.average():.4f} dB, SSIM: {ssim_adder_rec.average():.4f}, LPIPS: {lpips_adder_rec.average():.4f})')
# %%
if hasattr(args, "TOL"):
    file.remove_tol_param()
    merit.quickfocus()
        
logger.warning(f'lens system info')
logger.info(f'EFFL: {file.EFFL.item():.3f} mm, FNO: {file.FNO.item():.3f}, TOTR: {file.TOTR.item():.3f} mm')

rl = analysis.relative_illumination(sys_id, cfg_id, pupil_samp=63, field_samp=11, show=False).min()
distor = analysis.distortion(sys_id, cfg_id, pupil_samp=63, field_samp=11, show=False)[file.p_wvl].squeeze()
logger.info(f'min rl: {rl.item():.4f}')
logger.info(f'min distor: {(100 * distor).min().item():.4f}%, max distor: {(100 * distor).max().item():.4f}%')

merit.propagate_all_rays()
x = torch.where(merit.v_dic, merit.o_dic[-1, :, :, :, :, :, :, 0], 0.) # [wav, sys, cfg, ang, azi, M]
y = torch.where(merit.v_dic, merit.o_dic[-1, :, :, :, :, :, :, 1], 0.) # [wav, sys, cfg, ang, azi, M]        
ref_x = x.sum(dim=[0, -1], keepdim=True) / merit.v_dic.sum(dim=[0, -1], keepdim=True) # [1, sys, cfg, ang, azi, 1]
ref_y = y.sum(dim=[0, -1], keepdim=True) / merit.v_dic.sum(dim=[0, -1], keepdim=True) # [1, sys, cfg, ang, azi, 1]
dx = x - ref_x
dy = y - ref_y
avg_spot_ms = torch.where(merit.v_dic, dx ** 2 + dy ** 2, 0.).sum(dim=[0, 2, 3, 4, 5]) / merit.v_dic.sum(dim=[0, 2, 3, 4, 5])
logger.info(f'avg_spot_ms: {(1000 * avg_spot_ms.sqrt()).item():.4f} um')
