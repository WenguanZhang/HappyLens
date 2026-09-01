# %%
import torch
opt_dtype = torch.float64
net_dtype = torch.float32
torch.set_default_dtype(opt_dtype)
torch.set_printoptions(precision=10)
from torchvision.utils import make_grid
import torchmetrics.image as metrics
from torch.utils.tensorboard import SummaryWriter

from PIL import Image as Image
import time
import os
import shutil

import sys as system
system.path.append('..')
import lens
import nets
# %%
if __name__ == '__main__':
    name = 'cake_fix'
    args = lens.GetYaml(f'./lens_yaml/{name}.yaml')
    lens.configure_material_catalog(getattr(args, 'MATERIAL_CATALOG', None))
    torch.set_default_device(f'{args.DEVICE}')
    lens.set_random_seed(args.SEED)

    result_folder = f'./results/{str(time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))}_{name}'
    os.makedirs(result_folder)
    shutil.copyfile(f'./lens_yaml/{name}.yaml', f'{result_folder}/{name}.yaml')
    writer = SummaryWriter(f'{result_folder}/log')
    img_show_num = 1

    waveweights_rgb = torch.tensor([args.WAVEWEIGHTS_R, args.WAVEWEIGHTS_G, args.WAVEWEIGHTS_B])
    # initialize optical system
    file = lens.System(wavelengths=args.WAVELENGTHS, 
                       waveweights=args.WAVEWEIGHTS,
                       p_wvl=args.P_WAVE, 
                       max_view=args.MAX_VIEW, 
                       sys_num=args.SYS_NUM, 
                       cfg_num=args.CFG_NUM,
                       samp_method=args.SAMP_METHOD,
                       norm_views=args.NORM_VIEWS, 
                       azimuths=args.AZIMUTHS,
                       file=args.FILE)
    analysis = lens.Analysis(file)
    # save initial results
    analysis.save_analysis_results(result_folder)
    
    merit = lens.Merit(file, args.SAMP_RAYS)
    
    tols_dict_1 = {
        # 1st tol
        '1_1': {"decenter": [0.0, 0.0], "tilt": [0.0, 0.0, 0.]},
        '2_2': {"decenter": [0.0, 0.0], "tilt": [0.0, 0.0, 0.]},
        '3_3': {"decenter": [0.0, 0.0], "tilt": [0.0, 0.0, 0.]},
        '4_4': {"decenter": [0.0, 0.0], "tilt": [0.0, 0.0, 0.]},
    }
    file.ini_tol_sys(tols_dict_1)
    tols_dict_2 = {
        # 2nd tol
        '1_2': {"decenter": [0.0, 0.0], "tilt": [0.0, 0.0, 0.]},
    }
    file.ini_tol_sys(tols_dict_2)

    # initialize network
    net = nets.model.Model(args.NET).to(net_dtype)
    if args.NET_PTH != None: net.load(args.NET_PTH, args.DEVICE)

    lens.set_random_seed(args.SEED)
    # set psf parameters and batch sizes
    ang_num = args.PSF_ANG_NUM
    azi_num = args.PSF_AZI_NUM
    psf_sampling = args.PSF_SAMPLING
    psf_size = args.PSF_SIZE
    psf_delta = args.PSF_DELTA
    rl_sampling = args.RL_SAMPLING
    train_batch = ang_num * azi_num * file.cfg_num * file.sys_num
    valid_batch = file.cfg_num * file.sys_num
    
    trainloader = nets.dataload.train_rgbloader(f'../../Data/{args.DATASET}/train', args.TRAIN_PATCH_SIZE + psf_size - 1, train_batch, train_batch)
    validloader = nets.dataload.valid_rgbloader(f'../../Data/{args.DATASET}/valid', args.VALID_PATCH_SIZE + psf_size - 1, valid_batch, valid_batch)
    
    # set metrics
    psnr_measure = metrics.PeakSignalNoiseRatio().to(args.DEVICE)
    ssim_measure = metrics.StructuralSimilarityIndexMeasure(data_range=1.0).to(args.DEVICE)
    lpips_measure = metrics.LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(args.DEVICE)
    psnr_adder_rec = nets.utils.Adder()
    ssim_adder_rec = nets.utils.Adder()
    lpips_adder_rec = nets.utils.Adder()
    psnr_adder_blur = nets.utils.Adder()
    ssim_adder_blur = nets.utils.Adder()
    lpips_adder_blur = nets.utils.Adder()
    
    count = 0
    epoch_count = 0
    del_num = 0
    
    # set optimizers and schedulers
    optimizer_net = torch.optim.AdamW(net.parameters(), lr=args.LR_NET)
    scheduler_net = torch.optim.lr_scheduler.LinearLR(optimizer_net, start_factor=1., end_factor=0.50, total_iters=len(trainloader) * args.EPOCH)
    
    for epoch in range(args.EPOCH):
        epoch_count += 1
        
        net.train()
        for iter_idx, datas in nets.dataload.rgb_generator(trainloader, args.DEVICE, net_dtype):
            count += 1
            start = time.time()
            optimizer_net.zero_grad()

            # add tolerance
            scale = (epoch + 1) / args.EPOCH
            file.rand_decenter_tilt_thick_param(args.TOL_DECENTER * scale, args.TOL_TILT * scale, args.TOL_THICK * scale)
            # tolerance compensate
            merit.update_system(update_radius=False, quick_focus=True)
            
            with torch.no_grad():
            # generate psfs
                angle = (torch.sqrt(torch.rand(ang_num))).tolist()
                azimuth = (360. * torch.rand(azi_num)).tolist()
                psfs, grids = merit.psf_rs(psf_sampling, psf_size, psf_delta, angle, azimuth, auto=False, chief_o=True)
                psfs = merit.psf_to_rgb(psfs, waveweights_rgb, False)
                psfs = psfs.permute(1, 2, 3, 4, 0, 5, 6).reshape(-1, 3, psfs.shape[-2], psfs.shape[-1]).to(net_dtype)
                rls = merit.relative_illumination(rl_sampling, angle, azimuth).reshape(-1).to(net_dtype) # [sys * cfg * ang * azi]
                
            # generate blurs and labels
            sigma = torch.ones(datas.shape[0], dtype=net_dtype)[:, None, None, None] * args.NOISE_G
            lamb = torch.ones(datas.shape[0], dtype=net_dtype)[:, None, None, None] * args.NOISE_P
            blurs, labels = nets.utils.simulate_rgb(datas, psfs, rls, sigma, lamb) # [B, C, H, W]
            
            # recover images
            blurs = blurs / rls[:, None, None, None]
            match args.NET:
                case 'DEEPSN+F' | 'FSNET+F' | 'MIMOUNET+F':
                    line_sample = torch.linspace(-int((args.TRAIN_PATCH_SIZE - 1) / 2), int((args.TRAIN_PATCH_SIZE - 1) / 2), args.TRAIN_PATCH_SIZE) * args.PSF_DELTA / 1.e3
                    y, x = torch.meshgrid(-line_sample, line_sample, indexing='ij') # [H, W]
                    grids = grids[..., None, None] + torch.stack([x, y], dim=0)[:, None, None, None, None, :, :]
                    grids = grids.permute(1, 2, 3, 4, 0, 5, 6).reshape(-1, 2, args.TRAIN_PATCH_SIZE, args.TRAIN_PATCH_SIZE).to(net_dtype) / args.RENDER_R # [B, 2, H, W]
                    blurs = torch.cat([blurs, grids], dim=1) # [B, C+2, H, W]
            match args.NET:
                # calculate image loss
                case 'DWDN' | 'CDWDN':
                    recovs = net(blurs, psfs)
                    loss_img = net.loss(recovs, labels)
                    recovs = recovs[-1].clip(0., 1.)
                case 'SRCNN' | 'RESTORMER' | 'DEEPSN' | 'DEEPSN+F':
                    recovs = net(blurs)
                    loss_img = net.loss(recovs, labels)
                    recovs = recovs.clip(0., 1.)
                case 'MIMOUNET' | 'MIMOUNET+F' | 'FSNET' | 'FSNET+F':
                    recovs = net(blurs)
                    loss_img = net.loss(recovs, labels)
                    recovs = recovs[-1].clip(0., 1.)
                case 'DIFF':
                    recovs, noise_pred, noise_ref = net(labels, blurs)
                    loss_img = net.loss(labels, recovs, noise_pred, noise_ref)
            blurs = blurs[:, :3, :, :]
            
            # total loss
            loss = loss_img * args.IMG_WEIGHT
            loss.backward()
            
            # update sys and net
            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.01)
            optimizer_net.step()
            scheduler_net.step()
            print(time.time() - start)
            
            # record images and losses
            writer.add_image('images/label', make_grid(labels[0:img_show_num]), count)
            writer.add_image('images/blurs', make_grid(blurs[0:img_show_num]), count)
            writer.add_image('images/recovs', make_grid(recovs[0:img_show_num]), count)
            
            writer.add_scalar('total/loss_img', loss_img, count)
            writer.add_scalar('total/loss', loss, count)
            writer.add_scalar('total/lr_net', scheduler_net.get_last_lr()[0], count)
            
        # save results of the epoch
        os.makedirs(f'{result_folder}/epoch_{epoch_count}')
        nets.utils.save_images(make_grid(recovs, ang_num * azi_num), f'{result_folder}/epoch_{epoch_count}', f'epoch_{epoch_count}_recov')
        nets.utils.save_images(make_grid(blurs, ang_num * azi_num), f'{result_folder}/epoch_{epoch_count}', f'epoch_{epoch_count}_blur')
        nets.utils.save_images(make_grid(labels, ang_num * azi_num), f'{result_folder}/epoch_{epoch_count}', f'epoch_{epoch_count}_label')
        nets.utils.save_images(make_grid(psfs / psfs.amax(dim=[1, 2, 3], keepdim=True), ang_num * azi_num), f'{result_folder}/epoch_{epoch_count}', f'epoch_{epoch_count}_psf')
        
        # valid metrics
        if epoch_count % 10 != 0:
            continue
        
        net.eval()
        psnr_adder_rec.reset()
        ssim_adder_rec.reset()
        lpips_adder_rec.reset()
        psnr_adder_blur.reset()
        ssim_adder_blur.reset()
        lpips_adder_blur.reset()
        for iter_idx, datas in nets.dataload.rgb_generator(validloader, args.DEVICE, net_dtype):
            with torch.no_grad():
                # generate psfs
                # add tolerance
                file.rand_decenter_tilt_thick_param(args.TOL_DECENTER, args.TOL_TILT, args.TOL_THICK)
                # tolerance compensate
                merit.update_system(update_radius=False, quick_focus=True)

                angle = torch.rand(1).sqrt().item()
                azimuth = 360 * torch.rand(1).item()
                psfs, grids = merit.psf_rs(psf_sampling, psf_size, psf_delta, angle, azimuth, auto=False, chief_o=True)
                psfs = merit.psf_to_rgb(psfs, waveweights_rgb, False)
                psfs = psfs.permute(1, 2, 3, 4, 0, 5, 6).reshape(-1, 3, psfs.shape[-2], psfs.shape[-1]).to(net_dtype)
                rls = merit.relative_illumination(rl_sampling, angle, azimuth).reshape(-1).to(net_dtype) # [sys * cfg * ang * azi]
                
                # generate blurs and labels
                sigma = torch.ones(datas.shape[0], dtype=net_dtype)[:, None, None, None] * args.NOISE_G
                lamb = torch.ones(datas.shape[0], dtype=net_dtype)[:, None, None, None] * args.NOISE_P
                blurs, labels = nets.utils.simulate_rgb(datas, psfs, rls, sigma, lamb) # [B, C, H, W]
                
                blurs = blurs / rls[:, None, None, None]
                match args.NET:
                    case 'DEEPSN+F' | 'FSNET+F' | 'MIMOUNET+F':
                        line_sample = torch.linspace(-int((args.VALID_PATCH_SIZE - 1) / 2), int((args.VALID_PATCH_SIZE - 1) / 2), args.VALID_PATCH_SIZE) * args.PSF_DELTA / 1.e3
                        y, x = torch.meshgrid(-line_sample, line_sample, indexing='ij') # [H, W]
                        grids = grids[..., None, None] + torch.stack([x, y], dim=0)[:, None, None, None, None, :, :]
                        grids = grids.permute(1, 2, 3, 4, 0, 5, 6).reshape(-1, 2, args.VALID_PATCH_SIZE, args.VALID_PATCH_SIZE).to(net_dtype) / args.RENDER_R # [B, 2, H, W]
                        blurs = torch.cat([blurs, grids], dim=1) # [B, C+2, H, W]
                match args.NET:
                    # generate recovs
                    case 'DWDN' | 'CDWDN':
                        cut_size = args.PSF_SIZE // 2
                        recovs = net(blurs, psfs)
                        recovs = recovs[-1].clip(0., 1.)[:, :, cut_size:-cut_size, cut_size:-cut_size]
                        labels = labels[:, :, cut_size:-cut_size, cut_size:-cut_size]
                        blurs = blurs[:, :, cut_size:-cut_size, cut_size:-cut_size]
                    case 'SRCNN' | 'RESTORMER' | 'DEEPSN' | 'DEEPSN+F':
                        recovs = net(blurs)
                        recovs = recovs.clip(0., 1.)
                    case 'MIMOUNET' | 'MIMOUNET+F' | 'FSNET' | 'FSNET+F':
                        recovs = net(blurs)
                        recovs = recovs[-1].clip(0., 1.)
                    case 'DIFF':
                        recovs = net.model.inference(blurs)
                        recovs = recovs.clip(0., 1.)
                blurs = blurs[:, :3, :, :]

                # calculate metrics
                psnr_rec = psnr_adder_rec(psnr_measure(recovs, labels))
                ssim_rec = ssim_adder_rec(ssim_measure(recovs, labels))
                lpips_rec = lpips_adder_rec(lpips_measure(recovs, labels))
                psnr_blur = psnr_adder_blur(psnr_measure(blurs, labels))
                ssim_blur = ssim_adder_blur(ssim_measure(blurs, labels))
                lpips_blur = lpips_adder_blur(lpips_measure(blurs, labels))
                print(f'{iter_idx} - PSNR(rec/blur): {psnr_rec:.4f}/{psnr_blur:.4f} dB, SSIM(rec/blur): {ssim_rec:.4f}/{ssim_blur:.4f}, LPIPS(rec/blur): {lpips_rec:.4f}/{lpips_blur:.4f}')
                
                if torch.rand(1).item() < 0.333:
                    nets.utils.save_images(make_grid(recovs, ang_num * azi_num), f'{result_folder}/epoch_{epoch_count}', f'iter_{iter_idx}_recov')
                    nets.utils.save_images(make_grid(blurs, ang_num * azi_num), f'{result_folder}/epoch_{epoch_count}', f'iter_{iter_idx}_blur')
                    nets.utils.save_images(make_grid(labels, ang_num * azi_num), f'{result_folder}/epoch_{epoch_count}', f'iter_{iter_idx}_label')

        writer.add_scalar('metrics/rec_PSNR', psnr_adder_rec.average(), epoch_count)
        writer.add_scalar('metrics/rec_SSIM', ssim_adder_rec.average(), epoch_count)
        writer.add_scalar('metrics/rec_LPIPS', lpips_adder_rec.average(), epoch_count)
        writer.add_scalar('metrics/blur_PSNR', psnr_adder_blur.average(), epoch_count)
        writer.add_scalar('metrics/blur_SSIM', ssim_adder_blur.average(), epoch_count)
        writer.add_scalar('metrics/blur_LPIPS', lpips_adder_blur.average(), epoch_count)
        
    net_name = os.path.join(result_folder, 'net.pt')
    torch.save({'model': net.state_dict()}, net_name)
