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
import argparse

import sys as system
system.path.append('..')
import lens
import nets
# %%
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='g_014')
    args = parser.parse_args()
    name = args.name
    args = lens.GetYaml(f'./lens_yaml/{name}_co.yaml')
    lens.configure_material_catalog(getattr(args, 'MATERIAL_CATALOG', None))

    result_folder = f'./results/{name}_{args.DATASET}_co_{str(time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))}'
    os.makedirs(result_folder)
    writer = SummaryWriter(f'{result_folder}/log')
    img_show_num = 1
    
    shutil.copyfile(f'./lens_yaml/{name}_co.yaml', f'{result_folder}/{name}_co.yaml')
    torch.set_default_device(f'{args.DEVICE}')
    lens.set_random_seed(args.SEED)

    waveweights_rgb = torch.tensor([args.WAVEWEIGHTS_R, args.WAVEWEIGHTS_G, args.WAVEWEIGHTS_B])
    
    # initialize optical system
    sys = lens.System(wavelengths=args.WAVELENGTHS, waveweights=args.WAVEWEIGHTS, p_wvl=args.P_WAVE, max_view=args.MAX_VIEW, sys_num=args.SYS_NUM, cfg_num=args.CFG_NUM, pre_samp=args.PRE_SAMP, fix_radius_surf=args.FIX_RADIUS_SURF, norm_views=args.NORM_VIEWS, azimuths=args.AZIMUTHS, file=f'./lens_json/{name}_ref.json')
    analysis = lens.Analysis(sys)
    # save initial results
    analysis.save_analysis_results(result_folder)
    merit = lens.Deletion(system=sys, samp_rays=args.SAMP_RAYS)
    
    # set variables
    match name:
        case 'g_014':
            for i in range(1, len(sys.extract_surfs())-1):
                sys.freeze_sys_param(i, 'conic')
        case 'l_022' | 'g_015':
            for i in range(1, len(sys.extract_surfs())-1):
                sys.freeze_sys_param(i, 'conic')
            sys.freeze_sys_param(sys.stop_id, 'roc')
        case 'l_004':
            for i in range(1, len(sys.extract_surfs())-1):
                sys.freeze_sys_param(i, 'conic')
            sys.freeze_sys_param(sys.stop_id, 'roc')
            sys.freeze_sys_param(sys.stop_id, 'thick')

    # initialize networks
    net = nets.model.Model(args.NET).to(net_dtype)
    
    lens.set_random_seed(args.SEED)
    # set psf parameters and batch sizes
    ang_num = args.PSF_ANG_NUM
    azi_num = args.PSF_AZI_NUM
    psf_sampling = args.PSF_SAMPLING
    psf_size = args.PSF_SIZE
    psf_delta = args.PSF_DELTA
    rl_sampling = args.RL_SAMPLING
    train_batch = ang_num * azi_num * sys.cfg_num * sys.sys_num
    valid_batch = sys.cfg_num * sys.sys_num
    
    # set dataloaders
    trainloader = nets.dataload.train_rawloader(f'../../Data/{args.DATASET}/train', args.TRAIN_PATCH_SIZE + psf_size - 1, train_batch, train_batch)
    validloader = nets.dataload.valid_rawloader(f'../../Data/{args.DATASET}/valid', args.VALID_PATCH_SIZE + psf_size - 1, valid_batch, valid_batch)
    
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
    optimizer_opt = torch.optim.Adam(merit.params_lr(args.LR_OPT))
    optimizer_net = torch.optim.AdamW(net.parameters(), lr=args.LR_NET)
    
    scheduler_opt = torch.optim.lr_scheduler.LinearLR(optimizer_opt, start_factor=1., end_factor=0.05, total_iters=len(trainloader) * args.EPOCH)
    scheduler_net = torch.optim.lr_scheduler.LinearLR(optimizer_net, start_factor=1., end_factor=0.50, total_iters=len(trainloader) * args.EPOCH)
        
    for epoch in range(args.EPOCH):
        epoch_count += 1
        
        net.train()
        for iter_idx, datas in nets.dataload.raw_generator(trainloader, args.DEVICE, net_dtype):
            count += 1
            start = time.time()
            optimizer_opt.zero_grad()
            optimizer_net.zero_grad()
            
            angle = (torch.sqrt(torch.rand(ang_num))).tolist()
            azimuth = (360. * torch.rand(azi_num)).tolist()
            
            sys.norm_views = torch.tensor(angle)
            sys.azimuths = torch.tensor(azimuth)
            
            # calculate optical loss
            loss_opt = merit.forward_loss([], args.MERIT, writer=writer, count=count)

            psfs = merit.psf_rs(psf_sampling, psf_size, psf_delta, angle, azimuth, auto=False)
            psfs = merit.psf_to_rgb(psfs, waveweights_rgb, False)
            psfs = psfs.permute(1, 2, 3, 4, 0, 5, 6).reshape(-1, 3, psfs.shape[-2], psfs.shape[-1]).to(net_dtype)
            rls = merit.relative_illumination(rl_sampling, angle, azimuth).reshape(-1).to(net_dtype) # [sys * cfg * ang * azi]
            
            # generate blurs and labels
            raw_img, raw_color, raw_wb, raw_cm = datas
            sigma = torch.ones(raw_img.shape[0], dtype=net_dtype)[:, None, None] * args.NOISE_G
            lamb = torch.ones(raw_img.shape[0], dtype=net_dtype)[:, None, None] * args.NOISE_P
            raw_blur, raw_label, raw_color = nets.utils.simulate_raw(raw_img, raw_color, psfs, rls, sigma, lamb)
            blurs = nets.utils.isp_raw(raw_blur, rls ** -1, raw_color, raw_wb)
            labels = nets.utils.isp_raw(raw_label, torch.ones_like(rls), raw_color, raw_wb)
            
            alpha = torch.rand(ang_num * azi_num).to(net_dtype)
            match args.NET:
            # calculate image loss
                case 'DWDN' | 'CDWDN':
                    recovs = net(blurs, psfs)
                    recovs = [nets.utils.isp_rgb(recov, raw_cm, alpha) for recov in recovs]
                    labels = nets.utils.isp_rgb(labels, raw_cm, alpha)
                    loss_img = net.loss(recovs, labels)
                    recovs = recovs[-1].clip(0., 1.)
                case 'SRCNN' | 'DEEPSN':
                    recovs = net(blurs)
                    recovs = nets.utils.isp_rgb(recovs, raw_cm, alpha)
                    labels = nets.utils.isp_rgb(labels, raw_cm, alpha)
                    loss_img = net.loss(recovs, labels)
                    recovs = recovs.clip(0., 1.)
                case 'MIMOUNET' | 'FSNET':
                    recovs = net(blurs)
                    recovs = [nets.utils.isp_rgb(recov, raw_cm, alpha) for recov in recovs]
                    labels = nets.utils.isp_rgb(labels, raw_cm, alpha)
                    loss_img = net.loss(recovs, labels)
                    recovs = recovs[-1].clip(0., 1.)
            blurs = nets.utils.isp_rgb(blurs, raw_cm, alpha)
            
            # total loss
            loss = loss_opt * args.OPT_WEIGHT + loss_img * args.IMG_WEIGHT
            loss.backward()
            
            # print sys para and grad
            sys_id, cfg_id = torch.randint(0, sys.sys_num, (1,)).item(), torch.randint(0, sys.cfg_num, (1,)).item()
            sys.print_sys_para(sys_id, cfg_id)
            sys.print_sys_grad(sys_id, cfg_id)
            
            # record images and losses
            writer.add_image('images/label', make_grid(labels[0:img_show_num]), count)
            writer.add_image('images/blurs', make_grid(blurs[0:img_show_num]), count)
            writer.add_image('images/recovs', make_grid(recovs[0:img_show_num]), count)
            
            writer.add_scalar('total/loss_opt', loss_opt, count)
            writer.add_scalar('total/loss_img', loss_img, count)
            writer.add_scalar('total/loss', loss, count)
            writer.add_scalar('total/lr_opt', scheduler_opt.get_last_lr()[0], count)
            writer.add_scalar('total/lr_net', scheduler_net.get_last_lr()[0], count)
            
            # update sys and net
            optimizer_opt.step()
            scheduler_opt.step()
            
            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.01)
            optimizer_net.step()
            scheduler_net.step()
            print(time.time() - start)
            
            # update basic optical attributes
            if count % args.OPTIC_UPDATE_FREQ == 0:
                sys.norm_views = torch.tensor(args.NORM_VIEWS)
                sys.azimuths = torch.tensor(args.AZIMUTHS)
                merit.update_system(update_radius=False, quick_focus=False)
        
        # save results of the epoch
        # merit.update_system(args.MAX_RADIUS, True)
        os.makedirs(f'{result_folder}/epoch_{epoch_count}')
        sys.norm_views = torch.tensor(args.NORM_VIEWS)
        sys.azimuths = torch.tensor(args.AZIMUTHS)
        analysis.save_analysis_results(f'{result_folder}/epoch_{epoch_count}')
        nets.utils.save_images(make_grid(recovs, ang_num * azi_num), f'{result_folder}/epoch_{epoch_count}', f'epoch_{epoch_count}_recov')
        nets.utils.save_images(make_grid(blurs, ang_num * azi_num), f'{result_folder}/epoch_{epoch_count}', f'epoch_{epoch_count}_blur')
        nets.utils.save_images(make_grid(labels, ang_num * azi_num), f'{result_folder}/epoch_{epoch_count}', f'epoch_{epoch_count}_label')
        nets.utils.save_images(make_grid(psfs / psfs.amax(dim=[1, 2, 3], keepdim=True), ang_num * azi_num), f'{result_folder}/epoch_{epoch_count}', f'epoch_{epoch_count}_psf')
        
        # valid metrics
        net.eval()
        psnr_adder_rec.reset()
        ssim_adder_rec.reset()
        lpips_adder_rec.reset()
        psnr_adder_blur.reset()
        ssim_adder_blur.reset()
        lpips_adder_blur.reset()
        for iter_idx, datas in nets.dataload.raw_generator(validloader, args.DEVICE, net_dtype):
            with torch.no_grad():
                # generate psfs
                angle = torch.rand(1).sqrt().item()
                azimuth = 360 * torch.rand(1).item()
                psfs = merit.psf_rs(psf_sampling, psf_size, psf_delta, angle, azimuth, auto=False)
                psfs = merit.psf_to_rgb(psfs, waveweights_rgb, False)
                psfs = psfs.permute(1, 2, 3, 4, 0, 5, 6).reshape(-1, 3, psfs.shape[-2], psfs.shape[-1]).to(net_dtype)
                rls = merit.relative_illumination(rl_sampling, angle, azimuth).reshape(-1).to(net_dtype) # [sys * cfg * ang * azi]
                
                # generate blurs and labels
                raw_img, raw_color, raw_wb, raw_cm = datas
                sigma = torch.ones(raw_img.shape[0], dtype=net_dtype)[:, None, None] * args.NOISE_G
                lamb = torch.ones(raw_img.shape[0], dtype=net_dtype)[:, None, None] * args.NOISE_P
                raw_blur, raw_label, raw_color = nets.utils.simulate_raw(raw_img, raw_color, psfs, rls, sigma, lamb)
                blurs = nets.utils.isp_raw(raw_blur, rls ** -1, raw_color, raw_wb)
                labels = nets.utils.isp_raw(raw_label, torch.ones_like(rls), raw_color, raw_wb)

                alpha = torch.ones(ang_num * azi_num).to(net_dtype) * 0.5
                labels = nets.utils.isp_rgb(labels, raw_cm, alpha)
                match args.NET:
                # generate recovs
                    case 'DWDN' | 'CDWDN':
                        cut_size = args.PSF_SIZE // 2
                        recovs = net(blurs, psfs)
                        recovs = recovs[-1].clip(0., 1.)[:, :, cut_size:-cut_size, cut_size:-cut_size]
                        labels = labels[:, :, cut_size:-cut_size, cut_size:-cut_size]
                    case 'SRCNN' | 'DEEPSN':
                        recovs = net(blurs)
                        recovs = recovs.clip(0., 1.)
                    case 'MIMOUNET' | 'FSNET':
                        recovs = net(blurs)
                        recovs = recovs[-1].clip(0., 1.)
                recovs = nets.utils.isp_rgb(recovs, raw_cm, alpha)
                blurs = nets.utils.isp_rgb(blurs, raw_cm, alpha)
                
                # calculate metrics
                psnr_rec = psnr_adder_rec(psnr_measure(recovs, labels))
                ssim_rec = ssim_adder_rec(ssim_measure(recovs, labels))
                lpips_rec = lpips_adder_rec(lpips_measure(recovs, labels))
                psnr_blur = psnr_adder_blur(psnr_measure(blurs, labels))
                ssim_blur = ssim_adder_blur(ssim_measure(blurs, labels))
                lpips_blur = lpips_adder_blur(lpips_measure(blurs, labels))
                print(f'{iter_idx} - PSNR(rec/blur): {psnr_rec:.4f}/{psnr_blur:.4f} dB, SSIM(rec/blur): {ssim_rec:.4f}/{ssim_blur:.4f}, LPIPS(rec/blur): {lpips_rec:.4f}/{lpips_blur:.4f}')

        writer.add_scalar('metrics/rec_PSNR', psnr_adder_rec.average(), epoch_count)
        writer.add_scalar('metrics/rec_SSIM', ssim_adder_rec.average(), epoch_count)
        writer.add_scalar('metrics/rec_LPIPS', lpips_adder_rec.average(), epoch_count)
        writer.add_scalar('metrics/blur_PSNR', psnr_adder_blur.average(), epoch_count)
        writer.add_scalar('metrics/blur_SSIM', ssim_adder_blur.average(), epoch_count)
        writer.add_scalar('metrics/blur_LPIPS', lpips_adder_blur.average(), epoch_count)

    net_name = os.path.join(result_folder, 'net.pt')
    torch.save({'model': net.state_dict()}, net_name)
    sys.save_json(0, f'{result_folder}/sys_fin.json')
