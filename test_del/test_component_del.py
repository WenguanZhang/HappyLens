# %%
import torch
opt_dtype = torch.float64
net_dtype = torch.float32
torch.set_default_dtype(opt_dtype)
torch.set_printoptions(precision=10)
from torch.utils.tensorboard import SummaryWriter

from PIL import Image as Image
import time
import os
import shutil

import sys as system
system.path.append('..')
import lens
# %%
if __name__ == '__main__':
    result_folder = './results/{}'.format(str(time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())))
    os.makedirs(result_folder)
    writer = SummaryWriter(f'{result_folder}/log')
    
    name = 'g_014'
    args = lens.GetYaml(f'./lens_yaml/{name}.yaml')
    lens.configure_material_catalog(getattr(args, 'MATERIAL_CATALOG', None))
    shutil.copyfile(f'./lens_yaml/{name}.yaml', f'{result_folder}/{name}.yaml')
    torch.set_default_device(f'{args.DEVICE}')
    lens.set_random_seed(args.SEED)

    waveweights_rgb = torch.tensor([args.WAVEWEIGHTS_R, args.WAVEWEIGHTS_G, args.WAVEWEIGHTS_B])
    
    # initialize optical system
    sys = lens.System(wavelengths=args.WAVELENGTHS, waveweights=args.WAVEWEIGHTS, p_wvl=args.P_WAVE, max_view=args.MAX_VIEW, sys_num=args.SYS_NUM, cfg_num=args.CFG_NUM, pre_samp=args.PRE_SAMP, fix_radius_surf=args.FIX_RADIUS_SURF, norm_views=args.NORM_VIEWS, azimuths=args.AZIMUTHS, file=f'./lens_json/{name}.json')
    analysis = lens.Analysis(sys)
    # save initial results
    analysis.save_analysis_results(result_folder)
    merit = lens.Deletion(system=sys, samp_rays=args.SAMP_RAYS)
    
    # set variables
    match name:
        case 'g_014':
            for i in range(1, len(sys.extract_surfs())-1):
                sys.freeze_sys_param(i, 'conic')
        case 'l_004' | 'l_022' | 'g_015':
            for i in range(1, len(sys.extract_surfs())-1):
                sys.freeze_sys_param(i, 'conic')
            sys.freeze_sys_param(sys.stop_id, 'roc')

    count = 0
    del_num = 0
    
    # set optimizers and schedulers
    optimizer_opt = torch.optim.Adam(merit.params_lr(args.LR_OPT))
            
    while del_num < args.DEL_NUM:
        count += 1
        start = time.time()
        optimizer_opt.zero_grad()
        
        # find del element
        del_id = merit.find_del_surfs() if del_num != args.DEL_NUM else []
        print(del_id)

        # calculate optical loss
        loss_opt = merit.forward_loss(del_id, args.MERIT, writer=writer, count=count)
        
        # calculate del loss
        reses_loss, flats_loss = merit.del_surf_loss(del_id)
        loss_del = reses_loss + flats_loss
        del_thresh = args.DEL_THRESH * 2 * torch.amax(torch.stack([sys.system[del_id[0]].radius.amax(dim=-1), sys.system[del_id[1]].radius.amax(dim=-1)]), dim=0)
        if (reses_loss - del_thresh).max() < 0. or (flats_loss - del_thresh).max() < 0.:
        # if True:
            del_num += 1
            sys.del_surfs(del_id)
            loss_del = torch.zeros(sys.sys_num)
            merit.quickfocus()
            continue

        # total loss
        loss = loss_opt * args.OPT_WEIGHT + loss_del * args.DEL_WEIGHT
        loss.backward()
        
        # print sys para and grad
        sys.print_sys_para(0, 0)
        sys.print_sys_grad(0, 0)
        
        # record images and losses
        writer.add_scalar('total/loss_opt', loss_opt, count)
        writer.add_scalar('total/loss_del', loss_del, count)
        writer.add_scalar('total/loss', loss, count)
        
        # update sys
        optimizer_opt.step()
        print(time.time() - start)
        
        # update basic optical attributes
        if count % args.OPTIC_UPDATE_FREQ == 0:
            merit.update_system(args.MAX_RADIUS, True)

        if count % args.SAVE_FREQ == 0:
            # save results
            os.makedirs(f'{result_folder}/epoch_{count}')
            analysis.save_analysis_results(f'{result_folder}/epoch_{count}')

    os.makedirs(f'{result_folder}/final')
    analysis.save_analysis_results(f'{result_folder}/final')
    
    # save sys to zmx
    lens.read_prime_json_to_zmx(f'{result_folder}/final/sys_0_0.0000.json', f'{result_folder}/final/{name}_del.zmx', args.WAVELENGTHS, args.WAVEWEIGHTS, args.P_WAVE, args.NORM_VIEWS, args.MAX_VIEW)
