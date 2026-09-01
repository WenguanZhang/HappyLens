# %%
import os
import time
import shutil
import argparse
import torch.nn as nn
import torch

torch.set_default_dtype(torch.float64)
torch.set_printoptions(precision=10)
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt

import sys as system
system.path.append('..')
import lens
# %%
def main(args, seed, save_path):
    #!->1 print time
    t = time.time()
    t = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    print(t)

    #!->2 create result folder
    lens.set_random_seed(seed)
    result_folder = f'{save_path}/{str(t)}_{seed}'
    os.makedirs(result_folder)

    #!->3 delano randomization
    delano = lens.Delano(structure='S|S|S', sys_num=args.DELANO_SYS, 
                        target_fov=args.MAX_VIEW, 
                        target_effl=args.MERIT['EFL']['target'], 
                        target_fno=args.MERIT['FNO']['target'], 
                        target_totr=args.MERIT['EFL']['target'] * 1.1,
                        target_bfl=args.MERIT['BFL']['target'],
                        stop_pos=0, mat_type='K|R|R', 
                        dist_min=(args.MERIT['EFL']['target'] * 1.1 - args.MERIT['BFL']['target']) / 2)

    def optimize_cata(lr, save_dir):
        """
        Optimize lens materials based on the optimized y and ybar.
        """
        
        def fitness_cata():
            """
            Calculate the fitness of each system with catalog glasses
            """
            loss_dist_1 = torch.where(delano.elem_dist[:, 0] < delano.elem_dist[:, 1], delano.elem_dist[:, 1] - delano.elem_dist[:, 0], 0.)
            loss_dist_2 = torch.where(delano.elem_dist[:, 1] < delano.elem_dist[:, 2], delano.elem_dist[:, 2] - delano.elem_dist[:, 1], 0.)
            loss_dist = loss_dist_1 + loss_dist_2
            
            loss_a = delano.merit_stop() + delano.merit_dist() + delano.merit_totr()
            loss = loss_a + delano.merit_angle() * 0.1 + loss_dist
            return loss # [sys]
        
        p = tqdm()
        # optimizer initialization
        optimizer = torch.optim.Adam([delano._y, delano._ybar], lr = lr)
        # ======================================================== #
        # Merit y, ybar
        # ======================================================== #
        loss_min = torch.ones(delano.sys_num) * 1e10
        _y_m, _ybar_m = torch.zeros_like(delano._y), torch.zeros_like(delano._ybar)
        iters_tol = 1
        while iters_tol < delano.tol_iters:
            optimizer.zero_grad()
            delano.iters += 1
            
            loss = fitness_cata()
            p.set_description(f'min loss: {loss.min().item():.6f}, max loss: {loss.max().item():.6f}')
            
            valid = (loss_min > loss)
            loss_min[valid] = loss[valid]
            _y_m[valid], _ybar_m[valid] = delano._y.detach().clone()[valid], delano._ybar.detach().clone()[valid]
            
            iters_tol = 1 if loss.min() <= loss_min.min() else iters_tol + 1
            loss.sum().backward()
            optimizer.step()
            delano.update_y_ybar()
            delano.generate()
            
        # ======================================================== #
        # Judge valid system and choose the best id
        # ======================================================== #
        delano._y.data, delano._ybar.data = _y_m, _ybar_m
        delano.update_y_ybar()
        delano.generate()
        
        with torch.no_grad():
            loss = fitness_cata()
        
        _, idx = torch.topk(loss, 1, largest=False)
        delano.idx = int(idx)
        print(f'idx: {delano.idx}')
        delano.print_info(delano.idx)
        
        if save_dir is not None:
            delano.plot_set_up_with_trace(delano.idx, 3)
            plt.savefig(f'{save_dir}/opt_sys_iter_{delano.iters+1}.svg')
            plt.close()
            delano.plot_y_ybar(delano.idx)
            plt.savefig(f'{save_dir}/opt_y_ybar_iter_{delano.iters+1}.svg')
            plt.close()
            
    optimize_cata(lr=args.LR_DELANO, save_dir=result_folder)
    
    #!->4 create system
    sys = lens.System(wavelengths=args.WAVELENGTHS, 
                      waveweights=args.WAVEWEIGHTS,
                      p_wvl=args.P_WAVE, 
                      max_view=args.MAX_VIEW, 
                      sys_num=args.SYS_NUM, 
                      cfg_num=args.CFG_NUM,
                      samp_method=args.SAMP_METHOD,
                      norm_views=args.NORM_VIEWS, 
                      azimuths=args.AZIMUTHS)
    
    inner_radius = (delano.y[delano.idx, 1] * delano.scale).item()
    material = args.MAT
    try:
        n = lens.glass_catalog[material]['nd']
        mat_cata = 'G'
    except:
        n = lens.plastic_catalog[material]['nd']
        mat_cata = 'P'
    qorder = 7

    Q = (torch.rand(sys.sys_num) - 0.5) * 2 - 1
    roc1 = (delano.elem_effl[delano.idx, -1] * delano.scale) ** -1 * (Q + n / (n - 1))
    roc2 = (delano.elem_effl[delano.idx, -1] * delano.scale) ** -1 * (Q + 1)

    sys.system = nn.ModuleList([])
    sys.system.append(lens.OBJECT(material='VACUUM', distance=[None]))
    sys.system.append(lens.Qbfs(
        radius = [[(delano.elem_radius[delano.idx, 0] * delano.scale).item()]] * sys.sys_num,
        material = ['MIRROR'] * sys.sys_num,
        roc = [-2 * delano.elem_effl[delano.idx, 0] * delano.scale] * sys.sys_num,
        thick = [[-delano.elem_dist[delano.idx, 0] * delano.scale]] * sys.sys_num,
        conic = [0.] * sys.sys_num,
        qi_list=[[0.] * sys.sys_num] * qorder,
        rnorm=[(delano.elem_radius[delano.idx, 0] * delano.scale).item()] * sys.sys_num,
        aperture = 'circ',
        min_r = [inner_radius] * sys.sys_num,
        max_r = [(delano.elem_radius[delano.idx, 0] * delano.scale).item()] * sys.sys_num,
    ))
    sys.system.append(lens.Qbfs(
        radius = [[inner_radius]] * sys.sys_num,
        material = ['MIRROR'] * sys.sys_num,
        roc = [2 * delano.elem_effl[delano.idx, 1] * delano.scale] * sys.sys_num,
        thick = [[delano.elem_dist[delano.idx, 1] * delano.scale]] * sys.sys_num,
        conic = [0.] * sys.sys_num,
        qi_list=[[0.] * sys.sys_num] * qorder,
        rnorm=[inner_radius] * sys.sys_num,
    ))
    sys.system.append(lens.Qbfs(
        radius = [[(delano.elem_radius[delano.idx, 2] * delano.scale).item()]] * sys.sys_num,
        material = [material] * sys.sys_num,
        roc = (roc1 ** -1).tolist(),
        thick = [[0.]] * sys.sys_num,
        conic = [0.] * sys.sys_num,
        mat_cata = mat_cata,
        qi_list=[[0.] * sys.sys_num] * qorder,
        rnorm=[(delano.elem_radius[delano.idx, 2] * delano.scale).item()] * sys.sys_num,
    ))
    sys.system.append(lens.Qbfs(
        radius = [[(delano.elem_radius[delano.idx, 2] * delano.scale).item()]] * sys.sys_num,
        material = ['VACUUM'] * sys.sys_num,
        roc = (roc2 ** -1).tolist(),
        thick = [[(delano.elem_dist[delano.idx, 2] * delano.scale).item()]] * sys.sys_num,
        conic = [0.] * sys.sys_num,
        qi_list=[[0.] * sys.sys_num] * qorder,
        rnorm=[(delano.elem_radius[delano.idx, 2] * delano.scale).item()] * sys.sys_num,
    ))
    sys.system.append(lens.IMAGE(radius=[[(delano.elem_radius[delano.idx, -1] * delano.scale).item()]] * sys.sys_num))
    sys.stop_id = 1
    sys.samp_margin = 0.
    sys.update()
    
    #!->5 set variables
    analysis = lens.Analysis(sys)
    analysis.save_analysis_results(result_folder, 0, 0., args.SAMP_RAYS)
    merit = lens.Generation_Prime(system=sys, samp_rays=args.SAMP_RAYS)
    
    for i in range(1, len(sys.extract_surfs())-1):
        sys.freeze_sys_param(i, 'all')
        sys.unfreeze_sys_param(i, 'roc')
        sys.unfreeze_sys_param(i, 'thick')

    @torch.no_grad()
    def update():
        ray = sys.sample_ray_2d(args.SAMP_RAYS * 20 + 1, samp_method='line')
        ray = sys.system[0].propagate(ray)
        o = ray.o.unsqueeze(0)
        surfs = sys.extract_surfs()
        
        flags = [True, False, False, False]
        for i, elem in enumerate(surfs[1:-1]):
            o_s, _, ray = elem.propagate(ray, surfs[i], radius_flag=flags[i])
            o = torch.cat([o, o_s.unsqueeze(0)], dim=0)
        
        inner_radius = torch.where(ray.valid[:, :, :, 0, :, :], lens.length(o[2, :, :, :, 0, :, :, 0:2]), -torch.inf).amax(dim=[0, 2, 3, 4]) # [sys]
        surfs[1].min_r = inner_radius
        surfs[2].radius = inner_radius[:, None].repeat(1, sys.cfg_num)
        surfs[3].radius = torch.where(ray.valid, lens.length(o[-2, :, :, :, :, :, :, 0:2]), -torch.inf).amax(dim=[0, 2, 3, 4, 5])[:, None].repeat(1, sys.cfg_num) # [sys, cfg]
        surfs[4].radius = torch.where(ray.valid, lens.length(o[-1, :, :, :, :, :, :, 0:2]), -torch.inf).amax(dim=[0, 2, 3, 4, 5])[:, None].repeat(1, sys.cfg_num) # [sys, cfg]
        
        sys.update()
        merit.quickfocus(True)
        
    def params_lr(sys:lens.System, lr):
        param_list = []
        # for generation
        for name, params in sys.system.named_parameters():
            print(name)
            if name.endswith('roc'):
                param_list.append({'params': params, 'lr': lr * 1e-1})
            elif name.endswith('thick'):
                param_list.append({'params': params, 'lr': lr / 1e-1})
            elif name.endswith('conic'):
                param_list.append({'params': params, 'lr': lr})
            elif 'qi' in name:
                param_list.append({'params': params, 'lr': lr})
        return param_list

    #!->6 optimize system
    writer = SummaryWriter(f'{result_folder}/log')
    count = 0
    best_opt_data = sys.extract_all_sys_data()
    with torch.no_grad(): min_loss = torch.nan_to_num(merit.forward_loss(args.MERIT), torch.inf)
    for es, epochs in enumerate(args.EPOCH):
        for idx in range(1, len(sys.system)-1):
            if args.ADD_PARAM[es] is not None:
                for param in args.ADD_PARAM[es]:
                    sys.unfreeze_sys_param(idx, param)
                    
        optimizer = torch.optim.Adam(params_lr(sys, args.LR_OPT), fused=True)
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[
            torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1., total_iters=epochs//2),
            torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=epochs//2),
            ], milestones=[epochs//2])
        for epoch in range(epochs):
            print(count)
            count += 1
            start = time.time()
            optimizer.zero_grad()
            
            loss = merit.forward_loss(args.MERIT, writer=writer, count=count)
            loss[sys.valid].sum().backward()

            writer.add_scalar('total/loss', loss[sys.valid].min(), count)
            writer.add_scalar('total/loss_mean', loss[sys.valid].mean(), count)
            writer.add_scalar('total/lr', scheduler.get_last_lr()[0], count)
            writer.add_scalar('total/valid', sys.valid.sum(), count)

            lens.utils.rand_dropout(optimizer, args.DROPOUT)
            optimizer.step()
            scheduler.step()
            print(time.time() - start, f'valid: {sys.valid.sum()}')
            
            if count % args.OPTIC_UPDATE_FREQ == 0:
                merit.reborn_bad_system(args.MERIT, optimizer)
                update()
                if best_opt_data is not None and min_loss is not None:
                    cur_opt_data = sys.extract_all_sys_data()
                    with torch.no_grad(): cur_loss = torch.nan_to_num(merit.forward_loss(args.MERIT), torch.inf)
                    valid = (cur_loss < min_loss) & sys.valid
                    for s in best_opt_data:
                        for key in best_opt_data[s]:
                            best_opt_data[s][key][valid] = cur_opt_data[s][key][valid]
                    min_loss[valid] = cur_loss[valid]
                    
            if count % args.SAVE_FREQ == 0:
                save_path = f'{result_folder}/epoch_{count}'
                os.makedirs(save_path)
                update()
                with torch.no_grad(): loss = merit.forward_loss(args.MERIT, path=save_path)
                idx = torch.nan_to_num(loss, torch.inf).argmin()
                analysis.save_analysis_results(save_path, idx, loss[idx], args.SAMP_RAYS)

        if hasattr(args, 'SHADE'):
            update()
            merit.reborn_bad_system(args.MERIT)
            update()
            count = merit.differential_evolution_system_shade(args=args.MERIT, 
                                                              iters=args.SHADE['iters'],
                                                              F=args.SHADE['F'],
                                                              CR=args.SHADE['CR'],
                                                              c=args.SHADE['c'],
                                                              p=args.SHADE['p'],
                                                              writer=writer,
                                                              count=count)
            update()
            merit.reborn_bad_system(args.MERIT)
            update()
            
            save_path = f'{result_folder}/epoch_{count}'
            os.makedirs(save_path)
            update()
            with torch.no_grad(): loss = merit.forward_loss(args.MERIT)
            idx = torch.nan_to_num(loss, torch.inf).argmin()
            analysis.save_analysis_results(save_path, idx, loss[idx], args.SAMP_RAYS)
        
        cur_opt_data = sys.extract_all_sys_data()
        with torch.no_grad(): cur_loss = torch.nan_to_num(merit.forward_loss(args.MERIT), torch.inf)
        valid = (cur_loss < min_loss) & sys.valid
        for s in best_opt_data:
            for key in best_opt_data[s]:
                best_opt_data[s][key][valid] = cur_opt_data[s][key][valid]
        min_loss[valid] = cur_loss[valid]
                
    #!->7 save results
    save_file_num = 10
    sys.fit_opt_data(best_opt_data)
    merit.sys.update()
    with torch.no_grad(): loss = merit.forward_loss(args.MERIT)
    _, idx = torch.topk(torch.nan_to_num(loss, torch.inf), save_file_num, largest=False)
    
    for i, idc in enumerate(idx):
        save_path = f'{result_folder}/min_loss_{i+1}'
        os.makedirs(save_path)
        analysis.save_analysis_results(save_path, idc, loss[idc], args.SAMP_RAYS)
# %%
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='gen_cakex_inv')
    args = parser.parse_args()
    name = args.name

    args = lens.GetYaml(f'./lens_yaml/{name}.yaml') 
    lens.configure_material_catalog(getattr(args, 'MATERIAL_CATALOG', None))
    torch.set_default_device(f'{args.DEVICE}')

    t = time.time()
    t = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    
    save_path = f'./results/{str(t)}'
    os.makedirs(save_path)
    shutil.copyfile(f'./lens_yaml/{name}.yaml', f'{save_path}/{name}.yaml')
    
    for seed in args.SEED:
        main(args, seed, save_path)
