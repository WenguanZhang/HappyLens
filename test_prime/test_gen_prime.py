# %%
import os
import time
import shutil
import argparse

import torch
torch.set_default_dtype(torch.float64)
torch.set_printoptions(precision=10)
from torch.utils.tensorboard import SummaryWriter

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
    
    #!->3 system initialization
    if hasattr(args, 'DELANO'):
        delano = lens.Delano(structure=args.STRUCTURE, sys_num=args.DELANO['NUM'],
                             target_fov=args.MAX_VIEW, target_effl=args.MERIT['EFL']['target'], 
                             target_fno=args.MERIT['FNO']['target'], target_totr=args.MERIT['TOTR']['target'], target_bfl=args.MERIT['BFL']['target'], 
                             stop_pos=args.STOP_POS, mat_type=args.MAT_TYPE)
        delano.optimize_SA()
        delano.optimize(lr=args.DELANO['LR'], save_dir=result_folder)
        sys = lens.System(wavelengths=args.WAVELENGTHS,
                          waveweights=args.WAVEWEIGHTS,
                          p_wvl=args.P_WAVE,
                          max_view=args.MAX_VIEW,
                          sys_num=args.SYS_NUM,
                          cfg_num=args.CFG_NUM,
                          pre_samp=args.PRE_SAMP,
                          samp_method=args.SAMP_METHOD,
                          norm_views=args.NORM_VIEWS,
                          azimuths=args.AZIMUTHS,
                          delano=[delano, args.SURF_TYPE, args.MAT_TYPE, args.MAT_CATA, args.MERIT]
                          )
    else:
        sys = lens.System(wavelengths=args.WAVELENGTHS,
                          waveweights=args.WAVEWEIGHTS,
                          p_wvl=args.P_WAVE,
                          max_view=args.MAX_VIEW,
                          sys_num=args.SYS_NUM,
                          cfg_num=args.CFG_NUM,
                          pre_samp=args.PRE_SAMP,
                          samp_method=args.SAMP_METHOD,
                          norm_views=args.NORM_VIEWS,
                          azimuths=args.AZIMUTHS,
                          random=[args.STRUCTURE, args.SURF_TYPE, args.MAT_TYPE, args.MAT_CATA, args.STOP_POS, args.MERIT]
                          )

    vnum = len(args.NORM_VIEWS)
    VIG = {
        'VUY': torch.linspace(0.0, args.MAX_VIG, vnum).tolist(),
        'VLY': torch.zeros(vnum).tolist(),
        'VUX': torch.zeros(vnum).tolist(),
        'VLX': torch.zeros(vnum).tolist(),
    }
    sys.vig = VIG
    
    analysis = lens.Analysis(sys)
    merit = lens.Generation_Prime(system=sys, samp_rays=args.SAMP_RAYS)
    
    #!->4 set variables
    for idx in range(1, len(sys.system)-1):
        if idx != sys.stop_id:
            sys.freeze_sys_param(idx, 'all')
            sys.unfreeze_sys_param(idx, 'roc')
            sys.unfreeze_sys_param(idx, 'thick')
            if isinstance(sys.system[idx], (lens.Asphere, lens.Qcon, lens.Qbfs)):
                sys.unfreeze_sys_param(idx, 'conic')

    sys.freeze_sys_param(sys.stop_id, 'roc')
    sys.freeze_sys_param(sys.stop_id, 'conic')
        
    #!->5 start optimization
    writer = SummaryWriter(f'{result_folder}/log')
    count = 0
    count_view = 0
    best_opt_data, min_loss = None, None
    valid = torch.ones(sys.sys_num).bool()
    for es, epochs in enumerate(args.EPOCH):
        for idx in range(1, len(sys.system)-1):
            if idx != sys.stop_id and args.ADD_PARAM[es] is not None:
                for param in args.ADD_PARAM[es]:
                    sys.unfreeze_sys_param(idx, param)
    
        match args.OPT_MAT_STAGE[es]:
            case 0:
                for idx in range(1, len(sys.system)-1):
                    if 'VACUUM' not in sys.system[idx].material['name']:
                        name = sys.material_fit(idx)
                        sys.freeze_sys_param(idx, 'g1')
                        sys.freeze_sys_param(idx, 'g2')
                        print(name)
            case 1:
                for idx in range(1, len(sys.system)-1):
                    if 'VACUUM' not in sys.system[idx].material['name']:
                        sys.unfreeze_sys_param(idx, 'g1')
                        sys.unfreeze_sys_param(idx, 'g2')
        
        optimizer = torch.optim.Adam(merit.params_lr(args.LR_OPT, scale=args.LR_SCALE), fused=True)
        if count_view >= args.MAX_STATE_T:
            scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[
                torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1., total_iters=epochs//2),
                torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1., end_factor=0.1, total_iters=epochs//2),
            ], milestones=[epochs//2])
            merit.random_flip_elements(args.MERIT, mask=valid, p_fix=args.P_FIX, top_pick=args.TOP_PICK)
            merit.random_change_materials(args.MERIT, mask=valid, p_fix=args.P_FIX, top_pick=args.TOP_PICK)
        else:
            scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
            
        for epoch in range(epochs):
            print(count)
            count += 1
            count_view += 1
            start = time.time()
            optimizer.zero_grad()
            
            sys.max_view = torch.tensor(args.MAX_VIEW) * torch.sin(torch.tensor([torch.pi]) / 6 * (1 + 2 * min(count_view / args.MAX_STATE_T, 1)))
            target_FNO = (torch.tensor(args.MERIT['FNO']['target']) / torch.sin(torch.tensor([torch.pi]) / 6 * (1 + 2 * min(count_view / args.MAX_STATE_T, 1)))).tolist()
            if count_view == 1:
                merit.update_stop_radius(target_FNO)
                merit.update_system(rmax=args.MAX_RADIUS)
                merit.reborn_bad_system(args.MERIT)
                if hasattr(args, 'DE'):
                    if args.PRE_OPT:
                        merit.differential_evolution_system_shade(args=args.MERIT,
                                                                  iters=args.DE['iters'],
                                                                  F=args.DE['F'],
                                                                  CR=args.DE['CR'],
                                                                  c=args.DE['c'],
                                                                  p=args.DE['p'])
                        merit.update_stop_radius(target_FNO)
                        merit.update_system(rmax=args.MAX_RADIUS)
                        merit.reborn_bad_system(args.MERIT)
                
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
                merit.update_stop_radius(target_FNO)
                merit.update_system(rmax=args.MAX_RADIUS)
                merit.reborn_bad_system(args.MERIT, optimizer)
                merit.update_system(rmax=args.MAX_RADIUS)
                
                if best_opt_data is not None and min_loss is not None:
                    with torch.no_grad(): cur_loss = torch.nan_to_num(merit.forward_loss(args.MERIT), torch.inf)
                    valid = (cur_loss < min_loss) & sys.valid
                    min_loss[valid] = cur_loss[valid]
                    #TODO: save
                    cur_opt_data = sys.extract_all_sys_data()
                    for s in best_opt_data:
                        for key in best_opt_data[s]:
                            best_opt_data[s][key][valid] = cur_opt_data[s][key][valid]
                    
            if count % args.SAVE_FREQ == 0:
                save_path = f'{result_folder}/epoch_{count}'
                os.makedirs(save_path)
                merit.update_system(rmax=args.MAX_RADIUS)
                with torch.no_grad(): loss = merit.forward_loss(args.MERIT, path=save_path)
                idx = torch.nan_to_num(loss, torch.inf).argmin()
                analysis.save_analysis_results(save_path, idx, loss[idx], args.SAMP_RAYS, args.SAMP_METHOD)

        if (es + 1) <= len(args.EPOCH):
            merit.update_stop_radius(target_FNO)
            merit.update_system(rmax=args.MAX_RADIUS)
            merit.reborn_bad_system(args.MERIT)
            merit.update_system(rmax=args.MAX_RADIUS)
            
            match args.HYBRID_MODE:
                case 'switch':
                    if best_opt_data is None and min_loss is None and count_view >= args.MAX_STATE_T:
                        best_opt_data = sys.extract_all_sys_data()
                        with torch.no_grad(): min_loss = torch.nan_to_num(merit.forward_loss(args.MERIT), torch.inf)
                            
                case 'auto':
                    if best_opt_data is None and min_loss is None and count_view >= args.MAX_STATE_T:
                        best_opt_data = sys.extract_all_sys_data() #TODO: save
                        with torch.no_grad(): min_loss = torch.nan_to_num(merit.forward_loss(args.MERIT), torch.inf)
                        continue
                    elif count_view < args.MAX_STATE_T:
                        pass
                    else:
                        if min(min_loss) < min(cur_loss):
                            pass
                        else:
                            continue

                case _:
                    raise ValueError(f'HYBRID_MODE {args.HYBRID_MODE} is not implemented')
            
            if hasattr(args, 'SA'):
                count = merit.simulated_annealing_system(args=args.MERIT, 
                                                         T=args.SA['T'],
                                                         T_min=args.SA['T_min'],
                                                         step=args.SA['step'],
                                                         alpha=args.SA['alpha'],
                                                         iter=args.SA['iter'],
                                                         ptresh=args.SA['ptresh'],
                                                         writer=writer,
                                                         count=count)
            
            if hasattr(args, 'GA'):
                count = merit.genetic_system(args=args.MERIT, 
                                             iters=args.GA['iters'], 
                                             elitism_rate=args.GA['elitism_rate'], 
                                             mutation_rate=args.GA['mutation_rate'], 
                                             mutation_strength=args.GA['mutation_strength'],
                                             writer=writer,
                                             count=count)
            
            if hasattr(args, 'SHADE'):
                count = merit.differential_evolution_system_shade(args=args.MERIT,
                                                                  iters=args.SHADE['iters'],
                                                                  F=args.SHADE['F'],
                                                                  CR=args.SHADE['CR'],
                                                                  c=args.SHADE['c'],
                                                                  p=args.SHADE['p'],
                                                                  writer=writer,
                                                                  count=count)
            
            merit.update_stop_radius(target_FNO)
            merit.update_system(rmax=args.MAX_RADIUS)
            merit.reborn_bad_system(args.MERIT)
            merit.update_system(rmax=args.MAX_RADIUS)
            
            save_path = f'{result_folder}/epoch_{count}'
            os.makedirs(save_path, exist_ok=True)
            with torch.no_grad(): loss = merit.forward_loss(args.MERIT, path=save_path)
            idx = torch.nan_to_num(loss, torch.inf).argmin()
            analysis.save_analysis_results(save_path, idx, loss[idx], args.SAMP_RAYS, args.SAMP_METHOD)
            
            if best_opt_data is not None and min_loss is not None:
                with torch.no_grad(): cur_loss = torch.nan_to_num(merit.forward_loss(args.MERIT), torch.inf)
                valid = (cur_loss < min_loss) & sys.valid
                min_loss[valid] = cur_loss[valid]
                #TODO: save
                cur_opt_data = sys.extract_all_sys_data()
                for s in best_opt_data:
                    for key in best_opt_data[s]:
                        best_opt_data[s][key][valid] = cur_opt_data[s][key][valid]
                
    #!->6 save results
    save_file_num = args.SYS_NUM
    sys.fit_opt_data(best_opt_data) #TODO: save
    merit.update_system(rmax=args.MAX_RADIUS)
    with torch.no_grad(): loss = merit.forward_loss(args.MERIT)
    _, idx = torch.topk(torch.nan_to_num(loss, torch.inf), save_file_num, largest=False)
    
    for i, idc in enumerate(idx):
        save_path = f'{result_folder}/min_loss_{i+1}'
        os.makedirs(save_path)
        analysis.save_analysis_results(save_path, idc, loss[idc], args.SAMP_RAYS, args.SAMP_METHOD)
# %%
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='gen_prime')
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
